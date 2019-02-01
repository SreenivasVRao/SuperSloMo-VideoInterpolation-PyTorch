"""
ResNet18 + CLSTM + UPerNet
"""
import sys
sys.path.insert(0, "/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/SuperSloMo/models/")
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from resnet2D import *
from layers import *


log = logging.getLogger(__name__)


class FlowComputationModel(nn.Module):
    def __init__(self, in_channels, out_channels, cross_skip, cfg, stage, writer=None):
        super(FlowComputationModel, self).__init__()

        self.cfg = cfg
        self.writer = writer
        self.stage = stage
        self.cross_skip=cross_skip
        self.build_model(in_channels, out_channels)

        pix_mean = self.cfg.get('MODEL', 'PIXEL_MEAN').split(',')
        pix_mean = [float(p) for p in pix_mean]
        pix_std = self.cfg.get('MODEL', 'PIXEL_STD').split(',')
        pix_std = [float(p) for p in pix_std]
        pix_mean = torch.Tensor(pix_mean).float().view(1, -1, 1, 1).cuda()
        pix_std = torch.Tensor(pix_std).float().view(1, -1, 1, 1).cuda()

        self.register_buffer("pix_mean", pix_mean)
        self.register_buffer("pix_std", pix_std)

    def build_model(self, in_channels, out_channels):
        self.norm_type = self.cfg.get('MODEL', 'NORM_TYPE')
        assert self.norm_type.lower() == "bn", "Unsupported normalization method: %s"%self.norm_type

        # --------------------------------------
        # encoder
        # --------------------------------------
        # sample_size = self.cfg.getint('TRAIN', 'CROP_IMH')
        # sample_duration = self.cfg.getint('MISC', 'SAMPLE_DURATION')
        if self.cfg.get("STAGE1", "ENCODER") == "resnet18":
            log.info("Stage 1: ResNet18.")
            self.encoder = resnet18(in_channels=in_channels)
            encoder_dims = [64, 128, 256, 512]
        elif self.cfg.get("STAGE1", "ENCODER") == "resnet34":
            raise NotImplementedError
            self.encoder = resnet34(
                3, zero_init_residual=True,
                temporal_downsampling_last_layer=temporal_downsampling_last_layer
            )
            encoder_dims = [64, 128, 256, 512]
        else:
            raise NotImplementedError('Not supported 3D encoder: {}'.format(self.cfg.get('STAGE1', 'ENCODER')))

        if self.cfg.get("MODEL", "BOTTLENECK")=="CLSTM":
            log.info("Bottleneck Stage 1: CLSTM")
            self.bottleneck_layer = ConvBLSTM(in_channels=encoder_dims[-1], hidden_channels=encoder_dims[-1],
                                              kernel_size=(3, 3), num_layers=2, batch_first=True)

        elif self.cfg.get("MODEL", "BOTTLENECK") == "CGRU":
            log.info("Bottleneck Stage 1: CGRU")
            self.bottleneck_layer = ConvBGRU(in_channels=encoder_dims[-1], hidden_channels=encoder_dims[-1],
                                              kernel_size=(3, 3), num_layers=2, batch_first=True)

        elif self.cfg.get("MODEL", "BOTTLENECK")=="CONV":
            log.info("Bottleneck Stage 1: CONV")
            self.bottleneck_layer = nn.Sequential(
                nn.Conv2d(encoder_dims[-1], encoder_dims[-1], kernel_size=3, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(encoder_dims[-1], encoder_dims[-1], kernel_size=3, bias=False),
                nn.ReLU(inplace=True)
                )

        # --------------------------------------
        # decoder
        # --------------------------------------
        def conv3x3_norm_relu(inplanes, planes):
            layers = []
            layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, padding=1))
            layers.append(make_norm_layer(self.norm_type, planes))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        # 1
        self.fcn_type = self.cfg.get('MODEL', 'FCN_TYPE')
        # out_channels = 3  # * (sample_duration + 1)
        if self.fcn_type == '8s':
            self.upscales = nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            ])
            self.preds = nn.ModuleList([
                nn.Conv2d(d, out_channels, 1) for d in encoder_dims[:0:-1]
            ])
        elif self.fcn_type == '32s':
            self.upscales = nn.ModuleList([
                nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False),
            ])
            self.preds = nn.ModuleList([
                nn.Conv2d(encoder_dims[-1], out_channels, 1)
            ])
        elif self.fcn_type == '4s':
            # UPerNet
            # FPN Module
            fpn_dim = self.cfg.getint('MODEL', 'FPN_DIM')
            self.fpn_top = nn.Sequential(
                nn.Conv2d(encoder_dims[-1], fpn_dim, kernel_size=1, bias=False),
                make_norm_layer(self.norm_type, fpn_dim),
                nn.ReLU(inplace=True)
            )
            fpn_inplanes = encoder_dims
            self.fpn_in = []
            for fpn_inplane in fpn_inplanes[:-1]:  # skip the top layer
                self.fpn_in.append(nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                    make_norm_layer(self.norm_type, fpn_dim),
                    nn.ReLU(inplace=True)
                ))
            self.fpn_in = nn.ModuleList(self.fpn_in)

            self.fpn_out = []
            for i in range(len(fpn_inplanes) - 1):  # skip the top layer
                self.fpn_out.append(nn.Sequential(
                    conv3x3_norm_relu(fpn_dim, fpn_dim),
                ))
            self.fpn_out = nn.ModuleList(self.fpn_out)

            self.conv_last = nn.Sequential(
                conv3x3_norm_relu(len(fpn_inplanes) * fpn_dim, fpn_dim),
                nn.Conv2d(fpn_dim, out_channels, kernel_size=1)
            )

    def apply_bottleneck(self, tensor_list):
        if self.cfg.get("MODEL", "BOTTLENECK") in ["CLSTM", "CGRU"]:
            x_fwd = torch.stack(tensor_list, dim=1)
            x_rev = torch.stack(tensor_list, dim=1)
            output = self.bottleneck_layer(x_fwd, x_rev)
        elif self.cfg.get("MODEL", "BOTTLENECK") == "CONV":
            assert len(tensor_list) == 1, "Wrong number of timesteps."
            x_fwd = tensor_list[0]
            output = self.bottleneck_layer(x_fwd)
            output = output[:, None, ...] # B C H W -> B 1 C H W

        return output

    def decode_input(self, input_tensor, encoder_outputs):
        """
        :param input_tensor: bottleneck layer output from the ResNet encoder.
        :param encoder_outputs: intermediate outputs of the ResNet.
        :return:
        """
        # ------------------------
        # encoder
        # ------------------------
        # x1    # [1, 1/2, 1/2],        [1, 64, 4, 112, 112]
        # x2    # [1, 1/4, 1/4],        [1, 64, 4, 56, 56]
        # x3    # [1/2, 1/8, 1/8],      [1, 128, 2, 28, 28]
        # x4    # [1/4, 1/16, 1/16],    [1, 256, 1, 14, 14]
        # x5    # [1/4, 1/32, 1/32],    [1, 512, 1, 7, 7]
        (x1, x2, x3, x4, x5) = encoder_outputs

        # print('x1: ', x1.shape)
        # print('x2: ', x2.shape)
        # print('x3: ', x3.shape)
        # print('x4: ', x4.shape)
        # print('x5: ', x5.shape)

        # x5 = x5.mean(2)

        if self.fcn_type == '8s':
            x3 = x3.mean(2)
            x4 = x4.mean(2)
            feats = [x3, x4, x5]
            out = 0
            for i, feat in enumerate(feats[::-1]):
                out = self.upscales[i](self.preds[i](feat) + out)
            out = self.squash(out)
        elif self.fcn_type == '32s':
            feats = [x5]
            out = 0
            for i, feat in enumerate(feats[::-1]):
                out = self.upscales[i](self.preds[i](feat) + out)
            out = self.squash(out)
        elif self.fcn_type == '4s':
            # x2 = x2.mean(2)
            # x3 = x3.mean(2)
            # x4 = x4.mean(2)
            conv_out = [x2, x3, x4, x5]

            # # Top-down
            # p5 = self.RCNN_toplayer(c5)
            # p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
            # p4 = self.RCNN_smooth1(p4)
            # p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
            # p3 = self.RCNN_smooth2(p3)
            # p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
            # p2 = self.RCNN_smooth3(p2)

            # f = self.fpn_top(x5)
            f = self.fpn_top(input_tensor)
            fpn_feature_list = [f]
            for i in reversed(range(len(conv_out) - 1)):
                conv_x = conv_out[i]
                conv_x = self.fpn_in[i](conv_x)  # lateral branch

                f = F.upsample(
                    f, size=conv_x.size()[2:],
                    mode='bilinear',
                    align_corners=False
                )  # top-down branch
                f = conv_x + f

                fpn_feature_list.append(self.fpn_out[i](f))

            fpn_feature_list.reverse()  # [P2 - P5]
            output_size = fpn_feature_list[0].size()[2:]
            fusion_list = [fpn_feature_list[0]]
            for i in range(1, len(fpn_feature_list)):
                fusion_list.append(F.upsample(
                    fpn_feature_list[i],
                    output_size,
                    mode='bilinear', align_corners=False)
                )
            fusion_out = torch.cat(fusion_list, 1)
            out = self.conv_last(fusion_out)
            out = F.upsample(out, mode='bilinear', scale_factor=4, align_corners=False)

        else:
            raise RuntimeError('Not supported FCN type: {}'.format(self.fcn_type))

        # print('x1: ', x1.shape)
        # print('x2: ', x2.shape)
        # print('x3: ', x3.shape)
        # print('x4: ', x4.shape)
        # print('x5: ', x5.shape)

        # # -------------------------
        # # decoder
        # # -------------------------
        # x6_in = self.upsample6(x5)
        # raw_out = self.out_conv(x6_in)

        # batch_idxes = torch.LongTensor(range(raw_out.shape[0])).to(raw_out.device)

        # r = raw_out[batch_idxes, 3 * target_idxes].unsqueeze(1)
        # g = raw_out[batch_idxes, 3 * target_idxes + 1].unsqueeze(1)
        # b = raw_out[batch_idxes, 3 * target_idxes + 2].unsqueeze(1)
        # out = torch.cat((r, g, b), dim=1)

        # convert pixel range
        # out = (out - self.pix_mean) / self.pix_std

        return out

    def forward(self, input_tensor):
        """
        :param input_tensor: input tensors -> B, T, C_in, H, W
        :param stage1_encoder_output: if skip connection from stage1 goes to stage2.
        :return: T tuples with <(B, C_6, H, W) tensor, (B, C_out, H, W)> in case of cross stage skip connection.
        else T tuples with <None, (B, C_out, H, W)> in case of cross stage skip connection.
        :return:
        """

        T = input_tensor.shape[1] # B T C H W

        encodings = []

        bottleneck_in = []
        for t in range(T):
            x = input_tensor[:, t, ...]
            e = self.encoder(x, side_output=True)
            encodings.append(e)
            resnet_out = e[-1]
            bottleneck_in.append(resnet_out)

        h = self.apply_bottleneck(bottleneck_in)
        assert len(h.shape) == 5, "Tensor not of shape: B T C H W"
        assert (h.shape[1] == input_tensor.shape[1]), "Number of time steps do not match"

        decodings = []

        for t in range(T):
            h_t = h[:, t, ...]
            e = encodings[t]
            d = self.decode_input(h_t, e)
            # bottleneck output, and skip connections sent to decoder.
            decodings.append((h_t, d))

        return decodings


class FlowInterpolationModel(nn.Module):
    def __init__(self, in_channels, out_channels, cross_skip, cfg, stage, writer=None):
        super(FlowInterpolationModel, self).__init__()

        self.cfg = cfg
        self.writer = writer
        self.stage = stage
        self.cross_skip_connect = cross_skip
        self.build_model(in_channels, out_channels)

        pix_mean = self.cfg.get('MODEL', 'PIXEL_MEAN').split(',')
        pix_mean = [float(p) for p in pix_mean]
        pix_std = self.cfg.get('MODEL', 'PIXEL_STD').split(',')
        pix_std = [float(p) for p in pix_std]
        pix_mean = torch.Tensor(pix_mean).float().view(1, -1, 1, 1).cuda()
        pix_std = torch.Tensor(pix_std).float().view(1, -1, 1, 1).cuda()

        self.register_buffer("pix_mean", pix_mean)
        self.register_buffer("pix_std", pix_std)
        self.squash = nn.Sigmoid()

    def build_model(self, in_channels, out_channels):
        self.norm_type = self.cfg.get('MODEL', 'NORM_TYPE')
        assert self.norm_type.lower() == "bn", "Unsupported normalization method: %s"%self.norm_type

        # --------------------------------------
        # encoder
        # --------------------------------------
        # sample_size = self.cfg.getint('TRAIN', 'CROP_IMH')
        # sample_duration = self.cfg.getint('MISC', 'SAMPLE_DURATION')
        if self.cfg.get("STAGE2", "ENCODER") == "resnet18":
            log.info("Stage 2: ResNet18.")
            self.encoder = resnet18(in_channels=in_channels)
            encoder_dims = [64, 128, 256, 512]

        elif self.cfg.get("STAGE2", "ENCODER") == "resnet34":
            raise NotImplementedError
            self.encoder = resnet34(
                3, zero_init_residual=True,
                temporal_downsampling_last_layer=temporal_downsampling_last_layer
            )
            encoder_dims = [64, 128, 256, 512]
        else:
            raise NotImplementedError('Not supported 3D encoder: {}'.format(self.cfg.get('STAGE1', 'ENCODER')))

        if self.cfg.get("MODEL", "BOTTLENECK")=="CLSTM":
            log.info("Bottleneck Stage 2: CLSTM")
            self.bottleneck_layer = ConvBLSTM(in_channels=encoder_dims[-1], hidden_channels=encoder_dims[-1],
                                              kernel_size=(3, 3), num_layers=2, batch_first=True)
        elif self.cfg.get("MODEL", "BOTTLENECK") == "CGRU":
            log.info("Bottleneck Stage 1: CGRU")
            self.bottleneck_layer = ConvBGRU(in_channels=encoder_dims[-1], hidden_channels=encoder_dims[-1],
                                              kernel_size=(3, 3), num_layers=2, batch_first=True)
        elif self.cfg.get("MODEL", "BOTTLENECK")=="CONV":
            log.info("Bottleneck Stage 2: CONV")
            self.bottleneck_layer = nn.Sequential(
                nn.Conv2d(encoder_dims[-1], encoder_dims[-1], kernel_size=3, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(encoder_dims[-1], encoder_dims[-1], kernel_size=3, bias=False),
                nn.ReLU(inplace=True)
                )

        # --------------------------------------
        # decoder
        # --------------------------------------
        def conv3x3_norm_relu(inplanes, planes):
            layers = []
            layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, padding=1))
            layers.append(make_norm_layer(self.norm_type, planes))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        # 1
        self.fcn_type = self.cfg.get('MODEL', 'FCN_TYPE')
        # out_channels = 3  # * (sample_duration + 1)
        if self.fcn_type == '8s':
            self.upscales = nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            ])
            self.preds = nn.ModuleList([
                nn.Conv2d(d, out_channels, 1) for d in encoder_dims[:0:-1]
            ])
        elif self.fcn_type == '32s':
            self.upscales = nn.ModuleList([
                nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False),
            ])
            self.preds = nn.ModuleList([
                nn.Conv2d(encoder_dims[-1], out_channels, 1)
            ])
        elif self.fcn_type == '4s':
            # UPerNet
            # FPN Module
            fpn_dim = self.cfg.getint('MODEL', 'FPN_DIM')
            n_in = 2 if self.cross_skip_connect else 1
            self.fpn_top = nn.Sequential(
                nn.Conv2d(encoder_dims[-1]*n_in, fpn_dim, kernel_size=1, bias=False),
                make_norm_layer(self.norm_type, fpn_dim),
                nn.ReLU(inplace=True)
            )
            fpn_inplanes = encoder_dims
            self.fpn_in = []
            for fpn_inplane in fpn_inplanes[:-1]:  # skip the top layer
                self.fpn_in.append(nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                    make_norm_layer(self.norm_type, fpn_dim),
                    nn.ReLU(inplace=True)
                ))
            self.fpn_in = nn.ModuleList(self.fpn_in)

            self.fpn_out = []
            for i in range(len(fpn_inplanes) - 1):  # skip the top layer
                self.fpn_out.append(nn.Sequential(
                    conv3x3_norm_relu(fpn_dim, fpn_dim),
                ))
            self.fpn_out = nn.ModuleList(self.fpn_out)

            self.conv_last = nn.Sequential(
                conv3x3_norm_relu(len(fpn_inplanes) * fpn_dim, fpn_dim),
                nn.Conv2d(fpn_dim, out_channels, kernel_size=1)
            )

    def apply_bottleneck(self, tensor_list):
        if self.cfg.get("MODEL","BOTTLENECK") in ["CLSTM", "CGRU"]:
            x_fwd = torch.stack(tensor_list, dim=1)
            x_rev = torch.stack(tensor_list, dim=1)
            output = self.bottleneck_layer(x_fwd, x_rev)
        elif self.cfg.get("MODEL", "BOTTLENECK") == "CONV":
            assert len(tensor_list)==1, "Wrong number of timesteps."
            x_fwd = tensor_list[0]
            output = self.bottleneck_layer(x_fwd)
            output = output[:, None, ...] # B C H W -> B 1 C H W

        return output

    def decode_input(self, input_tensor, encoder_outputs, stage1_encoding=None):
        """
        :param input_tensor: bottleneck layer output from the ResNet encoder.
        :param encoder_outputs: intermediate outputs of the ResNet.
        :return:
        """
        # ------------------------
        # encoder
        # ------------------------
        # x1    # [1, 1/2, 1/2],        [1, 64, 4, 112, 112]
        # x2    # [1, 1/4, 1/4],        [1, 64, 4, 56, 56]
        # x3    # [1/2, 1/8, 1/8],      [1, 128, 2, 28, 28]
        # x4    # [1/4, 1/16, 1/16],    [1, 256, 1, 14, 14]
        # x5    # [1/4, 1/32, 1/32],    [1, 512, 1, 7, 7]
        (x1, x2, x3, x4, x5) = encoder_outputs

        # print('x1: ', x1.shape)
        # print('x2: ', x2.shape)
        # print('x3: ', x3.shape)
        # print('x4: ', x4.shape)
        # print('x5: ', x5.shape)

        # x5 = x5.mean(2)

        if self.fcn_type == '8s':
            raise NotImplementedError
            x3 = x3.mean(2)
            x4 = x4.mean(2)
            feats = [x3, x4, x5]
            out = 0
            for i, feat in enumerate(feats[::-1]):
                out = self.upscales[i](self.preds[i](feat) + out)
            out = self.squash(out)
        elif self.fcn_type == '32s':
            raise NotImplementedError
            feats = [x5]
            out = 0
            for i, feat in enumerate(feats[::-1]):
                out = self.upscales[i](self.preds[i](feat) + out)
            out = self.squash(out)
        elif self.fcn_type == '4s':
            # x2 = x2.mean(2)
            # x3 = x3.mean(2)
            # x4 = x4.mean(2)
            conv_out = [x2, x3, x4, x5]

            # # Top-down
            # p5 = self.RCNN_toplayer(c5)
            # p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
            # p4 = self.RCNN_smooth1(p4)
            # p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
            # p3 = self.RCNN_smooth2(p3)
            # p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
            # p2 = self.RCNN_smooth3(p2)

            # f = self.fpn_top(x5)
            # f = input_tensor

            if stage1_encoding is not None:
                f = torch.cat([stage1_encoding, input_tensor], dim=1)
            else:
                f = input_tensor

            f = self.fpn_top(f)
            fpn_feature_list = [f]
            for i in reversed(range(len(conv_out) - 1)):
                conv_x = conv_out[i]
                conv_x = self.fpn_in[i](conv_x)  # lateral branch

                f = F.upsample(
                    f, size=conv_x.size()[2:],
                    mode='bilinear',
                    align_corners=False
                )  # top-down branch
                f = conv_x + f

                fpn_feature_list.append(self.fpn_out[i](f))

            fpn_feature_list.reverse()  # [P2 - P5]
            output_size = fpn_feature_list[0].size()[2:]
            fusion_list = [fpn_feature_list[0]]
            for i in range(1, len(fpn_feature_list)):
                fusion_list.append(F.upsample(
                    fpn_feature_list[i],
                    output_size,
                    mode='bilinear', align_corners=False)
                )
            fusion_out = torch.cat(fusion_list, 1)
            out = self.conv_last(fusion_out)
            out = F.upsample(out, mode='bilinear', scale_factor=4, align_corners=False)
        else:
            raise RuntimeError('Not supported FCN type: {}'.format(self.fcn_type))

        # print('x1: ', x1.shape)
        # print('x2: ', x2.shape)
        # print('x3: ', x3.shape)
        # print('x4: ', x4.shape)
        # print('x5: ', x5.shape)

        # # -------------------------
        # # decoder
        # # -------------------------
        # x6_in = self.upsample6(x5)
        # raw_out = self.out_conv(x6_in)

        # batch_idxes = torch.LongTensor(range(raw_out.shape[0])).to(raw_out.device)

        # r = raw_out[batch_idxes, 3 * target_idxes].unsqueeze(1)
        # g = raw_out[batch_idxes, 3 * target_idxes + 1].unsqueeze(1)
        # b = raw_out[batch_idxes, 3 * target_idxes + 2].unsqueeze(1)
        # out = torch.cat((r, g, b), dim=1)

        # convert pixel range
        # out = (out - self.pix_mean) / self.pix_std

        return out

    def forward(self, input_tensor, stage1_encoder_output=None):
        """

        :param unet_in: input tensors -> B,T, C_in, H, W
        :param stage1_encoder_output: if skip connection from stage1 goes to stage2.
        :return: T tuples with <(B, C_6, H, W) tensor, (B, C_out, H, W)> in case of cross stage skip connection.
        else T tuples with <None, (B, C_out, H, W)> in case of cross stage skip connection.
        :return:
        """
        assert len(input_tensor.shape) == 5, "Tensor not of shape: B T C H W"

        T = input_tensor.shape[1]  # B T C H W

        encodings = []
        bottleneck_in = []
        for t in range(T):
            x = input_tensor[:, t, ...]
            e = self.encoder(x, side_output=True)
            encodings.append(e)
            resnet_out = e[-1]
            bottleneck_in.append(resnet_out)

        h = self.apply_bottleneck(bottleneck_in)

        decodings = []

        for t in range(T):
            h_t = h[:, t, ...]
            e = encodings[t]
            if self.cross_skip_connect:
                enc_stage1 = stage1_encoder_output[t]
            else:
                enc_stage1 = None

            d = self.decode_input(h_t, e, enc_stage1)
            # bottleneck output, and skip connections sent to decoder.
            # cross-stage encoding as well.

            decodings.append((None, d))
            # maintain some backward compatibility. Bad code.

        return decodings

    def compute_inputs(self, img_tensor, flow_pred_tensor, t):
        """
        Takes input and output from flow computation model, and required time step.
        Builds the required tensor for the interpolation model.

        :param img_tensor: B, 6, H, W image tensor (2 images , batch size B, input to flow computation model)
        :param flow_pred_tensor:  B, 4, H, W Flow 0->1, Flow 1->0 from flow computation model
        :param t: time step of interpolation t in (0, 1)
        :return: input_tensor: B, 16, H, W - input required for interpolation model
        """

        flow_01 = flow_pred_tensor[:, 0:2, :, :]  # flow from 0 to 1.
        flow_10 = flow_pred_tensor[:, 2:4, :, :]  # flow from 1 to 0.

        # estimated flow from t to 0
        est_flow_t0 = -(1 - t) * t * flow_01 + (t ** 2) * flow_10

        # estimated flow from t to 1
        est_flow_t1 = ((1 - t) ** 2) * flow_01 - t * (1 - t) * flow_10

        img_0 = img_tensor[:, 0:3, :, :]
        img_1 = img_tensor[:, 3:6, :, :]

        warped_img_1t = warp(img_1, est_flow_t1)  # backward warping
        warped_img_0t = warp(img_0, est_flow_t0)  # backward warping

        input_tensor = torch.cat([img_1, warped_img_1t, est_flow_t1,
                                  est_flow_t0, warped_img_0t, img_0], dim=1)

        return input_tensor

    def extract_outputs(self, output_tensor):
        """
        Extracts different elements in the output tensor.

        :param output_tensor: Output from the flow interpolation model.
        :return: The extract elements.
        """

        v_1t = output_tensor[:, 0, ...]  # Visibility Map 1-> t
        dflow_t1 = output_tensor[:, 1:3, ...]  # Residual of flow t->1
        dflow_t0 = output_tensor[:, 3:5, ...]  # Residual of flow t->0

        v_1t = v_1t[:, None, ...]  # making dimensions compatible

        v_1t = self.squash(v_1t)

        v_0t = 1 - v_1t  # Visibility Map 0->t

        return v_1t, dflow_t1, dflow_t0, v_0t

    def compute_output_image(self, img_tensor, input_tensor, output_tensor, t):
        """
        :param input_tensor: Input to flow interpolation model.
        :param output_tensor: Prediction from flow interpolation model
        :param t: Time step of interpolation (0 < t < 1)
        :return: I_t after enforcing constraints. B C H W
        """

        est_flow_t1 = input_tensor[:, 6:8, ...]  # Estimated flow t->1
        est_flow_t0 = input_tensor[:, 8:10, ...]  # Estimated flow t->0

        img_0 = img_tensor[:, 0:3, ...]
        img_1 = img_tensor[:, 3:6, ...]

        pred_v_1t, pred_dflow_t1, pred_dflow_t0, pred_v_0t = self.extract_outputs(output_tensor)

        pred_flow_t1 = est_flow_t1 + pred_dflow_t1
        pred_flow_t0 = est_flow_t0 + pred_dflow_t0

        pred_img_0t = warp(img_0, pred_flow_t0)  # backward warping to produce img at time t
        pred_img_1t = warp(img_1, pred_flow_t1)  # backward warping to produce img at time t

        pred_img_0t = pred_v_0t * pred_img_0t  # visibility map occlusion reasoning
        pred_img_1t = pred_v_1t * pred_img_1t  # visibility map occlusion reasoning

        weighted_sum = (1 - t) * pred_img_0t + t * pred_img_1t

        normalization_factor = (1 - t) * pred_v_0t + t * pred_v_1t  # Z (refer to paper)

        pred_img_t = weighted_sum / normalization_factor

        return pred_img_t


def get_model(path, in_channels, out_channels, cross_skip, stage=1, writer=None, cfg=None):

    if stage == 1:
        model = FlowComputationModel(in_channels, out_channels, cross_skip, cfg, stage, writer)

        if path:
            data = torch.load(path)
            if 'stage1_state_dict' in data.keys():
                log.info("Loading Stage 1 ResNet.")
                model.load_state_dict(data['stage1_state_dict'])
                log.info("Loaded weights for Flow Computation: "+str(path))
            else:
                model.load_state_dict(data)
        else:
            log.info("Not loading weights for stage %s."%stage)
        return model

    elif stage == 2:
        model = FlowInterpolationModel(in_channels, out_channels, cross_skip, cfg, stage, writer)

        if path:
            data = torch.load(path)
            if 'stage2_state_dict' in data.keys():
                log.info("Loading Stage 2 ResNet.")
                model.load_state_dict(data['stage2_state_dict'])
                log.info("Loaded weights for Flow Interpolation: " + str(path))
            else:
                model.load_state_dict(data)
        else:
            log.info("Not loading weights for stage %s." % stage)

        return model


