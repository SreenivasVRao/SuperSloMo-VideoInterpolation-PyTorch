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
        self.verbose= False

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
            decoder_dims = [512, 256, 128, 64, 64]

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
                nn.Conv2d(encoder_dims[-1], encoder_dims[-1], kernel_size=3, padding = 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(encoder_dims[-1], encoder_dims[-1], kernel_size=3, padding = 1, bias=False),
                nn.ReLU(inplace=True)
                )

        # --------------------------------------
        # decoder
        # --------------------------------------

        # 1/16
        self.conv7a = conv(in_planes=decoder_dims[0], out_planes=decoder_dims[0], kernel_size=3)
        self.conv7b = conv(in_planes=decoder_dims[0], out_planes=decoder_dims[0], kernel_size=3)

        # block 8

        self.upsample8 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                              mode='bilinear')  # 2 x 2 upsampling

        # 1/8

        self.conv8a = conv(in_planes=2*decoder_dims[0], out_planes=decoder_dims[1], kernel_size=3)
        self.conv8b = conv(in_planes=decoder_dims[1], out_planes=decoder_dims[1], kernel_size=3)


        # block 9
        self.upsample9 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                              mode='bilinear')  # 2 x 2 upsampling

        # 1/4

        self.conv9a = conv(in_planes=2*decoder_dims[1], out_planes=decoder_dims[2], kernel_size=3)
        self.conv9b = conv(in_planes=decoder_dims[2], out_planes=decoder_dims[2], kernel_size=3)

        # # block 10

        self.upsample10 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                               mode='bilinear')  # 2 x 2 upsampling

        # 1/2

        self.conv10a = conv(in_planes=2*decoder_dims[2], out_planes=decoder_dims[3], kernel_size=3)
        self.conv10b = conv(in_planes=decoder_dims[3], out_planes=decoder_dims[3], kernel_size=3)

        # block 11

        self.upsample11 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                               mode='bilinear')  # 2 x 2 upsampling
        # 1

        self.conv11a = conv(in_planes=2*decoder_dims[3], out_planes=decoder_dims[4], kernel_size=3)
        self.conv11b = conv(in_planes=decoder_dims[4], out_planes=decoder_dims[4], kernel_size=3)

        self.fuse_conv = conv(in_planes=2*decoder_dims[4], out_planes=decoder_dims[4]//2, kernel_size=3)

        self.final_conv = nn.Conv2d(in_channels=decoder_dims[4]//2, out_channels=out_channels,
                                    kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.upsample12 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                              mode='bilinear')  # 2 x 2 upsampling

    def apply_bottleneck(self, tensor_list):
        if self.cfg.get("MODEL", "BOTTLENECK") in ["CLSTM", "CGRU"]:
            x_fwd = torch.stack(tensor_list, dim=1)
            x_rev = torch.stack(tensor_list[::-1], dim=1)
            output = self.bottleneck_layer(x_fwd, x_rev)
        elif self.cfg.get("MODEL", "BOTTLENECK") == "CONV":
            assert len(tensor_list) == 1, "Wrong number of timesteps."
            x_fwd = tensor_list[0]
            output = self.bottleneck_layer(x_fwd)
            output = output[:, None, ...] # B C H W -> B 1 C H W

        return output

    def decoder(self, input_tensor, encoder_outputs):
        """
        :param input_tensor: output of LSTM.
        :param encoder_outputs: features from the encoder stages.
        :param stage1_encoder_output: Connection between stage1 and stage2.
        :return: Final result of the UNet as B, C, H, W tensor.
        """

        conv6_out = input_tensor
        conv1b_out, conv2b_out, conv3b_out, conv4b_out, conv5b_out = encoder_outputs


        if self.verbose:
            log.info("Output Block 6: " + str(conv6_out.shape))

        # conv7a_in = self.upsample7(conv6_out)

        conv7a_out = self.conv7a(conv6_out)
        conv7b_out = self.conv7b(conv7a_out)

        if self.verbose:
            log.info("Output Block 7: " + str(conv7b_out.shape))

        conv8a_in = torch.cat([conv7b_out, conv5b_out], dim=1)
        conv8a_in = self.upsample8(conv8a_in)

        conv8a_out = self.conv8a(conv8a_in)
        conv8b_out = self.conv8b(conv8a_out)

        if self.verbose:
            log.info("Output Block 8: " + str(conv8b_out.shape))

        conv9a_in = torch.cat([conv8b_out, conv4b_out], dim=1)
        conv9a_in = self.upsample9(conv9a_in)

        conv9a_out = self.conv9a(conv9a_in)
        conv9b_out = self.conv9b(conv9a_out)

        if self.verbose:
            log.info("Output Block 9: " + str(conv9b_out.shape))

        conv10a_in = torch.cat([conv9b_out, conv3b_out], dim=1)
        conv10a_in = self.upsample10(conv10a_in)

        conv10a_out = self.conv10a(conv10a_in)
        conv10b_out = self.conv10b(conv10a_out)

        if self.verbose:
            log.info("Output Block 10: " + str(conv10b_out.shape))

        conv11a_in = torch.cat([conv10b_out, conv2b_out], dim=1)
        conv11a_in = self.upsample11(conv11a_in)

        conv11a_out = self.conv11a(conv11a_in)
        conv11b_out = self.conv11b(conv11a_out)

        fuse_in = torch.cat([conv11b_out, conv1b_out], dim=1)
        fuse_in = self.upsample12(fuse_in)
        fuse_out = self.fuse_conv(fuse_in)

        final_out = self.final_conv(fuse_out)

        if self.verbose:
            log.info("Output Block 11: " + str(final_out.shape))

        return final_out

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
            d = self.decoder(h_t, e)
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
        self.verbose= False

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
        assert self.norm_type.lower() == "bn", "Unsupported normalization method: %s" % self.norm_type

        # --------------------------------------
        # encoder
        # --------------------------------------
        # sample_size = self.cfg.getint('TRAIN', 'CROP_IMH')
        # sample_duration = self.cfg.getint('MISC', 'SAMPLE_DURATION')
        if self.cfg.get("STAGE2", "ENCODER") == "resnet18":
            log.info("Stage 2: ResNet18.")
            self.encoder = resnet18(in_channels=in_channels)
            encoder_dims = [64, 128, 256, 512]
            decoder_dims = [512, 256, 128, 64, 64]

        elif self.cfg.get("STAGE2", "ENCODER") == "resnet34":
            raise NotImplementedError
            self.encoder = resnet34(
                3, zero_init_residual=True,
                temporal_downsampling_last_layer=temporal_downsampling_last_layer
            )
            encoder_dims = [64, 128, 256, 512]

        else:
            raise NotImplementedError('Not supported 3D encoder: {}'.format(self.cfg.get('STAGE1', 'ENCODER')))

        if self.cfg.get("MODEL", "BOTTLENECK") == "CLSTM":
            log.info("Bottleneck Stage 2: CLSTM")
            self.bottleneck_layer = ConvBLSTM(in_channels=encoder_dims[-1], hidden_channels=encoder_dims[-1],
                                              kernel_size=(3, 3), num_layers=2, batch_first=True)
        elif self.cfg.get("MODEL", "BOTTLENECK") == "CGRU":
            log.info("Bottleneck Stage 2: CGRU")
            self.bottleneck_layer = ConvBGRU(in_channels=encoder_dims[-1], hidden_channels=encoder_dims[-1],
                                             kernel_size=(3, 3), num_layers=2, batch_first=True)
        elif self.cfg.get("MODEL", "BOTTLENECK") == "CONV":
            log.info("Bottleneck Stage 2: CONV")
            self.bottleneck_layer = nn.Sequential(
                nn.Conv2d(encoder_dims[-1], encoder_dims[-1], kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(encoder_dims[-1], encoder_dims[-1], kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )

        # --------------------------------------
        # decoder
        # --------------------------------------

        # 1/16

        if self.cross_skip_connect:
            self.conv7a = conv(in_planes=2 * decoder_dims[0], out_planes=decoder_dims[0], kernel_size=3)
        else:
            self.conv7a = conv(in_planes=decoder_dims[0], out_planes=decoder_dims[0], kernel_size=3)

        self.conv7b = conv(in_planes=decoder_dims[0], out_planes=decoder_dims[0], kernel_size=3)

        # block 8

        self.upsample8 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                              mode='bilinear')  # 2 x 2 upsampling

        # 1/8

        self.conv8a = conv(in_planes=2 * decoder_dims[0], out_planes=decoder_dims[1], kernel_size=3)
        self.conv8b = conv(in_planes=decoder_dims[1], out_planes=decoder_dims[1], kernel_size=3)

        # block 9
        self.upsample9 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                              mode='bilinear')  # 2 x 2 upsampling

        # 1/4

        self.conv9a = conv(in_planes=2 * decoder_dims[1], out_planes=decoder_dims[2], kernel_size=3)
        self.conv9b = conv(in_planes=decoder_dims[2], out_planes=decoder_dims[2], kernel_size=3)

        # # block 10

        self.upsample10 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                               mode='bilinear')  # 2 x 2 upsampling

        # 1/2

        self.conv10a = conv(in_planes=2 * decoder_dims[2], out_planes=decoder_dims[3], kernel_size=3)
        self.conv10b = conv(in_planes=decoder_dims[3], out_planes=decoder_dims[3], kernel_size=3)

        # block 11

        self.upsample11 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                               mode='bilinear')  # 2 x 2 upsampling
        # 1

        self.conv11a = conv(in_planes=2 * decoder_dims[3], out_planes=decoder_dims[4], kernel_size=3)
        self.conv11b = conv(in_planes=decoder_dims[4], out_planes=decoder_dims[4], kernel_size=3)

        self.fuse_conv = conv(in_planes=2 * decoder_dims[4], out_planes=decoder_dims[4] // 2, kernel_size=3)

        self.final_conv = nn.Conv2d(in_channels=decoder_dims[4] // 2, out_channels=out_channels,
                                    kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.upsample12 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                              mode='bilinear')  # 2 x 2 upsampling

    def decoder(self, input_tensor, encoder_outputs, stage1_encoder_output=None):
        """
        :param input_tensor: output of LSTM.
        :param encoder_outputs: features from the encoder stages.
        :param stage1_encoder_output: Connection between stage1 and stage2.
        :return: Final result of the UNet as B, C, H, W tensor.
        """

        conv6_out = input_tensor
        conv1b_out, conv2b_out, conv3b_out, conv4b_out, conv5b_out = encoder_outputs

        if self.verbose:
            log.info("Output Block 6: " + str(conv6_out.shape))

        if self.cross_skip_connect:
            conv7a_in  = torch.cat([conv6_out, stage1_encoder_output], dim=1)
        else:
            conv7a_in = conv6_out

        #     conv7a_in = self.upsample7(concat_out)
        #     # upsample everything
        # else:  # only upsample and concatenate.
        #     conv7a_in = self.upsample7(conv6_out)

        conv7a_out = self.conv7a(conv7a_in)
        conv7b_out = self.conv7b(conv7a_out)

        if self.verbose:
            log.info("Output Block 7: " + str(conv7b_out.shape))

        conv8a_in = torch.cat([conv7b_out, conv5b_out], dim=1)
        conv8a_in = self.upsample8(conv8a_in)

        conv8a_out = self.conv8a(conv8a_in)
        conv8b_out = self.conv8b(conv8a_out)

        if self.verbose:
            log.info("Output Block 8: " + str(conv8b_out.shape))

        conv9a_in = torch.cat([conv8b_out, conv4b_out], dim=1)
        conv9a_in = self.upsample9(conv9a_in)

        conv9a_out = self.conv9a(conv9a_in)
        conv9b_out = self.conv9b(conv9a_out)

        if self.verbose:
            log.info("Output Block 9: " + str(conv9b_out.shape))

        conv10a_in = torch.cat([conv9b_out, conv3b_out], dim=1)
        conv10a_in = self.upsample10(conv10a_in)

        conv10a_out = self.conv10a(conv10a_in)
        conv10b_out = self.conv10b(conv10a_out)

        if self.verbose:
            log.info("Output Block 10: " + str(conv10b_out.shape))

        conv11a_in = torch.cat([conv10b_out, conv2b_out], dim=1)
        conv11a_in = self.upsample11(conv11a_in)

        conv11a_out = self.conv11a(conv11a_in)
        conv11b_out = self.conv11b(conv11a_out)

        fuse_in = torch.cat([conv11b_out, conv1b_out], dim=1)
        fuse_in = self.upsample12(fuse_in)
        fuse_out = self.fuse_conv(fuse_in)

        final_out = self.final_conv(fuse_out)
        if self.verbose:
            log.info("Output Block 11: " + str(final_out.shape))

        return final_out

    def apply_bottleneck(self, tensor_list):
        if self.cfg.get("MODEL","BOTTLENECK") in ["CLSTM", "CGRU"]:
            x_fwd = torch.stack(tensor_list, dim=1)
            x_rev = torch.stack(tensor_list[::-1], dim=1)
            output = self.bottleneck_layer(x_fwd, x_rev)
        elif self.cfg.get("MODEL", "BOTTLENECK") == "CONV":
            assert len(tensor_list)==1, "Wrong number of timesteps."
            x_fwd = tensor_list[0]
            output = self.bottleneck_layer(x_fwd)
            output = output[:, None, ...] # B C H W -> B 1 C H W

        return output

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

            d = self.decoder(h_t, e, enc_stage1)
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
