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
    def __init__(self, in_channels, out_channels, cross_skip=None, verbose=False, cfg = None):
        super(FlowComputationModel, self).__init__()
        self.norm_type = cfg.get("MODEL", "NORM_TYPE")
        self.bottleneck = cfg.get("MODEL", "BOTTLENECK")
        self.shortcut_type = cfg.get("MODEL", "SHORTCUT_TYPE")
        self.encoder_type = cfg.get("STAGE1", "ENCODER")
        log.info("Stage 1 model.")
        log.info("Encoder: %s. Normalization: %s. Bottleneck: %s. Shortcut: %s."%(self.encoder_type, self.norm_type,
                                                                                  self.bottleneck, self.shortcut_type))
        self.build_model(in_channels, out_channels)

    def build_model(self,in_channels, out_channels):

        # --------------------------------------
        # encoder
        # --------------------------------------
        if self.encoder_type == "resnet18":
            self.encoder = resnet18(in_channels=in_channels, norm_type=self.norm_type, shortcut_type=self.shortcut_type)

        elif self.encoder_type == "resnet34":
            raise NotImplementedError
            self.encoder = resnet34(
                3, zero_init_residual=True,
                temporal_downsampling_last_layer=temporal_downsampling_last_layer
            )
        else:
            raise NotImplementedError('Not supported encoder: %s'%self.encoder_type)

        # --------------------------------------
        # decoder
        # --------------------------------------
        def conv3x3_norm_relu(inplanes, planes, num_blocks=2):
            layers = []
            layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, padding=1))
            layers.append(make_norm_layer(self.norm_type, planes))
            layers.append(nn.ReLU())
            for _ in range(1, num_blocks):
                layers.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1))
                layers.append(make_norm_layer(self.norm_type, planes))
                layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        # 1/16
        self.upsample6 = nn.Upsample(scale_factor=2, mode='bilinear')

        if self.bottleneck == "CONV":
            log.info("Bottleneck: CONV.")
            self.layer6 = conv3x3_norm_relu(512, 512, 2)

        elif self.bottleneck == "CLSTM":
            log.info("Bottleneck: CLSTM")
            self.layer6 = ConvBLSTM(in_channels=512, hidden_channels=512,
                                    kernel_size=(3, 3), num_layers=2, batch_first=True)
        elif self.bottleneck == "CGRU":
            log.info("Bottleneck: CGRU")
            self.layer6 = ConvBGRU(in_channels=512, hidden_channels=512,
                                   kernel_size=(3, 3), num_layers=2, batch_first=True)

        # 1/8
        self.upsample7 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer7 = conv3x3_norm_relu(768, 256, 2)

        # 1/4
        self.upsample8 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer8 = conv3x3_norm_relu(384, 128, 2)

        # 1/2
        self.upsample9 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer9 = conv3x3_norm_relu(192, 64, 2)

        # 1/1
        self.upsample10 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer10 = conv3x3_norm_relu(128, 32, 2)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

    def apply_bottleneck(self, tensor_list):
        if self.bottleneck in ["CLSTM", "CGRU"]:
            x_fwd = torch.stack(tensor_list, dim=1)
            x_rev = torch.stack(tensor_list[::-1], dim=1)
            output = self.layer6(x_fwd, x_rev)
        elif self.bottleneck == "CONV":
            assert len(tensor_list) == 1, "Wrong number of timesteps."
            x_fwd = tensor_list[0]
            output = self.layer6(x_fwd)
            output = output[:, None, ...] # B C H W -> B 1 C H W

        return output

    def decoder(self, input_tensor, encoder_outputs):
        (x1, x2, x3, x4, x5) = encoder_outputs
        x7_in = torch.cat((x4, input_tensor), dim=1)
        x7_in = self.upsample7(x7_in)
        x7 = self.layer7(x7_in)  # [1/8, 1/8]

        x8_in = torch.cat((x3, x7), dim=1)
        x8_in = self.upsample8(x8_in)
        x8 = self.layer8(x8_in)  # [1/4, 1/4]

        x9_in = torch.cat((x2, x8), dim=1)
        x9_in = self.upsample9(x9_in)
        x9 = self.layer9(x9_in)  # [1/2, 1/2]

        x10_in = torch.cat((x1, x9), dim=1)
        x10_in = self.upsample10(x10_in)
        x10 = self.layer10(x10_in)  # [1, 1]

        final_out = self.final_conv(x10)

        # encoder out and decoding sent to next stage.
        return x5, final_out

    def forward(self, input_tensor):
        """
        :param input_tensor: input tensors -> B, T, C_in, H, W
        :param stage1_encoder_output: if skip connection from stage1 goes to stage2.
        :return: T tuples with <(B, C_6, H, W) tensor, (B, C_out, H, W)> in case of cross stage skip connection.
        else T tuples with <None, (B, C_out, H, W)> in case of cross stage skip connection.
        :return:
        """

        T = input_tensor.shape[1] # B T C H W
        if self.bottleneck == "CONV":
            assert T==1, "Expected 1 time step for CONV bottleneck. Found %s timesteps."%T
        else:
            assert T > 1, "Expected > 1 time step. Found %s timesteps." % T

        encodings = []

        bottleneck_in = []
        for t in range(T):
            x = input_tensor[:, t, ...]
            e = self.encoder(x, side_output=True)
            encodings.append(e)
            resnet_out = e[-1]
            resnet_out = self.upsample6(resnet_out) # x6_in = self.upsample6(x5)
            bottleneck_in.append(resnet_out)

        h = self.apply_bottleneck(bottleneck_in)
        assert len(h.shape) == 5, "Tensor not of shape: B T C H W"
        assert (h.shape[1] == input_tensor.shape[1]), "Number of time steps do not match"

        decodings = []

        for t in range(T):
            h_t = h[:, t, ...]
            e = encodings[t]
            # bottleneck output, and skip connections sent to decoder.
            d = self.decoder(h_t, e)
            decodings.append(d)

        return decodings


class FlowInterpolationModel(nn.Module):
    def __init__(self, in_channels, out_channels, cross_skip, verbose = False, cfg = None):
        super(FlowInterpolationModel, self).__init__()
        self.norm_type = cfg.get("MODEL", "NORM_TYPE")
        self.bottleneck = cfg.get("MODEL", "BOTTLENECK")
        self.shortcut_type = cfg.get("MODEL", "SHORTCUT_TYPE")
        self.encoder_type = cfg.get("STAGE2", "ENCODER")
        log.info("Stage 2 model.")
        log.info("Encoder: %s. Normalization: %s. Bottleneck: %s. Shortcut: %s."%(self.encoder_type, self.norm_type,
                                                                                  self.bottleneck, self.shortcut_type))
        self.build_model(in_channels, out_channels)

    def build_model(self, in_channels, out_channels):

        # --------------------------------------
        # encoder
        # --------------------------------------
        if self.encoder_type == "resnet18":
            self.encoder = resnet18(in_channels=in_channels, norm_type=self.norm_type, shortcut_type=self.shortcut_type)

        elif self.encoder_type == "resnet34":
            raise NotImplementedError
            self.encoder = resnet34(
                3, zero_init_residual=True,
                temporal_downsampling_last_layer=temporal_downsampling_last_layer
            )
        else:
            raise NotImplementedError('Not supported encoder: %s'%self.encoder_type)

        # --------------------------------------
        # decoder
        # --------------------------------------
        def conv3x3_norm_relu(inplanes, planes, num_blocks=2):
            layers = []
            layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, padding=1))
            layers.append(make_norm_layer(self.norm_type, planes))
            layers.append(nn.ReLU())
            for _ in range(1, num_blocks):
                layers.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1))
                layers.append(make_norm_layer(self.norm_type, planes))
                layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        # 1/16
        self.upsample6 = nn.Upsample(scale_factor=2, mode='bilinear')

        if self.bottleneck == "CONV":
            log.info("Bottleneck: CONV.")
            self.layer6 = conv3x3_norm_relu(1024, 512, num_blocks=2)

        elif self.bottleneck == "CLSTM":
            log.info("Bottleneck: CLSTM")
            self.layer6 = ConvBLSTM(in_channels=1024, hidden_channels=512,
                                    kernel_size=(3, 3), num_layers=2, batch_first=True)
        elif self.bottleneck == "CGRU":
            log.info("Bottleneck: CGRU")
            self.layer6 = ConvBGRU(in_channels=1024, hidden_channels=512,
                                   kernel_size=(3, 3), num_layers=2, batch_first=True)

        # 1/8
        self.upsample7 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer7 = conv3x3_norm_relu(768, 256, 2)

        # 1/4
        self.upsample8 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer8 = conv3x3_norm_relu(384, 128, 2)

        # 1/2
        self.upsample9 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer9 = conv3x3_norm_relu(192, 64, 2)

        # 1/1
        self.upsample10 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer10 = conv3x3_norm_relu(128, 32, 2)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

    def apply_bottleneck(self, tensor_list):
        if self.bottleneck in ["CLSTM", "CGRU"]:
            x_fwd = torch.stack(tensor_list, dim=1)
            x_rev = torch.stack(tensor_list[::-1], dim=1)
            output = self.layer6(x_fwd, x_rev)
        elif self.bottleneck == "CONV":
            assert len(tensor_list) == 1, "Wrong number of timesteps."
            x_fwd = tensor_list[0]
            output = self.layer6(x_fwd)
            output = output[:, None, ...] # B C H W -> B 1 C H W

        return output

    def decoder(self, input_tensor, encoder_outputs):
        (x1, x2, x3, x4, x5) = encoder_outputs
        x7_in = torch.cat((x4, input_tensor), dim=1)
        x7_in = self.upsample7(x7_in)
        x7 = self.layer7(x7_in)  # [1/8, 1/8]

        x8_in = torch.cat((x3, x7), dim=1)
        x8_in = self.upsample8(x8_in)
        x8 = self.layer8(x8_in)  # [1/4, 1/4]

        x9_in = torch.cat((x2, x8), dim=1)
        x9_in = self.upsample9(x9_in)
        x9 = self.layer9(x9_in)  # [1/2, 1/2]

        x10_in = torch.cat((x1, x9), dim=1)
        x10_in = self.upsample10(x10_in)
        x10 = self.layer10(x10_in)  # [1, 1]

        final_out = self.final_conv(x10)

        return  final_out

    def forward(self, input_tensor, stage1_outputs=None):
        """
        :param input_tensor: input tensors -> B, T, C_in, H, W
        :param stage1_encoder_output: if skip connection from stage1 goes to stage2.
        :return: T tuples with <(B, C_6, H, W) tensor, (B, C_out, H, W)> in case of cross stage skip connection.
        else T tuples with <None, (B, C_out, H, W)> in case of cross stage skip connection.
        :return:
        """

        T = input_tensor.shape[1] # B T C H W
        if self.bottleneck == "CONV":
            assert T==1, "Expected 1 time step for %s bottleneck. Found %s timesteps."%(self.bottleneck, T)
        else:
            assert T > 1, "Expected > 1 time step for %s bottleneck. Found %s timesteps."%(self.bottleneck, T)

        encodings = []

        bottleneck_in = []
        for t in range(T):
            x = input_tensor[:, t, ...]
            e = self.encoder(x, side_output=True)
            encodings.append(e)
            resnet_out = e[-1]
            if stage1_outputs is not None: # cross stage skip
                stage1_out = stage1_outputs[t]
                resnet_out = torch.cat([resnet_out, stage1_out], dim=1)
            resnet_out = self.upsample6(resnet_out) # x6_in = self.upsample6(x5)
            bottleneck_in.append(resnet_out)

        h = self.apply_bottleneck(bottleneck_in)
        assert len(h.shape) == 5, "Tensor not of shape: B T C H W"
        assert (h.shape[1] == input_tensor.shape[1]), "Number of time steps do not match"

        decodings = []

        for t in range(T):
            h_t = h[:, t, ...]
            e = encodings[t]
            # bottleneck output, and skip connections sent to decoder.
            d = self.decoder(h_t, e)
            enc_out = e[-1]
            decodings.append(d)

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

        v_1t = torch.sigmoid(v_1t)

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


def get_model(path, in_channels, out_channels, cross_skip, verbose=False, stage=1, cfg=None):

    if stage == 1:
        model = FlowComputationModel(in_channels, out_channels, cross_skip, verbose=verbose, cfg=cfg)

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
        model = FlowInterpolationModel(in_channels, out_channels, cross_skip, verbose=verbose, cfg=cfg)

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

    else:
        raise Exception("Expected stage = 1 or 2. Got stage = %s."%stage)
