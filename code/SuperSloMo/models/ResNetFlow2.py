import sys
sys.path.insert(0, "/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/SuperSloMo/models/")
from layers import *
import torch.nn as nn
import torch.nn.functional as F
from resnet2D import *

import torch
import logging

log = logging.getLogger(__name__)


class FlowComputationModel(nn.Module):

    def __init__(self, in_channels, out_channels, cross_skip=None, verbose=False, cfg = None):
        super(FlowComputationModel, self).__init__()
        log.info("USING UPDATED RESNET MODEL.")
        self.norm_type = cfg.get("MODEL", "NORM_TYPE")
        self.bottleneck_type = cfg.get("MODEL", "BOTTLENECK")
        self.shortcut_type = cfg.get("MODEL", "SHORTCUT_TYPE")
        self.encoder_type = cfg.get("STAGE1", "ENCODER")
        self.cross_skip_connect= cross_skip
        log.info("Stage 1 model.")
        log.info("Encoder: %s. Normalization: %s. Bottleneck: %s. Shortcut: %s."%(self.encoder_type, self.norm_type,
                                                                                  self.bottleneck_type, self.shortcut_type))
        self.build_model(in_channels, out_channels)
        self.verbose=False

    def build_model(self, in_channels, out_channels):
        """
        :param in_channels: Number of channels for input tensor.
        :param out_channels: Number of channels for output tensor.
        :return:
        """
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

        if self.bottleneck_type == "CONV":
            self.conv6 = conv3x3_norm_relu(512, 512, num_blocks=2)

        elif self.bottleneck_type == "CLSTM":
            self.conv6 = ConvBLSTM(in_channels=512, hidden_channels=512,
                                   kernel_size=(3, 3), num_layers=2, batch_first=True)
        elif self.bottleneck_type == "CGRU":
            self.conv6 = ConvBGRU(in_channels=512, hidden_channels=512,
                                  kernel_size=(3, 3), num_layers=2, batch_first=True)

        # block 7

        self.upsample7 = nn.Upsample(scale_factor=2, mode='bilinear')  # 2 x 2 upsampling

        # 1/16
        self.conv7 = conv3x3_norm_relu(512, 512, num_blocks=2)

        # block 8

        self.upsample8 = nn.Upsample(scale_factor=2, mode='bilinear')  # 2 x 2 upsampling

        # 1/8
        self.conv8 = conv3x3_norm_relu(1024, 256, num_blocks=2)

        # block 9
        self.upsample9 = nn.Upsample(scale_factor=2, mode='bilinear')  # 2 x 2 upsampling

        # 1/4
        self.conv9 = conv3x3_norm_relu(512, 128, num_blocks=2)

        # # block 10

        self.upsample10 = nn.Upsample(scale_factor=2, mode='bilinear')  # 2 x 2 upsampling

        # 1/2
        self.conv10 = conv3x3_norm_relu(256, 64, num_blocks=2)

        # block 11

        self.upsample11 = nn.Upsample(scale_factor=2, mode='bilinear')  # 2 x 2 upsampling
        # 1

        self.conv11 = conv3x3_norm_relu(128, 64, num_blocks=2)

        self.fuse_conv = conv3x3_norm_relu(128, 64, num_blocks=1)

        self.final_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                    dilation=1, bias=True)

    def bottleneck(self, tensor_list):
        if self.bottleneck_type in ["CLSTM", "CGRU"]:
            x_fwd = torch.stack(tensor_list, dim=1)
            x_rev = torch.stack(tensor_list[::-1], dim=1)
            output = self.conv6(x_fwd, x_rev)
        elif self.bottleneck_type == "CONV":
            assert len(tensor_list) == 1, "Wrong number of timesteps."
            x_fwd = tensor_list[0]
            output = self.conv6(x_fwd)
            output = output[:, None, ...]  # B C H W -> B 1 C H W
        return output

    def decoder(self, input_tensor, encoder_outputs):
        """
        :param input_tensor: output of LSTM.
        :param encoder_outputs: features from the encoder stages.
        :param stage1_encoder_output: Connection between stage1 and stage2.
        :return: Final result of the UNet as B, C, H, W tensor.
        """

        conv6_out = input_tensor
        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out = encoder_outputs

        conv5_out = self.upsample7(conv5_out)
        conv6_out = self.upsample7(conv6_out)
        conv7_out = self.conv7(conv6_out) #  1/16
        conv7_out = torch.cat([conv7_out, conv5_out], dim=1)  # 1/16

        conv8_out = self.conv8(conv7_out)
        conv8_out = torch.cat([conv8_out, conv4_out], dim=1) # 1/16
        conv8_out = self.upsample8(conv8_out) # 1/8

        conv9_out = self.conv9(conv8_out)
        conv9_out = torch.cat([conv9_out, conv3_out], dim=1) # 1/8
        conv9_out = self.upsample9(conv9_out) # 1/4

        conv10_out = self.conv10(conv9_out)
        conv10_out = torch.cat([conv10_out, conv2_out], dim=1) # 1/8
        conv10_out = self.upsample10(conv10_out) # 1/4

        conv11_out = self.conv11(conv10_out)
        conv11_out = torch.cat([conv11_out, conv1_out], dim=1) # 1/8
        conv11_out = self.upsample11(conv11_out) # 1/4

        fuse_out = self.fuse_conv(conv11_out)
        final_out = self.final_conv(fuse_out)

        if self.cross_skip_connect:
            encoder_output = input_tensor
            return encoder_output, final_out
        else:
            return None, final_out

    def forward(self, unet_in):
        """
        :param input_tensor: input tensors -> B, T, C_in, H, W
        :param stage1_encoder_output: if skip connection from stage1 goes to stage2.
        :return: T tuples with <(B, C_6, H, W) tensor, (B, C_out, H, W)> in case of cross stage skip connection.
        else T tuples with <None, (B, C_out, H, W)> in case of cross stage skip connection.
        :return:
        """

        T = unet_in.shape[1]  # B T C H W

        encodings = []

        bottleneck_in = []
        for t in range(T):
            x = unet_in[:, t, ...]
            e = self.encoder(x, side_output=True)
            encodings.append(e)
            pool6_out = e[-1]
            bottleneck_in.append(pool6_out)
        h = self.bottleneck(bottleneck_in)
        assert len(h.shape) == 5, "Tensor not of shape: B T C H W"
        assert (h.shape[1] == unet_in.shape[1]), "Number of time steps do not match"

        decodings = []

        for t in range(T):
            h_t = h[:, t, ...]
            e = encodings[t]
            d = self.decoder(h_t, e)
            # bottleneck output, and skip connections sent to decoder.
            decodings.append(d)

        return decodings


class FlowInterpolationModel(nn.Module):

    def __init__(self, in_channels, out_channels, cross_skip=None, verbose=False, cfg = None):
        super(FlowInterpolationModel, self).__init__()
        self.norm_type = cfg.get("MODEL", "NORM_TYPE")
        self.bottleneck_type = cfg.get("MODEL", "BOTTLENECK")
        self.shortcut_type = cfg.get("MODEL", "SHORTCUT_TYPE")
        self.encoder_type = cfg.get("STAGE1", "ENCODER")
        self.cross_skip_connect= cross_skip

        log.info("Stage 1 model.")
        log.info("Encoder: %s. Normalization: %s. Bottleneck: %s. Shortcut: %s."%(self.encoder_type, self.norm_type,
                                                                                  self.bottleneck_type, self.shortcut_type))
        self.build_model(in_channels, out_channels)
        self.verbose=False

    def build_model(self, in_channels, out_channels):
        """
        :param in_channels: Number of channels for input tensor.
        :param out_channels: Number of channels for output tensor.
        :return:
        """
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

        if self.bottleneck_type == "CONV":
            self.conv6 = conv3x3_norm_relu(512, 512, num_blocks=2)

        elif self.bottleneck_type == "CLSTM":
            self.conv6 = ConvBLSTM(in_channels=512, hidden_channels=512,
                                   kernel_size=(3, 3), num_layers=2, batch_first=True)
        elif self.bottleneck_type == "CGRU":
            self.conv6 = ConvBGRU(in_channels=512, hidden_channels=512,
                                  kernel_size=(3, 3), num_layers=2, batch_first=True)

        # block 7

        self.upsample7 = nn.Upsample(scale_factor=2, mode='bilinear')  # 2 x 2 upsampling
        if self.cross_skip_connect:
            self.conv7 = conv3x3_norm_relu(1024, 512, num_blocks=2)
        else:
            self.conv7 = conv3x3_norm_relu(512, 512, num_blocks=2)

        # 1/8

        self.upsample8 = nn.Upsample(scale_factor=2, mode='bilinear')  # 2 x 2 upsampling

        # 1/8
        self.conv8 = conv3x3_norm_relu(1024, 256, num_blocks=2)

        # block 9
        self.upsample9 = nn.Upsample(scale_factor=2, mode='bilinear')  # 2 x 2 upsampling

        # 1/4
        self.conv9 = conv3x3_norm_relu(512, 128, num_blocks=2)

        # # block 10

        self.upsample10 = nn.Upsample(scale_factor=2, mode='bilinear')  # 2 x 2 upsampling

        # 1/2
        self.conv10 = conv3x3_norm_relu(256, 64, num_blocks=2)

        # block 11

        self.upsample11 = nn.Upsample(scale_factor=2, mode='bilinear')  # 2 x 2 upsampling
        # 1

        self.conv11 = conv3x3_norm_relu(128, 64, num_blocks=2)

        self.fuse_conv = conv3x3_norm_relu(128, 64, num_blocks=1)

        self.final_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                    dilation=1, bias=True)

    def bottleneck(self, tensor_list):
        if self.bottleneck_type in ["CLSTM", "CGRU"]:
            x_fwd = torch.stack(tensor_list, dim=1)
            x_rev = torch.stack(tensor_list[::-1], dim=1)
            output = self.conv6(x_fwd, x_rev)
        elif self.bottleneck_type == "CONV":
            assert len(tensor_list) == 1, "Wrong number of timesteps."
            x_fwd = tensor_list[0]
            output = self.conv6(x_fwd)
            output = output[:, None, ...]  # B C H W -> B 1 C H W
        return output

    def decoder(self, input_tensor, encoder_outputs, stage1_encoder_output=None):
        """
        :param input_tensor: output of LSTM.
        :param encoder_outputs: features from the encoder stages.
        :param stage1_encoder_output: Connection between stage1 and stage2.
        :return: Final result of the UNet as B, C, H, W tensor.
        """

        conv6_out = input_tensor
        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out = encoder_outputs

        conv5_out = self.upsample7(conv5_out)

        if self.cross_skip_connect:
            concat_out = torch.cat([conv6_out, stage1_encoder_output], dim=1)
            # concatenate encoder outputs
            conv6_out = self.upsample7(concat_out)
            # upsample everything
        else:  # only upsample and concatenate.
            conv6_out = self.upsample7(conv6_out)

        conv7_out = self.conv7(conv6_out) #  1/16
        conv7_out = torch.cat([conv7_out, conv5_out], dim=1)  # 1/16

        conv8_out = self.conv8(conv7_out)
        conv8_out = torch.cat([conv8_out, conv4_out], dim=1) # 1/16
        conv8_out = self.upsample8(conv8_out) # 1/8

        conv9_out = self.conv9(conv8_out)
        conv9_out = torch.cat([conv9_out, conv3_out], dim=1) # 1/8
        conv9_out = self.upsample9(conv9_out) # 1/4

        conv10_out = self.conv10(conv9_out)
        conv10_out = torch.cat([conv10_out, conv2_out], dim=1) # 1/8
        conv10_out = self.upsample10(conv10_out) # 1/4

        conv11_out = self.conv11(conv10_out)
        conv11_out = torch.cat([conv11_out, conv1_out], dim=1) # 1/8
        conv11_out = self.upsample11(conv11_out) # 1/4

        fuse_out = self.fuse_conv(conv11_out)
        final_out = self.final_conv(fuse_out)

        return final_out

    def forward(self, unet_in, stage1_outputs=None):
        """
        :param input_tensor: input tensors -> B, T, C_in, H, W
        :param stage1_encoder_output: if skip connection from stage1 goes to stage2.
        :return: T tuples with <(B, C_6, H, W) tensor, (B, C_out, H, W)> in case of cross stage skip connection.
        else T tuples with <None, (B, C_out, H, W)> in case of cross stage skip connection.
        :return:
        """

        T = unet_in.shape[1]  # B T C H W

        encodings = []

        bottleneck_in = []
        for t in range(T):
            x = unet_in[:, t, ...]
            e = self.encoder(x, side_output=True)
            encodings.append(e)
            pool6_out = e[-1]
            bottleneck_in.append(pool6_out)
        h = self.bottleneck(bottleneck_in)
        assert len(h.shape) == 5, "Tensor not of shape: B T C H W"
        assert (h.shape[1] == unet_in.shape[1]), "Number of time steps do not match"
        decodings = []

        for t in range(T):
            h_t = h[:, t, ...]
            e = encodings[t]
            stage1_out = stage1_outputs[t]
            d = self.decoder(h_t, e, stage1_out)
            # bottleneck output, and skip connections sent to decoder.
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

        if self.verbose:
            log.info("Generated Input tensor of shape:" + str(input_tensor.shape))

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
                log.info("Loading Stage 1 UNet.")
                model.load_state_dict(data['stage1_state_dict'])
                log.info("Loaded weights for Flow Computation: " + str(path))
            else:
                model.load_state_dict(data)
        else:
            log.info("Not loading weights for stage %s." % stage)
        return model

    elif stage == 2:
        model = FlowInterpolationModel(in_channels, out_channels, cross_skip, verbose=verbose, cfg=cfg)

        if path:
            data = torch.load(path)
            if 'stage2_state_dict' in data.keys():
                log.info("Loading Stage 2 UNet.")
                model.load_state_dict(data['stage2_state_dict'])
                log.info("Loaded weights for Flow Interpolation: " + str(path))
            else:
                model.load_state_dict(data)
        else:
            log.info("Not loading weights for stage %s." % stage)

        return model

    else:
        raise Exception("Expected stage = 1 or 2. Got stage = %s." % stage)


if __name__ == '__main__':
    logging.basicConfig(filename="test.log", level=logging.INFO)

    flowC_model = get_model(path=None, in_channels=6, out_channels=4, cross_skip=True, verbose=True, stage=1)
    flowI_model = get_model(path=None, in_channels=16, out_channels=5, cross_skip=True, verbose=True, stage=2)

    flowC_model = flowC_model.cuda()
    flowI_model = flowI_model.cuda()

    stage1_input = torch.randn([1, 6, 320, 640]).cuda()
    encoder_out, flow_tensor = flowC_model(stage1_input)
    log.info("Encoder: " + str(encoder_out.shape))
    stage2_input = torch.randn([1, 16, 320, 640]).cuda()
    stage2_out = flowI_model(stage2_input, encoder_out)
    logging.info("Done.")

##########################################
# // And all you touch and all you see,//#
# // Is all your life will ever be!    //#
##########################################
