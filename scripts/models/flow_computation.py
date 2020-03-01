import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .CLSTM.convgru import ConvBGRU
from .CLSTM.convlstm import ConvBLSTM
from .layers import avg_pool, conv

log = logging.getLogger(__name__)


class FlowComputationModel(nn.Module):
    def __init__(self, in_channels, out_channels, cross_skip, verbose=False, cfg=None):
        super(FlowComputationModel, self).__init__()
        self.cross_skip_connect = cross_skip
        self.cfg = cfg
        self.verbose = verbose
        self.bottleneck_type = self.cfg.get("STAGE1", "BOTTLENECK")
        log.info("Stage 1 model.")
        log.info("Encoder: UNET.  Bottleneck: %s.", self.bottleneck_type)

        # skip connection from stage1 to stage2
        self.build_model(in_channels, out_channels)

    def build_model(self, in_channels, out_channels):
        """
        :param in_channels: Number of channels for input tensor.
        :param out_channels: Number of channels for output tensor.
        :return:
        """

        # block 1

        self.conv1a = conv(
            in_planes=in_channels, out_planes=32, kernel_size=7, padding=3
        )
        self.conv1b = conv(in_planes=32, out_planes=32, kernel_size=7, padding=3)

        # block 2

        self.pool2 = avg_pool(kernel_size=2, stride=None, padding=0)  # 1/2
        self.conv2a = conv(in_planes=32, out_planes=64, kernel_size=5, padding=2)
        self.conv2b = conv(in_planes=64, out_planes=64, kernel_size=5, padding=2)

        # block 3

        self.pool3 = avg_pool(kernel_size=2, stride=None, padding=0)  # 1/4
        self.conv3a = conv(in_planes=64, out_planes=128, kernel_size=3)
        self.conv3b = conv(in_planes=128, out_planes=128, kernel_size=3)

        # block 4

        self.pool4 = avg_pool(kernel_size=2, stride=None, padding=0)  # 1/8
        self.conv4a = conv(in_planes=128, out_planes=256, kernel_size=3)
        self.conv4b = conv(in_planes=256, out_planes=256, kernel_size=3)

        # block 5

        self.pool5 = avg_pool(kernel_size=2, stride=None, padding=0)  # 1/16
        self.conv5a = conv(in_planes=256, out_planes=512, kernel_size=3)
        self.conv5b = conv(in_planes=512, out_planes=512, kernel_size=3)

        # block 6
        self.pool6 = avg_pool(kernel_size=2, stride=None, padding=0)  # 1/32

        if self.bottleneck_type == "CONV":
            self.conv6 = nn.Sequential(
                conv(512, 512, kernel_size=3), conv(512, 512, kernel_size=3)
            )

        elif self.bottleneck_type == "CLSTM":
            self.conv6 = ConvBLSTM(
                in_channels=512,
                hidden_channels=512,
                kernel_size=(3, 3),
                num_layers=2,
                batch_first=True,
            )
        elif self.bottleneck_type == "CGRU":
            self.conv6 = ConvBGRU(
                in_channels=512,
                hidden_channels=512,
                kernel_size=(3, 3),
                num_layers=2,
                batch_first=True,
            )

        # block 7

        self.upsample7 = lambda x: F.upsample(
            x, size=(2 * x.shape[2], 2 * x.shape[3]), mode="bilinear"
        )  # 2 x 2 upsampling

        # 1/16

        self.conv7a = conv(in_planes=512, out_planes=512, kernel_size=3)
        self.conv7b = conv(in_planes=512, out_planes=512, kernel_size=3)

        # block 8

        self.upsample8 = lambda x: F.upsample(
            x, size=(2 * x.shape[2], 2 * x.shape[3]), mode="bilinear"
        )  # 2 x 2 upsampling

        # 1/8

        self.conv8a = conv(in_planes=1024, out_planes=256, kernel_size=3)
        self.conv8b = conv(in_planes=256, out_planes=256, kernel_size=3)

        # block 9
        self.upsample9 = lambda x: F.upsample(
            x, size=(2 * x.shape[2], 2 * x.shape[3]), mode="bilinear"
        )  # 2 x 2 upsampling

        # 1/4

        self.conv9a = conv(in_planes=512, out_planes=128, kernel_size=3)
        self.conv9b = conv(in_planes=128, out_planes=128, kernel_size=3)

        # # block 10

        self.upsample10 = lambda x: F.upsample(
            x, size=(2 * x.shape[2], 2 * x.shape[3]), mode="bilinear"
        )  # 2 x 2 upsampling

        # 1/2

        self.conv10a = conv(in_planes=256, out_planes=64, kernel_size=3)
        self.conv10b = conv(in_planes=64, out_planes=64, kernel_size=3)

        # block 11

        self.upsample11 = lambda x: F.upsample(
            x, size=(2 * x.shape[2], 2 * x.shape[3]), mode="bilinear"
        )  # 2 x 2 upsampling
        # 1

        self.conv11a = conv(in_planes=128, out_planes=32, kernel_size=3)
        self.conv11b = conv(in_planes=32, out_planes=32, kernel_size=3)

        self.fuse_conv = conv(in_planes=64, out_planes=32, kernel_size=3)

        self.final_conv = nn.Conv2d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
        )

    def encoder(self, img_tensor):
        """ Encode the input `img_tensor`.


        :param img_tensor: B C H W tensor where images are C = 6 (2 images)
        :returns: each layer's output as a B C H W tensor
        :rtype: tuple of output tensors.

        """

        if self.verbose:
            log.info("Input: %s ", str(img_tensor.shape))

        conv1a_out = self.conv1a(img_tensor)
        conv1b_out = self.conv1b(conv1a_out)

        if self.verbose:
            log.info("Output Block 1: %s ", str(conv1b_out.shape))

        pool2_out = self.pool2(conv1b_out)
        conv2a_out = self.conv2a(pool2_out)
        conv2b_out = self.conv2b(conv2a_out)

        if self.verbose:
            log.info("Output Block 2: %s ", str(conv2b_out.shape))

        pool3_out = self.pool3(conv2b_out)
        conv3a_out = self.conv3a(pool3_out)
        conv3b_out = self.conv3b(conv3a_out)

        if self.verbose:
            log.info("Output Block 3: %s ", str(conv3b_out.shape))

        pool4_out = self.pool4(conv3b_out)
        conv4a_out = self.conv4a(pool4_out)
        conv4b_out = self.conv4b(conv4a_out)

        if self.verbose:
            log.info("Output Block 4: %s ", str(conv4b_out.shape))

        pool5_out = self.pool5(conv4b_out)
        conv5a_out = self.conv5a(pool5_out)
        conv5b_out = self.conv5b(conv5a_out)

        if self.verbose:
            log.info("Output Block 5: %s ", str(conv5b_out.shape))

        pool6_out = self.pool6(conv5b_out)

        return conv1b_out, conv2b_out, conv3b_out, conv4b_out, conv5b_out, pool6_out

    def bottleneck(self, tensor_list):

        if self.bottleneck_type in ["CLSTM", "CGRU"]:
            x_fwd = torch.stack(tensor_list, dim=1)
            x_rev = torch.stack(tensor_list[::-1], dim=1)
            output = self.conv6(x_fwd, x_rev)

        elif self.bottleneck_type == "CONV":
            output = []
            for x_fwd in tensor_list:
                out = self.conv6(x_fwd)
                output.append(out)
            output = torch.stack(output, dim=1)  # B C H W -> B 1 C H W

        return output

    def decoder(self, input_tensor, encoder_outputs):
        """
        :param input_tensor: output of LSTM.
        :param encoder_outputs: features from the encoder stages.
        :param stage1_encoder_output: Connection between stage1 and stage2.
        :return: Final result of the UNet as B, C, H, W tensor.
        """

        conv6_out = input_tensor
        conv1b_out, conv2b_out, conv3b_out, conv4b_out, conv5b_out, _ = encoder_outputs

        if self.verbose:
            log.info("Output Block 6 %s ", str(conv6_out.shape))

        conv7a_in = self.upsample7(conv6_out)

        conv7a_out = self.conv7a(conv7a_in)
        conv7b_out = self.conv7b(conv7a_out)

        if self.verbose:
            log.info("Output Block 7 %s ", str(conv7b_out.shape))

        conv8a_in = torch.cat([conv7b_out, conv5b_out], dim=1)
        conv8a_in = self.upsample8(conv8a_in)

        conv8a_out = self.conv8a(conv8a_in)
        conv8b_out = self.conv8b(conv8a_out)

        if self.verbose:
            log.info("Output Block 8 %s ", str(conv8b_out.shape))

        conv9a_in = torch.cat([conv8b_out, conv4b_out], dim=1)
        conv9a_in = self.upsample9(conv9a_in)

        conv9a_out = self.conv9a(conv9a_in)
        conv9b_out = self.conv9b(conv9a_out)

        if self.verbose:
            log.info("Output Block 9 %s ", str(conv9b_out.shape))

        conv10a_in = torch.cat([conv9b_out, conv3b_out], dim=1)
        conv10a_in = self.upsample10(conv10a_in)

        conv10a_out = self.conv10a(conv10a_in)
        conv10b_out = self.conv10b(conv10a_out)

        if self.verbose:
            log.info("Output Block 10 %s ", str(conv10b_out.shape))

        conv11a_in = torch.cat([conv10b_out, conv2b_out], dim=1)
        conv11a_in = self.upsample11(conv11a_in)

        conv11a_out = self.conv11a(conv11a_in)
        conv11b_out = self.conv11b(conv11a_out)

        fuse_in = torch.cat([conv11b_out, conv1b_out], dim=1)
        fuse_out = self.fuse_conv(fuse_in)

        final_out = self.final_conv(fuse_out)

        if self.verbose:
            log.info("Output Block 11 %s ", str(final_out.shape))

        if self.cross_skip_connect:
            encoder_output = conv6_out
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
            e = self.encoder(x)
            encodings.append(e)
            pool6_out = e[-1]
            bottleneck_in.append(pool6_out)

        h = self.bottleneck(bottleneck_in)
        assert len(h.shape) == 5, "Tensor not of shape: B T C H W"
        assert h.shape[1] == unet_in.shape[1], "Number of time steps do not match"

        decodings = []

        for t in range(T):
            h_t = h[:, t, ...]
            e = encodings[t]
            d = self.decoder(h_t, e)
            # bottleneck output, and skip connections sent to decoder.
            decodings.append(d)

        return decodings
