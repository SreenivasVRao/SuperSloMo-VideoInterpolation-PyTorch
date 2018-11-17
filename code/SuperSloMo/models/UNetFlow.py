from layers import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging

log = logging.getLogger(__name__)

class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=False, verbose=False):
        super(UNet, self).__init__()
        self.batchNorm = batch_norm
        self.verbose = verbose
        self.build_model(in_channels, out_channels)
        self.squash = nn.Sigmoid()

    def build_model(self, in_channels, out_channels):
        """

        :param in_channels: Number of channels for input tensor.
        :param out_channels: Number of channels for output tensor.
        :return:
        """

        # block 1

        self.conv1a = conv(in_planes=in_channels, out_planes=32, kernel_size=7, padding=3)
        self.conv1b = conv(in_planes=32, out_planes=32, kernel_size=7, padding=3)

        # block 2

        self.pool2 = avg_pool(kernel_size=2, stride=None, padding=0) # 1/2
        self.conv2a = conv(in_planes=32, out_planes=64, kernel_size=5, padding=2)
        self.conv2b = conv(in_planes=64, out_planes=64, kernel_size=5, padding=2)

        # block 3

        self.pool3 = avg_pool(kernel_size=2, stride=None, padding=0) # 1/4
        self.conv3a = conv(in_planes=64, out_planes=128,kernel_size=3)
        self.conv3b = conv(in_planes=128, out_planes=128,kernel_size=3)


        # block 4

        self.pool4 = avg_pool(kernel_size=2, stride=None, padding=0) # 1/8
        self.conv4a = conv(in_planes=128, out_planes=256,kernel_size=3)
        self.conv4b = conv(in_planes=256, out_planes=256, kernel_size=3)

        # block 5

        self.pool5 = avg_pool(kernel_size=2, stride=None, padding=0) # 1/16
        self.conv5a = conv(in_planes=256, out_planes=512, kernel_size=3)
        self.conv5b = conv(in_planes=512, out_planes=512, kernel_size=3)

        # block 6
        self.pool6 = avg_pool(kernel_size=2, stride=None, padding=0) # 1/32
        self.conv6a = conv(in_planes=512, out_planes=512, kernel_size=3)
        self.conv6b = conv(in_planes=512, out_planes=512, kernel_size=3)

        # block 7

        self.upsample7 = lambda x: F.upsample(x, scale_factor=2,
                                              mode='bilinear')  # 2 x 2 upsampling
        # 1/16

        self.conv7a = conv(in_planes=512, out_planes=512,kernel_size=3)
        self.conv7b = conv(in_planes=512, out_planes=512, kernel_size=3)


        # block 8

        self.upsample8 = lambda x: F.upsample(x, scale_factor=2,
                                              mode='bilinear')  # 2 x 2 upsampling
        # 1/8

        self.conv8a = conv(in_planes=512, out_planes=256, kernel_size=3)
        self.conv8b = conv(in_planes=256, out_planes=256, kernel_size=3)


        # block 9

        self.upsample9 = lambda x: F.upsample(x, scale_factor=2,
                                              mode='bilinear')  # 2 x 2 upsampling
        # 1/4

        self.conv9a = conv(in_planes=256, out_planes=128, kernel_size=3)
        self.conv9b = conv(in_planes=128, out_planes=128, kernel_size=3)

        # # block 10
        #
        self.upsample10 = lambda x: F.upsample(x, scale_factor=2,
                                               mode='bilinear')  # 2 x 2 upsampling
        # 1/2

        self.conv10a = conv(in_planes=128, out_planes=64, kernel_size=3)
        self.conv10b = conv(in_planes=64, out_planes=64, kernel_size=3)

        # block 11

        self.upsample11 = lambda x: F.upsample(x, scale_factor=2,
                                               mode='bilinear')  # 2 x 2 upsampling
        # 1

        self.conv11a = conv(in_planes=64, out_planes=32, kernel_size=3)
        self.conv11b = nn.Sequential(nn.ReplicationPad2d(1),
                                       nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True))


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_normal(m.weight.data, mode='fan_in')
        #         if m.bias is not None:
        #             m.bias.data.zero_()


    def forward(self, flowI_input):
        """
        :param input_tensor: input: N,16, H, W,
        batch_size = N

        :return: output_tensor: N, 5, H, W, C
        interpolation result

        """

        if self.verbose:
            log.info("Input: " + str(flowI_input.shape))

        conv1a_out = self.conv1a(flowI_input)
        conv1b_out = self.conv1b(conv1a_out)

        if self.verbose:
            log.info("Output Block 1: "+str(conv1b_out.shape))

        pool2_out  = self.pool2(conv1b_out)
        conv2a_out = self.conv2a(pool2_out)
        conv2b_out = self.conv2b(conv2a_out)

        if self.verbose:
            log.info("Output Block 2: "+str(conv2b_out.shape))

        pool3_out  = self.pool3(conv2b_out)
        conv3a_out = self.conv3a(pool3_out)
        conv3b_out = self.conv3b(conv3a_out)

        if self.verbose:
            log.info("Output Block 3: "+str(conv3b_out.shape))

        pool4_out  = self.pool4(conv3b_out)
        conv4a_out = self.conv4a(pool4_out)
        conv4b_out = self.conv4b(conv4a_out)

        if self.verbose:
            log.info("Output Block 4: "+str(conv4b_out.shape))

        pool5_out  = self.pool5(conv4b_out)
        conv5a_out = self.conv5a(pool5_out)
        conv5b_out = self.conv5b(conv5a_out)

        if self.verbose:
            log.info("Output Block 5: "+str(conv5b_out.shape))

        pool6_out  = self.pool6(conv5b_out)
        conv6a_out = self.conv6a(pool6_out)
        conv6b_out = self.conv6b(conv6a_out)

        if self.verbose:
            log.info("Output Block 6: "+str(conv6b_out.shape))

        upsample7_out = self.upsample7(conv6b_out)
        conv7a_out = self.conv7a(upsample7_out) + conv5b_out
        conv7b_out = self.conv7b(conv7a_out)

        if self.verbose:
            log.info("Output Block 7: "+str(conv7b_out.shape))

        upsample8_out = self.upsample8(conv7b_out)
        conv8a_out = self.conv8a(upsample8_out) + conv4b_out
        conv8b_out = self.conv8b(conv8a_out)

        if self.verbose:
            log.info("Output Block 8: "+str(conv8b_out.shape))

        upsample9_out = self.upsample8(conv8b_out)
        conv9a_out = self.conv9a(upsample9_out)  + conv3b_out
        conv9b_out = self.conv9b(conv9a_out)

        if self.verbose:
            log.info("Output Block 9: "+str(conv9b_out.shape))

        upsample10_out = self.upsample10(conv9b_out)
        conv10a_out = self.conv10a(upsample10_out) + conv2b_out
        conv10b_out = self.conv10b(conv10a_out)

        if self.verbose:
            log.info("Output Block 10: "+str(conv10b_out.shape))

        upsample11_out = self.upsample11(conv10b_out)
        conv11a_out = self.conv11a(upsample11_out) + conv1b_out
        conv11b_out = self.conv11b(conv11a_out)
        if self.verbose:
            log.info("Output Block 11: "+str(conv11b_out.shape))

        flowI_output = conv11b_out

        return flowI_output

    def compute_inputs(self, img_tensor, flow_pred_tensor, t):
        """
        Takes input and output from flow computation model, and required time step.
        Builds the required tensor for the interpolation model.

        :param img_tensor: B, 6, H, W image tensor (2 images , batch size B, input to flow computation model)
        :param flow_pred_tensor:  B, 4, H, W Flow 0->1, Flow 1->0 from flow computation model
        :param t: time step of interpolation t in (0, 1)
        :return: input_tensor: B, 16, H, W - input required for interpolation model
        """

        flow_01 = flow_pred_tensor[:,:2,:,:] # flow from 0 to 1.
        flow_10 = flow_pred_tensor[:,2:,:,:] # flow from 1 to 0.

        # estimated flow from t to 0
        est_flow_t0 = -(1 - t) * t * flow_01 + (t ** 2) * flow_10

        # estimated flow from t to 1
        est_flow_t1 = ((1 - t) ** 2) * flow_01  - t * (1 - t) * flow_10

        img_0 = img_tensor[:,:3,:,:]
        img_1 = img_tensor[:,3:,:,:]

        warped_img_1t = warp(img_1, - est_flow_t1) # backward warping
        warped_img_0t = warp(img_0, - est_flow_t0) # backward warping

        input_tensor = torch.cat([img_1, warped_img_1t, est_flow_t1,
                                  est_flow_t0, warped_img_0t, img_0], dim=1)

        if self.verbose:
            log.info("Generated Input tensor of shape:"+str(input_tensor.shape))

        return input_tensor

    def extract_outputs(self, output_tensor):
        """
        Extracts different elements in the output tensor.

        :param output_tensor: Output from the flow interpolation model.
        :return: The extract elements.
        """

        v_1t = output_tensor[:, 0, ...] # Visibility Map 1-> t
        dflow_t1 = output_tensor[:, 1:3, ...] # Residual of flow t->1
        dflow_t0 = output_tensor[:, 3:, ...] # Residual of flow t->0

        v_1t = v_1t[:, None, ...] # making dimensions compatible
        v_1t = self.squash(v_1t)

        v_0t = 1 - v_1t # Visibility Map 0->t

        return v_1t, dflow_t1, dflow_t0, v_0t

    def compute_output_image(self,img_tensor, input_tensor, output_tensor, t):
        """
        :param input_tensor: Input to flow interpolation model.
        :param output_tensor: Prediction from flow interpolation model
        :param t: Time step of interpolation (0 < t < 1)
        :return: I_t after enforcing constraints. B C H W
        """

        est_flow_t1 = input_tensor[:, 6:8, ...] # Estimated flow t->1
        est_flow_t0= input_tensor[:, 8:10, ...] # Estimated flow t->0

        img_0 = img_tensor[:,:3,...]
        img_1 = img_tensor[:,3:,...]

        pred_v_1t, pred_dflow_t1, pred_dflow_t0, pred_v_0t = self.extract_outputs(output_tensor)

        pred_flow_t1 = est_flow_t1 + pred_dflow_t1
        pred_flow_t0 = est_flow_t0 + pred_dflow_t0

        pred_img_0t = warp(img_0, -pred_flow_t0) # backward warping to produce img at time t
        pred_img_1t = warp(img_1, -pred_flow_t1) # backward warping to produce img at time t

        pred_img_0t = pred_v_0t * pred_img_0t # visibility map occlusion reasoning
        pred_img_1t = pred_v_1t * pred_img_1t # visibility map occlusion reasoning

        weighted_sum =(1 - t) * pred_img_0t  + t * pred_img_1t

        normalization_factor = (1 - t) * pred_v_0t + t * pred_v_1t # Z (refer to paper)

        pred_img_t = weighted_sum/normalization_factor

        return pred_img_t


def get_model(path, in_channels, out_channels, verbose=False):
    model = UNet(in_channels, out_channels, verbose=verbose)
    if path is not None:
        data = torch.load(path)
        if 'stage1_state_dict' in data.keys() and out_channels==4:
            log.info("Loading Stage 1 UNet.")
            model.load_state_dict(data['stage1_state_dict'])
            log.info("Loaded weights for Flow Computation: "+str(path))
        elif 'stage2_state_dict' in data.keys() and out_channels==5:
            log.info("Loading Stage 2 UNet.")
            model.load_state_dict(data['stage2_state_dict'])
            log.info("Loaded weights for Flow Interpolation: "+str(path))
        else:
            model.load_state_dict(data)
    else:
        log.info("Not loading weights for UNET.")
    return model



if __name__=='__main__':
    logging.basicConfig(filename="test.log", level=logging.INFO)

    model = get_model(path=None, in_channels=16, out_channels=11, verbose=True)
    model = model.cuda()

    input_sample = Variable(torch.randn([1, 16, 320, 640])).cuda()
    output_sample = model(input_sample)

    gt_sample = Variable(torch.randn([1, 3, 320, 640])).cuda()

    loss_weights = (0.5, 0.6, 1, 1)
    img_tensor = Variable(torch.randn([1, 6, 320, 640])).cuda()
    flow_tensor = Variable(torch.randn([1, 4, 320, 640])).cuda()

    test, _ = model.compute_interpolation_losses(img_tensor, flow_tensor, input_sample, output_sample,
                                                 target_image=gt_sample, loss_weights=loss_weights, t=0.5)
    test.backward()


##########################################
# // And all you touch and all you see,//#
# // Is all your life will ever be!    //#
##########################################
