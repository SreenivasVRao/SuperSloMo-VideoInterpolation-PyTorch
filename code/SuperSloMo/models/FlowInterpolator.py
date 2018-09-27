from layers import *
from custom_losses import PerceptualLoss, SmoothnessLoss
import torch.nn as nn
import torch.nn.functional as F
import torch


class InterpolationModel(nn.Module):

    def __init__(self, args=None, batch_norm=False, verbose=False):
        super(InterpolationModel, self).__init__()
        self.batchNorm = batch_norm
        self.args = args
        self.verbose = verbose
        self.build_model()
        self.define_losses()

    def concat_tensors(self, tensor1, tensor2):
        """
        Makes tensor1 dimensions equal to tensor2.
        :param tensor1: tensor after interpolation
        :param tensor2: tensor from earlier layer.
        :return: concatenation of tensor1 and tensor2 along channels
        """
        _, _, h1, w1 = tensor1.shape
        _, _, h2, w2 = tensor2.shape

        padding = [0, 0, 0, 0] # (left, right, top, bottom)
        extra_padding = False

        rows = abs(h2 - h1)
        cols = abs(w2 - w1)

        top = int(rows / 2)
        bottom = rows - top

        left = int(cols / 2)
        right = cols - left

        if h1 < h2:
            padding[2] = top
            padding[3] = bottom
            extra_padding = True
        elif h2 < h1:
            tensor1 = tensor1[:, :, :-rows, :]

        if w1 < w2:
            padding[0] = left
            padding[1] = right
            extra_padding = True
        elif w2 < w1:
            tensor1 = tensor1[:, :, :, :-cols]

        if extra_padding:
            pad = nn.ZeroPad2d(padding)
            tensor1 = pad(tensor1)

        new_tensor = torch.cat([tensor1, tensor2], dim=1)

        return new_tensor

    def build_model(self):

        # block 1

        self.conv1a = conv(in_planes=16, out_planes=32, kernel_size=7)
        self.conv1b = conv(in_planes=32, out_planes=32, kernel_size=7)

        # block 2

        self.pool2 = avg_pool(kernel_size=2, stride=None, padding=0)
        self.conv2a = conv(in_planes=32, out_planes=64, kernel_size=5)
        self.conv2b = conv(in_planes=64, out_planes=64, kernel_size=5)

        # block 3

        self.pool3 = avg_pool(kernel_size=2, stride=None, padding=0)
        self.conv3a = conv(in_planes=64, out_planes=128,kernel_size=3)
        self.conv3b = conv(in_planes=128, out_planes=128,kernel_size=3)


        # block 4

        self.pool4 = avg_pool(kernel_size=2, stride=None, padding=0)
        self.conv4a = conv(in_planes=128, out_planes=256,kernel_size=3)
        self.conv4b = conv(in_planes=256, out_planes=256, kernel_size=3)

        # block 5

        self.pool5 = avg_pool(kernel_size=2, stride=None, padding=0)
        self.conv5a = conv(in_planes=256, out_planes=512, kernel_size=3)
        self.conv5b = conv(in_planes=512, out_planes=512, kernel_size=3)

        # block 6
        self.pool6 = avg_pool(kernel_size=2, stride=None, padding=0)
        self.conv6a = conv(in_planes=512, out_planes=512, kernel_size=3)
        self.conv6b = conv(in_planes=512, out_planes=512, kernel_size=3)

        # block 7

        self.upsample7 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                              mode='bilinear')  # 2 x 2 upsampling
        self.conv7a = conv(in_planes=1024, out_planes=512,kernel_size=3)
        self.conv7b = conv(in_planes=512, out_planes=512, kernel_size=3)


        # block 8

        self.upsample8 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                              mode='bilinear')  # 2 x 2 upsampling
        self.conv8a = conv(in_planes=768, out_planes=256, kernel_size=3)
        self.conv8b = conv(in_planes=256, out_planes=256, kernel_size=3)


        # block 9

        self.upsample9 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                              mode='bilinear')  # 2 x 2 upsampling
        self.conv9a = conv(in_planes=384, out_planes=128, kernel_size=3)
        self.conv9b = conv(in_planes=128, out_planes=128, kernel_size=3)

        # # block 10
        #
        self.upsample10 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                               mode='bilinear')  # 2 x 2 upsampling
        self.conv10a = conv(in_planes=192, out_planes=64, kernel_size=3)
        self.conv10b = conv(in_planes=64, out_planes=64, kernel_size=3)

        # block 11

        self.upsample11 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                               mode='bilinear')  # 2 x 2 upsampling
        self.conv11a = conv(in_planes=96, out_planes=32, kernel_size=3)
        self.conv11b = conv(in_planes=32, out_planes=11, kernel_size=3)

        # block 12

        self.upsample12 = lambda x: F.upsample(x, size=(2 * x.shape[2], 2 * x.shape[3]),
                                               mode='bilinear')  # 2 x 2 upsampling

    def define_losses(self):
        self.reconstr_loss_fn = nn.L1Loss()
        self.perceptual_loss_fn = PerceptualLoss()
        self.smooth_loss_1 = SmoothnessLoss()
        self.smooth_loss_2 = SmoothnessLoss()

        self.warp_loss_1 = nn.L1Loss()
        self.warp_loss_2 = nn.L1Loss()
        self.warp_loss_3 = nn.L1Loss()
        self.warp_loss_4 = nn.L1Loss()

    def forward(self, input_tensor):
        """
        :param input_tensor: input: N,18,  H, W,
        batch_size = N

        :return: output_tensor: N, 18, H, W, C
        interpolation result

        """
        if self.verbose:
            print "Input: ", input_tensor.shape

        conv1a_out = self.conv1a(input_tensor)
        conv1b_out = self.conv1b(conv1a_out)

        if self.verbose:
            print "Output Block 1: ", conv1b_out.shape

        pool2_out  = self.pool2(conv1b_out)
        conv2a_out = self.conv2a(pool2_out)
        conv2b_out = self.conv2b(conv2a_out)

        if self.verbose:
            print "Output Block 2: ", conv2b_out.shape

        pool3_out  = self.pool3(conv2b_out)
        conv3a_out = self.conv3a(pool3_out)
        conv3b_out = self.conv3b(conv3a_out)

        if self.verbose:
            print "Output Block 3: ", conv3b_out.shape

        pool4_out  = self.pool4(conv3b_out)
        conv4a_out = self.conv4a(pool4_out)
        conv4b_out = self.conv4b(conv4a_out)

        if self.verbose:
            print "Output Block 4: ", conv4b_out.shape

        pool5_out  = self.pool5(conv4b_out)
        conv5a_out = self.conv5a(pool5_out)
        conv5b_out = self.conv5b(conv5a_out)

        if self.verbose:
            print "Output Block 5: ", conv5b_out.shape

        pool6_out  = self.pool6(conv5b_out)
        conv6a_out = self.conv6a(pool6_out)
        conv6b_out = self.conv6b(conv6a_out)

        if self.verbose:
            print "Output Block 6: ", conv6b_out.shape

        upsample7_out = self.upsample7(conv6b_out)
        input_7 = self.concat_tensors(upsample7_out, conv5b_out)
        # = torch.cat([upsample7_out, conv5b_out], dim=1)
        conv7a_out = self.conv7a(input_7)
        conv7b_out = self.conv7b(conv7a_out)

        if self.verbose:
            print "Output Block 7: ", conv7b_out.shape

        upsample8_out = self.upsample8(conv7b_out)
        input_8 = self.concat_tensors(upsample8_out, conv4b_out)
        # input_8 = torch.cat([upsample8_out, conv4b_out], dim=1)
        conv8a_out = self.conv8a(input_8)
        conv8b_out = self.conv8b(conv8a_out)

        if self.verbose:
            print "Output Block 8: ", conv8b_out.shape

        upsample9_out = self.upsample8(conv8b_out)
        input_9 = self.concat_tensors(upsample9_out, conv3b_out)
        # input_9 = torch.cat([upsample9_out, conv3b_out], dim=1)
        conv9a_out = self.conv9a(input_9)
        conv9b_out = self.conv9b(conv9a_out)

        if self.verbose:
            print "Output Block 9: ", conv9b_out.shape

        upsample10_out = self.upsample10(conv9b_out)
        input_10 = self.concat_tensors(upsample10_out, conv2b_out)
        # input_10 = torch.cat([upsample10_out, conv2b_out], dim=1)
        conv10a_out = self.conv10a(input_10)
        conv10b_out = self.conv10b(conv10a_out)

        if self.verbose:
            print "Output Block 10: ", conv10b_out.shape

        upsample11_out = self.upsample11(conv10b_out)
        input_11 = self.concat_tensors(upsample11_out, conv1b_out)
        # input_11 = torch.cat([upsample11_out, conv1b_out], dim=1)
        conv11a_out = self.conv11a(input_11)
        conv11b_out = self.conv11b(conv11a_out)

        upsample12_out = F.upsample(conv11b_out, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear')

        if self.verbose:
            print "Output Block 11: ", conv11b_out.shape
            print "Upsampled Output Shape:", upsample12_out.shape

        return upsample12_out

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
            print "Generated Input tensor of shape:", input_tensor.shape

        return input_tensor

    def extract_outputs(self, output_tensor):
        """
        Extracts different elements in the output tensor.

        :param output_tensor: Output from the flow interpolation model.
        :return: The extract elements.
        """

        img_1 = output_tensor[:, :3, ...] # Image 1
        v_1t = output_tensor[:, 3, ...] # Visibility Map 1-> t
        dflow_t1 = output_tensor[:, 4:6, ...] # Residual of flow t->1
        dflow_t0 = output_tensor[:, 6:8, ...] # Residual of flow t->0
        img_0 = output_tensor[:, 8:, ...] # Image 0

        v_0t = 1 - v_1t # Visibility Map 0->t

        return img_1, v_1t, dflow_t1, dflow_t0, v_0t, img_0

    def compute_output_image(self, input_tensor, output_tensor, t):
        """
        :param input_tensor: Input to flow interpolation model.
        :param output_tensor: Prediction from flow interpolation model
        :param t: Time step of interpolation (0 < t < 1)
        :return: I_t after enforcing constraints. B C H W
        """

        est_flow_t1 = input_tensor[:, 6:8, ...] # Estimated flow t->1
        est_flow_t0= input_tensor[:, 8:10, ...] # Estimated flow t->0

        pred_img_1, pred_v_1t, pred_dflow_t1, \
        pred_dflow_t0, pred_v_0t, pred_img_0 = self.extract_outputs(output_tensor)

        pred_flow_t1 = est_flow_t1 + pred_dflow_t1
        pred_flow_t0 = est_flow_t0 + pred_dflow_t0

        pred_img_0t = warp(pred_img_0, -pred_flow_t0) # backward warping to produce img at time t
        pred_img_1t = warp(pred_img_1, -pred_flow_t1) # backward warping to produce img at time t

        pred_img_0t = pred_v_0t * pred_img_0t # visibility map occlusion reasoning
        pred_img_1t = pred_v_1t * pred_img_1t # visibility map occlusion reasoning

        weighted_sum =(1 - t) * pred_img_0t  + t * pred_img_1t

        normalization_factor = (1 - t) * pred_v_0t + t * pred_v_1t # Z (refer to paper)

        pred_img_t = weighted_sum/normalization_factor

        return pred_img_t

    def get_reconstruction_loss(self, interpolated_image, target_image):

        loss_reconstr = self.reconstr_loss_fn(interpolated_image, target_image)

        return loss_reconstr

    def get_warp_loss(self, img_tensor, flow_tensor, input_tensor, output_tensor, target_image):

        flow_01 = flow_tensor[:,:2,  ...]
        flow_10 = flow_tensor[:, 2:, ...]

        img_1 = img_tensor[:,:3,...]
        img_0 = img_tensor[:,3:,...]

        pred_img_10 = warp(img_1, - flow_01) # flow img 1 to img 0
        pred_img_01 = warp(img_0, - flow_10) # flow img 0 to img 1

        flow_t1 = input_tensor[:, 6:8, ...] # Estimated flow t->1
        flow_t0= input_tensor[:, 8:10, ...] # Estimated flow t->0

        pred_img_1, pred_v_1t, pred_dflow_t1, \
        pred_dflow_t0, pred_v_0t, pred_img_0 = self.extract_outputs(output_tensor)

        pred_flow_t1 = flow_t1 + pred_dflow_t1
        pred_flow_t0 = flow_t0 + pred_dflow_t0

        pred_img_0t = warp(pred_img_0, -pred_flow_t0) # backward warping to produce img at time t
        pred_img_1t = warp(pred_img_1, -pred_flow_t1) # backward warping to produce img at time t

        loss_warp = self.warp_loss_1(pred_img_10, img_0) + \
                    self.warp_loss_2(pred_img_01, img_1) + \
                    self.warp_loss_3(pred_img_0t, target_image) + \
                    self.warp_loss_4(pred_img_1t, target_image)

        return loss_warp

    def get_perceptual_loss(self, img, target_img):
        loss_perceptual = self.perceptual_loss_fn(img, target_img)
        return loss_perceptual

    def get_smooth_loss(self, flow_tensor, image_tensor):
        assert image_tensor.shape[1]==6, "Expected B 6 H W tensor."
        assert flow_tensor.shape[1]==4, "Expected B 4 H W tensor."
        img_0 = image_tensor[:, :3, ...] # first half of B 6 H W tensor
        img_1 = image_tensor[:, 3:, ...] # second half

        flow_01 = flow_tensor[:, :2, ...] # flow 0 -> 1
        flow_10 = flow_tensor[:, 2:, ...] # flow 1 -> 0

        loss_smooth_01 = self.smooth_loss_1(flow_01, img_0)
        loss_smooth_10 = self.smooth_loss_2(flow_10, img_1)

        loss_smooth = loss_smooth_01 + loss_smooth_10

        return loss_smooth

    def compute_loss(self, img_tensor, flow_tensor, input_tensor,
                     output_tensor, target_image, loss_weights, t):
        """

        :param img_tensor: Input to Flow Computation Model B, 6, H, W
        :param flow_tensor: Output from Flow Computation Model B, 4, H, W
        :param input_tensor: Input to Flow Interpolation Model B, 16, H, W
        :param output_tensor: Output from Flow Interpolation Model B, 11, H, W
        :param target_image: interpolation ground truth B, 3, H, W
        :param loss_weights: tuple of 4 weights (reconstr, smooth, warp, perceptual)
        :param t: time step of interpolation
        :return: total loss as weighted sum of losses.
        """

        interpolated_image = self.compute_output_image(input_tensor, output_tensor, t)
        loss_reconstr = self.get_reconstruction_loss(interpolated_image, target_image)
        loss_perceptual = self.get_perceptual_loss(interpolated_image, target_image)
        loss_smooth = self.get_smooth_loss(flow_tensor, img_tensor)
        loss_warp = self.get_warp_loss(img_tensor, flow_tensor, input_tensor, output_tensor, target_image)

        lambda_r, lambda_p, lambda_w, lambda_s = loss_weights

        total_loss = lambda_r * loss_reconstr + lambda_s * loss_smooth + \
                     lambda_w * loss_warp + lambda_p * loss_perceptual

        return total_loss, (loss_reconstr, loss_perceptual, loss_smooth, loss_warp)


def flow_interpolator(path):
    model = InterpolationModel()
    if path is not None:
        data = torch.load(path)
        if 'stage2_state_dict' in data.keys():
            model.load_state_dict(data['stage2_state_dict'])
        else:
            model.load_state_dict(data)
        print "Loaded weights for Flow Interpolator: ", path
    else:
        print "Not loading weights for Flow Interpolator."
    return model



if __name__=='__main__':

    model = InterpolationModel(verbose=False)
    model = model.cuda()

    input_sample = Variable(torch.randn([1, 16, 384, 384])).cuda()
    output_sample = model(input_sample)

    gt_sample = Variable(torch.randn([1, 3, 384, 384])).cuda()

    loss_weights = (0.5, 0.6, 1, 1)
    img_tensor = Variable(torch.randn([1, 6, 384, 384])).cuda()
    flow_tensor = Variable(torch.randn([1, 4, 384, 384])).cuda()

    test, _ = model.compute_loss(img_tensor, flow_tensor, input_sample, output_sample,
                              target_image=gt_sample, loss_weights=loss_weights, t=0.5)
    test.backward()


##########################################
# // And all you touch and all you see,//#
# // Is all your life will ever be!    //#
##########################################