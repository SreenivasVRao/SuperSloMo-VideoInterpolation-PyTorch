import logging

import torch
import torch.nn as nn
import torchvision

from .layers import warp

log = logging.getLogger(__name__)


class PerceptualLoss(nn.Module):
    def __init__(self):
        """
        Creates a loss function that compares the VGG-16 deep features
        of two tensors using L-2 loss.

        :returns:
        :rtype:

        """
        super(PerceptualLoss, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)

        self.l2_loss = nn.MSELoss(reduce=False)

        self.vgg16.eval()
        self.eval()

        for param in self.vgg16.parameters():
            param.requires_grad = False
        for param in self.parameters():
            param.requires_grad = False
        self.vgg_conv4_3 = self.vgg16.features[:23]

    def forward(self, x_input, x_target):

        x_input = self.vgg_conv4_3(x_input)
        x_target = self.vgg_conv4_3(x_target)
        perceptual_loss = self.l2_loss(x_input, x_target)
        return perceptual_loss


class SSMLosses(nn.Module):
    def __init__(self, cfg):
        """This module holds all the various loss functions that are required
        for SuperSloMo. It loads the hyper parameters from the cfg file.

        :param cfg: the parsed config from the `configs/ssmr.ini`
        """

        super(SSMLosses, self).__init__()
        self.cfg = cfg
        self.perceptual_loss_fn = PerceptualLoss()
        self.reconstr_loss_fn = nn.L1Loss(reduce=False)
        self.warp_loss_1 = nn.L1Loss(reduce=False)
        self.warp_loss_2 = nn.L1Loss(reduce=False)
        self.warp_loss_3 = nn.L1Loss(reduce=False)
        self.warp_loss_4 = nn.L1Loss(reduce=False)
        self.loss_weights = self.read_loss_weights(cfg)
        self.squash = nn.Sigmoid()

    def read_loss_weights(self, cfg):
        """ Loads the weights for each loss function from the config

        :param cfg:  parsed from `configs/ssmr.ini`

        :returns: hyper parameters for loss weights

        :rtype: tuple of float values

        """
        # reconstruction loss weighting
        lambda_r = cfg.getfloat("TRAIN", "LAMBDA_R")
        lambda_w = cfg.getfloat("TRAIN", "LAMBDA_W")  # warp loss weighting
        # perceptual loss weighting
        lambda_p = cfg.getfloat("TRAIN", "LAMBDA_P")
        return lambda_r, lambda_p, lambda_w

    def extract_visibility_and_flow(self, output_tensor):
        """
        Extracts the visibility and flow maps from the tensor.
        2nd and 3rd values are residual optical flow tensors of B C H W shape, with C = 2

        :param output_tensor: B C H W output tensor
        :return: Tuple of BCHW tensors, where 1st and last elements
        each have channel dim = 1, and are equal to probability values.

        """

        v_1t = output_tensor[:, 0, ...]  # Visibility Map 1-> t
        dflow_t1 = output_tensor[:, 1:3, ...]  # Residual of flow t->1
        dflow_t0 = output_tensor[:, 3:5, ...]  # Residual of flow t->0

        v_1t = v_1t[:, None, ...]  # making dimensions compatible

        v_1t = self.squash(v_1t)

        v_0t = 1 - v_1t  # Visibility Map 0->t

        return v_1t, dflow_t1, dflow_t0, v_0t

    def get_reconstruction_loss(self, interpolated_image, target_image):
        """ Get L-1 loss between the two images
        :param interpolated_image:
        :param target_image:
        :returns: The pixel-wise L-1 loss between the images
        :rtype: B 1 H W tensor

        """
        return self.reconstr_loss_fn(interpolated_image, target_image)

    def get_warp_loss(self, img_tensor, flowC_output, flowI_input, flowI_output, target_image):
        """
        Get the warp loss in each stage if the stage is included in training.
        Warp loss for 1st stage is based on coarse optical flow.
        2nd stage includes residual flow predicted by the flow-interpolation model.


        Estimate optical flow from T= t -> 0, and T= t->1
        Use the optical flow to warp img at T=0 and T= 1 and produce the image at T=t.

        Apply L-1 loss on the estimated and target images.

        :param img_tensor: B 6 H W tensor of original input images
        :param flowC_output: the output tensor of the flow computation model.
        :param flowI_input: B C H W tensor which is used as input to the flow interpolation model
        :param flowI_output: B C H W tensor which is output from the flow interpolation model
        :param target_image: B 3 H W image
        :returns: total warp loss
        :rtype: a B 1 H W tensor.

        """

        flow_01 = flowC_output[:, 0:2, ...]
        flow_10 = flowC_output[:, 2:4, ...]

        img_0 = img_tensor[:, 0:3, ...]
        img_1 = img_tensor[:, 3:6, ...]

        flow_t1 = flowI_input[:, 6:8, ...]  # Estimated flow t->1
        flow_t0 = flowI_input[:, 8:10, ...]  # Estimated flow t->0

        pred_v_1t, pred_dflow_t1, pred_dflow_t0, pred_v_0t = self.extract_visibility_and_flow(
            flowI_output
        )

        pred_flow_t1 = flow_t1 + pred_dflow_t1
        pred_flow_t0 = flow_t0 + pred_dflow_t0

        # backward warping to produce img at time t
        pred_img_0t = warp(img_0, pred_flow_t0)
        # backward warping to produce img at time t
        pred_img_1t = warp(img_1, pred_flow_t1)

        loss_warp = 0
        loss_warp_stage1 = 0
        loss_warp_stage2 = 0
        if not self.cfg.getboolean("STAGE1", "FREEZE"):
            loss_warp_stage1 = self.warp_loss_1(warp(img_1, flow_01), img_0) + self.warp_loss_2(
                warp(img_0, flow_10), img_1
            )

        if not self.cfg.getboolean("STAGE2", "FREEZE"):
            loss_warp_stage2 = self.warp_loss_3(pred_img_0t, target_image) + self.warp_loss_4(
                pred_img_1t, target_image
            )

        loss_warp = loss_warp_stage1 + loss_warp_stage2
        return loss_warp, loss_warp_stage1, loss_warp_stage2

    def get_perceptual_loss(self, img, target_img):
        """ Get VGG-16 feature L-2 loss as a B C H W tensor.

        :param img:
        :param target_img:
        :returns:
        :rtype:

        """
        return self.perceptual_loss_fn(img, target_img)

    def get_batch_mean(self, loss_tensor):
        """
        Get the average loss over each sample in the batch.
        We retain the batch dimension because that allows `torch.nn.DataParallel`
        to be more GPU-memory efficient

        :param loss_tensor = B, C, H, W tensor.
        :return batch_mean = B, 1 tensor. The mean loss for each sample as B, 1 tensor.
        """
        loss_tensor = loss_tensor.contiguous().view(loss_tensor.shape[0], -1)
        batch_mean = loss_tensor.mean(dim=1)[:, None]
        return batch_mean

    def forward(
        self, flowC_input, flowC_output, flowI_input, flowI_output, interpolated_image, target_image
    ):
        """ Compute the losses and return a [B, 4] shape tensor where B = batch size,
        and 4 values are (total loss, reconstruction loss, warp loss, perceptual loss)

        We retain the batch dimension for GPU-memory efficiency.

        :param flowC_input: Input to the flow computation model  (B C H W)
        :param flowC_output: Output from the flow computation model (B C H W)
        :param flowI_input: Input to the flow interpolation model (B C H W)
        :param flowI_output: Output to the flow interpolation model (B C H W)
        :param interpolated_image: Final interpolated image from the model (B C H W)
        :param target_image: Ground truth image
        :returns: losses
        :rtype: [B, 4] tensor

        """

        lambda_r, lambda_p, lambda_w = self.loss_weights

        loss_reconstr = lambda_r * self.get_reconstruction_loss(interpolated_image, target_image)
        loss_perceptual = lambda_p * self.get_perceptual_loss(interpolated_image, target_image)
        loss_warp, loss_warp_stage1, loss_warp_stage2 = self.get_warp_loss(
            flowC_input, flowC_output, flowI_input, flowI_output, target_image
        )
        loss_warp *= lambda_w
        loss_warp_stage1 *= lambda_w
        loss_warp_stage2 *= lambda_w

        loss_reconstr = loss_reconstr.view(loss_reconstr.shape[0], -1).mean(dim=1)[:, None]
        loss_perceptual = loss_perceptual.view(loss_perceptual.shape[0], -1).mean(dim=1)[:, None]
        loss_warp = loss_warp.view(loss_warp.shape[0], -1).mean(dim=1)[:, None]

        loss_reconstr = self.get_batch_mean(loss_reconstr)
        loss_warp = self.get_batch_mean(loss_warp)

        loss_perceptual = self.get_batch_mean(loss_perceptual)

        # gets a [B, 1] with mean loss over each sample tensor
        # need this because I'm using multi-gpu
        # and need to accumulate the loss over samples.
        # Such bad code :/

        total_loss = loss_reconstr + loss_warp + loss_perceptual

        loss_list = [total_loss, loss_reconstr, loss_warp, loss_perceptual]

        loss_tensor = torch.stack(loss_list).squeeze()
        if len(loss_tensor.shape) == 1:
            loss_tensor = loss_tensor[:, None]

        loss_tensor = loss_tensor.permute(1, 0)  # [B, 4] for 4 losses
        return loss_tensor


if __name__ == "__main__":

    VGGLoss = PerceptualLoss()
    print(VGGLoss.training)

    tensor_1 = torch.autograd.Variable(torch.randn([2, 3, 100, 100]), requires_grad=True).cuda()
    tensor_2 = torch.autograd.Variable(torch.randn([2, 3, 100, 100])).cuda()
    result = VGGLoss(tensor_1, tensor_2)
    # for param in VGGLoss.parameters():
    #     print param.requires_grad
    result.backward()
