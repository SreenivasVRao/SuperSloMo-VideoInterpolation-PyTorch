import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import logging
import numpy as np
from layers import warp

log = logging.getLogger(__name__)


class PerceptualLoss(nn.Module):

    def __init__(self, use_cuda=True):
        super(PerceptualLoss, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)

        self.l2_loss = nn.MSELoss()

        vgg16_mean = np.asarray([0.485, 0.456, 0.406])[None, :, None, None]
        vgg16_std = np.asarray([0.229, 0.224, 0.225])[None, :, None, None]
        self.vgg16_std = torch.autograd.Variable(torch.from_numpy(vgg16_std), requires_grad=False)
        self.vgg16_mean = torch.autograd.Variable(torch.from_numpy(vgg16_mean), requires_grad=False)

        self.vgg16_std = self.vgg16_std.float().cuda()
        self.vgg16_mean = self.vgg16_mean.float().cuda()

        self.vgg16.eval()
        self.eval()

        for param in self.vgg16.parameters():
            param.requires_grad = False
        for param in self.parameters():
            param.requires_grad = False

        self.modulelist = list(self.vgg16.features.modules())
        self.modulelist = self.modulelist[1:23] # until conv4_3

        if use_cuda:
            self.vgg16.cuda()
            self.cuda()

    def rescale(self, tensor):
        """
        :param tensor: B C H W tensor
        :return: tensor after rescaling to [0, 1]
        """
        b, c, h, w = tensor.shape
        tensor = tensor.contiguous()
        max_val, _ = tensor.view(b, c, -1).max(dim=2)
        min_val, _ = tensor.view(b, c, -1).min(dim=2)

        max_val = max_val[..., None, None]
        min_val = min_val[..., None, None]

        tensor_rescaled = (tensor - min_val)/(max_val-min_val)
        tensor_normed = (tensor_rescaled - self.vgg16_mean) / self.vgg16_std

        return tensor_normed

    def forward(self, x_input, x_target):
        x_input = self.rescale(x_input)
        x_target = self.rescale(x_target)

        for aLayer in self.modulelist: # until conv4_3
            x_input = aLayer(x_input)
            x_target = aLayer(x_target)

        perceptual_loss = self.l2_loss(x_input, x_target)

        return perceptual_loss


class SmoothnessLoss(nn.Module):
    def __init__(self, use_cuda=True):
        super(SmoothnessLoss, self).__init__()
        if use_cuda:
            self.cuda()
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, flow_input, image):

        gx_flow = self.gradient_x(flow_input)
        gx_image = self.gradient_x(image)

        w_gx_image = torch.mean(torch.abs(gx_image), dim=1, keepdim=True)
        w_gx_image = torch.exp(- w_gx_image)
        gx_flow = gx_flow * w_gx_image
        smooth_x_loss = torch.mean(torch.abs(gx_flow))

        gy_flow = self.gradient_y(flow_input)
        gy_image = self.gradient_y(image)
        w_gy_image = torch.mean(torch.abs(gy_image), dim=1, keepdim=True)
        w_gy_image = torch.exp(- w_gy_image)
        gy_flow = gy_flow * w_gy_image
        smooth_y_loss = torch.mean(torch.abs(gy_flow))

        loss_term = smooth_y_loss + smooth_x_loss

        return loss_term

    def spatial_gradient(self, image):
        """
        Compute magnitude of first order spatial gradient.
        :param image: tensor B, C, H, W
        :return: B C H W tensor - magnitude of first order spatial gradient
        """
        gx = self.gradient_x(image)
        gy = self.gradient_y(image)
        total_gradient = torch.sqrt(gx**2 + gy**2)
        return total_gradient

    def gradient_x(self, img):

        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):

        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy


class SSMLosses(nn.Module):

    def __init__(self, cfg):
        super(SSMLosses, self).__init__()

        self.reconstr_loss_fn = nn.L1Loss()
        self.perceptual_loss_fn = PerceptualLoss()
        self.smooth_loss_1 = SmoothnessLoss()
        self.smooth_loss_2 = SmoothnessLoss()

        self.warp_loss_1 = nn.L1Loss()
        self.warp_loss_2 = nn.L1Loss()
        self.warp_loss_3 = nn.L1Loss()
        self.warp_loss_4 = nn.L1Loss()
        self.loss_weights = self.read_loss_weights(cfg)

    def read_loss_weights(self, cfg):
        lambda_r = cfg.getfloat("TRAIN", "LAMBDA_R") # reconstruction loss weighting
        lambda_w = cfg.getfloat("TRAIN", "LAMBDA_W") # warp loss weighting
        lambda_s = cfg.getfloat("TRAIN", "LAMBDA_S") # smoothness loss weighting
        lambda_p = cfg.getfloat("TRAIN", "LAMBDA_P") # perceptual loss weighting
        self.loss_weights = lambda_r, lambda_p, lambda_w, lambda_s

    def get_reconstruction_loss(self, interpolated_image, target_image):
        return self.reconstr_loss_fn(interpolated_image, target_image)

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
        return self.perceptual_loss_fn(img, target_img)

    def forward(self, flowC_input, flowC_output, flowI_input, flowI_output, interpolated_image, target_image):
        loss_reconstr = self.get_reconstruction_loss(interpolated_image, target_image)
        loss_perceptual = self.get_perceptual_loss(interpolated_image, target_image)
        loss_warp = self.get_warp_loss(flowC_input, flowC_output, flowI_input, flowI_output, target_image)

        lambda_r, lambda_p, lambda_w, lambda_s = self.loss_weights

        total_loss = lambda_r * loss_reconstr + lambda_w * loss_warp + lambda_p * loss_perceptual
        loss_list = [total_loss, loss_reconstr, loss_warp, loss_perceptual]

        loss_tensor = torch.stack(loss_list).squeeze()
        return loss_tensor


def get_loss(config):
    return SSMLosses(config)


if __name__=='__main__':

    VGGLoss = PerceptualLoss()
    print(VGGLoss.training)

    tensor_1 = torch.autograd.Variable(torch.randn([2, 3, 100, 100]), requires_grad=True).cuda()
    tensor_2 = torch.autograd.Variable(torch.randn([2, 3, 100, 100])).cuda()
    result = VGGLoss(tensor_1, tensor_2)
    # for param in VGGLoss.parameters():
    #     print param.requires_grad
    result.backward()

    loss_smooth = SmoothnessLoss()
    tensor_1 = torch.autograd.Variable(torch.randn([2, 2, 100, 100]), requires_grad=True).cuda()
    tensor_2 = torch.autograd.Variable(torch.randn([2, 3, 100, 100]), requires_grad=True).cuda()
    result = loss_smooth(tensor_1, tensor_2)
    log.info(result.shape)
    result.backward()

