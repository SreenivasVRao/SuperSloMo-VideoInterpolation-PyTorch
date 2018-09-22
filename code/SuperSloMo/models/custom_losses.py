import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch


class PerceptualLoss(nn.Module):

    def __init__(self, use_cuda=True):
        super(PerceptualLoss, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)

        self.modulelist = list(self.vgg16.features.modules())

        self.modulelist = self.modulelist[1:23] # until conv4_3

        for param in self.vgg16.parameters():
            param.requires_grad = False

        for param in self.parameters():
            param.requires_grad = False


        if use_cuda:
            self.vgg16.cuda()
            self.cuda()

    def forward(self, x_input, x_target):

        for aLayer in self.modulelist: # until conv4_3
            x_input = aLayer(x_input)
            x_target = aLayer(x_target)

        difference = x_input - x_target

        diff_sq = difference**2

        l2_norm = torch.sum(diff_sq)

        return l2_norm


class SmoothnessLoss(nn.Module):
    def __init__(self, use_cuda=True):
        super(SmoothnessLoss, self).__init__()
        if use_cuda:
            self.cuda()

        for param in self.parameters():
            param.requires_grad=False

    def forward(self, x_input):
        gradient_flow = self.spatial_gradient(x_input) # gradient of the flow
        abs_gradient = torch.abs(gradient_flow)
        l1_norm = torch.sum(abs_gradient)
        return l1_norm

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


if __name__=='__main__':

    VGGLoss = PerceptualLoss()
    tensor_1 = torch.autograd.Variable(torch.randn([2, 3, 100, 100]), requires_grad=True).cuda()
    tensor_2 = torch.autograd.Variable(torch.randn([2, 3, 100, 100])).cuda()
    result = VGGLoss(tensor_1, tensor_2)
    for param in VGGLoss.parameters():
        print param.requires_grad
    result.backward()

    loss_smooth = SmoothnessLoss()
    tensor_1 = torch.autograd.Variable(torch.randn([2, 3, 100, 100]), requires_grad=True).cuda()
    result = loss_smooth(tensor_1)
    for param in loss_smooth.parameters():
        print param.requires_grad
    result.backward()

