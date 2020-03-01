import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def make_norm_layer(norm_type, out_planes, is_2d=True, gn_planes=32):
    if norm_type.lower() == "bn":
        if is_2d:
            return torch.nn.BatchNorm2d(out_planes)
        else:
            return torch.nn.BatchNorm3d(out_planes)
    elif norm_type.lower() == "gn":
        return torch.nn.GroupNorm(gn_planes, out_planes)
    else:
        raise Exception("Not supported normalization layer type: {}.".format(norm_type))


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def conv_norm(
    in_planes,
    out_planes,
    norm_type="bn",
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        make_norm_layer(norm_type, out_planes),
        nn.ReLU(inplace=True),
    )


def avg_pool(kernel_size=2, stride=None, padding=0):
    return nn.AvgPool2d(
        kernel_size, stride, padding, ceil_mode=False, count_include_pad=True
    )


def upsample(in_planes, out_planes, scale=2, mode="bilinear"):
    return nn.Sequential(
        nn.Conv1d(in_planes, out_planes, kernel_size=(1, 1)),
        nn.Upsample(scale_factor=scale, mode=mode),
    )


def warp(x, flo):
    """
    Apply optical flow warping to an image to generate a new image.

    1. Take a grid of points which are the pixel locations in the input image (x)
    2. Add the optical flow vectors to these points to find their new locations.
    3. Use `nn.functional.grid_sample()` to calculate the pixel value at the new location
    based on bi-linear interpolation from values in the original image.

    Source: https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py#L370

    :param x: [B, C, H, W] image
    :param flo: [B, 2, H, W] flow (u,v) vectors
    :return output: warped image as a tensor
    :rtype: `torch.Tensor`
    """

    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    # @sreenivasvrao: PyTorch backward pass fails with the next two lines of code.

    # vgrid[:, 0, :, :] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    # vgrid[:, 1, :, :] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    # replace with the next 6
    u_tmp = vgrid[:, 0, :, :].clone()
    v_tmp = vgrid[:, 1, :, :].clone()

    u_tmp = 2.0 * u_tmp / max(W - 1, 1) - 1.0
    v_tmp = 2.0 * v_tmp / max(H - 1, 1) - 1.0

    vgrid[:, 0, :, :] = u_tmp
    vgrid[:, 1, :, :] = v_tmp

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    return output


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(
        in_planes, out_planes, kernel_size, stride, padding, bias=True
    )
