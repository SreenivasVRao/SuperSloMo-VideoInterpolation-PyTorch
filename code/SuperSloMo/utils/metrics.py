import numpy as np
import math
from skimage.measure import compare_ssim
import torch

IE_function = torch.nn.MSELoss()
MSE_function = torch.nn.MSELoss()

"""
img1, img2 should be in numpy format with type uint8.
"""


def psnr(img1, img2):
    assert (img1.dtype == img2.dtype == np.uint8)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    diff = img1 - img2
    diff_sq = diff **2

    mse = np.mean(diff_sq, axis=(1, 2, 3))
    return 20 * np.log10(255.0 / np.sqrt(mse+1e-7))


def ssim(img1, img2):
    assert (img1.dtype == img2.dtype == np.uint8)
    scores = []
    for idx in range(img1.shape[0]):
        scores.append(compare_ssim(img1[idx, ...], img2[idx, ...], multichannel=True))

    scores = np.asarray(scores)
    return scores


def interpolation_error(input_batch, target):

    scores = []
    for idx in range(input_batch.shape[0]):
        img = input_batch[idx, ...][None,...] # 1 C H W
        img_t = target[idx, ...][None,...] # 1 C H W
        current_score = IE_function(img, img_t)
        current_score = torch.sqrt(current_score)
        scores.append(current_score)

    return scores

