import numpy as np
import math
from skimage.measure import compare_ssim

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
    return 20 * math.log10(255.0 / math.sqrt(mse)+1e-7)


def ssim(img1, img2):
    assert (img1.dtype == img2.dtype == np.uint8)
    scores = []
    for idx in range(img1.shape[0]):
        scores.append(compare_ssim(img1[idx, ...], img2[idx, ...]))

    scores = np.asarray(scores)
    return scores


def interpolation_error(image1, image2):
    diff = image1 - image2
    diff_sq = diff **2
    mse = np.mean(diff_sq, axis=(1, 2, 3)) # mean over H W C
    rmse = np.sqrt(mse)
    return rmse



