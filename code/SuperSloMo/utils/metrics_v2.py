import numpy as np
from skimage.measure import compare_ssim


def get_scores(output_batch, target_batch):
    """
    output_batch = B, C, H, W tensor 0 - 1 range
    target_batch = B, C, H, W tensor 0 - 1 range
    """
    assert output_batch.shape == target_batch.shape, "Batch shape mismatch."
    B = output_batch.shape[0]

    psnr_scores = []
    ie_scores   = []
    ssim_scores = []
    
    output_batch = output_batch.permute(0, 2, 3, 1) # BCHW -> BHWC
    target_batch = target_batch.permute(0, 2, 3, 1) # BCHW -> BHWC
    

    for idx in range(B):
        output_image = output_batch[idx, ...] * 255.0 # 1 H W C
        target_image = target_batch[idx, ...] * 255.0 # 1 H W C

        output_image = output_image.cpu().data.numpy()
        target_image = target_image.cpu().data.numpy()

        mse_score = np.square(output_image - target_image).mean()
        
        rmse_score = np.sqrt(mse_score)
        psnr_score = 20*np.log10(255.0/(rmse_score+1e-7)) # avoid divide by zero.
        ssim_score = compare_ssim(output_image, target_image, multichannel=True)

        psnr_scores.append(psnr_score)
        ie_scores.append(rmse_score)
        ssim_scores.append(ssim_score)
        
    return psnr_scores, ie_scores, ssim_scores
