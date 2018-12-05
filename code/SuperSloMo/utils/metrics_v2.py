import numpy as np
from skimage.measure import compare_ssim, compare_psnr


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

    h1 = output_batch.shape[1]
    h2 = target_batch.shape[1]
    assert h1==h2 and h2==736, "Image Heights are wrong."

    output_batch = output_batch[:, 8:728, ...]
    target_batch = target_batch[:, 8:728, ...]

    assert output_batch.shape[1:4]==(720, 1280, 3), "Dimensions are incorrect."
    assert target_batch.shape[1:4]==(720, 1280, 3), "Dimensions are incorrect."

    for idx in range(B):
        output_image = output_batch[idx, ...] * 255.0 # 1 H W C
        target_image = target_batch[idx, ...] * 255.0 # 1 H W C

        output_image = output_image.cpu().data.numpy()
        target_image = target_image.cpu().data.numpy()

        mse_score = np.square(output_image - target_image).mean()
        
        rmse_score = np.sqrt(mse_score)
        # psnr_score = 20*np.log10(255.0/(rmse_score+1e-7)) # avoid divide by zero.
        psnr_score = compare_psnr(output_image.astype(np.uint8), target_image.astype(np.uint8))
        ssim_score = compare_ssim(output_image.astype(np.uint8), target_image.astype(np.uint8),
                                  multichannel=True, gaussian_weights=True)

        psnr_scores.append(psnr_score)
        ie_scores.append(rmse_score)
        ssim_scores.append(ssim_score)
        
    return psnr_scores, ie_scores, ssim_scores
