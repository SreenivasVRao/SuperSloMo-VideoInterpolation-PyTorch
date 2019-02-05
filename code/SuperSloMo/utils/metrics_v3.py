"""
Use DataLoader to get batches of frames for the model.

"""
import numpy as np
from skimage.measure import compare_ssim, compare_psnr
import logging
import torch
import configparser, cv2, os, glob
from argparse import ArgumentParser
import adobe_240fps

import sys
sys.path.insert(0, "/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/SuperSloMo/models/")
import SSMR

log = logging.getLogger(__name__)


def getargs():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        default="config.ini",
                        help="Path to config.ini file.")
    parser.add_argument("--expt", required=True,
                        help="Experiment Name.")
    parser.add_argument("--log", required=True, help="Path to logfile.")
    args = parser.parse_args()
    return args


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
        output_image = output_batch[idx, ...] # * 255.0 # 1 H W C
        target_image = target_batch[idx, ...] # * 255.0 # 1 H W C

        output_image = output_image.cpu().data.numpy()
        target_image = target_image.cpu().data.numpy()

        mse_score = np.square(output_image - target_image).mean()
        
        rmse_score = np.sqrt(mse_score)
        psnr_score = compare_psnr(output_image.astype(np.uint8), target_image.astype(np.uint8))
        ssim_score = compare_ssim(output_image.astype(np.uint8), target_image.astype(np.uint8),
                                  multichannel=True, gaussian_weights=True)

        psnr_scores.append(psnr_score)
        ie_scores.append(rmse_score)
        ssim_scores.append(ssim_score)
        
    return psnr_scores, ie_scores, ssim_scores


def denormalize(batch):
    pix_mean=torch.tensor([0.485,0.456,0.406])[None, :, None, None].cuda()
    pix_std=torch.tensor([0.229,0.224,0.225])[None, :, None, None].cuda()
    batch = batch * pix_std + pix_mean
    batch = batch * 255.0
    return batch



def interpolate_frames(ssm_model, current_batch, interp_locations, info, iteration):
    interpolation_results = []
    data_batch = current_batch.cuda().float()
    T = data_batch.shape[1]
    input_idx = [i for i in range(T) if i not in interp_locations]

    if iteration==1:
        log.info("T: %s"%T)
        log.info("Using as input: ")
        log.info(input_idx)
        log.info("Targets: ")
        log.info(interp_locations)
        
    image_tensor = data_batch[:, input_idx, ...] # indices = I0, I1, I2, I3

    ground_truths = []

    for idx, t_idx in enumerate(interp_locations):
        t_interp = float(idx+1)/8
        img_t = data_batch[:, t_idx, ...] # most intermediate frame.

        if iteration==1:
            log.info("Interpolating: T=%s t=%s"%(idx+1, t_interp))

        est_img_t = ssm_model(image_tensor, info, t_interp, 'VAL', iteration, compute_loss=False)
        interpolation_results.append(est_img_t)
        ground_truths.append(img_t)
    
    interpolation_results = torch.stack(interpolation_results).squeeze()

    interpolation_results = denormalize(interpolation_results)

    ground_truths = torch.stack(ground_truths).squeeze()

    ground_truths = denormalize(ground_truths)
    
    assert interpolation_results.shape == ground_truths.shape, "Shape mismatch."

    return interpolation_results, ground_truths


def get_interp_idx(n_frames):
    if n_frames == 2:
        return list(range(1, 8))
    elif n_frames == 4:
        return list(range(2, 9))
    elif n_frames == 6:
        return list(range(3, 10))
    elif n_frames == 8:
        return list(range(4, 11))
    else:
        raise Exception("Wrong number of frames : %s"%n_frames)

if __name__ == '__main__':

    args = getargs()
    config = configparser.RawConfigParser()
    logging.basicConfig(filename=args.log, level=logging.INFO)
    config.read(args.config)
    
    ssm_net = SSMR.full_model(config) # get the Super SloMo model.

    ssm_net.cuda()
    ssm_net.eval()

    log.info("Loaded the Super SloMo model.")

    video_PSNR = []
    video_IE = []
    video_SSIM = []

    iteration = 0
    val_samples = adobe_240fps.data_generator(config, "VAL", eval=True)
    n_frames = config.getint("TRAIN", "N_FRAMES")

    interp_locations = get_interp_idx(n_frames)
    log.info(n_frames)
    log.info("Interpolating at: ")
    log.info(interp_locations)
    for aBatch, t_idx in val_samples:
        iteration +=1
        log.info("Iteration: %s"%iteration)
        output_batch, targets = interpolate_frames(ssm_net, aBatch, interp_locations, info=None, iteration=iteration)
        PSNR_score, SSIM_score, IE_score = get_scores(output_batch, targets)
        video_PSNR.extend(PSNR_score)
        video_IE.extend(IE_score)
        video_SSIM.extend(SSIM_score)
        if iteration%5==0:
            log.info(aBatch.shape)
            log.info(t_idx)
            log.info("So far: PSNR: %.3f IE: %.3f SSIM: %.3f"%(np.mean(video_PSNR), np.mean(video_IE), np.mean(video_SSIM)))

    mean_avg_psnr = np.mean(video_PSNR)
    mean_avg_IE = np.mean(video_IE)
    mean_avg_SSIM = np.mean(video_SSIM)
    log.info("Avg. per video. PSNR: %.3f IE: %.3f SSIM: %.3f"%(mean_avg_psnr, mean_avg_IE, mean_avg_SSIM))
