import numpy as np
from skimage.measure import compare_ssim, compare_psnr
import logging
import torch
import ConfigParser, cv2, os, glob
from argparse import ArgumentParser
import sys
sys.path.insert(0, "/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/SuperSloMo/models/")
import SSM

log = logging.getLogger(__name__)

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
        psnr_score = compare_psnr(output_image.astype(np.uint8), target_image.astype(np.uint8))
        ssim_score = compare_ssim(output_image.astype(np.uint8), target_image.astype(np.uint8),
                                  multichannel=True, gaussian_weights=True)

        psnr_scores.append(psnr_score)
        ie_scores.append(rmse_score)
        ssim_scores.append(ssim_score)
        
    return psnr_scores, ie_scores, ssim_scores


def interpolate_frames(current_window, vFlag, info):
    current_images = current_window[0::8] #[I_0, I_1, I_2, I_3] [0, 8, 16, 24]

    image_tensor = [get_image(impath, vFlag) for impath in current_images]
    image_tensor = torch.stack(image_tensor, dim=1)

    img_0 = image_tensor[:, 1, ...] #[I0, I1, I2, I3]
    img_1 = image_tensor[:, 2, ...] #[I0, I1, I2, I3]

    interpolation_results = []

    for idx in range(1, 8):
        t_interp = float(idx)/8
        est_image_t = ssm_net(image_tensor, info, t_interp, split="VAL", compute_loss=False)
        interpolation_results.append(est_image_t)

    interpolation_results = torch.stack(interpolation_results).squeeze()
    ground_truth = [get_image(impath, vFlag) for impath in current_window[9:16]] # interpolations between I1 and I2.
    ground_truth = torch.stack(ground_truth).squeeze()

    return interpolation_results, ground_truth


def process_one_clip(clip_id, fpath):
    images_list = glob.glob(os.path.join(fpath, "*.png"))
    images_list.sort()
    log.info("Interpolation beginning.")
    log.info("Clip: %s has %s frames"%(clip_id, len(images_list)))

    img_0 = cv2.imread(images_list[0])
    h,w,c = img_0.shape
    vFlag = h>w # horizontal video => h<=w vertical video => w< h
    info=(736, 1280),(1.0, 1.0)

    count = 0
    start_idx = 0
    window = 25
    overlap = 17
    end_idx = start_idx + window
    iteration = 0

    PSNR_score = []
    IE_score = []
    SSIM_score = []

    while end_idx <= len(images_list):
        iteration +=1
        current_window = images_list[start_idx:end_idx]
        assert len(current_window) == window, "Size of input is wrong." #[I_0 - I_3] [0 - 24]
        interpolation_results, ground_truth = interpolate_frames(current_window, vFlag, info)
        psnr, ie, ssim = get_scores(interpolation_results, ground_truth)
        PSNR_score.append(psnr)
        IE_score.append(ie)
        SSIM_score.append(ssim)
        start_idx = start_idx + window - overlap
        end_idx = start_idx + window

    PSNR_score = [psnr for window in PSNR_score for psnr in window]
    SSIM_score = [ssim for window in SSIM_score for ssim in window]
    IE_score = [ie for window in IE_score for ie in window]

    return PSNR_score, SSIM_score, IE_score
    

def get_image(path, flipFlag):
    img = cv2.imread(path)
    img = img/255.0
    if flipFlag:
        img = img.swapaxes(0, 1)
    img =  torch.from_numpy(img)
    img = img.cuda().float()
    img = img[None, ...]
    img = img.permute(0, 3, 1, 2) # bhwc => bchw
    pad = torch.nn.ZeroPad2d([0,0, 8, 8])
    img = pad(img)
    return img


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


if __name__ == '__main__':

    args = getargs()
    config = ConfigParser.RawConfigParser()
    logging.basicConfig(filename=args.log, level=logging.INFO)
    config.read(args.config)
    
    ssm_net = SSM.full_model(config) # get the Super SloMo model.

    ssm_net.cuda()
    ssm_net.eval()

    log.info("Loaded the Super SloMo model.")

    val_file = config.get("ADOBE_DATA", "VALPATHS")

    clips = set()

    with open(val_file, 'rb') as f:
        data = f.readlines()
        data = [d.strip() for d in data]
        for d in data:
            if 'clip' in d:
                clip_id = d.split('/')[-2]
                clips.add(clip_id)

    video_PSNR = []
    video_IE = []
    video_SSIM = []

    for c in clips:
        log.info(c)
                
    for clip_id in clips:
        fpath = "/mnt/nfs/work1/elm/hzjiang/Data/VideoInterpolation/Adobe240fps/Clips/"+clip_id
        PSNR_score, SSIM_score, IE_score = process_one_clip(clip_id, fpath)
        video_PSNR.extend(PSNR_score)
        video_IE.extend(IE_score)
        video_SSIM.extend(SSIM_score)
        log.info("Interpolation complete.")

    mean_avg_psnr = np.mean(video_PSNR)
    mean_avg_IE = np.mean(video_IE)
    mean_avg_SSIM = np.mean(video_SSIM)
    log.info("Avg. per video. PSNR: %.3f IE: %.3f SSIM: %.3f"%(mean_avg_psnr, mean_avg_IE, mean_avg_SSIM))
