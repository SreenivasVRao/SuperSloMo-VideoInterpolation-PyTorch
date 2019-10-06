import configparser
import logging
from argparse import ArgumentParser
import numpy as np
from SuperSloMo.models import SSMR
from SuperSloMo.utils.vimeo_eval import *
from skimage.measure import compare_psnr, compare_ssim

def denormalize(batch):
    pix_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda()\
                                                                    .float()
    pix_std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()\
                                                                   .float()
    batch = batch * pix_std + pix_mean
    batch = batch * 255.0
    return batch

def eval_single_image(tensor1, tensor2):
    image1 = tensor1.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
    image2 = tensor2.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
    assert image1.shape[0]==image2.shape[0]==1
    image1 = image1[0, ...]
    image2 = image2[0, ...]
    PSNR = compare_psnr(image1, image2)
    SSIM = compare_ssim(image1, image2,
                        multichannel=True,
                        gaussian_weights=True)
    IE = image1.astype(float) - image2.astype(float)
    IE = np.mean(np.sqrt(np.sum(IE * IE, axis=2)))
    return PSNR, SSIM, IE

parser = ArgumentParser()
parser.add_argument("--log")
parser.add_argument("-c", "--config")  # config
args = parser.parse_args()

logging.basicConfig(filename=args.log, level=logging.INFO)

config = configparser.RawConfigParser()
config.read(args.config)
logging.info("Read config")

PSNR_scores = []
SSIM_scores = []
IE_scores = []


n_frames = config.getint("TRAIN", "N_FRAMES")

vimeo_samples = data_generator(config, split="VAL", eval_mode=True)
superslomo = SSMR.full_model(config).cuda().eval()

t_interp = torch.Tensor([0.5]* (n_frames-1)).cuda().float()
t_interp = t_interp.view(1, -1, 1, 1, 1)

# data = []

for idx, (input_tensor, seq, pos) in enumerate(vimeo_samples):
    if n_frames == 4:
        image_tensor = input_tensor[:, [0, 1, 3, 4], ...].cuda().float()
        # B 5 C H W -> B 4 C H W
        target_tensor = input_tensor[:, 2, ...].cuda().float()
    elif n_frames == 2:
        image_tensor = input_tensor[:, [0, 2], ...].cuda().float()
        # B 3 C H W -> B 2 C H W
        target_tensor = input_tensor[:, 1, ...].cuda().float()

    t_interp = t_interp.expand(image_tensor.shape[0], n_frames-1, 1, 1, 1)
    # for multiple gpus.
    est_img_t = superslomo(image_tensor, None, t_interp, iteration=None,
                           compute_loss=False)[0]

    est_img_t = denormalize(est_img_t)
    target_tensor = denormalize(target_tensor)
    psnr, ssim, ie = eval_single_image(est_img_t, target_tensor)
    PSNR_scores.append(psnr)
    SSIM_scores.append(ssim)
    IE_scores.append(ie)

    if idx%10 == 0:
        logging.info("Iteration: %s of %s"%(idx, len(vimeo_samples)))
        logging.info(image_tensor.shape)
        mean_avg_psnr = np.mean(PSNR_scores)
        mean_avg_IE = np.mean(IE_scores)
        mean_avg_SSIM = np.mean(SSIM_scores)
        logging.info("So far: PSNR: %.3f IE: %.3f SSIM: %.3f" % (mean_avg_psnr,
                                                                 mean_avg_IE,
                                                                 mean_avg_SSIM))

mean_avg_psnr = np.mean(PSNR_scores)
mean_avg_IE = np.mean(IE_scores)
mean_avg_SSIM = np.mean(SSIM_scores)
logging.info("Avg. per video. PSNR: %.3f IE: %.3f SSIM: %.3f" % (mean_avg_psnr,
                                                                 mean_avg_IE,
                                                                 mean_avg_SSIM))








