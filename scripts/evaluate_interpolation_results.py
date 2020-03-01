"""
Use DataLoader to get batches of frames for the model.

"""
import configparser
import logging
from argparse import ArgumentParser

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models import superslomo_r
from utils.dataset import get_dataset
from utils.validators import (
    validate_batch_crop_dimensions,
    validate_evaluation_interpolation_result,
    validate_t_interp,
)

log = logging.getLogger(__name__)


def getargs():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, default="config.ini", help="Path to config.ini file.",
    )
    parser.add_argument("--expt", required=True, help="Experiment Name.")
    parser.add_argument("--log", required=True, help="Path to logfile.")
    args = parser.parse_args()
    return args


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.video_PSNR = []
        self.video_IE = []
        self.video_SSIM = []
        self.val_samples = get_dataset(cfg, "VAL")
        self.dataset = self.cfg.get("DATA", "DATASET")
        assert self.dataset in ["SINTEL_HFR", "ADOBE", "SLOWFLOW", "VIMEO"], "Invalid dataset."

        (
            (self.H_REF, self.W_REF),
            (self.H_IN, self.W_IN),
            (self.H_START, self.W_START),
        ) = self.get_dims()

        self.n_frames = config.getint("TRAIN", "N_FRAMES")
        self.interp_factor = 32 if self.dataset == "SINTEL_HFR" else 8

        log.info("Using %s input frames.", self.n_frames)
        log.info("Interpolating %s frames: ", (self.interp_factor - 1))

        self.load_model()

    def load_model(self):
        # self.model = superslomo.FullModel(self.cfg)
        # get the Super SloMo model.
        self.model = superslomo_r.FullModel(self.cfg)
        self.model.eval()  # freeze backward pass

        if torch.cuda.device_count() > 1:
            log.info("Found %s GPUS. Using DataParallel.", torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)
        else:
            log.warning("GPUs found: %s", str(torch.cuda.device_count()))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s", device)
        self.model.to(device)
        log.info("Loaded the Super SloMo model.")

    def get_dims(self):
        """ Retrieves the dimensions of the images in the dataset.
        Also calculates the nearest multiple of 32 that can be used to crop the image.
        This is necessary to make the U-Net model with 5 stages work.

        :returns:
        :rtype:

        """
        section = self.dataset+"_DATA"
        H_IN = self.cfg.getint(section, "H_IN")
        W_IN = self.cfg.getint(section, "W_IN")

        H_REF = int(np.ceil(H_IN / 32) * 32)
        W_REF = int(np.ceil(W_IN / 32) * 32)

        H_START = (H_REF - H_IN) // 2
        W_START = (W_REF - W_IN) // 2

        log.info("H_IN: %s W_IN: %s", H_IN, W_IN)
        log.info("H_REF: %s W_REF: %s", H_REF, W_REF)
        log.info("H_START: %s W_START: %s", H_START, W_START)

        return (H_REF, W_REF), (H_IN, W_IN), (H_START, W_START)

    def eval_single_image(self, target_image, output_image):
        PSNR = peak_signal_noise_ratio(target_image, output_image)
        SSIM = structural_similarity(
            target_image, output_image, multichannel=True, gaussian_weights=True
        )
        IE = target_image.astype(float) - output_image.astype(float)
        IE = np.mean(np.sqrt(np.sum(IE * IE, axis=2)))
        return PSNR, SSIM, IE

    def eval_batch(self, input_batch, target_batch, n_avail):
        outputs = self.interpolate_frames(input_batch)
        # gets a list of length = self.interp_factor - 1 frames (B C H W) interpolation..

        outputs = torch.stack(outputs, dim=1)  # B 7 C H W
        outputs = list(outputs.split(dim=0, split_size=1))  # B length list. 1 T C H W
        targets = list(target_batch.split(dim=0, split_size=1))

        assert len(outputs) == len(n_avail)
        new_outputs = []
        new_targets = []
        for idx, n in enumerate(n_avail):
            if n < self.interp_factor - 1:  # handles the edges of the original clip.
                log.info("Found an end clip: %s", n)
                log.info(outputs[idx].shape)
                log.info(outputs[idx][:, :n, ...].shape)
                # trim if less than 7 interpolations to be considered.
                outputs[idx] = outputs[idx][:, :n, ...]
                targets[idx] = targets[idx][:, :n, ...]  # 1 T C H W tensor.
            new_outputs.append(outputs[idx])
            new_targets.append(targets[idx])

        new_outputs = torch.cat(new_outputs, dim=1)  # concat along t axis. 1 [T1+T2+T3] C H W
        new_targets = torch.cat(new_targets, dim=1)  # concat along t axis.

        new_outputs = new_outputs.view(-1, 3, self.H_REF, self.W_REF)  # [T1+T2+T3] C H W
        new_targets = new_targets.view(-1, 3, self.H_REF, self.W_REF)

        PSNR_score, IE_score, SSIM_score = self.get_scores(new_outputs, new_targets)
        self.video_PSNR.extend(PSNR_score)
        self.video_IE.extend(IE_score)
        self.video_SSIM.extend(SSIM_score)

    @validate_batch_crop_dimensions
    def get_crop(self, batch):
        """
        Removes the padding to make dims a factor of 32.
        """
        batch = batch.permute(0, 2, 3, 1)  # BCHW -> BHWC

        batch = batch[
            :,
            self.H_START : self.H_START + self.H_IN,
            self.W_START : self.W_START + self.W_IN,
            ...,
        ]

        return batch

    def convert_tensor_to_numpy_image(self, batch):
        batch = self.get_crop(batch)
        batch = self.denormalize(batch)
        batch = batch.cpu().data.numpy().astype(np.uint8)
        return batch

    def get_scores(self, output_batch, target_batch):
        """
        output_batch = B, C, H, W tensor 0 - 1 range
        target_batch = B, C, H, W tensor 0 - 1 range
        """

        B = output_batch.shape[0]

        psnr_scores = []
        ie_scores = []
        ssim_scores = []

        output_batch = self.convert_tensor_to_numpy_image(output_batch)
        target_batch = self.convert_tensor_to_numpy_image(target_batch)

        for idx in range(B):
            output_image = output_batch[idx, ...]  # * 255.0 # 1 H W C
            target_image = target_batch[idx, ...]  # * 255.0 # 1 H W C

            psnr_score, ssim_score, ie_score = self.eval_single_image(target_image, output_image)

            psnr_scores.append(psnr_score)
            ie_scores.append(ie_score)
            ssim_scores.append(ssim_score)

        return psnr_scores, ie_scores, ssim_scores

    def denormalize(self, batch):
        pix_mean = self.cfg.get("MODEL", "PIXEL_MEAN").split(",")
        pix_mean = [float(p) for p in pix_mean]
        pix_std = self.cfg.get("MODEL", "PIXEL_STD").split(",")
        pix_std = [float(p) for p in pix_std]

        pix_mean = torch.tensor(pix_mean).view(1, 1, 1, -1).cuda()
        pix_std = torch.tensor(pix_std).view(1, 1, 1, -1).cuda()
        batch = batch * pix_std + pix_mean
        batch = batch * 255.0
        return batch

    @validate_t_interp
    def get_t_interp_tensor(self, t_mid):

        t_interp = [t_mid] * (self.n_frames - 1)
        t_interp = torch.Tensor(t_interp).view(1, (self.n_frames - 1), 1, 1, 1)  # B T C H W
        t_interp = t_interp.cuda().float()
        t_interp = t_interp / self.interp_factor
        return t_interp

    @validate_evaluation_interpolation_result
    def interpolate_frames(self, current_batch):

        interpolation_results = []

        # VIMEO = only 30 FPS sequences.
        # interpolate most intermediate frame.
        if self.dataset == "VIMEO":
            t_interp = self.get_t_interp_tensor(4)

            t_interp = t_interp.expand(
                current_batch.shape[0], self.n_frames - 1, 1, 1, 1
            )  # for multiple gpus.

            est_img_t, _ = self.model(current_batch, t_interp, inference_mode=True)
            interpolation_results.append(est_img_t)  # B C H W

        else:
            # use every interp position
            # as input value in the equation from Jiang et al.
            # (1-7) or (1-31) depending on dataset
            for idx in range(1, self.interp_factor):
                t_interp = self.get_t_interp_tensor(idx)

                t_interp = t_interp.expand(
                    current_batch.shape[0], self.n_frames - 1, 1, 1, 1
                )  # for multiple gpus.

                est_img_t, _ = self.model(current_batch, t_interp, inference_mode=True)
                interpolation_results.append(est_img_t)  # B C H W

        return interpolation_results

    def run_evaluation(self):
        iteration = 0
        n_iterations = len(self.val_samples)
        log.info("Running evaluations for %s iterations", n_iterations)
        for input_batch, target_batch, n_avail in self.val_samples:
            if input_batch.shape[0] < torch.cuda.device_count():
                continue
            input_batch = input_batch.cuda().float()
            target_batch = target_batch.cuda().float()
            self.eval_batch(
                input_batch, target_batch, n_avail
            )  # applies a sliding window evaluation.
            if iteration % 10 == 0:
                log.info("Iteration: %s of %s", iteration, n_iterations)
                log.info(input_batch.shape)
                log.info(target_batch.shape)
                log.info(
                    "So far: PSNR: %.3f IE: %.3f SSIM: %.3f",
                    np.mean(self.video_PSNR),
                    np.mean(self.video_IE),
                    np.mean(self.video_SSIM),
                )
            iteration += 1

        mean_avg_psnr = np.mean(self.video_PSNR)
        mean_avg_IE = np.mean(self.video_IE)
        mean_avg_SSIM = np.mean(self.video_SSIM)
        log.info(
            "Avg. per video. PSNR: %.3f IE: %.3f SSIM: %.3f",
            mean_avg_psnr,
            mean_avg_IE,
            mean_avg_SSIM,
        )


if __name__ == "__main__":
    args = getargs()
    config = configparser.RawConfigParser()
    logging.basicConfig(filename=args.log, level=logging.INFO)
    config.read(args.config)
    torch.backends.cudnn.benchmark = True
    e = Evaluator(config)
    e.run_evaluation()
