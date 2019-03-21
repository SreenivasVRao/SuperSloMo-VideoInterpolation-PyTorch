"""
Use DataLoader to get batches of frames for the model.

"""
import numpy as np
from skimage.measure import compare_ssim, compare_psnr
import logging
import torch
import configparser, cv2, os, glob
from argparse import ArgumentParser
import adobe_eval, slowflow, sintel_hfr
import torch.nn.functional as F
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


class Evaluator:

    def __init__(self, cfg):
        self.cfg = cfg

        self.video_PSNR = []
        self.video_IE = []
        self.video_SSIM = []
        self.val_samples, self.dataset = self.get_dataset()

        assert self.dataset in ["SINTEL_HFR", "ADOBE", "SLOWFLOW"], "Invalid dataset."

        dims = self.get_dims()

        (self.H_REF, self.W_REF), (self.H_IN, self.W_IN), (self.H_START, self.W_START) = dims

        self.n_frames = config.getint("TRAIN", "N_FRAMES")
        self.interp_factor = 32 if self.dataset == "SINTEL_HFR" else 8
        self.interp_locations = self.get_interp_idx()
        log.info("Using %s input frames." % self.n_frames)
        log.info("Interpolating %s frames: " % (self.interp_factor - 1))
        log.info("Interpolating at: ")
        log.info(self.interp_locations)

        self.load_model()

    def load_model(self):
        self.model = SSMR.full_model(self.cfg)  # get the Super SloMo model.
        self.model.eval()  # freeze backward pass

        if torch.cuda.device_count() > 1:
            log.info("Found %s GPUS. Using DataParallel." % torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)
        else:
            log.warning("GPUs found: " + str(torch.cuda.device_count()))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s" % device)
        self.model.to(device)
        log.info("Loaded the Super SloMo model.")

    def get_dataset(self):

        dataset = self.cfg.get("MISC", "DATASET")
        if dataset == "ADOBE":
            val_samples = adobe_eval.data_generator(config, "VAL", eval_mode=True)
        elif dataset == "SINTEL_HFR":
            val_samples = sintel_hfr.data_generator(config, "VAL", eval_mode=True)
        elif dataset == "SLOWFLOW":
            val_samples = slowflow.data_generator(config, "VAL", eval_mode=True)
        else:
            raise Exception("Dataset not supported: ")
        log.info("Evaluating on %s" % dataset)

        return val_samples, dataset

    def get_dims(self):

        H_IN = self.cfg.getint(self.dataset + "_DATA", "H_IN")
        W_IN = self.cfg.getint(self.dataset + "_DATA", "W_IN")

        H_REF = int(np.ceil(H_IN / 32) * 32)
        W_REF = int(np.ceil(W_IN / 32) * 32)

        H_START = (H_REF - H_IN) // 2
        W_START = (W_REF - W_IN) // 2

        log.info("H_IN: %s W_IN: %s" % (H_IN, W_IN))
        log.info("H_REF: %s W_REF: %s" % (H_REF, W_REF))
        log.info("H_START: %s W_START: %s" % (H_START, W_START))

        return (H_REF, W_REF), (H_IN, W_IN), (H_START, W_START)

    def get_interp_idx(self):

        if self.dataset == "SINTEL_HFR":
            log.info("USING SINTEL HFR settings.")
            if self.n_frames == 2:
                return list(range(1, 32))
            elif self.n_frames == 4:
                return list(range(2, 33))
            elif self.n_frames == 6:
                return list(range(3, 34))
            elif self.n_frames == 8:
                return list(range(4, 35))
            else:
                raise Exception("Wrong number of frames : %s" % self.n_frames)
        else:
            if self.n_frames == 2:
                return list(range(1, 8))
            elif self.n_frames == 4:
                return list(range(2, 9))
            elif self.n_frames == 6:
                return list(range(3, 10))
            elif self.n_frames == 8:
                return list(range(4, 11))
            else:
                raise Exception("Wrong number of frames : %s" % self.n_frames)

    def eval_single_image(self, image1, image2):
        PSNR = compare_psnr(image1, image2)
        SSIM = compare_ssim(image1, image2,
                            multichannel=True,
                            gaussian_weights=True)
        IE = image1.astype(float) - image2.astype(float)
        IE = np.mean(np.sqrt(np.sum(IE * IE, axis=2)))
        return PSNR, SSIM, IE

    def eval_batch(self, batch, n_avail):
        outputs, targets = self.interpolate_frames(batch, info=None)
        assert len(outputs)== self.interp_factor -1 , "Wrong number of outputs."
        # gets a list of length = self.interp_factor - 1 frames (B C H W) interpolation..

        outputs = torch.stack(outputs, dim=1) # B 7 C H W
        targets = torch.stack(targets, dim=1)
        outputs = list(outputs.split(dim=0, split_size=1)) # B length list. 1 T C H W
        targets = list(targets.split(dim=0, split_size=1))
        
        assert len(outputs) == len(n_avail)
        new_outputs = []
        new_targets = []
        for idx, n in enumerate(n_avail):
            if n < self.interp_factor-1: # handles the edges of the original clip.
                log.info("Found an end clip: %s"%n)
                log.info(outputs[idx].shape)
                log.info(outputs[idx][:, :n, ...].shape)
                outputs[idx] = outputs[idx][:, :n, ...] # trim if less than 7 interpolations to be considered.
                targets[idx] = targets[idx][:, :n, ...] # 1 T C H W tensor.
            new_outputs.append(outputs[idx])
            new_targets.append(targets[idx])
            
        new_outputs = torch.cat(new_outputs, dim=1) # concat along t axis. 1 [T1+T2+T3] C H W
        new_targets = torch.cat(new_targets, dim=1) # concat along t axis.
        
        new_outputs = new_outputs.view(-1, 3, self.H_REF, self.W_REF) # [T1+T2+T3] C H W
        new_targets = new_targets.view(-1, 3, self.H_REF, self.W_REF)
        
        PSNR_score, IE_score, SSIM_score = self.get_scores(new_outputs, new_targets)
        self.video_PSNR.extend(PSNR_score)
        self.video_IE.extend(IE_score)
        self.video_SSIM.extend(SSIM_score)

    def get_crop(self, batch):
        """
        Removes the padding to make dims a factor of 32.
        """
        assert batch.shape[2:] == (self.H_REF, self.W_REF), "Invalid shape"

        B = batch.shape[0]
        
        batch = batch.permute(0, 2, 3, 1)  # BCHW -> BHWC

        batch = batch[:, self.H_START:self.H_START + self.H_IN,
                       self.W_START:self.W_START + self.W_IN, ...]

        assert batch.shape[1:4] == (self.H_IN, self.W_IN, 3), "Dimensions are incorrect."
        
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

        output_batch = self.get_crop(output_batch)
        target_batch = self.get_crop(target_batch)

        output_batch = self.denormalize(output_batch)
        target_batch = self.denormalize(target_batch)

        output_batch = output_batch.cpu().data.numpy().astype(np.uint8)
        target_batch = target_batch.cpu().data.numpy().astype(np.uint8)

        for idx in range(B):
            output_image = output_batch[idx, ...]  # * 255.0 # 1 H W C
            target_image = target_batch[idx, ...]  # * 255.0 # 1 H W C

            psnr_score, ssim_score, ie_score = self.eval_single_image(output_image, target_image)

            psnr_scores.append(psnr_score)
            ie_scores.append(ie_score)
            ssim_scores.append(ssim_score)

        return psnr_scores, ie_scores, ssim_scores

    def denormalize(self, batch):
        pix_mean = torch.tensor([0.485, 0.456, 0.406])[None, None, None, :].cuda()
        pix_std = torch.tensor([0.229, 0.224, 0.225])[None, None, None, :].cuda()
        batch = batch * pix_std + pix_mean
        batch = batch * 255.0
        return batch

    def get_t_interp(self, t_mid):

        t_interp = [t_mid] * (self.n_frames - 1)
        raw_values = t_interp[:]
        t_interp = torch.Tensor(t_interp).view(1, (self.n_frames - 1), 1, 1, 1)  # B T C H W
        t_interp = t_interp.cuda().float()
        t_interp = t_interp / self.interp_factor
        assert (0 < t_interp).all() and (t_interp < 1).all(), "Incorrect values."
        return t_interp, raw_values

    def interpolate_frames(self, current_batch, info):

        data_batch = current_batch.cuda().float()
        T = data_batch.shape[1]
        input_idx = [i for i in range(T) if i not in self.interp_locations]

        image_tensor = data_batch[:, input_idx, ...]  # indices = I0, I1, I2, I3

        ground_truths = []
        interpolation_results = []

        for idx, t_idx in enumerate(self.interp_locations):
            img_t = data_batch[:, t_idx, ...]  # most intermediate frames.
            t_interp, raw_values = self.get_t_interp(idx + 1)  # 1 - 7 or 1 - 31 depending on dataset.

            t_interp = t_interp.expand(image_tensor.shape[0], self.n_frames - 1, 1, 1, 1)  # for multiple gpus.

            est_img_t = self.model(image_tensor, info, t_interp, iteration=None, compute_loss=False)
            interpolation_results.append(est_img_t) # B C H W
            ground_truths.append(img_t)

        return interpolation_results, ground_truths

    def run_evaluation(self):
        iteration = 0
        n_iterations = len(self.val_samples)
        for aBatch, n_avail in self.val_samples:
            if aBatch.shape[0] < torch.cuda.device_count():
                continue
            
            self.eval_batch(aBatch, n_avail) # applies a sliding window evaluation.
            if iteration % 10 == 0:
                log.info("Iteration: %s of %s" % (iteration, n_iterations))
                log.info(aBatch.shape)
                log.info("So far: PSNR: %.3f IE: %.3f SSIM: %.3f" % (np.mean(self.video_PSNR),
                                                                     np.mean(self.video_IE),
                                                                     np.mean(self.video_SSIM)))
            iteration += 1

        mean_avg_psnr = np.mean(self.video_PSNR)
        mean_avg_IE = np.mean(self.video_IE)
        mean_avg_SSIM = np.mean(self.video_SSIM)
        log.info("Avg. per video. PSNR: %.3f IE: %.3f SSIM: %.3f" % (mean_avg_psnr, mean_avg_IE, mean_avg_SSIM))


if __name__ == '__main__':
    args = getargs()
    config = configparser.RawConfigParser()
    logging.basicConfig(filename=args.log, level=logging.INFO)
    config.read(args.config)
    torch.backends.cudnn.benchmark = True
    e = Evaluator(config)
    e.run_evaluation()


