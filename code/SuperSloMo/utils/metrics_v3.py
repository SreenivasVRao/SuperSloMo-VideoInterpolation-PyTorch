"""
Use DataLoader to get batches of frames for the model.

"""
import numpy as np
from skimage.measure import compare_ssim, compare_psnr
import logging
import torch
import configparser, cv2, os, glob
from argparse import ArgumentParser
import adobe_240fps, slowflow, sintel_hfr

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

        dims = self.get_dims()

        (self.H_REF, self.W_REF), (self.H_IN, self.W_IN), (self.H_START, self.W_START) = dims

        self.n_frames = config.getint("TRAIN", "N_FRAMES")

        self.interp_locations = self.get_interp_idx()
        log.info(self.n_frames)
        log.info("Interpolating at: ")
        log.info(self.interp_locations)
        self.model = SSMR.full_model(cfg)  # get the Super SloMo model.
        self.model.eval() # freeze backward pass
        
        if torch.cuda.device_count()>1:
            log.info("Found %s GPUS. Using DataParallel."%torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)
        else:
            log.warning("GPUs found: "+str(torch.cuda.device_count()))
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s"%device)
        self.model.to(device)
        log.info("Loaded the Super SloMo model.")
        

    def get_dataset(self):

        dataset = self.cfg.get("MISC", "DATASET")
        if dataset  == "ADOBE":
            val_samples = adobe_240fps.data_generator(config, "VAL", eval_mode=True)
        elif dataset == "SINTEL_HFR":
            val_samples = sintel_hfr.data_generator(config, "VAL", eval_mode=True)
        elif dataset  == "SLOWFLOW":
            val_samples = slowflow.data_generator(config, "VAL", eval_mode=True)
        else:
            raise Exception("Dataset not supported: ")
        log.info("Evaluating on %s"%dataset)

        return val_samples, dataset

    def get_dims(self):

        H_IN = self.cfg.getint(self.dataset + "_DATA", "H_IN")
        W_IN = self.cfg.getint(self.dataset + "_DATA", "W_IN")

        H_REF = int(np.ceil(H_IN / 32) * 32)
        W_REF = int(np.ceil(W_IN / 32) * 32)

        H_START = (H_REF - H_IN) // 2
        W_START = (W_REF - W_IN) // 2

        log.info("H_IN: %s W_IN: %s"%(H_IN, W_IN))
        log.info("H_REF: %s W_REF: %s"%(H_REF, W_REF))
        log.info("H_START: %s W_START: %s"%(H_START, W_START))

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
                raise Exception("Wrong number of frames : %s"%self.n_frames)
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
                raise Exception("Wrong number of frames : %s"%self.n_frames)

    def get_scores(self, output_batch, target_batch):
        """
        output_batch = B, C, H, W tensor 0 - 1 range
        target_batch = B, C, H, W tensor 0 - 1 range
        """
        assert output_batch.shape == target_batch.shape, "Batch shape mismatch."
        assert output_batch.shape[2:]==(self.H_REF, self.W_REF), "Invalid shape"
        assert target_batch.shape[2:]==(self.H_REF, self.W_REF), "Invalid shape"

        B = output_batch.shape[0]

        psnr_scores = []
        ie_scores   = []
        ssim_scores = []

        output_batch = output_batch.permute(0, 2, 3, 1) # BCHW -> BHWC
        target_batch = target_batch.permute(0, 2, 3, 1) # BCHW -> BHWC

        output_batch = output_batch[:, self.H_START:self.H_START+self.H_IN,
                                    self.W_START:self.W_START+self.W_IN, ...]
        target_batch = target_batch[:, self.H_START:self.H_START+self.H_IN,
                                    self.W_START:self.W_START+self.W_IN, ...]

        assert output_batch.shape[1:4]==(self.H_IN, self.W_IN, 3), "Dimensions are incorrect."
        assert target_batch.shape[1:4]==(self.H_IN, self.W_IN, 3), "Dimensions are incorrect."

        output_batch = output_batch.cpu().data.numpy()
        target_batch = target_batch.cpu().data.numpy()
        
        for idx in range(B):
            output_image = output_batch[idx, ...] # * 255.0 # 1 H W C
            target_image = target_batch[idx, ...] # * 255.0 # 1 H W C
            
            mse_score = np.square(output_image - target_image).mean()

            rmse_score = np.sqrt(mse_score)
            psnr_score = compare_psnr(output_image.astype(np.uint8), target_image.astype(np.uint8))
            ssim_score = compare_ssim(output_image.astype(np.uint8), target_image.astype(np.uint8),
                                      multichannel=True, gaussian_weights=True)

            psnr_scores.append(psnr_score)
            ie_scores.append(rmse_score)
            ssim_scores.append(ssim_score)

        return psnr_scores, ie_scores, ssim_scores

    def denormalize(self, batch):
        pix_mean = torch.tensor([0.485,0.456,0.406])[None, :, None, None].cuda()
        pix_std  = torch.tensor([0.229,0.224,0.225])[None, :, None, None].cuda()
        batch = batch * pix_std + pix_mean
        batch = batch * 255.0
        return batch

    def get_t_interp(self, t_mid, sample_type):
        n_interp = 32 if self.dataset=="SINTEL_HFR" else 8

        if self.n_frames > 2:
            if sample_type=="A":
                left  = [np.random.randint(1, n_interp) for _ in range((self.n_frames-1) // 2)]
                right = [np.random.randint(1, n_interp) for _ in range((self.n_frames-1) // 2)]
                t_interp = left+[t_mid]+right
            elif sample_type=="B":
                t_interp = [t_mid]*(self.n_frames-1)
        else:
            t_interp = [t_mid]

        assert len(t_interp) == (self.n_frames -1) , "Incorrect number of interpolation points."
        
        raw_values = t_interp[:]

        t_interp = torch.Tensor(t_interp).view(1, (self.n_frames-1), 1, 1, 1) # B T C H W

        t_interp = t_interp.cuda().float()
        t_interp = t_interp/n_interp
        assert (0<t_interp).all() and (t_interp<1).all(), "Incorrect values."
        return t_interp, raw_values

    def interpolate_frames(self, current_batch, info, iteration):

        data_batch = current_batch.cuda().float()
        T = data_batch.shape[1]
        input_idx = [i for i in range(T) if i not in self.interp_locations]

        if iteration==1:
            log.info("T: %s"%T)
            log.info("Using as input: ")
            log.info(input_idx)
            log.info("Targets: ")
            log.info(self.interp_locations)

        image_tensor = data_batch[:, input_idx, ...] # indices = I0, I1, I2, I3

        ground_truths = []
        interpolation_results = []

        for idx, t_idx in enumerate(self.interp_locations):
            img_t = data_batch[:, t_idx, ...] # most intermediate frames.
            t_interp, raw_values = self.get_t_interp(idx+1, sample_type="B") # 1 - 7 or 1 - 31 depending on dataset.

            if iteration==1 and self.dataset!="SINTEL_HFR":
                log.info("Interpolating: T=%s t=%s"%(idx+1, float((idx+1)/8)))
                log.info(raw_values)
            elif iteration==1 and self.dataset=="SINTEL_HFR":
                log.info("Interpolating: T=%s t=%s"%(idx+1, float((idx+1)/32)))
                log.info(raw_values)

            t_interp = t_interp.expand(image_tensor.shape[0], 3, 1, 1, 1) # for multiple gpus.

            est_img_t = self.model(image_tensor, info, t_interp, iteration=iteration, compute_loss=False)
            interpolation_results.append(est_img_t)
            ground_truths.append(img_t)

        interpolation_results = torch.stack(interpolation_results)
        interpolation_results = interpolation_results.view(-1, 3, self.H_REF, self.W_REF)

        interpolation_results = self.denormalize(interpolation_results)

        ground_truths = torch.stack(ground_truths)
        ground_truths = ground_truths.view(-1, 3, self.H_REF, self.W_REF)

        ground_truths = self.denormalize(ground_truths)

        assert interpolation_results.shape == ground_truths.shape, "Shape mismatch."

        return interpolation_results, ground_truths

    def run_evaluation(self):
        iteration = 0

        for aBatch, _ in self.val_samples:
            log.info("Iteration: %s" % iteration)
            output_batch, targets = self.interpolate_frames(aBatch, info=None, iteration=iteration)
            PSNR_score, IE_score, SSIM_score= self.get_scores(output_batch, targets)

            self.video_PSNR.extend(PSNR_score)
            self.video_IE.extend(IE_score)
            self.video_SSIM.extend(SSIM_score)
            
            if iteration % 5 == 0:
                log.info(aBatch.shape)
                log.info("So far: PSNR: %.3f IE: %.3f SSIM: %.3f" % (np.mean(self.video_PSNR),
                                                                     np.mean(self.video_IE),
                                                                     np.mean(self.video_SSIM)))
            iteration+= 1

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


