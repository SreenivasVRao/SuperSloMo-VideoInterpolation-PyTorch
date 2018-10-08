from SuperSloMo.models import SSM, SSMLosses
from SuperSloMo.utils import adobe_240fps, metrics
import numpy as np
import time
import torch.optim
import torch

from tensorboardX import SummaryWriter
import os, logging, ConfigParser

log = logging.getLogger(__name__)


def read_config(configpath='config.ini'):
    config = ConfigParser.RawConfigParser()
    config.read(configpath)
    return config


class SSM_Main:

    def __init__(self, config, expt_name, message=None):
        """
        Initializes various objects, and creates an instance of the model.
        Creates a summary writer callback for tensorboard.
        :param config: Config object.
        """

        self.cfg = config
        self.get_hyperparams()
        self.expt_name = expt_name
        self.msg = message
        log_dir = os.path.join(self.cfg.get("PROJECT","DIR"), "logs")

        os.makedirs(os.path.join(log_dir, self.expt_name, "plots"))

        self.writer = SummaryWriter(os.path.join(log_dir, self.expt_name, "plots"))
        if message:
            self.writer.add_text("ExptInfo", message)
        self.superslomo = SSM.full_model(self.cfg).cuda()
        if torch.cuda.device_count()>0:
            self.superslomo = torch.nn.DataParallel(self.superslomo)
        self.loss = SSMLosses.get_loss(self.cfg).cuda()

    def get_hyperparams(self):
        """
        Reads the config to get training hyperparameters.
        :return:
        """

        self.n_epochs = self.cfg.getint("TRAIN", "N_EPOCHS")
        self.learning_rate = self.cfg.getfloat("TRAIN", "LEARNING_RATE")
        self.lr_decay = self.cfg.getfloat("TRAIN", "LR_DECAY")
        self.lr_period = self.cfg.getfloat("TRAIN", "LR_PERIOD")

        self.t_interp = self.cfg.getfloat("TRAIN", "T_INTERP")
        self.save_every= self.cfg.getint("TRAIN", "SAVE_EVERY")

    def write_losses(self, total_loss, individual_losses, iteration, split):
        """
        Writes the losses to tensorboard for given iteration and split.

        :param total_loss: Weighted sum of all losses.
        :param individual_losses: Tuple of 4 losses.
        :param iteration: Current iteration.
        :param split: Train/Val,
        :return:
        """

        loss_reconstr, loss_perceptual, loss_warp = individual_losses

        self.writer.add_scalars('Total_Loss', {split: total_loss.data[0]}, iteration)
        self.writer.add_scalars('Reconstruction_Loss', {split: loss_reconstr.data[0]}, iteration)
        self.writer.add_scalars('Perceptual_Loss', {split: loss_perceptual.data[0]}, iteration)
        # self.writer.add_scalars('Smoothness_Loss', {split: loss_smooth.data[0]}, iteration)
        self.writer.add_scalars('Warping_Loss', {split: loss_warp.data[0]}, iteration)

    def forward_pass(self, data_batch, dataset_info, split, iteration, get_interpolation=False):
        """
        :param data_batch: B H W C 0-255, np.uint8
        :param dataset_info: dataset object with corresponding split.
        :param split: "TRAIN"/"TEST"/"VAL"
        :param get_interpolation: flag to return interpolation result
        :return: if get_interpolation is set, returns interpolation result as BCHW Variable.
        otherwise returns the losses.
        """
        data_batch = data_batch.cuda().float()
        img_0 = data_batch[:, 0, ...]
        img_t = data_batch[:, 1, ...]
        img_1 = data_batch[:,-1, ...]

        results = self.superslomo(img_0, img_1, dataset_info, self.t_interp)
        loss_flag = split in ["TRAIN", "VAL"]
        if loss_flag:
            total_loss = self.loss(*results, target_image=img_t)[0]
            return total_loss

    def train(self):
        """
        Training schedule for the SuperSloMo architecture.

        :return:
        """

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.superslomo.parameters()),
                                     lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_period,gamma=self.lr_decay)
        iteration = 0

        prev_t = time.time()

        for epoch in range(self.n_epochs):
            # shuffles the data on each epoch
            adobe_train_samples = adobe_240fps.data_generator(self.cfg, split="TRAIN")
            adobe_val_samples = adobe_240fps.data_generator(self.cfg, split="VAL")

            train_info = adobe_240fps.get_data_info(self.cfg, split="TRAIN")
            val_info = adobe_240fps.get_data_info(self.cfg, split="VAL")

            log.info("Epoch: "+str(epoch)+" Iteration: "+str(iteration))

            for train_batch in adobe_train_samples:
                iteration +=1

                train_loss = self.forward_pass(train_batch, train_info, "TRAIN", iteration)


                optimizer.zero_grad()
                train_loss.backward()
                lr_scheduler.step()

                try:
                    val_batch = next(adobe_val_samples)
                except StopIteration:
                    adobe_val_samples = adobe_240fps.data_generator(self.cfg, split="VAL")
                    val_batch = next(adobe_val_samples)

                self.forward_pass(val_batch, val_info, "VAL", iteration)
                delta_t = time.time() - prev_t
                prev_t = time.time()
                log.info(str(iteration)+" "+str(delta_t))
                if iteration==20:
                    break
            break

            if epoch%self.save_every==0:
                if isinstance(self.superslomo, torch.nn.DataParallel):
                    model = self.superslomo.module
                else:
                    model = self.superslomo
                state = {
                    'epoch': epoch,
                    'stage1_state_dict': model.stage1_model.state_dict(),
                    'stage2_state_dict': model.stage2_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                fpath = os.path.join(self.cfg.get("PROJECT", "DIR"), "logs",
                                     self.expt_name, self.expt_name+"_EPOCH_"+str(epoch).zfill(4)+".pt")

                torch.save(state, fpath)

        self.writer.close()

    def compute_metrics(self, dataset):
        """
        Computes PSNR, Interpolation Error, and SSIM scores for the given split of the dataset.
        :param dataset:
        :return: avg PSNR, avg IE, avg SSIM
        """
        total_ssim = 0
        total_IE = 0
        total_PSNR = 0

        nframes = 0

        for a_batch in dataset.get_clips():
            nframes += a_batch.shape[0]/3
            est_image_t, gt_image_t = self.forward_pass(a_batch, dataset,"TRAIN", iter,  get_interpolation=True)
            est_image_t = est_image_t * 255.0

            est_image_t = est_image_t.permute(0, 2, 3, 1) # BCHW -> BHWC
            gt_image_t  = est_image_t.permute(0, 2, 3, 1) # BCHW -> BHWC

            est_image_t = est_image_t.cpu().data.numpy()
            gt_image_t  =  gt_image_t.cpu().data.numpy()

            est_image_t = est_image_t.astype(np.uint8)
            gt_image_t  =  gt_image_t.astype(np.uint8)

            IE_scores = metrics.interpolation_error(est_image_t, gt_image_t)
            ssim_scores = metrics.ssim(est_image_t, gt_image_t)
            psnr_scores = metrics.psnr(est_image_t, gt_image_t)

            total_IE   += np.sum(IE_scores)
            total_ssim += np.sum(ssim_scores)
            total_PSNR += np.sum(psnr_scores)

        avg_IE = float(total_IE)/nframes
        avg_ssim = float(total_ssim)/nframes
        avg_PSNR = float(total_PSNR)/nframes

        return avg_PSNR, avg_IE, avg_ssim


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        default="config.ini",
                        help="Path to config.ini file.")
    parser.add_argument("--expt", required=True,
                        help="Experiment Name.")

    parser.add_argument("--log", required=True, help="Path to logfile.")

    parser.add_argument("--msg", help="(Optional) Details of experiment stored with TensorBoard.")

    args = parser.parse_args()

    logging.basicConfig(filename=args.log, level=logging.INFO)

    cfg = read_config(args.config)

    model = SSM_Main(cfg, args.expt, args.msg)

    model.train()

    """
    adobe_train = adobe_240fps.Reader(cfg, split="TRAIN")
    adobe_val = adobe_240fps.Reader(cfg, split="VAL")

    PSNR, IE, SSIM = model.compute_metrics(adobe_train)
    logging.info("ADOBE TRAIN: PSNR ", PSNR, " IE: ", IE, " SSIM: ", SSIM)

    PSNR, IE, SSIM = model.compute_metrics(adobe_val)
    logging.info("ADOBE VAL: PSNR ", PSNR, " IE: ", IE, " SSIM: ", SSIM)

    PSNR, IE, SSIM = model.compute_metrics(adobe_test)
    logging.info("ADOBE TEST: PSNR ", PSNR, " IE: ", IE, " SSIM: ", SSIM)
    """


##################################################
# //Set the controls for the heart of the sun!// #
##################################################
