from SuperSloMo.models import SSM
from SuperSloMo.utils import adobe_240fps, metrics
import datetime
import numpy as np

import torch.optim
import torch
from torch.autograd import Variable

from tensorboardX import SummaryWriter
import os
import ConfigParser


def read_config(configpath='config.ini'):
    config = ConfigParser.RawConfigParser()
    config.read(configpath)
    return config


class SSM_Main:

    def __init__(self, config):
        """
        Initializes various objects, and creates an instance of the model.
        Creates a summary writer callback for tensorboard.
        :param config: Config object.
        """

        self.cfg = config
        self.get_hyperparams()

        expt_name = config.get("EXPERIMENT", "NAME")
        expt_time= datetime.datetime.now().strftime("%d%b_%H%M%S")
        self.expt_name = expt_name + "_" + expt_time

        log_dir = os.path.join(self.cfg.get("PROJECT", "DIR"), "code", "logs", self.expt_name)
        os.makedirs(os.path.join(log_dir, self.expt_name), exist_ok=True)

        with open(os.path.join(log_dir, self.expt_name +".ini")) as f:
            config.write(f)

        self.writer = SummaryWriter(os.path.join(log_dir, self.expt_name))
        self.superslomo = self.load_model()

    def load_model(self):
        """
        Loads the models, optionally with weights, and optionally freezing individual stages.
        :return: the SuperSloMo model.
        """

        top_dir = self.cfg.get("PROJECT", "DIR")
        stage1_weights = None
        stage2_weights = None

        if self.cfg.getboolean("STAGE1", "LOADPREV"):
            stage1_weights = self.cfg.get("STAGE1", "WEIGHTS")
            stage1_weights = os.path.join(top_dir, stage1_weights)

        if self.cfg.getboolean("STAGE2", "LOADPREV"):
            stage1_weights = self.cfg.get("STAGE1", "WEIGHTS")
            stage1_weights = os.path.join(top_dir, stage1_weights)

        model = SSM.full_model(stage1_weights, stage2_weights).cuda()
        if self.cfg.getboolean("STAGE1", "FREEZE"):
            print("Freezing stage1 model.")
            model.stage1_model.eval()
            for param in model.stage1_model.parameters():
                param.requires_grad = False
        else:
            print("Training stage1 model.")

        if self.cfg.getboolean("STAGE2", "FREEZE"):
            print("Freezing stage2 model.")
            model.stage2_model.eval()
            for param in model.stage2_model.parameters():
                param.requires_grad = False
        else:
            print("Training stage2 model.")

        return model

    def get_hyperparams(self):
        """
        Reads the config to get training hyperparameters.
        :return:
        """
        lambda_r = self.cfg.getfloat("TRAIN", "LAMBDA_R") # reconstruction loss weighting
        lambda_w = self.cfg.getfloat("TRAIN", "LAMBDA_W") # warp loss weighting
        lambda_s = self.cfg.getfloat("TRAIN", "LAMBDA_S") # smoothness loss weighting
        lambda_p = self.cfg.getfloat("TRAIN", "LAMBDA_P") # perceptual loss weighting

        self.n_epochs = self.cfg.getint("TRAIN", "N_EPOCHS")
        self.learning_rate = self.cfg.getfloat("TRAIN", "LEARNING_RATE")
        self.lr_decay = self.cfg.getfloat("TRAIN", "LR_DECAY")
        self.lr_period = self.cfg.getfloat("TRAIN", "LR_PERIOD")

        self.loss_weights = lambda_r, lambda_p, lambda_w, lambda_s
        self.t_interp = self.cfg.getfloat("TRAIN", "T_INTERP")
        self.save_every= self.cfg.getint("TRAIN", "SAVE_EVERY")

    def np2torch(self, data):
        """
        Converts numpy data into Torch Variable. B H W C
        :param data:
        :return: three tensors - BCHW I_0, I_t, I_1
        """
        oneBatch = Variable(torch.from_numpy(data)).cuda().float()

        oneBatch = oneBatch.permute(0, 3, 1, 2)
        image_0 = oneBatch[::3, ...].float()
        image_t = oneBatch[1::3, ...].float()
        image_1 = oneBatch[2::3, ...].float()

        return image_0, image_t, image_1

    def write_losses(self, total_loss, individual_losses, iteration, split):
        """
        Writes the losses to tensorboard for given iteration and split.

        :param total_loss: Weighted sum of all losses.
        :param individual_losses: Tuple of 4 losses.
        :param iteration: Current iteration.
        :param split: Train/Val,
        :return:
        """

        loss_reconstr, loss_perceptual, loss_smooth, loss_warp = individual_losses

        self.writer.add_scalars('Total Loss', {split: total_loss.data[0]}, iteration)
        self.writer.add_scalars('Reconstruction Loss', {split: loss_reconstr.data[0]}, iteration)
        self.writer.add_scalars('Perceptual Loss', {split: loss_perceptual.data[0]}, iteration)
        self.writer.add_scalars('Smoothness Loss', {split: loss_smooth.data[0]}, iteration)
        self.writer.add_scalars('Warping Loss', {split: loss_warp.data[0]}, iteration)

    def forward_pass(self, numpy_batch, dataset, split, get_interpolation=False):
        """
        :param numpy_batch: B H W C 0-255, np.uint8
        :param dataset: dataset object with corresponding split.
        :param split: "TRAIN"/"TEST"/"VAL"
        :param get_interpolation: flag to return interpolation result
        :return: if get_interpolation is set, returns interpolation result as BCHW Variable.
        otherwise returns the losses.
        """

        img_0, img_t, img_1 = self.np2torch(numpy_batch)

        results = self.superslomo(img_0, img_t, dataset.dims, dataset.scale_factors, t=self.t_interp)

        flowC_input, flowC_output, flowI_input, flowI_output = results
        target_image = img_t

        if get_interpolation:

            output_image = self.superslomo.stage2_model.compute_output_image(flowI_input,
                                                                             flowI_output,
                                                                             self.t_interp)
            return output_image, target_image

        else:

            train_losses = self.superslomo.stage2_model.compute_loss(flowC_input, flowC_output,
                                                                     flowI_input, flowI_output,
                                                                     target_image, self.loss_weights,
                                                                     t=self.t_interp)
            total_loss, individual_losses = train_losses
            self.write_losses(total_loss, individual_losses, iter, split)
            return total_loss, individual_losses

    def train(self):
        """
        Training schedule for the SuperSloMo architecture.

        :return:
        """

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.superslomo.parameters()),
                                     lr=self.learning_rate)
        iter = 0

        for epoch in range(self.n_epochs):
            # shuffles the data on each epoch
            adobe_train = adobe_240fps.Reader(self.cfg, split="TRAIN")
            adobe_val = adobe_240fps.Reader(self.cfg, split="VAL")
            val_generator = adobe_val.get_clips()

            print ("Epoch: ", epoch, " Iteration: ", iter)

            for train_batch in adobe_train.get_clips():
                iter +=1

                train_loss, _ = self.forward_pass(train_batch, adobe_train, "TRAIN")

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                try:
                    val_batch = next(val_generator)
                except StopIteration:
                    adobe_val = adobe_240fps.Reader(self.cfg, split="VAL")
                    val_generator = adobe_val.get_clips()
                    val_batch = next(val_generator)

                self.forward_pass(val_batch, adobe_val, "VAL")

            if epoch%self.lr_period==0 and epoch>0:
                self.learning_rate = self.learning_rate*self.lr_decay

            if epoch%self.save_every==0:
                state = {
                    'epoch': epoch,
                    'stage1_state_dict': self.superslomo.stage1_model.state_dict(),
                    'stage2_state_dict': self.superslomo.stage2_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                fpath = os.path.join(self.cfg.get("PROJECT", "DIR"),
                                     "weights/SuperSloMo_EPOCH_"+str(epoch).zfill(4)+".pt")

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
            est_image_t, gt_image_t = self.forward_pass(a_batch, dataset,
                                                        split="TRAIN", get_interpolation=True)
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
    args = parser.parse_args()
    cfg = read_config(args.config)

    model = SSM_Main(cfg)

    model.train()
    adobe_train = adobe_240fps.Reader(cfg, split="TRAIN")
    adobe_val = adobe_240fps.Reader(cfg, split="VAL")
    adobe_test = adobe_240fps.Reader(cfg, split="TEST")

    PSNR, IE, SSIM = model.compute_metrics(adobe_train)
    print("ADOBE TRAIN: PSNR ", PSNR, " IE: ", IE, " SSIM: ", SSIM)

    PSNR, IE, SSIM = model.compute_metrics(adobe_val)
    print("ADOBE VAL: PSNR ", PSNR, " IE: ", IE, " SSIM: ", SSIM)

    PSNR, IE, SSIM = model.compute_metrics(adobe_test)
    print("ADOBE TEST: PSNR ", PSNR, " IE: ", IE, " SSIM: ", SSIM)




##################################################
# //Set the controls for the heart of the sun!// #
##################################################