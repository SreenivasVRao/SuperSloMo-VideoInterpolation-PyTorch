from SuperSloMo.models import SSM
from SuperSloMo.utils import adobe_240fps
import datetime

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

    def get_batch(self, data):
        oneBatch = Variable(torch.from_numpy(data)).cuda().float()

        oneBatch = oneBatch.permute(0, 3, 1, 2)
        image_0 = oneBatch[::3, ...].float()
        image_t = oneBatch[1::3, ...].float()
        image_1 = oneBatch[2::3, ...].float()

        return image_0, image_t, image_1

    def write_losses(self, total_loss, individual_losses, iteration, split):

        loss_reconstr, loss_perceptual, loss_smooth, loss_warp = individual_losses

        self.writer.add_scalars('Total Loss', {split: total_loss.data[0]}, iteration)
        self.writer.add_scalars('Reconstruction Loss', {split: loss_reconstr.data[0]}, iteration)
        self.writer.add_scalars('Perceptual Loss', {split: loss_perceptual.data[0]}, iteration)
        self.writer.add_scalars('Smoothness Loss', {split: loss_smooth.data[0]}, iteration)
        self.writer.add_scalars('Warping Loss', {split: loss_warp.data[0]}, iteration)

    def forward_pass(self, aClip, dataset, split):

        img_0, img_t, img_1 = self.get_batch(aClip)

        results = self.superslomo(img_0, img_t, dataset.dims, dataset.scale_factors, t=self.t_interp)

        img_tensor, flow_tensor, flowI_input, flowI_output = results

        target_image = img_t

        train_losses = self.superslomo.stage2_model.compute_loss(img_tensor, flow_tensor,
                                                            flowI_input, flowI_output,
                                                            target_image, self.loss_weights, t=0.5)
        total_loss, individual_losses = train_losses
        self.write_losses(total_loss, individual_losses, iter, split)
        return total_loss, individual_losses

    def train(self):

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.superslomo.parameters()),
                                     lr=self.learning_rate)
        iter = 0

        for epoch in range(self.n_epochs):
            # shuffles the data on each epoch
            adobe_train = adobe_240fps.Reader(self.cfg, split="TRAIN")
            adobe_val = adobe_240fps.Reader(self.cfg, split="VAL")
            val_generator = adobe_val.get_clips()

            print ("Epoch: ", epoch, " Iteration: ", iter)

            for aClip in adobe_train.get_clips():
                iter +=1

                train_loss, _ = self.forward_pass(aClip, adobe_train, "TRAIN")

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                try:
                    valClip = next(val_generator)
                except StopIteration:
                    adobe_val = adobe_240fps.Reader(self.cfg, split="VAL")
                    val_generator = adobe_val.get_clips()
                    valClip = next(val_generator)

                self.forward_pass(valClip, adobe_val, "VAL")

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




if __name__ == '__main__':

    cfg = read_config("./config.ini")

    model = SSM_Main(cfg)

    model.train()


##################################################
# //Set the controls for the heart of the sun!// #
##################################################