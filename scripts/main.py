import configparser
import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.optim
from tensorboardX import SummaryWriter

# from models import superslomo as SSM
from models import superslomo_r as SSMR
from utils.dataset import get_dataset
from utils.validators import validate_forward_pass_inputs


log = logging.getLogger(__name__)


def getargs():
    """ Parse CLI args.

    :returns: the arguments from the parser
    :rtype: ArgumentParser

    """
    parser = ArgumentParser()

    parser.add_argument(
        "-c", "--config", required=True, default="config.ini", help="Path to config.ini file.",
    )
    parser.add_argument("--expt", required=True, help="Experiment Name.")
    parser.add_argument("--log", required=True, help="Path to log file.")
    parser.add_argument("--msg", help="(Optional) Details of experiment stored with TensorBoard.")

    return parser.parse_args()


class Trainer:
    def __init__(self, config, expt_name, message=None):
        """
        Initializes various objects, and creates an instance of the model.
        Creates a callback for tensorboard.
        :param config: Config object.
        """
        self.cfg = config
        self.expt_name = expt_name
        self.msg = message

        log_dir = self.cfg.get("PROJECT", "LOGDIR")

        os.makedirs(os.path.join(log_dir, self.expt_name, "plots"), exist_ok=True)
        os.makedirs(
            os.path.join(self.cfg.get("TRAIN", "CKPT_DIR"), self.expt_name), exist_ok=True,
        )

        self.writer = SummaryWriter(os.path.join(log_dir, self.expt_name, "plots"))
        self.get_hyperparams()
        self.create_model()
        self.configure_trainer()

    def create_model(self):
        """ Creates the SuperSlomo model based on the config and assigns it to the GPU.
        Uses `torch.nn.DataParallel` in-case multiple GPUs are available

        :returns:
        :rtype:

        """
        # self.superslomo = SSM.FullModel(self.cfg, self.writer)
        self.superslomo = SSMR.FullModel(self.cfg, self.writer)

        if torch.cuda.device_count() > 1:
            log.info("Found %s GPUS. Using DataParallel.", torch.cuda.device_count())
            self.superslomo = torch.nn.DataParallel(self.superslomo)
        else:
            log.warning("GPUs found: %s", str(torch.cuda.device_count()))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s", device)
        self.superslomo.to(device)

    def get_hyperparams(self):
        """ Loads hyperparameters for training from the config file.

        :returns: None
        :rtype:

        """

        self.n_epochs = self.cfg.getint("TRAIN", "N_EPOCHS")
        self.learning_rate = self.cfg.getfloat("TRAIN", "LEARNING_RATE")
        self.lr_decay = self.cfg.getfloat("TRAIN", "LR_DECAY")
        self.lr_period = self.cfg.getfloat("TRAIN", "LR_PERIOD")
        self.save_every = self.cfg.getint("TRAIN", "SAVE_EVERY")

    def write_losses(self, losses, iteration, split):
        """ Writes the losses to tensorboard using `TensorboardX::SummaryWriter`

        :param losses: tuple of `(Total Loss, Reconstruction Loss, Warping loss, Perceptual Loss)`
        :param iteration: integer
        :param split: `TRAIN` or `VAL`
        :returns:
        :rtype:

        """

        total_loss, loss_reconstr, loss_warp, loss_perceptual = losses

        self.writer.add_scalars("Total_Loss", {split: total_loss.item()}, iteration)
        self.writer.add_scalars("Reconstruction_Loss", {split: loss_reconstr.item()}, iteration)
        self.writer.add_scalars("Perceptual_Loss", {split: loss_perceptual.item()}, iteration)
        self.writer.add_scalars("Warping_Loss", {split: loss_warp.item()}, iteration)

    @validate_forward_pass_inputs
    def forward_pass(self, input_images, target_images, split, iteration, t_interp):
        """
        Runs a single forward pass and returns the total loss
        from the iteration. Writes the losses to tensorboard as well.

        :param input_images: B T H W C 0-255, np.uint8
        :param target_images: B T-1 H W C 0-255, np.uint8
        :param dataset_info: dataset object with corresponding split.
        :param split: "TRAIN"/"TEST"/"VAL"
        :param get_interpolation: flag to return interpolation result
        :return: returns the total loss.
        """

        est_img_t, losses = self.superslomo(
            input_images,
            t_interp,
            target_images=target_images,
            iteration=iteration,
            inference_mode=False,
        )

        losses = losses.mean(dim=0)
        # averages the loss over the batch. Horrific code. [B, 4] -> [4]
        self.write_losses(losses, iteration, split)

        if iteration % 100 == 0:
            self.write_image(est_img_t, split, iteration)
        total_loss = losses[0]
        return total_loss

    def write_image(self, output_images, split, iteration):
        """ Helper function to write the image to tensorboard after
        re-centering by mean and standard deviation.

        :param output_images: B C H W tensor with float values.
        :param split: "TRAIN"/"VAL"/"TEST"
        :param iteration: int
        :returns:
        :rtype:

        """
        pred_img = output_images[0, ...]
        pix_mean = self.cfg.get("MODEL", "PIXEL_MEAN").split(",")
        pix_mean = [float(p) for p in pix_mean]
        pix_std = self.cfg.get("MODEL", "PIXEL_STD").split(",")
        pix_std = [float(p) for p in pix_std]
        pix_std = torch.tensor(pix_std)[None, :, None, None].cuda()
        pix_mean = torch.tensor(pix_mean)[None, :, None, None].cuda()
        pred_img = pred_img[None, ...] * pix_std + pix_mean
        self.writer.add_image(split, pred_img[0, ...], iteration)

    def train(self):
        """
        Training schedule for the SuperSloMo architecture.

        :return:
        """
        iteration = 0
        train_samples = get_dataset(self.cfg, split="TRAIN")

        for epoch in range(self.start, self.n_epochs + 1):
            # shuffles the data on each epoch
            self.writer.add_scalars("Learning_Rate", {"TRAIN": self.get_learning_rate()}, iteration)
            for input_tensor, target_tensor, t_interp in train_samples:
                iteration += 1
                if iteration % 100 == 0:
                    log.info("Iterations: %s", iteration)

                if input_tensor.shape[0] < torch.cuda.device_count():
                    continue

                t_interp = t_interp.cuda().float()
                input_tensor = input_tensor.cuda().float()
                target_tensor = target_tensor.cuda(non_blocking=True).float()

                train_loss = self.forward_pass(
                    input_tensor, target_tensor, "TRAIN", iteration, t_interp,
                )
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

            self.lr_scheduler.step()

            log.info("Epoch: %s, Iteration: %s ", epoch, iteration)
            if epoch % self.save_every == 0:
                self.save_model(epoch)

        self.writer.close()

    def get_learning_rate(self):
        """ Gets the learning rate from the optimizer
        Assuming we only have one param group.

        :returns: learning rate
        :rtype: float value
        """
        assert len(self.optimizer.param_groups) == 1
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def save_model(self, epoch):
        """ Helper function to save the model checkpoint.

        :param epoch: int
        :returns:
        :rtype:

        """

        if isinstance(self.superslomo, torch.nn.DataParallel):
            model = self.superslomo.module
        else:
            model = self.superslomo
        state = {
            "epoch": epoch,
            "stage1_state_dict": model.stage1_model.state_dict(),
            "stage2_state_dict": model.stage2_model.state_dict(),
            "self.optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
        }

        fpath = os.path.join(
            self.cfg.get("TRAIN", "CKPT_DIR"),
            self.expt_name,
            self.expt_name + "_EPOCH_" + str(epoch).zfill(4) + ".pt",
        )

        torch.save(state, fpath)

    def configure_trainer(self):
        """ Configure the optimizer and the learning rate scheduler.
        Loads weights if configured, and freezes them too if necessary.

        :returns:
        :rtype:

        """
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.superslomo.parameters()), lr=self.learning_rate,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.lr_period, gamma=self.lr_decay
        )
        self.start = 1

        load_1 = self.cfg.getboolean("STAGE1", "LOADPREV")
        load_2 = self.cfg.getboolean("STAGE2", "LOADPREV")
        freeze_1 = self.cfg.getboolean("STAGE1", "FREEZE")
        freeze_2 = self.cfg.getboolean("STAGE2", "FREEZE")
        if load_1 and not freeze_1:
            ckpt_path = self.cfg.get("STAGE1", "WEIGHTS")
        elif load_2 and not freeze_2:
            ckpt_path = self.cfg.get("STAGE2", "WEIGHTS")
        else:
            return

        # configure optimizer if stage1 or stage2 are resuming

        checkpoint = torch.load(ckpt_path)

        self.optimizer.load_state_dict(checkpoint["self.optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        self.start = max(checkpoint["epoch"], 1)
        log.info("Starting training from: %s", self.start)
        log.info("Scheduler: %s", self.lr_scheduler.get_last_lr())
        for param_group in self.optimizer.param_groups:
            log.info("Learning rate: %s", param_group["lr"])


if __name__ == "__main__":

    ARGS = getargs()

    logging.basicConfig(filename=ARGS.log, level=logging.INFO)
    cfg = configparser.RawConfigParser()
    cfg.read(ARGS.config)

    torch.manual_seed(cfg.getint("SEED", "VALUE"))
    torch.backends.cudnn.benchmark = True
    np.random.seed(cfg.getint("SEED", "VALUE"))
    random.seed(cfg.getint("SEED", "VALUE"))

    log.info("SEED: %s", torch.initial_seed())

    my_network = Trainer(cfg, ARGS.expt, ARGS.msg)
    my_network.train()

    log.info("Training complete.")
