from SuperSloMo.models import SSM, SSMLoss
from SuperSloMo.utils import adobe_240fps, metrics_v2
import numpy as np,  random
import torch.optim
import torch

from tensorboardX import SummaryWriter
import os, logging, ConfigParser

log = logging.getLogger(__name__)


def read_config(configpath='config.ini'):
    config = ConfigParser.RawConfigParser()
    config.read(configpath)
    return config


class SSMNet:

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
        os.makedirs(os.path.join(log_dir, self.expt_name, "checkpoints"))

        self.writer = SummaryWriter(os.path.join(log_dir, self.expt_name, "plots"))
        self.get_hyperparams()

        self.superslomo = SSM.full_model(self.cfg, self.writer)
        
        if torch.cuda.device_count()>1:
            log.info("Found %s GPUS. Using DataParallel."%torch.cuda.device_count())
            self.superslomo = torch.nn.DataParallel(self.superslomo)
        else:
            log.warning("GPUs found: "+str(torch.cuda.device_count()))
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s"%device)
        self.superslomo.to(device)
        self.loss_module = SSMLoss.get_loss(config).cuda()

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

    def write_losses(self, losses, iteration, split):
        """
        Writes the losses to tensorboard for given iteration and split.

        :param total_loss: Weighted sum of all losses.
        :param individual_losses: Tuple of 4 losses.
        :param iteration: Current iteration.
        :param split: Train/Val
        :return:
        """
        total_loss, loss_reconstr,  loss_warp,loss_perceptual = losses

        self.writer.add_scalars('Total_Loss', {split: total_loss.item()}, iteration)
        self.writer.add_scalars('Reconstruction_Loss', {split: loss_reconstr.item()}, iteration)
        self.writer.add_scalars('Perceptual_Loss', {split: loss_perceptual.item()}, iteration)
        self.writer.add_scalars('Warping_Loss', {split: loss_warp.item()}, iteration)
        # self.writer.add_scalars('Smoothness_Loss', {split: loss_smooth.item()}, iteration)

    def forward_pass(self, data_batch, dataset_info, split, iteration, t_idx, get_interpolation=False):
        """
        :param data_batch: B H W C 0-255, np.uint8
        :param dataset_info: dataset object with corresponding split.
        :param split: "TRAIN"/"TEST"/"VAL"
        :param get_interpolation: flag to return interpolation result
        :return: if get_interpolation is set, returns interpolation result as BCHW Variable.
        otherwise returns the losses.
        """
        data_batch = data_batch.cuda().float()
        assert 1<=t_idx<=7, "Invalid time-step: "+str(t_idx)
        image_tensor = data_batch[:, 0::2, ...] # [0, 2, 4, 6] indices = I0, I1, I2, I3
        img_t = data_batch[:, 1::2, ...] # [1, 3, 5] indices = I4, I12, I20.
        t_interp = float(t_idx)/8

        if get_interpolation:
            if iteration==1:
                log.info("Getting only Interpolation Result.")
            interpolation_result = self.superslomo(img_0, img_1, dataset_info, t_interp, split=split, iteration=iteration)
            return interpolation_result
        
        elif not get_interpolation:
            losses = self.superslomo(image_tensor, dataset_info, t_interp, split=split, iteration=iteration, target_images=img_t, compute_loss=True)
            losses = losses.mean(dim=0) # averages the loss over the batch. Horrific code. [B, 4] -> [4]
            self.write_losses(losses, iteration, split)
            total_loss = losses[0]
            return total_loss

    def train(self):
        """
        Training schedule for the SuperSloMo architecture.

        :return:
        """
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.superslomo.parameters()),
                                     lr=self.learning_rate)        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_period, gamma=self.lr_decay)
            
        iteration = 0

        train_info = adobe_240fps.get_data_info(self.cfg, split="TRAIN")
        # val_info = adobe_240fps.get_data_info(self.cfg, split="VAL")

        for epoch in range(1, self.n_epochs+1):
            # shuffles the data on each epoch
            adobe_train_samples = adobe_240fps.data_generator(self.cfg, split="TRAIN")
            # adobe_val_samples = adobe_240fps.data_generator(self.cfg, split="VAL")
            lr_scheduler.step()

            for train_batch in adobe_train_samples:
                iteration +=1
                data_batch, t_idx = train_batch
                if data_batch.shape[0]<torch.cuda.device_count():
                    continue
                
                train_loss = self.forward_pass(data_batch, train_info, "TRAIN", iteration, t_idx)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                # try:
                #     val_batch = next(adobe_val_samples)
                # except StopIteration:
                #     adobe_val_samples = adobe_240fps.data_generator(self.cfg, split="VAL")
                #     val_batch = next(adobe_val_samples)

                # data_batch, t_idx = val_batch
                # if data_batch.shape[0]<torch.cuda.device_count():
                #     continue
                
                # self.forward_pass(data_batch, val_info, "VAL", iteration, t_idx)
                    
            log.info("Epoch: "+str(epoch)+" Iteration: "+str(iteration))

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
                    'scheduler': lr_scheduler.state_dict()
                }

                fpath = os.path.join(self.cfg.get("PROJECT", "DIR"), "logs", self.expt_name, "checkpoints",
                                     self.expt_name+"_EPOCH_"+str(epoch).zfill(4)+".pt")

                torch.save(state, fpath)

        self.writer.close()

    def compute_metrics(self, dataset, info, split):
        """
        Computes PSNR, Interpolation Error, and SSIM scores for the given split of the dataset.
        :param dataset:
        :return: avg PSNR, avg IE, avg SSIM
        """
        total_ssim = 0
        total_IE = 0
        total_PSNR = 0

        nframes = 0

        for iteration, a_batch in enumerate(dataset):
            data_batch, _ = a_batch
            data_batch = data_batch.cuda().float()
            if iteration==1:
                log.info(data_batch.shape)
            for t_idx in range(1, 8):
                est_image_t = self.forward_pass(data_batch, info, split, iteration, t_idx, get_interpolation=True)
                gt_image_t = data_batch[:, t_idx, ...]
                psnr_scores, IE_scores, ssim_scores = metrics_v2.get_scores(est_image_t, gt_image_t)
                total_IE   += np.sum(IE_scores)
                total_ssim += np.sum(ssim_scores)
                total_PSNR += np.sum(psnr_scores)
            n_interpolations = data_batch.shape[1]-2 # exclude i_0, i_1
            nframes += data_batch.shape[0]*n_interpolations  # interpolates nframes Batch size - 2 frames (i0, i1)
        log.info(data_batch.shape)

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
    torch.manual_seed(cfg.getint("SEED","VALUE"))
    np.random.seed(cfg.getint("SEED","VALUE"))
    random.seed(cfg.getint("SEED", "VALUE"))

    log.info("SEED: %s"%torch.initial_seed())

    ssm_net = SSMNet(cfg, args.expt, args.msg)

    ssm_net.train()

    log.info("Training complete.")
    
    # log.info("Evaluating metrics.")

    # ssm_net.superslomo.eval()

    # adobe_train = adobe_240fps.data_generator(cfg, split="TRAIN", eval=True)
    # train_info = adobe_240fps.get_data_info(cfg, split="TRAIN")
        
    # PSNR, IE, SSIM = ssm_net.compute_metrics(adobe_train, train_info, "TRAIN")
    # logging.info("ADOBE TRAIN: Average PSNR %.3f IE %.3f SSIM %.3f"%(PSNR, IE, SSIM))

    # adobe_val = adobe_240fps.data_generator(cfg, split="VAL", eval=True)
    # val_info = adobe_240fps.get_data_info(cfg, split="VAL")
    
    # PSNR, IE, SSIM = ssm_net.compute_metrics(adobe_val, val_info, "VAL")
    # logging.info("ADOBE VAL: Average PSNR %.3f IE %.3f SSIM %.3f"%(PSNR, IE, SSIM))
    

##################################################
# //Set the controls for the heart of the sun!// #
##################################################
