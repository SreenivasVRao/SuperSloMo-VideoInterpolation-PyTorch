import PWCNet
import UNetFlow
import SSMLoss
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

log = logging.getLogger(__name__)

class FullModel(nn.Module):

    def __init__(self, cfg, writer):
        super(FullModel, self).__init__()
        self.cfg = cfg
        self.writer = writer
        self.iternum = 0
        self.load_model()
        self.loss = SSMLoss.get_loss(cfg)

    def load_model(self):
        """
        Loads the models, optionally with weights, and optionally freezing individual stages.
        :return:
        """

        stage1_weights = None
        stage2_weights = None
        if self.cfg.getboolean("STAGE1", "LOADPREV"):
            stage1_weights = self.cfg.get("STAGE1", "WEIGHTS")

        if self.cfg.getboolean("STAGE2", "LOADPREV"):
            stage2_weights = self.cfg.get("STAGE2", "WEIGHTS")

        if self.cfg.get("STAGE1", "MODEL")=="PWC":
            self.stage1_model = PWCNet.pwc_dc_net(stage1_weights)  # Flow Computation Model

        elif self.cfg.get("STAGE1", "MODEL")=="UNET":
            self.stage1_model = UNetFlow.get_model(stage1_weights,
                                                   in_channels=6, out_channels=4)
        # Flow Computation Model
        self.stage2_model = UNetFlow.get_model(stage2_weights, in_channels=16,
                                               out_channels=5)  # Flow Interpolation Model

        if self.cfg.getboolean("STAGE1", "FREEZE"):
            log.info("Freezing stage1 model.")
            self.stage1_model.eval()
            for param in self.stage1_model.parameters():
                param.requires_grad = False
        else:
            log.info("Training stage1 model.")

        if self.cfg.getboolean("STAGE2", "FREEZE"):
            log.info("Freezing stage2 model.")
            self.stage2_model.eval()
            for param in self.stage2_model.parameters():
                param.requires_grad = False
        else:
            log.info("Training stage2 model.")

    def stage1_computations(self, img0, img1, dataset_info):
        """
        Refer to PWC-Net repo for more details.
        :param img0, img1: torch tensor BGR (0, 255.0)
        :return: output from flowC model, multiplied by 20
        """

        input_pair_01 = torch.cat([img0, img1], dim=1)
        input_pair_10 = torch.cat([img1, img0], dim=1)
        img_tensor = input_pair_01

        if self.cfg.get("STAGE1","MODEL")=="PWC":

            est_flow_01 = self.stage1_model(input_pair_01)
            est_flow_10 = self.stage1_model(input_pair_10)

            flow_tensor = torch.cat([est_flow_01, est_flow_10], dim=1)

            flow_tensor = self.post_process_flow(flow_tensor, dataset_info)

        elif self.cfg.get("STAGE1", "MODEL")=="UNET":
            flow_tensor = self.stage1_model(input_pair_01)

        return img_tensor, flow_tensor

    def post_process_flow(self, flow_tensor, dataset_info):
        """
        Refer to PWC Net repo for details.
        :param flow_tensor:
        :param dataset_info:
        :return:
        """
        dims, scale_factors = dataset_info
        flow_tensor = flow_tensor * 20.0
        H, W = dims
        upsampled_flow = F.upsample(flow_tensor, size=(H, W), mode='bilinear')

        s_H, s_W = scale_factors
        upsampled_flow[:, 0::2, ...] = upsampled_flow[:, 0::2, ...] * s_W
        # u vectors

        upsampled_flow[:, 1::2, ...] = upsampled_flow[:, 1::2, ...] * s_H
        # v vectors

        return upsampled_flow

    def forward(self, image_0, image_1, dataset_info, t_interp, target_image=None, output_buffer=None, split=None, iteration=None):

        img_tensor, flow_tensor = self.stage1_computations(image_0, image_1, dataset_info)

        flowI_input = self.stage2_model.compute_inputs(img_tensor, flow_tensor, t=t_interp)

        flowI_output = self.stage2_model(flowI_input)
        interpolation_result = self.stage2_model.compute_output_image(img_tensor, flowI_input, flowI_output, t=t_interp)

        if iteration % 100 == 0:
            self.writer.add_image(split, interpolation_result[0, [2,1,0], ...], iteration)
        if output_buffer is not None:
            losses = self.loss(img_tensor, flow_tensor, flowI_input, flowI_output, interpolation_result, target_image)
            output_buffer[0, :] += losses
            output_buffer

            return output_buffer
        else:
            return interpolation_result

def full_model(config, writer):
    model = FullModel(config, writer)
    return model


if __name__ == '__main__':

    import ConfigParser, cv2, os, glob, logging
    from argparse import ArgumentParser
    from tensorboardX import SummaryWriter
    from torch.autograd import Variable
    import torch, numpy as np
    import sys

    sys.path.insert(0, "/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/SuperSloMo/utils/")
    import adobe_240fps

    config = ConfigParser.RawConfigParser()
    parser = ArgumentParser()

    parser.add_argument("-c", "--config", required=True,
                        default="config.ini",
                        help="Path to config.ini file.")
    parser.add_argument("--expt", required=True,
                        help="Experiment Name.")

    parser.add_argument("--log", required=True, help="Path to logfile.")

    args = parser.parse_args()
    logging.basicConfig(filename=args.log, level=logging.INFO)
    config.read(args.config)

    log_dir = os.path.join(config.get("PROJECT","DIR"), "logs")

    os.makedirs(os.path.join(log_dir, args.expt, "plots"))
    os.makedirs(os.path.join(log_dir, args.expt, "images"))

    writer = SummaryWriter(os.path.join(log_dir, args.expt, "plots"))

    ssm_net = full_model(config, writer) # get the SSM Network.
    ssm_net.cuda()
    ssm_net.eval()

    info = adobe_240fps.get_data_info(config, "VAL")
    img_dir = os.path.join(log_dir, args.expt, "images")

    # for idx in range(20):
    #     fprefix = "/clip"+str(idx).zfill(3)+"_"
    #     batch = next(samples).float().cuda()
    #     img_0 = batch[:,  0, ...]
    #     img_1 = batch[:, -1, ...]

    #     img_0_np = img_0.permute(0, 2, 3, 1)[0,...] * 255.0
    #     img_0_np = img_0_np.cpu().data.numpy()
    #     img_1_np = img_1.permute(0, 2, 3, 1)[0,...] * 255.0
    #     img_1_np = img_1_np.cpu().data.numpy()

    #     cv2.imwrite(img_dir+fprefix+str(0).zfill(3)+".png", img_0_np)
    #     cv2.imwrite(img_dir+fprefix+str(8).zfill(3)+".png", img_1_np)

    #     for t_idx in range(1, 8):
    #         t_interp = float(t_idx)/8
    #         interpolation_result  = ssm_net(img_0, img_1, info, t_interp, "VAL", iteration=idx)
    #         est_image_t = interpolation_result * 255.0
    #         est_image_t = est_image_t.permute(0,2, 3, 1)[0, ...]
    #         est_image_t = est_image_t.cpu().data.numpy()

    #         cv2.imwrite(img_dir+fprefix+str(t_idx).zfill(3)+".png", est_image_t)


    def get_image(path, flipFlag):
        img = cv2.imread(path)
        img = img/255.0
        if flipFlag:
            img = img.swapaxes(0, 1)
        # img = cv2.resize(img, (640, 360))
        img = Variable(torch.from_numpy(img))
        img = img.cuda().float()
        img = img[None, ...]
        img = img.permute(0, 3, 1, 2) # bhwc => bchw
        return img


    for clip_idx in [0]:#, 42, 51]:
        fpath = "/mnt/nfs/work1/elm/hzjiang/Data/VideoInterpolation/YouTube240fps/Clips/clip_"+str(clip_idx).zfill(5)+"/"
        images_list = glob.glob(os.path.join(fpath, "*.png"))
        images_list.sort()
        images_list = images_list[::8] # get 30 fps version.


        img_0 = cv2.imread(images_list[0])
        h,w,c = img_0.shape
        vFlag = h>w # horizontal video => h<=w vertical video => w< h
        h_start = np.random.randint(0, 720-704+1)

        #h_start = np.random.randint(0, 1080-1024+1)
        info=(704, 1280),(1.0, 1.0)

        img_dir = os.path.join(log_dir, args.expt, "images", "clip_"+str(clip_idx).zfill(5))
        os.makedirs(img_dir)
        log.info("h w %s %s"%(h,w))
        logging.info("H start: %s"%h_start)
        logging.info("Interpolation beginning.")


        count = 0
        for iteration, im_pair in enumerate(zip(images_list[0:-1], images_list[1:])):
            # if iteration > 20:
            #     break
            impath0, impath1= im_pair
            log.info(iteration)
            img_0 = get_image(impath0, vFlag)[:, :,  h_start:h_start+704, :]
            img_1 = get_image(impath1, vFlag)[:, :,  h_start:h_start+704, :]

            img_0_np = img_0 * 255.0
            img_0_np = img_0_np.permute(0,2, 3, 1)[0, ...]
            img_0_np = img_0_np.cpu().data.numpy()
            cv2.imwrite(img_dir+"/clip_"+str(clip_idx).zfill(5)+"_"+str(count).zfill(3)+".png", img_0_np)
            count+=1

            for idx in range(1, 8):
                t_interp = float(idx)/8
                interpolation_result  = ssm_net(img_0, img_1, info, t_interp, "VAL", iteration=iteration)
                est_image_t = interpolation_result * 255.0
                est_image_t = est_image_t.permute(0,2, 3, 1)[0, ...]
                est_image_t = est_image_t.cpu().data.numpy()
                cv2.imwrite(img_dir+"/clip_"+str(clip_idx).zfill(5)+"_"+str(count).zfill(3)+".png", est_image_t)
                count+=1

        img_1_np = img_1 * 255.0
        img_1_np = img_1_np.permute(0,2, 3, 1)[0, ...]
        img_1_np = img_1_np.cpu().data.numpy()
        cv2.imwrite(img_dir+"/clip_"+str(clip_idx).zfill(5)+"_"+str(count).zfill(3)+".png", img_1_np)
        count+=1

        logging.info("Interpolation complete.")
