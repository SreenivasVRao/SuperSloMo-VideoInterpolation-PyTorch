from SuperSloMo.models import *
from SuperSloMo.utils import adobe_240fps, flo_utils

import torch.optim
import torch
from torch.autograd import Variable

import os
import ConfigParser
import cv2, numpy as np
from math import ceil
import sys
from scipy.ndimage import imread


def read_config(configpath='config.ini'):
    config = ConfigParser.RawConfigParser()
    config.read(configpath)
    return config


class FullModel:

    def __init__(self, config, freeze_stages=(False, False), use_gpu=True):
        self.cfg = config

        top_dir = self.cfg.get("PROJECT", "DIR")

        flowC_weights_path = self.cfg.get("STAGE1", "WEIGHTS")
        flowC_weights_path = os.path.join(top_dir, flowC_weights_path)

        self.flowC_model = PWCNet.pwc_dc_net(flowC_weights_path) # Flow Computation Model
        self.flowI_model = FlowInterpolator.flow_interpolator() # Flow Interpolation Model
        self.use_gpu = use_gpu

        if use_gpu:
            self.flowC_model.cuda()
            self.flowI_model.cuda()
        self.freeze_model(freeze_stages)

    def freeze_model(self, freeze_stages):
        if freeze_stages[0]:
            self.flowC_model.training = False
            for param in self.flowC_model.parameters():
                param.requires_grad = False

        if freeze_stages[1]:
            for param in self.flowI_model.parameters():
                param.requires_grad = False

    def stage1_computations(self, img0, img1):
        """
        Refer to PWC-Net repo for more details.
        :param img0, img1: torch tensor BGR float32 (0, 1.0)
        :return: output from flowC model, multiplied by 20
        """
        input_pair_01 = torch.cat([img0, img1], dim=1)
        input_pair_10 = torch.cat([img1, img0], dim=1)

        est_flow_01 = self.flowC_model(input_pair_01)
        est_flow_10 = self.flowC_model(input_pair_10)

        img_tensor =  input_pair_01
        _, _, H, W = img_tensor.shape

        # postprocess the flow back
        new_flow_01 = self.postprocess_flow(est_flow_01, H, W)
        new_flow_10 = self.postprocess_flow(est_flow_10, H, W)

        flow_tensor = torch.cat([new_flow_01, new_flow_10], dim=1)

        return img_tensor, flow_tensor

    def postprocess_flow(self, flow, H, W):
        n, c, _, _ = flow.shape
        new_flow = np.zeros([n, H, W, c])

        flow = flow * 20.0 # refer to PWC Net repo

        flow = flow.cpu().data.numpy()

        for idx in range(flow.shape[0]):
            flo = flow[idx,...]

            flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2)

            u_ = cv2.resize(flo[:, :, 0], (W, H))
            v_ = cv2.resize(flo[:, :, 1], (W, H))
            # u_ *= W / float(W_)
            # v_ *= H / float(H_)

            flo = np.dstack((u_, v_))
            new_flow[idx,...] = flo

        new_flow = Variable(torch.from_numpy(new_flow))
        new_flow = new_flow.permute(0, 3, 1, 2) # B C H W format
        if self.use_gpu:
            new_flow= new_flow.cuda()

        return new_flow

    def train(self):

        lambda_r = self.cfg.getfloat("TRAIN", "LAMBDA_R") # reconstruction loss weighting
        lambda_w = self.cfg.getfloat("TRAIN", "LAMBDA_W") # warp loss weighting
        lambda_s = self.cfg.getfloat("TRAIN", "LAMBDA_S") # smoothness loss weighting
        lambda_p = self.cfg.getfloat("TRAIN", "LAMBDA_P") # perceptual loss weighting

        n_epochs = self.cfg.getint("TRAIN", "N_EPOCHS")
        learning_rate = self.cfg.getfloat("TRAIN", "LEARNING_RATE")
        lr_decay = self.cfg.getfloat("TRAIN", "LR_DECAY")
        lr_period = self.cfg.getfloat("TRAIN", "LR_PERIOD")

        loss_weights = lambda_r, lambda_p, lambda_w, lambda_s

        # optimizer = torch.optim.Adam(self.flowI_model.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            adobe_dataset = adobe_240fps.DataReader(self.cfg)
            # TODO: shuffle the list in the generator

            for aClip in adobe_dataset.get_clips():
                oneBatch = Variable(torch.from_numpy(aClip)).cuda().float()

                oneBatch = oneBatch.permute(0, 3, 1, 2)
                oneBatch = 1.0 * oneBatch / 255.0

                image_0 = oneBatch[ ::3, ...].float()
                image_t = oneBatch[1::3, ...].float()
                image_1 = oneBatch[2::3, ...].float()

                img_tensor, flow_tensor = self.stage1_computations(image_0, image_1)
                return img_tensor, flow_tensor
                # print(img_tensor.shape, flow_tensor.shape)
                # print(img_tensor.requires_grad, flow_tensor.requires_grad)


                # flowI_input = self.flowI_model.compute_inputs(img_tensor, flow_tensor, t=0.5)
                # flowI_output = self.flowI_model(flowI_input)
                #
                # interpolation_result = self.flowI_model.compute_output_image(flowI_input, flowI_output, t=0.5)
                #
                # target_image = Variable(torch.from_numpy(image_t))
                #
                # total_loss = self.flowI_model.compute_loss(img_tensor, flow_tensor, flowI_input,
                #                                            flowI_output, target_image, loss_weights, t=0.5)
                #
                # optimizer.zero_grad()
                # total_loss.backward()
                # optimizer.step()

            if epoch%lr_period==0 and epoch>0:
                learning_rate = learning_rate*lr_decay


if __name__ == '__main__':


    cfg = read_config("./config.ini")

    model = FullModel(cfg, freeze_stages = (True, True))
    img_tensor, flow_tensor = model.train()
    # img_tensor, flow_tensor = model.stage1_computations(img0=img_0, img1=img_1)

    images = img_tensor.cpu().data.numpy()
    # print(np.linalg.norm(images[0,0:3,...] - im_all[0].numpy()))
    # print(np.linalg.norm(images[0,3:,...] - im_all[1].numpy()))

    flow_tensor = flow_tensor.permute(0, 2, 3, 1)
    flows = flow_tensor.cpu().data.numpy()

    flow_01 = flows[0, ..., 0:2].astype(np.float32)
    flow_10 = flows[0, ..., 2:].astype(np.float32)

    flo_utils.write("/media/sreenivas/Data/UMASS/Thesis/code/SuperSloMo/tmp/adobe_flow01.flo", flow_01)
    flo_utils.write("/media/sreenivas/Data/UMASS/Thesis/code/SuperSloMo/tmp/adobe_flow10.flo", flow_10)

    img_tensor = img_tensor.permute(0, 2, 3, 1)*255.0
    images = img_tensor.cpu().data.numpy()[0,...]

    cv2.imwrite("Adobe_image0.png", images[...,0:3])
    cv2.imwrite("Adobe_image1.png", images[...,3:])

    print(img_tensor.shape, flow_tensor.shape)


