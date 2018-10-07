import PWCNet
import UNetFlow
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

log = logging.getLogger(__name__)


class FullModel(nn.Module):

    def __init__(self, cfg):
        super(FullModel, self).__init__()
        self.cfg = cfg
        self.load_model()

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
                                               out_channels=11)  # Flow Interpolation Model

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
        img0 = (1.0 * img0) / 255.0
        img1 = (1.0 * img1) / 255.0

        input_pair_01 = torch.cat([img0, img1], dim=1)
        input_pair_10 = torch.cat([img1, img0], dim=1)

        est_flow_01 = self.stage1_model(input_pair_01)
        est_flow_10 = self.stage1_model(input_pair_10)

        img_tensor = input_pair_01
        flow_tensor = torch.cat([est_flow_01, est_flow_10], dim=1)

        if self.cfg.get("STAGE1","MODEL")=="PWC":
            flow_tensor = self.post_process_flow(flow_tensor, dataset_info)
        else:
            pass

        return img_tensor, flow_tensor

    def post_process_flow(self, flow_tensor, dataset_info):
        dims, scale_factors = dataset_info
        flow_tensor = flow_tensor * 20.0
        H, W = dims
        upsampled_flow = F.upsample(flow_tensor, size=(H, W), mode='bilinear')

        s_H, s_W = scale_factors
        upsampled_flow[:, 0::2, ...] = upsampled_flow[:, 0::2, ...] * s_W
        # u vectors

        upsampled_flow[:, 1::3, ...] = upsampled_flow[:, 1::3, ...] * s_H
        # v vectors

        return upsampled_flow

    def forward(self, image_0, image_1, dataset_info, t_interp):

        img_tensor, flow_tensor = self.stage1_computations(image_0, image_1, dataset_info)
        flowI_input = self.stage2_model.compute_inputs(img_tensor, flow_tensor, t=t_interp)
        flowI_output, interpolation_result = self.stage2_model(img_tensor, flow_tensor)

        return img_tensor, flow_tensor, flowI_input, flowI_output, interpolation_result


def full_model(config):
    model = FullModel(config)
    return model


if __name__ == '__main__':

    from code.SuperSloMo.utils import adobe_240fps
    from torch.autograd import Variable
    import ConfigParser, os
    config = ConfigParser.RawConfigParser()
    config.read("../../config.ini")

    stage1_weights = os.path.join(config.get("PROJECT", "DIR"), config.get("STAGE1", "WEIGHTS"))
    model = full_model(config)
    print(type(model))
    print isinstance(model, nn.DataParallel)
    print isinstance(model, nn.Module)

    model = nn.DataParallel(model)
    print(type(model))
    print isinstance(model, nn.Module)
    print isinstance(model, nn.DataParallel)
    exit(0)

    adobe_dataset = adobe_240fps.Reader(config, split="TRAIN")

    aClip = next(adobe_dataset.get_clips())

    oneBatch = Variable(torch.from_numpy(aClip)).cuda().float()

    oneBatch = oneBatch.permute(0, 3, 1, 2)
    image_0 = oneBatch[::3, ...].float()
    image_t = oneBatch[1::3, ...].float()
    image_1 = oneBatch[2::3, ...].float()

    stage1_weights = os.path.join(config.get("PROJECT", "DIR"), config.get("STAGE1", "WEIGHTS"))
    superslomo = FullModel(stage1_weights)
    superslomo.cuda()
    superslomo.train(False)
    interpolated_image = superslomo(image_0, image_1, adobe_dataset, config.getfloat("TRAIN", "T_INTERP"))

    log.info(str(interpolated_image.shape))
