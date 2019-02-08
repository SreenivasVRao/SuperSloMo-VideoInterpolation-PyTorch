"""
SuperSloMo-R (R= Recurrent model)
"""
import sys
sys.path.insert(0, "/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/SuperSloMo/models/")
from SSMLoss import get_loss as get_ssm_loss
import torch
import torch.nn as nn
import logging
import ResNetFlow2 as resnet
import UNetFlow as unet

log = logging.getLogger(__name__)


class FullModel(nn.Module):

    def __init__(self, cfg, writer=None):
        super(FullModel, self).__init__()
        self.cfg = cfg
        self.writer = writer
        self.load_model()
        self.loss = get_ssm_loss(cfg)

    def build_stage(self, stage, in_channels, out_channels):
        stage_id= "STAGE%s"%stage
        weights = None
        if self.cfg.getboolean(stage_id, "LOADPREV"):
            weights = self.cfg.get(stage_id, "WEIGHTS")

        self.cross_skip = self.cfg.getboolean("STAGE2", "CROSS_SKIP")

        if self.cfg.get(stage_id, "ENCODER") == "unet":
            model = unet.get_model(weights, in_channels, out_channels, self.cross_skip,
                                   verbose = False, stage = stage, cfg = self.cfg)

        elif self.cfg.get(stage_id, "ENCODER") in ["resnet18", "resnet34"]:
            model = resnet.get_model(weights, in_channels, out_channels, self.cross_skip,
                                     verbose = False, stage = stage, cfg = self.cfg)
        else:
            raise Exception("Unsupported encoder: %s"%self.cfg.get(stage_id, "ENCODER"))
        return model

    def load_model(self):
        """
        Loads the models, optionally with weights, and optionally freezing individual stages.
        :return:
        """
        self.stage1_model = self.build_stage(stage=1, in_channels=6, out_channels=4)
        self.stage2_model = self.build_stage(stage=2, in_channels=16, out_channels=5)

        if self.cross_skip:
            log.info("Cross stage Skip Connections: ENABLED")
        else:
            log.info("Cross stage Skip Connections: DISABLED")

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

    def stage1_computations(self, img_tensor, dataset_info):
        """
        Refer to PWC-Net repo for more details.
        :param img0, img1: torch tensor BGR (0, 255.0)
        :return: output from flowC model, multiplied by 20
        """

        if self.cfg.get("STAGE1", "MODEL") == "PWC":
            raise NotImplementedError("PWC Net not implemented.")
            # input_pair_01 = torch.cat([img0, img1], dim=1)
            # input_pair_10 = torch.cat([img1, img0], dim=1)
            # img_tensor = input_pair_01
            #
            # est_flow_01 = self.stage1_model(input_pair_01)
            # est_flow_10 = self.stage1_model(input_pair_10)
            #
            # flow_tensor = torch.cat([est_flow_01, est_flow_10], dim=1)
            #
            # flow_tensor = self.post_process_flow(flow_tensor, dataset_info)

        elif self.cfg.get("STAGE1", "MODEL") in ["UNET", "UNETA", "UNETC"]:
            flow_tensor = self.stage1_model(img_tensor)

        return flow_tensor

    # def post_process_flow(self, flow_tensor, dataset_info):
    #     """
    #     Refer to PWC Net repo for details.
    #     :param flow_tensor:
    #     :param dataset_info:
    #     :return:
    #     """
    #     dims, scale_factors = dataset_info
    #     flow_tensor = flow_tensor * 20.0
    #     H, W = dims
    #     upsampled_flow = F.upsample(flow_tensor, size=(H, W), mode='bilinear')
    #
    #     s_H, s_W = scale_factors
    #     upsampled_flow[:, 0::2, ...] = upsampled_flow[:, 0::2, ...] * s_W
    #     # u vectors
    #
    #     upsampled_flow[:, 1::2, ...] = upsampled_flow[:, 1::2, ...] * s_H
    #     # v vectors
    #
    #     return upsampled_flow

    def get_image_pairs(self, img_tensor):
        """
        :param img_tensor: B T 3 H W tensor of images.
        :return: B (T-1) 6 H W tensor -> every 2 adjacent images are paired.
        """

        images = list(img_tensor.split(dim=1, split_size=1))
        image_pairs = list(zip(images[:-1], images[1:]))
        image_pairs = [torch.cat([imgA, imgB], dim=2).squeeze() for imgA, imgB in image_pairs]
        if len(image_pairs[0].shape) < 4:  # bad code.
            image_pairs = [img_pair[None, ...] for img_pair in image_pairs]
        image_pairs = torch.stack(image_pairs, dim=1)
        return image_pairs

    def forward(self, image_tensor, dataset_info, t_interp, target_images=None,
                split=None, iteration=None, compute_loss=False):

        image_pairs = self.get_image_pairs(image_tensor)
        T = image_pairs.shape[1]
        mid_idx = T // 2
        if iteration == 1:
            log.info("%s interpolation windows. Mid_idx: %s" % (T, mid_idx))

        flowC_outputs = self.stage1_model(image_pairs)

        combined_encoding = []
        stage2_inputs = []

        for time_step in range(T):
            img_pair = image_pairs[:, time_step, ...]
            stage1_encoding, flow_tensor = flowC_outputs[time_step]
            input_tensor = self.stage2_model.compute_inputs(img_pair, flow_tensor, t=t_interp)
            stage2_inputs.append(input_tensor)
            combined_encoding.append(stage1_encoding)

        stage2_inputs = torch.stack(stage2_inputs, dim=1)
        flowI_outputs = self.stage2_model(stage2_inputs, combined_encoding)

        b = image_pairs.shape[0]
        losses = torch.zeros([b, 4]).cuda()

        est_img_t = None

        for t in range(T):
            flowI_out = flowI_outputs[t]
            flowI_in = stage2_inputs[:, t, ...]
            img_pair = image_pairs[:, t, ...]

            interpolation_result = self.stage2_model.compute_output_image(img_pair, flowI_in,
                                                                          flowI_out, t=t_interp)
            flow_tensor = flowC_outputs[t][1]
            if compute_loss:
                assert target_images is not None, "No target found for loss."
                assert target_images.shape[1]==T, "Insufficient number of targets."
                current_target = target_images[:, t, ...]
                losses = losses + self.loss(img_pair, flow_tensor, flowI_in,
                                            flowI_out, interpolation_result, current_target)

            if t==mid_idx:
                est_img_t = interpolation_result

        if compute_loss:
            # losses = [B, 4] tensor-> total_loss, loss_r, loss_w, loss_p
            return est_img_t, losses
        else:
            if iteration==1:
                log.info("Returning middle result: %s"%mid_idx)
            return est_img_t # middle result.


def full_model(config, writer=None):
    """
    Returns the SuperSloMo model with config and writer as specified.
    """
    model = FullModel(config, writer)
    return model


if __name__ == "__main__":
    import configparser

    def read_config(configpath='config.ini'):
        config = configparser.RawConfigParser()
        config.read(configpath)
        return config

    cfg = read_config("/media/sreenivas/Data/UMASS/gypsum/code/configs/ssmr.ini")
    logging.basicConfig(filename="test.log", level=logging.INFO)

    ssm_net = full_model(cfg).cuda()
    N = 4
    test_in = torch.rand([1, N, 3, 64, 64]).cuda()
    target = torch.rand([1, N-1, 3, 64, 64]).cuda()
    _, loss_value = ssm_net(test_in, None, 0.5, target, split="TRAIN", iteration=1, compute_loss=True)

    x = loss_value.mean()
    x.backward()
