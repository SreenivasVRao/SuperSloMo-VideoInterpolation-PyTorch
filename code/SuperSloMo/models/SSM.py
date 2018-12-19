import PWCNet
import UNetFlow
import SSMLoss
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

log = logging.getLogger(__name__)


class FullModel(nn.Module):

    def __init__(self, cfg, writer=None):
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

        self.cross_skip = self.cfg.getboolean("STAGE2", "CROSS_SKIP")
            
        if self.cfg.get("STAGE1", "MODEL")=="PWC":
            log.info("STAGE1 PWC")
            self.stage1_model = PWCNet.pwc_dc_net(stage1_weights)  # Flow Computation Model

        elif self.cfg.get("STAGE1", "MODEL") in ["UNET", "UNETC", "UNETA"]:
            log.info("STAGE 1 %s"%self.cfg.get("STAGE1", "MODEL"))
            
            self.stage1_model = UNetFlow.get_model(stage1_weights, in_channels=6, out_channels=4,
                                                   cross_skip=self.cross_skip, stage=1)
            
        # Flow Computation Model
        log.info("STAGE 2 %s"%self.cfg.get("STAGE2", "MODEL"))
        self.stage2_model = UNetFlow.get_model(stage2_weights, in_channels=16, out_channels=5,
                                               cross_skip=self.cross_skip, stage=2)
        # Flow Interpolation Model

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

    def forward(self, image_tensor, dataset_info, t_interp, target_images=None,
                split=None, iteration=None, compute_loss=False):
        img0 = image_tensor[:, 0, ...]
        img1 = image_tensor[:, 1, ...]
        img2 = image_tensor[:, 2, ...]
        img3 = image_tensor[:, 3, ...]

        img_01 = torch.cat([img0, img1], dim=1)
        img_12 = torch.cat([img1, img2], dim=1)
        img_23 = torch.cat([img2, img3], dim=1)

        flowC_outputs = self.stage1_model(img_01, img_12, img_23)

        combined_encoding = []
        stage2_inputs = []

        for idx, img_pair in enumerate([img_01, img_12, img_23]):
            stage1_encoding, flow_tensor = flowC_outputs[idx]
            input_tensor = self.stage2_model.compute_inputs(img_pair, flow_tensor, t=t_interp)
            stage2_inputs.append(input_tensor)
            combined_encoding.append(stage1_encoding)

        x01, x12, x23 = stage2_inputs

        flowI_outputs = self.stage2_model(x01, x12, x23, combined_encoding)

        b = image_tensor.shape[0]

        losses = torch.zeros([b, 4]).cuda()

        img_t = []

        for idx, img_pair in enumerate([img_01, img_12, img_23]):
            _, flowI_out = flowI_outputs[idx]
            flowI_in = stage2_inputs[idx]
            interpolation_result = self.stage2_model.compute_output_image(img_pair, flowI_in,
                                                                          flowI_out, t=t_interp)

            if compute_loss and target_images is not None:
                flow_tensor = flowC_outputs[idx][1]
                target = target_images[:, idx, ...]
                losses += self.loss(img_pair, flow_tensor, flowI_in, flowI_out, interpolation_result, target)
            else:
                img_t.append(interpolation_result)

        # if iteration % 100 == 0 and self.writer is not None:
        #     self.writer.add_image(split, interpolation_result[0, [2, 1, 0], ...], iteration)

        if compute_loss:
            return losses

        return img_t[1] # middle result.

        
def full_model(config, writer=None):
    """
    Returns the SuperSloMo model with config and writer as specified.
    """
    model = FullModel(config, writer)
    return model
