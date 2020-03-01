"""
SuperSloMo
"""

import logging
from ..utils.validators import validate_target_tensor

import torch
import torch.nn as nn

from . import unetflow as unet
from .losses import SSMLosses

log = logging.getLogger(__name__)


class FullModel(nn.Module):
    def __init__(self, cfg, writer=None):
        """ Defines the superslomo model as per Jiang et. al. (2017)

        :param cfg: Config parsed from `configs/ssm.ini`
        :param writer: TensorboardX object
        """

        super(FullModel, self).__init__()
        self.cfg = cfg
        self.writer = writer
        self.load_weights()
        self.freeze_weights()  # only if cfg specifies it
        self.loss = SSMLosses(cfg)

    def load_weights(self):
        """
        Loads the weights for each stage.
        """
        stage1_weights = (
            self.cfg.get("STAGE1", "WEIGHTS")
            if self.cfg.getboolean("STAGE1", "LOADPREV")
            else None
        )

        stage2_weights = (
            self.cfg.get("STAGE2", "WEIGHTS")
            if self.cfg.getboolean("STAGE1", "LOADPREV")
            else None
        )

        self.cross_skip = self.cfg.getboolean("STAGE2", "CROSS_SKIP")

        if self.cfg.get("STAGE1", "ENCODER") != "UNET":
            raise NotImplementedError

        log.info("STAGE 1 UNET")
        # Flow Computation Model
        self.stage1_model = unet.get_model(
            stage1_weights, 6, 4, self.cross_skip, stage=1, cfg=self.cfg
        )

        # Flow Interpolation Model
        log.info("STAGE 2 %s", self.cfg.get("STAGE2", "ENCODER"))
        self.stage2_model = unet.get_model(
            stage2_weights, 16, 5, self.cross_skip, stage=2, cfg=self.cfg
        )

        log.info("Cross stage Skip Connections Present? %s ", self.cross_skip)

    def freeze_weights(self):
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

    def get_image_pairs(self, img_tensor):
        """ Pair adjacent images in the tensor.

        :param img_tensor: B T 3 H W tensor of images.
        :return: B (T-1) 6 H W tensor -> every 2 adjacent images are paired.
        :rtype:
        """
        B = img_tensor.shape[0]
        images = list(img_tensor.split(dim=1, split_size=1))
        image_pairs = list(zip(images[:-1], images[1:]))
        image_pairs = [
            torch.cat([imgA, imgB], dim=2).squeeze() for imgA, imgB in image_pairs
        ]

        # bad code to handle B = 1
        if B == 1:
            image_pairs = [img_pair[None, ...] for img_pair in image_pairs]

        return torch.stack(image_pairs, dim=1)

    def get_intermediate_outputs(self, flowC_outputs, flowI_inputs, flowI_outputs):
        """ Get the outputs from the flow computation and flow interpolation models.

        Optical flow outputs, estimated and refined optical flows, visibility maps.

        :param flowC_outputs:
        :param flowI_inputs:
        :param flowI_outputs:
        :returns:
        :rtype:

        """

        _, mid_flowC = flowC_outputs

        flowC_01 = mid_flowC[:, [0, 1], :, :]  # flow from 0 to 1.
        flowC_10 = mid_flowC[:, [2, 3], :, :]  # flow from 1 to 0.

        est_flow_t1 = flowI_inputs[:, [6, 7], ...]
        est_flow_t0 = flowI_inputs[:, [8, 9], ...]

        v_1t = flowI_outputs[:, 0, ...]  # Visibility Map 1-> t
        dflow_t1 = flowI_outputs[:, 1:3, ...]  # Residual of flow t->1
        dflow_t0 = flowI_outputs[:, 3:5, ...]  # Residual of flow t->0

        v_1t = v_1t[:, None, ...]  # making dimensions compatible

        v_1t = torch.sigmoid(v_1t)

        v_0t = 1 - v_1t  # Visibility Map 0->t

        refined_flow_t1 = est_flow_t1 + dflow_t1
        refined_flow_t0 = est_flow_t0 + dflow_t0

        return (
            flowC_01,
            flowC_10,
            est_flow_t1,
            est_flow_t0,
            refined_flow_t1,
            refined_flow_t0,
            v_0t,
            v_1t,
        )

    @validate_target_tensor
    def forward(
        self,
        image_tensor,
        t_interp,
        target_images=None,
        iteration=None,
        inference_mode=True,
    ):
        """ Forward pass of the model.

        :param image_tensor:  B T 3 H W tensor with dtype=`np.float32` and normalized by imagenet mean and standard deviation.
        :param t_interp: B T 1 1 1 tensor (broadcasted) of values between 0 and 1 (float)
        representing interpolation points.
        :param target_images: B T C H W tensor of ground truth images
        normalized by imagenet mean and standard deviation
        :param iteration:
        :param compute_loss: flag which indicates whether to compute loss
        (during inference, set to True)
        :returns:
        :rtype:

        """

        image_pairs = self.get_image_pairs(image_tensor)
        T = image_pairs.shape[1]
        assert T == 1, "Expected one interpolation window"
        mid_idx = T // 2

        sampled_t = t_interp[:, mid_idx, ...]

        combined_encoding = []
        flowI_inputs = []

        time_step = 0  # since T = 1
        flowC_outputs = self.stage1_model(image_pairs)

        img_pair = image_pairs[:, time_step, ...]
        stage1_encoding, flow_tensor = flowC_outputs[time_step]
        input_tensor = self.stage2_model.compute_inputs(
            img_pair, flow_tensor, t=sampled_t
        )
        flowI_inputs.append(input_tensor)
        combined_encoding.append(stage1_encoding)

        flowI_inputs = torch.stack(flowI_inputs, dim=1)

        flowI_outputs = self.stage2_model(flowI_inputs, combined_encoding)

        b = image_pairs.shape[0]
        losses = torch.zeros([b, 4]).cuda()

        t = mid_idx
        img_pair = image_pairs[:, t, ...]
        flowI_in = flowI_inputs[:, t, ...]
        flowI_out = flowI_outputs[t]

        if iteration == 1:
            log.info("%s interpolation windows. Mid_idx: %s", T, t)

        flow_tensor = flowC_outputs[t][1]

        interpolation_result = self.stage2_model.compute_output_image(
            img_pair, flowI_in, flowI_out, t=sampled_t
        )

        if not inference_mode:  # and t == mid_idx:
            current_target = target_images[:, t, ...]
            losses = losses + self.loss(
                img_pair,
                flow_tensor,
                flowI_in,
                flowI_out,
                interpolation_result,
                current_target,
            )
            return interpolation_result, losses
        else:
            outputs = self.get_intermediate_outputs(
                flowC_outputs[mid_idx], flowI_inputs[mid_idx], flowI_outputs[mid_idx]
            )
            return interpolation_result, outputs  # middle result.
