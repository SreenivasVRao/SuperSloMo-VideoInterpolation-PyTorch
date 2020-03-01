"""
SuperSloMo-R (R= Recurrent model)
"""

import logging

import torch
import torch.nn as nn

from . import unetflow as unet
from .losses import SSMLosses

log = logging.getLogger(__name__)


def validate_target_tensor(model_forward_func):
    def func_wrapper(
        self, image_tensor, t_interp, target_images=None, iteration=None, inference_mode=True,
    ):
        if not inference_mode:
            assert target_images is not None, "No target found for loss."
            assert (
                target_images.shape[1] == self.cfg.getint("TRAIN", "N_FRAMES") - 1
            ), "Insufficient number of targets."

        return model_forward_func(
            self, image_tensor, t_interp, target_images, iteration, inference_mode
        )

    return func_wrapper


class FullModel(nn.Module):
    def __init__(self, cfg, writer=None):
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
            self.cfg.get("STAGE1", "WEIGHTS") if self.cfg.getboolean("STAGE1", "LOADPREV") else None
        )

        stage2_weights = (
            self.cfg.get("STAGE2", "WEIGHTS") if self.cfg.getboolean("STAGE1", "LOADPREV") else None
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

        images = list(img_tensor.split(dim=1, split_size=1))
        image_pairs = list(zip(images[:-1], images[1:]))
        image_pairs = [torch.cat([imgA, imgB], dim=2).squeeze() for imgA, imgB in image_pairs]

        # bad code to handle B = 1
        if len(image_pairs[0].shape) < 4:
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
        )

    def get_stage1_outputs(self, image_pairs, t_interp, T, flowC_outputs):
        """ Computes flow computation model output at each time step.

        :param image_pairs:  B T-1 C H W tensor (C= 6)
        :param t_interp:  B T-1 1 1 1 tensor (values in 0-1)
        :param T: number of input frames (2, 4, 6, 8)
        :param flowC_outputs: raw outputs from flow computation model
        :returns: flow interpolation model inputs, encodings from flow computation model
        :rtype: tuple whose elements are each a list of tensors

        """

        flowC_encodings = []
        flowI_inputs = []

        for time_step in range(T):
            img_pair = image_pairs[:, time_step, ...]
            stage1_encoding, flow_tensor = flowC_outputs[time_step]

            sampled_t = t_interp[:, time_step, ...]
            # these are the actual values of t in 0-1 range
            # as specified in Jiang et. al. (2018)

            input_tensor = self.stage2_model.compute_inputs(img_pair, flow_tensor, t=sampled_t)
            flowI_inputs.append(input_tensor)
            flowC_encodings.append(stage1_encoding)

        flowI_inputs = torch.stack(flowI_inputs, dim=1)
        return flowI_inputs, flowC_encodings

    def get_stage2_outputs(
        self,
        image_pairs,
        t_interp,
        T,
        flowC_outputs,
        flowI_inputs,
        flowI_outputs,
        target_images=None,
        inference_mode=True,
    ):

        """ Compute flow interpolation outputs for every time step.

        :param image_pairs: B T-1 C H W tensor (C = 6)
        :param t_interp: B T-1 1 1 1 tensor (between 0 and 1)
        :param T: number of frames as input (2, 4, 6, 8)
        :param flowC_outputs: flow computation model outputs
        :param flowI_inputs: flow interpolation model inputs
        :param flowI_outputs: flow interpolation model raw outputs
        :param target_images: B T-1 C H W tensor ground truth images
        :param inference_mode: boolean
        :returns: tuple consisting of estimated image and loss tensor during training,
        and estimated image and intermediate flow interpolation output during inference
        :rtype:

        """

        b = image_pairs.shape[0]
        losses = torch.zeros([b, 4]).cuda()
        mid_idx = T // 2
        est_img_t = None

        for time_step in range(T):
            sampled_t = t_interp[:, time_step, ...]
            flowI_out = flowI_outputs[time_step]
            flowI_in = flowI_inputs[:, time_step, ...]
            img_pair = image_pairs[:, time_step, ...]

            interpolation_result = self.stage2_model.compute_output_image(
                img_pair, flowI_in, flowI_out, t=sampled_t
            )
            flow_tensor = flowC_outputs[time_step][1]

            if not inference_mode:
                current_target = target_images[:, time_step, ...]
                losses = losses + self.loss(
                    img_pair,
                    flow_tensor,
                    flowI_in,
                    flowI_out,
                    interpolation_result,
                    current_target,
                )

            if time_step == mid_idx:
                est_img_t = interpolation_result

        losses /= T  # average over timesteps

        if not inference_mode:
            return est_img_t, losses
        else:
            outputs = self.get_intermediate_outputs(
                flowC_outputs[mid_idx], flowI_inputs[:, mid_idx, :, :, :], flowI_outputs[mid_idx],
            )
            return est_img_t, outputs

    @validate_target_tensor
    def forward(
        self, image_tensor, t_interp, target_images=None, iteration=None, inference_mode=True,
    ):
        """ Forward pass of superslomo recurrent model.

        note: This model can be used to implement plain superslomo model by making bottleneck = CONV and N_FRAMES=2 in the config

        :param image_tensor: B T C H W tensor of normalized images
        :param t_interp: interpolation time in 0-1 range B T-1 1 1 1 tensor
        :param target_images: B T-1 C H W tensor
        :param iteration: int
        :param inference_mode: boolean
        :returns: estimated image and loss during training or estimated image and intermediate outputs during test.
        :rtype:

        """

        image_pairs = self.get_image_pairs(image_tensor)
        T = image_pairs.shape[1]
        mid_idx = T // 2

        if iteration == 1:
            log.info("%s interpolation windows. Mid_idx: %s", T, mid_idx)

        flowC_outputs = self.stage1_model(image_pairs)

        flowI_inputs, flowC_encodings = self.get_stage1_outputs(
            image_pairs, t_interp, T, flowC_outputs
        )

        flowI_outputs = self.stage2_model(flowI_inputs, flowC_encodings)
        final_outputs = self.get_stage2_outputs(
            image_pairs,
            t_interp,
            T,
            flowC_outputs,
            flowI_inputs,
            flowI_outputs,
            target_images,
            inference_mode,
        )

        return final_outputs


if __name__ == "__main__":
    import configparser

    config = configparser.RawConfigParser()
    config.read("configs/ssmr.ini")

    logging.basicConfig(filename="test.log", level=logging.INFO)

    ssm_net = FullModel(config).cuda()
    N = 4
    B = 2
    test_in = torch.rand([B, N, 3, 64, 64]).cuda()
    target = torch.rand([B, N - 1, 3, 64, 64]).cuda()
    t_interp = torch.rand([B, N - 1, 1, 1, 1]).cuda()

    est_image, losses = ssm_net(test_in, None, t_interp, target, iteration=1, inference_mode=False)
    x = losses.mean(dim=0)
    log.info(x.size())
    x[0].backward()
