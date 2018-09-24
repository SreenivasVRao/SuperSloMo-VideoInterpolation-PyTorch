import PWCNet
import FlowInterpolator

import torch
import torch.nn as nn
from torch.nn import functional as F


class FullModel(nn.Module):

    def __init__(self, stage1_weights=None, stage2_weights=None):
        super(FullModel, self).__init__()
        self.stage1_model = PWCNet.pwc_dc_net(stage1_weights)  # Flow Computation Model
        self.stage2_model = FlowInterpolator.flow_interpolator(stage2_weights)  # Flow Interpolation Model

    def stage1_computations(self, img0, img1, dims, scale_factors):
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
        flow_tensor = self.post_process_flow(flow_tensor, dims, scale_factors)

        return img_tensor, flow_tensor

    def post_process_flow(self, flow_tensor, dims, scale_factors):
        flow_tensor = flow_tensor * 20.0
        H, W = dims
        upsampled_flow = F.upsample(flow_tensor, size=(H, W), mode='bilinear')

        s_H, s_W = scale_factors
        upsampled_flow[:, 0::2, ...] = upsampled_flow[:, 0::2, ...] * s_W
        # u vectors

        upsampled_flow[:, 1::3, ...] = upsampled_flow[:, 1::3, ...] * s_H
        # v vectors

        return upsampled_flow

    def forward(self, image_0, image_1, dims, scale_factors, t_interp=0.5):
        img_tensor, flow_tensor = self.stage1_computations(image_0, image_1, dims, scale_factors)

        flowI_input = self.stage2_model.compute_inputs(img_tensor, flow_tensor, t=t_interp)
        flowI_output = self.stage2_model(flowI_input)

        interpolation_result = self.stage2_model.compute_output_image(flowI_input, flowI_output, t=t_interp)

        if self.training:
            return img_tensor, flow_tensor, flowI_input, flowI_output
        else:
            return interpolation_result


def full_model(stage1_weights, stage2_weights):
    model = FullModel(stage1_weights, stage2_weights)
    return model


if __name__ == '__main__':

    from code.SuperSloMo.utils import adobe_240fps
    from torch.autograd import Variable
    import ConfigParser, os
    config = ConfigParser.RawConfigParser()
    config.read("../../config.ini")

    stage1_weights = os.path.join(config.get("PROJECT", "DIR"), config.get("STAGE1", "WEIGHTS"))
    full_model(stage1_weights, None)
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
    interpolated_image = superslomo(image_0, image_1, adobe_dataset.dims, (adobe_dataset.s_y, adobe_dataset.s_x))

    print(interpolated_image.shape)
