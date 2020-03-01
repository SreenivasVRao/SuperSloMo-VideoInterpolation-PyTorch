import logging

import torch

from .flow_computation import FlowComputationModel
from .flow_interpolation import FlowInterpolationModel

log = logging.getLogger(__name__)


def get_model(path, in_channels, out_channels, cross_skip,
              verbose=False, stage=1, cfg=None):
    assert stage in [1, 2], "Unsupported stage id."
    if stage == 1:
        model = FlowComputationModel(in_channels, out_channels, cross_skip,
                                     verbose=verbose, cfg=cfg)
    else:
        model = FlowInterpolationModel(in_channels, out_channels, cross_skip,
                                       verbose=verbose, cfg=cfg)
    if path is None:
        log.info("Not loading weights for stage %s.", stage)
        return model

    key = 'stage%s_state_dict' % stage
    data = torch.load(path)
    if key in data.keys():
        data = data[key]
        log.info("Loading weights for Stage %s UNet.", stage)

    model.load_state_dict(data)

    return model


if __name__ == '__main__':
    logging.basicConfig(filename="test.log", level=logging.INFO)

    flowC_model = get_model(path=None, in_channels=6,
                            out_channels=4, cross_skip=True, verbose=True, stage=1)
    flowI_model = get_model(path=None, in_channels=16,
                            out_channels=5, cross_skip=True, verbose=True, stage=2)

    flowC_model = flowC_model.cuda()
    flowI_model = flowI_model.cuda()

    stage1_input = torch.randn([1, 6, 320, 640]).cuda()
    encoder_out, flow_tensor = flowC_model(stage1_input)
    log.info("Encoder: %s", str(encoder_out.shape))
    stage2_input = torch.randn([1, 16, 320, 640]).cuda()
    stage2_out = flowI_model(stage2_input, encoder_out)
    logging.info("Done.")
