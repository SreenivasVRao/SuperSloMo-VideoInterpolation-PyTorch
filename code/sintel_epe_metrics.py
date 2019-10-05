import configparser
import logging
from argparse import ArgumentParser
import numpy as np
from SuperSloMo.models import SSMR
from SuperSloMo.utils.sintel_epe import *

parser = ArgumentParser()
parser.add_argument("--log")
parser.add_argument("-c", "--config") # config
args = parser.parse_args()


def compute_metrics(flow, gt_flow):
    assert flow.shape[0]==gt_flow.shape[0]==1
    flow = flow[0, ...]
    gt_flow = gt_flow[0, ...]
    err_map = gt_flow - flow
    err_map = np.sqrt(np.sum(err_map ** 2, axis=2))
    epe = np.mean(err_map)
    idxes = np.where(err_map > 3)
    err_pct = len(idxes[0]) / err_map.shape[0] / err_map.shape[1]

    return epe, err_pct

logging.basicConfig(filename=args.log, level=logging.INFO)

config = configparser.RawConfigParser()
config.read(args.config)
logging.info("Read config")

superslomo_model = SSMR.full_model(config).cuda().eval()

import time
total = 0
samples = data_generator(config, "TRAIN") # TRAINING Set of Sintel data.

EPE = []
pct_error = []
total = len(samples)

n_frames = config.getint("TRAIN", "N_FRAMES")

for idx, (images, gt_flow) in enumerate(samples):
    b = images.shape[0]
    if b< torch.cuda.device_count():
        continue
    images = images.cuda().float()

    t_interp = [0.5]*(n_frames - 1) # doesn't really matter.
    t_interp = torch.Tensor(t_interp).view(1, (n_frames - 1), 1, 1, 1)

    t_interp = torch.Tensor(t_interp).expand(b, (n_frames - 1), 1, 1, 1).cuda().float()

    model_outputs = superslomo_model(images, dataset_info=None, t_interp=t_interp, iteration=None, compute_loss=False)
    flowC_01 = model_outputs[1]

    gt_flow = gt_flow.cpu().numpy()
    flowC_01 = flowC_01.permute(0, 2, 3, 1)[:, 6:442, ...].cpu().data.numpy()

    epe, pe_3 = compute_metrics(flowC_01, gt_flow)
    EPE.append(epe)
    pct_error.append(pe_3)

    if idx%10 ==0 :
        log.info("Iteration: %s of %s"%(idx, total))
        log.info(images.shape)
        log.info(gt_flow.shape)
        log.info("So Far: EPE: %.3f 3_pct_error: %.3f" %(np.mean(EPE), np.mean(pct_error)))

        log.info("Final average: EPE: %.3f 3_pct_error: %.3f" %(np.mean(EPE), np.mean(pct_error)))

