import sys
sys.path.insert(0, "/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/SuperSloMo/utils/")
import numpy as np
import logging
import cv2, glob, os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from common import (ResizeCrop, EvalPad, Normalize, ToTensor)
import pickle
import flo_utils2


cv2.setNumThreads(0)
log = logging.getLogger(__name__)


class Reader(Dataset):

    def __init__(self, cfg, split="TRAIN", eval_mode=False, transform=None):
        """
        :param cfg: Config file.
        :param split: TRAIN/VAL/TEST
        """

        self.cfg = cfg
        self.split = split

        self.eval_mode = eval_mode

        self.custom_transform = transform

        REQD_IMAGES = {2: 2, 4: 4}
        self.n_frames = self.cfg.getint("TRAIN", "N_FRAMES")
        self.reqd_images = REQD_IMAGES[self.n_frames]
        log.info("Using %s input frames." % self.n_frames)

        self.SRC_DIR = self.cfg.get("SINTEL_EPE_DATA", "ROOTDIR")
        self.setting = self.cfg.get("SINTEL_EPE_DATA", "SETTING").lower()
        self.FLOW_DIR = os.path.join(self.SRC_DIR, 'flow')
        log.info("Using render setting: %s"% self.setting)
        self.clips = self.read_clip_list(split)

    def read_clip_list(self, split):
        """
        :param split: TRAIN/VAL/TEST
        :return: list of all clips in split
        """


        clips = glob.glob(os.path.join(self.SRC_DIR, self.setting, '*'))
        log.info("Found %s clips."%(len(clips)))

        data = []

        for clip in sorted(clips):
            clip_dir = os.path.join(self.SRC_DIR, self.setting, clip)
            clip_name = clip_dir.split('/')[-1]
            img_paths = glob.glob(clip_dir+'/*.png')
            img_paths = sorted(img_paths)
            current_flow_dir = os.path.join(self.FLOW_DIR, clip_name)
            flow_paths = glob.glob(current_flow_dir+'/*.flo')
            flow_paths = sorted(flow_paths)
            assert len(img_paths)==len(flow_paths)+1
            
            for indexes, sample in self.sliding_window(img_paths):
                flo_idx = indexes[self.n_frames//2 - 1]
                current_flow_path = flow_paths[flo_idx]
                data.append((sample, current_flow_path))
        log.info("Found %s samples" %len(data))
        return data

    def sliding_window(self, img_paths):
        """
        Generates samples of length= self.reqd_images.
        First compute input indexes for the interpolation windows.
        Then compute left most and right most indexes.
        Check bounds. Replicate interp inputs (first and last) if necessary.
        """
        interp_inputs = list(range(len(img_paths)))
        interp_pairs = list(zip(interp_inputs[:-1], interp_inputs[1:]))
        
        for interp_start, interp_end in interp_pairs:
            left_start = interp_start - ((self.n_frames - 1)//2)
            right_end = interp_end + ((self.n_frames - 1)//2)
            input_locations = list(range(left_start, right_end+1))
            for idx in range(len(input_locations)):
                if input_locations[idx]<0:
                    input_locations[idx]= 0
                elif input_locations[idx]>=len(img_paths):
                    input_locations[idx] = len(img_paths)-1 # final index.
            sample = [img_paths[i] for i in input_locations]
            
            yield input_locations, sample

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        """

        :param idx: index of sample clip in dataset
        :return: Gets the required sample, and ensures only 9 frames from the clip are returned.
        """
        img_paths, flow_path = self.clips[idx]
        assert len(img_paths)==self.reqd_images

        img_buffer = []
        for impath in img_paths:
            img_buffer.append(cv2.imread(impath)[..., ::-1]) # RGB
        img_buffer = np.array(img_buffer)
        flow_data = flo_utils2.read_flow(flow_path)

        if self.custom_transform:
            img_buffer = self.custom_transform(img_buffer)
        flow_data = torch.from_numpy(flow_data.copy())
            
        return img_buffer, flow_data



def get_transform(config, split, eval_mode):

    pix_mean = config.get('MODEL', 'PIXEL_MEAN').split(',')
    pix_mean = [float(p) for p in pix_mean]
    pix_std = config.get('MODEL', 'PIXEL_STD').split(',')
    pix_std = [float(p) for p in pix_std]

    if eval_mode:
        custom_transform = transforms.Compose([Normalize(pix_mean, pix_std), ToTensor(), EvalPad(torch.nn.ZeroPad2d([0,0,6,6]))])
    else:
        raise Exception("This module is not useful for training phase.")
    return custom_transform


def data_generator(config, split, eval_mode=True):

    assert eval_mode

    custom_transform = get_transform(config, split, eval_mode)

    batch_size = config.getint(split, "BATCH_SIZE")
    n_workers = config.getint("MISC", "N_WORKERS")

    dataset = Reader(config, split, eval_mode, custom_transform)

    n_frames = config.getint("TRAIN", "N_FRAMES")
    log.info("Generating %s frame input."%n_frames)

    shuffle_flag = not eval_mode
    log.info("Shuffle: %s"%shuffle_flag)
    sintel_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag, num_workers=n_workers,
                              worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()%(2**32 -1))),
                              pin_memory=True)

    return sintel_loader


if __name__ == '__main__':
    import configparser
    import logging
    from argparse import ArgumentParser
    import numpy as np
    import sys
    sys.path.insert(0, "/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/SuperSloMo/models/")
    import SSMR
    
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
    