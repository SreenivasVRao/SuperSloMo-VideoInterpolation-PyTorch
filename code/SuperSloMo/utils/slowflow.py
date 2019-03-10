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

        REQD_IMAGES={2:9, 4:25, 6:41, 8:57}
        self.n_frames = self.cfg.getint("TRAIN", "N_FRAMES")
        self.reqd_images = REQD_IMAGES[self.n_frames]
        self.interp_factor = 8
        log.info("Using %s input frames." % self.n_frames)

        self.pad_mode = self.cfg.get("EVAL", "PADDING")
        assert self.pad_mode in ["NONE", "REPLICATE"]

        self.clips = self.read_clip_list(split)

    def read_clip_list(self, split):
        """
        :param split: TRAIN/VAL/TEST
        :return: list of all clips in split
        """

        SRC_DIR = self.cfg.get("SLOWFLOW_DATA", "ROOTDIR")

        clips = glob.glob(SRC_DIR+"/*")
        log.info("Found %s clips."%(len(clips)))

        data = []

        for clip in sorted(clips):
            clip_dir = os.path.join(SRC_DIR, clip)
            img_paths = glob.glob(clip_dir+'/*.png')
            img_paths = sorted(img_paths)
            log.info("Found %s frames in: %s"%(len(img_paths), clip))
            for sample in self.sliding_window(img_paths):
                data.append(sample)
                
        log.info("Total: %s"%len(data))
        
        return data

    def sliding_window(self, img_paths):
        """
        Generates samples of length= self.reqd_images.
        First compute input indexes for the interpolation windows.
        Then compute left most and right most indexes.
        Check bounds. Replicate interp inputs (first and last) if necessary.
        """

        T = len(img_paths)
        img_indexes = list(range(T))

        input_idx = [i for i in range(0, T, self.interp_factor)]
        interp_windows = list(zip(input_idx[0:-1], input_idx[1:]))

        last_window = interp_windows[-1]
        interp_start, interp_end = last_window

        if interp_end < T - 1:
            # at least one more frame to interpolate. Maybe more.
            interp_start = interp_end
            interp_end = interp_start + self.interp_factor

        interp_windows.append((interp_start, interp_end))

        for idx, pair in enumerate(interp_windows):
            interp_start, interp_end = pair
            left_start = interp_start - (self.interp_factor * ((self.n_frames - 1) // 2))
            right_end = interp_end + (self.interp_factor * ((self.n_frames - 1) // 2)) + 1
            # +1 so that the final index is included.
            # ensures that the interp_start:interp_end window is at the center always.

            if left_start >=0 and right_end <= T:
                current_window = img_indexes[left_start:right_end]

            elif left_start < 0:
                current_window = img_indexes[0:right_end]
                n_pad = abs(left_start)
                first_interp_idx = 0
                left_pad = [img_indexes[first_interp_idx]]*n_pad
                current_window = left_pad + current_window

            elif right_end > T:
                # replicate the last interpolation input.
                last_interp_idx = ((T-1) // self.interp_factor) * self.interp_factor
                # if there are T = 49 frames, and interp_factor = 8, last_interp_input = frame 48 [0 indexed].
                # if there are T = 48 frames, 47 // 8 * 8 = 40
                n_pad = (right_end - T)
                current_window = img_indexes[left_start:T]
                right_pad = [img_indexes[last_interp_idx]]*n_pad
                current_window = current_window + right_pad

            assert len(current_window)==self.reqd_images

            if idx == len(interp_windows)-1:
                if interp_end >= T:
                    n_avail = (T-1) - interp_start
                    log.info((interp_start, interp_end, T-1, n_avail))
            else:
                assert interp_end < T
                n_avail = self.interp_factor-1
            # if idx < 2 or len(interp_windows)-2<= idx<=len(interp_windows)-1:
            #     log.info("%s --- %s"%(interp_start, interp_end))
            #     log.info(current_window[::8])
                
            sample_paths = [img_paths[i] for i in current_window]
            assert len(sample_paths)==self.reqd_images
            yield (sample_paths, n_avail)
            # move to the next interpolation point.

    def get_reqd_idx(self):
        n_frames = self.cfg.getint("TRAIN", "N_FRAMES")
        t_index = [i * self.interp_factor for i in range(n_frames)] # [0, 8, 16, 24 ... ] input frames.
        mid_idx = int(np.mean(t_index)) 
        t1 = mid_idx - 3
        t2 = t1 + 7
        t_index.extend(range(t1, t2))  # most intermediate frames to be interpolated.
        t_index = sorted(t_index)
        interp_idx = None

        return t_index, interp_idx


    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        """

        :param idx: index of sample clip in dataset
        :return: Gets the required sample, and ensures only 9 frames from the clip are returned.
        """
        img_paths, n_avail = self.clips[idx]
        t_idx, interp_idx = self.get_reqd_idx()
        sample = read_sample(img_paths, t_idx)
        sample = self.custom_transform(sample)
        return sample, n_avail

    
def read_sample(img_paths, t_index=None):
    if t_index:
        img_paths = [img_paths[idx] for idx in t_index]
        
    img = cv2.imread(img_paths[0])
    h, w, c = img.shape
    frames = np.zeros([len(img_paths), h, w, c])  # images are sometimes flipped for vertical videos.

    for idx, fpath in enumerate(img_paths):
        img = cv2.imread(fpath)
        frames[idx,...] = img[..., ::-1] # BGR -> RGB

    if h > w: # vertical video. W = 720, H =1280
        frames = frames.swapaxes(1, 2)

    return frames


def get_transform(config, split, eval_mode):

    pix_mean = config.get('MODEL', 'PIXEL_MEAN').split(',')
    pix_mean = [float(p) for p in pix_mean]
    pix_std = config.get('MODEL', 'PIXEL_STD').split(',')
    pix_std = [float(p) for p in pix_std]

    if eval_mode:
        custom_transform = transforms.Compose([Normalize(pix_mean, pix_std), ToTensor(), EvalPad(padding=None, target_dims=(1024, 1280))])

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
    slowflow_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag, num_workers=n_workers,
                              worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()%(2**32 -1))),
                              pin_memory=True)

    return slowflow_loader


if __name__ == '__main__':
    import configparser
    import logging
    from argparse import ArgumentParser
    import numpy as np
    
    parser = ArgumentParser()
    parser.add_argument("--log")
    parser.add_argument("--config") # config
    args = parser.parse_args()

    logging.basicConfig(filename=args.log, level=logging.INFO)

    config = configparser.RawConfigParser()
    config.read("/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/configs/ssmr.ini")
    # config.read(args.config)
    logging.info("Read config")

    import time
    total = 0
    samples = data_generator(config, "VAL")
    for idx, batch in enumerate(samples):
        data, n_avail = batch
        data = data.permute(0, 1, 3, 4, 2)
        log.info(n_avail.shape)
        log.info(data.shape)
        data = data.cpu().numpy()
        data = data[0, ...]
        for t in range(data.shape[0]):
            img = data[t, ...]
            cv2.imwrite(str(t).zfill(3)+'.png', img[..., ::-1])
        break
