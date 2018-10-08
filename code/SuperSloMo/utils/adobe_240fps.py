import glob, logging
import cv2
import numpy as np
import random, os
from math import ceil
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

log = logging.getLogger(__name__)


class Reader(DataLoader):

    def __init__(self, cfg, split="TRAIN", transform=None):

        self.cfg = cfg
        self.compute_scale_factors()
        self.t_interp = self.cfg.getfloat("TRAIN", "T_INTERP")
        self.t_index = int(math.floor(self.t_interp*9)) # bad code.

        self.clips = self.read_clip_list(split)
        self.split = split
        self.transform = transform

        log.info(split+ ": Extracted clip list.")


    def read_clip_list(self, split):

        fpath = self.cfg.get("ADOBE_DATA", split+"PATHS")
        with open(fpath, "rb") as f:
            data = f.readlines()
            data = [d.strip() for d in data]

        clips = []

        data = [d.replace("/home/", "/mnt/nfs/work1/elm/") for d in data]
        data = [d.replace("/workspace", "") for d in data]

        for idx, d in enumerate(data):
            if len(d)<=2:
                nframes = int(d)
                img_paths = data[idx + 1 : idx + 1 + nframes]
                img_paths = self.get_required_images(img_paths)
                clips.append(img_paths)
            else:
                continue

        return clips

    def get_required_images(self, image_list):
        n_frames = len(image_list)

        if n_frames == 9:
            pass

        elif n_frames < 12:
            start_idx = random.randint(0, n_frames - 9) # random starting point to get 9 frames.
            image_list = image_list[start_idx: start_idx + 9]

        elif n_frames >= 12:
            subset = random.randint(0, n_frames-12) # random starting point to get subset of 12 frames.
            image_list = image_list[subset:subset+12]

            start_idx = random.randint(0, 3) # random starting point to get subset of 9 frames.
            image_list = image_list[start_idx:start_idx+9]

        assert len(image_list)==9 and 0<self.t_index < 9, "Something went wrong."
        image_list = [image_list[0], image_list[self.t_index], image_list[-1]]

        return image_list



    def compute_scale_factors(self):

        self.H = self.cfg.getint("ADOBE_DATA", "H")
        self.W = self.cfg.getint("ADOBE_DATA", "W")
        self.dims = (self.H, self.W)

        divisor = 64.

        H_ = int(ceil(self.H / divisor) * divisor)
        W_ = int(ceil(self.W / divisor) * divisor)

        self.s_x = float(self.W) / W_
        self.s_y = float(self.H) / H_

        self.scale_factors= (self.s_y, self.s_x)

    def __len__(self):

        return len(self.clips)

    def __getitem__(self, idx):

        img_paths = self.clips[idx]
        H = 720 # known size of Adobe240FPS dataset
        W = 1280 # known size of Adobe dataset.
        frames = np.zeros([len(img_paths), H, W, 3])
        for idx, fpath in enumerate(img_paths):
            img = cv2.imread(fpath)
            h, w, c = img.shape
            if (h,w) == (720, 1280):
                frames[idx,...] = img
            elif (h,w) == (1280, 720):
                frames[idx, ...] = img.swapaxes(0, 1) # flipping the image for vertical videos
            else:
                log.info(str(img.shape) + " "+fpath)

        if self.transform:
            frames = self.transform(frames)

        return frames

class ResizeCrop(object):
    """
    Convert 720 x 1280 frames to 320 x 640 -> Resize + Random Cropping
    """

    def __call__(self, sample_frames):

        new_frames = np.zeros((sample_frames.shape[0], 360, 640, 3))
        for idx in range(sample_frames.shape[0]):
            new_frames[idx, ...] = cv2.resize(sample_frames[idx, ...], (640, 360))

        h_start  =random.randint(0, 360-320)

        new_frames = new_frames[:, h_start:h_start+320, ...]
        return new_frames

class ToTensor(object):

    def __call__(self, sample):
        sample = torch.from_numpy(sample)
        sample = sample.permute(0, 3, 1, 2) # n_frames, H W C -> n_frames, C, H, W
        return sample


def data_generator(config, split):
    transformations = transforms.Compose([ResizeCrop(), ToTensor()])
    batch_size = config.getint(split, "BATCH_SIZE")
    n_workers = config.getint("MISC", "N_WORKERS")

    dataset = Reader(config, split, transform=transformations)
    adobe_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    for batch_sample in adobe_loader:
        yield batch_sample


def get_data_info(config, split):
    transformations = transforms.Compose([ResizeCrop(), ToTensor()])
    dataset = Reader(config, split, transform=transformations)
    return dataset.dims, dataset.scale_factors


if __name__ == '__main__':
    import ConfigParser
    import logging
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--log")
    parser.add_argument("--config") # config
    args = parser.parse_args()


    logging.basicConfig(filename=args.log, level=logging.INFO)

    config = ConfigParser.RawConfigParser()
    config.read(args.config)
    logging.info("Read config")

    samples = data_generator(config, "TRAIN")
    import time
    start = time.time()

    for idx in range(20):
        log.info(str(idx))
        batch = next(samples)
        log.info(batch.shape)

    stop = time.time()
    total = (stop - start)
    log.info(stop-start)

    log.info(str(float(total)/20.0))
