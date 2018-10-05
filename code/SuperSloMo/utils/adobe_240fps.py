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
                clips.append(img_paths)
            else:
                continue

        return clips

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
        sample = sample.permute(0, 3, 2, 1) # n_frames, H W C -> n_frames, C, H, W
        return sample


def collate_fn(data):
    """
    Custom collate function.
    :params data: list of tensors to collate. each of size [n_frames, C, H, W]
    :return batch_data: batch of size B, 9, C, H, W
    """

    for idx in range(len(data)):
        tensor_data = data[idx]
        assert len(tensor_data.shape)==4, "Shape: "+str(tensor_data.shape)
        n_frames = tensor_data.shape[0]
        if n_frames == 9:
            continue

        elif n_frames < 12:
            start_idx = random.randint(0, n_frames - 9) # random starting point to get 9 frames.
            clip = tensor_data[start_idx: start_idx + 9, ...]
            data[idx] = clip

        elif n_frames >= 12:
            subset = random.randint(0, n_frames-12) # random starting point to get subset of 12 frames.
            clip_12 = tensor_data[subset:subset+12, ...]

            start_idx = random.randint(0, 3) # random starting point to get subset of 9 frames.
            clip = clip_12[start_idx:start_idx+9, ...]
            data[idx] = clip

    batch = torch.stack(data)
    return batch




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

    transformations = transforms.Compose([ResizeCrop(), ToTensor()])

    batch_size = config.getint("TRAIN", "BATCH_SIZE")

    adobe_dataset = Reader(config, split="TRAIN", transform=transformations)

    import time

    adobe_loader = DataLoader(adobe_dataset, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn)

    start = time.time()

    for idx, batch_sample in enumerate(adobe_loader):
        log.info(str(idx))
        if idx==19:
            break


    log.info(str(batch_sample.shape)+" shape of input tensor.")

    stop = time.time()
    total = (stop - start)
    log.info(stop-start)

    log.info(str(float(total)/20.0))
