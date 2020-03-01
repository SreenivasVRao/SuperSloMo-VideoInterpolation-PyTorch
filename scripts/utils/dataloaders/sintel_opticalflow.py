import glob
import logging
import os

import cv2
import more_itertools
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.flo_utils import read_flow  # works due to python path conventions

from .augmentations import EvalPad, Normalize, ToTensor


log = logging.getLogger(__name__)


class Reader(Dataset):
    def __init__(self, cfg, split="TRAIN", eval_mode=True):
        """
        :param cfg: Config file.
        :param split: TRAIN/VAL/TEST
        """

        self.cfg = cfg
        self.split = split

        self.eval_mode = eval_mode

        self.custom_transform = self.get_transform()

        REQD_IMAGES = {2: 2, 4: 4}
        self.n_frames = self.cfg.getint("TRAIN", "N_FRAMES")
        self.reqd_images = REQD_IMAGES[self.n_frames]
        log.info("Using %s input frames." % self.n_frames)

        self.SRC_DIR = self.cfg.get("SINTEL_EPE_DATA", "ROOTDIR")
        self.setting = self.cfg.get("SINTEL_EPE_DATA", "SETTING").lower()
        self.FLOW_DIR = os.path.join(self.SRC_DIR, "flow")
        log.info("Using render setting: %s" % self.setting)
        self.clips = self.read_clip_list()

    def read_clip_list(self):
        """
        :return: list of all clips in split
        """

        clips = glob.glob(os.path.join(self.SRC_DIR, self.setting, "*"))
        log.info("Found %s clips." % (len(clips)))

        data = []

        for clip in sorted(clips):
            clip_dir = os.path.join(self.SRC_DIR, self.setting, clip)
            clip_name = clip_dir.split("/")[-1]
            img_paths = glob.glob(clip_dir + "/*.png")
            img_paths = sorted(img_paths)
            current_flow_dir = os.path.join(self.FLOW_DIR, clip_name)
            flow_paths = glob.glob(current_flow_dir + "/*.flo")
            flow_paths = sorted(flow_paths)
            if "training" in self.cfg.get("SINTEL_EPE_DATA", "ROOTDIR"):
                assert len(img_paths) == len(flow_paths) + 1  # sanity checking
            else: # test optical flow by uploading .flo to server
                pass

            for input_indexes, target_idx in self.sliding_window(img_paths):
                sample = [img_paths[i] for i in input_indexes]
                current_flow_path = flow_paths[target_idx]
                data.append((sample, current_flow_path))
        log.info("Found %s samples" % len(data))
        return data

    def sliding_window(self, img_paths):
        """
        Generates samples of length= self.reqd_images.
        First compute input indexes for the interpolation windows.
        Then compute left most and right most indexes.
        Check bounds. Replicate interp inputs (first and last) if necessary.
        """
        indexes = list(range(len(img_paths)))
        if self.n_frames == 4:
            indexes = [0] + indexes  # padding left
            indexes = indexes + [indexes[-1]]  # padding right

        for each_window in more_itertools.windowed(indexes, n=self.n_frames, step=1):
            input_indexes = each_window
            if self.n_frames == 2:
                target_flow_idx = each_window[0]
            else:
                target_flow_idx = each_window[1]
            yield input_indexes, target_flow_idx

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        """

        :param idx: index of sample clip in dataset
        :return: Gets the required sample, and ensures only 9 frames from the clip are returned.
        """
        img_paths, flow_path = self.clips[idx]
        assert len(img_paths) == self.reqd_images

        img_buffer = []
        for impath in img_paths:
            img_buffer.append(cv2.imread(impath)[..., ::-1])  # RGB
        img_buffer = np.array(img_buffer)
        flow_data = read_flow(flow_path)

        if self.custom_transform:
            img_buffer = self.custom_transform(img_buffer)
        flow_data = torch.from_numpy(flow_data.copy())

        return img_buffer, flow_data

    def get_transform(self):

        pix_mean = self.cfg.get("MODEL", "PIXEL_MEAN").split(",")
        pix_mean = [float(p) for p in pix_mean]
        pix_std = self.cfg.get("MODEL", "PIXEL_STD").split(",")
        pix_std = [float(p) for p in pix_std]

        assert self.eval_mode, "This module is not used during training"
        custom_transform = transforms.Compose(
            [Normalize(pix_mean, pix_std), ToTensor(), EvalPad(torch.nn.ZeroPad2d([0, 0, 6, 6])),]
        )

        return custom_transform


def data_generator(config):

    # since this module is used only during inference
    batch_size = 1 # overriding config
    # TODO fix in evaluation script
    log.warn("Setting batchsize = 1. This may not match the config.")
    n_workers = config.getint("DATALOADER", "N_WORKERS")

    dataset = Reader(config)

    n_frames = config.getint("TRAIN", "N_FRAMES")
    log.info("Generating %s frame input." % n_frames)

    shuffle_flag = False
    log.info("Shuffle: %s" % shuffle_flag)
    sintel_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=n_workers,
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed() % (2 ** 32 - 1))),
        pin_memory=True,
    )

    return sintel_loader
