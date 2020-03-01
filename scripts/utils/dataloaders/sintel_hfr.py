import glob
import logging
import os

import cv2

import torch

from torchvision import transforms

from . import default_reader
from .augmentations import EvalPad, Normalize, ToTensor

cv2.setNumThreads(0)
log = logging.getLogger(__name__)


class SintelHFRReader(default_reader.Reader):
    def __init__(self, cfg, split="VAL", eval_mode=True):
        """
        :param cfg: Config file.
        :param split: TRAIN/VAL/TEST
        """
        super(SintelHFRReader, self).__init__(cfg, split, eval_mode)
        REQD_IMAGES = {2: 33, 4: 97, 6: 161, 8: 225}
        self.n_frames = self.cfg.getint("TRAIN", "N_FRAMES")
        self.reqd_images = REQD_IMAGES[self.n_frames]
        self.interp_factor = 32
        log.info("Using %s input frames." , self.n_frames)

        self.custom_transform = self.get_torchvision_transform()
        self.clips = self.read_inference_clip_list()

    def read_inference_clip_list(self):
        """
        :param split: TRAIN/VAL/TEST
        :return: list of all clips in split
        """

        SRC_DIR = self.cfg.get("SINTEL_HFR_DATA", "ROOTDIR")

        clips = glob.glob(SRC_DIR + "/*")
        log.info("Found %s clips." % (len(clips)))

        data = []

        for clip in sorted(clips):
            clip_dir = os.path.join(SRC_DIR, clip)
            img_paths = glob.glob(clip_dir + "/*.png")
            img_paths = sorted(img_paths)
            log.info("Found %s frames in: %s" % (len(img_paths), clip))
            for sample in self.generate_sliding_windows(img_paths):
                data.append(sample)

        log.info("Total: %s" % len(data))

        return data

    def __len__(self):
        return len(self.clips)

    def get_torchvision_transform(self):

        pix_mean = self.cfg.get("MODEL", "PIXEL_MEAN").split(",")
        pix_mean = [float(p) for p in pix_mean]
        pix_std = self.cfg.get("MODEL", "PIXEL_STD").split(",")
        pix_std = [float(p) for p in pix_std]

        assert self.eval_mode, "This module is not useful for training phase."
        custom_transform = transforms.Compose(
            [Normalize(pix_mean, pix_std), ToTensor(), EvalPad(torch.nn.ZeroPad2d([0, 0, 6, 6]))]
        )

        return custom_transform
