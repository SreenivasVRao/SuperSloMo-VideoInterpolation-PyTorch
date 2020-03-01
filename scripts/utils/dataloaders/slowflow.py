import glob
import logging
import os

import cv2
from torchvision import transforms

from . import default_reader
from .augmentations import EvalPad, Normalize, ToTensor


log = logging.getLogger(__name__)


class SlowflowReader(default_reader.Reader):
    def __init__(self, cfg, split="VAL", eval_mode=True):
        """
        :param cfg: Config file.
        :param split: TRAIN/VAL/TEST
        """
        super(SlowflowReader, self).__init__(cfg, split, eval_mode)
        self.custom_transform = self.get_torchvision_transform()
        if not eval_mode:
            self.clips = self.read_train_clip_list()
        else:
            self.clips = self.read_inference_clip_list()

    def read_inference_clip_list(self):
        """
        :param split: TRAIN/VAL/TEST
        :return: list of all clips in split
        """

        SRC_DIR = self.cfg.get("SLOWFLOW_DATA", "ROOTDIR")

        clips = glob.glob(SRC_DIR + "/*")
        log.info("Found %s clips.", len(clips))

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

    def get_torchvision_transform(self):
        """ Applies transforms on the input tensor.

        :returns: 
        :rtype:

        """

        pix_mean = self.cfg.get("MODEL", "PIXEL_MEAN").split(",")
        pix_mean = [float(p) for p in pix_mean]
        pix_std = self.cfg.get("MODEL", "PIXEL_STD").split(",")
        pix_std = [float(p) for p in pix_std]

        if not self.eval_mode:
            raise Exception("This module is not used for training.")
        custom_transform = transforms.Compose(
            [
                Normalize(pix_mean, pix_std),
                ToTensor(),
                EvalPad(padding=None, target_dims=(1024, 1280)),
            ]
        )

        return custom_transform
