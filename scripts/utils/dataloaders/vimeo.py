import logging
import os

import numpy as np
from torchvision import transforms

from . import default_reader
from .augmentations import Normalize, RandomCrop, ToTensor

log = logging.getLogger(__name__)


class VimeoReader(default_reader.Reader):
    def __init__(self, cfg, split="TRAIN", eval_mode=False):
        super(VimeoReader, self).__init__(cfg, split, eval_mode)
        REQD_IMAGES = {2: 3, 4: 7}
        self.reqd_images = REQD_IMAGES[self.n_frames]
        self.t_sample = "FIXED"
        self.custom_transform = self.get_torchvision_transform()
        if not eval_mode:
            self.clips = self.read_train_clip_list()
        else:
            self.clips = self.read_inference_clip_list()

    def read_train_clip_list(self):
        """
        :param split: TRAIN/VAL/TEST
        :return: list of all clips in split
        """
        assert self.split == "TRAIN"
        fpath = self.cfg.get("VIMEO_DATA", "TRAINPATHS")
        with open(fpath, "r") as f:
            data = f.readlines()
            data = [d.strip() for d in data]

        clips = []

        SRC_DIR = self.cfg.get("VIMEO_DATA", "ROOTDIR")

        for sequence in data:
            sequence_dir = os.path.join(SRC_DIR, "sequences", sequence)
            image_paths = [sequence_dir + "/im" + str(i) + ".png" for i in range(1, 8)]
            image_paths = sorted(image_paths)
            assert len(image_paths) == 7
            clips.append(image_paths)
        return clips

    def read_inference_clip_list(self):
        """
        :param split: TRAIN/VAL/TEST
        :return: list of all clips in split
        """

        fpath = self.cfg.get("VIMEO_DATA", "VALPATHS")
        with open(fpath, "r") as f:
            data = f.readlines()
            sequences = [d.strip() for d in data]

        clips = []

        SRC_DIR = self.cfg.get("VIMEO_DATA", "ROOTDIR")

        for seq in sequences:
            img_dir = os.path.join(SRC_DIR, "sequences", seq)
            img_list = [os.path.join(img_dir, "im%s.png" % i) for i in range(1, 8)]
            if self.n_frames == 4:
                clips.append(([img_list[i] for i in [0, 0, 1, 2, 4]], 1))  # use 0 0 2 4 to interp 1.
                clips.append(([img_list[i] for i in [0, 2, 3, 4, 6]], 1))  # use 0 2 4 6 to interp 3.
                clips.append(([img_list[i] for i in [2, 4, 5, 6, 6]], 1))  # use 2 4 6 6 to interp 5.
            else:
                clips.append(([img_list[i] for i in [0, 1, 2]], 1))  # use 0 2 to interp 1.
                clips.append(([img_list[i] for i in [2, 3, 4]], 1))  # use 2 4 to interp 3.
                clips.append(([img_list[i] for i in [4, 5, 6]], 1))  # use 4 6 to interp 5.

        log.info("Total: %s samples. " % len(clips))

        return clips

    def get_train_item_indexes(self):
        """ Vimeo data consist of septuplets i.e., 7 frame clips
        indexes = [0, 1, 2, 3, 4, 5, 6]
        We train using this setting: possible input = [0, 2, 4, 6]
        possible interpolation targets = [1, 3, 5]

        :returns: 
        :rtype:

        """

        if self.n_frames == 2:
            interp_choice = np.random.choice([1, 3, 5])
            train_idx = [interp_choice - 1, interp_choice + 1]
            target_idx = [interp_choice]
            sampled_idx = [4]

        elif self.n_frames == 4:
            interp_choice = np.random.choice([1, 3, 5])
            if interp_choice == 1:
                train_idx = [0, 0, 2, 4]  # replicate edge.
                target_idx = [0, 1, 3]

            elif interp_choice == 3:
                train_idx = [0, 2, 4, 6]
                target_idx = [1, 3, 5]
            else:
                train_idx = [2, 4, 6, 6]  # replicate edge.
                target_idx = [3, 5, 6]  # replicate edge.

            sampled_idx = [4, 4, 4]
        else:
            raise Exception("Only supports 2, or 4 frames.")

        assert len(train_idx + target_idx) == (2 * self.n_frames - 1), "Incorrect number of frames."

        return train_idx, target_idx, sampled_idx

    def get_inference_item_indexes(self):
        input_idx = []
        groundtruth_idx = []
        if self.n_frames == 4:
            input_idx = [0, 1, 3, 4]
            groundtruth_idx = [2]
        elif self.n_frames == 2:
            input_idx = [0, 2]
            groundtruth_idx = [1]
        else:
            raise Exception("Evaluation supported only for input = 2 or 4 frames")

        return input_idx, groundtruth_idx

    def get_torchvision_transform(self):
        pix_mean = self.cfg.get("MODEL", "PIXEL_MEAN").split(",")
        pix_mean = [float(p) for p in pix_mean]
        pix_std = self.cfg.get("MODEL", "PIXEL_STD").split(",")
        pix_std = [float(p) for p in pix_std]

        if self.eval_mode:
            custom_transform = transforms.Compose([Normalize(pix_mean, pix_std), ToTensor()])
        else:
            assert self.split in ["TRAIN", "VAL"], "Invalid Split %s" % self.split
            crop_imh = self.cfg.getint(self.split, "CROP_IMH")
            crop_imw = self.cfg.getint(self.split, "CROP_IMW")
            custom_transform = transforms.Compose(
                [
                    RandomCrop((crop_imh, crop_imw)),
                    # RandomMirrorRotate(),
                    Normalize(pix_mean, pix_std),
                    ToTensor(),
                ]
            )

        return custom_transform
