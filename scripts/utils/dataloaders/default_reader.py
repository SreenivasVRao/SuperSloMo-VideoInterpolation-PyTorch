import itertools
import logging

import cv2
import more_itertools
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..validators import (
    validate_image_paths_length,
    validate_inference_item_indexes,
    validate_inference_tensor_shapes,
    validate_train_tensor_shapes,
)
from .augmentations import EvalPad, Normalize, RandomCrop, ToTensor

cv2.setNumThreads(0)
log = logging.getLogger(__name__)


class Reader(Dataset):
    def __init__(self, cfg, split="TRAIN", eval_mode=False):
        """
        :param cfg: Config file.
        :param split: TRAIN/VAL/TEST
        """
        self.cfg = cfg
        self.dataset_name = self.cfg.get("DATA", "DATASET")
        # typical experimental setup is to convert 30fps t0 240fps
        self.interp_factor = 32 if self.dataset_name == "SINTEL_HFR" else 8

        self.split = split
        self.WINDOW_LENGTH = self.cfg.getint("DATA", "WINDOW_LENGTH")
        REQD_IMAGES = {2: 9, 4: 25, 6: 41, 8: 57}
        self.n_frames = self.cfg.getint("TRAIN", "N_FRAMES")
        self.reqd_images = REQD_IMAGES[self.cfg.getint("TRAIN", "N_FRAMES")]
        self.eval_mode = eval_mode
        self.t_sample = self.cfg.get("DATALOADER", "T_SAMPLE")
        log.info("Using SAMPLE MODE: %s", self.t_sample)
        self.clips = []
        self.custom_transform = self.get_torchvision_transform()

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        """
        :param idx: index of sample clip in dataset
        :return: Gets the required sample, and ensures only 9 frames from the clip are returned.
        """
        if self.eval_mode:
            return self.get_inference_item(idx)
        else:
            return self.get_train_item(idx)

    @validate_train_tensor_shapes
    def get_train_item(self, idx):
        img_paths = self.clips[idx]
        if self.dataset_name in ["ADOBE", "NFS"]: # lazy bad code
            img_paths = self.get_random_window_in_clip(img_paths)

        if np.random.randint(0, 2) == 1:
            img_paths = img_paths[::-1]

        (
            train_idx,
            target_idx,
            sampled_idx,
        ) = self.get_train_item_indexes()  # handles the sampling.
        combined_idx = train_idx + target_idx

        sample = self.read_sample(img_paths, combined_idx)
        sample = self.custom_transform(sample)
        input_tensor = sample[0 : self.n_frames, :, :, :]
        target_tensor = sample[self.n_frames :, :, :, :]

        interp_idx = torch.Tensor(sampled_idx).view(-1, 1, 1, 1)  # T C H W

        t_interp = interp_idx / 8.0
        # get value between 0 - 1 for interpolation to 240FPS

        return input_tensor, target_tensor, t_interp

    @validate_inference_tensor_shapes
    def get_inference_item(self, idx):
        """ Return a single sample of input, ground truth and number of valid ground truths.

        :param idx: index of the clips list
        :returns: input tensor, target_tensor, n_targets
        :rtype: N C H W tensor, (N-1) C H W tensor, int where N = `self.n_frames`

        """
        img_paths, n_targets = self.clips[idx]
        input_idx, target_idx = self.get_inference_item_indexes()
        combined_idx = input_idx + target_idx
        sample = self.read_sample(img_paths, combined_idx)

        # transform all the images in the same manner
        sample = self.custom_transform(sample)

        input_tensor = sample[0 : self.n_frames, :, :, :]  # input images
        target_tensor = sample[self.n_frames :, :, :, :]  # target images

        return input_tensor, target_tensor, n_targets

    @validate_image_paths_length
    def get_random_window_in_clip(self, img_paths):
        """ Gets a random subwindow of the clip such that the length of the window
        equals the number of required images


        :param img_paths: all the images in the clip.
        :returns: subwindow of the clip
        :rtype: list of image paths with length = `reqd_images`

        """
        start_idx = np.random.randint(0, len(img_paths) - self.reqd_images + 1)
        end_idx = start_idx + self.reqd_images

        return img_paths[start_idx:end_idx]

    def read_train_clip_list(self):
        raise NotImplementedError

    def read_inference_clip_list(self):
        raise NotImplementedError

    @validate_inference_item_indexes
    def get_inference_item_indexes(self):
        """ Returns the input and groundtruth indexes in the full clip.
        Ground truth index is always between the most intermediate input indexes.

        :returns: list of input indexes, list of ground truth indexes
        :rtype: tuple of lists

        """
        # [0, 8, 16, 24 ... ] input frames when interpolating 8x
        # [0, 32, 64, 96 ... ] input frames when interpolating 32x
        input_idx = [i * self.interp_factor for i in range(self.n_frames)]
        # during evaluation, no sampling should happen
        # all images in the most intermediate window become target ground truth.
        # most intermediate frames to be interpolated.
        # for example if input_idx = [0, 8, 16, 24]
        # interp_idx = [9, 10, 11, 12, 13, 14, 15] -> images to be interpolated
        mid_idx = len(input_idx) // 2 - 1
        t1 = input_idx[mid_idx] + 1
        t2 = input_idx[mid_idx + 1]
        groundtruth_idx = list(range(t1, t2))
        return input_idx, groundtruth_idx

    def get_train_item_indexes(self):
        """ Get input indexes, and interpolation points that were randomly sampled.
        During training, randomly choose a single idx in each window
        Same interpolation idx across all windows = slightly better performance

        :returns: list of all indexes (input + groundtruth), list of ground truth indexes
        :rtype: tuple of lists

        """

        assert self.interp_factor == 8, "Expected 240FPS input during training"
        input_idx = [i * self.interp_factor for i in range(self.n_frames)]

        if self.t_sample == "RANDOM":
            sampled_idx = [np.random.randint(1, self.interp_factor)] * (self.n_frames - 1)

        elif self.t_sample == "MIDDLE":
            sampled_idx = [self.interp_factor // 2] * (self.n_frames - 1)

        else:
            raise NotImplementedError

        # add frame offset for reading from the list of images
        # during training, we only have 240FPS images in the dataset
        # so we multiply by 8 here (interp factor)
        interp_idx = [t + i * self.interp_factor for i, t in enumerate(sampled_idx)]

        return input_idx, interp_idx, sampled_idx

    def read_sample(self, img_paths, t_index):
        """ Loads images from the img_paths list based on t_index

        :param img_paths: list of full paths to each image in the clip
        :param t_index: list of indexes to read from the clip
        :returns: N C H W numpy array where N = self.n_frames, C = 3
        :rtype: numpy uint8 array

        """
        assert t_index is not None
        img_paths = [img_paths[idx] for idx in t_index]

        img = cv2.imread(img_paths[0])
        h, w, c = img.shape
        frames = np.zeros([len(img_paths), h, w, c])

        for idx, fpath in enumerate(img_paths):
            frames[idx, ...] = cv2.imread(fpath)

        frames = frames[..., ::-1]  # BGR -> RGB

        if h > w:
            # images are sometimes flipped for vertical videos.
            frames = frames.swapaxes(1, 2)

        return frames

    def pad_clip_edges(self, indexes):
        left_padding = self.interp_factor * (self.n_frames // 2 - 1)
        right_padding = self.interp_factor * (self.n_frames // 2 - 1)

        # handle the last interpolation window
        last_idx = len(indexes) - 1
        if (last_idx % self.interp_factor) == 0:
            n_last_window = self.interp_factor - 1  # last window is full
        else:
            n_last_window = last_idx % self.interp_factor
            right_padding += self.interp_factor - n_last_window

        # pad the clip so that first window will start at idx = 0
        # and last window will include last idx = N -1
        indexes = [0] * left_padding + indexes

        # if there's trailing images at the end to interpolate.
        # ensure padding uses correct input
        last_input = (last_idx // self.interp_factor) * self.interp_factor
        log.info("Last index: %s Last input %s", last_idx, last_input)
        indexes = indexes + [indexes[last_input]] * right_padding

        return indexes, n_last_window

    def generate_sliding_windows(self, img_paths):
        indexes = list(range(len(img_paths)))
        indexes, n_last = self.pad_clip_edges(indexes)

        generator_fn = more_itertools.windowed(indexes, n=self.reqd_images, step=self.interp_factor)
        generator_1, generator_2 = itertools.tee(generator_fn)  # copy it
        n_windows = len(list(generator_1))
        log.info("number of windows: %s", n_windows)
        # generate sliding window of size = reqd_images, and step size = interpolation factor
        for idx, each_window in enumerate(generator_2):
            log.info("WINDOW: %s", each_window)
            sample_paths = [img_paths[i] for i in each_window]
            if idx == n_windows - 1:  # last window
                yield sample_paths, n_last
            else:  # all other windows
                yield sample_paths, self.interp_factor - 1

    def get_torchvision_transform(self):
        """ Get torchvision transform to apply to each sample in the batch.

        :returns: torchvision.transforms object
        :rtype: 

        """

        pix_mean = self.cfg.get("MODEL", "PIXEL_MEAN").split(",")
        pix_mean = [float(p) for p in pix_mean]
        pix_std = self.cfg.get("MODEL", "PIXEL_STD").split(",")
        pix_std = [float(p) for p in pix_std]

        log.info("EVAL MODE %s", self.eval_mode)

        if self.eval_mode:
            custom_transform = transforms.Compose(
                [
                    Normalize(pix_mean, pix_std),
                    ToTensor(),
                    EvalPad(torch.nn.ZeroPad2d([0, 0, 8, 8])),
                ]
            )
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


def get_dataloader(dataset):
    config = dataset.cfg
    split = dataset.split

    batch_size = config.getint(split, "BATCH_SIZE")
    n_workers = config.getint("DATALOADER", "N_WORKERS")

    n_frames = config.getint("TRAIN", "N_FRAMES")
    log.info("Model trained with %s frame input.", n_frames)

    shuffle_flag = not dataset.eval_mode
    log.info("Shuffle: %s", shuffle_flag)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=n_workers,
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed() % (2 ** 32 - 1))),
        pin_memory=True,
        drop_last=(not dataset.eval_mode),
    )

    return loader
