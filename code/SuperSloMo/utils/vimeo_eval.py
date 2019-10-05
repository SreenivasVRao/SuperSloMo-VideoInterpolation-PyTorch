import numpy as np
import logging
import os, glob
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
from .common import (AugmentData, EvalPad, Normalize, ToTensor, RandomCrop)


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
        log.info("USING VIMEO %s DATA."%split)
        self.n_frames = self.cfg.getint("TRAIN", "N_FRAMES")
        self.custom_transform = transform
        self.clips = self.read_clip_list(split)

    def read_clip_list(self, split):
        """
        :param split: TRAIN/VAL/TEST
        :return: list of all clips in split
        """

        fpath = self.cfg.get("VIMEO_DATA", split + "PATHS")
        with open(fpath, "r") as f:
            data = f.readlines()
            sequences = [d.strip() for d in data]

        clips = []

        SRC_DIR = self.cfg.get("VIMEO_DATA", "ROOTDIR")

        for seq in sequences:
            img_dir = os.path.join(SRC_DIR, "sequences", seq)
            img_list = [os.path.join(img_dir, 'im%s.png'%i) for i in range(1, 8)]
            if self.n_frames == 4:
                clips.append(([img_list[i] for i in [0, 0, 1, 2, 4]], seq, 1)) # use 0 0 2 4 to interp 1.
                clips.append(([img_list[i] for i in [0, 2, 3, 4, 6]], seq, 3)) # use 0 2 4 6 to interp 3.
                clips.append(([img_list[i] for i in [2, 4, 5, 6, 6]], seq, 5)) # use 2 4 6 6 to interp 5.
            else:
                clips.append(([img_list[i] for i in [0, 1, 2]], seq, 1)) # use 0 2 to interp 1.
                clips.append(([img_list[i] for i in [2, 3, 4]], seq, 3)) # use 2 4 to interp 3.
                clips.append(([img_list[i] for i in [4, 5, 6]], seq, 5)) # use 4 6 to interp 5.

        log.info("Total: %s samples. "%len(clips))

        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        """

        :param idx: index of sample clip in dataset
        :return: Gets the required sample, and ensures only 9 frames from the clip are returned.
        """
        img_paths, seq, pos = self.clips[idx]
        # most intermediate.
        sample = read_sample(img_paths)
        sample = self.custom_transform(sample)
        return sample, seq, pos


def read_sample(img_paths, t_index=None):
    if t_index:
        img_paths = [img_paths[idx] for idx in t_index]

    img = cv2.imread(img_paths[0])
    h, w, c = img.shape
    frames = np.zeros([len(img_paths), h, w, c])  # images are sometimes flipped for vertical videos.

    for idx, fpath in enumerate(img_paths):
        frames[idx, ...] = cv2.imread(fpath)[..., ::-1] # RGB

    if h > w:  # vertical video. W = 720, H =1280
        frames = frames.swapaxes(1, 2) # n h w c

    return frames


def get_transform(config, split):

    pix_mean = config.get('MODEL', 'PIXEL_MEAN').split(',')
    pix_mean = [float(p) for p in pix_mean]
    pix_std = config.get('MODEL', 'PIXEL_STD').split(',')
    pix_std = [float(p) for p in pix_std]

    custom_transform = transforms.Compose([Normalize(pix_mean, pix_std), ToTensor()])

    return custom_transform


def data_generator(config, split, eval_mode=True):
    assert eval_mode
    batch_size = config.getint(split, "BATCH_SIZE")

    n_workers = config.getint("MISC", "N_WORKERS")
    
    custom_transform = get_transform(config, split)

    dataset = Reader(config, split, eval_mode, custom_transform)

    shuffle_flag = not eval_mode
    log.info("Shuffle: %s" % shuffle_flag)
    vimeo_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag, num_workers=n_workers,
                            worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed() % (2 ** 32 - 1))),
                            pin_memory=True)

    return vimeo_loader
