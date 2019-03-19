import sys
import os
sys.path.insert(0, "/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/SuperSloMo/utils/")
import numpy as np
import logging
import cv2
from math import ceil
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from common import (AugmentData, EvalPad, Normalize, ToTensor, RandomCrop)

cv2.setNumThreads(0)
log = logging.getLogger(__name__)


class VimeoReader:

    def __init__(self, cfg, split="TRAIN", eval_mode=False, transform=None, t_sample="FIXED"):
        """
        :param cfg: Config file.
        :param split: TRAIN/VAL/TEST
        """
        self.cfg = cfg
        assert t_sample=="FIXED"
        self.t_sample = t_sample
        self.n_frames = self.cfg.getint("TRAIN", "N_FRAMES")
        self.clips = self.read_clip_list(split)
        self.split = split
        REQD_IMAGES = {2: 3, 4: 7}
        self.reqd_images = REQD_IMAGES[self.n_frames]
        self.eval_mode = eval_mode
        log.info("Launching Vimeo reader with T_SAMPLE: %s N_FRAMES: %s"%(self.t_sample, self.n_frames))
        
        self.custom_transform = transform

    def get_start_end(self, img_paths):
        """
        gets start-end indexes for each N_FRAMES setting such that the most intermediate frames of the full sample are at the center.
        """
        assert len(img_paths) == 7, "Expected 7 frames per sample."
        n_frames = self.cfg.getint("TRAIN", "N_FRAMES")

        if self.eval_mode:
            raise NotImplementedError
            return range(0, 8)
        else:
            return (0, 8)

    def read_clip_list(self, split):
        """
        :param split: TRAIN/VAL/TEST
        :return: list of all clips in split
        """

        fpath = self.cfg.get("VIMEO_DATA", split + "PATHS")
        with open(fpath, "r") as f:
            data = f.readlines()
            data = [d.strip() for d in data]

        clips = []

        SRC_DIR = self.cfg.get("VIMEO_DATA", "ROOTDIR")

        for idx, sequence in enumerate(data):
            sequence_dir = os.path.join(SRC_DIR, "sequences", sequence)
            image_paths = [sequence_dir + "/im" + str(i) + ".png" for i in range(1, 8)]
            image_paths = sorted(image_paths)
            assert len(image_paths) == 7
            clips.append(image_paths)
        return clips

    def get_reqd_idx(self):
        t_sample = self.t_sample
        n_frames = self.n_frames
        if t_sample == "NIL":
            raise NotImplementedError

        elif t_sample == "FIXED":
            if n_frames == 2:
                interp_choice = np.random.choice([1, 3, 5])
                indexes = [interp_choice - 1, interp_choice, interp_choice + 1]
                t_index = sorted(indexes)
                interp_idx = None

            elif n_frames == 4:
                interp_choice = np.random.choice([1, 3, 5])
                if interp_choice == 1:
                    indexes = [0, 0, 2, 4] + [0, 1, 3]  # replicate edge.
                elif interp_choice == 3:
                    indexes = [0, 2, 4, 6] + [1, 3, 5]
                elif interp_choice == 5:
                    indexes = [2, 4, 6, 6] + [3, 5, 6]  # replicate edge.
                else:
                    raise Exception("Incorrect center.")
                t_index = sorted(indexes)
                interp_idx = None
            else:
                raise Exception("Only supports 2, or 4 frames.")

            assert len(t_index) == (2 * n_frames - 1), "Incorrect number of frames."
        else:
            raise Exception("Invalid sampling argument.")

        return t_index, interp_idx

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        """

        :param idx: index of sample clip in dataset
        :return: Gets the required sample, and ensures only 9 frames from the clip are returned.
        """
        img_paths = self.clips[idx]
        assert len(img_paths) == 7
        # start, end = self.get_start_end(img_paths)
        # img_paths = img_paths[start:end]
        # assert len(img_paths)==self.reqd_images, "Incorrect length of input sequence."
        if self.split == "TRAIN" and np.random.randint(0, 2) == 1:
            img_paths = img_paths[::-1]
        reqd_idx, _ = self.get_reqd_idx()  # handles the sampling.

        sample = read_sample(img_paths, reqd_idx)
        sample = self.custom_transform(sample)

        t_interp = [4] * (self.n_frames - 1)  # backwards compatible code.
        t_interp = torch.Tensor(t_interp).view(-1, 1, 1, 1)  # T C H W

        return sample, t_interp


class AdobeReader(Dataset):

    def __init__(self, cfg, split="TRAIN", eval_mode=False, transform=None, t_sample="FIXED"):
        """
        :param cfg: Config file.
        :param split: TRAIN/VAL/TEST
        """

        self.cfg = cfg
        self.clips = self.read_clip_list(split)
        self.split = split
        REQD_IMAGES = {2: 9, 4: 25, 6: 41, 8: 57}
        self.reqd_images = REQD_IMAGES[self.cfg.getint("TRAIN", "N_FRAMES")]
        self.eval_mode = eval_mode
        self.custom_transform = transform
        self.t_sample = t_sample
        self.n_frames = self.cfg.getint("TRAIN", "N_FRAMES")
        log.info("Launching Adobe reader with T_SAMPLE: %s N_FRAMES: %s"%(self.t_sample, self.n_frames))

    def get_start_end(self, img_paths):
        """
        gets start-end indexes for each N_FRAMES setting such that the most intermediate frames of the full sample are at the center.
        """
        assert len(img_paths) == 57, "Expected 8 frames per sample."
        n_frames = self.cfg.getint("TRAIN", "N_FRAMES")

        if self.eval_mode:
            return (0, 57)
            # if n_frames==2:
            #     return (24, 33)
            # elif n_frames==4:
            #     return (16, 41)
            # elif n_frames==6:
            #     return (8, 49)
            # elif n_frames==8:
            #     return (0, 57)
            # else:
            #     raise Exception("Incorrect number of input frames.")
        else:
            if len(img_paths) > self.reqd_images:
                start_idx = np.random.randint(0, len(img_paths) - self.reqd_images + 1)
                end_idx = start_idx + self.reqd_images
                return (start_idx, end_idx)
            else:
                return (0, 57)

    def read_clip_list(self, split):
        """
        :param split: TRAIN/VAL/TEST
        :return: list of all clips in split
        """

        fpath = self.cfg.get("ADOBE_DATA", split + "PATHS")
        with open(fpath, "r") as f:
            data = f.readlines()
            data = [d.strip() for d in data]

        clips = []

        data = [d.replace("/home/", "/mnt/nfs/work1/elm/") for d in data]
        data = [d.replace("/workspace", "") for d in data]

        for idx, d in enumerate(data):
            if len(d) <= 2:
                nframes = int(d)
                img_paths = data[idx + 1: idx + 1 + nframes]
                clips.append(img_paths)
            else:
                continue
        return clips

    def get_reqd_idx(self):
        t_sample = self.t_sample
        n_frames = self.n_frames
        if t_sample == "NIL":
            raise NotImplementedError

        elif t_sample == "FIXED":
            # get I_0, I_0.5, I_1, I_1.5, I_2, ... I_n.
            interp_idx = [4] * (n_frames - 1)

            if n_frames == 2:
                input_idx = [0, 8]
                sample_idx = [4]
            elif n_frames == 4:
                input_idx = [0, 8, 16, 24]
                sample_idx = [4, 12, 20]
            elif n_frames == 6:
                input_idx = [0, 8, 16, 24, 32, 40]
                sample_idx = [4, 12, 20, 28, 36]
            elif n_frames == 8:
                input_idx = [0, 8, 16, 24, 32, 40, 48, 56]
                sample_idx = [4, 12, 20, 28, 36, 44, 52]
            else:
                raise Exception("Wrong n_frames")

            t_index = sorted(input_idx + sample_idx)
            assert len(t_index) == (2 * n_frames - 1), "Incorrect number of frames."

        elif t_sample == "RANDOM":

            interp_idx = [np.random.randint(1, 8)] * (n_frames - 1)
            # used to calculate t_interp.
            sample_idx = [t + i * 8 for i, t in enumerate(interp_idx)]
            # add frame offset.

            if n_frames == 2:
                input_idx = [0, 8]
            elif n_frames == 4:
                input_idx = [0, 8, 16, 24]
            elif n_frames == 6:
                input_idx = [0, 8, 16, 24, 32, 40]
            elif n_frames == 8:
                input_idx = [0, 8, 16, 24, 32, 40, 48, 56]
            else:
                raise Exception("Unsupported n_frames %s" % n_frames)

            t_index = sorted(input_idx + sample_idx)
            assert len(t_index) == (2 * n_frames - 1), "Incorrect number of frames."

        else:
            raise Exception("Invalid sampling argument.")

        return t_index, interp_idx

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        """

        :param idx: index of sample clip in dataset
        :return: Gets the required sample, and ensures only 9 frames from the clip are returned.
        """
        img_paths = self.clips[idx]
        start, end = self.get_start_end(img_paths)
        img_paths = img_paths[start:end]
        assert len(img_paths) == self.reqd_images, "Incorrect length of input sequence."
        if self.split == "TRAIN" and np.random.randint(0, 2) == 1:
            img_paths = img_paths[::-1]
        reqd_idx, interp_idx = self.get_reqd_idx()  # handles the sampling.

        sample = read_sample(img_paths, reqd_idx)
        sample = self.custom_transform(sample)

        assert interp_idx is not None
        interp_idx = torch.Tensor(interp_idx).view(-1, 1, 1, 1)  # T C H W
        return sample, interp_idx


class CombinedReader(Dataset):
    def __init__(self, cfg, split="TRAIN", eval_mode=False, transform=None):
        """
        :param cfg: Config file.
        :param split: TRAIN/VAL/TEST
        """

        self.cfg = cfg
        t_sample = self.cfg.get("MISC", "T_SAMPLE")
        self.adobe_reader = AdobeReader(cfg, split, eval_mode, transform, t_sample)
        self.vimeo_reader = VimeoReader(cfg, split, eval_mode, transform, t_sample="FIXED")

        n_adobe = len(self.adobe_reader.clips)
        n_vimeo = len(self.vimeo_reader.clips)

        log.info("Adobe: %s clips. Vimeo: %s clips. " %(n_adobe, n_vimeo))

        self.clips = self.generate_combined_clips(n_adobe, n_vimeo)
        assert len(self.clips) == n_vimeo + n_adobe
        log.info("Using the combined Adobe and Vimeo Readers.")
        log.info("Total length: %s"%len(self.clips))

    def generate_combined_clips(self, n_adobe, n_vimeo):

        adobe_list = [("adobe", i) for i in range(n_adobe)]
        vimeo_list = [("vimeo", i) for i in range(n_vimeo)]

        combined_list = vimeo_list + adobe_list
        return combined_list

    def __getitem__(self, idx):
        sample_dataset, sample_idx = self.clips[idx]
        if sample_dataset == "adobe":
            return self.adobe_reader[sample_idx]
        elif sample_dataset == "vimeo":
            return self.vimeo_reader[sample_idx]

    def __len__(self):
        return len(self.clips)


def read_sample(img_paths, t_index=None):
    if t_index:
        img_paths = [img_paths[idx] for idx in t_index]

    img = cv2.imread(img_paths[0])
    h, w, c = img.shape
    frames = np.zeros([len(img_paths), h, w, c])  # images are sometimes flipped for vertical videos.

    for idx, fpath in enumerate(img_paths):
        img = cv2.imread(fpath)
        frames[idx, ...] = img[..., ::-1]  # BGR -> RGB

    if h == 1280 and w == 720:  # vertical video. W = 720, H =1280
        frames = frames.swapaxes(1, 2)

    return frames


def get_transform(config, split, eval_mode):
    pix_mean = config.get('MODEL', 'PIXEL_MEAN').split(',')
    pix_mean = [float(p) for p in pix_mean]
    pix_std = config.get('MODEL', 'PIXEL_STD').split(',')
    pix_std = [float(p) for p in pix_std]

    if eval_mode:
        custom_transform = transforms.Compose([Normalize(pix_mean, pix_std),
                                               ToTensor(), EvalPad(torch.nn.ZeroPad2d([0, 0, 8, 8]))])
    elif split == "VAL" and not eval_mode:
        crop_imh = config.getint('VAL', 'CROP_IMH')
        crop_imw = config.getint('VAL', 'CROP_IMW')
        custom_transform = transforms.Compose([
            RandomCrop((crop_imh, crop_imw)),
            Normalize(pix_mean, pix_std),
            ToTensor()
        ])

    elif split == "TRAIN" and not eval_mode:
        crop_imh = config.getint('TRAIN', 'CROP_IMH')
        crop_imw = config.getint('TRAIN', 'CROP_IMW')
        custom_transform = transforms.Compose([
            RandomCrop((crop_imh, crop_imw)),
            AugmentData(),
            Normalize(pix_mean, pix_std),
            ToTensor()
        ])
    else:
        raise Exception("Invalid Split: %s" % split)

    return custom_transform


def data_generator(config, split, eval_mode=False):
    if eval_mode:
        assert config.get("MISC", "T_SAMPLE") == "NIL", "Invalid sampling argument for eval mode."

    custom_transform = get_transform(config, split, eval_mode)

    batch_size = config.getint(split, "BATCH_SIZE")
    n_workers = config.getint("MISC", "N_WORKERS")

    dataset = CombinedReader(config, split, eval_mode, custom_transform)

    n_frames = config.getint("TRAIN", "N_FRAMES")
    log.info("Model trained with %s frame input." % n_frames)

    shuffle_flag = not eval_mode
    log.info("Shuffle: %s" % shuffle_flag)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag, num_workers=n_workers,
                              worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed() % (2 ** 32 - 1))),
                              pin_memory=True)

    return data_loader


if __name__ == '__main__':
    import configparser
    import logging
    from argparse import ArgumentParser
    import numpy as np

    parser = ArgumentParser()
    parser.add_argument("--log")
    parser.add_argument("--config")  # config
    args = parser.parse_args()

    logging.basicConfig(filename=args.log, level=logging.INFO)

    config = configparser.RawConfigParser()
    # config.read("/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/SuperSloMo/utils/ssmr.ini")
    config.read(args.config)
    logging.info("Read config")

    import time

    total = 0

    for epoch in range(10):
        samples = data_generator(config, "TRAIN")
        tic = time.time()
        for idx, x in enumerate(samples):
            batch, t_idx = x
            log.info(batch.shape)
            if idx > 10:
                exit(0)
        toc = time.time()
        tic = time.time()
        total += toc - tic

    log.info("Average %s" % (total / 10))
