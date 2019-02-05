import sys
sys.path.insert(0, "/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/SuperSloMo/utils/")
import numpy as np
import logging
import cv2
from math import ceil
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from common import (AugmentData, ResizeCrop, EvalPad, Normalize, ToTensor)


cv2.setNumThreads(0)
log = logging.getLogger(__name__)


class Reader(Dataset):

    def __init__(self, cfg, split="TRAIN", eval_mode=False):
        """
        :param cfg: Config file.
        :param split: TRAIN/VAL/TEST
        """

        self.cfg = cfg
        # self.compute_scale_factors()
        self.clips = self.read_clip_list(split)
        self.split = split
        self.reqd_images = 8 * (self.cfg.getint("TRAIN","N_FRAMES") -1 )+ 1
        # log.info(split+ ": Extracted clip list.")
        self.eval_mode = eval_mode

    def read_clip_list(self, split):
        """
        :param split: TRAIN/VAL/TEST
        :return: list of all clips in split
        """

        fpath = self.cfg.get("ADOBE_DATA", split+"PATHS")
        with open(fpath, "r") as f:
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
        """
        scale factors required for PWC Net.
        :return:
        """

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
        """

        :param idx: index of sample clip in dataset
        :return: Gets the required sample, and ensures only 9 frames from the clip are returned.
        """

        img_paths = self.clips[idx]
        if len(img_paths)>self.reqd_images and not self.eval_mode:
            start_idx = np.random.randint(0, len(img_paths)- self.reqd_images + 1)
            img_paths = img_paths[start_idx:start_idx+self.reqd_images]
        elif len(img_paths)>self.reqd_images and self.eval_mode:
            mid_idx = len(img_paths)//2
            start = mid_idx - self.reqd_images//2
            end = start + self.reqd_images
            img_paths = img_paths[start:end]
        assert len(img_paths)==self.reqd_images, "Incorrect length of input sequence."
        if self.split=="TRAIN" and np.random.randint(0, 2)==1:
            img_paths = img_paths[::-1]

        return img_paths


def collate_data(aBatch, custom_transform, t_sample, n_frames):
    """
    :param aBatch: List[List] B samples of 8 frames (frames given as paths)
    :param custom_transform: torchvision transform
    :param t_sample: NIL => No sampling. Read all frames.
                     FIXED => Fixed sampling of middle frame. t_index= 4
                     RANDOM => Uniform random sampling from 1, 7.
    :return: tensor N, K, C, H, W and index of time step sampled (None, or int)
    """

    if t_sample=="NIL":
        t_index = [i*8 for i in range(n_frames)]
        mid_idx = int(np.mean(t_index))
        t1 = mid_idx - 3
        t2 = t1 + 7
        t_index.extend(range(t1, t2)) # intermediate frames to be interpolated.
        t_index = sorted(t_index)
        
    elif t_sample=="FIXED":
        # get I_0, I_0.5, I_1, I_1.5, I_2, ... I_n.
        if n_frames == 2:
            input_idx = [0, 8]
            interp_idx = [4]
        elif n_frames == 4:
            input_idx = [0, 8, 16, 24]
            interp_idx = [4, 12, 20]
        elif n_frames == 6:
            input_idx = [0, 8, 16, 24, 32, 40]
            interp_idx =  [4, 12, 20, 28, 36]
        elif n_frames == 8:
            input_idx = [0, 8, 16, 24, 32, 40, 48, 56]
            interp_idx =  [4, 12, 20, 28, 36, 44, 52]

        t_index = sorted(input_idx + interp_idx)
        assert len(t_index)==(2*n_frames - 1), "Incorrect number of frames."

    elif t_sample=="RANDOM":
        raise NotImplementedError
        t_index = np.random.randint(1, 8) #uniform sampling
        t_index = [0, 8, 8+t_index, 16, 24]
    else:
        raise Exception("Invalid sampling argument.")

    frame_buffer = [read_sample(sample, t_index) for sample in aBatch]
    
    for idx, a_buffer in enumerate(frame_buffer):
        frame_buffer[idx] = custom_transform(a_buffer)

    frame_buffer = torch.stack(frame_buffer)

    if t_sample=="FIXED":
        return frame_buffer, 4 # middle index. lol such bad code.

    else:
        return frame_buffer, None

    
def read_sample(img_paths, t_index=None):
    if t_index:
        img_paths = [img_paths[idx] for idx in t_index]
        
    img = cv2.imread(img_paths[0])
    h, w, c = img.shape
    frames = np.zeros([len(img_paths), h, w, c])  # images are sometimes flipped for vertical videos.

    for idx, fpath in enumerate(img_paths):
        img = cv2.imread(fpath)
        frames[idx,...] = img[..., ::-1] # BGR -> RGB

    if h==1280 and w==720: # vertical video. W = 720, H =1280
        frames = frames.swapaxes(1, 2)

    return frames


def get_transform(config, split, eval):

    pix_mean = config.get('MODEL', 'PIXEL_MEAN').split(',')
    pix_mean = [float(p) for p in pix_mean]
    pix_std = config.get('MODEL', 'PIXEL_STD').split(',')
    pix_std = [float(p) for p in pix_std]

    if eval:
        custom_transform = transforms.Compose([Normalize(pix_mean, pix_std),
                                               ToTensor(), EvalPad(torch.nn.ZeroPad2d([0,0,8,8]))])
    elif split == "VAL":
        crop_imh = config.getint('VAL', 'CROP_IMH')
        crop_imw = config.getint('VAL', 'CROP_IMW')
        custom_transform = transforms.Compose([
            ResizeCrop(crop_imh, crop_imw),
            Normalize(pix_mean, pix_std),
            ToTensor()
        ])

    elif split == "TRAIN":
        crop_imh = config.getint('TRAIN', 'CROP_IMH')
        crop_imw = config.getint('TRAIN', 'CROP_IMW')
        custom_transform = transforms.Compose([
            ResizeCrop(crop_imh, crop_imw),
            AugmentData(),
            Normalize(pix_mean, pix_std),
            ToTensor()
        ])
    else:
        raise Exception("Invalid Split: %s"%split)

    return custom_transform


def data_generator(config, split, eval=False):

    if eval:
        assert config.get("MISC", "T_SAMPLE") == "NIL", "Invalid sampling argument for eval mode."

    custom_transform = get_transform(config, split, eval)
    t_sample = config.get("MISC", "T_SAMPLE")

    batch_size = config.getint(split, "BATCH_SIZE")
    n_workers = config.getint("MISC", "N_WORKERS")

    dataset = Reader(config, split, eval)

    n_frames = config.getint("TRAIN", "N_FRAMES")
    log.info("Model trained with %s frame input."%n_frames)

    adobe_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers,
                              worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()%(2**32 -1))),
                              collate_fn=lambda batch: collate_data(batch, custom_transform, t_sample, n_frames))

    for batch_sample in adobe_loader:        
        yield batch_sample


def get_data_info(config, split):
    # dataset = Reader(config, split)
    # return dataset.dims, dataset.scale_factors
    log.warning("This function should not be called.")
    return None, None


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
    config.read(args.config)
    logging.info("Read config")
    import time
    total = 0
    
    for epoch in range(10):
        samples = data_generator(config, "TRAIN")
        tic = time.time()
        for x in samples:
            pass
        toc = time.time()
        tic = time.time()
        total += toc - tic

    log.info("Average %s"%(total/10))
