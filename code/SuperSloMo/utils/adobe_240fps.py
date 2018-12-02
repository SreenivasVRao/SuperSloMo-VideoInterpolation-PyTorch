import numpy as np
import logging
import cv2
from math import ceil
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


log = logging.getLogger(__name__)


class Reader(Dataset):

    def __init__(self, cfg, split="TRAIN"):
        """
        :param cfg: Config file.
        :param split: TRAIN/VAL/TEST
        """

        self.cfg = cfg
        self.compute_scale_factors()
        self.clips = self.read_clip_list(split)
        self.split = split
        # log.info(split+ ": Extracted clip list.")

    def read_clip_list(self, split):
        """
        :param split: TRAIN/VAL/TEST
        :return: list of all clips in split
        """

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

    def get_required_images(self, image_list):
        """
        :param image_list: list of  frames in clip
        :return: returns sample of 9 frames.
        """
        n_frames = len(image_list)

        if self.split=="TRAIN" and np.random.randint(0, 2)==1:
            image_list = image_list[::-1]

        if n_frames == 9:
            pass

        elif n_frames < 12:
            start_idx = np.random.randint(0, n_frames - 9+1) # random starting point to get 9 frames.
            image_list = image_list[start_idx: start_idx + 9]

        elif n_frames >= 12:
            subset = np.random.randint(0, n_frames-12+1) # random starting point to get subset of 12 frames.
            image_list = image_list[subset:subset+12]

            start_idx = np.random.randint(0, 3+1) # random starting point to get subset of 9 frames.
            image_list = image_list[start_idx:start_idx+9]

        assert len(image_list)==9, "Something went wrong."

        return image_list

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
        img_paths = self.get_required_images(img_paths)
    
        return img_paths

    
class AugmentData(object):
    """
    Flips the images horizontally 50% of the time.
    Performs a random rotation of the data.
    """

    def __call__(self, frames):
        """
        :param frames: N H W C array.
        :return: same array after augmentation.
        """
        if np.random.randint(0, 2)==1:
            frames = frames[:,:,::-1,:] # horizontal flip 50% of the time

        N, H, W, C = frames.shape

        cx = np.random.randint(0, W)
        cy = np.random.randint(0, H)
        theta = np.random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((cx, cy),theta,1)
        # rotate around random center.
 
        for idx in range(N):
            img = frames[idx,...]
            frames[idx,...] = cv2.warpAffine(img, M, (W, H))

        return frames

    
class ResizeCrop(object):
    """
    Convert 720 x 1280 frames to 352 x 352 -> Resize + Random Cropping
    """

    def __call__(self, sample_frames):

        _, h, w, c = sample_frames.shape
        assert h==720 and w==1280, "invalid dimensions"

        new_frames = np.zeros((sample_frames.shape[0], 360, 640, 3))

        for idx in range(sample_frames.shape[0]):
            new_frames[idx, ...] = cv2.resize(sample_frames[idx, ...], (640, 360))
        h_start = np.random.randint(0, 360-352+1)
        w_start = np.random.randint(0, 640-352+1)

        new_frames = new_frames[:, h_start:h_start+352, w_start:w_start+352, ...]

        return new_frames

    
class ToTensor(object):
    """
    Converts np 0-255 uint8 to 0-1 tensor
    """
    def __call__(self, sample):
        sample = torch.from_numpy(sample.copy())/255.0
        sample = sample.permute(0, 3, 1, 2) # n_frames, H W C -> n_frames, C, H, W
        return sample


def collate_data(aBatch, custom_transform, t_sample):
    """
    :param aBatch: List[List] B samples of 8 frames (frames given as paths)
    :param custom_transform: torchvision transform
    :param t_sample: NIL => No sampling. Read all frames.
                     FIXED => Fixed sampling of middle frame. t_index= 4
                     RANDOM => Uniform random sampling from 1, 7.
    :return: tensor N, K, C, H, W and index of time step sampled (None, or int)
    """

    if t_sample=="NIL":
        t_index = None
    elif t_sample=="FIXED":
        t_index = 4
    elif t_sample=="RANDOM":
        t_index = np.random.randint(1, 8) #uniform sampling
    else:
        raise Exception, "Invalid sampling argument."

    frame_buffer = [read_sample(sample, t_index) for sample in aBatch]
    
    for idx, a_buffer in enumerate(frame_buffer):
        frame_buffer[idx] = custom_transform(a_buffer)

    frame_buffer = torch.stack(frame_buffer)
    
    return frame_buffer, t_index

    
def read_sample(img_paths, t_index=None):
    if t_index:
        img_paths = [img_paths[0], img_paths[t_index], img_paths[-1]]
        
    img = cv2.imread(img_paths[0])
    h, w, c = img.shape
    frames = np.zeros([len(img_paths), h, w, c])  # images are sometimes flipped for vertical videos.

    for idx, fpath in enumerate(img_paths):
        img = cv2.imread(fpath)
        frames[idx,...] = img

    if h==1280 and w==720: # vertical video. W = 720, H =1280
        frames = frames.swapaxes(1, 2)

    return frames


def data_generator(config, split, eval=False):

    if eval:
        custom_transform = transforms.Compose([ResizeCrop(), ToTensor()])
        t_sample = "NIL"
    else:
        custom_transform = transforms.Compose([ResizeCrop(), AugmentData(), ToTensor()])
        t_sample = config.get("MISC", "T_SAMPLE")

    batch_size = config.getint(split, "BATCH_SIZE")
    n_workers = config.getint("MISC", "N_WORKERS")

    dataset = Reader(config, split)

    adobe_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers,
                              worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed()%(2**32 -1))),
                              collate_fn = lambda batch: collate_data(batch, custom_transform, t_sample))

    for batch_sample in adobe_loader:        
        yield batch_sample


def get_data_info(config, split):
    dataset = Reader(config, split)
    return dataset.dims, dataset.scale_factors



if __name__ == '__main__':
    import ConfigParser
    import logging
    from argparse import ArgumentParser
    import numpy as np
    
    parser = ArgumentParser()
    parser.add_argument("--log")
    parser.add_argument("--config") # config
    args = parser.parse_args()


    logging.basicConfig(filename=args.log, level=logging.INFO)

    config = ConfigParser.RawConfigParser()
    config.read(args.config)
    logging.info("Read config")

    samples = data_generator(config, "TRAIN")
    aBatch, t_idx = next(samples)
    log.info(aBatch.shape)
    log.info(t_idx)
    import time
    start = time.time()
    k = 0
    for epoch in range(10):
        log.info("Epoch: %s K: %s"%(epoch, k))
        for aBatch, t_idx in samples:
            k+=1
        
    stop = time.time()
    total = (stop-start)
    average = total/k
    log.info("Total: %.2f seconds"%total)
    log.info("Average: %.2f seconds"%average)

    start = time.time()
    k = 0
    for epoch in range(10):
        log.info("Epoch: %s"%epoch)
        samples = data_generator(config, "TRAIN")
        for aBatch, t_idx in samples:
            k+=1
        
    stop = time.time()
    total = (stop-start)
    average = total/k
    log.info("Total: %.2f seconds"%total)
    log.info("Average: %.2f seconds"%average)
    
