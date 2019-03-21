import sys
sys.path.insert(0, "/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/SuperSloMo/utils/")
import numpy as np
import logging
import os, glob
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
from common import (AugmentData, EvalPad, Normalize, ToTensor, RandomCrop)


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


if __name__ == '__main__':
    import configparser
    import logging
    from argparse import ArgumentParser
    import numpy as np
    import sys
    sys.path.insert(0, "/home/sreenivasv/CS701/SuperSloMo-PyTorch/code/SuperSloMo/models/")
    import SSMR
    from skimage.measure import compare_psnr, compare_ssim

    def denormalize(batch):
        pix_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda().float()
        pix_std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda().float()
        batch = batch * pix_std + pix_mean
        batch = batch * 255.0
        return batch

    def eval_single_image(tensor1, tensor2):
        image1 = tensor1.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
        image2 = tensor2.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
        assert image1.shape[0]==image2.shape[0]==1
        image1 = image1[0, ...]
        image2 = image2[0, ...]
        PSNR = compare_psnr(image1, image2)
        SSIM = compare_ssim(image1, image2,
                            multichannel=True,
                            gaussian_weights=True)
        IE = image1.astype(float) - image2.astype(float)
        IE = np.mean(np.sqrt(np.sum(IE * IE, axis=2)))
        return PSNR, SSIM, IE

    parser = ArgumentParser()
    parser.add_argument("--log")
    parser.add_argument("-c", "--config")  # config
    args = parser.parse_args()

    logging.basicConfig(filename=args.log, level=logging.INFO)

    config = configparser.RawConfigParser()
    config.read(args.config)
    logging.info("Read config")

    PSNR_scores = []
    SSIM_scores = []
    IE_scores = []
    

    n_frames = config.getint("TRAIN", "N_FRAMES")

    vimeo_samples = data_generator(config, split="VAL", eval_mode=True)
    superslomo = SSMR.full_model(config).cuda().eval()
    
    t_interp = torch.Tensor([0.5]* (n_frames-1)).cuda().float()
    t_interp = t_interp.view(1, -1, 1, 1, 1)

    # data = []

    for idx, (input_tensor, seq, pos) in enumerate(vimeo_samples):
        if n_frames == 4:
            image_tensor = input_tensor[:, [0, 1, 3, 4], ...].cuda().float() # B 5 C H W -> B 4 C H W
            target_tensor = input_tensor[:, 2, ...].cuda().float()
        elif n_frames == 2:
            image_tensor = input_tensor[:, [0, 2], ...].cuda().float() # B 3 C H W -> B 2 C H W
            target_tensor = input_tensor[:, 1, ...].cuda().float()
       
        t_interp = t_interp.expand(image_tensor.shape[0], n_frames-1, 1, 1, 1)  # for multiple gpus.
        est_img_t = superslomo(image_tensor, None, t_interp, iteration=None, compute_loss=False)

        est_img_t = denormalize(est_img_t)
        target_tensor = denormalize(target_tensor)
        psnr, ssim, ie = eval_single_image(est_img_t, target_tensor)
        PSNR_scores.append(psnr)
        SSIM_scores.append(ssim)
        IE_scores.append(ie)

        if idx%10 == 0:
            logging.info("Iteration: %s of %s"%(idx, len(vimeo_samples)))
            logging.info(image_tensor.shape)
            mean_avg_psnr = np.mean(PSNR_scores)
            mean_avg_IE = np.mean(IE_scores)
            mean_avg_SSIM = np.mean(SSIM_scores)
            logging.info("So far: PSNR: %.3f IE: %.3f SSIM: %.3f" % (mean_avg_psnr, mean_avg_IE, mean_avg_SSIM))
    #     if idx>200:
    #         break
    # data.sort(key=lambda tup: tup[0], reverse=True)

    # print(data[0:100])
    
    mean_avg_psnr = np.mean(PSNR_scores)
    mean_avg_IE = np.mean(IE_scores)
    mean_avg_SSIM = np.mean(SSIM_scores)
    logging.info("Avg. per video. PSNR: %.3f IE: %.3f SSIM: %.3f" % (mean_avg_psnr, mean_avg_IE, mean_avg_SSIM))



        
        

    

