import glob, logging
import cv2
import numpy as np
import random, os
from math import ceil


log = logging.getLogger(__name__)

class Reader:

    def __init__(self, cfg, split="TRAIN"):
        self.cfg = cfg
        self.batch_size = self.cfg.getint("TRAIN", "BATCH_SIZE")
        self.compute_scale_factors()
        self.clips = self.read_clip_list(split)
        self.split = split
        log.info(split+ ": Extracted clip list.")

    def read_clip_list(self, split):
        fpath = self.cfg.get("ADOBE_DATA", split+"PATHS")
        with open(fpath, "rb") as f:
            data = f.readlines()
            data = [d.strip() for d in data]

        clips = {}

        data = [d.replace("/home/", "/mnt/nfs/work1/elm/") for d in data]
        data = [d.replace("/workspace", "") for d in data]

        for idx, d in enumerate(data):
            if len(d)<=2:
                nframes = int(d)
                img_paths = data[idx + 1 : idx + 1 + nframes]
                if nframes in clips.keys():
                    clips[nframes].append(img_paths)

                else:
                    clips[nframes]=[img_paths]
            else:
                continue

        return clips



    def compute_scale_factors(self):
        self.H = self.cfg.getint("ADOBE_DATA", "H")
        self.W = self.cfg.getint("ADOBE_DATA", "W")
        self.dims = (self.H, self.W)

        divisor = 64.

        H_ = int(ceil(self.H / divisor) * divisor)
        W_ = int(ceil(self.W / divisor) * divisor)

        self.s_x = float(self.W) / W_
        self.s_y = float(self.H) / H_

        self.scale_factors= (self.s_y, self.s_x)


    def get_clips(self):
        """
        Generator that can yield clips from video, each clip is of size n_frames.

        numpy array, BGR format. uint8.

        :param video_path: full path to video
        :param n_frames: number of frames to extract for each clip
        :return:
        """
        if self.split=="TRAIN":
            clips_list = self.clips[12]
        elif self.split=="VAL":
            clips_list = self.clips[9]

        log.info(self.split+ ": Running generator for extracting clips.")


        frame_buffer = np.zeros([self.batch_size, 9, self.H, self.W, 3])
        count = 0

        while True:
            clip_idx = random.randint(0, len(clips_list)-1) # random clip id
            clip = clips_list[clip_idx]
            start_idx = random.randint(0, len(clip)-9) # random starting point to get subset of 9 frames.
            img_paths = clip[start_idx: start_idx +9]
            images = [cv2.imread(fpath) for fpath in img_paths]
            images = [cv2.resize(image, (640, 360)) for image in images]
            images = np.array(images)

            h_start = random.randint(0, 360-self.H) # random height crop.
            images = images[:, h_start:h_start+self.H, ...]
            frame_buffer[count, ...] = images

            count+=1
            if count == self.batch_size:
                yield frame_buffer
                frame_buffer = np.zeros([self.batch_size, 9, self.H, self.W, 3])
                count = 0



if __name__ == '__main__':
    import ConfigParser
    config = ConfigParser.RawConfigParser()
    config.read("../../config.ini")
    adobe_dataset = Reader(config, split="TRAIN")
    for aClip in adobe_dataset.get_clips():
        np.save("Clips.npy", aClip)
        log.info(""+str(aClip.shape))
        exit(0)
