import glob
import cv2
import numpy as np
import random, os
from math import ceil


class Reader:

    def __init__(self, cfg, split="TRAIN"):
        self.cfg = cfg
        self.splits = self.get_splits()
        self.file_list = self.splits[split]
        self.n_frames = self.cfg.getint("TRAIN", "N_FRAMES")
        self.compute_scale_factors()

    def get_splits(self):
        project_dir = self.cfg.get("PROJECT", "DIR")
        adobe_path = self.cfg.get("ADOBE_DATA", "PATH")
        full_path = os.path.join(project_dir, adobe_path)
        file_list = glob.glob(os.path.join(full_path,"*"))
        print "ADOBE 240FPS: Using videos from:"
        print adobe_path
        print "Found "+str(len(file_list))+" videos."

        train_list = file_list[:93]
        test_list = file_list[93:113]
        val_list = file_list[113:]

        random.shuffle(train_list)
        random.shuffle(test_list)
        random.shuffle(val_list)

        splits = {"TRAIN":train_list, "TEST":test_list, "VAL":val_list}
        return splits

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
        # TODO: shuffle the list in the generator
        """

        """
        https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python
        https://stackoverflow.com/questions/2974625/opencv-seek-function-rewind
        """

        assert self.n_frames>0, "Number of frames as input has to be positive value"

        for aVideo in self.file_list:
            cap = cv2.VideoCapture(aVideo)

            ret, img = cap.read()

            H, W, C = img.shape
            H = H/2
            W = W/2

            frame_buffer = np.zeros([self.n_frames, H, W, C])

            current_idx = 0

            frame_buffer[current_idx, ...] = cv2.resize(img, (W, H))
            count = 0
            while ret:
                current_idx+=1
                if count == self.n_frames-1:
                    h_start = random.randint(0, H-self.H)
                    frame_buffer = frame_buffer[:, h_start:h_start+self.H, :, :]
                    yield frame_buffer
                    frame_buffer = np.zeros([self.n_frames, H, W, C])
                    current_idx = 0

                ret, img = cap.read(cv2.CAP_PROP_POS_FRAMES)
                if ret:
                    count+=1
                    frame_buffer[count, ...] = cv2.resize(img, (W, H))

if __name__ == '__main__':
    import ConfigParser
    config = ConfigParser.RawConfigParser()
    config.read("../../config.ini")
    adobe_dataset = Reader(config)
    for aClip in adobe_dataset.get_clips():
        print(aClip.shape)
        for idx in range(aClip.shape[0]):
            img = aClip[idx,...]
            cv2.imwrite("Sample_"+str(idx)+".png", img)
        exit(0)


