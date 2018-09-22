import glob
import cv2
import numpy as np
import random, os


class Reader:

    def __init__(self, cfg):
        self.cfg = cfg

        project_dir = self.cfg.get("PROJECT", "DIR")

        adobe_path = self.cfg.get("ADOBE_DATA", "PATH")

        self.full_path = os.path.join(project_dir, adobe_path)
        self.file_list = glob.glob(os.path.join(self.full_path,"*"))
        self.batch_size = self.cfg.getint("TRAIN", "BATCH_SIZE")
        self.n_frames = self.cfg.getint("TRAIN", "N_FRAMES")
        print "Using videos from:"
        print self.full_path
        print "Found "+str(len(self.file_list))+" videos."

    def get_clips(self):

        """
        Generator that can yield clips from video, each clip is of size n_frames.

        numpy array, BGR format. uint8.

        :param video_path: full path to video
        :param n_frames: number of frames to extract for each clip
        :return:

        """

        """
        https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python
        https://stackoverflow.com/questions/2974625/opencv-seek-function-rewind
        """

        assert self.n_frames>0, "Number of frames as input has to be positive value"

        for aVideo in self.file_list:
            cap = cv2.VideoCapture(aVideo)

            ret, img = cap.read()

            H = self.cfg.getint("ADOBE_DATA", "H_IMG")
            W = self.cfg.getint("ADOBE_DATA", "W_IMG")
            C = 3

            frame_buffer = np.zeros([self.n_frames, H, W, C])

            current_idx = 0

            frame_buffer[current_idx, ...] = cv2.resize(img, (W, H))
            cv2.imwrite("Sample.png", frame_buffer[current_idx, ...])
            cv2.imwrite("Sample_original.png", img)

            while ret:
                current_idx+=1
                if current_idx == self.n_frames:
                    # W_max = self.cfg.getint("TRAIN", "W")
                    H_max = self.cfg.getint("TRAIN", "H")

                    h_start = random.randint(0, H - H_max)

                    frame_buffer = frame_buffer[:, h_start:h_start + H_max, :, :]

                    yield frame_buffer
                    frame_buffer = np.zeros([self.n_frames, H, W, C])
                    current_idx = 0

                ret, img = cap.read()
                if ret:
                    frame_buffer[current_idx, ...] = cv2.resize(img, (W, H))


def DataReader(config):
    adobe_reader = Reader(config)
    # does not return PDFs!
    # I hope the dataset's pdfs are fine though.
    # #ReallyBadStatsJokes

    return adobe_reader


if __name__ == '__main__':
    import ConfigParser

    config = ConfigParser.RawConfigParser()
    config.read("../../config.ini")

    adobe_dataset = DataReader(config)
    for aClip in adobe_dataset.get_clips():
        print aClip.shape
        exit(0)
        # if aClip.shape!=(12, 448, 1024,3):
        #     exit(0)

