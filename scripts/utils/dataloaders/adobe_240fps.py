import glob
import logging
import os
import pickle

from . import default_reader

log = logging.getLogger(__name__)


class AdobeReader(default_reader.Reader):
    def __init__(self, cfg, split="TRAIN", eval_mode=False):
        super(AdobeReader, self).__init__(cfg, split, eval_mode)
        self.custom_transform = self.get_torchvision_transform()
        if not eval_mode:
            self.clips = self.read_train_clip_list()
        else:
            self.clips = self.read_inference_clip_list()

    def read_train_clip_list(self):
        """
        :return: list of all clips in split
        """

        fpath = self.cfg.get("ADOBE_DATA", self.split + "PATHS")
        with open(fpath, "r") as f:
            data = f.readlines()
            data = [d.strip() for d in data]
            # data = [os.path.join(self.cfg.get("ADOBE_DATA", "ROOTDIR"), d) for d in data if len(d)>2]
        clips = []

        for idx, d in enumerate(data):
            if len(d) <= 2:
                nframes = int(d)
                img_paths = data[idx + 1 : idx + 1 + nframes]
                clips.append(img_paths)
            else:
                continue
        return clips

    def read_inference_clip_list(self):
        """
        For inference, we want all possible interpolation windows in the 30FPS input.
        We start with a 240FPS video, and sample every 9 frame sequence.

        :returns: a list of clips
        :rtype:

        """

        clips_src = self.cfg.get("ADOBE_DATA", self.split + "_clips")

        with open(clips_src, "rb") as f:
            clips = pickle.load(f)
        log.info("Found %s clips for split: %s ", len(clips), self.split)

        SRC_DIR = self.cfg.get("ADOBE_DATA", "ROOTDIR")

        data = []

        for clip in sorted(clips):
            clip_dir = os.path.join(SRC_DIR, clip)
            img_paths = glob.glob(clip_dir + "/*.png")
            img_paths = sorted(img_paths)
            log.info("Found %s frames in: %s", len(img_paths), clip)
            for sample in self.generate_sliding_windows(img_paths):
                data.append(sample)

        log.info("Total: %s", len(data))

        return data
