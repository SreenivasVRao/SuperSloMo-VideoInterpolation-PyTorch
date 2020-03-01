from . import default_reader


class NFSReader(default_reader.Reader):
    def __init__(self, cfg, split="TRAIN", eval_mode=False):
        super(NFSReader, self).__init__(cfg, split, eval_mode)

    def read_train_clip_list(self):
        """
        :param split: TRAIN/VAL/TEST
        :return: list of all clips in split
        """

        fpath = self.cfg.get("NFS_DATA", "TRAINPATHS")
        with open(fpath, "r") as f:
            data = f.readlines()
            data = [d.strip() for d in data]
            # data = [os.path.join(self.cfg.get("NFS_DATA", "ROOTDIR"), d) for d in data if len(d)>2]
        clips = []

        for idx, d in enumerate(data):
            if len(d) <= 2:
                nframes = int(d)
                img_paths = data[idx + 1 : idx + 1 + nframes]
                clips.append(img_paths)
            else:
                continue
        return clips
