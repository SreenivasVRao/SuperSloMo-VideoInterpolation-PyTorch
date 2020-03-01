from . import default_reader
from .adobe_240fps import AdobeReader
from .nfs import NFSReader
from .vimeo import VimeoReader


import logging

log = logging.getLogger(__name__)


class CombinedReader(default_reader.Reader):
    def __init__(self, cfg, split="TRAIN", eval_mode=False):
        """
        :param cfg: Config file.
        :param split: TRAIN/VAL/TEST
        """
        super(CombinedReader, self).__init__(cfg, split, eval_mode)

        self.adobe_reader = AdobeReader(cfg, split, eval_mode)
        self.nfs_reader = NFSReader(cfg, split, eval_mode)
        self.vimeo_reader = VimeoReader(cfg, split, eval_mode)

        n_adobe = len(self.adobe_reader.clips)
        n_nfs = len(self.nfs_reader.clips)
        n_vimeo = len(self.vimeo_reader.clips)

        log.info("Adobe: %s clips. NFS: %s clips. Vimeo: %s clips. " % (n_adobe, n_nfs, n_vimeo))

        self.clips = self.generate_dataset_indexes(n_adobe, n_nfs, n_vimeo)
        assert len(self.clips) == n_nfs + n_adobe + n_vimeo
        log.info("Using the combined Adobe, Vimeo and NFS Readers.")
        log.info("Total length: %s" % len(self.clips))

    def read_train_clip_list(self):
        """
        Place holder function to override the base function.
        Clips are read by individual readers.
        """
        log.info("No-op for reading train clips in CombinedReader.")
        pass

    def generate_dataset_indexes(self, n_adobe, n_nfs, n_vimeo):
        combined_list = []

        combined_list += [("adobe", i) for i in range(n_adobe)]
        combined_list += [("nfs", i) for i in range(n_nfs)]
        combined_list += [("vimeo", i) for i in range(n_vimeo)]

        return combined_list

    def __getitem__(self, idx):
        sample_dataset, sample_idx = self.clips[idx]
        if sample_dataset == "adobe":
            return self.adobe_reader[sample_idx]
        elif sample_dataset == "nfs":
            return self.nfs_reader[sample_idx]
        elif sample_dataset == "vimeo":
            return self.vimeo_reader[sample_idx]
