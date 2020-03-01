import logging

from .dataloaders import adobe_240fps, combined_dataset, default_reader, slowflow, sintel_hfr, vimeo
from .validators import validate_sampling

log = logging.getLogger(__name__)


@validate_sampling
def get_dataset(config, split):
    NAME = config.get("DATA", "DATASET")
    eval_mode = config.getboolean("EVAL", "EVAL_MODE")

    log.info("Loading dataset: %s", NAME)

    if NAME == "ALL":  # NFS + ADOBE + VIMEO
        dataset_reader = combined_dataset.CombinedReader(config, split)

    elif NAME == "ADOBE":
        assert (eval_mode and split == "VAL") or (not eval_mode and split == "TRAIN")
        dataset_reader = adobe_240fps.AdobeReader(config, split, eval_mode)

    elif NAME == "VIMEO" and split == "VAL":
        dataset_reader = vimeo.VimeoReader(config, split, eval_mode=True)

    elif NAME == "SLOWFLOW" and split == "VAL":
        dataset_reader = slowflow.SlowflowReader(config, split, eval_mode)

    elif NAME == "SINTEL_HFR" and split == "VAL":
        dataset_reader = sintel_hfr.SintelHFRReader(config)

    else:
        raise Exception("Unsupported Dataset.")

    return default_reader.get_dataloader(dataset_reader)
