"""Visualize a loss landscape computed via full-batch GD
This is more a less calls a rewritten wrapper of the crunch function of
https://github.com/tomgoldstein/loss-landscape

The loss landscape saved by this script is saved within an LMDB and not compatible with the original hdf-based implementation.

This CLI interface based on the hydra configuration received at startup (most prominently of cfg.viz)"""

import torch
import hydra

import time
import logging

import fullbatch

import os

os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="cfg", version_base="1.1")
def main_launcher(cfg):
    fullbatch.utils.job_startup(main_process, cfg, log, "loss landscape visualization")


def main_process(process_idx, local_group_size, cfg):
    local_time = time.time()
    setup = fullbatch.utils.system_startup(process_idx, local_group_size, cfg)

    trainloader, validloader = fullbatch.data.construct_dataloader(cfg.data, cfg.impl, cfg.dryrun)

    model = fullbatch.models.construct_model(cfg.model, cfg.data.channels, cfg.data.classes)
    model = fullbatch.models.prepare_model(model, cfg, process_idx, setup)

    fullbatch.visualization.crunch(model, trainloader, validloader, setup, cfg)

    if cfg.impl.setup.dist:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main_launcher()
