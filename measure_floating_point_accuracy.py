"""Verify a given checkpoint."""

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
    fullbatch.utils.job_startup(main_process, cfg, log, "floating-point evaluation")


def main_process(process_idx, local_group_size, cfg):
    local_time = time.time()
    setup = fullbatch.utils.system_startup(process_idx, local_group_size, cfg)

    trainloader, validloader = fullbatch.data.construct_dataloader(cfg.data, cfg.impl, cfg.hyp, cfg.dryrun)

    model = fullbatch.models.construct_model(cfg.model, cfg.data.channels, cfg.data.classes)
    model = fullbatch.models.prepare_model(model, cfg, process_idx, setup)

    fullbatch.training.training._measure_implementation_noise(model, trainloader, validloader, setup, cfg)

    if cfg.impl.setup.dist:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main_launcher()
