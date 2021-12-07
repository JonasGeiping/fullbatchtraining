"""Verify a given checkpoint."""

import torch
import hydra

import time
import logging

import fullbatch

import os
os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="cfg")
def main_launcher(cfg):
    fullbatch.utils.job_startup(main_process, cfg, log, 'evaluation')


def main_process(process_idx, local_group_size, cfg):
    local_time = time.time()
    setup = fullbatch.utils.system_startup(process_idx, local_group_size, cfg)

    trainloader, validloader = fullbatch.data.construct_dataloader(cfg.data, cfg.impl, cfg.hyp, cfg.dryrun)

    model = fullbatch.models.construct_model(cfg.model, cfg.data.channels, cfg.data.classes)
    model = fullbatch.models.prepare_model(model, cfg, process_idx, setup)

    if cfg.impl.checkpoint.name is not None:
        file = os.path.join(cfg.original_cwd, 'checkpoints', cfg.impl.checkpoint.name)
        _, model_state, _, _, step = torch.load(file, map_location=setup['device'])
        model.load_state_dict(model_state)
        log.info(f'Loaded model checkpoint from step {step} successfully.')
    else:
        raise ValueError('Could not load checkpoint')

    stats = fullbatch.training.evaluate(model, validloader, None, setup, cfg.impl, cfg.hyp, dryrun=cfg.dryrun)
    log.info(f'VAL loss {stats["valid_loss"][-1]:7.4f} | VAL Acc: {stats["valid_acc"][-1]:7.2%} |')

    if cfg.impl.setup.dist:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main_launcher()
