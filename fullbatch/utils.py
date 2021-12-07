"""Various utilities."""

import socket
import sys

import time
import datetime
import os
import csv

import torch
import random
import numpy as np

import hydra
from omegaconf import OmegaConf, open_dict

import logging

def job_startup(main_process, cfg, log, name=None):
    log.info('---------------------------------------------------')
    log.info(f'-----Launching {name} job! --------')

    launch_time = time.time()
    if cfg.seed is None:
        cfg.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    ngpus_per_node = torch.cuda.device_count()
    if cfg.impl.setup.dist:
        if cfg.impl.setup.url == "env://" and cfg.impl.setup.world_size == -1:
            cfg.impl.setup.world_size = int(os.environ["WORLD_SIZE"])
        # Randomize port in tcp setup
        if 'tcp://' in cfg.impl.setup.url and '?' in cfg.impl.setup.url:
            random_port = torch.randint(2000, 65535 - 1, (1,)).item()
            cfg.impl.setup.url = cfg.impl.setup.url.replace('?', str(random_port))

    log.info(OmegaConf.to_yaml(cfg))
    initialize_multiprocess_log(cfg)  # manually save log configuration

    if cfg.impl.setup.dist and ngpus_per_node > 1:
        cfg.impl.setup.world_size = ngpus_per_node * cfg.impl.setup.world_size
        # main_worker process function
        log.info(f'Distributed mode launching on {cfg.impl.setup.world_size} GPUs'
                 f' with backend {cfg.impl.setup.backend} with sharing strategy {cfg.impl.setup.strategy}')
        torch.multiprocessing.spawn(main_process, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        main_process(0, 1, cfg)

    log.info('---------------------------------------------------')
    log.info(f'Finished computations with total train time: '
             f'{str(datetime.timedelta(seconds=time.time() - launch_time))}')
    log.info('-------------Job finished.-------------------------')

def system_startup(process_idx, local_group_size, cfg):
    """Decide and print GPU / CPU / hostname info. Generate local distributed setting if running in distr. mode."""
    log = get_log(cfg)
    torch.backends.cudnn.benchmark = cfg.impl.benchmark
    torch.multiprocessing.set_sharing_strategy(cfg.impl.setup.strategy)
    # 100% reproducibility?
    if cfg.impl.deterministic:
        set_deterministic()
    if cfg.seed is not None:
        set_random_seed(cfg.seed + 10 * process_idx)

    dtype = getattr(torch, cfg.impl.dtype)  # :> dont mess this up
    memory_format = torch.contiguous_format if cfg.impl.memory == 'contiguous' else torch.channels_last

    device = torch.device(f'cuda:{process_idx}') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=dtype, memory_format=memory_format)
    python_version = sys.version.split(" (")[0]
    log.info(f'Platform: {sys.platform}, Python: {python_version}, PyTorch: {torch.__version__}')
    # log.info(torch.__config__.show())
    log.info(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')

    if torch.cuda.is_available():
        torch.cuda.set_device(process_idx)
        log.info(f'GPU : {torch.cuda.get_device_name(device=device)}')

    if not torch.cuda.is_available() and not cfg.dryrun:
        raise ValueError('No GPU allocated to this process. Training in CPU-mode is likely a bad idea. Complain to your admin.')

    if cfg.impl.setup.dist:
        if cfg.impl.setup.MASTER_PORT is not None:
            os.environ['MASTER_PORT'] = str(cfg.impl.setup.MASTER_PORT)
        if cfg.impl.setup.MASTER_ADDR is not None:
            os.environ['MASTER_ADDR'] = str(cfg.impl.setup.MASTER_ADDR)
        # Find the global cluster rank of this node
        if cfg.impl.setup.rank == 'SLURM':
            cfg.impl.setup.rank = int(os.environ["SLURM_NODEID"])
        elif cfg.impl.setup.rank == 'PBS':
            cfg.impl.setup.rank = int(os.environ["PBS_NODENUM"])
        elif cfg.impl.setup.url == "env://" and cfg.impl.setup.rank == -1:
            cfg.impl.setup.rank = int(os.environ["RANK"])  # cluster rank
        if local_group_size > 1:
            cfg.impl.setup.rank = cfg.impl.setup.rank * local_group_size + process_idx  # cluster + local rank
        torch.distributed.init_process_group(backend=cfg.impl.setup.backend, init_method=cfg.impl.setup.url,
                                             world_size=cfg.impl.setup.world_size, rank=cfg.impl.setup.rank)
        log.info(f'Distributed worker on rank {cfg.impl.setup.rank} for local size {local_group_size} '
                 f'and {cfg.impl.setup.world_size} total processes.')

    return setup


def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def get_log(cfg, name=os.path.basename(__file__)):
    """Solution via https://github.com/facebookresearch/hydra/issues/1126#issuecomment-727826513"""
    if is_main_process():
        logging.config.dictConfig(
            OmegaConf.to_container(cfg.job_logging_cfg, resolve=True)
        )
        logger = logging.getLogger(name)
    else:
        def logger(*args, **kwargs):
            pass
        logger.info = logger
    return logger


def initialize_multiprocess_log(cfg):
    with open_dict(cfg):
        # manually save log config to cfg
        log_config = hydra.core.hydra_config.HydraConfig.get().job_logging
        # but resolve any filenames
        cfg.job_logging_cfg = OmegaConf.to_container(log_config, resolve=True)
        cfg.original_cwd = hydra.utils.get_original_cwd()


def save_summary(cfg, stats, local_time):
    """Save two summary tables. A detailed table of iterations/loss+acc and a summary of the end results."""
    log = get_log(cfg)
    # 1) detailed table:
    for step in range(len(stats['train_loss'])):
        iteration = dict()
        for key in stats:
            iteration[key] = stats[key][step] if step < len(stats[key]) else None
        save_to_table('.', f'{cfg.name}_convergence_results', dryrun=cfg.dryrun, **iteration)

    def _maybe_record(key):
        if len(stats[key]) > 0:
            return stats[key][-1]
        else:
            return ''

    # 1b) Compute some extra quantities:
    max_val_acc = max(stats['valid_acc']) if len(stats['valid_acc']) > 0 else ''
    try:
        # Skip all warmup blocks and then find the block at which the full training loss is minimal
        # Use that block for validation
        warmup = cfg.hyp.warmup
        blocks_of_full_loss = np.array(stats['full_loss'][warmup:]).reshape([-1, cfg.impl.validate_every_nth_step])
        block_with_min_loss = blocks_of_full_loss.mean(axis=-1).argmin() + warmup // cfg.impl.validate_every_nth_step
        acc_at_block_with_min_loss = stats['valid_acc'][block_with_min_loss]
    except ValueError:
        acc_at_block_with_min_loss = ''

    # 2) save a reduced summary
    summary = dict(name=cfg.name,
                   model=cfg.model.name,
                   optimizer=cfg.hyp.optim.name,
                   stoch=cfg.hyp.train_stochastic,
                   augmentations=True if cfg.data.augmentations_train not in [None, '', ' '] else False,
                   valid_acc=_maybe_record('valid_acc'),
                   valid_acc_at_min_loss=acc_at_block_with_min_loss,
                   max_val_acc=max_val_acc,  # Use this only to verify that the min_loss implementation works. Not a stat to report!
                   train_acc=_maybe_record('train_acc'),
                   valid_loss=_maybe_record('valid_loss'),
                   train_loss=_maybe_record('train_loss'),
                   full_loss=_maybe_record('full_loss'),
                   grad_norm=_maybe_record('grad_norm'),
                   avg_step_time=np.median(np.asarray(stats['train_time'], dtype=np.float)),
                   total_time=str(datetime.timedelta(seconds=local_time)).replace(',', ''),
                   param_norm=_maybe_record('param_norm'),
                   grad_SNR=_maybe_record('analysis_grad_SNR'),
                   grad_mean_norm=_maybe_record('analysis_grad_mean_norm'),
                   grad_noise_scale=_maybe_record('analysis_grad_noise_scale'),
                   empirical_flatness=_maybe_record('analysis_empirical_flatness'),
                   batch_size=cfg.data.batch_size,
                   **cfg.hyp,
                   **{k: v for k, v in cfg.impl.items() if k != 'setup'},
                   seed=cfg.seed,
                   folder=os.getcwd().split('outputs/')[1])
    save_to_table(os.path.join(cfg.original_cwd, 'tables'),
                  f'fb_{cfg.data.name}_runs', dryrun=cfg.dryrun, **summary)


def save_to_table(out_dir, table_name, dryrun, **kwargs):
    """Save keys to .csv files. Function adapted from Micah."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f'table_{table_name}.csv')
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)  # noqa  # this line is testing the header
            # assert header == fieldnames[:len(header)]  # new columns are ok, but old columns need to be consistent
            # dont test, always write when in doubt to prevent erroneous table rewrites
    except Exception as e:  # noqa
        if not dryrun:
            # print('Creating a new .csv table...')
            with open(fname, 'w') as f:
                writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
                writer.writeheader()
        else:
            pass
            # print(f'Would create new .csv table {fname}.')

    # Write a new row
    if not dryrun:
        # Add row for this experiment
        with open(fname, 'a') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writerow(kwargs)
        # print('\nResults saved to ' + fname + '.')
    else:
        pass
        # print(f'Would save results to {fname}.')


def set_random_seed(seed=233):
    """."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
    # Can't be too careful :>


def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
