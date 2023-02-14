"""Repeatable code parts concerning data loading.
Data Config Structure (cfg_data): See config/data
"""


import torch
import torchvision
import torchvision.transforms as transforms
from .datasets import TinyImageNet
from .auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform

import os
from .cached_dataset import CachedDataset


# Block ImageNet corrupt EXIF warnings
import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def construct_dataloader(cfg_data, cfg_impl, cfg_hyp, dryrun=False):
    """Return a dataloader with given dataset. Choose number of workers and their settings."""
    trainset, validset = _build_dataset(cfg_data, can_download=not cfg_impl.setup.dist)

    if cfg_data.db.name == "LMDB":
        from .lmdb_datasets import LMDBDataset  # this also depends on py-lmdb, that's why it's a lazy import

        trainset = LMDBDataset(trainset, cfg_data.db, "train", can_create=not cfg_impl.setup.dist)
        validset = LMDBDataset(validset, cfg_data.db, "val", can_create=not cfg_impl.setup.dist)

    if dryrun:
        # Limit datasets to just one batch
        # This comes after LMDB for safety reasons - an invalid DB might be written in that step
        num_machines = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        trainset = torch.utils.data.Subset(trainset, torch.arange(0, cfg_data.batch_size * num_machines))
        validset = torch.utils.data.Subset(validset, torch.arange(0, cfg_data.batch_size * num_machines))

    if cfg_data.caching:
        trainset = CachedDataset(trainset, num_workers=cfg_impl.threads, pin_memory=cfg_impl.pin_memory)
        validset = CachedDataset(validset, num_workers=cfg_impl.threads, pin_memory=cfg_impl.pin_memory)

    if cfg_impl.threads > 0:
        num_workers = (
            min(torch.get_num_threads(), cfg_impl.threads * max(1, torch.cuda.device_count())) if torch.get_num_threads() > 1 else 0
        )
    else:
        num_workers = 0

    if cfg_impl.setup.dist:
        train_sampler = torch.utils.data.DistributedSampler(trainset, shuffle=cfg_hyp.shuffle)
    else:
        if cfg_hyp.shuffle:
            train_sampler = torch.utils.data.RandomSampler(trainset, replacement=cfg_hyp.sample_with_replacement)
        else:
            train_sampler = torch.utils.data.SequentialSampler(trainset)

        # Patch the sampler to return nothing when set_epoch is called
        def set_epoch(*args, **kwargs):
            pass

        train_sampler.set_epoch = set_epoch

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=min(cfg_data.batch_size, len(trainset)),
        sampler=train_sampler,
        drop_last=True,  # just throw these images away forever :>
        num_workers=num_workers,
        pin_memory=cfg_impl.pin_memory,
        persistent_workers=cfg_impl.persistent_workers if num_workers > 0 else False,
    )
    # Distributed samplers can split data across machines,

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=min(cfg_data.batch_size, len(trainset)),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=cfg_impl.pin_memory,
        persistent_workers=False,
    )
    # but all machines replicate the validation procedure

    return trainloader, validloader


def construct_subset_dataloader(dataloader, cfg, step):
    """Subset dataloader from large dataloader."""
    random_idx = step % cfg.data.db.rounds  # torch.randint(0, cfg.data.db.rounds, (1,))
    dataset_subset_ids = torch.arange(0, cfg.data.size) + random_idx * cfg.data.size
    dataset = torch.utils.data.Subset(dataloader.dataset, dataset_subset_ids)
    if cfg.impl.setup.dist:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=cfg.hyp.shuffle)
    else:
        sampler = torch.utils.data.RandomSampler(dataset) if cfg.hyp.shuffle else torch.utils.data.SequentialSampler(dataset)

        # Patch the sampler to return nothing when set_epoch is called
        def set_epoch(*args, **kwargs):
            pass

        sampler.set_epoch = set_epoch
    localloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(cfg.data.batch_size, len(dataset)),
        sampler=sampler,
        drop_last=True,
        num_workers=dataloader.num_workers,
        pin_memory=cfg.impl.pin_memory,
    )
    return localloader


def _build_dataset(cfg_data, can_download=True):
    cfg_data.path = os.path.expanduser(cfg_data.path)
    if cfg_data.name == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root=cfg_data.path, train=True, download=can_download, transform=transforms.ToTensor())
        validset = torchvision.datasets.CIFAR10(root=cfg_data.path, train=False, download=can_download, transform=None)
    elif cfg_data.name == "CIFAR100":
        trainset = torchvision.datasets.CIFAR100(root=cfg_data.path, train=True, download=can_download, transform=transforms.ToTensor())
        validset = torchvision.datasets.CIFAR100(root=cfg_data.path, train=False, download=can_download, transform=None)
    elif cfg_data.name == "ImageNet":
        trainset = torchvision.datasets.ImageNet(root=cfg_data.path, split="train", transform=transforms.ToTensor())
        validset = torchvision.datasets.ImageNet(root=cfg_data.path, split="val", transform=None)
    elif cfg_data.name == "TinyImageNet":
        trainset = TinyImageNet(root=cfg_data.path, split="train", download=can_download, transform=transforms.ToTensor(), cached=True)
        validset = TinyImageNet(root=cfg_data.path, split="val", download=can_download, transform=None, cached=True)
    else:
        raise ValueError(f"Invalid dataset {cfg_data.name} provided.")

    if cfg_data.mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cfg_data.mean, cfg_data.std

    train_transforms, valid_transforms = _parse_data_augmentations(cfg_data)

    # Apply transformations
    trainset.transform = train_transforms if train_transforms is not None else None
    validset.transform = valid_transforms if valid_transforms is not None else None

    # Reduce train dataset according to cfg_data.size:
    if cfg_data.size < len(trainset):
        trainset = torch.utils.data.Subset(trainset, torch.arange(0, cfg_data.size))

    return trainset, validset


def _get_meanstd(dataset):
    cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std


def _get_autoaugment(auto_augment, img_size_min=32, mean=(0, 0, 0)):
    """The auto_augment key could be something like rand-m7-mstd0.5-inc1"""
    assert isinstance(auto_augment, str)
    aa_params = dict(
        translate_const=int(img_size_min * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in mean]),
    )
    if auto_augment.startswith("rand"):
        return rand_augment_transform(auto_augment, aa_params)
    elif auto_augment.startswith("augmix"):
        aa_params["translate_pct"] = 0.3
        return augment_and_mix_transform(auto_augment, aa_params)
    else:
        return auto_augment_transform(auto_augment, aa_params)


def _parse_data_augmentations(cfg_data, PIL_only=False):
    def _parse_cfg_dict(cfg_dict):
        list_of_transforms = []
        if hasattr(cfg_dict, "keys"):
            for key in cfg_dict.keys():
                if key in ["RandAugment", "AutoAugment", "AugMix"]:
                    # TIMM implementations
                    transform = _get_autoaugment(cfg_dict[key], img_size_min=cfg_data.pixels, mean=cfg_data.mean)
                else:
                    # Torchvision implementations:
                    try:  # ducktype iterable
                        transform = getattr(transforms, key)(*cfg_dict[key])
                    except TypeError:
                        transform = getattr(transforms, key)(cfg_dict[key])
                list_of_transforms.append(transform)
        return list_of_transforms

    train_transforms = _parse_cfg_dict(cfg_data.augmentations_train)
    valid_transforms = _parse_cfg_dict(cfg_data.augmentations_val)

    if not PIL_only:
        train_transforms.append(transforms.ToTensor())
        valid_transforms.append(transforms.ToTensor())
        if cfg_data.normalize:
            train_transforms.append(transforms.Normalize(cfg_data.mean, cfg_data.std))
            valid_transforms.append(transforms.Normalize(cfg_data.mean, cfg_data.std))

    return transforms.Compose(train_transforms), transforms.Compose(valid_transforms)
