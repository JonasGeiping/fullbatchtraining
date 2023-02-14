"""LMBD dataset wrap an existing dataset and create a database if necessary."""

import os

import pickle
import copy
import warnings
import time
import datetime

import platform
import tempfile
import lmdb

import torch
import torchvision
import numpy as np
from PIL import Image

from .data_preparation import _parse_data_augmentations
import logging

warnings.filterwarnings("ignore", "The given buffer is not writable", UserWarning)

log = logging.getLogger(__name__)
NUM_DB_ATTEMPTS = 10


class LMDBDataset(torch.utils.data.Dataset):
    """Implement LMDB caching and access.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py
    and
    https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
    """

    def __init__(self, dataset, cfg_db, name="train", can_create=True):
        """Initialize with a given pytorch dataset."""
        # self.dataset = dataset

        self.live_transform = copy.deepcopy(dataset.transform)
        if self.live_transform is not None:
            if isinstance(self.live_transform.transforms[0], torchvision.transforms.ToTensor):
                self.skip_pillow = True
                self.live_transform.transforms.pop(0)
                db_channels_first = True
            else:
                db_channels_first = False
                self.skip_pillow = False
        else:
            db_channels_first = False
            self.skip_pillow = True
        self.path, self.handle = _choose_lmdb_path(dataset, cfg_db, db_channels_first, name=name)
        if can_create:
            _maybe_create_lmdb(dataset, self.path, cfg_db, db_channels_first, name=name)

        # Setup database
        self.cfg = cfg_db
        self.access = cfg_db.access
        for attempt in range(NUM_DB_ATTEMPTS):
            self.db = lmdb.open(
                self.path,
                subdir=False,
                max_readers=cfg_db.max_readers,
                readonly=True,
                lock=False,
                readahead=cfg_db.readahead,
                meminit=cfg_db.meminit,
                max_spare_txns=cfg_db.max_spare_txns,
            )
            try:
                with self.db.begin(write=False) as txn:
                    self.length = pickle.loads(txn.get(b"__len__"))
                    self.keys = pickle.loads(txn.get(b"__keys__"))
                    self.labels = pickle.loads(txn.get(b"__labels__"))
                    self.shape = pickle.loads(txn.get(b"__shape__"))
                break
            except TypeError:
                warnings.warn(f"The provided LMDB dataset at {self.path} is unfinished or damaged. Waiting and retrying.")
                time.sleep(13)
        else:
            raise ValueError(f"Database at path {self.path} damaged and could not be loaded after repeated attempts.")

        if self.access == "cursor":
            self._init_cursor()

        # Generate some references from the original dataset
        self.transform = self.live_transform
        self.classes = dataset.classes

    def __getstate__(self):
        state = self.__dict__.copy()
        state["db"] = None
        state["handle"] = None  # only the first instance of the db will keep a handle back to its path, good luck...
        if self.access == "cursor":
            state["_txn"] = None
            state["cursor"] = None

        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.handle = None
        # Regenerate db handle after pickling:
        self.db = lmdb.open(
            self.path,
            subdir=False,
            max_readers=self.cfg.max_readers,
            readonly=True,
            lock=False,
            readahead=self.cfg.readahead,
            meminit=self.cfg.meminit,
            max_spare_txns=self.cfg.max_spare_txns,
        )
        if self.access == "cursor":
            self._init_cursor()

    def _init_cursor(self):
        """Initialize cursor position."""
        self._txn = self.db.begin(write=False)
        self.cursor = self._txn.cursor()
        self.cursor.first()
        self.internal_index = 0

    # def __getattr__(self, name):
    #     """Call this only if all attributes of Subset are exhausted."""
    #     return getattr(self.dataset, name)

    def __len__(self):
        """Draw length from target dataset."""
        return self.length

    def __getitem__(self, index):
        """Get from database. This is either unordered or cursor access for now.
        Future: Write this class as a proper https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """

        if self.access == "cursor":
            index_key = "{}".format(index).encode("ascii")
            if index_key != self.cursor.key():
                self.cursor.set_key(index_key)

            byteflow = self.cursor.value()
            self.cursor.next()
        else:
            with self.db.begin(write=False) as txn:
                byteflow = txn.get(self.keys[index])

        # crime, but ok - we just disabled the warning...:
        # Tested this and the LMDB cannot be corrupted this way, even though byteflow is technically non-writeable
        data_block = torch.frombuffer(byteflow, dtype=torch.uint8).view(self.shape)

        if not self.skip_pillow:
            img = Image.fromarray(data_block.numpy())
        else:
            img = data_block.to(torch.float) / 255
        if self.live_transform is not None:
            img = self.live_transform(img)

        # load label
        label = self.labels[index]

        return img, label


def _choose_lmdb_path(raw_dataset, cfg_db, db_channels_first=False, name="train"):
    os.makedirs(os.path.expanduser(cfg_db.path), exist_ok=True)
    if os.path.isfile(os.path.expanduser(cfg_db.path)):
        raise ValueError("LMDB path must lead to a folder containing the databases, not a file.")
    augmentations = cfg_db.augmentations_train if "train" in name else cfg_db.augmentations_val
    if not cfg_db.temporary_database:
        round_info = f"R{cfg_db.rounds}" if "train" in name else ""
        round_info += "_first_clean" if "train" in name and cfg_db.first_round_clean else ""
        round_info += "_shuffled" if cfg_db.shuffle_while_writing else ""
        round_info += "_CHW" if db_channels_first else "HWC"
        full_name = f"{name}_{len(raw_dataset)}_" + "".join([l for l in repr(augmentations) if l.isalnum()]) + round_info

        path = os.path.join(os.path.expanduser(cfg_db.path), f"{type(raw_dataset).__name__}_{full_name}.lmdb")
        handle = None
    else:
        handle = tempfile.NamedTemporaryFile(dir=os.path.join(os.path.expanduser(cfg_db.path)))
        path = handle.name

    return path, handle


def _maybe_create_lmdb(raw_dataset, path, cfg_db, db_channels_first=False, name="train"):
    if cfg_db.rebuild_existing_database:
        if os.path.isfile(path):
            os.remove(path)
            os.remove(path + "-lock")

    # Load or create database
    if os.path.isfile(path) and not cfg_db.temporary_database:
        log.info(f"Reusing cached database at {path}.")
    else:
        os.makedirs(os.path.expanduser(cfg_db.path), exist_ok=True)
        log.info(f"Creating database at {path}. This may take some time ...")

        checksum = _create_database(raw_dataset, path, cfg_db, db_channels_first, name)


def _create_database(dataset, database_path, cfg_db, db_channels_first=False, name="train"):
    """Create an LMDB database from the given pytorch dataset.
    https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
    Removed pyarrow dependency
    """
    sample_train_transforms, sample_val_transforms = _parse_data_augmentations(cfg_db, PIL_only=True)
    if name == "train":
        repeat_dataset = True
        sample_transforms = sample_train_transforms
    else:
        repeat_dataset = False
        sample_transforms = sample_val_transforms

    data_transforms = copy.deepcopy(dataset.transform.transforms)

    if cfg_db.first_round_clean:
        dataset.transform.transforms = [torchvision.transforms.PILToTensor()]
    else:
        # keep transform alive as ref to original dataset, modify only its transforms attribute
        dataset.transform.transforms = [sample_transforms, torchvision.transforms.PILToTensor()]

    if platform.system() == "Linux":
        map_size = 1099511627776 * 2  # Linux can grow memory as needed.
    else:
        raise ValueError("Provide a reasonable default map_size for your operating system.")
    db = lmdb.open(
        database_path,
        subdir=False,
        map_size=map_size,
        readonly=False,
        meminit=cfg_db.meminit,
        writemap=True,
        map_async=True,
    )
    txn = db.begin(write=True)

    iterations = cfg_db.rounds if repeat_dataset else 1

    num_workers = min(16, torch.get_num_threads())
    batch_size = min(len(dataset) // max(num_workers, 1), 512)

    cacheloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=cfg_db.shuffle_while_writing,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if ((num_workers > 0) and (iterations > 1)) else None,
    )

    idx = 0
    timestamp = time.time()
    labels = []
    for round in range(iterations):
        for batch_idx, (img_batch, label_batch) in enumerate(cacheloader):
            # Run data transformations in (multiprocessed) batches
            for img, label in zip(img_batch, label_batch):
                # But we have to write sequentially anyway
                labels.append(label.item())
                # serialize
                if db_channels_first:
                    byteflow = np.asarray(img.numpy(), dtype=np.uint8).tobytes()
                else:
                    byteflow = np.asarray(img.permute(1, 2, 0).numpy(), dtype=np.uint8).tobytes()
                txn.put("{}".format(idx).encode("ascii"), byteflow)
                idx += 1

                if idx % cfg_db.write_frequency == 0:
                    time_taken = (time.time() - timestamp) / cfg_db.write_frequency
                    estimated_finish = str(datetime.timedelta(seconds=time_taken * len(dataset) * iterations))
                    timestamp = time.time()

                    txn.commit()
                    txn = db.begin(write=True)
                    log.info(f"[{idx} / {len(dataset) * iterations}] Estimated total time: {estimated_finish}")

        # reset dataloader after first round if first round clean:
        if cfg_db.first_round_clean and round == 0:
            dataset.transform.transforms = [sample_transforms, torchvision.transforms.PILToTensor()]
            cacheloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=cfg_db.shuffle_while_writing,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=False,
                persistent_workers=True if num_workers > 0 else None,
            )
    # finalize dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(idx)]
    shape = img.shape if db_channels_first else img.permute(1, 2, 0).shape
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", pickle.dumps(keys))
        txn.put(b"__labels__", pickle.dumps(labels))
        txn.put(b"__len__", pickle.dumps(len(keys)))
        txn.put(b"__shape__", pickle.dumps(shape))
    log.info(f"Database written successfully with {len(keys)} entries of shape {shape}.")
    dataset.transform.transforms = data_transforms
