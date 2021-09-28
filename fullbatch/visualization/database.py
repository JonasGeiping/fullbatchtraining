"""Utility functions for visualization."""

import os
import platform
import pickle
import io

import torch

from .normalized_directions import compute_randomized_directions


def _load_serialized(bytes_data, device=torch.device('cpu')):
    buffer = io.BytesIO()
    buffer.write(bytes_data)
    buffer.seek(0)
    return torch.load(buffer, map_location=device)

def _save_serialized(data):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    buffer.seek(0)
    return buffer.read()

def load_loss_database(model, cfg_impl, cfg_viz, original_cwd, setup, log):
    """Prepare loss database."""
    import lmdb  # Import this lazily

    base_name = cfg_impl.checkpoint.name if cfg_viz.database_name is None else cfg_viz.database_name
    if base_name is None:
        base_name = 'debug_db_'
    full_name = os.path.splitext(base_name)[0] + f'_{cfg_viz.ignore_layers}_{cfg_viz.norm}_losses.lmdb'
    db_path = os.path.join(original_cwd, 'checkpoints', full_name)

    if cfg_viz.rebuild_existing_database:
        if os.path.isfile(db_path):
            os.remove(db_path)
            os.remove(db_path + '-lock')

    # Create DB
    if not os.path.isfile(db_path):
        log.info(f'Creating new database at {db_path}.')
        _create_loss_db(model, cfg_impl, cfg_viz, db_path)
        log.info(f'Database written successfully at {db_path}.')
    else:
        log.info(f'Reusing cached database at {db_path}.')

    # Load DB
    db = lmdb.open(db_path, subdir=False, max_readers=cfg_viz.max_readers, readonly=False, lock=False,
                   readahead=cfg_viz.readahead, meminit=cfg_viz.meminit, max_spare_txns=cfg_viz.max_spare_txns,
                   map_size=_get_mapsize(cfg_viz.map_size))

    with db.begin(write=False) as txn:
        try:
            model_state_dict = _load_serialized(txn.get(b'model_state_dict'), device=setup['device'])
            x_direction = _load_serialized(txn.get(b'x_direction'), device=setup['device'])
            y_direction = _load_serialized(txn.get(b'y_direction'), device=setup['device'])
            # Verify model correctness:
            for p_loaded, p_model in zip(model.state_dict(), model_state_dict):
                assert torch.equal(model.state_dict()[p_loaded], model_state_dict[p_model])
        except TypeError:
            raise ValueError(f'The provided LMDB dataset at {db_path} is unfinished or damaged.')

    return db, x_direction, y_direction


def _get_mapsize(cfg_map_size):
    if platform.system() == 'Linux':
        map_size = 1099511627776 * 2  # Linux can grow memory as needed.
    else:
        map_size = cfg_map_size
        print(f'Remember to provide a reasonable default map_size for your {platform.system()} operating system.')
    return map_size

def _create_loss_db(model, cfg_impl, cfg_viz, db_path):
    """Create new LMDB at location."""
    import lmdb  # Import this lazily

    db = lmdb.open(db_path, subdir=False,
                   map_size=_get_mapsize(cfg_viz.map_size), readonly=False,
                   meminit=cfg_viz.meminit, map_async=True)

    x_direction, y_direction = compute_randomized_directions(model, cfg_viz)

    with db.begin(write=True) as txn:
        txn.put(b'model_state_dict', _save_serialized(model.state_dict()))
        txn.put(b'x_direction', _save_serialized(x_direction))
        txn.put(b'y_direction', _save_serialized(y_direction))



def load_surface_from_lmdb(db_path, positions):
    import lmdb  # Import this lazily

    db = lmdb.open(db_path, subdir=False, readonly=True, meminit=True, map_async=True)

    landscape = dict(train_loss=torch.zeros(len(positions)) * float('nan'),
                     train_acc=torch.zeros(len(positions)) * float('nan'),
                     full_loss=torch.zeros(len(positions)) * float('nan'))
    with db.begin(write=False) as txn:
        for idx, position in enumerate(positions):
            db_key = pickle.dumps([position])
            value = txn.get(db_key, default=None)
            if value is not None:
                try:
                    payload = pickle.loads(value)
                    for key in ['train_loss', 'train_acc', 'full_loss']:
                        landscape[key][idx] = payload.get(key, float('nan'))
                except pickle.UnpicklingError:
                    pass
    return landscape
