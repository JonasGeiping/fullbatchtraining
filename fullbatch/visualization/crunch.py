"""Compute loss landscape in normalized random directions."""

import torch

import time
from collections import defaultdict
import os

import pickle


from ..training import optim_interface, get_loss_fn
from ..utils import get_log

from .database import load_loss_database


def crunch(model, trainloader, validloader, setup, cfg):

    log = get_log(cfg)
    model.eval()
    optimizer, _ = optim_interface(model, cfg.hyp)
    stats = defaultdict(list)

    # Load checkpointed state:
    # This checkpoint has to match the current cfg!
    if cfg.impl.checkpoint.name is not None:
        file = os.path.join(cfg.original_cwd, 'checkpoints', cfg.impl.checkpoint.name)
        optim_state, model_state, _, _, step = torch.load(file, map_location=setup['device'])
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)
        log.info(f'Loaded model checkpoint from step {step} successfully.')
    else:
        step = 0
        cfg.impl.checkpoint.name = cfg.name
        log.info('No checkpoint supplied! Loss landscape will be computed for the model initialization without training.')

    num_blocks = len(trainloader)
    num_machines = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    acc_dtype = getattr(torch, cfg.impl.accumulation_dtype)

    loss_fn = get_loss_fn(cfg.hyp, cfg.data.batch_size)
    compute_grads = True if cfg.viz.compute_full_loss and cfg.hyp.grad_reg.block_strength != 0 else False

    # Prepare database and choose random directions
    # Also allows for reuse of an existing db
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        db, x_direction, y_direction = load_loss_database(model, cfg.impl, cfg.viz, cfg.original_cwd, setup, log)
    else:
        db = None
        x_direction, y_direction = [torch.zeros_like(model.parameters())], [torch.zeros_like(model.parameters())]
    base_parameters = [p.detach().clone() for p in model.parameters()]

    # Distribute random directions:
    if torch.distributed.is_initialized():
        def _send_direction(direction):
            concat_dir = torch.cat([tensor.data.reshape(-1) for tensor in direction]).to(**setup)
            torch.distributed.broadcast(concat_dir, 0, async_op=False)
            pointer = 0
            for tensor in direction:
                num_values = tensor.numel()
                tensor.data = concat_dir[pointer:pointer + num_values].view_as(tensor)
                pointer += num_values
        _send_direction(x_direction)
        _send_direction(y_direction)

    xcoords = torch.linspace(cfg.viz.coordinates.x.min, cfg.viz.coordinates.x.max, cfg.viz.coordinates.x.num)
    ycoords = torch.linspace(cfg.viz.coordinates.y.min, cfg.viz.coordinates.y.max, cfg.viz.coordinates.y.num)

    valid_positions = [[x.item(), y.item()] for x in xcoords for y in ycoords]

    def _set_parameter_offset(directions, coordinates):
        dx = directions[0]
        dy = directions[1]
        changes = [d0 * coordinates[0] + d1 * coordinates[1] for (d0, d1) in zip(dx, dy)]
        for (p, w, d) in zip(model.parameters(), base_parameters, changes):
            p.data = w + d.to(**setup)

    def _compute_batched_gradient(inputs, labels, compute_grads=False, create_graph=False):
        with torch.inference_mode(mode=not compute_grads):
            with torch.cuda.amp.autocast(enabled=cfg.impl.mixed_precision):
                outputs = model(inputs)
                block_loss = loss_fn(outputs, labels)  # the running average takes the number of machines into account
                block_correct_preds = (outputs.argmax(dim=-1) == labels).float().sum()

        grads = torch.autograd.grad(block_loss, model.parameters(), create_graph=False) if compute_grads else None
        return grads, block_loss.detach(), block_correct_preds.detach()

    @torch.no_grad()
    def _communicate_full_loss(step_loss, step_preds, datapoints, grad_norms):
        # Compute full loss:
        param_norm = sum([p.detach().pow(2).sum() for p in model.parameters()])
        full_loss = step_loss / num_blocks + 0.5 * getattr(cfg.hyp.optim, 'weight_decay', 0.0) * param_norm
        if cfg.hyp.grad_reg.block_strength != 0:
            reg_strength = optimizer.param_groups[0]["lr"] / 4 * cfg.hyp.grad_reg.block_strength
            full_grad_norm = grad_norms.mean()  # Mean here because the gradient contribs are averaged as well
            full_loss += reg_strength * full_grad_norm
        if cfg.hyp.grad_reg.acc_strength != 0:
            raise ValueError('Loss landscape does not contain acc_strength!')

        if torch.distributed.is_initialized():
            package = torch.stack([step_loss, step_preds, full_loss])
            # Add loss terms from all machines:
            torch.distributed.reduce(package, dst=0, async_op=False)
            step_loss, step_preds, full_loss = package

        train_loss = step_loss.item() / num_blocks / num_machines
        train_acc = step_preds.item() / datapoints / num_machines
        full_loss = full_loss.item() / num_machines

        return train_loss, train_acc, full_loss

    def _accumulate_full_loss(trainloader):
        train_time = time.time()
        average_grads = [torch.zeros_like(p, dtype=acc_dtype) for p in model.parameters()]

        # Central code block starts here ###############################################################################
        step_loss, step_preds, datapoints = 0.0, 0.0, 0
        grad_norms = torch.zeros(num_blocks, device=setup['device'], dtype=setup['dtype'])
        trainloader.sampler.set_epoch(step)
        for block, (inputs, labels) in enumerate(trainloader):
            datapoints += labels.shape[0]
            inputs = inputs.to(**setup, non_blocking=cfg.impl.non_blocking)
            labels = labels.to(dtype=torch.long, device=setup['device'], non_blocking=cfg.impl.non_blocking)

            grads, block_loss, block_correct_preds = _compute_batched_gradient(inputs, labels, compute_grads=compute_grads,
                                                                               create_graph=False)
            if compute_grads:
                grad_norms[block] = torch.stack([g.detach().pow(2).sum() for g in grads]).sum()

            step_loss += block_loss
            step_preds += block_correct_preds
        # ##############################################################################################################
        train_loss, train_acc, full_loss = _communicate_full_loss(step_loss, step_preds, datapoints, grad_norms)
        time_stamp = time.time() - train_time
        return train_loss, train_acc, full_loss, time_stamp

    # Run through all positions and find runs that are not yet in the database:
    time.sleep(torch.rand((1,)).mul(10).item())  # Sleep for a random amount of time to prevent deadlocks at start

    # Loop over all positions:
    for position in valid_positions:
        db_key = pickle.dumps([position])

        # Check if loss value already exists:
        with db.begin(write=False) as txn:
            value = txn.get(db_key, default=None)

        # If not, compute it (possibly distributed over all machines)
        if value is None:
            current_position = torch.as_tensor(position, device=setup['device'])
            if torch.distributed.is_initialized():
                # Assert that all workers are at the same position:
                current_position = torch.distributed.broadcast(current_position, 0, async_op=False)
            # Write placeholder from rank 0, in case other processes are also computing values:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                with db.begin(write=True) as txn:
                    txn.put(db_key, u'{}'.format(True).encode('ascii'))

            # Start computation:
            _set_parameter_offset([x_direction, y_direction], current_position)
            train_loss, train_acc, full_loss, time_stamp = _accumulate_full_loss(trainloader)

            # Save results to db:
            log.info(status_message(train_loss, train_acc, full_loss, time_stamp, current_position))
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                with db.begin(write=True) as txn:
                    payload = pickle.dumps(dict(train_loss=train_loss, train_acc=train_acc, full_loss=full_loss))
                    txn.replace(db_key, payload)
        else:
            log.info(f'Skipping loss at position {position}')
            pass  # pass to next key


def status_message(train_loss, train_acc, full_loss, time_stamp, current_position):
    """Basic console printout during loss crunch."""

    msg = f'Pos: [{current_position[0].item():4.2f}, {current_position[1].item():4.2f}] | Time: {time_stamp:4.2f}s |'
    msg += f'TRAIN loss {train_loss:7.4f} | TRAIN Acc: {train_acc:7.2%} |'
    msg += f'Full loss {full_loss:7.4f} |'
    return msg
