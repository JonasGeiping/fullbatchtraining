"""Main training routine.

A single computation unit is a "step" which represents a single step in the optimization, possibly containing multiple
evaluations of the gradient. If train_stochastic=False, then "the" gradient is the gradient over the whole dataset,
when stochastic, this is a stochastic sample.

In the full-batch setting the train function implements the following structure:

while step < cfg.steps:
    def closure():
        eval _accumulate_full_gradient:
            for block in loader:
                _compute_batch_gradient()               [includes gradreg object to modify batch gradients]
                gradreg                                 [Modify gradients in-place to account for gradient reg]
                update_avg_gradients                    [inplace, using torch._foreach_.]
            _record_stats()                             [Record loss vals, gradient norms, param norms]
            _allreduce_coalesced(model, avg_gradients)  [allreduce + assign to model.parameters().grad]
        _modify_gradient_params()    [This function modifies the full averaged gradient, e.g. to clip]

    optimizer.step(closure)
    step += 1

    evaluate()            [This is validation]
    status_message()      [Console and log printout]
    analyze()             [Optionally compute additional stats]

"""
import torch

import time
from collections import defaultdict
import copy
import os


from .optimizers import optim_interface
from ..utils import get_log
from .utils import _clip_gradient_list, _update_ema, _allreduce_coalesced
from .utils import _save_state_for_visualization, _save_to_checkpoint, _load_from_checkpoint
from ..models.modules import LabelSmoothCrossEntropyLoss, MaxupLoss, GradRegularizer, IncorrectCrossEntropyLoss
from ..analysis import analyze
from ..data import construct_subset_dataloader


def _stable_mean_accumulation(running_mean, new_value, counter):
    torch._foreach_sub_(new_value, running_mean)
    torch._foreach_add_(running_mean, new_value, alpha=1 / counter)


def train(model, trainloader, validloader, setup, cfg):
    """Train given model based on implementation details and hyperparameters."""
    log = get_log(cfg)
    model.train()
    optimizer, scheduler = optim_interface(model, cfg.hyp)
    stats = defaultdict(list)

    class Counter:
        step: int = 0

    # Optionally: Load checkpoint:
    if cfg.impl.checkpoint.name is not None:
        file = os.path.join(cfg.original_cwd, "checkpoints", cfg.impl.checkpoint.name)
        _load_from_checkpoint(model, optimizer, scheduler, None, Counter, cfg.hyp.steps, device=setup["device"], file=file)

    num_blocks = len(trainloader)
    num_chunks = max(cfg.data.batch_size // cfg.hyp.sub_batch, 1)
    num_machines = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    acc_dtype = getattr(torch, cfg.impl.accumulation_dtype)

    loss_fn = get_loss_fn(cfg.hyp, cfg.data.batch_size)
    gradreg = GradRegularizer(model, optimizer, loss_fn, **cfg.hyp.grad_reg, mixed_precision=False)

    if cfg.hyp.evaluate_ema:
        ema_model = copy.deepcopy(model)

    def _compute_batched_gradient(inputs, labels, create_graph=False):
        with torch.cuda.amp.autocast(enabled=cfg.impl.mixed_precision):
            outputs = model(inputs)
            block_loss = loss_fn(outputs, labels)  # the running average takes the number of machines into account
            block_correct_preds = (outputs.argmax(dim=-1) == labels).float().sum()

        grads = torch.autograd.grad(block_loss, model.parameters(), create_graph=create_graph)
        return grads, block_loss.detach(), block_correct_preds.detach()

    @torch.no_grad()
    def _record_stats(stats, pre_grads, step_loss, step_preds, datapoints, train_time, grad_norms):
        for idx, entry in enumerate(grad_norms.sqrt().tolist()):
            # Recorded stats for grad_norm_train are not reduced to rank 0, too much effort.
            stats[f"grad_norm_train_{idx}"] += [entry]

        # Compute full loss:
        param_norm = sum([p.detach().pow(2).sum() for p in model.parameters()])
        full_grad_norm = grad_norms.mean()  # Mean here because the gradient contribs are averaged as well

        full_loss = step_loss / num_blocks + 0.5 * getattr(cfg.hyp.optim, "weight_decay", 0.0) * param_norm
        if cfg.hyp.grad_reg.block_strength != 0:
            reg_strength = optimizer.param_groups[0]["lr"] / 4 * cfg.hyp.grad_reg.block_strength
            full_loss += reg_strength * full_grad_norm
        if cfg.hyp.grad_reg.acc_strength != 0:
            avg_grad_norm = torch.stack([g.detach().pow(2).sum() for g in pre_grads]).sum()
            reg_strength = optimizer.param_groups[0]["lr"] / 4 * cfg.hyp.grad_reg.acc_strength
            full_loss += reg_strength * avg_grad_norm

        if torch.distributed.is_initialized():
            package = torch.stack([step_loss, step_preds, full_loss, full_grad_norm])
            # Add loss terms from all machines:
            torch.distributed.reduce(package, dst=0, async_op=False)
            step_loss, step_preds, full_loss, full_grad_norm = package

        stats["train_loss"] += [step_loss.item() / num_blocks / num_machines]
        stats["train_acc"] += [step_preds.item() / datapoints / num_machines]
        stats["train_time"] += [time.time() - train_time]
        stats["param_norm"] += [param_norm.item()]
        stats["grad_norm"] += [full_grad_norm.sqrt().item() / num_machines]
        stats["full_loss"] += [full_loss.item() / num_machines]

        if cfg.hyp.batch_clip is not None:
            stats["clipped_batches"] += [clipped_batches]
            log.info(f"{clipped_batches} of {num_blocks * num_chunks} batches clipped to {cfg.hyp.batch_clip} in this step.")

    def _accumulate_full_gradient(trainloader, stats):
        train_time = time.time()
        average_grads = [torch.zeros_like(p, dtype=acc_dtype) for p in model.parameters()]

        # The following block precomputes the full gradient to allow for a computation of the GD regularization effect
        # in addition to the SGD regularization effect
        # This is usually skipped
        if cfg.hyp.grad_reg.acc_strength != 0:
            pre_grads = [torch.zeros_like(p, dtype=acc_dtype) for p in model.parameters()]
            trainloader.sampler.set_epoch(Counter.step)
            for block, (inputs, labels) in enumerate(trainloader):
                inputs = inputs.to(**setup, non_blocking=cfg.impl.non_blocking)
                labels = labels.to(dtype=torch.long, device=setup["device"], non_blocking=cfg.impl.non_blocking)

                grads, _, _ = _compute_batched_gradient(inputs, labels, create_graph=False)
                with torch.no_grad():
                    grads = [g.to(dtype=acc_dtype) for g in grads]
                    if cfg.hyp.batch_clip is not None:
                        _clip_gradient_list(grads, cfg.hyp.batch_clip, cfg)
                    _stable_mean_accumulation(pre_grads, grads, counter=num_machines * (block + 1))
        else:
            pre_grads = None

        # Central code block starts here ###############################################################################
        step_loss, step_preds, datapoints, clipped_batches = 0.0, 0.0, 0, 0
        grad_norms = torch.zeros(num_chunks * num_blocks, device=setup["device"], dtype=setup["dtype"])
        trainloader.sampler.set_epoch(Counter.step)
        for block, (inputs, labels) in enumerate(trainloader):
            datapoints += labels.shape[0]
            chunks_in_block = max(labels.shape[0] // cfg.hyp.sub_batch, 1)
            inputs = inputs.to(**setup, non_blocking=cfg.impl.non_blocking)
            labels = labels.to(dtype=torch.long, device=setup["device"], non_blocking=cfg.impl.non_blocking)

            # Optionally chunk batches further
            input_chunks = torch.chunk(inputs, chunks_in_block, dim=0)
            label_chunks = torch.chunk(labels, chunks_in_block, dim=0)

            for idx, (input_chunk, label_chunk) in enumerate(zip(input_chunks, label_chunks)):
                grads, chunk_loss, chunk_correct_preds = _compute_batched_gradient(
                    input_chunk, label_chunk, create_graph=gradreg.create_graph
                )
                grad_norms[block * num_chunks + idx] = torch.stack([g.detach().pow(2).sum() for g in grads]).sum()
                grads = gradreg(grads, input_chunk, label_chunk, pre_grads)
                with torch.no_grad():
                    grads = [g.to(dtype=acc_dtype) for g in grads]
                    if cfg.hyp.batch_clip is not None:
                        clipped_batches += _clip_gradient_list(grads, cfg.hyp.batch_clip, cfg)
                    _stable_mean_accumulation(average_grads, grads, counter=num_machines * (block * chunks_in_block + idx + 1))

                    # torch._foreach_sub_(grads, average_grads)
                    # torch._foreach_add_(average_grads, grads, alpha=1 / (num_machines * num_blocks * chunks_in_block))
                    step_loss += chunk_loss / chunks_in_block
                    step_preds += chunk_correct_preds
        # ##############################################################################################################
        _record_stats(stats, pre_grads, step_loss, step_preds, datapoints, train_time, grad_norms)

        # Distribute gradients across multiple machines
        model.to(dtype=acc_dtype)  # param and param.grad need to share their dtype
        if torch.distributed.is_initialized():
            _allreduce_coalesced(model, average_grads)
        else:
            for param, grad in zip(model.parameters(), average_grads):
                param.grad = grad

        return step_loss.detach() / num_blocks

    @torch.no_grad()
    def _modify_gradient_params():
        if cfg.hyp.norm_bias.strength > 0.0:
            param_norm_l2 = sum([p.pow(2).sum() for p in model.parameters()])
            if cfg.hyp.norm_bias.norm_type == 1:
                diff_value_sign = (param_norm_l2 - cfg.hyp.norm_bias.bias**2).sign()
                [p.grad.add_(cfg.hyp.norm_bias.strength * diff_value_sign) for p in model.parameters()]
            else:
                factor = 2 * (param_norm_l2 - cfg.hyp.norm_bias.bias**2)
                [p.grad.add_(cfg.hyp.norm_bias.strength * factor * p) for p in model.parameters()]

        if cfg.hyp.grad_clip is not None:  # this is full clipping, we could also have block-level clipping
            if cfg.hyp.grad_clip_norm == float("inf"):
                grad_norm = max(p.grad.abs().max() for p in model.parameters())
            else:
                grad_norm = torch.norm(
                    torch.stack([torch.norm(p.grad, cfg.hyp.grad_clip_norm) for p in model.parameters()]), cfg.hyp.grad_clip_norm
                )
            stats["preclip_gradnorm"] += [grad_norm.item()]
            if grad_norm > cfg.hyp.grad_clip:
                [p.grad.mul_(cfg.hyp.grad_clip / (grad_norm + 1e-6)) for p in model.parameters()]
                log.info(f"Gradient total norm was {grad_norm}. Clipping to {cfg.hyp.grad_clip}.")
                stats["clipped_step"] += [1]
            else:
                stats["clipped_step"] += [0]
        if cfg.hyp.grad_noise["additive"] is not None:  # additive noise as in Langevin dynamics or diff. privacy
            [p.grad.add_(cfg.hyp.grad_noise["additive"] * torch.randn_like(p)) for p in model.parameters()]
        if cfg.hyp.grad_noise["multiplicative"] is not None:  # multiplicative noise as in Hoffer et al.
            [p.grad.mul_(1 + cfg.hyp.grad_noise["multiplicative"] * torch.randn_like(p)) for p in model.parameters()]

    train_stochastic = cfg.hyp.train_stochastic
    # ## MAIN LOOP is controlled here ## ###################
    while Counter.step < cfg.hyp.steps:
        model.train()
        # Optionally switch between stoch. and non.stoch. mode
        if cfg.hyp.train_switch_stochastic is not None and cfg.hyp.train_switch_stochastic >= Counter.step:
            train_stochastic = not cfg.hyp.train_stochastic
        if not train_stochastic:

            def gradient_evaluation():
                """This is a full-blown closure that is passed to the optimizer."""
                # Gradient evaluation part:
                model.to(dtype=setup["dtype"])
                # this may move the model to a different precision
                loss = _accumulate_full_gradient(trainloader, stats)
                # Modify gradient:
                _modify_gradient_params()
                return loss

            # Take the actual steps:
            optimizer.step(gradient_evaluation)
            scheduler.step()
            Counter.step += 1

        else:  # stochastic sanity check!!
            train_time = time.time()

            if cfg.hyp.train_semi_stochastic:
                localloader = construct_subset_dataloader(trainloader, cfg, Counter.step)
            else:
                localloader = trainloader

            localloader.sampler.set_epoch(Counter.step)
            step_loss, step_preds, datapoints = 0.0, 0.0, 0
            grad_norms = torch.zeros(num_blocks, device=setup["device"], dtype=setup["dtype"])

            for block, (inputs, labels) in enumerate(localloader):
                inputs = inputs.to(**setup, non_blocking=cfg.impl.non_blocking)
                labels = labels.to(dtype=torch.long, device=setup["device"], non_blocking=cfg.impl.non_blocking)

                def gradient_evaluation():
                    """This is a full-blown closure that is passed to the optimizer."""
                    nonlocal grad_norms, step_loss, step_preds, datapoints  # :>
                    # Gradient evaluation part:
                    model.to(dtype=setup["dtype"])
                    datapoints += labels.shape[0]
                    grads, block_loss, block_correct_preds = _compute_batched_gradient(inputs, labels)
                    grad_norms[block] = torch.stack([g.detach().pow(2).sum() for g in grads]).sum()
                    grads = gradreg(grads, inputs, labels, None)
                    grads = [g.to(dtype=acc_dtype) for g in grads]
                    model.to(dtype=acc_dtype)  # param and param.grad need to share their dtype
                    if torch.distributed.is_initialized():
                        # slower than DDP but consistent with the fullbatch implementation
                        _allreduce_coalesced(model, grads)
                    else:
                        for param, grad in zip(model.parameters(), grads):
                            param.grad = grad
                    if cfg.hyp.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyp.grad_clip, norm_type=2.0)
                    # _modify_gradient_params()
                    step_loss += block_loss
                    step_preds += block_correct_preds
                    return step_loss.detach()

                step_loss = optimizer.step(gradient_evaluation)
                optimizer.zero_grad()

            _record_stats(stats, None, step_loss, step_preds, datapoints, train_time, grad_norms)
            scheduler.step()
            Counter.step += 1

        # Update EMA
        model.to(dtype=setup["dtype"])
        if cfg.hyp.evaluate_ema:
            _update_ema(model, ema_model, momentum=cfg.hyp.eval_ema_momentum)
            eval_model = ema_model
        else:
            eval_model = model

        # Validate
        if (Counter.step - 1) % cfg.impl.validate_every_nth_step == 0 or Counter.step >= cfg.hyp.steps or cfg.dryrun:
            evaluate(eval_model, validloader, stats, setup, cfg.impl, cfg.hyp, dryrun=cfg.dryrun)

        # Print log
        log.info(status_message(optimizer, stats, Counter.step))

        # Run optional analysis stuff
        if cfg.analysis.type is not None:
            if Counter.step % cfg.analysis.check_every_nth_step == 0 or Counter.step >= cfg.hyp.steps or cfg.dryrun:
                analyze(eval_model, loss_fn, optimizer, trainloader, stats, setup, cfg)

        # Optionally save model dict and gradients for visualization:
        if cfg.analysis.save_model_every_nth_step is not None:
            if (Counter.step - 1) % cfg.analysis.save_model_every_nth_step == 0 or Counter.step >= cfg.hyp.steps:
                path = f"{cfg.name}_{cfg.model.name}_step_{Counter.step}.pth"
                _save_state_for_visualization(model, optimizer, cfg, path=path)

        # Early stopping if loss is not finite
        if not torch.as_tensor(stats["train_loss"][-1]).isfinite():
            log.info("Terminating iterations due to divergence of loss...")
            break

        # Optional stopping if the last n steps were at 100% training accuracy.
        if cfg.hyp.stop_at_full_training_accuracy > 0:
            last_n_accs = stats["train_acc"][-cfg.hyp.stop_at_full_training_accuracy :]
            if min(last_n_accs) == 1:
                log.info("Terminating training after fitting all datapoints.")
                # Do a final validation/analysis pass in this case:
                evaluate(eval_model, validloader, stats, setup, cfg.impl, cfg.hyp, dryrun=cfg.dryrun)
                if cfg.analysis.type is not None:
                    analyze(eval_model, loss_fn, optimizer, trainloader, stats, setup, cfg)
                break

        # Save internal checkpoints from rank 0 [Separate from model dict saves]
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if cfg.impl.checkpoint.name is not None:
                if (Counter.step - 1) % cfg.impl.checkpoint.save_every_nth_step == 0 or Counter.step >= cfg.hyp.steps:
                    file = os.path.join(cfg.original_cwd, "checkpoints", cfg.impl.checkpoint.name)
                    _save_to_checkpoint(model, optimizer, scheduler, None, Counter, file=file)

        if cfg.dryrun:
            break

    return stats


def evaluate(model, dataloader, stats, setup, cfg_impl, cfg_hyp, dryrun=False):
    """Validation. In a distributed setting this operation is replicated on all machines and work is not shared."""
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()

    if cfg_impl.setup.dist:
        # Synchronize statistics across machines if any exist:
        if len(list(model.buffers())) > 0:
            concat_buf = torch.cat([b.data.reshape(-1) for b in model.buffers()])
            torch.distributed.all_reduce(concat_buf, async_op=False)
            pointer = 0
            for buffer in model.buffers():
                num_values = buffer.numel()
                buffer.data = concat_buf[pointer : pointer + num_values].view_as(buffer) / cfg_impl.setup.world_size
                pointer += num_values

    if stats is None:
        stats = defaultdict(list)
    with torch.inference_mode():
        step_loss, step_preds, datapoints = 0.0, 0.0, 0

        # Iterate over all blocks in the validation dataset
        for block, (inputs, labels) in enumerate(dataloader):
            datapoints += labels.shape[0]
            inputs = inputs.to(**setup, non_blocking=cfg_impl.non_blocking)
            labels = labels.to(dtype=torch.long, device=setup["device"], non_blocking=cfg_impl.non_blocking)

            if cfg_hyp.test_time_flips:
                outputs_left = model(inputs).softmax(dim=1)
                outputs_right = model(torch.flip(inputs, [3])).softmax(dim=1)
                outputs = outputs_left + outputs_right  # averaging is a waste
            else:
                outputs = model(inputs)
            block_loss = loss_fn(outputs, labels)
            block_correct_preds = (outputs.argmax(dim=-1) == labels).float().sum()

            step_loss += block_loss.item() * labels.shape[0]
            step_preds += block_correct_preds.item()

            if dryrun:
                break

    stats["valid_loss"] += [step_loss / datapoints]
    stats["valid_acc"] += [step_preds / datapoints]
    model.train()
    return stats


def get_loss_fn(cfg_hyp, batch_size):
    if cfg_hyp.label_smoothing not in [None, ""]:
        if cfg_hyp.loss_modification is None:
            loss_fn = torch.jit.script(
                LabelSmoothCrossEntropyLoss(smoothing=cfg_hyp.label_smoothing, loss_modification=cfg_hyp.loss_modification)
            )
        elif cfg_hyp.loss_modification == "incorrect-xent":
            loss_fn = torch.jit.script(IncorrectCrossEntropyLoss(smoothing=cfg_hyp.label_smoothing))
        else:
            raise ValueError("Loss modification not implemented in conjunction with label smoothing.")
    else:
        if cfg_hyp.loss_modification is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        elif cfg_hyp.loss_modification == "incorrect-xent":
            loss_fn = torch.jit.script(IncorrectCrossEntropyLoss(smoothing=0.0))
        elif cfg_hyp.loss_modification == "batch-maxup":
            loss_fn = torch.jit.script(MaxupLoss(ntrials=batch_size))
        elif "maxup" in cfg_hyp.loss_modification:
            loss_fn = torch.jit.script(MaxupLoss(ntrials=int(cfg_hyp.loss_modification.split("maxup-")[1])))
        else:
            raise ValueError(f"Invalid loss modification {cfg_hyp.loss_modification}.")

    return loss_fn


def status_message(optimizer, stats, step):
    """A basic console printout."""
    current_lr = f'{optimizer.param_groups[0]["lr"]:.4f}'

    def _maybe_print(key):
        return stats[key][-1] if len(stats[key]) > 0 else float("NaN")

    msg = f'Step: {step:<4}| lr: {current_lr} | Time: {stats["train_time"][-1]:4.2f}s |'
    msg += f'TRAIN loss {stats["train_loss"][-1]:7.4f} | TRAIN Acc: {stats["train_acc"][-1]:7.2%} |'
    msg += f'VAL loss {_maybe_print("valid_loss"):7.4f} | VAL Acc: {_maybe_print("valid_acc"):7.2%} |'
    return msg


def _measure_implementation_noise(model, trainloader, validloader, setup, cfg):
    """Measure FP difference between successive runs. Todo: Make this nice and unify with the main loop."""

    log = get_log(cfg)
    model.train()

    class Counter:
        step: int = 0

    optimizer, scheduler = optim_interface(model, cfg.hyp)
    stats = defaultdict(list)
    # Optionally: Load checkpoint:
    if cfg.impl.checkpoint.name is None:
        print("Could not load checkpoint. Using newly initalized model.")
        cfg.impl.checkpoint.name = cfg.name
        file = os.path.join(cfg.original_cwd, "checkpoints", cfg.impl.checkpoint.name)
        _save_to_checkpoint(model, optimizer, scheduler, None, Counter, file=file)

    num_blocks = len(trainloader)
    num_chunks = max(cfg.data.batch_size // cfg.hyp.sub_batch, 1)
    num_machines = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    acc_dtype = getattr(torch, cfg.impl.accumulation_dtype)

    loss_fn = get_loss_fn(cfg.hyp, cfg.data.batch_size)
    gradreg = GradRegularizer(model, optimizer, loss_fn, **cfg.hyp.grad_reg, mixed_precision=False)

    def _compute_batched_gradient(inputs, labels, create_graph=False):
        with torch.cuda.amp.autocast(enabled=cfg.impl.mixed_precision):
            outputs = model(inputs)
            block_loss = loss_fn(outputs, labels)  # the running average takes the number of machines into account
            block_correct_preds = (outputs.argmax(dim=-1) == labels).float().sum()

        grads = torch.autograd.grad(block_loss, model.parameters(), create_graph=create_graph)
        return grads, block_loss.detach(), block_correct_preds.detach()

    def _accumulate_full_gradient(trainloader, stats):
        train_time = time.time()
        average_grads = [torch.zeros_like(p, dtype=acc_dtype) for p in model.parameters()]

        # The following block precomputes the full gradient to allow for a computation of the GD regularization effect
        # in addition to the SGD regularization effect
        # This is usually skipped
        if cfg.hyp.grad_reg.acc_strength != 0:
            pre_grads = [torch.zeros_like(p, dtype=acc_dtype) for p in model.parameters()]
            trainloader.sampler.set_epoch(Counter.step)
            for block, (inputs, labels) in enumerate(trainloader):
                inputs = inputs.to(**setup, non_blocking=cfg.impl.non_blocking)
                labels = labels.to(dtype=torch.long, device=setup["device"], non_blocking=cfg.impl.non_blocking)

                grads, _, _ = _compute_batched_gradient(inputs, labels, create_graph=False)
                with torch.no_grad():
                    grads = [g.to(dtype=acc_dtype) for g in grads]
                    if cfg.hyp.batch_clip is not None:
                        _clip_gradient_list(grads, cfg.hyp.batch_clip, cfg)
                    torch._foreach_sub_(grads, pre_grads)
                    torch._foreach_add_(pre_grads, grads, alpha=1 / (num_machines * num_blocks))
        else:
            pre_grads = None

        # Central code block starts here ###############################################################################
        step_loss, step_preds, datapoints, clipped_batches = 0.0, 0.0, 0, 0
        grad_norms = torch.zeros(num_chunks * num_blocks, device=setup["device"], dtype=setup["dtype"])
        trainloader.sampler.set_epoch(Counter.step)
        for block, (inputs, labels) in enumerate(trainloader):
            datapoints += labels.shape[0]
            chunks_in_block = max(labels.shape[0] // cfg.hyp.sub_batch, 1)
            inputs = inputs.to(**setup, non_blocking=cfg.impl.non_blocking)
            labels = labels.to(dtype=torch.long, device=setup["device"], non_blocking=cfg.impl.non_blocking)

            # Optionally chunk batches further
            input_chunks = torch.chunk(inputs, chunks_in_block, dim=0)
            label_chunks = torch.chunk(labels, chunks_in_block, dim=0)

            for idx, (input_chunk, label_chunk) in enumerate(zip(input_chunks, label_chunks)):
                grads, chunk_loss, chunk_correct_preds = _compute_batched_gradient(
                    input_chunk, label_chunk, create_graph=gradreg.create_graph
                )
                grad_norms[block * num_chunks + idx] = torch.stack([g.detach().pow(2).sum() for g in grads]).sum()
                grads = gradreg(grads, input_chunk, label_chunk, pre_grads)
                with torch.no_grad():
                    grads = [g.to(dtype=acc_dtype) for g in grads]
                    if cfg.hyp.batch_clip is not None:
                        clipped_batches += _clip_gradient_list(grads, cfg.hyp.batch_clip, cfg)
                    torch._foreach_sub_(grads, average_grads)
                    torch._foreach_add_(average_grads, grads, alpha=1 / (num_machines * num_blocks * chunks_in_block))
                    step_loss += chunk_loss / chunks_in_block
                    step_preds += chunk_correct_preds
        # ##############################################################################################################

        # Distribute gradients across multiple machines
        model.to(dtype=acc_dtype)  # param and param.grad need to share their dtype
        if torch.distributed.is_initialized():
            _allreduce_coalesced(model, average_grads)
        else:
            for param, grad in zip(model.parameters(), average_grads):
                param.grad = grad

        return step_loss.detach() / num_blocks

    @torch.no_grad()
    def _modify_gradient_params():
        if cfg.hyp.norm_bias.strength > 0.0:
            param_norm_l2 = sum([p.pow(2).sum() for p in model.parameters()])
            if cfg.hyp.norm_bias.norm_type == 1:
                diff_value_sign = (param_norm_l2 - cfg.hyp.norm_bias.bias**2).sign()
                [p.grad.add_(cfg.hyp.norm_bias.strength * diff_value_sign) for p in model.parameters()]
            else:
                factor = 2 * (param_norm_l2 - cfg.hyp.norm_bias.bias**2)
                [p.grad.add_(cfg.hyp.norm_bias.strength * factor * p) for p in model.parameters()]

        if cfg.hyp.grad_clip is not None:  # this is full clipping, we could also have block-level clipping
            if cfg.hyp.grad_clip_norm == float("inf"):
                grad_norm = max(p.grad.abs().max() for p in model.parameters())
            else:
                grad_norm = torch.norm(
                    torch.stack([torch.norm(p.grad, cfg.hyp.grad_clip_norm) for p in model.parameters()]), cfg.hyp.grad_clip_norm
                )
            stats["preclip_gradnorm"] += [grad_norm.item()]
            if grad_norm > cfg.hyp.grad_clip:
                [p.grad.mul_(cfg.hyp.grad_clip / (grad_norm + 1e-6)) for p in model.parameters()]
                log.info(f"Gradient total norm was {grad_norm}. Clipping to {cfg.hyp.grad_clip}.")
                stats["clipped_step"] += [1]
            else:
                stats["clipped_step"] += [0]
        if cfg.hyp.grad_noise["additive"] is not None:  # additive noise as in Langevin dynamics or diff. privacy
            [p.grad.add_(cfg.hyp.grad_noise["additive"] * torch.randn_like(p)) for p in model.parameters()]
        if cfg.hyp.grad_noise["multiplicative"] is not None:  # multiplicative noise as in Hoffer et al.
            [p.grad.mul_(1 + cfg.hyp.grad_noise["multiplicative"] * torch.randn_like(p)) for p in model.parameters()]

    def gradient_evaluation():
        """This is a full-blown closure that is passed to the optimizer."""
        # Gradient evaluation part:
        model.to(dtype=setup["dtype"])
        # this may move the model to a different precision
        loss = _accumulate_full_gradient(trainloader, stats)
        # Modify gradient:
        _modify_gradient_params()
        return loss

    file = os.path.join(cfg.original_cwd, "checkpoints", cfg.impl.checkpoint.name)
    _, model_state, _, _, step = torch.load(file, map_location=setup["device"])
    model.load_state_dict(model_state)
    model.train()
    log.info(f"Loaded model checkpoint from step {step} successfully.")
    loss1 = gradient_evaluation()
    grads_1 = [p.grad.detach().clone() for p in model.parameters()]
    print(f"Completed first pass with loss {loss1.item()}.")

    file = os.path.join(cfg.original_cwd, "checkpoints", cfg.impl.checkpoint.name)
    _, model_state, _, _, step = torch.load(file, map_location=setup["device"])
    model.load_state_dict(model_state)
    model.train()
    log.info(f"Loaded model checkpoint from step {step} successfully.")
    loss2 = gradient_evaluation()
    grads_2 = [p.grad.detach().clone() for p in model.parameters()]
    print(f"Completed first pass with loss {loss2.item()}.")

    norm_linf = torch.stack([g.max() for g in grads_1]).max()
    norm_l2 = torch.stack([g.pow(2).sum() for g in grads_1]).sum().sqrt()
    norm_l1 = torch.stack([g.abs().sum() for g in grads_1]).sum()

    print(f"Gradient Norms | L^Inf: {norm_linf.item()} | L2: {norm_l2.item()} | L1: {norm_l1.item()}.")

    error_linf = torch.stack([(g1 - g2).max() for g1, g2 in zip(grads_1, grads_2)]).max()
    error_l2 = torch.stack([(g1 - g2).pow(2).sum() for g1, g2 in zip(grads_1, grads_2)]).sum().sqrt()
    error_l1 = torch.stack([(g1 - g2).abs().sum() for g1, g2 in zip(grads_1, grads_2)]).sum()

    print(f"Error in L^inf Norm: Total: {error_linf.item()} | Relative: {(error_linf / norm_linf).item()}.")
    print(f"Error in L^2 Norm: Total: {error_l2.item()} | Relative: {(error_l2 / norm_l2).item()}.")
    print(f"Error in L^1 Norm: Total: {error_l1.item()} | Relative: {(error_l1 / norm_l1).item()}.")

    return
