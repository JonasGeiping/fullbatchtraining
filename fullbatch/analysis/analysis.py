"""Defines the main analysis function that can be called to analyze the current state of a model."""

import torch
from .welford import WelfordAccumulation
from .rollouts import perturb2threshold
from ..utils import get_log


def analyze(model, loss_fn, optimizer, dataloader, stats, setup, cfg):
    """Collect some statistics about the current model.

    This function requires knowledge of cfg.analysis, but also cfg.impl and cfg.data.

    For analysis purposes the data.batch_size rules and defines the data included in a "single" sample of a model gradient.
    """
    log = get_log(cfg)
    model.eval()
    # needs to be reshape for channels_last:
    param_vector = torch.cat([param.cpu().reshape(-1) for param in model.parameters()])

    if cfg.analysis.measure_param_norm:
        stats['analysis_param_norm'] += [param_vector.norm().item()]  # Saxe

    if cfg.analysis.measure_grad_norm:
        norm_type = cfg.hyp.grad_clip_norm
        try:
            if norm_type == float('inf'):
                stats['analysis_grad_norm'] += [max(p.grad.abs().max() for p in model.parameters())]
            else:
                stats['analysis_grad_norm'] += [torch.norm(torch.stack([torch.norm(p.grad, norm_type) for p in model.parameters()]),
                                                           norm_type).item()]  # this is the "pytorch" norm-of-norm l2-norm
        except AttributeError:  # sometimes not all gradients have been recorded
            stats['analysis_grad_norm'] += [float('NaN')]

    if cfg.analysis.check_momentum:
        if cfg.hyp.optim.momentum > 0:
            grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
            momentum = torch.cat([optimizer.state[p]['momentum_buffer'].reshape(-1) for p in model.parameters()])

            stats['analysis_momentum_dist'] += [torch.linalg.norm(grad - momentum).item()]
            stats['analysis_momentum_sim'] += [((grad * momentum).sum() / grad.norm() / momentum.norm()).item()]

    # Do we want advanced gradient statistics (mean/variance)?
    if cfg.analysis.compute_gradient_SNR or cfg.analysis.compute_gradient_noise_scale or cfg.analysis.record_gradient_norm_per_batch:
        grads = []
        num_blocks = len(dataloader)

        collector = WelfordAccumulation()

        def collect_gradients(inputs, labels):
            inputs = inputs.to(**setup, non_blocking=cfg.impl.non_blocking)
            labels = labels.to(dtype=torch.long, device=setup['device'], non_blocking=cfg.impl.non_blocking)

            with torch.cuda.amp.autocast(enabled=cfg.impl.mixed_precision):
                outputs = model(inputs)
                block_loss = loss_fn(outputs, labels) / num_blocks

            grad_list = torch.autograd.grad(block_loss, model.parameters())
            grad_vector = torch.cat([g.detach().cpu().reshape(-1) for g in grad_list])
            collector(grad_vector)
            return grad_vector.norm()

        # Compute gradient informations [this is a limited sample in a DDP distributed setting]
        subblock_counter = 0
        grad_norms = torch.zeros(num_blocks * cfg.analysis.internal_batch_size_chunks,
                                 device=setup['device'], dtype=setup['dtype'])
        for block, (inputs, labels) in enumerate(dataloader):
            input_chunks = torch.chunk(inputs, cfg.analysis.internal_batch_size_chunks, dim=0)
            label_chunks = torch.chunk(labels, cfg.analysis.internal_batch_size_chunks, dim=0)
            for input, label in zip(input_chunks, label_chunks):
                grad_norms[subblock_counter] = collect_gradients(input, label)
                subblock_counter += 1

        # Itemization to records separated from computations to improve GPU queuing:
        if cfg.analysis.record_gradient_norm_per_batch:
            for i in range(subblock_counter):
                stats[f'analysis_grad_norm_{i}'] += [grad_norms[i].item()]
        grad_mean, grad_variance, grad_std, grad_norm, squared_norm = collector.finalize()

        if cfg.analysis.compute_gradient_SNR:
            stats['analysis_grad_mean_mean'] += [grad_mean.mean().item()]  # Saxe
            stats['analysis_grad_mean_norm'] += [grad_mean.norm().item()]  # Saxe
            stats['analysis_grad_std_mean'] += [grad_std.mean().item()]  # Saxe
            stats['analysis_grad_std_norm'] += [grad_std.norm().item()]  # Saxe
            stats['analysis_grad_SNR'] += [stats['grad_mean_norm'][-1] / (stats['grad_std_norm'][-1] + 1e-10)]
            log.info(f'Gradient SNR is {stats["grad_SNR"][-1]}')

        if cfg.analysis.compute_gradient_noise_scale:
            b_local = cfg.data.batch_size // cfg.analysis.internal_batch_size_chunks
            b_full = max(len(dataloader.dataset), cfg.data.size)  # Dataset size might have been artifically inflated
            g_local = squared_norm
            g_full = grad_mean.pow(2).sum()

            candlish_S = 1 / (1 / b_local - 1 / b_full + 1e-10) * (g_local - g_full)
            candlish_G = 1 / (b_full - b_local + 1e-10) * (b_full * g_full - b_local * g_local)
            stats['analysis_grad_noise_scale'] += [(candlish_S / candlish_G).item()]
            log.info(f'Gradient Noise Scale is {stats["grad_noise_scale"][-1]}')

    if cfg.analysis.compute_flatness:
        empirical_flatness, counter = perturb2threshold(model, dataloader, torch.nn.CrossEntropyLoss(reduction='sum'),
                                                        setup, step_size=cfg.analysis.flatness_step_size,
                                                        threshold=cfg.analysis.flatness_threshold,
                                                        norm=cfg.analysis.flatness_norm, ignore='biasbn', dryrun=cfg.dryrun)
        stats['analysis_empirical_flatness'] += [empirical_flatness]
        log.info(f'Empirical flatness from random directions with threshold {cfg.analysis.flatness_threshold} '
                 f'is {stats["empirical_flatness"][-1]} after {counter} steps.')

    model.train()
