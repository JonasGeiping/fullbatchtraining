"""Interface with optimizers."""

import re
import torch

from .additional_optimizers import FISTA, LARS, SGD_AGC, LBFGS, SAM, GradualWarmupScheduler
from .additional_optimizers import RestartingLineSearch, NonMonotoneLinesearch, WolfeGradientDescent
from .additional_optimizers import AdaptiveGradientClipping

def optim_interface(model, cfg_hyp):
    """Construct optimizer and scheduler objects."""
    optim_params = {k: v for k, v in cfg_hyp.optim.items() if k != 'name'}

    if cfg_hyp.only_linear_layers_weight_decay and not cfg_hyp.optim.name == 'GD-AGC':
        parameter_iterable = []
        for key, value in model.named_parameters():
            # this regex snippet is modified from https://github.com/benjs/nfnets_pytorch/blob/master/train.py
            if len(re.findall('(bias|gain)|skip_gain', key)) > 0:
                parameter_iterable += [{'params': [value], 'weight_decay':0.0}]
            else:
                parameter_iterable += [{'params': [value]}]
    else:
        parameter_iterable = model.parameters()

    if cfg_hyp.optim.name == 'Gradient Descent':
        optim_params = {k: v for k, v in optim_params.items() if k != 'line_search'}
        if cfg_hyp.optim.line_search == 'none':
            optimizer = torch.optim.SGD(parameter_iterable, **optim_params)
        elif cfg_hyp.optim.line_search == 'wolfe':
            optimizer = WolfeGradientDescent(parameter_iterable, **optim_params)
        elif cfg_hyp.optim.line_search == 'non-monotone':
            optimizer = NonMonotoneLinesearch(parameter_iterable, **optim_params)
        elif cfg_hyp.optim.line_search == 'restarting':
            optimizer = RestartingLineSearch(parameter_iterable, **optim_params)
        else:
            raise ValueError(f'Invalid linesearch {cfg_hyp.optim.line_search} defined.')
    elif cfg_hyp.optim.name == 'Adaptive Gradient Descent':
        optimizer = AdaptiveGradientClipping(parameter_iterable, **optim_params)
    elif cfg_hyp.optim.name == 'Adam':
        optimizer = torch.optim.AdamW(parameter_iterable, **optim_params)
    elif cfg_hyp.optim.name == 'L-BFGS':
        optimizer = LBFGS(parameter_iterable, **optim_params)
    elif cfg_hyp.optim.name == 'FISTA':
        optimizer = FISTA(parameter_iterable, **optim_params)
    elif cfg_hyp.optim.name == 'GD-AGC':
        optimizer = SGD_AGC(model.named_parameters(), **optim_params)
        for group in optimizer.param_groups:
            if group['name'].startswith('linear'):
                group['clipping'] = None
            if cfg_hyp.only_linear_layers_weight_decay:
                if len(re.findall('stem.*(bias|gain)|conv.*(bias|gain)|skip_gain', group['name'])) > 0:
                    group['weight_decay'] = 0
    else:
        raise ValueError(f'Invalid optimizer {cfg_hyp.optim.name} provided.')

    if cfg_hyp.optim_modification.name == 'none':
        optimizer_to_schedule = optimizer
    else:
        if cfg_hyp.optim_modification.name == 'LARS':
            optimizer = LARS(optimizer, trust_coefficient=cfg_hyp.optim_modification.trust_coefficient,
                             clip=False, eps=cfg_hyp.optim_modification.eps)
        elif cfg_hyp.optim_modification.name == 'LARC':
            optimizer = LARS(optimizer, trust_coefficient=cfg_hyp.optim_modification.trust_coefficient,
                             clip=True, eps=cfg_hyp.optim_modification.eps)
        elif cfg_hyp.optim_modification.name == 'SAM':
            optimizer = SAM(optimizer, rho=cfg_hyp.optim_modification.rho)
        optimizer_to_schedule = optimizer.optim

    if cfg_hyp.scheduler == 'linear':
        # Drop at 5/8, 6/8, 7/8:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_to_schedule,
                                                         milestones=[cfg_hyp.steps // 2.667, cfg_hyp.steps // 1.6,
                                                                     cfg_hyp.steps // 1.142], gamma=0.1)
    elif cfg_hyp.scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_to_schedule, gamma=0.99)
    elif cfg_hyp.scheduler == 'cosine-decay-floored':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_to_schedule, cfg_hyp.steps, eta_min=cfg_hyp.optim.lr / 25)
    elif cfg_hyp.scheduler == 'cosine-decay':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_to_schedule, cfg_hyp.steps, eta_min=0.0)
    elif cfg_hyp.scheduler == 'cosine-4000':
        # Cosine decay, hardcoded to 4000 steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_to_schedule, 4000, eta_min=0.0)
    elif cfg_hyp.scheduler in ['', ' ', None]:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_to_schedule, milestones=[], gamma=1)
    else:
        raise ValueError(f'Invalid scheduler {cfg_hyp.scheduler} provided.')

    if cfg_hyp.warmup > 0:
        scheduler = GradualWarmupScheduler(optimizer_to_schedule, multiplier=1.0,
                                           total_epoch=cfg_hyp.warmup, after_scheduler=scheduler)

    return optimizer, scheduler
