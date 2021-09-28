"""Parse config files and load appropriate model."""
import torch

from contextlib import nullcontext
import os

from .resnets import ResNet, resnet_depths_to_config
from .densenets import DenseNet, densenet_depths_to_config
from .nfnets import NFNet
from .vgg import VGG


def construct_model(cfg_model, channels, classes):
    """cfg_model templates can be found under config/model."""

    if 'resnet' in cfg_model.name.lower():
        block, layers = resnet_depths_to_config(cfg_model.depth)
        model = ResNet(block, layers, channels, classes, stem=cfg_model.stem, convolution_type=cfg_model.convolution,
                       nonlin=cfg_model.nonlin_fn, norm=cfg_model.normalization,
                       downsample=cfg_model.downsample, width_per_group=cfg_model.width,
                       zero_init_residual=True if 'skip_residual' in cfg_model.initialization else False)
    elif 'densenet' in cfg_model.name.lower():
        growth_rate, block_config, num_init_features = densenet_depths_to_config(cfg_model.depth)
        model = DenseNet(growth_rate=growth_rate,
                         block_config=block_config,
                         num_init_features=num_init_features,
                         bn_size=cfg_model.bn_size,
                         drop_rate=cfg_model.drop_rate,
                         channels=channels,
                         num_classes=classes,
                         memory_efficient=cfg_model.memory_efficient,
                         norm=cfg_model.normalization,
                         nonlin=cfg_model.nonlin_fn,
                         stem=cfg_model.stem,
                         convolution_type=cfg_model.convolution)
    elif 'vgg' in cfg_model.name.lower():
        model = VGG(cfg_model.name, in_channels=channels, num_classes=classes, norm=cfg_model.normalization,
                    nonlin=cfg_model.nonlin_fn, head=cfg_model.head, convolution_type=cfg_model.convolution,
                    drop_rate=cfg_model.drop_rate, classical_weight_init=cfg_model.classical_weight_init)
    elif 'linear' in cfg_model.name:
        model = torch.nn.Sequential(torch.nn.Flatten(), _Select(
            100), torch.nn.Linear(100, classes))  # for debugging only

    elif 'nfnet' in cfg_model.name:
        model = NFNet(channels, classes, variant=cfg_model.variant, stochdepth_rate=cfg_model.stochdepth_rate,
                      alpha=cfg_model.alpha, se_ratio=cfg_model.se_ratio, activation=cfg_model.nonlin, stem=cfg_model.stem,
                      use_dropout=cfg_model.use_dropout)

    return model


def prepare_model(model, cfg, process_idx, setup):
    model.to(**setup)
    if cfg.impl.JIT == 'trace':  # only rarely possible
        with torch.cuda.amp.autocast(enabled=cfg.impl.mixed_precision):
            template = torch.zeros([cfg.data.batch_size, cfg.data.channels, cfg.data.pixels, cfg.data.pixels]).to(**setup)
            model = torch.jit.trace(model, template)
    elif cfg.impl.JIT == 'script':
        model = torch.jit.script(model)
    if cfg.impl.setup.dist:
        # if cfg.hyp.train_stochastic:
        #     # Use DDP only in stochastic mode
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[process_idx],
        #                                                       output_device=process_idx, broadcast_buffers=False)
        # else:
        # For both full-batch and stochastic we'll run our own distributed solution, to be able to switch more efficiently
        for param in model.parameters():
            torch.distributed.broadcast(param.data, 0, async_op=True)
        torch.distributed.barrier()
    else:
        model.no_sync = nullcontext

    os.makedirs(os.path.join(cfg.original_cwd, 'checkpoints'), exist_ok=True)

    return model


class _Select(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :self.n]
