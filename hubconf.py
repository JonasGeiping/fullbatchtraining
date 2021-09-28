# Optional list of dependencies required by the package
from fullbatch.models.resnets import resnet_depths_to_config, ResNet
dependencies = ['torch']

names = ['highreg']
model_urls = ['']


def _resnet18(name, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 with default config used in this repo"""
    depth = 18
    width = 64
    # Architecture specifics:
    stem = 'CIFAR'

    convolution = 'Standard'
    nonlin_fn = 'ReLU'
    normalization = 'BatchNorm2d'

    downsample = 'C'  # as in He et al., 2019
    initialization = 'skip-residual'

    block, layers = resnet_depths_to_config(depth)
    model = ResNet(block, layers, channels, classes, stem=stem, convolution_type=convolution,
                   nonlin=nonlin_fn, norm=normalization,
                   downsample=downsample, width_per_group=width,
                   zero_init_residual=True if 'skip_residual' in initialization else False)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[name], progress=progress)
        model.load_state_dict(state_dict)

    return model

def resnet18_highreg(pretrained=False, progress=True, **kwargs):
    r"""Loads a Resnet18 model pretrained with fullbatch gradient descent with "highreg" hyperparams."""
    return _resnet18('highreg', pretrained=False, progress=True, **kwargs)
