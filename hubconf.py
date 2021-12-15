import torch

# Optional list of dependencies required by the package
from fullbatch.models.resnets import resnet_depths_to_config, ResNet

dependencies = ["torch"]

names = ["highreg"]
url = "https://github.com/JonasGeiping/fullbatchtraining/releases/download/v1/"
model_urls = {
    "final_fbaug_highreg_lr08_resnet18": url + "final_fbaug_highreg_lr08_resnet18" + ".pth",
    "final_fbaug_gradreg_lr08_resnet18": url + "final_fbaug_gradreg_lr08_resnet18" + ".pth",
    "final_fbaug_gradreg_lr16_resnet18": url + "final_fbaug_gradreg_lr16_resnet18" + ".pth",
    "final_fbaug_clip_lr04_resnet18": url + "final_fbaug_clip_lr04_resnet18" + ".pth",
    "final_fbaug_highreg_lr08_shuffle_resnet152": url + "final_fbaug_highreg_lr08_shuffle_resnet152" + ".pth",
}


def _resnet18(name, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 with default config used in this repo"""
    # Architecture:
    block, layers = resnet_depths_to_config(18)
    model = ResNet(
        block,
        layers,
        channels=3,
        classes=10,
        stem="CIFAR",
        convolution_type="Standard",
        nonlin="ReLU",
        norm="BatchNorm2d",
        downsample="C",
        width_per_group=64,
        zero_init_residual="skip-residual",
    )
    if pretrained:
        _, state_dict, _, _, _ = torch.hub.load_state_dict_from_url(
            model_urls[name], progress=progress, map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)

    return model


def _resnet152(name, pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 with default config used in this repo"""
    # Architecture:
    block, layers = resnet_depths_to_config(152)
    model = ResNet(
        block,
        layers,
        channels=3,
        classes=10,
        stem="CIFAR",
        convolution_type="Standard",
        nonlin="ReLU",
        norm="BatchNorm2d",
        downsample="C",
        width_per_group=64,
        zero_init_residual="skip-residual",
    )
    if pretrained:
        _, state_dict, _, _, _ = torch.hub.load_state_dict_from_url(
            model_urls[name], progress=progress, map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)

    return model


def resnet18_fbaug_clip(pretrained=False, progress=True, **kwargs):
    r"""Loads a Resnet18 model pretrained with fullbatch gradient descent with "clip" hyperparams
    trained with data augmentations but without data shuffling as described in section 3."""
    return _resnet18("final_fbaug_clip_lr04_resnet18", pretrained, progress, **kwargs)


def resnet18_fbaug_gradreg(pretrained=False, progress=True, **kwargs):
    r"""Loads a Resnet18 model pretrained with fullbatch gradient descent with "gradreg" hyperparams
    trained with data augmentations but without data shuffling as described in section 3."""
    return _resnet18("final_fbaug_gradreg_lr08_resnet18", pretrained, progress, **kwargs)


def resnet18_fbaug_gradreg_v2(pretrained=False, progress=True, **kwargs):
    r"""Loads a Resnet18 model pretrained with fullbatch gradient descent with "gradreg" hyperparams,
    but a doubled learning rate compared to the arxiv version,
    trained with data augmentations but without data shuffling as described in section 3."""
    return _resnet18("final_fbaug_gradreg_lr16_resnet18", pretrained, progress, **kwargs)


def resnet18_fbaug_highreg(pretrained=False, progress=True, **kwargs):
    r"""Loads a Resnet18 model pretrained with fullbatch gradient descent with "highreg" hyperparams
    trained with data augmentations but without data shuffling as described in section 3."""
    return _resnet18("final_fbaug_highreg_lr08_resnet18", pretrained, progress, **kwargs)


def resnet152_fbaug_highreg(pretrained=False, progress=True, **kwargs):
    r"""Loads a Resnet152 model pretrained with fullbatch gradient descent with "highreg" hyperparams
    trained with data augmentations and data shuffling as described in section 3."""
    return _resnet152("final_fbaug_highreg_lr08_shuffle_resnet152", pretrained, progress, **kwargs)
