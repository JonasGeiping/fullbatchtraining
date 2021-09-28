# Training competitive vision models without stochasticity

## Requirements
* PyTorch 1.9.*
* Hydra 1.* (via `pip install hydra-core --upgrade`)
* python-lmdb (only for N x CIFAR experiments.)

Pytorch 1.9.* is used for multithreaded persistent workers, `_for_each_`, functionality, and `torch.inference_mode`, all of which have to be replaced when using earlier versions.


## Training Runs

Scripts for training runs can be found in `train.sh`


## Loading pretrained models
Pretrained models can be loaded via
```python
model = torch.hub.load('JonasGeiping/fullbatch[:v1]', 'resnet18_highreg',
                       pretrained=True)
```
