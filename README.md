# Stochastic Training is Not Necessary for Generalization -- Training competitive vision models without stochasticity
This repository implements training routines to train vision models for classification on CIFAR-10 without SGD as described in our publication https://arxiv.org/abs/2109.14119.

## Abstract:
It is widely believed that the implicit regularization of SGD is fundamental to the impressive generalization behavior we observe in neural networks.  In this work, we demonstrate that non-stochastic full-batch training can achieve comparably strong performance to SGD on CIFAR-10 using modern architectures.  To this end, we show that the implicit regularization of SGD can be completely replaced with explicit regularization.  Our observations indicate that the perceived difficulty of full-batch training is largely the result of its optimization properties and the disproportionate time and effort spent by the ML community tuning optimizers and hyperparameters for small-batch training.

## Requirements
* PyTorch 1.9.*
* Hydra 1.* (via `pip install hydra-core --upgrade`)
* python-lmdb (only for N x CIFAR experiments.)

Pytorch 1.9.* is used for multithreaded persistent workers, `_for_each_`, functionality, and `torch.inference_mode`, all of which have to be replaced when using earlier versions.


## Training Runs

Scripts for training runs can be found in `train.sh`

## Pretrained Models

While this project is mostly about analysis of models trained in the full batch setting, we do now provide a few model checkpoints. Ideally at some point all experiments of interest from the paper will be downloadadable as checkpoints. We hope this helps with further empirical analysis of these types of models.

Models can be loaded via `torch.hub` without having to install this repository manually in the following way:
```
model = torch.hub.load("JonasGeiping/fullbatchtraining", "resnet18_fbaug_highreg", pretrained=True)  # resnet18 with strong reg. (no shuffle)
model = torch.hub.load("JonasGeiping/fullbatchtraining", "resnet152_fbaug_highreg", pretrained=True)   # resnet152 with shuffle
```

All currently available checkpoints can be listed with `torch.hub.list("jonasgeiping/fullbatchtraining")`.


## Guide

The main script for fullbatch training should be `train_with_gradient_descent.py`. This scripts also runs the stochastic gradient descent sanity check with the same codebase. A crucial flag to distinguish between both settings is `hyp.train_stochastic=False`. The gradient regularization is activated by setting `hyp.grad_reg.block_strength=0.5`. Have a look at the various folders under `config` to see all options. The script `crunch_loss_landscape.py` can measure the loss landscape of checkpointed models which can be used later for visualization.

If you are interesting in extracting components from this repository, you can use the gradient regularization separately, which can be found under `fullbatch/models/modules.py` in the class `GradRegularizer`. This regularizer can be instantiated like so:
```
gradreg = GradRegularizer(model, optimizer, loss_fn, block_strength=0.5, acc_strength=0.0, eps=1e-2,
                          implementation='finite_diff', mixed_precision=False)
```
and then used during training to modify a list of gradients (such as from `torch.autograd.grad`) in-place:
```
grads = gradreg(grads, inputs, labels, pre_grads=None)
```

## Contact

Just open an issue here on github or send us an email if you have any questions.
