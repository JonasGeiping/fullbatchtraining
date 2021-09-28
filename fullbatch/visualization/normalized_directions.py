"""Compute normalized (random) directions for a given neural network.

A rewritten version of https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py.
"""
import torch

def compute_randomized_directions(model, cfg_viz):

    @torch.no_grad()
    def _get_randomized_direction(model):
        base_direction = [torch.randn_like(p) for p in model.parameters()]
        for d, w in zip(base_direction, model.parameters()):
            if d.dim() <= 1:
                if cfg_viz.ignore_layers == 'biasbn':
                    d.fill_(0)  # ignore directions for weights with 1 dimension
                else:
                    d.copy_(w)  # keep directions for weights/bias that are only 1 per node
            else:
                normalize_direction(d, w, cfg_viz.norm)
        return base_direction

    x_direction = _get_randomized_direction(model)
    y_direction = _get_randomized_direction(model)
    return x_direction, y_direction


"""This is the normalize function from https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py."""
################################################################################
#                        Normalization Functions
################################################################################


def normalize_direction(direction, weights, norm='filter', eps=1e-10):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.
        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm() / (d.norm() + eps))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm() / direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + eps)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())
