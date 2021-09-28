"""Manipulate network parameters and setup random directions with normalization. Stuff from Micah and Liam."""

import torch
import copy
import numpy as np
import torch.nn.utils as nnutils

################################################################################
#                 Supporting functions for weights manipulation
################################################################################


def npvec_to_tensorlist(vec, params):
    """ Convert a numpy vector to a list of tensor with the same dimensions as params
        Args:
            vec: a 1D numpy vector
            params: a list of parameters from net
        Returns:
            rval: a list of tensors with the same shape as params
    """
    loc = 0
    rval = []
    for p in params:
        numel = p.data.numel()
        rval.append(torch.from_numpy(vec[loc:loc + numel]).view(p.data.shape).float())
        loc += numel
    assert loc == vec.size, 'The vector has more elements than the net has parameters'
    return rval


def get_weights(net, setup=None):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data.to(dtype=setup['dtype'], device=setup['device']) for p in net.parameters()]


def set_weights(net, weights, directions=None, step=None, setup=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.to(device=p.device, dtype=p.dtype))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d * step for d in directions]
        for (p, w, d) in zip(net.parameters(), weights, changes):
            p.data = w.to(dtype=p.dtype, device=p.device) + d.to(dtype=p.dtype, device=p.device)


def set_states(net, states, directions=None, step=None):
    """
        Overwrite the network's state_dict or change it along directions with a step size.
    """
    if directions is None:
        net.load_state_dict(states)
    else:
        assert step is not None, 'If direction is provided then the step must be specified as well'
        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d * step for d in directions[0]]

        new_states = copy.deepcopy(states)
        assert (len(new_states) == len(changes))
        for (k, v), d in zip(new_states.items(), changes):
            d = torch.tensor(d)
            v.add_(d.type(v.type()))

        net.load_state_dict(new_states)


def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn_like(w) for w in weights]


def get_random_states(states):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's state_dict(), so one direction entry
        per weight, including BN's running_mean/var.
    """
    return [torch.randn_like(w) for k, w in states.items()]

################################################################################
#                        Normalization Functions
################################################################################


def normalize_direction(direction, weights, norm='filter'):
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
            d.mul_(w.norm() / (d.norm() + 1e-10))
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
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())
    elif norm == 'entire':
        # Rescale direction vector to have same norm as weight vector
        c = scaling_constant(direction, weights)
        for d in direction:
            d.mul_(torch.tensor([c]))


def make_unit_norm(direction):
    direction_norm = np.linalg.norm(nnutils.parameters_to_vector(direction).numpy())
    for d in direction:
        d.div_(direction_norm)


def scaling_constant(direction, weights):
    weights_norm = np.linalg.norm(nnutils.parameters_to_vector(weights).cpu().numpy())
    direction_norm = np.linalg.norm(nnutils.parameters_to_vector(direction).cpu().numpy())
    return weights_norm / direction_norm


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn', setup=None):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert(len(direction) == len(weights))
    if norm == 'filter':
        for d, w in zip(direction, weights):
            if d.dim() <= 1:
                if ignore == 'biasbn':
                    d.fill_(0)  # ignore directions for weights with 1 dimension
                else:
                    temp = torch.randn(w.size())
                    temp = temp.div_(torch.abs(temp))
                    d.copy_(w * (temp.to(**setup)))  # keep directions for weights/bias that are only 1 per node
            else:
                normalize_direction(d, w, norm)
    elif norm == 'layer':
        for d, w in zip(direction, weights):
            if d.dim() <= 1:
                if ignore == 'biasbn':
                    d.fill_(0)  # ignore directions for weights with 1 dimension
                else:
                    temp = torch.randn(w.size())
                    temp = temp * (w.norm() / temp.norm())
                    d.copy_(temp)  # keep directions for weights/bias that are only 1 per node
            else:
                normalize_direction(d, w, norm)
    elif norm == 'entire':
        scalar = scaling_constant(direction, weights)
        for d, w in zip(direction, weights):
            if d.dim() <= 1:
                if ignore == 'biasbn':
                    d.fill_(0)  # ignore directions for weights with 1 dimension
                else:
                    temp = torch.randn(w.size())
                    temp = temp * scalar
                    d.copy_(temp)  # keep directions for weights/bias that are only 1 per node
            else:
                normalize_direction(d, w, norm)
    else:
        for d, w in zip(direction, weights):
            if d.dim() <= 1:
                if ignore == 'biasbn':
                    d.fill_(0)  # ignore directions for weights with 1 dimension
                else:
                    temp = torch.randn(w.size())
                    temp = temp.div_(torch.abs(temp))
                    d.copy_(w * temp)  # keep directions for weights/bias that are only 1 per node


def normalize_directions_for_states(direction, states, norm='filter', ignore='ignore'):
    assert(len(direction) == len(states))
    for d, (k, w) in zip(direction, states.items()):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0)  # ignore directions for weights with 1 dimension
            else:
                d.copy_(w)  # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def ignore_biasbn(directions):
    """ Set bias and bn parameters in directions to zero """
    for d in directions:
        if d.dim() <= 1:
            d.fill_(0)

################################################################################
#                       Create directions
################################################################################


def create_random_direction(net, dir_type='weights', ignore='biasbn', norm='filter', setup=None):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.
        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'
        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    # random direction
    if dir_type == 'weights':
        weights = get_weights(net, setup=setup)  # a list of parameters.
        direction = get_random_weights(weights)
        normalize_directions_for_weights(direction, weights, norm, ignore, setup)
    elif dir_type == 'states':
        states = net.state_dict()  # a dict of parameters, including BN's running mean/var.
        direction = get_random_states(states)
        normalize_directions_for_states(direction, states, norm, ignore)

    return direction

################################################################################
#                       Perturb to threshold
################################################################################


def total_loss(loader, criterion, model, setup, dryrun=False):
    '''
    Do not use a loss function that averages over the batch.
    '''
    model.eval()
    running_average = 0
    num_samples = 0
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(**setup)
            targets = targets.to(dtype=torch.long, device=setup['device'])
            loss = criterion(model(inputs), targets)
            batch_size = inputs.shape[0]
            running_average = (num_samples / (num_samples + batch_size)) * running_average + loss.item() / (num_samples + batch_size)
            num_samples += batch_size

            if dryrun:
                break
        return running_average


def perturb2threshold(net, loader, criterion, setup, step_size=0.1, threshold=1.0,
                      norm='filter', ignore='biasbn', dryrun=False):
    net = net.to(**setup)
    direction = create_random_direction(net, dir_type='weights', ignore=ignore, norm=norm, setup=setup)
    direction_norm = torch.linalg.norm(torch.cat([d.reshape(-1) for d in direction])).item()
    counter = 0
    while True:
        loss = total_loss(loader, criterion, net, setup, dryrun=dryrun)
        # this_trajectory.append(loss)
        if loss > threshold:
            return direction_norm * counter, counter
        set_weights(net, get_weights(net, setup=setup), directions=direction, step=step_size, setup=setup)
        counter += 1
