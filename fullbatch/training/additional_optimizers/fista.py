"""
Custom implementation of Fast Iterative Thresholding (FISTA), which is basically Nesterov acceleration gradient descent.

Implementations from https://github.com/JonasGeiping/ParametricMajorization/blob/master/bilevelsurrogates/optim/optim.py
"""
import torch
import torch.optim
import numpy as np

from torch.optim.optimizer import required


class FISTA(torch.optim.Optimizer):
    """
    Implement the FISTA algorithm, or FISTA-MOD
    borrows heavily from the pytorch ADAM implementation!

    This variant explictely constructs the a_k sequence (which is fixed for the default optim.SGD(nesterov=True))
    and allows for variations of the sequence as described in https://arxiv.org/pdf/1807.04005.pdf.
    fista_mod = (p, q, r)
    Also possible: p=1/50, q=1/10 [Lazy FISTA]
    r = 4 should rarely be changed.
    """

    def __init__(self, params, projection=None, lr=1e-4,
                 fista_mod=(1.0, 1.0, 4.0)):
        """
        This requires that projetion is a function handle to be applied to
        the parameter
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if projection is None:
            self.projection = None
        else:
            self.projection = projection

        defaults = dict(lr=lr, fista_mod=fista_mod, projection=projection)
        super(FISTA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FISTA, self).__setstate__(state)

    def step(self, closure=None):
        """
        Single optimization step
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data

                state = self.state[param]

                # Initialization of State:
                if len(state) == 0:
                    state['x+'] = param.clone().detach()
                    state['x-'] = param.clone().detach()
                    state['tk'] = param.new_ones(1, requires_grad=False)

                # Gradient step
                state['x+'] = param.data - grad * group['lr']
                if group['projection'] is not None:
                    state['x+'] = group['projection'](state['x+'])

                # Overrelaxation factor
                p_factor, q_factor, r_factor = group['fista_mod']
                tk = (p_factor + torch.sqrt(q_factor + r_factor * state['tk']**2)) / 2
                ak = (state['tk'] - 1) / tk
                state['tk'] = tk

                # The actual parameter corresponds to 'yk'
                param.data = state['x+'] * (1 + ak) - state['x-'] * ak
                state['x-'].data = state['x+'].clone()

        return loss


class FISTALineSearch(torch.optim.Optimizer):
    """
    Implement the FISTA algorithm, or FISTA-MOD
    borrows heavily from the pytorch ADAM implementation!
    with added linesearch
    """

    def __init__(self, params, projection=None, lr=10, eta=0.8,
                 max_searches=25, fista_mod=(1.0, 1.0, 4.0), tk=1.0):
        """
        This requires that projection is a function handle to be applied to the
        parameter
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if projection is None:
            self.projection = lambda x: x
        else:
            self.projection = projection

        defaults = dict(lr=lr, eta=eta, max_searches=max_searches,
                        fista_mod=fista_mod, projection=projection, tk=tk)
        super(FISTALineSearch, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FISTALineSearch, self).__setstate__(state)

    def step(self, closure=None):
        """
        Single optimization step
        but backtracking is done for every group, many groups will thus lead to
        a long computation times.
        """

        if closure is None:
            raise ValueError('A closure is necessary, containing the keyword '
                             ' "requires_grad", which determines whether '
                             'loss.backward is called')
        else:
            loss = closure(requires_grad=True)

        for group in self.param_groups:

            # Phase 0: Update overrelaxation factor
            p_factor, q_factor, r_factor = group['fista_mod']
            tk = (p_factor + np.sqrt(q_factor + r_factor * group['tk']**2)) / 2
            ak = (group['tk'] - 1) / tk
            group['tk'] = tk

            # Phase I Linesearch for largest possible lr
            loss_yk = closure(requires_grad=False)

            for searches in range(group['max_searches']):
                linearization = group['params'][0].new_zeros(1)
                distance = group['params'][0].new_zeros(1)
                for param in group['params']:
                    if param.grad is None:
                        continue
                    grad = param.grad.data
                    state = self.state[param]

                    # Initialization of State:
                    if len(state) == 0:
                        state['x-'] = param.clone().detach()

                    # Gradient step
                    state['yk'] = param.data.clone().detach()
                    param.data -= grad * group['lr']
                    if self.projection is not None:
                        param.data = self.projection(param.data)
                    linearization += torch.sum(grad * (param.data - state['yk']))
                    distance += torch.sum((param.data - state['yk'])**2) / 2

                loss_xk = closure(requires_grad=False)
                D_h_xk_yk = loss_xk - loss_yk - linearization

                # Check lineseach condition:
                if D_h_xk_yk * group['lr'] > distance:
                    # Reduce lr:
                    group['lr'] *= group['eta']
                    # Undo gradient step
                    # If we had no projection this would be easier done via
                    # adding the gradient back on
                    for param in group['params']:
                        if param.grad is None:
                            continue
                        param.data = self.state[param]['yk'].clone()

                else:
                    # Break loop and continue to next phase
                    break

            # Phase II - Overrelaxation Step
            for param in group['params']:
                if param.grad is None:
                    continue
                # Get param state
                state = self.state[param]

                # The value of param is currently x^{k+1}, due to the step
                param_xp = param.data.clone()

                # The actual parameter corresponds to 'yk'
                param.data = param.data * (1 + ak) - state['x-'] * ak
                state['x-'].data = param_xp

        return loss


class SGDLineSearch(torch.optim.Optimizer):
    r"""
    WHAT FOLLOWS IS A   1to1 version of the pytorch SGD fom
    https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
    adapted to do linesearch

    ---------------------------------------------------------------------------
    Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                        momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, alpha=0.2, beta=0.5):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        alpha=alpha, beta=beta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum "
                             " and zero dampening")
        super(linesearchSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(linesearchSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        if closure is None:
            raise ValueError('A closure is necessary.')
        loss_prime = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # First candidate step
                p.data.add_(-group['lr'], d_p)
                with torch.no_grad():
                    loss = closure(backward=False)

                # This is a short hack and not a good idea for multiple groups
                threshold = loss_prime - group['alpha'] * group['lr'] * d_p.norm()

                for i in range(25):
                    if loss > threshold:
                        # remove old gradient
                        p.data.add_(group['lr'], d_p)
                        # Update learning rate
                        group['lr'] *= group['beta']
                        # Take a new step
                        p.data.add_(-group['lr'], d_p)
                        with torch.no_grad():
                            loss = closure(backward=False)
                    else:
                        break
                print(i)

        return loss
