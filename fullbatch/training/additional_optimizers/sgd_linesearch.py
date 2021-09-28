"""This implementation is a minor modification of the original pytorch SGD implementation

found at https://github.com/pytorch/pytorch/blob/master/torch/optim/_multi_tensor/sgd.py
"""

import torch

from math import copysign, sqrt
from collections import defaultdict


class RestartingLineSearch(torch.optim.SGD):
    r"""Implements gradient descent (optionally with momentum) with interpolating Wolfe linesearch

    Reduce learning rate until the loss is smaller than the maximum over the last interval many steps
    """

    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, interval=10, factor=0.25, max_iter=10):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov,
                        weight_decay=weight_decay, interval=interval, factor=factor, max_iter=max_iter)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        torch.optim.Optimizer.__init__(self, params, defaults)

    def _save_initial_state(self):
        self.buffer = dict()
        self.momentum_buffer = dict()
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                self.buffer[param] = param.clone().detach()
                if self.state[param].get('momentum_buffer') is not None:
                    self.momentum_buffer[param] = self.state[param]['momentum_buffer'].clone().detach()
                else:
                    self.momentum_buffer[param] = None

    def _load_initial_state(self):
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                param.copy_(self.buffer[param])
                if self.momentum_buffer[param] is not None:
                    self.state[param]['momentum_buffer'].copy_(self.momentum_buffer[param])
                else:
                    self.state[param]['momentum_buffer'] = None

    def _reset_momentum(self):
        for group in self.param_groups:
            for param in group['params']:
                self.state[param]['momentum_buffer'] = torch.zeros_like(param)

    @torch.no_grad()
    def step(self, closure):
        """Perform a step with backtracking line-search.
        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        self._save_initial_state()
        # Abuse group0, param0 for global state:
        global_group = self.param_groups[0]

        with torch.enable_grad():
            loss = closure()
        super().step()

        if self.state.get('loss') is None:
            self.state['loss'] = [loss.item()]

        if len(self.state['loss']) < global_group['interval']:
            self.state['loss'].append(loss.item())
            return
        else:
            recent_loss_max = max(self.state['loss'][-global_group['interval']:])

            if loss.item() < recent_loss_max:
                self.state['loss'].append(loss.item())
                return
            else:
                print(f'Recent maximum was {recent_loss_max}, but new loss is {loss.item()}. Resetting momentum ...')
                # Repeat step with smaller lr
                self._load_initial_state()
                self._reset_momentum()
                super().step()


class NonMonotoneLinesearch(torch.optim.SGD):
    r"""Implements gradient descent (optionally with momentum) with interpolating Wolfe linesearch

    Reduce learning rate until the loss is smaller than the maximum over the last interval many steps
    """

    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, interval=10, factor=0.25, max_iter=10):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov,
                        weight_decay=weight_decay, interval=interval, factor=factor, max_iter=max_iter)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        torch.optim.Optimizer.__init__(self, params, defaults)

    def _save_initial_state(self):
        self.buffer = dict()
        self.momentum_buffer = dict()
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                self.buffer[param] = param.clone().detach()
                if self.state[param].get('momentum_buffer') is not None:
                    self.momentum_buffer[param] = self.state[param]['momentum_buffer'].clone().detach()
                else:
                    self.momentum_buffer[param] = None

    def _load_initial_state(self):
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                param.copy_(self.buffer[param])
                if self.momentum_buffer[param] is not None:
                    self.state[param]['momentum_buffer'].copy_(self.momentum_buffer[param])
                else:
                    self.state[param]['momentum_buffer'] = None

    def _reduce_lr(self, factor):
        """Implement this via .grad modification so that it resets after every iteration."""
        # for group in self.param_groups:
        #     group['lr'] *= factor
        for group in self.param_groups:
            for param in group['params']:
                param.grad *= factor

    @torch.no_grad()
    def step(self, closure):
        """Perform a step with backtracking line-search.
        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        self._save_initial_state()
        # Abuse group0, param0 for global state:
        global_group = self.param_groups[0]

        with torch.enable_grad():
            loss = closure()
        super().step()

        if self.state.get('loss') is None:
            self.state['loss'] = [loss.item()]

        if len(self.state['loss']) < global_group['interval']:
            self.state['loss'].append(loss.item())
            return
        else:
            recent_loss_max = max(self.state['loss'][-global_group['interval']:])

            for iteration in range(global_group['max_iter']):
                if loss.item() < recent_loss_max:
                    self.state['loss'].append(loss.item())
                    return
                else:
                    # Repeat step with smaller lr
                    self._load_initial_state()
                    print(f'Recent maximum was {recent_loss_max}, but new loss is {loss.item()}. '
                          f'Reducing lr by factor {global_group["factor"]}.')
                    self._reduce_lr(factor=global_group['factor'])
                    super().step()
                    with torch.enable_grad():
                        loss = closure()




class WolfeGradientDescent(torch.optim.Optimizer):
    r"""Implements gradient descent (optionally with momentum) with interpolating Wolfe linesearch

    This modification uses global c1, c2, max_iter from the first group
    """

    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, c1=1e-4, c2=0.9, alpha_max=10.0, max_iter=10):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        c1=c1, c2=c2, alpha_max=alpha_max, max_iter=max_iter)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        torch.optim.Optimizer.__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def _find_descent_direction(self):
        p_k_groups = []
        p_k_offset = 0
        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']

            p_k = []
            for i, param in enumerate(group['params']):
                grad = param.grad
                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)

                if momentum != 0:
                    buf = self.state[param].get('momentum_buffer')
                    if buf is None:
                        buf = torch.clone(grad).detach()
                        self.state[param]['momentum_buffer'] = buf
                    else:
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                    if nesterov:
                        p_k.append(-grad.add(buf, alpha=momentum))
                    else:
                        p_k.append(-buf)
                else:
                    p_k.append(-grad)
                p_k_offset += (grad * p_k[-1]).sum()

            p_k_groups.append(p_k)

        return p_k_groups, p_k_offset

    def _evaluate_phi(self, alpha, p_k_groups, closure, LUT_ref):
        if alpha not in LUT_ref:
            self._attempt_step(p_k_groups, alpha)
            with torch.enable_grad():
                loss = closure().item()
            phi_grad = self._get_phi_grad(p_k_groups).item()
            LUT_ref[alpha] = dict(val=loss, grad=phi_grad)
            return loss, phi_grad
        else:
            return LUT_ref[alpha]['val'], LUT_ref[alpha]['grad']

    def _get_phi_grad(self, p_k_groups):
        p_k_offset = 0
        for group, p_k in zip(self.param_groups, p_k_groups):
            for i, param in enumerate(group['params']):
                grad = param.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(param, alpha=group['weight_decay'])
                p_k_offset += (grad * p_k[i]).sum()
        return p_k_offset

    def _attempt_step(self, p_k_groups, alpha):
        if self.has_stepped:
            self._load_initial_state()

        for group, p_k in zip(self.param_groups, p_k_groups):
            torch._foreach_add_(group['params'], p_k, alpha=group['lr'] * alpha)
        self.has_stepped = True

    def _save_initial_state(self):
        self.buffer = dict()
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                self.buffer[param] = param.clone().detach()

    def _load_initial_state(self):
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                param.copy_(self.buffer[param])

    @torch.no_grad()
    def step(self, closure):
        """Perform a step with backtracking line-search.
        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        self.has_stepped = False
        self._save_initial_state()

        # Compute initial gradient and inital loss
        with torch.enable_grad():
            loss = closure()
        p_k, p_k_offset = self._find_descent_direction()  # This also updates the momentum term
        if p_k_offset > 0:
            print(f"phi'={p_k_offset} is positive. p_k is not a descent direction.")  # otherwise p_k is not a descent direction


        # Abuse group0, param0 for global state:
        global_group = self.param_groups[0]
        # init:
        if 'alpha' not in self.state:
            alpha = self.state['alpha'] = 1.0
        else:
            alpha = 1.0  # reload previous alpha?  #self.state['alpha']

        # Generate a look-up table for previously considered alpha values
        phi = defaultdict(dict)
        phi[0] = dict(val=loss.item(), grad=p_k_offset.item())

        prev_loss = float('inf')
        prev_alpha = 0
        for iteration in range(global_group['max_iter']):
            # Take the step to figure out phi(a_i)
            self._evaluate_phi(alpha, p_k, closure, phi)
            # Step prediction from here:
            sufficient_loss = phi[0]['val'] + global_group['c1'] * alpha * phi[0]['grad']
            if (phi[alpha]['val'] > sufficient_loss) or (phi[alpha]['val'] > prev_loss):
                alpha = self._zoom(prev_alpha, alpha, p_k, closure, phi)
                print(f'alpha={alpha} from rule 1 zoom.')
                break

            if abs(phi[alpha]['grad']) <= -global_group['c2'] * phi[0]['grad']:
                print(f'alpha={alpha} fulfills strong Wolfe.')
                break

            if phi[alpha]['grad'] >= 0:
                alpha = self._zoom(alpha, prev_alpha, p_k, closure, phi)
                print(f'alpha={alpha} from rule 2 zoom.')
                break

            # If both conditions work, we can increase the alpha toward alpha_max
            prev_alpha = alpha
            alpha = min(alpha * 2.5, global_group['alpha_max'])
            print(f'alpha={alpha} increased from {prev_alpha}.')
            if alpha == global_group['alpha_max']:
                break

        # Save the final alpha to global state for the next step
        self.state['alpha'] = alpha
        # There is no need to take a final step, the last step attempt counts as sucessful

    def _zoom(self, alpha_low, alpha_high, p_k, closure, phi):
        global_group = self.param_groups[0]

        for iteration in range(global_group['max_iter']):
            if abs(alpha_low - alpha_high) < 1e-4:  # Nocedal spinning in his chair :<
                return alpha_low
            alpha = self._interpolate(alpha_low, alpha_high, phi)
            print(f'alpha={alpha} interpolated from interval [{alpha_low}, {alpha_high}].')
            self._evaluate_phi(alpha, p_k, closure, phi)
            sufficient_loss = phi[0]['val'] + global_group['c1'] * alpha * phi[0]['grad']
            if (phi[alpha]['val'] > sufficient_loss) or (phi[alpha]['val'] > phi[alpha_low]['val']):
                alpha_high = alpha
            else:
                if (phi[alpha]['grad']) <= -global_group['c2'] * phi[0]['grad']:
                    return alpha

                if phi[alpha]['grad'] * (alpha_high - alpha_low) >= 0:
                    alpha_high = alpha_low
                alpha_low = alpha

        return self._interpolate(alpha_low, alpha_high, phi)

    @staticmethod
    def _interpolate(alpha1, alpha2, phi):
        """Cubic interpolation as in Nocedal and Wright"""
        # assert alpha1 < alpha2  # just let it fail terribly :>
        if alpha1 == alpha2:
            return alpha1
        quotient = (phi[alpha1]['val'] - phi[alpha2]['val']) / (alpha1 - alpha2)
        d_1 = phi[alpha1]['grad'] + phi[alpha2]['grad'] - 3 * quotient
        d_2 = copysign(1.0, alpha2 - alpha1) * sqrt(d_1**2 - phi[alpha1]['grad'] * phi[alpha2]['grad'])

        update_nom = phi[alpha2]['grad'] + d_2 - d_1
        update_denom = phi[alpha2]['grad'] - phi[alpha1]['grad'] + 2 * d_2
        return alpha2 - (alpha2 - alpha1) * update_nom / update_denom
