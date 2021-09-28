"""Some tests with adaptive clipping (adaptive across iterations)."""

import torch

class AdaptiveGradientClipping(torch.optim.SGD):
    r"""Implements gradient descent with momentum with adaptive gradient clipping.
    The gradient norm is forced to be at most as large as the last 10 gradient norms.

    This is often a bad idea, proceed with caution.

    The parameters "interval" and "norm_type" are global and the values of group[0] will always be used.
    """

    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, interval=10, norm_type=2):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov,
                        weight_decay=weight_decay, interval=interval, norm_type=float(norm_type))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        torch.optim.Optimizer.__init__(self, params, defaults)

    def _compute_gradient_norm(self):
        global_group = self.param_groups[0]
        norm_type = global_group['norm_type']
        if norm_type == float('inf'):
            grad_norm = max(p.grad.abs().max() for group in self.param_groups for p in group['params'])
        else:
            norm_stack = torch.stack([torch.norm(p.grad, norm_type) for group in self.param_groups for p in group['params']])
            grad_norm = torch.norm(norm_stack, norm_type)
        return grad_norm.detach()

    def _scale_gradients(self, current_gradient_norm, grad_target_norm):
        scale_factor = grad_target_norm / (current_gradient_norm + 1e-6)
        for group in self.param_groups:
            torch._foreach_mul_(group['params'], scale_factor)

    @torch.no_grad()
    def step(self, closure):
        """Perform a step with optional gradient scaling.
        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        # Abuse group0, param0 for global state:
        global_group = self.param_groups[0]

        with torch.enable_grad():
            loss = closure()

        grad_norm = self._compute_gradient_norm()

        if self.state.get('norms') is None:
            self.state['norms'] = [grad_norm]

        if len(self.state['norms']) < global_group['interval']:
            # Accumulate norm values
            self.state['norms'].append(grad_norm)
            super().step()
            return
        else:
            # Check and recommend scaling here
            recent_norm_max = max(self.state['norms'][-global_group['interval']:])

            if grad_norm < recent_norm_max:
                self.state['norms'].append(grad_norm)
                return
            else:
                print(f'Recent maximum grad norm was {recent_norm_max}, but new norm is {grad_norm.item()}. Rescaling ...')
                self._scale_gradients(grad_norm, recent_norm_max)
                super().step()
                return
