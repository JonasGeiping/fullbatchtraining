"""Additional modules."""
import torch


class Skipper(torch.nn.Module):
    """Semi-drop-in replacement for batchnorm."""

    def __init__(self, channels, initial_scale=0, gain=0.2):
        """Takes channels argument as input for compatibility without using it."""
        super().__init__()

        self.alpha = torch.nn.Parameter(torch.ones(()) * initial_scale)
        self.register_buffer('gain', torch.tensor(gain, requires_grad=False), persistent=False)

    def forward(self, inputs):
        return inputs * self.alpha * self.gain


class SequentialGhostNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, virtual_batch_size=64):
        super().__init__()
        self.batchnorm = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.virtual_batch_size = virtual_batch_size

        # spoof weights, biases
        self.weight = self.batchnorm.weight
        self.bias = self.batchnorm.bias

        self.reset_parameters()

    def forward(self, x):
        num_chunks = max(x.shape[0] // self.virtual_batch_size, 1)
        chunks = torch.chunk(x, num_chunks, 0)
        seq_normed = [self.batchnorm(chunk) for chunk in chunks]
        return torch.cat(seq_normed, 0)

    def reset_parameters(self):
        torch.nn.init.constant_(self.batchnorm.weight, 1)
        torch.nn.init.constant_(self.batchnorm.bias, 0)


class ParallelGhostNorm(torch.nn.modules.batchnorm._NormBase):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 num_chunks=16, virtual_batch_size=64):
        """Initialize transformation."""
        torch.nn.Module.__init__(self)
        self.num_features = num_features
        self.num_chunks = num_chunks

        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_chunks, 1, num_features, 1, 1))
            self.bias = torch.nn.Parameter(torch.Tensor(num_chunks, 1, num_features, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_chunks, 1, num_features, 1))
            self.register_buffer('running_var', torch.ones(num_chunks, 1, num_features, 1))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
        self.reset_parameters()

    def forward(self, input):
        chunks = torch.stack(torch.chunk(input, self.num_chunks, 0), dim=0)
        if self.training:
            var, mean = torch.var_mean(chunks, dim=(1, 3, 4), keepdim=True, unbiased=True)
            output = (chunks - mean) / (var + self.eps).sqrt()
            output = output * self.weight + self.bias

            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.num_batches_tracked += 1

        else:
            output = (chunks - self.running_mean) / (self.running_var + self.eps).sqrt() * self.weight + self.bias
        return output.view_as(input)


class LabelSmoothCrossEntropyLoss(torch.nn.Module):
    """See https://github.com/pytorch/pytorch/issues/7455.

    This is huanglianghua's variant
    """

    def __init__(self, smoothing=0.0, loss_modification=''):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = torch.nn.functional.log_softmax(input, dim=-1)
        weight = torch.ones_like(input) * self.smoothing / (input.shape[-1] - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss_per_sample = (-weight * log_prob).sum(dim=-1)
        return loss_per_sample.mean()


class IncorrectCrossEntropyLoss(torch.nn.Module):
    """CrossEntropyLoss, but only on incorrectly classified examples. Optionally: Includes label smoothing as above."""

    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        with torch.no_grad():
            correct_preds = input.argmax(dim=1) == target
        log_prob = torch.nn.functional.log_softmax(input, dim=-1)
        weight = torch.ones_like(input) * self.smoothing / (input.shape[-1] - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss_per_sample = (-weight * log_prob).sum(dim=-1)
        # return loss_per_sample[~correct_preds].mean()  # Take the mean only over incorrect samples
        return (loss_per_sample * (1 - correct_preds.float())).mean()


class MaxupLoss(torch.nn.Module):

    def __init__(self, ntrials=10):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.ntrials = 10

    def forward(self, outputs, labels):
        batch_size = outputs.shape[0] // self.ntrials
        stacked_loss = self.loss(outputs, labels).view(batch_size, self.ntrials, -1)
        loss = stacked_loss.max(dim=1)[0].mean()
        return loss


class GradRegularizer():
    """Modify given iterable of gradients outside of autograd."""

    def __init__(self, model, optimizer, loss_fn, norm=2, block_strength=0.1, acc_strength=0.0, eps=1e-2,
                 implementation='finite_diff', mixed_precision=False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.norm = norm
        self.block_strength = block_strength
        self.acc_strength = acc_strength
        self.eps = eps
        self.mixed_precision = mixed_precision

        if self.block_strength == 0 and self.acc_strength == 0:
            self.forward = self._pass
            self.create_graph = False
        elif implementation == 'autograd-pen':
            if self.acc_strength != 0 and self.block_strength == 0:
                raise ValueError('Requires non-zero block strength if computing pre_grads')
            self.forward = self._gradpen_function
            self.create_graph = True
        elif implementation == 'autograd':
            self.forward = self._double_autograd
            self.create_graph = True
        elif implementation == 'central-differences':
            self.forward = self._central_differences
            self.create_graph = False
        elif implementation == 'complex-step':
            self.forward = self._complex_differences
            self.create_graph = False
        elif implementation == 'forward-differences':
            self.forward = self._forward_differences
            self.create_graph = False
        elif implementation == 'forward-differences-legacy':
            self.forward = self._forward_differences_legacy
            self.create_graph = False
        else:
            raise ValueError(f'Invalid spec. given for regularizer implementation: {implementation}')

    def _pass(self, grads, inputs, labels, pre_grads):
        return grads

    def _gradpen_function(self, grads, inputs, labels, pre_grads):
        """Implement this via pytorch's autograd. Only works for block_strength > 0"""
        grad_penalty = 0
        if pre_grads is not None:
            fac = 1 / (2 * self.block_strength)  # this accounts for the fact that pre_grads does not require_grad
            for grad, pre_grad in zip(grads, pre_grads):
                grad_penalty += fac * (self.block_strength * grad + self.acc_strength * pre_grad).pow(self.norm).sum()
        else:
            for grad in grads:
                grad_penalty += self.block_strength * grad.pow(self.norm).sum()

        vhp = torch.autograd.grad(grad_penalty, self.model.parameters(), create_graph=False)
        correction_factor = self.optimizer.param_groups[0]["lr"] / 4
        torch._foreach_add_(grads, vhp, alpha=correction_factor)
        return grads

    def _double_autograd(self, grads, inputs, labels, pre_grads):
        """Implement this via pytorch's autograd. Are the pre_grad mods autograd safe?"""
        vhp = torch.autograd.grad(grads, self.model.parameters(), grad_outputs=grads, create_graph=False,
                                  retain_graph=True if pre_grads is not None else False)

        correction_factor = self.optimizer.param_groups[0]["lr"] / 4
        torch._foreach_add_(grads, vhp, alpha=correction_factor * self.block_strength)
        if pre_grads is not None:  # double autograd twice for pre_grads is a safe sanity check but non-optimal. Use gradpen in practice
            vhp = torch.autograd.grad(grads, self.model.parameters(), grad_outputs=pre_grads, create_graph=False,
                                      retain_graph=True if pre_grads is not None else False)

            correction_factor = self.optimizer.param_groups[0]["lr"] / 4
            torch._foreach_add_(grads, vhp, alpha=correction_factor * self.acc_strength)
        return grads

    def _forward_differences(self, grads, inputs, labels, pre_grads):
        # evaluate at theta + eps * grad
        # use the darts rule for eps
        correction_factor = self.optimizer.param_groups[0]["lr"] / 4
        original_parameters = [p.detach().clone() for p in self.model.parameters()]

        grad_vec = torch._foreach_mul(grads, self.block_strength)  # This assigns the vector in the vhp
        # This is one mem assignment more than the legacy fd, but should be ok on most machines. Use legacy_fd if necessary.

        if pre_grads is not None:
            torch._foreach_add_(grad_vec, pre_grads, alpha=self.acc_strength)

        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in grad_vec]).sum().sqrt()  # Adapts eps to different block strengths
        # eps around 1e-2 to 1e-4 appear most stable, but this can be model-dependent. See derivative check notebook.

        torch._foreach_add_(list(self.model.parameters()), grad_vec, alpha=eps_n)
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            outputs = self.model(inputs)
            block_loss = self.loss_fn(outputs, labels)
        offset_grads = torch.autograd.grad(block_loss, self.model.parameters())

        torch._foreach_sub_(offset_grads, grads)
        vhp = offset_grads  # this is due to the inplace modification and saves one copy
        torch._foreach_div_(vhp, eps_n)

        # repair model parameters
        for param, original_param in zip(self.model.parameters(), original_parameters):
            param.data.copy_(original_param)

        torch._foreach_add_(grads, vhp, alpha=correction_factor)
        return grads

    def _forward_differences_legacy(self, grads, inputs, labels, pre_grads):
        """Legacy FD implementation. Do not use if acc_strength>0, the parameter will be disregarded."""
        # evaluate at theta + eps * grad
        # use the darts rule for eps
        correction_factor = self.optimizer.param_groups[0]["lr"] / 4 * self.block_strength
        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in grads]).sum().sqrt()

        torch._foreach_add_(list(self.model.parameters()), grads, alpha=eps_n)
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            outputs = self.model(inputs)
            block_loss = self.loss_fn(outputs, labels)
        offset_grads = torch.autograd.grad(block_loss, self.model.parameters())

        torch._foreach_sub_(offset_grads, grads)
        vhp = offset_grads  # this is due to the inplace modification and saves one copy
        torch._foreach_div_(vhp, eps_n)

        # repair model parameters inplace by subtracting and hoping for the best with finite precision:
        torch._foreach_sub_(list(self.model.parameters()), grads, alpha=eps_n)

        torch._foreach_add_(grads, vhp, alpha=correction_factor)
        return grads

    def _central_differences(self, grads, inputs, labels, pre_grads):
        # evaluate at theta + 0.5 * eps * grad and theta - 0.5 * eps * grad
        # use the darts rule for eps
        correction_factor = self.optimizer.param_groups[0]["lr"] / 4
        original_parameters = [p.detach().clone() for p in self.model.parameters()]

        grad_vec = torch._foreach_mul(grads, self.block_strength)  # This assigns the vector in the vhp
        if pre_grads is not None:
            torch._foreach_add_(grad_vec, pre_grads, alpha=self.acc_strength)

        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in grad_vec]).sum().sqrt()

        # Evaluate x+
        torch._foreach_add_(list(self.model.parameters()), grad_vec, alpha=0.5 * eps_n)
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            outputs = self.model(inputs)
            block_loss = self.loss_fn(outputs, labels)
        xplus_grads = torch.autograd.grad(block_loss, self.model.parameters())

        # Evaluate x-
        torch._foreach_sub_(list(self.model.parameters()), grad_vec, alpha=eps_n)
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            outputs = self.model(inputs)
            block_loss = self.loss_fn(outputs, labels)
        xminus_grads = torch.autograd.grad(block_loss, self.model.parameters())

        vhp = torch._foreach_sub(xplus_grads, xminus_grads)
        torch._foreach_div_(vhp, eps_n)

        # repair model parameters
        for param, original_param in zip(self.model.parameters(), original_parameters):
            param.data.copy_(original_param)

        torch._foreach_add_(grads, vhp, alpha=correction_factor)
        return grads

    def _complex_differences(self, grads, inputs, labels, pre_grads):
        r"""Compute derivatives via the complex step trick.

        see e.g. https://timvieira.github.io/blog/post/2014/08/07/complex-step-derivative/
        however here the computations are changed to allow multivariate gradients
        for f:\R^n \to \R with parameter x in \R^n and gradient g \in \R^n we have
        Imag[Grad f(x - 1i*g*eps)] / eps = Grad(Grad(f(x))

        As such the 2nd-order backprop is replaced by a 1st-order complex backprop.

        !!! This option is currently not working in pytorch 1.9, but future versions might enable enough functionality
        to allow for it !!!
        """
        # Cast into complex tensors
        self.model.to(dtype=torch.complex64)

        grad_vec = torch._foreach_mul(grads, self.block_strength)  # This assigns the vector in the vhp
        if pre_grads is not None:
            torch._foreach_add_(grad_vec, pre_grads, alpha=self.acc_strength)

        # evaluate at theta + eps * grad_vec
        # no need for the darts rule for eps here?
        eps_n = torch.tensor(1j * self.eps, dtype=torch.complex64)
        torch._foreach_sub_(list(self.model.parameters()), [g.to(dtype=torch.complex64) for g in grad_vec],
                            alpha=eps_n)

        # complex forward pass
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):  # this should likely always be a no-op context
            outputs = self.model(inputs.to(dtype=torch.complex64))
            # block_loss = outputs.pow(2).sum()  # This just a sanity check until more complex loss ops are supported in pytorch
            block_loss = self.loss_fn(outputs, labels)
        complex_grads = torch.autograd.grad(block_loss, self.model.parameters())
        vhp = [(g.imag / eps_n).to(dtype=inputs.dtype) for g in complex_grads]

        # repair model parameters inplace by removing imaginary parts
        # for param in self.model.parameters():
        #    param.data = param.real
        self.model.to(inputs.dtype)

        # Add to grad
        correction_factor = self.optimizer.param_groups[0]["lr"] / 4
        torch._foreach_add_(grads, vhp, alpha=correction_factor)
        return grads

    def __call__(self, *args):
        # Cannot assign directly to __call__ per object so we assign to forward
        return self.forward(*args)
