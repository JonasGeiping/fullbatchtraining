"""This is a modification of https://github.com/hjmshi/PyTorch-LBFGS [a mod of pytorch's L-BFGS]

Modifications:
* Allow parameter groups [+ weight decay]
* Move all options into optimizer parameters
* Move all optimizer state into optimizer state
* Remove manual .cuda(), .to(dtype) modifications
* Remove debug prints
"""

import torch
import numpy as np

from copy import deepcopy


def is_legal(v):
    """
    Checks that tensor is not NaN or Inf.

    Inputs:
        v (tensor): tensor to be checked

    """
    legal = not torch.isnan(v).any() and not torch.isinf(v)

    return legal


def polyinterp(points, x_min_bound=None, x_max_bound=None):
    """
    Gives the minimizer and minimum of the interpolating polynomial over given points
    based on function and derivative information. Defaults to bisection if no critical
    points are valid.

    Based on polyinterp.m Matlab function in minFunc by Mark Schmidt with some slight
    modifications.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 12/6/18.

    Inputs:
        points (nparray): two-dimensional array with each point of form [x f g]
        x_min_bound (float): minimum value that brackets minimum (default: minimum of points)
        x_max_bound (float): maximum value that brackets minimum (default: maximum of points)

    Outputs:
        x_sol (float): minimizer of interpolating polynomial
        F_min (float): minimum of interpolating polynomial

    Note:
      . Set f or g to np.nan if they are unknown

    """
    no_points = points.shape[0]
    order = np.sum(1 - np.isnan(points[:, 1:3]).astype('int')) - 1

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])

    # compute bounds of interpolation area
    if x_min_bound is None:
        x_min_bound = x_min
    if x_max_bound is None:
        x_max_bound = x_max

    # explicit formula for quadratic interpolation
    if no_points == 2 and order == 2:
        # Solution to quadratic interpolation is given by:
        # a = -(f1 - f2 - g1(x1 - x2))/(x1 - x2)^2
        # x_min = x1 - g1/(2a)
        # if x1 = 0, then is given by:
        # x_min = - (g1*x2^2)/(2(f2 - f1 - g1*x2))

        if points[0, 0] == 0:
            x_sol = -points[0, 2] * points[1, 0] ** 2 / \
                (2 * (points[1, 1] - points[0, 1] - points[0, 2] * points[1, 0]))
        else:
            a = -(points[0, 1] - points[1, 1] - points[0, 2] *
                  (points[0, 0] - points[1, 0])) / (points[0, 0] - points[1, 0]) ** 2
            x_sol = points[0, 0] - points[0, 2] / (2 * a)

        x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)

    # explicit formula for cubic interpolation
    elif no_points == 2 and order == 3:
        # Solution to cubic interpolation is given by:
        # d1 = g1 + g2 - 3((f1 - f2)/(x1 - x2))
        # d2 = sqrt(d1^2 - g1*g2)
        # x_min = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2))
        d1 = points[0, 2] + points[1, 2] - 3 * ((points[0, 1] - points[1, 1]) / (points[0, 0] - points[1, 0]))
        d2 = np.sqrt(d1 ** 2 - points[0, 2] * points[1, 2])
        if np.isreal(d2):
            x_sol = points[1, 0] - (points[1, 0] - points[0, 0]) * \
                ((points[1, 2] + d2 - d1) / (points[1, 2] - points[0, 2] + 2 * d2))
            x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)
        else:
            x_sol = (x_max_bound + x_min_bound) / 2

    # solve linear system
    else:
        # define linear constraints
        A = np.zeros((0, order + 1))
        b = np.zeros((0, 1))

        # add linear constraints on function values
        for i in range(no_points):
            if not np.isnan(points[i, 1]):
                constraint = np.zeros((1, order + 1))
                for j in range(order, -1, -1):
                    constraint[0, order - j] = points[i, 0] ** j
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 1])

        # add linear constraints on gradient values
        for i in range(no_points):
            if not np.isnan(points[i, 2]):
                constraint = np.zeros((1, order + 1))
                for j in range(order):
                    constraint[0, j] = (order - j) * points[i, 0] ** (order - j - 1)
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 2])

        # check if system is solvable
        if A.shape[0] != A.shape[1] or np.linalg.matrix_rank(A) != A.shape[0]:
            x_sol = (x_min_bound + x_max_bound) / 2
            f_min = np.Inf
        else:
            # solve linear system for interpolating polynomial
            coeff = np.linalg.solve(A, b)

            # compute critical points
            dcoeff = np.zeros(order)
            for i in range(len(coeff) - 1):
                dcoeff[i] = coeff[i] * (order - i)

            crit_pts = np.array([x_min_bound, x_max_bound])
            crit_pts = np.append(crit_pts, points[:, 0])

            if not np.isinf(dcoeff).any():
                roots = np.roots(dcoeff)
                crit_pts = np.append(crit_pts, roots)

            # test critical points
            f_min = np.Inf
            x_sol = (x_min_bound + x_max_bound) / 2  # defaults to bisection
            for crit_pt in crit_pts:
                if np.isreal(crit_pt) and crit_pt >= x_min_bound and crit_pt <= x_max_bound:
                    F_cp = np.polyval(coeff, crit_pt)
                    if np.isreal(F_cp) and F_cp < f_min:
                        x_sol = np.real(crit_pt)
                        f_min = np.real(F_cp)

    return x_sol


class LBFGS(torch.optim.Optimizer):
    """
    Implements the L-BFGS algorithm. Compatible with multi-batch and full-overlap
    L-BFGS implementations and (stochastic) Powell damping. Partly based on the
    original L-BFGS implementation in PyTorch, Mark Schmidt's minFunc MATLAB code,
    and Michael Overton's weak Wolfe line search MATLAB code.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 10/20/20.

    Warnings:
      . Only the weight_decay and lr options can be group-specific. All other options are sourced from group 0.

    Inputs:
        lr (float): steplength or learning rate (default: 1)
        history_size (int): update history size (default: 10)
        line_search (str): designates line search to use (default: 'Wolfe')
            Options:
                'None': uses steplength designated in algorithm
                'Armijo': uses Armijo backtracking line search
                'Wolfe': uses Armijo-Wolfe bracketing line search

        General Options:
            'eps' (float): constant for curvature pair rejection or damping (default: 1e-2)
            'damping' (bool): flag for using Powell damping (default: False)

        Options for Armijo backtracking line search:
            'eta' (tensor): factor for decreasing steplength > 0 (default: 2)
            'c1' (tensor): sufficient decrease constant in (0, 1) (default: 1e-4)
            'max_ls' (int): maximum number of line search steps permitted (default: 10)

        Options for Wolfe line search:
            'eta' (float): factor for extrapolation (default: 2)
            'c1' (float): sufficient decrease constant in (0, 1) (default: 1e-4)
            'c2' (float): curvature condition constant in (0, 1) (default: 0.9)
            'max_ls' (int): maximum number of line search steps permitted (default: 10)


    References:
    [1] Berahas, Albert S., Jorge Nocedal, and Martin Takác. "A Multi-Batch L-BFGS
        Method for Machine Learning." Advances in Neural Information Processing
        Systems. 2016.
    [2] Bollapragada, Raghu, et al. "A Progressive Batching L-BFGS Method for Machine
        Learning." International Conference on Machine Learning. 2018.
    [3] Lewis, Adrian S., and Michael L. Overton. "Nonsmooth Optimization via Quasi-Newton
        Methods." Mathematical Programming 141.1-2 (2013): 135-163.
    [4] Liu, Dong C., and Jorge Nocedal. "On the Limited Memory BFGS Method for
        Large Scale Optimization." Mathematical Programming 45.1-3 (1989): 503-528.
    [5] Nocedal, Jorge. "Updating Quasi-Newton Matrices With Limited Storage."
        Mathematics of Computation 35.151 (1980): 773-782.
    [6] Nocedal, Jorge, and Stephen J. Wright. "Numerical Optimization." Springer New York,
        2006.
    [7] Schmidt, Mark. "minFunc: Unconstrained Differentiable Multivariate Optimization
        in Matlab." Software available at http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
        (2005).
    [8] Schraudolph, Nicol N., Jin Yu, and Simon Günter. "A Stochastic Quasi-Newton
        Method for Online Convex Optimization." Artificial Intelligence and Statistics.
        2007.
    [9] Wang, Xiao, et al. "Stochastic Quasi-Newton Methods for Nonconvex Stochastic
        Optimization." SIAM Journal on Optimization 27.2 (2017): 927-956.

    """

    def __init__(self, params, lr=1., history_size=10, line_search='Wolfe', eps=1e-2, damping=True, eta=2, c1=1e-4,
                 c2=0.9, max_linesearches=10, weight_decay=0.0):

        # ensure inputs are valid yourself
        defaults = dict(lr=lr, history_size=history_size, line_search=line_search, eps=eps, damping=damping, eta=eta,
                        c1=c1, c2=c2, max_linesearches=max_linesearches, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.numel = sum([p.numel() for group in self.param_groups for p in group['params']])

        state = self.state['global_state']
        state.setdefault('n_iter', 0)
        state.setdefault('curv_skips', 0)
        state.setdefault('fail_skips', 0)
        state.setdefault('H_diag', 1)
        state.setdefault('fail', True)

        state['old_dirs'] = []
        state['old_stps'] = []


    def _gather_flat_grad(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    view = p.data.new(p.data.numel()).zero_()
                else:
                    view = p.grad.data.view(-1)
                view += group['weight_decay'] * p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_update(self, step_size, update):
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                # view as to avoid deprecated pointwise semantics
                p.data.add_(update[offset:offset + numel].view_as(p.data), alpha=step_size * group['lr'])
                offset += numel
        assert offset == self.numel

    def _copy_params(self):
        current_params = []
        for group in self.param_groups:
            for p in group['params']:
                current_params.append(deepcopy(p.data))
        return current_params

    def _load_params(self, current_params):
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                param.data = current_params[i]
            i += 1

    def two_loop_recursion(self, vec):
        """
        Performs two-loop recursion on given vector to obtain Hv.

        Inputs:
            vec (tensor): 1-D tensor to apply two-loop recursion to

        Output:
            r (tensor): matrix-vector product Hv

        """

        group = self.param_groups[0]
        history_size = group['history_size']

        state = self.state['global_state']
        old_dirs = state.get('old_dirs')    # change in gradients
        old_stps = state.get('old_stps')    # change in iterates
        H_diag = state.get('H_diag')

        # compute the product of the inverse Hessian approximation and the gradient
        num_old = len(old_dirs)

        if 'rho' not in state:
            state['rho'] = [None] * history_size
            state['alpha'] = [None] * history_size
        rho = state['rho']
        alpha = state['alpha']

        for i in range(num_old):
            rho[i] = 1. / old_stps[i].dot(old_dirs[i])

        q = vec
        for i in range(num_old - 1, -1, -1):
            alpha[i] = old_dirs[i].dot(q) * rho[i]
            q.add_(-alpha[i], old_stps[i])

        # multiply by initial Hessian
        # r/d is the final direction
        r = torch.mul(q, H_diag)
        for i in range(num_old):
            beta = old_stps[i].dot(r) * rho[i]
            r.add_(alpha[i] - beta, old_dirs[i])

        return r

    def curvature_update(self, flat_grad, eps=1e-2, damping=False):
        """
        Performs curvature update.

        Inputs:
            flat_grad (tensor): 1-D tensor of flattened gradient for computing
                gradient difference with previously stored gradient
            eps (float): constant for curvature pair rejection or damping (default: 1e-2)
            damping (bool): flag for using Powell damping (default: False)
        """

        # load parameters
        if(eps <= 0):
            raise(ValueError('Invalid eps; must be positive.'))

        group = self.param_groups[0]
        history_size = group['history_size']

        # variables cached in state (for tracing)
        state = self.state['global_state']
        fail = state.get('fail')

        # check if line search failed
        if not fail:

            d = state.get('d')
            t = state.get('t')
            old_dirs = state.get('old_dirs')
            old_stps = state.get('old_stps')
            H_diag = state.get('H_diag')
            prev_flat_grad = state.get('prev_flat_grad')
            Bs = state.get('Bs')

            # compute y's
            y = flat_grad.sub(prev_flat_grad)
            s = d.mul(t)
            sBs = s.dot(Bs)
            ys = y.dot(s)  # y*s

            # update L-BFGS matrix
            if ys > eps * sBs or damping:

                # perform Powell damping
                if damping and ys < eps * sBs:
                    theta = ((1 - eps) * sBs) / (sBs - ys)
                    y = theta * y + (1 - theta) * Bs

                # updating memory
                if len(old_dirs) == history_size:
                    # shift history by one (limited-memory)
                    old_dirs.pop(0)
                    old_stps.pop(0)

                # store new direction/step
                old_dirs.append(s)
                old_stps.append(y)

                # update scale of initial Hessian approximation
                H_diag = ys / y.dot(y)  # (y*y)

                state['old_dirs'] = old_dirs
                state['old_stps'] = old_stps
                state['H_diag'] = H_diag

            else:
                # save skip
                state['curv_skips'] += 1

        else:
            # save skip
            state['fail_skips'] += 1

        return

    def _step(self, p_k, g_Ok, closure):
        """
        Performs a single optimization step based on a precomputed direction p_k based on a gradient g_Ok.
        """

        # load parameter options
        group = self.param_groups[0]
        lr = group['lr']
        line_search = group['line_search']

        # variables cached in state (for tracing)
        state = self.state['global_state']
        d = state.get('d')
        t = state.get('t')
        prev_flat_grad = state.get('prev_flat_grad')
        Bs = state.get('Bs')

        # keep track of nb of iterations
        state['n_iter'] += 1

        # set search direction
        g_Sk = g_Ok.clone()
        d = p_k
        gtd = g_Sk.dot(d)

        # modify previous gradient
        if prev_flat_grad is None:
            prev_flat_grad = g_Ok.clone()
        else:
            prev_flat_grad.copy_(g_Ok)

        # set initial step size
        t = 1.0

        # closure evaluation counter
        closure_eval = 0



        # perform Armijo backtracking line search
        if group['line_search'] == 'Armijo':
            # initialize values
            F_prev = torch.tensor(np.nan, dtype=p_k.dtype, device=p_k.device)

            ls_step = 0
            t_prev = 0  # old steplength
            fail = False  # failure flag

            # store values if not in-place
            if not inplace:
                current_params = self._copy_params()

            # update and evaluate at new point
            self._add_update(t, d)
            F_new = closure()

            # check Armijo condition
            while F_new > F_k + group['c1'] * t * gtd or not is_legal(F_new):

                # check if maximum number of iterations reached
                if ls_step >= group['max_linesearches']:
                    self._add_update(-t, d)

                    t = 0
                    F_new = closure()
                    fail = True
                    break

                else:
                    # store current steplength
                    t_new = t

                    # if first step or not interpolating, then multiply by factor
                    if ls_step == 0 or not is_legal(F_new):
                        t = t / group['eta']

                    # if second step, use function value at new point along with
                    # gradient and function at current iterate
                    elif ls_step == 1 or not is_legal(F_prev):
                        t = polyinterp(np.array([[0, F_k.item(), gtd.item()], [t_new, F_new.item(), np.nan]]))

                    # otherwise, use function values at new point, previous point,
                    # and gradient and function at current iterate
                    else:
                        t = polyinterp(np.array([[0, F_k.item(), gtd.item()], [t_new, F_new.item(), np.nan],
                                                 [t_prev, F_prev.item(), np.nan]]))

                    # if values are too extreme, adjust t
                    if t < 1e-3 * t_new:
                        t = 1e-3 * t_new
                    elif t > 0.6 * t_new:
                        t = 0.6 * t_new

                    # store old point
                    F_prev = F_new
                    t_prev = t_new

                    # update iterate and reevaluate
                    self._add_update(t - t_new, d)

                    F_new = closure()
                    ls_step += 1  # iterate

            # store Bs
            if Bs is None:
                Bs = (g_Sk.mul(-t)).clone()
            else:
                Bs.copy_(g_Sk.mul(-t))

            state['d'] = d
            state['prev_flat_grad'] = prev_flat_grad
            state['t'] = t
            state['Bs'] = Bs
            state['fail'] = fail

            return F_new

        # perform weak Wolfe line search
        elif line_search == 'Wolfe':
            # initialize counters
            ls_step = 0
            grad_eval = 0  # tracks gradient evaluations
            t_prev = 0  # old steplength

            # initialize bracketing variables and flag
            alpha = 0
            beta = float('Inf')
            fail = False

            # initialize values for line search
            F_a = F_k = closure()
            g_a = gtd

            F_b = torch.tensor(np.nan, dtype=p_k.dtype, device=p_k.device)
            g_b = torch.tensor(np.nan, dtype=p_k.dtype, device=p_k.device)

            # update and evaluate at new point
            self._add_update(t, d)
            F_new = closure()

            # main loop
            while True:
                # check if maximum number of line search steps have been reached
                if ls_step >= group['max_linesearches']:
                    self._add_update(-t, d)

                    t = 0
                    F_new = closure()
                    g_new = self._gather_flat_grad()

                    fail = True
                    break

                # check Armijo condition
                if F_new > F_k + group['c1'] * t * gtd:

                    # set upper bound
                    beta = t
                    t_prev = t

                    # update interpolation quantities
                    F_b = F_new
                    g_b = torch.tensor(np.nan, dtype=p_k.dtype, device=p_k.device)
                else:

                    # compute gradient
                    g_new = self._gather_flat_grad()
                    gtd_new = g_new.dot(d)

                    # check curvature condition
                    if gtd_new < group['c2'] * gtd:

                        # set lower bound
                        alpha = t
                        t_prev = t
                        F_a = F_new
                        g_a = gtd_new

                    else:
                        break

                # compute new steplength

                # if first step or not interpolating, then bisect or multiply by factor
                if not is_legal(F_b):
                    if beta == float('Inf'):
                        t = group['eta'] * t
                    else:
                        t = (alpha + beta) / 2.0

                # otherwise interpolate between a and b
                else:
                    t = polyinterp(np.array([[alpha, F_a.item(), g_a.item()], [beta, F_b.item(), g_b.item()]]))

                    # if values are too extreme, adjust t
                    if beta == float('Inf'):
                        if t > 2 * eta * t_prev:
                            t = 2 * eta * t_prev
                        elif t < group['eta'] * t_prev:
                            t = group['eta'] * t_prev
                    else:
                        if t < alpha + 0.2 * (beta - alpha):
                            t = alpha + 0.2 * (beta - alpha)
                        elif t > (beta - alpha) / 2.0:
                            t = (beta - alpha) / 2.0

                    # if we obtain nonsensical value from interpolation
                    if t <= 0:
                        t = (beta - alpha) / 2.0

                # update parameters
                self._add_update(t - t_prev, d)

                # evaluate closure
                F_new = closure()
                ls_step += 1

            # store Bs
            if Bs is None:
                Bs = (g_Sk.mul(-t)).clone()
            else:
                Bs.copy_(g_Sk.mul(-t))

            state['d'] = d
            state['prev_flat_grad'] = prev_flat_grad
            state['t'] = t
            state['Bs'] = Bs
            state['fail'] = fail

            return F_new

        else:

            # perform update
            self._add_update(t, d)

            # store Bs
            if Bs is None:
                Bs = (g_Sk.mul(-t)).clone()
            else:
                Bs.copy_(g_Sk.mul(-t))

            state['d'] = d
            state['prev_flat_grad'] = prev_flat_grad
            state['t'] = t
            state['Bs'] = Bs
            state['fail'] = False

            return t

    def step(self, closure):
        """
        Performs a single optimization step.

        Inputs:
            callable closure

        Outputs:
            None
        """
        # gather gradient
        grad = self._gather_flat_grad()

        # update curvature if after 1st iteration
        state = self.state['global_state']
        if state['n_iter'] > 0:
            self.curvature_update(grad, self.param_groups[0]['eps'], self.param_groups[0]['damping'])

        # compute search direction
        p = self.two_loop_recursion(-grad)

        # take step
        return self._step(p, grad, closure=closure)
