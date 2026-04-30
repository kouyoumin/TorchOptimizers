import torch
import torch.optim as optim
import math


class Paratrooper(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01,
                 lookahead_k=6,
                 lookahead_alpha=0.5,
                 use_lars=True,
                 lars_epsilon=1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 1 <= lookahead_k:
            raise ValueError(f"Invalid lookahead_k: {lookahead_k}")
        if not 0.0 <= lookahead_alpha <= 1.0:
            raise ValueError(f"Invalid lookahead_alpha: {lookahead_alpha}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        lookahead_k=lookahead_k, lookahead_alpha=lookahead_alpha,
                        use_lars=use_lars, lars_epsilon=lars_epsilon)
        super().__init__(params, defaults)

        for group in self.param_groups:
            group['lookahead_step_counter'] = 0
            for p in group['params']:
                state = self.state[p]
                state['slow_buffer'] = p.data.clone().detach()
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            use_lars = group['use_lars']
            lars_eps = group['lars_epsilon']
            p_inf = 2.0 / (1.0 - beta2) - 1.0

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                state['step'] += 1
                t = state['step']

                grad = p.grad.data.float()
                m = state['exp_avg']
                v = state['exp_avg_sq']

                # Moment update (pure loss gradient, no WD — decoupled style)
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                beta2_t = beta2 ** t
                beta1_t = beta1 ** t
                p_t = p_inf - 2.0 * t * beta2_t / (1.0 - beta2_t)

                # Default clip for the degenerated (warm-up) regime
                R_t = 10.0

                if p_t >= 5.0:
                    # RAdam rectified regime
                    b_t = math.sqrt(1.0 - beta2_t) / (1.0 - beta1_t)
                    r_t = math.sqrt(
                        (p_t - 4.0) * (p_t - 2.0) * p_inf /
                        ((p_inf - 4.0) * (p_inf - 2.0) * p_t)
                    )
                    # RAdamW update: bias-corrected Adam + decoupled weight decay
                    D_theta = (r_t * b_t) * m / (v.sqrt().add(eps))
                    if wd != 0.0:
                        D_theta = D_theta.add(p.data.float(), alpha=wd)
                    # Paratrooper dynamic clip: starts large, decays toward 1
                    R_t = 1.0 / (r_t * b_t)
                else:
                    # Degenerated regime: bias-corrected momentum only (no v)
                    b_t = 1.0 / (1.0 - beta1_t)
                    D_theta = b_t * m
                    if wd != 0.0:
                        D_theta = D_theta.add(p.data.float(), alpha=wd)
                    # R_t stays at 10.0

                # LARS trust ratio applied to Adam update D_theta (not raw grad)
                if use_lars:
                    w_norm = p.data.float().norm(2).item()
                    d_norm = D_theta.norm(2).item()
                    if w_norm == 0.0 or d_norm == 0.0:
                        T_t = 1.0
                    else:
                        T_t = max(1.0, min(R_t, w_norm / (d_norm + lars_eps)))
                else:
                    T_t = 1.0

                p.data.add_(D_theta.to(p.data.dtype), alpha=-lr * T_t)

        # Lookahead: interpolate all params (not just those with gradients)
        for group in self.param_groups:
            group['lookahead_step_counter'] += 1
            if group['lookahead_step_counter'] >= group['lookahead_k']:
                group['lookahead_step_counter'] = 0
                alpha = group['lookahead_alpha']
                for p in group['params']:
                    state = self.state[p]
                    slow_p = state['slow_buffer']
                    slow_p.add_(p.data - slow_p, alpha=alpha)
                    p.data.copy_(slow_p)

        return loss
