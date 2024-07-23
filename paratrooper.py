import torch
from torch.optim.optimizer import Optimizer

class Paratrooper(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.95, 0.999), eps=1e-8, weight_decay=0.0, mu=0.5, k=6, lambda_=0.001):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= mu <= 1.0:
            raise ValueError("Invalid mu value: {}".format(mu))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 1 <= k:
            raise ValueError("Invalid k value: {}".format(k))
        if not 0.0 <= lambda_:
            raise ValueError("Invalid lambda value: {}".format(lambda_))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, mu=mu, k=k, lambda_=lambda_)
        super(Paratrooper, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Paratrooper, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Paratrooper does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['theta'] = torch.randn_like(p.data)
                    state['phi'] = state['theta'].clone()

                m, v = state['m'], state['v']
                theta, phi = state['theta'], state['phi']

                beta1, beta2 = group['betas']
                lambda_ = group['lambda_']
                mu = group['mu']
                k = group['k']
                eps = group['eps']
                lr = group['lr']

                state['step'] += 1
                t = state['step']

                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first moment estimate
                m_hat = m / (1 - beta1 ** t)
                # Compute bias-corrected second moment estimate
                v_hat = v / (1 - beta2 ** t)

                # RAdam adjustment terms
                p_inf = 2 / (1 - beta2) - 1
                p_t = p_inf - 2 * t * beta2 ** t / (1 - beta2 ** t)
                Rt = 10.0  # Default value

                if p_t >= 5:
                    bt = ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)
                    rt = ((p_t - 4) * (p_t - 2) * p_inf) / ((p_inf - 4) * (p_inf - 2) * p_t)
                    Dtheta = rt * bt * (m_hat / (v_hat.sqrt() + eps)) + lambda_ * theta
                    Rt = 1 / (rt * bt)
                else:
                    use_radamw = (1 / t) * torch.sum(grad ** 2) <= 0.5 * ((1 - beta1) / (1 - beta2)) ** 2
                    if use_radamw:
                        bt = 1 / (1 - beta1 ** t)
                        Dtheta = bt * (m_hat / (v_hat.sqrt() + eps)) + lambda_ * theta
                    else:
                        # Use degenerated RAdamW (not implemented in this basic version)
                        Dtheta = grad
                # Weight decay
                if group['weight_decay'] != 0:
                    Dtheta.add_(p.data, alpha=group['weight_decay'])

                # Adaptive learning rate
                omega = torch.sqrt(torch.sum(Dtheta ** 2))
                omega_dtheta = torch.sqrt(torch.sum((Dtheta + p.grad.data) ** 2))

                if omega == 0 or omega_dtheta == 0:
                    Tt = 1
                else:
                    Tt = max(1, min(Rt, omega / omega_dtheta))

                # Update theta
                theta.add_(Dtheta, alpha=-lr * Tt)
                # Update parameters
                #p.data.add_(Dtheta, alpha=-lr * Tt)

                if t % k == 0:
                    phi.mul_(mu).add_(theta, alpha=1 - mu)
                    theta.copy_(phi)
                    #state['theta'] = phi.clone()
                
                # Update the parameter
                p.data.copy_(theta)

        return loss

# Usage example:
# model = ...  # Your model
# optimizer = Paratrooper(model.parameters(), lr=0.001)
# for input, target in dataset:
#     optimizer.zero_grad()
#     loss = loss_fn(model(input), target)
#     loss.backward()
#     optimizer.step()
