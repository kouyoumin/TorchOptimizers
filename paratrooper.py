import torch
import torch.optim as optim
import math

class Paratrooper(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, # Default AdamW style weight decay
                 # Lookahead parameters
                 lookahead_k=6,
                 lookahead_alpha=0.5,
                 # Paratrooper LARS-like component parameters
                 use_lars=True,
                 paratrooper_start_clip_val=10.0,
                 paratrooper_end_clip_val=1.0,
                 paratrooper_total_steps=10000, # Total training steps for decay schedule
                 lars_epsilon=1e-8, # Epsilon for LARS trust ratio calculation
                 lars_weight_decay=0.0 # Specific weight decay for LARS part if different
                ):

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay: # General weight decay
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= lookahead_alpha <= 1.0:
            raise ValueError(f"Invalid Lookahead alpha value: {lookahead_alpha}")
        if not 1 <= lookahead_k:
            raise ValueError(f"Invalid Lookahead k value: {lookahead_k}")
        if not paratrooper_start_clip_val >= paratrooper_end_clip_val:
            raise ValueError("paratrooper_start_clip_val must be >= paratrooper_end_clip_val")
        if not paratrooper_total_steps > 0:
            raise ValueError("paratrooper_total_steps must be positive for Paratrooper decay schedule")
        if not 0.0 <= lars_weight_decay:
            raise ValueError(f"Invalid LARS weight_decay value: {lars_weight_decay}")


        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        lookahead_k=lookahead_k, lookahead_alpha=lookahead_alpha,
                        use_lars=use_lars,
                        paratrooper_start_clip_val=paratrooper_start_clip_val,
                        paratrooper_end_clip_val=paratrooper_end_clip_val,
                        paratrooper_total_steps=paratrooper_total_steps,
                        lars_epsilon=lars_epsilon,
                        lars_weight_decay=lars_weight_decay)

        super(Paratrooper, self).__init__(params, defaults)

        # Initialize RAdam as the fast optimizer
        # If LARS is handling weight decay, RAdam should not apply it.
        radam_weight_decay = 0.0 if use_lars else weight_decay
        self.radam_optimizer = optim.RAdam(
            self.param_groups, # RAdam will use the groups defined in Paratrooper
            lr=lr,             # RAdam's LR, can be modulated effectively by LARS scaling
            betas=betas,
            eps=eps,
            weight_decay=radam_weight_decay
        )

        # Lookahead state: store slow weights and RAdam's original parameters
        # This might seem redundant as radam_optimizer uses self.param_groups,
        # but Lookahead often conceptualizes operating on a copy.
        # Here, self.param_groups ARE the 'fast' weights.
        for group in self.param_groups:
            group['lookahead_step_counter'] = 0
            for p in group['params']:
                param_state = self.state[p]
                param_state['slow_buffer'] = torch.clone(p.data).detach()

        self.global_step = 0

    def _get_current_paratrooper_clip_val(self, group):
        start_val = group['paratrooper_start_clip_val']
        end_val = group['paratrooper_end_clip_val']
        # Ensure global_step doesn't exceed total_steps for calculation
        progress = min(self.global_step / group['paratrooper_total_steps'], 1.0)
        # Linear decay
        current_clip_val = start_val - (start_val - end_val) * progress
        return current_clip_val

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        # LARS-like gradient scaling (Paratrooper modification)
        # This happens BEFORE RAdam's step, by modifying p.grad in place
        current_max_clip_for_lars = self._get_current_paratrooper_clip_val(self.param_groups[0]) # Assuming one group for clip schedule

        for group in self.param_groups:
            if not group['use_lars']:
                continue # Skip LARS-like scaling for this group

            for p in group['params']:
                if p.grad is None:
                    continue

                grad_data = p.grad.data
                param_data = p.data

                # Effective weight decay for LARS component
                lars_wd = group['lars_weight_decay']

                # Create a temporary gradient for LARS processing (if WD is applied)
                # This avoids modifying p.grad.data if WD is zero.
                temp_grad = grad_data
                if lars_wd > 0:
                    temp_grad = grad_data.add(param_data, alpha=lars_wd)

                weight_norm = torch.norm(param_data)
                grad_norm = torch.norm(temp_grad)

                trust_ratio = 1.0 # Default if norms are zero or LARS conditions not met

                if weight_norm > 0 and grad_norm > 0:
                    lars_trust_raw = weight_norm / (grad_norm + group['lars_epsilon'])
                    # Apply Paratrooper clipping to the LARS trust ratio
                    trust_ratio = min(lars_trust_raw, current_max_clip_for_lars)

                # Scale the original gradient (p.grad.data) by this trust_ratio.
                # If LARS weight decay was applied, temp_grad holds that.
                # We want RAdam to see `trust_ratio * (original_grad + lars_wd * param)`
                if lars_wd > 0:
                    p.grad.data = temp_grad.mul_(trust_ratio)
                else: # if lars_wd is 0, temp_grad is grad_data
                    p.grad.data.mul_(trust_ratio)


        # RAdam step (uses the potentially modified p.grad)
        self.radam_optimizer.step()

        # Lookahead update
        for group in self.param_groups:
            group['lookahead_step_counter'] += 1
            if group['lookahead_step_counter'] >= group['lookahead_k']:
                group['lookahead_step_counter'] = 0
                alpha = group['lookahead_alpha']
                for p in group['params']:
                    if p.grad is None: # Should not happen if p was in radam_optimizer
                        continue
                    param_state = self.state[p]
                    slow_p = param_state['slow_buffer']
                    fast_p = p.data # p.data is now the result of RAdam's step (fast weights)

                    slow_p.add_(fast_p - slow_p, alpha=alpha) # Update slow weights: slow += alpha * (fast - slow)
                    fast_p.copy_(slow_p)                      # Copy slow weights back to fast weights

        return loss

    def zero_grad(self, set_to_none: bool = False):
        # Zero gradients for the main parameters
        super(Paratrooper, self).zero_grad(set_to_none)
        # RAdam internally refers to the same param_groups, so its grads are also zeroed.
        # If RAdam had its own copy of params, we'd call self.radam_optimizer.zero_grad()
        # but since it uses self.param_groups, this is sufficient.

    def state_dict(self):
        # Get base optimizer state (includes slow_buffers and lookahead_step_counters)
        base_state_dict = super(Paratrooper, self).state_dict()
        # Get RAdam's state
        radam_state_dict = self.radam_optimizer.state_dict()

        # Combine them
        return {
            "base_state": base_state_dict,
            "radam_optimizer_state": radam_state_dict,
            "global_step": self.global_step
        }

    def load_state_dict(self, state_dict):
        # Load RAdam's state first
        self.radam_optimizer.load_state_dict(state_dict["radam_optimizer_state"])
        # Load base optimizer state (this will restore slow_buffers etc. into self.state)
        super(Paratrooper, self).load_state_dict(state_dict["base_state"])
        self.global_step = state_dict["global_step"]

        # PyTorch optimizers (like RAdam) re-initialize their state for param_groups
        # when load_state_dict is called, if the param_groups structure changed.
        # We need to ensure self.param_groups are correctly configured before this.
        # After RAdam's state is loaded, self.param_groups used by Paratrooper
        # should align with what RAdam expects. Our structure where RAdam uses
        # self.param_groups directly should simplify this.

        # Important: Ensure that the `slow_buffer` is correctly associated.
        # The base `load_state_dict` should handle restoring `self.state[p]['slow_buffer']`.
        # And `self.radam_optimizer.param_groups` are the same objects as `self.param_groups`.
