import torch
from rllib.algorithms.mpc import CEMShooting


class SafeMPC(CEMShooting):
    """Safe CEM algorithm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unsafe = False

    def get_best_action(self, action_sequence, returns):
        """Get best action by averaging the num_elites samples."""
        costs = returns[..., 1:]
        returns = returns[..., 0]

        unsafe_actions = torch.any(costs > 0, dim=-1)
        if torch.all(unsafe_actions):
            self.unsafe = True
        else:
            self.unsafe = False

        returns[unsafe_actions] = -1e6
        # Make the returns extremely negative on unsafe actions

        idx = torch.topk(returns, k=self.num_elites, largest=True, dim=-1)[1]
        idx = idx.unsqueeze(0).unsqueeze(-1)  # Expand dims to action_sequence.
        idx = idx.repeat_interleave(self.num_model_steps, 0).repeat_interleave(
            self.dim_action, -1
        )
        return torch.gather(action_sequence, -2, idx)
