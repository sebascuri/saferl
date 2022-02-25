"""Verification algorithm.

Maximize J^pi, for hallucination
"""
import torch
from hucrl.model.hallucinated_model import HallucinatedModel
from rllib.algorithms.mpc import CEMShooting


class MaximallySafeMPC(CEMShooting):
    def __init__(self, *args, **kwargs):
        super(MaximallySafeMPC, self).__init__(*args, **kwargs)
        if isinstance(self.dynamical_model, HallucinatedModel):
            self.h_dim_action = self.dynamical_model.dim_state
        else:
            self.h_dim_action = (0,)

    def _get_action_by_index(self, action_sequence, index):
        idx = index.unsqueeze(0).unsqueeze(-1)  # Expand dims to action_sequence.
        idx = idx.repeat_interleave(self.horizon, 0).repeat_interleave(
            self.dim_action, -1
        )
        return torch.gather(action_sequence, -2, idx)

    def get_best_action(self, action_sequence, returns):
        """Get best action by: see below.

        - Maximize true actions w.r.t. cost.
        - Minimize hallucinated actions w.r.t. cost.
        """
        cost = returns[..., 1:].sum(-1)
        idx = torch.topk(cost, k=self.num_elites, largest=True, dim=-1)[1]
        max_actions = self._get_action_by_index(action_sequence, idx)
        idx = torch.topk(cost, k=self.num_elites, largest=False, dim=-1)[1]
        min_actions = self._get_action_by_index(action_sequence, idx)

        actions = min_actions[..., : -self.h_dim_action[0]]
        hall_actions = max_actions[..., -self.h_dim_action[0] :]

        return torch.cat((actions, hall_actions), dim=-1)
