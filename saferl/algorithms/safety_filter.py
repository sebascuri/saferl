"""Safety Filter Algorithm."""
import torch
from hucrl.model.hallucinated_model import HallucinatedModel
from rllib.algorithms.mpc import CEMShooting
from rllib.model.abstract_model import AbstractModel
from rllib.model.closed_loop_model import ClosedLoopModel


class SafetyFilterMPC(CEMShooting):
    def __init__(self, dynamical_model, reward_model, policy, *args, **kwargs):
        if isinstance(dynamical_model, HallucinatedModel):
            self.h_dim_action = self.dynamical_model.dim_state
        else:
            self.h_dim_action = (0,)
        dynamical_model = ClosedLoopModel(dynamical_model, policy)
        reward_model = SafetyFilterReward(reward_model, policy)
        super(SafetyFilterMPC, self).__init__(
            dynamical_model=dynamical_model, reward_model=reward_model, *args, **kwargs
        )
        self.policy = policy

    def get_best_action(self, action_sequence, returns):
        """Get best action by: see below.

        - Maximize true actions w.r.t. cost.
        - Minimize hallucinated actions w.r.t. cost.
        """

        cost = returns[..., 1:]
        returns = returns[..., 1:]
        valid_indexes = torch.where(torch.all(cost <= 0, dim=-1))[0]
        invalid_indexes = torch.where(~torch.all(cost <= 0, dim=-1))[0]

        returns[valid_indexes] = cost[valid_indexes].sum(-1)
        returns[invalid_indexes] = -float("inf")

        idx = torch.topk(
            returns, k=min(self.num_elites, len(valid_indexes)), largest=True, dim=-1
        )[1]

        # TODO: optimize hallucinated action to maximize the cost.

        idx = idx.unsqueeze(0).unsqueeze(-1)  # Expand dims to action_sequence.
        idx = idx.repeat_interleave(self.horizon, 0).repeat_interleave(
            self.dim_action, -1
        )
        return torch.gather(action_sequence, -2, idx)


class SafetyFilterReward(AbstractModel):
    def __init__(self, base_model, policy, *args, **kwargs):
        super().__init__(
            dim_state=base_model.dim_state,
            dim_action=base_model.dim_action,
            dim_reward=(1,),
            num_states=base_model.num_states,
            num_actions=base_model.num_actions,
            model_kind=base_model.model_kind,
            *args,
            **kwargs
        )
        self.policy = policy
        assert self.model_kind == "rewards", "only reward model is implemented."

    def forward(self, state, action, next_state=None):
        self.policy(state)
        reward = torch.norm(action - self.policy(state)[0], dim=-1).unsqueeze(-1)
        return reward, torch.zeros_like(reward)
