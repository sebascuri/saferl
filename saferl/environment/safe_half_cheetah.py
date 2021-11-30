"""Utility files to run experiments."""
import torch
from rllib.reward.locomotion_reward import LocomotionReward
from rllib.util.neural_networks.utilities import to_torch


class HalfCheetahSafeReward(LocomotionReward):
    """Reward for Mujoco."""

    def __init__(
        self,
        dim_action,
        ctrl_cost_weight,
        forward_reward_weight=1.0,
        healthy_reward=0.0,
        max_velocity=5.0,
    ):
        self.dim_action = dim_action
        super().__init__(dim_action=dim_action, ctrl_cost_weight=ctrl_cost_weight)
        self.forward_reward_weight = forward_reward_weight
        self.healthy_reward = healthy_reward
        self.max_velocity = max_velocity

    def forward(self, state, action, next_state=None):
        """Get reward distribution for state, action, next_state."""
        state, action = to_torch(state), to_torch(action)
        reward_state = self.state_reward(state, next_state)
        reward_ctrl = self.action_reward(action)

        cost = state[..., 0] - self.max_velocity
        # cost = tolerance(state[..., 0], lower=0.05, upper=0.15, margin=0.0015) - 1.0
        reward = torch.cat(
            (reward_state + self.ctrl_cost_weight * reward_ctrl, cost), dim=-1
        )

        try:
            self._info.update(
                reward_state=reward_state.sum().item(),
                reward_ctrl=reward_ctrl.sum().item(),
            )
            reward = reward.type(torch.get_default_dtype()).unsqueeze(-1)
        except AttributeError:
            pass
        return reward, torch.zeros_like(reward).unsqueeze(-1)
