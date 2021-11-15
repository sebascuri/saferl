"""Utility files to run experiments."""
import yaml
from rllib.reward.locomotion_reward import LocomotionReward
from rllib.model.abstract_model import AbstractModel
import torch
from rllib.reward.utilities import tolerance



def parse_config_file(file_dir):
    """Parse configuration file."""
    with open(file_dir, "r") as file:
        args = yaml.safe_load(file)
    return args

class MujocoReward(LocomotionReward):
    """Reward for Inverted Pendulum."""

    def __init__(self,
        dim_action,
        ctrl_cost_weight,
        forward_reward_weight=1.0,
        healthy_reward=0.0,
    ):
        self.dim_action = dim_action
        super().__init__(dim_action=dim_action, ctrl_cost_weight=ctrl_cost_weight)
        self.forward_reward_weight = forward_reward_weight
        self.healthy_reward = healthy_reward

    def forward(self, state, action, next_state=None):
        """Get reward distribution for state, action, next_state."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        reward_state = self.state_reward(state, next_state)
        reward_ctrl = self.action_reward(action)

        cost = tolerance(state[..., 0], lower=0.05, upper=0.15, margin=0.0015) - 1.0
        reward = reward_state + self.ctrl_cost_weight * reward_ctrl + cost

        try:
            self._info.update(
                reward_state=reward_state.sum().item(),
                reward_ctrl=reward_ctrl.sum().item(),
            )
            reward = reward.type(torch.get_default_dtype()).unsqueeze(-1)
        except AttributeError:
            pass
        return reward, torch.zeros_like(reward).unsqueeze(-1)