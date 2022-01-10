"""Airplane Pitch Control example."""
import torch
import numpy as np
from rllib.environment import SystemEnvironment
from rllib.environment.systems.pitch_control import PitchControl
from rllib.model import AbstractModel
from rllib.util.neural_networks.utilities import torch_quadratic


class PitchReward(AbstractModel):
    """Reward for Inverted Pendulum."""

    def __init__(self, action_cost=1.0 / 50):
        super().__init__(
            dim_state=(3,), dim_action=(1,), dim_reward=(1,), model_kind="rewards"
        )
        self.action_cost = action_cost

    def forward(self, state, action, next_state=None):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        pitch_angle = state[..., 2:]

        state_cost = torch_quadratic(pitch_angle, torch.tensor(2.0))
        action_cost = torch_quadratic(action, torch.tensor(self.action_cost))

        cost = -(state_cost + action_cost)

        return cost, torch.zeros(self.dim_reward)


def get_pitch_control_environment(
    action_cost=0.0,
    initial_distribution=torch.distributions.Uniform(
        torch.tensor([-0.001, -0.001, -0.2 - 0.001]),
        torch.tensor([+0.001, +0.001, -0.2 + 0.001]),
    ),
):
    """Get pendulum environment."""
    env = SystemEnvironment(
        system=PitchControl(),
        initial_state=initial_distribution.sample,
        reward=PitchReward(action_cost=action_cost),
    )
    return env


if __name__ == "__main__":
    from rllib.agent.random_agent import RandomAgent
    from rllib.util.utilities import RandomAgent

    env = get_pitch_control_environment()
    agent = RandomAgent.default(env)
