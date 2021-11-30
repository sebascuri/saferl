import numpy as np
import torch
import torch.distributions
from rllib.environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum
from rllib.model import AbstractModel
from rllib.reward.utilities import tolerance


class PendulumReward(AbstractModel):
    """Reward for Inverted Pendulum."""

    def __init__(self, action_cost=0.0, indicator=True):
        super().__init__(
            dim_state=(2,), dim_action=(1,), dim_reward=(3,), model_kind="rewards"
        )
        self.action_cost = action_cost
        self.reward_offset = 0
        self.indicator = indicator

    def forward(self, state, action, next_state):
        """See `abstract_reward.forward'."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        cos_angle = torch.cos(state[..., 0])
        velocity = state[..., 1]

        angle_tolerance = tolerance(cos_angle, lower=0.95, upper=1.0, margin=0.1)
        velocity_tolerance = tolerance(velocity, lower=-0.5, upper=0.5, margin=0.5)
        state_cost = angle_tolerance * velocity_tolerance

        action_tolerance = tolerance(action[..., 0], lower=-0.1, upper=0.1, margin=0.1)
        action_cost = self.action_cost * (action_tolerance - 1)

        if self.indicator:
            cost = torch.stack(
                [
                    state_cost + action_cost,
                    1.0 * (state[..., 0] > 0.1 * np.pi),
                    1.0 * (state[..., 0] < -1.5 * np.pi),
                ],
                dim=-1,
            )
        else:
            cost = torch.stack(
                [
                    state_cost + action_cost,
                    -state[..., 0] + 0.1 * np.pi,
                    state[..., 0] + 1.5 * np.pi,
                ],
                dim=-1,
            )
        return cost, torch.zeros(self.dim_reward)


def get_pendulum_environment(action_cost=0.0, initial_distribution=None):
    """Get pendulum environment."""
    if initial_distribution is None:
        initial_distribution = torch.distributions.Uniform(
            torch.tensor([-1.2 * np.pi, -0.001]), torch.tensor([0, +0.001])
        )
    env = SystemEnvironment(
        system=InvertedPendulum(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80),
        initial_state=initial_distribution.sample,
        reward=PendulumReward(action_cost=action_cost),
    )
    return env
