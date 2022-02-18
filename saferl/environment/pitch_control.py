"""Airplane Pitch Control example."""
import torch
from rllib.environment import SystemEnvironment, SystemEnvironmentBuilder
from rllib.environment.systems.pitch_control import PitchControl
from rllib.model import AbstractModel
from rllib.util.neural_networks.utilities import torch_quadratic


class PitchControlReward(AbstractModel):
    """Reward for Airplane pitch control.

    The reward is given by a quadratic cost with pitch_cost and action_cost controlling
    the coefficients.

    The first constraint is that the pitch angle can't be positive.
    The second constraint is that the action must be between -1 and 1.
    """

    def __init__(self, pitch_cost=2.0, action_cost=1.0 / 50, indicator=False):
        super().__init__(
            dim_state=(3,), dim_action=(1,), dim_reward=(3,), model_kind="rewards"
        )
        self.action_cost = torch.tensor([action_cost])
        self.pitch_cost = torch.tensor([pitch_cost])
        self.indicator = indicator

    def forward(self, state, action, next_state=None):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        pitch_angle = state[..., 2:]

        state_cost = torch_quadratic(pitch_angle, self.pitch_cost)
        action_cost = torch_quadratic(action, self.action_cost)

        reward = -(state_cost + action_cost)

        if self.indicator:
            state_constraint = 1 * [pitch_angle >= 0]
            action_constraint = torch.abs(action) >= 1.0
        else:
            state_constraint = pitch_angle
            action_constraint = torch.abs(action) - 1.0

        return torch.cat(
            [reward, state_constraint, action_constraint], dim=-1
        ), torch.zeros(self.dim_reward)


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
        reward=PitchControlReward(action_cost=action_cost),
    )
    return env


class PitchControlBuilder(SystemEnvironmentBuilder):
    def __init__(self, action_cost=0.0, initial_distribution_fn=None, horizon=1000):
        self.action_cost = action_cost
        if initial_distribution_fn is None:
            initial_distribution_fn = torch.distributions.Uniform(
                torch.tensor([-0.001, -0.001, -0.2 - 0.001]),
                torch.tensor([+0.001, +0.001, -0.2 + 0.001]),
            ).sample
        self.initial_distribution_fn = initial_distribution_fn
        self.horizon = horizon

    def get_system_model(self):
        """Get dynamical model."""
        return PitchControl()

    def get_reward_model(self):
        """Get reward model."""
        return PitchControlReward(action_cost=self.action_cost)

    def initial_distribution_fn(self):
        """Get Initial Distribution Sample function."""
        return self.initial_distribution_fn
