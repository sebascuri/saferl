# import sys
#
# sys.path.append('/home/alederer25/Documents/Code/rllib-dev')
# sys.path.append('/home/alederer25/Documents/Code/hucrl-master')
# sys.path.append('/home/alederer25/Documents/Code/safety-gym-master')

from rllib.environment.systems import InvertedPendulum
from rllib.environment import SystemEnvironment
import torch
import torch.distributions
from rllib.model import AbstractModel
from rllib.reward.utilities import tolerance
from rllib.util.training.agent_training import train_agent
from rllib.agent import SACAgent
import numpy as np


class PendulumReward(AbstractModel):
    """Reward for Inverted Pendulum."""

    def __init__(self, action_cost=0.0):
        super().__init__(
            dim_state=(2,), dim_action=(1,), dim_reward=(3,), model_kind="rewards"
        )
        self.action_cost = action_cost
        self.reward_offset = 0

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

        angle_constraint = tolerance(state[...,0],lower=-1.5*np.pi,upper=0.1,margin=0.01)-1.0

        cost = torch.stack([state_cost, action_cost, angle_constraint], dim=-1)
        return cost, torch.zeros(3)


initial_distribution = torch.distributions.Uniform(
        torch.tensor([-1.2*np.pi, +0.0]), torch.tensor([0, -0.0])
    )

env = SystemEnvironment(
    system=InvertedPendulum(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80),
    initial_state=initial_distribution.sample,
    reward=PendulumReward(action_cost=0.0),
)

agent = SACAgent.default(env)

train_agent(
    agent=agent,
    environment=env,
    num_episodes=1000,
    max_steps=200,
    eval_frequency=5,
    print_frequency=1,
    render=False,
)
