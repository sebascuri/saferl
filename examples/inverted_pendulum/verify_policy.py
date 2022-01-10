from saferl.environment.inverted_pendulum import (
    get_pendulum_environment,
    PendulumReward,
)
from functools import reduce

from rllib.util.utilities import set_random_seed
from saferl.algorithms.verification import verify
from utilities import load_policy, load_model, load_critic, get_parser
from saferl.policy.mixture_policy import MixturePolicy
from rllib.value_function import IntegrateQValueFunction
from hucrl.model.hallucinated_model import HallucinatedModel
from rllib.model.expected_model import ExpectedModel
from rllib.util.rollout import rollout_policy

import torch
import numpy as np


def main(args):
    """Run main experiment."""
    env = get_pendulum_environment()
    reward_model = PendulumReward()

    dynamical_model = load_model(env, "max_safe", reward_model=reward_model)

    if args.hallucinate:
        dynamical_model = HallucinatedModel(
            base_model=dynamical_model.base_model,
            transformations=dynamical_model.transformations,
        )
    else:
        dynamical_model = ExpectedModel(
            base_model=dynamical_model.base_model,
            transformations=dynamical_model.transformations,
        )

    # Verify if another policy is unsafe.
    collection_policy = load_policy(env, "max_safe")
    test_policy = load_policy(env, "max_unsafe")

    mixture_policy = MixturePolicy(collection_policy, test_policy, alpha=0.3)
    terminal_reward = None
    for k in range(5):

        trajectory = rollout_policy(
            environment=env,
            policy=mixture_policy,
            max_steps=args.horizon,
            num_episodes=1,
        )
        returns = reduce(lambda a, b: a + b, map(lambda x: x.reward, trajectory[0]))
        print("true", returns)

        is_safe, trajectories = verify(
            dynamical_model,
            reward_model,
            mixture_policy,
            trajectory[0][0].state,
            termination_model=None,
            terminal_reward=terminal_reward,
            gamma=0.99,
            horizon=args.horizon,
            num_samples=1000,
        )
        print("sim", trajectories.reward.mean(0).sum(0))
        print("is_safe?", is_safe)


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(horizon=200)

    args = parser.parse_args()

    set_random_seed(args.seed)
    main(args)
