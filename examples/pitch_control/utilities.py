from saferl.environment.pitch_control import PitchControlReward
import matplotlib.pyplot as plt
from rllib.value_function.integrate_q_value_function import IntegrateQValueFunction

import torch
import argparse
import os
from saferl.algorithms.safe_mpc import SafeMPC
from saferl.algorithms.safety_filter import SafetyFilterMPC
from rllib.policy.mpc_policy import MPCPolicy

from rllib.agent import ModelBasedAgent, FittedValueEvaluationAgent
from rllib.policy.zero_policy import ZeroPolicy
from rllib.policy.proportional_policy import ProportionalPolicy
from rllib.policy.random_policy import RandomPolicy
from saferl.utilities.utilities import plot_violation


def plot_pitch_angle_callback(agent, environment, episode: int):
    with torch.no_grad():
        angle = [observation.state[2].numpy() for observation in agent.last_trajectory]

    fig, ax = plt.subplots(1)
    plot_violation(angle, violation_limit=0, ax=ax)
    ax.set_ylim([-0.25, 0.1])
    ax.set_title(f"{agent.name}, episode: {episode}")
    plt.show()


def get_parser():
    parser = argparse.ArgumentParser("Run inverted pitch control experiment.")
    parser.add_argument("--seed", type=int, default=0, help="Random Seed.")
    parser.add_argument("--horizon", type=int, default=1000, help="Horizon to run.")

    parser.add_argument(
        "--num-episodes", type=int, default=10, help="Number of episodes."
    )
    parser.add_argument(
        "--hallucinate",
        action="store_true",
        default=False,
        help="Whether to hallucinate or not.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=False,
        help="Whether to calibrate or not.",
    )

    parser.add_argument(
        "--objective",
        type=str,
        default="safe",
        choices=["safe", "unsafe", "random", "open_loop", "safe_mpc", "safety_filter"],
        help="Objective to optimize.",
    )

    parser.add_argument(
        "--xi", type=float, default=-0.001, help="Safety filter parameter.",
    )
    return parser


def get_name(args):
    """Get experiment name from args."""
    return f"{args.objective}_{args.calibrate}"


def load_policy(env, args, name=None):
    """Load a policy."""
    name = args.objective if name is None else name
    if name == "safe":
        policy = ProportionalPolicy.default(
            env, gain=-torch.tensor([[-0.5034, 52.8645, 1.4142]])
        )
    elif name == "unsafe":
        policy = ProportionalPolicy.default(
            env, gain=-10 * torch.tensor([[-0.5034, 52.8645, 1.4142]])
        )
    elif name == "random":
        policy = RandomPolicy.default(env)
        policy.action_scale = torch.tensor(1.0)
    elif name == "open_loop":
        policy = ZeroPolicy.default(env)
    elif name == "safe_mpc":
        model = load_model(env, args)
        safe_mpc = SafeMPC(dynamical_model=model, reward_model=PitchControlReward())
        policy = MPCPolicy(safe_mpc)
    elif name == "safety_filter":
        model = load_model(env, args)
        critic = load_critic(env, args)
        safe_policy = load_policy(env, args, name="safe")

        critic = IntegrateQValueFunction(q_function=critic, policy=safe_policy)

        guidance_policy = load_policy(env, args, name="safe")
        safety_filter = SafetyFilterMPC(
            dynamical_model=model,
            reward_model=PitchControlReward(),
            safety_critic=critic,
            policy=guidance_policy,
            safe_policy=safe_policy,
            xi=args.xi,
        )
        policy = MPCPolicy(safety_filter)
    else:
        raise NotImplementedError(f"{args.objective} not implemented.")

    return policy


def create_agent(env, args, **kwargs):
    policy = load_policy(env, args, name=args.objective)
    reward_model = PitchControlReward()

    agent_fn = get_model_based_agent_factory(
        policy=policy,
        reward_model=reward_model,
        num_episodes=args.num_episodes,
        name=get_name(args),
        calibrate=args.calibrate,
        **kwargs,
    )
    agent = agent_fn(env)
    return agent


def load_agent(env, args):
    name = f"safe_{args.calibrate}"
    base_dir = "runs/ModelBasedAgent"
    folder = [x for x in filter(lambda x: x.startswith(name), os.listdir(base_dir))][0]
    path = f"{base_dir}/{folder}/last.pkl"

    policy = load_policy(env, args, name="safe")

    agent_factory = get_model_based_agent_factory(
        name, reward_model=PitchControlReward(), policy=policy
    )
    agent = agent_factory(env)
    agent.load(path)
    return agent


def load_model(env, args):
    agent = load_agent(env, args)
    agent.logger.delete_directory()
    return agent.dynamical_model


def load_critic(env, args):
    agent = load_agent(env, args)
    agent.logger.delete_directory()
    return agent.algorithm.critic


def get_model_based_agent_factory(name, policy, num_episodes=10, *args, **kwargs):
    def make_agent(env):
        fve = FittedValueEvaluationAgent.default(env, policy=policy)
        fve.logger.delete_directory()
        agent = ModelBasedAgent.default(
            policy_learning_algorithm=fve.algorithm,
            optimizer=fve.optimizer,
            environment=env,
            model_learn_exploration_episodes=min(num_episodes - 1, 5),
            comment=name,
            *args,
            **kwargs,
        )
        return agent

    return make_agent
