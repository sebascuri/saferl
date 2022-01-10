import argparse
from rllib.agent import ModelBasedAgent, TD3Agent
from rllib.util.multi_objective_reduction import GetIndexMultiObjectiveReduction

from saferl.utilities.multi_objective_reduction import (
    LagrangianReduction,
    CostReduction,
    NegCostReduction,
)
import os
from saferl.algorithms.safe_rl_wrapper import SafeRLPathwiseLoss


def get_parser():
    parser = argparse.ArgumentParser("Run inverted pendulum experiment.")
    parser.add_argument("--seed", type=int, default=0, help="Random Seed.")
    parser.add_argument("--horizon", type=int, default=200, help="Horizon to evaluate.")

    parser.add_argument("--num-episodes", type=int, default=10, help="Random Seed.")
    parser.add_argument("--action-cost", type=float, default=0.00, help="Action Cost.")
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
        default="lagrangian",
        choices=["standard", "lagrangian", "max_safe", "max_unsafe"],
        help="Objective to optimize.",
    )
    return parser


def load_policy(env, name="lagrangian"):
    base_dir = "runs/TD3Agent"
    folder = [x for x in filter(lambda x: x.startswith(name), os.listdir(base_dir))][0]
    path = f"{base_dir}/{folder}/best.pkl"

    agent_factory = get_model_free_agent_factory(name)
    agent = agent_factory(env)
    agent.load(path)
    agent.logger.delete_directory()
    return agent.policy


def load_critic(env, name="lagrangian"):
    base_dir = "runs/TD3Agent"
    folder = [x for x in filter(lambda x: x.startswith(name), os.listdir(base_dir))][0]
    path = f"{base_dir}/{folder}/last.pkl"

    agent_factory = get_model_free_agent_factory(name)
    agent = agent_factory(env)
    agent.load(path)
    agent.logger.delete_directory()

    critic = agent.algorithm.critic
    return critic


def load_model(env, name="lagrangian", reward_model=None):
    base_dir = "runs/ModelBasedAgent"
    folder = [x for x in filter(lambda x: x.startswith(name), os.listdir(base_dir))][0]
    path = f"{base_dir}/{folder}/last.pkl"

    policy = load_policy(env, name=name)

    agent_factory = get_model_based_agent_factory(
        name, reward_model=reward_model, policy=policy
    )
    agent = agent_factory(env)
    agent.load(path)
    agent.logger.delete_directory()
    return agent.dynamical_model


def get_model_free_agent_factory(name, *args, **kwargs):
    if name == "lagrangian":

        def make_agent(env):
            agent = TD3Agent.default(
                env,
                multi_objective_reduction=LagrangianReduction(
                    components=env.dim_reward[0] - 1
                ),
                comment=name,
                *args,
                **kwargs,
            )
            agent.algorithm.pathwise_loss = SafeRLPathwiseLoss(
                critic=agent.algorithm.critic,
                policy=agent.algorithm.policy,
                multi_objective_reduction=agent.algorithm.multi_objective_reduction,
            )
            return agent

    elif name == "max_safe":

        def make_agent(env):
            agent = TD3Agent.default(
                env,
                multi_objective_reduction=NegCostReduction(),
                comment=name,
                *args,
                **kwargs,
            )
            return agent

    elif name == "standard":

        def make_agent(env):
            return TD3Agent.default(
                env,
                multi_objective_reduction=GetIndexMultiObjectiveReduction(idx=0),
                comment=name,
                *args,
                **kwargs,
            )

    elif name == "max_unsafe":

        def make_agent(env):
            return TD3Agent.default(
                env,
                multi_objective_reduction=GetIndexMultiObjectiveReduction(idx=1),
                comment=name,
                *args,
                **kwargs,
            )

    else:
        raise NotImplementedError

    return make_agent


def get_model_based_agent_factory(name, num_episodes=10, *args, **kwargs):
    def make_agent(env):
        agent = ModelBasedAgent.default(
            environment=env,
            model_learn_exploration_episodes=num_episodes - 1,
            comment=name,
            *args,
            **kwargs,
        )
        return agent

    return make_agent
