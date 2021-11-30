# import sys
#
# sys.path.append('/home/alederer25/Documents/Code/rllib-dev')
# sys.path.append('/home/alederer25/Documents/Code/hucrl-master')
# sys.path.append('/home/alederer25/Documents/Code/safety-gym-master')

from saferl.environment.inverted_pendulum import get_pendulum_environment
from saferl.algorithms.safe_rl_wrapper import make_safe_agent

from rllib.util.training.agent_training import train_agent
from rllib.agent import TD3Agent
from saferl.utilities.lagrangian import LagrangianReduction

env = get_pendulum_environment()
agent = TD3Agent.default(
    env, multi_objective_reduction=LagrangianReduction(components=env.dim_reward[0] - 1))

safe_agent = make_safe_agent(agent)
train_agent(
    agent=agent,
    environment=env,
    num_episodes=1000,
    max_steps=200,
    eval_frequency=5,
    print_frequency=1,
    render=True,
)
