from saferl.environment.inverted_pendulum import (
    get_pendulum_environment,
    PendulumReward,
)
from rllib.algorithms.mpc import CEMShooting, RandomShooting
from rllib.util.utilities import set_random_seed
from utilities import load_policy, load_model, load_critic, get_parser
from rllib.model import ExpectedModel
from hucrl.model.hallucinated_model import HallucinatedModel
from rllib.value_function.integrate_q_value_function import IntegrateQValueFunction
from saferl.algorithms.safety_filter import SafetyFilterMPC, SafetyFilterReward
from rllib.agent import MPCAgent
from rllib.util.training.agent_training import evaluate_agent
from saferl.policy.mixture_policy import MixturePolicy


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
    safe_critic = load_critic(env, "max_safe")
    safe_policy = load_policy(env, "max_safe")
    safe_terminal_reward = IntegrateQValueFunction(safe_critic, safe_policy)
    test_policy = load_policy(env, "max_unsafe")

    mixture_policy = MixturePolicy(safe_policy, test_policy, alpha=0.35)

    safety_filter_reward = SafetyFilterReward(
        base_model=reward_model, policy=mixture_policy
    )

    safe_mpc = RandomShooting(
        dynamical_model=dynamical_model,
        reward_model=safety_filter_reward,
        policy=mixture_policy,
        terminal_reward=safe_terminal_reward,
    )
    agent = MPCAgent(mpc_solver=safe_mpc)
    evaluate_agent(
        environment=env, agent=agent, num_episodes=args.num_episodes, max_steps=200
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)
