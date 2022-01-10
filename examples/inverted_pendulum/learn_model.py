import argparse

from saferl.environment.inverted_pendulum import (
    get_pendulum_environment,
    PendulumReward,
)

from rllib.util.training.agent_training import train_agent

from rllib.util.utilities import set_random_seed

from utilities import load_policy, get_model_based_agent_factory, get_parser


def main(args):
    """Run main experiment."""
    env = get_pendulum_environment(action_cost=args.action_cost)
    # collect data with a policy and learn a model.
    policy = load_policy(env, name=args.objective)
    reward_model = PendulumReward(args.action_cost)

    agent_fn = get_model_based_agent_factory(
        policy=policy,
        reward_model=reward_model,
        num_episodes=args.num_episodes,
        name=f"{args.objective}_{args.calibrate}",
        calibrate=args.calibrate,
        tensorboard=True,
        num_epochs=50,
    )
    agent = agent_fn(env)

    train_agent(
        agent=agent,
        environment=env,
        num_episodes=args.num_episodes,
        max_steps=args.horizon,
        eval_frequency=0,
        print_frequency=0,
        render=False,
    )
    print("here")


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(num_episodes=10)

    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)
