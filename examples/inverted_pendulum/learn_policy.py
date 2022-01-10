from saferl.environment.inverted_pendulum import get_pendulum_environment

from rllib.util.training.agent_training import train_agent
from rllib.util.utilities import set_random_seed
from utilities import get_model_free_agent_factory, get_parser


def main(args):
    """Run main experiment."""
    env = get_pendulum_environment(action_cost=args.action_cost)
    agent_factory = get_model_free_agent_factory(
        args.objective, exploration_episodes=10
    )
    agent = agent_factory(env)

    train_agent(
        agent=agent,
        environment=env,
        num_episodes=args.num_episodes,
        max_steps=args.horizon,
        eval_frequency=0,
        print_frequency=1,
        render=True,
    )


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(num_episodes=100)

    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)
