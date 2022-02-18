import torch
from saferl.environment.pitch_control import get_pitch_control_environment

from rllib.util.training.agent_training import train_agent

from rllib.util.utilities import set_random_seed

from utilities import get_parser, create_agent


def main(args):
    """Run main experiment."""
    initial_distribution = torch.distributions.Uniform(
        torch.tensor([-0.1, -0.1, -0.2]),
        torch.tensor([+0.1, +0.1, -0.1]),
    )

    env = get_pitch_control_environment(initial_distribution=initial_distribution)
    # collect data with a policy and learn a model.
    agent = create_agent(
        env,
        args,
        deterministic=False,
        tensorboard=True,
        num_epochs=5,
    )

    train_agent(
        agent=agent,
        environment=env,
        num_episodes=args.num_episodes,
        max_steps=args.horizon,
        eval_frequency=0,
        print_frequency=1,
        render=False,
    )


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(num_episodes=20)

    args_ = parser.parse_args()
    set_random_seed(args_.seed)
    main(args_)
