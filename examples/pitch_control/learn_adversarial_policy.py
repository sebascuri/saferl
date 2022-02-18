"""Given the safe policy, get a hallucination adversarial policy."""
from rllib.util.utilities import set_random_seed

from utilities import get_parser


def main(args):
    pass
    # load model
    # hallucinate model
    # make reduction that maximizes the costs
    # run MB algorithm.


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(num_episodes=100)

    args_ = parser.parse_args()
    set_random_seed(args_.seed)
    main(args_)
