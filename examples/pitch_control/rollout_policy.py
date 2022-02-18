"""Rollout a policy."""

from saferl.environment.pitch_control import get_pitch_control_environment

from utilities import plot_pitch_angle, load_policy, get_parser
from rllib.util.rollout import rollout_policy

if __name__ == "__main__":
    # ["safe", "unsafe", "random", "open_loop", "safe_mpc", "safety_filter"]
    parser = get_parser()
    args = parser.parse_args()
    env = get_pitch_control_environment()
    policy = load_policy(env, args)

    trajectories = rollout_policy(
        environment=env, policy=policy, num_episodes=1, max_steps=1000
    )
    plot_pitch_angle(trajectory=trajectories[0], title=args.objective)
