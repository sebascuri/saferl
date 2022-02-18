"""Run the main experiment."""
from builder import parser, AgentBuilder
from saferl.environment.pitch_control import PitchControlBuilder
from utilities import plot_pitch_angle_callback
from saferl.utilities.utilities import compute_constraint_violation
from rllib.util.rollout import rollout_agent

if __name__ == "__main__":

    experiment = parser.parse_args()
    agent_builder = AgentBuilder(experiment)
    environment_builder = PitchControlBuilder()

    env = environment_builder.create_environment()
    agent = agent_builder.create_agent(
        environment=env,
        reward_model=environment_builder.get_reward_model(),
        exploration_episodes=5,
        model_learn_exploration_episodes=4,
        pre_train_iterations=100,
        num_model_steps=1,
        num_particles=1,
        num_epochs=1,
        target_entropy=-1.0,
        entropy_regularization=False,  # entropy constraint.
    )
    rollout_agent(
        environment=env,
        agent=agent,
        num_episodes=experiment.num_episodes,
        max_steps=environment_builder.horizon,
        print_frequency=1,
        callback_frequency=1,
        callbacks=[plot_pitch_angle_callback, compute_constraint_violation],
    )
