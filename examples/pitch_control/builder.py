"""File that builds the agent for the pitch control experiments."""
from rllib.agent.fixed_policy_agent import FixedPolicyAgent
from rllib.policy.proportional_policy import ProportionalPolicy
import torch
from examples.agent_builder import AgentBuilder


class PitchControlAgentBuilder(AgentBuilder):
    """Builder for Pitch control experiments."""

    def _get_known_safe(self, environment, *args, **kwargs):
        return FixedPolicyAgent(
            policy=ProportionalPolicy.default(
                environment, gain=-torch.tensor([[-0.5034, 52.8645, 1.4142]])
            ),
            name="KnownSafeAgent",
        )

    def _get_known_unsafe(self, environment, *args, **kwargs):
        return FixedPolicyAgent(
            policy=ProportionalPolicy.default(
                environment, gain=-10 * torch.tensor([[-0.5034, 52.8645, 1.4142]])
            ),
            name="KnownUnsafeAgent",
        )
