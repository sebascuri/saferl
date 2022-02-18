import torch
from rllib.dataset.datatypes import TupleDistribution
from rllib.policy import AbstractPolicy
from torch import Tensor
from torch.distributions import Categorical


class MixturePolicy(AbstractPolicy):
    """Return the mixture of two policies."""

    def __init__(
        self, policy_1: AbstractPolicy, policy_2: AbstractPolicy, alpha: float
    ):
        super().__init__(
            dim_state=policy_1.dim_state,
            dim_action=policy_1.dim_action,
            num_states=policy_1.num_states,
            num_actions=policy_1.num_actions,
            tau=policy_1.tau,
            deterministic=policy_1.deterministic,
            action_scale=policy_1.action_scale,
            goal=policy_1.goal,
            dist_params=policy_1.dist_params,
        )
        self.policy_1 = policy_1
        self.policy_2 = policy_2
        self.alpha = alpha

    def reset(self):
        """Reset policy parameters."""
        self.policy_1.reset()
        self.policy_2.reset()

    def update(self):
        """Update policy parameters."""
        self.policy_1.update()
        self.policy_2.update()

    def set_goal(self, goal):
        """Update policy parameters."""
        self.policy_1.set_goal(goal)
        self.policy_2.set_goal(goal)

    def _mix(self, x, y):
        return self.alpha * x + (1 - self.alpha) * y

    def forward(self, state: Tensor) -> TupleDistribution:
        distribution_1 = self.policy_1(state)
        distribution_2 = self.policy_2(state)
        if len(distribution_1) == 1:
            prob_1 = Categorical(logits=distribution_1).probs
            prob_2 = Categorical(logits=distribution_2).probs
            return Categorical(probs=self._mix(prob_1, prob_2)).logits
        else:
            mean_1, cov_1 = distribution_1
            mean_2, cov_2 = distribution_2
            return self._mix(mean_1, mean_2), self._mix(cov_1, cov_2)
