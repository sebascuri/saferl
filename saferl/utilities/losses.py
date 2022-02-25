"""Safe RL losses."""
import torch
from hucrl.policy.augmented_policy import AugmentedPolicy
from rllib.dataset.datatypes import Loss
from rllib.util.losses.pathwise_loss import PathwiseLoss, get_q_value_pathwise_gradients
from rllib.util.utilities import sample_action

from saferl.utilities.multi_objective_reduction import (
    NegCostReduction,
    LagrangianReduction,
)


class CMDPPathwiseLoss(PathwiseLoss):
    """Pathwise loss for CMDPs."""

    multi_objective_reduction: LagrangianReduction

    def forward(self, observation, **kwargs):
        state = observation.state
        action = sample_action(self.policy, state)
        q = get_q_value_pathwise_gradients(
            self.critic, state, action, self.multi_objective_reduction
        )

        return Loss(
            policy_loss=-q,
            dual_loss=self.multi_objective_reduction.lagrangian.mean(dim=-1),
        )


class MaximallySafeLoss(PathwiseLoss):
    """Pathwise loss for Maximally safe algorithm with hallucination.

    gradient descent on lambda_i J_i  w.r.t. true policy.
    gradient ascent on lambda_i J_i   w.r.t. hallucination policy.
    gradient ascent on lambda_i J_i   w.r.t. lambda.
    """

    policy: AugmentedPolicy

    multi_objective_reduction: NegCostReduction

    def forward(self, observation, **kwargs):
        state = observation.state

        if isinstance(self.policy, AugmentedPolicy):
            true_action = sample_action(self.policy.true_policy, state)
            hall_action = sample_action(self.policy.hallucination_policy, state)

            action = torch.cat((true_action.detach(), hall_action), dim=-1)
            q_hall = get_q_value_pathwise_gradients(
                self.critic, state, action, self.multi_objective_reduction
            )

            action = torch.cat((true_action, hall_action.detach()), dim=-1)
            q_true = get_q_value_pathwise_gradients(
                self.critic, state, action, self.multi_objective_reduction
            )
            # we are doing gradient ascent w.r.t. the hallucinated actions and gradient
            # descent w.r.t the true actions.
            policy_loss = -q_true + q_hall
        else:
            action = sample_action(self.policy, state)
            q = get_q_value_pathwise_gradients(
                self.critic, state, action, self.multi_objective_reduction
            )
            policy_loss = -q
        return Loss(policy_loss=policy_loss)


class VerifySafeLoss(PathwiseLoss):
    """Pathwise loss for verification algorithm.

    policy is fixed.
    gradient ascent on J_i  w.r.t. hallucination policy.
    gradient ascent on J_i w.r.t. lambda.
    """

    multi_objective_reduction: LagrangianReduction

    def forward(self, observation, **kwargs):
        state = observation.state

        if isinstance(self.policy, AugmentedPolicy):
            true_action = sample_action(self.policy.true_policy, state).detach()
            hall_action = sample_action(self.policy.hallucination_policy, state)
            action = torch.cat((true_action, hall_action), dim=-1)
        else:
            action = sample_action(self.policy, state).detach()
        q = get_q_value_pathwise_gradients(
            self.critic, state, action, self.multi_objective_reduction
        )

        # we are doing gradient ascent w.r.t. the hallucinated actions.
        return Loss(policy_loss=+q, dual_loss=self.multi_objective_reduction.lagrangian)
