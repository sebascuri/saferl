"""Verification algorithm.

Maximize J^pi, for hallucination
"""
import torch
from hucrl.model.hallucinated_model import HallucinatedModel
from hucrl.policy.augmented_policy import AugmentedPolicy
from rllib.algorithms.mpc import CEMShooting
from rllib.dataset.datatypes import Loss
from rllib.util.losses.pathwise_loss import PathwiseLoss
from rllib.util.utilities import sample_action

from saferl.utilities.lagrangian import LagrangianReduction
from saferl.utilities.utilities import get_q_value_pathwise_gradients


class MaximallySafeLoss(PathwiseLoss):
    """
    gradient descent on lambda_i J_i  w.r.t. true policy.
    gradient ascent on lambda_i J_i   w.r.t. hallucination policy.
    gradient ascent on lambda_i J_i   w.r.t. lambda.
    """

    policy: AugmentedPolicy

    multi_objective_reduction: LagrangianReduction

    def forward(self, observation, **kwargs):
        state = observation.state

        if isinstance(self.policy, AugmentedPolicy):
            true_action = sample_action(self.policy.true_policy, state)
            hall_action = sample_action(self.policy.hallucination_policy, state)

            action = torch.cat((true_action.detach(), hall_action), dim=-1)
            q_hall = get_q_value_pathwise_gradients(
                self.critic, state, action, self.multi_objective_reduction
            )

            q_true = torch.cat((true_action, hall_action.detach()), dim=-1)
            # we are doing gradient ascent w.r.t. the hallucinated actions and gradient
            # descent w.r.t the true actions.
            policy_loss = -q_true + q_hall
        else:
            action = sample_action(self.policy, state)
            q = get_q_value_pathwise_gradients(
                self.critic, state, action, self.multi_objective_reduction
            )
            policy_loss = -q
        return Loss(
            policy_loss=policy_loss, dual_loss=self.multi_objective_reduction.lagrangian
        )


class MaximallySafeMPC(CEMShooting):
    def __init__(self, *args, **kwargs):
        super(MaximallySafeMPC, self).__init__(*args, **kwargs)
        if isinstance(self.dynamical_model, HallucinatedModel):
            self.h_dim_action = self.dynamical_model.dim_state
        else:
            self.h_dim_action = (0,)

    def _get_action_by_index(self, action_sequence, index):
        idx = index.unsqueeze(0).unsqueeze(-1)  # Expand dims to action_sequence.
        idx = idx.repeat_interleave(self.horizon, 0).repeat_interleave(
            self.dim_action, -1
        )
        return torch.gather(action_sequence, -2, idx)

    def get_best_action(self, action_sequence, returns):
        """Get best action by: see below.

        - Maximize true actions w.r.t. cost.
        - Minimize hallucinated actions w.r.t. cost.
        """
        cost = returns[..., 1:].sum(-1)
        idx = torch.topk(cost, k=self.num_elites, largest=True, dim=-1)[1]
        max_actions = self._get_action_by_index(action_sequence, idx)
        idx = torch.topk(cost, k=self.num_elites, largest=False, dim=-1)[1]
        min_actions = self._get_action_by_index(action_sequence, idx)
        actions = min_actions[..., : -self.h_dim_action[0]]
        hall_actions = max_actions[..., -self.h_dim_action[0] :]
        return torch.cat((actions, hall_actions), dim=-1)
