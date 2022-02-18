"""Wrap an agent with a safe RL algorithm."""
from typing import Any

from rllib.algorithms.data_augmentation import DataAugmentation
from rllib.algorithms.sac import SAC
from rllib.dataset.datatypes import Loss
from rllib.util.losses.pathwise_loss import PathwiseLoss, get_q_value_pathwise_gradients
from rllib.util.utilities import sample_action

from saferl.utilities.multi_objective_reduction import LagrangianReduction


class CMDPPathwiseLoss(PathwiseLoss):
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


#
# class SafeSAC(SAC):
#     """
#     gradient descent on J + lambda_i J_i   w.r.t. policies.
#     gradient ascent on J + lambda_i J_i    w.r.t. lambda.
#     """
#
#     def __init__(
#         self, policy, critic, multi_objective_reduction=None, *args: Any, **kwargs: Any
#     ) -> None:
#         del multi_objective_reduction
#         super().__init__(
#             policy=policy,
#             critic=critic,
#             multi_objective_reduction=LagrangianReduction(
#                 components=critic.dim_reward[0] - 1
#             ),
#             *args,
#             **kwargs
#         )
#         self.pathwise_loss = CMDPPathwiseLoss(
#             policy=self.policy,
#             critic=self.critic,
#             multi_objective_reduction=self.multi_objective_reduction,
#         )
#
#
# class SafeDASAC(DataAugmentation):
#     def __init__(self, *args, **kwargs):
#         super().__init__(base_algorithm=SafeSAC(*args, **kwargs), *args, **kwargs)
