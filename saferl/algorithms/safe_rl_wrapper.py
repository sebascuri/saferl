"""Wrap an agent with a safe RL algorithm."""
from typing import Any, Type

from rllib.agent import AbstractAgent
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.algorithms.derived_algorithm import DerivedAlgorithm
from rllib.dataset.datatypes import Loss, Observation
from rllib.util.losses.pathwise_loss import PathwiseLoss
from rllib.util.utilities import sample_action

from saferl.utilities.lagrangian import LagrangianReduction
from saferl.utilities.utilities import get_q_value_pathwise_gradients


class SafeRLPathwiseLoss(PathwiseLoss):
    multi_objective_reduction: LagrangianReduction

    def forward(self, observation, **kwargs):
        state = observation.state
        action = sample_action(self.policy, state)
        q = get_q_value_pathwise_gradients(
            self.critic, state, action, self.multi_objective_reduction
        )

        return Loss(policy_loss=-q, dual_loss=self.multi_objective_reduction.lagrangian)


class SafeRLAlgorithm(DerivedAlgorithm):
    """
    gradient descent on J + lambda_i J_i   w.r.t. policies.
    gradient ascent on J + lambda_i J_i    w.r.t. lambda.
    """

    def __init__(
        self, base_algorithm: AbstractAlgorithm, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(base_algorithm, *args, **kwargs)

    def forward(self, observation: Observation, **kwargs) -> Loss:
        loss = self.base_algorithm(observation)
        lagrangian = self.base_algorithm.multi_objective_reduction.lagrangian
        if lagrangian.dim() > 1:
            lagrangian = lagrangian.mean(dim=-1)
        return loss + Loss(dual_loss=lagrangian)


def make_safe_agent(agent: AbstractAgent) -> AbstractAgent:
    """Make an agent safe by wrapping the algorithm around the SafeRL algorithm."""
    agent.algorithm = SafeRLAlgorithm(base_algorithm=agent.algorithm)

    agent.optimizer = type(agent.optimizer)(
        [
            p
            for n, p in agent.algorithm.named_parameters()
            if "target" not in n and "old_policy" not in n
        ],
        **agent.optimizer.defaults,
    )
    return agent
