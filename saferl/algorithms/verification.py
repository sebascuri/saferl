"""Verification algorithm.

Maximize J^pi, for hallucination
"""
import torch
from hucrl.model.hallucinated_model import HallucinatedModel
from hucrl.policy.augmented_policy import AugmentedPolicy
from rllib.algorithms.mpc import CEMShooting
from rllib.dataset.datatypes import Loss
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.model.closed_loop_model import ClosedLoopModel
from rllib.util.losses.pathwise_loss import PathwiseLoss
from rllib.util.rollout import rollout_actions
from rllib.util.utilities import sample_action
from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.value_estimation import discount_sum
from torch.distributions import MultivariateNormal

from saferl.utilities.multi_objective_reduction import LagrangianReduction
from saferl.utilities.utilities import get_q_value_pathwise_gradients


class VerifySafeLoss(PathwiseLoss):
    """
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


class VerifySafeMPC(CEMShooting):
    """Verify MPC algorithm."""

    def __init__(self, *args, **kwargs):
        super(VerifySafeMPC, self).__init__(*args, **kwargs)
        if isinstance(self.dynamical_model, HallucinatedModel):
            self.h_dim_action = self.dynamical_model.dim_state
        else:
            self.h_dim_action = (0,)

    def get_best_action(self, action_sequence, returns):
        """Get best action by: see below.

        - Maximize true actions w.r.t. cost.
        - Minimize hallucinated actions w.r.t. cost.
        """
        cost = returns[..., 1:]

        valid_indexes = torch.where(torch.all(cost <= 0, dim=-1))[0]
        invalid_indexes = torch.where(~torch.all(cost <= 0, dim=-1))[0]

        returns[valid_indexes] = cost[valid_indexes].sum(-1)
        returns[invalid_indexes] = -float("inf")

        idx = torch.topk(
            returns, k=min(self.num_elites, len(valid_indexes)), largest=True, dim=-1
        )[1]

        idx = idx.unsqueeze(0).unsqueeze(-1)  # Expand dims to action_sequence.
        idx = idx.repeat_interleave(self.horizon, 0).repeat_interleave(
            self.dim_action, -1
        )
        return torch.gather(action_sequence, -2, idx)


def verify(
    dynamical_model,
    reward_model,
    policy,
    state,
    termination_model=None,
    terminal_reward=None,
    gamma=0.99,
    horizon=10,
    num_samples=1000,
):
    """Verify if a policy is safe."""
    state = repeat_along_dimension(state, number=num_samples, dim=-2)

    if isinstance(dynamical_model, HallucinatedModel):
        h_dim = state.shape[-1]
        action_distribution = MultivariateNormal(torch.zeros(h_dim), torch.eye(h_dim))
        sample_shape = (horizon, num_samples)
        action_sequence = action_distribution.sample(sample_shape).clamp(-1.0, 1.0)
    else:
        action_sequence = torch.zeros((horizon, num_samples, 0))
    dynamical_model = ClosedLoopModel(dynamical_model, policy)
    reward_model = ClosedLoopModel(reward_model, policy)

    trajectory = stack_list_of_tuples(
        rollout_actions(
            dynamical_model,
            reward_model,
            action_sequence,
            state,
            termination_model,
        ),
        dim=-2,
    )

    returns = discount_sum(trajectory.reward, gamma)
    if terminal_reward:
        terminal_reward = terminal_reward(trajectory.next_state[..., -1, :])
        if terminal_reward.dim() > returns.dim():
            # the terminal reward is an ensemble.
            terminal_reward = terminal_reward.min(dim=-1)[0]
        returns = returns + gamma ** horizon * terminal_reward
    cost = returns[..., 1:]

    is_safe = torch.all(torch.all(cost <= 0, dim=-1))
    return is_safe, trajectory
