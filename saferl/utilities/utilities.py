import matplotlib.pyplot as plt
import numpy as np
import torch
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.value_function import NNEnsembleQFunction


def get_q_value_pathwise_gradients(critic, state, action, multi_objective_reduction):
    with DisableGradient(critic):
        q = critic(state, action)
        if isinstance(critic, NNEnsembleQFunction):
            q = q[..., 0]

    # Take multi-objective reduction.
    q = multi_objective_reduction(q)
    # Take mean over time coordinate.
    if q.dim() > 1:
        q = q.mean(dim=1)
    return q


def _compute_constraint_violation(agent):
    violations = np.array(
        [
            observation.reward[1:].detach().numpy() > 0
            for observation in agent.last_trajectory
        ]
    )
    violation_dict = {
        "violation": np.sum(violations),
    }
    violation_dict.update(
        **{
            f"violation-{i + 1}": np.sum(violations[:, i])
            for i in range(violations.shape[1])
        }
    )
    agent.logger.update(**violation_dict)


def compute_constraint_violation(agent, environment, episode):
    _compute_constraint_violation(agent)


def plot_violation(coordinate, violation_limit=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(coordinate)
    ax.axhline(y=violation_limit, color="k", linestyle="dashed")
    return ax
