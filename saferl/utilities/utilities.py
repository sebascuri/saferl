"""Utilities for Safe RL package."""
import matplotlib.pyplot as plt
import numpy as np


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
