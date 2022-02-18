"""This files contains utilities for Lagrangian based algorithms."""
import torch
from rllib.util.losses.lagrangian_loss import LagrangianLoss
from rllib.util.multi_objective_reduction import AbstractMultiObjectiveReduction


class LagrangianReduction(AbstractMultiObjectiveReduction):
    """The lagrangian reduction uses a lagrangian loss.

    The Lagrangian Loss is for inequalities:
        g(x) - c < epsilon,
    i.e., negative g(x) is good.

    """

    def __init__(self, components=1, dim=2):
        super(LagrangianReduction, self).__init__(dim=dim)
        self._lagrangian = LagrangianLoss(
            inequality_zero=1e-3, dual=torch.ones(components), regularization=False
        )
        self.lagrangian = torch.zeros(components)

    def forward(self, value):
        """Reduce the value."""
        base = value[..., 0]
        others = value[..., 1:]
        loss = self._lagrangian(others)
        self.lagrangian = loss.dual_loss.mean(dim=self.dim)
        # maximize the loss while minimizing the constraints.
        return base - loss.reg_loss.sum(dim=self.dim)


class CostReduction(AbstractMultiObjectiveReduction):
    """Multi-objective reduction that returns the costs."""

    def __init__(self, dim=2):
        super(CostReduction, self).__init__(dim=dim)

    def forward(self, value):
        """Reduce the value."""
        cost = value[..., 1:]
        # maximize the constraints.
        return cost.sum(dim=self.dim)


class NegCostReduction(AbstractMultiObjectiveReduction):
    """Multi-objective reduction that returns the costs.

    This is useful when maximizing the safety constraints.
    """

    def __init__(self, dim=2):
        super(NegCostReduction, self).__init__(dim=dim)

    def forward(self, value):
        """Reduce the value."""
        cost = value[..., 1:]
        # minimize the constraints.
        return -cost.sum(dim=self.dim)
