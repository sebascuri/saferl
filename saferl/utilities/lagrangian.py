"""This files contains utilities for Lagrangian based algorithms."""
import torch
from rllib.util.losses.lagrangian_loss import LagrangianLoss
from rllib.util.multi_objective_reduction import AbstractMultiObjectiveReduction


class LagrangianReduction(AbstractMultiObjectiveReduction):
    def __init__(self, components=1, dim=2):
        super(LagrangianReduction, self).__init__(dim=dim)
        self._lagrangian = LagrangianLoss(
            inequality_zero=1e-3, dual=torch.ones(components), regularization=False
        )
        self.lagrangian = torch.zeros(components)

    def __call__(self, value):
        """Reduce the value."""
        base = value[..., 0]
        others = value[..., 1:]
        loss = self._lagrangian(others)
        self.lagrangian = loss.dual_loss.mean(dim=self.dim)
        # maximize the loss while minimizing the constraints.
        return base - loss.reg_loss.sum(dim=self.dim)