from rllib.agent import ModelBasedAgent, MPCAgent

from saferl.algorithms.safe_mpc import SafeMPC
from saferl.algorithms.maximally_safe_mpc import MaximallySafeMPC


class SafeMPCAgent(MPCAgent):
    """Safe MPC Agent."""

    @classmethod
    def default(cls, environment, solver_class=SafeMPC, *args, **kwargs):
        """Default agent creation."""
        agent = ModelBasedAgent.default(environment, *args, **kwargs)
        agent.logger.delete_directory()
        kwargs.update(
            dynamical_model=agent.dynamical_model,
            reward_model=agent.reward_model,
            termination_model=agent.termination_model,
            gamma=agent.gamma,
        )
        mpc_solver = solver_class(
            action_scale=environment.action_scale, *args, **kwargs
        )
        kwargs.pop("policy")
        return super().default(environment, mpc_solver=mpc_solver, *args, **kwargs)


class MaximallySafeMPCAgent(MPCAgent):
    """Safe MPC Agent."""

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """Default agent creation."""
        return super().default(
            environment, solver_class=MaximallySafeMPC, *args, **kwargs
        )
