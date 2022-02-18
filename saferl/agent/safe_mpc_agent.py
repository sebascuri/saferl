from rllib.agent import ModelBasedAgent, MPCAgent

from saferl.algorithms.safe_cem import SafeCEM


class SafeMPCAgent(MPCAgent):
    @classmethod
    def default(cls, environment, *args, **kwargs):
        agent = ModelBasedAgent.default(environment, *args, **kwargs)
        agent.logger.delete_directory()
        kwargs.update(
            dynamical_model=agent.dynamical_model,
            reward_model=agent.reward_model,
            termination_model=agent.termination_model,
            gamma=agent.gamma,
        )
        mpc_solver = SafeCEM(action_scale=environment.action_scale, *args, **kwargs)

        return super().default(environment, mpc_solver=mpc_solver, *args, **kwargs)
