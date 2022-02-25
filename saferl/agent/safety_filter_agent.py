from rllib.agent import SACAgent
from rllib.agent.model_based.model_based_agent import ModelBasedAgent
from rllib.dataset.experience_replay import ExperienceReplay
from saferl.algorithms.safety_filter import SafetyFilterMPC
from saferl.utilities.multi_objective_reduction import (
    LagrangianReduction,
    NegCostReduction,
)
from saferl.utilities.losses import CMDPPathwiseLoss, MaximallySafeLoss
from rllib.policy.mpc_policy import MPCPolicy


class SafetyFilterAgent(ModelBasedAgent):
    def __init__(
        self,
        dynamical_model,
        reward_model,
        guidance_agent,
        maximally_safe_agent,
        xi=-0.0001,
        gamma=0.99,
        *args,
        **kwargs
    ):
        kwargs.pop("policy", None)
        memory = ExperienceReplay(max_len=100000, num_memory_steps=0)
        safety_filter = SafetyFilterMPC(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            policy=guidance_agent.policy,
            safety_critic=maximally_safe_agent.algorithm.critic,
            safe_policy=maximally_safe_agent.algorithm.policy,
            xi=xi,
            gamma=gamma,
            *args,
            **kwargs,
        )
        policy = MPCPolicy(mpc_solver=safety_filter, solver_frequency=1)

        self.guidance_agent = guidance_agent
        self.maximally_safe_agent = maximally_safe_agent

        self.guidance_agent.model_learning_algorithm = None
        self.maximally_safe_agent.model_learning_algorithm = None

        super(SafetyFilterAgent, self).__init__(
            memory=memory,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            policy=policy,
            gamma=gamma,
            simulation_frequency=0,
            *args,
            **kwargs,
        )

    def __str__(self):
        return (
            super().__str__()
            + str(self.guidance_agent)
            + str(self.maximally_safe_agent)
        )

    def observe(self, observation):
        super().observe(observation.clone())
        self.guidance_agent.observe(observation.clone())
        self.maximally_safe_agent.observe(observation.clone())

    def start_episode(self):
        super().start_episode()
        self.guidance_agent.start_episode()
        self.maximally_safe_agent.start_episode()

    def end_episode(self) -> None:
        super().end_episode()
        self.guidance_agent.end_episode()
        self.maximally_safe_agent.end_episode()

    def end_interaction(self) -> None:
        super().end_interaction()
        self.guidance_agent.end_interaction()
        self.maximally_safe_agent.end_interaction()

    @classmethod
    def default(
        cls,
        environment,
        guidance_agent=None,
        maximally_safe_agent=None,
        *args,
        **kwargs
    ):
        if guidance_agent is None:
            guidance_agent = SACAgent.default(
                environment,
                pathwise_loss_class=CMDPPathwiseLoss,
                multi_objective_reduction=LagrangianReduction(
                    components=environment.dim_reward[0] - 1
                ),
            )
        if maximally_safe_agent is None:
            maximally_safe_agent = SACAgent.default(
                environment,
                pathwise_loss_class=MaximallySafeLoss,
                multi_objective_reduction=NegCostReduction(),
            )
        return super().default(
            environment,
            guidance_agent=guidance_agent,
            maximally_safe_agent=maximally_safe_agent,
            *args,
            **kwargs,
        )
