from itertools import chain

from rllib.agent import TD3Agent
from rllib.agent.model_based.model_based_agent import ModelBasedAgent
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.policy.nn_policy import NNPolicy
from rllib.value_function.nn_ensemble_value_function import NNEnsembleQFunction
from torch.optim import Adam

from saferl.algorithms.maximally_safe import MaximallySafeDADPG
from saferl.utilities.multi_objective_reduction import NegCostReduction


class MaximallySafeAgent(TD3Agent):
    def __init__(self, multi_objective_reduction=NegCostReduction(), *args, **kwargs):

        super().__init__(
            multi_objective_reduction=multi_objective_reduction,
            *args,
            **kwargs,
        )


class ModelBasedMaximallySafeAgent(ModelBasedAgent):
    def __init__(
        self,
        dynamical_model,
        reward_model,
        policy,
        critic,
        gamma=0.99,
        memory=None,
        *args,
        **kwargs
    ):
        if memory is None:
            memory = ExperienceReplay(
                max_len=100000,
                num_memory_steps=0,
            )
        policy_learning_algorithm = MaximallySafeDADPG(
            memory=memory,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            policy=policy,
            critic=critic,
            gamma=gamma,
            *args,
            **kwargs,
        )

        super().__init__(
            memory=memory,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            gamma=gamma,
            policy_learning_algorithm=policy_learning_algorithm,
            *args,
            **kwargs,
        )

    @classmethod
    def default(cls, environment, policy=None, critic=None, lr=1e-3, *args, **kwargs):
        if policy is None:
            policy = NNPolicy.default(environment, *args, **kwargs)
        if critic is None:
            critic = NNEnsembleQFunction.default(environment, *args, **kwargs)

        return super().default(
            environment,
            policy=policy,
            critic=critic,
            optimizer=Adam(chain(policy.parameters(), critic.parameters()), lr=lr),
            *args,
            **kwargs,
        )
