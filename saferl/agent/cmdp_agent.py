from itertools import chain

import torch.nn.modules.loss as loss
from rllib.agent import SACAgent
from rllib.agent.model_based.model_based_agent import ModelBasedAgent
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.policy.nn_policy import NNPolicy
from rllib.value_function.nn_ensemble_value_function import NNEnsembleQFunction
from torch.optim import Adam

from saferl.algorithms.safe_rl_wrapper import SafeDASAC, SafeSAC


class CMDPAgent(SACAgent):
    def __init__(
        self,
        critic,
        policy,
        criterion=loss.MSELoss,
        *args,
        **kwargs,
    ):
        super().__init__(critic=critic, policy=policy, *args, **kwargs)

        self.algorithm = SafeSAC(
            critic=critic,
            policy=policy,
            criterion=criterion(reduction="none"),
            *args,
            **kwargs,
        )
        self.optimizer = type(self.optimizer)(
            [
                p
                for n, p in self.algorithm.named_parameters()
                if "target" not in n and "old_policy" not in n
            ],
            **self.optimizer.defaults,
        )


class ModelBasedCMDPAgent(ModelBasedAgent):
    def __init__(
        self,
        dynamical_model,
        reward_model,
        policy,
        critic,
        gamma=0.99,
        memory=None,
        *args,
        **kwargs,
    ):
        if memory is None:
            memory = ExperienceReplay(
                max_len=100000,
                num_memory_steps=0,
            )
        policy_learning_algorithm = SafeDASAC(
            memory=memory,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            policy=policy,
            critic=critic,
            gamma=gamma,
            *args,
            **kwargs,
        )

        super(ModelBasedCMDPAgent, self).__init__(
            memory=memory,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            gamma=gamma,
            policy_learning_algorithm=policy_learning_algorithm,
            *args,
            **kwargs,
        )

    @classmethod
    def default(
        cls,
        environment,
        policy=None,
        critic=None,
        lr=1e-3,
        memory=None,
        *args,
        **kwargs,
    ):
        if policy is None:
            policy = NNPolicy.default(environment, *args, **kwargs)
        if critic is None:
            critic = NNEnsembleQFunction.default(environment)

        return super().default(
            environment,
            memory=memory,
            policy=policy,
            critic=critic,
            optimizer=Adam(chain(policy.parameters(), critic.parameters()), lr=lr),
            *args,
            **kwargs,
        )
