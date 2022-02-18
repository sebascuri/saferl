from itertools import chain

from rllib.agent.model_based.model_based_agent import ModelBasedAgent
from rllib.algorithms.dpg import DPG
from rllib.dataset.experience_replay import BootstrapExperienceReplay, ExperienceReplay
from rllib.model.ensemble_model import EnsembleModel
from rllib.policy.nn_policy import NNPolicy
from rllib.value_function.nn_value_function import NNQFunction
from torch.optim import Adam

from saferl.algorithms.algorithm_list import AlgorithmList
from saferl.algorithms.maximally_safe import MaximallySafeDADPG
from saferl.algorithms.safety_filter import SafetyFilterMPC


class SafetyFilterAgent(ModelBasedAgent):
    def __init__(
        self,
        dynamical_model,
        reward_model,
        guidance_policy,
        guidance_critic,
        safe_policy,
        safety_critic,
        xi=-0.0001,
        gamma=0.9,
        memory=None,
        *args,
        **kwargs
    ):
        if memory is None:
            if isinstance(dynamical_model.base_model, EnsembleModel):
                memory = BootstrapExperienceReplay(
                    num_bootstraps=dynamical_model.base_model.num_heads,
                    max_len=100000,
                    num_steps=0,
                    bootstrap=True,
                )
            else:
                memory = ExperienceReplay(
                    max_len=100000,
                    num_steps=0,
                )

        kwargs.pop("base_algorithm", None)

        mpc_kwargs = kwargs.copy()
        mpc_kwargs.pop("num_samples", 1)
        mpc_kwargs.pop("num_iter", 1)
        mpc_kwargs.pop("horizon", 1)
        planner = SafetyFilterMPC(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            policy=guidance_policy,
            safety_critic=safety_critic,
            safe_policy=safe_policy,
            horizon=1,
            num_samples=1,
            num_iter=1,
            xi=xi,
            gamma=gamma,
            *args,
            **mpc_kwargs,
        )
        algorithm = AlgorithmList(
            [
                # SAC(
                #     memory=memory,
                #     dynamical_model=dynamical_model,
                #     reward_model=reward_model,
                #     policy=guidance_policy,
                #     critic=guidance_critic,
                #     gamma=gamma,
                #     num_memory_samples=4,
                #     *args,
                #     **kwargs,
                # ),
                # SAC(
                #     memory=memory,
                #     dynamical_model=dynamical_model,
                #     reward_model=reward_model,
                #     policy=guidance_policy,
                #     critic=safety_critic,
                #     gamma=gamma,
                #     num_memory_samples=4,
                #     *args,
                #     **kwargs,
                # ),
            ]
        )

        super(SafetyFilterAgent, self).__init__(
            memory=memory,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            planning_algorithm=planner,
            gamma=gamma,
            *args,
            **kwargs,
        )

        self.algorithm = algorithm
        self.optimizer = type(self.optimizer)(
            [
                p
                for name, p in algorithm.named_parameters()
                if (
                    "model" not in name
                    and "target" not in name
                    and "old_policy" not in name
                    and p.requires_grad
                )
            ],
            **self.optimizer.defaults,
        )

    @classmethod
    def default(
        cls,
        environment,
        guidance_policy=None,
        guidance_critic=None,
        safe_policy=None,
        safety_critic=None,
        lr=1e-3,
        cmdp_algorithm=DPG,
        memory=None,
        *args,
        **kwargs
    ):
        if guidance_policy is None:
            guidance_policy = NNPolicy.default(environment, *args, **kwargs)
        if safe_policy is None:
            safe_policy = NNPolicy.default(environment, *args, **kwargs)
        if guidance_critic is None:
            guidance_critic = NNQFunction.default(environment, *args, **kwargs)
        if safety_critic is None:
            safety_critic = NNQFunction.default(environment, *args, **kwargs)

        return super().default(
            environment,
            memory=memory,
            guidance_policy=guidance_policy,
            guidance_critic=guidance_critic,
            safe_policy=safe_policy,
            safety_critic=safety_critic,
            cmdp_algorithm=cmdp_algorithm,
            optimizer=Adam(
                chain(guidance_policy.parameters(), guidance_critic.parameters()), lr=lr
            ),
            *args,
            **kwargs,
        )
