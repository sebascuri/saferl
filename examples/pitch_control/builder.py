"""File that builds the agents."""
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
import torch
from rllib.agent import AbstractAgent, BPTTAgent, MPCAgent, SACAgent
from rllib.agent.fixed_policy_agent import FixedPolicyAgent
from rllib.policy.proportional_policy import ProportionalPolicy
from rllib.util.multi_objective_reduction import GetIndexMultiObjectiveReduction
from hucrl.model.hallucinated_model import HallucinatedModel
from rllib.model.transformed_model import TransformedModel
from rllib.policy import AbstractPolicy, NNPolicy
from hucrl.policy.augmented_policy import AugmentedPolicy
from saferl.algorithms.safe_rl_wrapper import CMDPPathwiseLoss
from saferl.agent.safety_filter_agent import SafetyFilterAgent
from saferl.agent.safe_mpc_agent import SafeMPCAgent
from saferl.utilities.multi_objective_reduction import (
    NegCostReduction,
    LagrangianReduction,
)


@dataclass
class Experiment:
    agent: str = field(
        metadata=dict(
            choiches=[
                "cmdp",
                "model_based_cmdp",
                "maximally_safe",
                "model_based_maximally_safe",
                "sac",
                "bptt",
                "mpc",
                "safe_mpc",
                "safety_filter",
                "known_safe",
                "known_unsafe",
                "random",
            ]
        ),
        default="cmdp",
    )
    hallucinate: bool = field(metadata=dict(), default=False)
    num_episodes: int = field(metadata=dict(), default=100)


parser = ArgumentParser(Experiment)


class AgentBuilder:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        if not (
            "model_based" in self.experiment.agent or "mpc" in self.experiment.agent
        ):
            self.experiment.hallucinate = False

    def create_agent(self, environment, *args, **kwargs) -> AbstractAgent:
        if self.experiment.agent == "cmdp":
            return self._get_cmdp(environment, *args, **kwargs)
        elif self.experiment.agent == "model_based_cmdp":
            return self._get_model_based_cmdp(environment, *args, **kwargs)
        elif self.experiment.agent == "maximally_safe":
            return self._get_maximally_safe(environment, *args, **kwargs)
        elif self.experiment.agent == "model_based_maximally_safe":
            return self._get_model_based_maximally_safe(environment, *args, **kwargs)
        elif self.experiment.agent == "sac":
            return self._get_sac(environment, *args, **kwargs)
        elif self.experiment.agent == "bptt":
            return self._get_bptt(environment, *args, **kwargs)
        elif self.experiment.agent == "safe_mpc":
            return self._get_safe_mpc(environment, *args, **kwargs)
        elif self.experiment.agent == "mpc":
            return self._get_mpc(environment, *args, **kwargs)
        elif self.experiment.agent == "safety_filter":
            return self._get_safety_filter(environment, *args, **kwargs)
        elif self.experiment.agent == "known_safe":
            return self._get_known_safe(environment, *args, **kwargs)
        elif self.experiment.agent == "known_unsafe":
            return self._get_known_unsafe(environment, *args, **kwargs)
        elif self.experiment.agent == "random":
            return self._get_random(environment, *args, **kwargs)

    def _create_dynamical_model(self, environment, *args, **kwargs) -> TransformedModel:
        if self.experiment.hallucinate:
            return HallucinatedModel.default(environment, *args, **kwargs)
        else:
            return TransformedModel.default(environment, *args, **kwargs)

    def _create_policy(self, environment, *args, **kwargs) -> AbstractPolicy:
        if self.experiment.hallucinate:
            return AugmentedPolicy.default(environment, *args, **kwargs)
        else:
            return NNPolicy.default(environment, *args, **kwargs)

    def _get_cmdp(self, environment, *args, **kwargs):
        return SACAgent.default(
            environment=environment,
            policy=self._create_policy(environment, *args, **kwargs),
            multi_objective_reduction=LagrangianReduction(
                components=environment.dim_reward[0] - 1
            ),
            pathwise_loss_class=CMDPPathwiseLoss,
            name="CMDPAgent",
            *args,
            **kwargs
        )

    def _get_model_based_cmdp(self, environment, *args, **kwargs):
        return BPTTAgent.default(
            environment=environment,
            policy=self._create_policy(environment, *args, **kwargs),
            dynamical_model=self._create_dynamical_model(environment, *args, **kwargs),
            multi_objective_reduction=LagrangianReduction(
                components=environment.dim_reward[0] - 1
            ),
            pathwise_loss_class=CMDPPathwiseLoss,
            num_epochs=1,
            name="ModelBasedCMDPAgent",
            *args,
            **kwargs
        )

    def _get_maximally_safe(self, environment, *args, **kwargs):
        return SACAgent.default(
            environment=environment,
            policy=self._create_policy(environment, *args, **kwargs),
            multi_objective_reduction=NegCostReduction(),
            name="MaximallySafeAgent",
            *args,
            **kwargs
        )

    def _get_model_based_maximally_safe(self, environment, *args, **kwargs):
        return BPTTAgent.default(
            environment=environment,
            dynamical_model=self._create_dynamical_model(environment, *args, **kwargs),
            policy=self._create_policy(environment, *args, **kwargs),
            multi_objective_reduction=NegCostReduction(),
            name="ModelBasedMaximallySafeAgent",
            *args,
            **kwargs
        )

    def _get_sac(self, environment, *args, **kwargs):
        return SACAgent.default(
            environment,
            policy=self._create_policy(environment, *args, **kwargs),
            multi_objective_reduction=GetIndexMultiObjectiveReduction(),
            *args,
            **kwargs
        )

    def _get_bptt(self, environment, *args, **kwargs):
        return BPTTAgent.default(
            environment,
            dynamical_model=self._create_dynamical_model(environment, *args, **kwargs),
            policy=self._create_policy(environment, *args, **kwargs),
            multi_objective_reduction=GetIndexMultiObjectiveReduction(),
            *args,
            **kwargs
        )

    def _get_mpc(self, environment, *args, **kwargs):
        return MPCAgent.default(
            environment,
            multi_objective_reduction=GetIndexMultiObjectiveReduction(),
            *args,
            **kwargs
        )

    def _get_safe_mpc(self, environment, *args, **kwargs):
        return SafeMPCAgent.default(environment, *args, **kwargs)

    def _get_safety_filter(self, environment, *args, **kwargs):
        return SafetyFilterAgent.default(
            environment,
            dynamical_model=self._create_dynamical_model(environment, *args, **kwargs),
            policy=self._create_policy(environment, *args, **kwargs),
            *args,
            **kwargs
        )

    def _get_known_safe(self, environment, *args, **kwargs):
        return FixedPolicyAgent(
            policy=ProportionalPolicy.default(
                environment, gain=-torch.tensor([[-0.5034, 52.8645, 1.4142]])
            ),
            name="KnownSafeAgent",
        )

    def _get_known_unsafe(self, environment, *args, **kwargs):
        return FixedPolicyAgent(
            policy=ProportionalPolicy.default(
                environment, gain=-10 * torch.tensor([[-0.5034, 52.8645, 1.4142]])
            ),
            name="KnownUnsafeAgent",
        )

    def _get_random(self, environment, *args, **kwargs):
        return FixedPolicyAgent.default(environment, name="RandomAgent")
