"""Agent Builder base class."""
import importlib

from examples.utilities import Experiment
from rllib.agent import AbstractAgent
from rllib.agent.fixed_policy_agent import FixedPolicyAgent
from rllib.util.multi_objective_reduction import GetIndexMultiObjectiveReduction
from hucrl.model.hallucinated_model import HallucinatedModel
from rllib.model.transformed_model import TransformedModel
from rllib.policy import AbstractPolicy, NNPolicy
from rllib.policy.zero_policy import ZeroPolicy
from hucrl.policy.augmented_policy import AugmentedPolicy
from rllib.util.losses.pathwise_loss import PathwiseLoss
from saferl.utilities.losses import CMDPPathwiseLoss, MaximallySafeLoss
from rllib.util.utilities import RewardTransformer

from saferl.agent.safety_filter_agent import SafetyFilterAgent
from saferl.agent.safe_mpc_agent import SafeMPCAgent
from saferl.utilities.multi_objective_reduction import (
    NegCostReduction,
    LagrangianReduction,
)


class AgentBuilder(object):
    """Default agent builder class."""

    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        if not (
            "model_based" in self.experiment.agent or "mpc" in self.experiment.agent
        ):
            self.experiment.hallucinate = False

    def create_agent(self, environment, *args, **kwargs) -> AbstractAgent:
        """Create the agent."""
        if self.experiment.agent == "known_safe":
            return self._get_known_safe(environment, *args, **kwargs)
        elif self.experiment.agent == "known_unsafe":
            return self._get_known_unsafe(environment, *args, **kwargs)
        elif self.experiment.agent == "open_loop":
            return self._get_open_loop(environment, *args, **kwargs)
        elif self.experiment.agent == "random":
            return self._get_random(environment, *args, **kwargs)

        agent_class = self._get_base_agent_class()
        reduction, loss = self._get_reduction_and_loss(environment)

        return agent_class.default(
            environment,
            dynamical_model=self._create_dynamical_model(environment, *args, **kwargs),
            policy=self._create_policy(environment, *args, **kwargs),
            multi_objective_reduction=reduction,
            pathwise_loss_class=loss,
            name=self.experiment.agent,
            # num_model_steps=10,
            reward_transformer=self._get_reward_transformer(),
            *args,
            **kwargs,
        )

    def _get_reward_transformer(self):
        if "MPO" in self.experiment.agent:
            return RewardTransformer(offset=-300, scale=1, low=0, high=1000)
        return RewardTransformer()

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

    def _get_base_agent_class(self):
        """Create agent."""
        if self.experiment.agent == "SafeMPC":
            return SafeMPCAgent
        elif self.experiment.agent == "SafetyFilter":
            return SafetyFilterAgent

        agent_module = importlib.import_module("rllib.agent")

        agent_name = self.experiment.agent.split("_")[-1]
        return getattr(agent_module, f"{agent_name}Agent")

    def _get_reduction_and_loss(self, environment):
        if "cmdp" in self.experiment.agent.lower():
            reduction = LagrangianReduction(components=environment.dim_reward[0] - 1)
            loss = CMDPPathwiseLoss
        elif "maximally_safe" in self.experiment.agent.lower():
            reduction = NegCostReduction()
            loss = MaximallySafeLoss
        else:
            reduction = GetIndexMultiObjectiveReduction()
            loss = PathwiseLoss
        return reduction, loss

    def _get_known_safe(self, environment, *args, **kwargs):
        raise NotImplementedError

    def _get_known_unsafe(self, environment, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_random(environment, *args, **kwargs):
        return FixedPolicyAgent.default(environment, name="RandomAgent")

    @staticmethod
    def _get_open_loop(environment, *args, **kwargs):
        policy = ZeroPolicy.default(environment)
        return FixedPolicyAgent.default(
            environment, policy=policy, name="OpenLoopAgent"
        )
