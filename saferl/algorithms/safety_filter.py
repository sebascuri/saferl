"""Safety Filter Algorithm."""
import torch
from rllib.model.abstract_model import AbstractModel
from rllib.model.closed_loop_model import ClosedLoopModel
from rllib.value_function.abstract_value_function import AbstractQFunction
from rllib.value_function.integrate_q_value_function import IntegrateQValueFunction

from saferl.algorithms.safe_cem import SafeCEM


class SafetyFilterReward(AbstractModel):
    def __init__(
        self,
        reward_model,
        policy,
        safety_critic=None,
        safe_policy=None,
        xi=-0.001,
        gamma=0.99,
        *args,
        **kwargs
    ):
        super().__init__(
            dim_state=reward_model.dim_state,
            dim_action=reward_model.dim_action,
            dim_reward=(reward_model.dim_reward[0] + 1,),
            num_states=reward_model.num_states,
            num_actions=reward_model.num_actions,
            model_kind=reward_model.model_kind,
            *args,
            **kwargs,
        )
        self.base_model = reward_model
        self.policy = policy
        if isinstance(safety_critic, AbstractQFunction):
            safety_critic = IntegrateQValueFunction(
                q_function=safety_critic, policy=safe_policy
            )
        self.safety_critic = safety_critic
        self.safe_policy = safe_policy
        self.xi = xi
        self.gamma = gamma
        assert self.model_kind == "rewards", "only reward model is implemented."

    def forward(self, state, action, next_state=None):
        self.policy(state)
        reward = -torch.norm(action - self.policy(state)[0], dim=-1).unsqueeze(-1)
        cost = self.base_model(state, action)[0][..., 1:]

        if self.safety_critic is not None:
            # add the terminal rewards.
            v = self.safety_critic(next_state)
            c = self.base_model(next_state, self.safe_policy(next_state)[0])[0]
            to_go_cost = (v - c - self.xi * self.gamma)[..., 1:].sum(-1).unsqueeze(-1)
        else:
            to_go_cost = torch.zeros_like(reward)
        reward = torch.cat([reward, cost, to_go_cost], dim=-1)

        return reward, torch.zeros_like(reward)


class SafetyFilterMPC(SafeCEM):
    def __init__(
        self,
        dynamical_model,
        reward_model,
        policy,
        safety_critic=None,
        safe_policy=None,
        xi=-0.001,
        gamma=0.99,
        *args,
        **kwargs
    ):
        dynamical_model = ClosedLoopModel(dynamical_model, policy)
        reward_model = SafetyFilterReward(
            reward_model=reward_model,
            policy=policy,
            safety_critic=safety_critic,
            safe_policy=safe_policy,
            xi=xi,
            gamma=gamma,
        )
        super(SafetyFilterMPC, self).__init__(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            gamma=gamma,
            *args,
            **kwargs,
        )
