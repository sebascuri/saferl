from rllib.util.neural_networks.utilities import DisableGradient
from rllib.value_function import NNEnsembleQFunction


def get_q_value_pathwise_gradients(critic, state, action, multi_objective_reduction):
    with DisableGradient(critic):
        q = critic(state, action)
        if isinstance(critic, NNEnsembleQFunction):
            q = q[..., 0]

    # Take multi-objective reduction.
    q = multi_objective_reduction(q)
    # Take mean over time coordinate.
    if q.dim() > 1:
        q = q.mean(dim=1)
    return q
