from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.dataset.datatypes import Loss


class AlgorithmList(AbstractAlgorithm):
    def __init__(self, algorithms, *args, **kwargs):
        super().__init__(
            gamma=algorithms[0].gamma,
            critic=algorithms[0].critic,
            policy=algorithms[0].policy,
            *args,
            **kwargs
        )
        self.algorithms = algorithms

    def forward(self, observation):
        """Rollout model and call base algorithm with transitions."""
        loss = Loss()
        for algorithm in self.algorithms:
            loss += algorithm(observation)
        return loss

    def update(self):
        """Update base algorithm."""
        for algorithm in self.algorithms:
            algorithm.update()

    def reset(self):
        """Reset base algorithm."""
        for algorithm in self.algorithms:
            algorithm.reset()

    def info(self):
        """Get info from base algorithm."""
        info = {**self._info}
        for algorithm in self.algorithms:
            info.update(**algorithm.info())
        return info

    def reset_info(self):
        """Reset info from base algorithm."""
        for algorithm in self.algorithms:
            algorithm.reset_info()

    def set_policy(self, new_policy):
        """Set policy in base algorithm."""
        for algorithm in self.algorithms:
            algorithm.set_policy(new_policy)
