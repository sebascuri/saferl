from dataclasses import asdict

import numpy as np
import torch
from rllib.dataset.datatypes import Observation
from rllib.dataset.experience_replay import ExperienceReplay


class BatchExperienceReplay(ExperienceReplay):
    """Experience Replay that accepts a batch of elements."""

    def __init__(self, max_len):
        super().__init__(max_len)
        self.batch_memory = dict()

    def __getitem__(self, idx):
        """Return any desired observation.

        Parameters
        ----------
        idx: int

        Returns
        -------
        observation: Observation
        idx: int
        weight: torch.tensor.

        """
        if self.valid[idx] == 0:  # when a non-valid index is sampled.
            idx = np.random.choice(self.valid_indexes).item()

        return self._get_dict_observation(self, idx), idx, self.weights[idx]

    def _init_observation(self, observation):
        super()._init_observation(observation)
        for key, value in asdict(observation).items():
            if torch.isnan(value).any():
                continue
            if observation.reward.ndim == 1:
                value = value.unsqueeze(0)

            self.batch_memory[key] = value.clone().unsqueeze(1)

    def append(self, observation):
        """Append a new observation."""
        if observation.reward.ndim > 1:
            new_size = observation.state.shape[0]
        else:
            new_size = 1

        if self.zero_observation is None:
            self._init_observation(observation)
            self.valid[self.ptr : self.ptr + new_size] = 1
            self.data_count += new_size
            return

        for key, value in asdict(observation).items():
            if torch.isnan(value).any():
                continue
            if observation.reward.ndim == 1:
                value = value.unsqueeze(0)  # add batch coordinate.
            value = value.unsqueeze(1).clone()  # add time coordinate.

            if self.is_full:
                self.batch_memory[key][self.ptr : self.ptr + new_size] = value
            else:
                self.batch_memory[key] = torch.cat(
                    (self.batch_memory[key], value),
                    dim=0,
                )
        self.valid[self.ptr : self.ptr + new_size] = 1
        self.data_count += new_size

    def _get_dict_observation(self, idx):
        return {key: value[idx] for key, value in self.batch_memory.items()}

    def reset(self):
        """Reset memory to empty."""
        super().reset()
        self.batch_memory = dict()

    def sample_batch(self, batch_size):
        """Sample a batch of observations."""
        indices = np.random.choice(self.valid_indexes, batch_size)
        obs = self._get_dict_observation(indices)
        return Observation(**obs), torch.tensor(indices), self.weights[indices]
