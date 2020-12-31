from abc import ABC, abstractmethod

from more_itertools import chunked

import numpy as np


class Dataset(ABC):
    def __init__(self, segment):
        self.segment = segment

    @abstractmethod
    def dataset(self, seed):
        """
        Generates spec, program pairs.
        """
        pass

    def batched_dataset_iter(self, *, seed, batch_size):
        if self.segment != "train":
            assert seed == 0
        for batch in chunked(self.dataset(seed), batch_size):
            yield tuple(zip(*batch))

    def multiple_epochs_iter(self, *, seed, batch_size, epochs):
        for specific_seed in np.random.RandomState(seed).choice(2 ** 32, size=epochs):
            yield from self.batched_dataset_iter(
                seed=specific_seed, batch_size=batch_size
            )
