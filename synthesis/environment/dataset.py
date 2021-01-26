from abc import ABC, abstractmethod

from more_itertools import chunked

import numpy as np

from ..utils.utils import shuffle_chunks


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

    def shuffle_chunks(self, chunk_size):
        return ShuffledChunksDataset(self, chunk_size)

    def limit(self, limit):
        return LimitedDataset(self, limit)


class ShuffledChunksDataset(Dataset):
    def __init__(self, dataset, chunk_size):
        super().__init__(dataset.segment)
        self.underlying = dataset
        self.chunk_size = chunk_size

    def dataset(self, seed):
        rng = np.random.RandomState(seed)
        yield from shuffle_chunks(
            self.underlying.dataset(rng.randint(2 ** 32)), self.chunk_size, rng
        )


class LimitedDataset(Dataset):
    def __init__(self, dataset, limit):
        super().__init__(dataset.segment)
        self.underlying = dataset
        self.limit = limit

    def dataset(self, seed):
        yield from itertools.islice(self.underlying.dataset(seed), self.limit)
