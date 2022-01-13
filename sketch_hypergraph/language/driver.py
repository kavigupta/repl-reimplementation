from abc import ABC, abstractmethod
from collections import Counter

import numpy as np


class Driver(ABC):
    @abstractmethod
    def select(self, elements):
        pass


class SamplingDriver(Driver):
    def __init__(self, seed, weights):
        self.rng = np.random.RandomState(seed)
        self.weights = weights

    def select(self, elements):
        if len(elements) == 0:
            raise SamplerError("Cannot sample from empty set")
        tags = [el.node_class() for el in elements]
        tag_counts = Counter(tags)
        if len(tag_counts) == 1:
            weights = np.ones(len(elements))
        else:
            weights = [self.weights[tag] / tag_counts[tag] for tag in tags]
            weights = np.array(weights, dtype=np.float)
        weights = weights / weights.sum()
        idx = self.rng.choice(weights.size, p=weights)
        return elements[idx]


class SamplerError(Exception):
    pass
