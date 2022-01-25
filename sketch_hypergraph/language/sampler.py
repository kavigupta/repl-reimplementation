from collections import Counter

import numpy as np


from .ast_constructor import ASTConstructionState
from .driver import DriverError, Driver
from .types import BaseType, TypeEnv, WithinContext


class SamplingDriver(Driver):
    def __init__(self, seed, weights):
        self.rng = np.random.RandomState(seed)
        self.weights = weights

    def select(self, elements):
        if len(elements) == 0:
            raise DriverError("Cannot sample from empty set")
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


def sample(driver_config, grammar, typenv, *, to_sample=BaseType.block):
    driver_config = driver_config.copy()
    assert driver_config.pop("type") == "SamplingDriver"
    driver = SamplingDriver(**driver_config)
    return sample_with_driver(driver, grammar, typenv, to_sample=to_sample)


def sample_with_driver(driver, grammar, typenv, *, to_sample):
    if isinstance(to_sample, TypeEnv):
        return {
            k: sample_with_driver(driver, grammar, typenv, to_sample=typ)
            for k, typ in to_sample.type_map.items()
        }

    while True:
        try:
            return ASTConstructionState.run_full_with_driver(
                grammar, driver, WithinContext(to_sample, typenv)
            )
        except DriverError:
            continue
