from abc import ABC, abstractmethod

import attr

import numpy as np

from .grammar import grammar


class Sampler(ABC):
    def __init__(self, rng):
        self.rng = rng

    @abstractmethod
    def sample(self, variables, production="D"):
        pass

    def sample_from_list(self, items):
        return items[rng.choice(len(items))]


def probabilities_for_weights(items, weights):
    probs = []
    for item in items:
        if isinstance(item, list):
            item = item[0]
        probs.append(weights.get(item, 1))
    probs = np.array(probs, dtype=np.float64)
    probs /= probs.sum()
    return probs


class PCFGSamplerConfig(Sampler):
    def __init__(self, rng, grammar=grammar, weights={}):
        super().__init__(rng)
        self.weights = weights
        self.grammar = grammar

    def sample_from_list(self, items):
        probs = probabilities_for_weights(items, self.weights)
        idx = self.rng.choice(len(probs), p=probs)
        return items[idx]

    def sample(self, variables, production="D"):
        variables = set(variables)
        if isinstance(production, type):
            tag = self.sample_from_list(production.tags())
            return production.parse(tag)

        if isinstance(production, list):
            start, *rules = production
            tag = self.sample_from_list(start.tags())
            return start.parse(tag, [self.sample(variables, rule) for rule in rules])

        assert isinstance(production, str)
        rule = self.sample_from_list(self.grammar[production])
        if isinstance(rule, list):
            start = rule[0]
        else:
            start = rule
        custom_sample = start.custom_sample(self, variables)
        if custom_sample is not None:
            return custom_sample
        return self.sample(variables, rule)
