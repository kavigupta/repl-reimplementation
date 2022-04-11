from abc import ABC, abstractmethod
import attr
import numpy as np
from permacache import permacache, stable_hash

from ..language.sampler import sample_valid_datapoint


class ExperimentalSetting(ABC):
    def __init__(self, *, max_type_size, num_elements_dist, minimal_objects):
        self.max_type_size = max_type_size
        self.num_elements_dist = num_elements_dist
        self.minimal_objects = minimal_objects

    @abstractmethod
    def variable_alphabet(self, context):
        pass

    @abstractmethod
    def grammar(self, context):
        pass

    @abstractmethod
    def value_grammar(self):
        pass

    @abstractmethod
    def type_distribution(self):
        pass

    @abstractmethod
    def sampler_spec(self):
        pass

    def sample(self, context, seed):
        return sample_experimental_setting(self, context, seed)

    def __permacache_hash__(self):
        return dict(type="ExperimentalSetting", content=self.__dict__)


@permacache(
    "sketch_hypergraph/experiments/experimental_setting/sample_experimental_setting",
    key_function=dict(es=stable_hash, context=stable_hash),
)
def sample_experimental_setting(es, context, seed):
    return sample_valid_datapoint(
        np.random.RandomState(seed),
        sampler_spec=es.sampler_spec(),
        grammar=es.grammar(context),
        g_value=es.value_grammar(),
        t_value=es.type_distribution(),
        e_context=context,
        max_type_size=es.max_type_size,
        num_elements_dist=es.num_elements_dist,
        minimal_objects=es.minimal_objects,
    )
