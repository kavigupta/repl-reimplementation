from abc import ABC, abstractmethod
import attr
import numpy as np
from permacache import permacache, stable_hash

from ..language.sampler import sample_valid_datapoint


class ExperimentalSetting(ABC):
    sampler_spec = attr.ib()

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

    def sample(self, context, seed, **kwargs):
        return sample_experimental_setting(self, context, seed, **kwargs)


@permacache(
    "sketch_hypergraph/experiments/experimental_setting/sample_experimental_setting",
    key_function=dict(es=stable_hash, context=stable_hash),
)
def sample_experimental_setting(
    es, context, seed, *, max_type_size, num_elements, minimal_objects
):
    return sample_valid_datapoint(
        np.random.RandomState(seed),
        sampler_spec=es.sampler_spec(),
        grammar=es.grammar(context),
        g_value=es.value_grammar(),
        t_value=es.type_distribution(),
        e_context=context,
        max_type_size=max_type_size,
        num_elements=num_elements,
        minimal_objects=minimal_objects,
    )
