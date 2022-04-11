from collections import Counter
from itertools import count

import numpy as np

from .canvas import ValidatingCanvas
from .value import DrawnObjectInvalidError


from .ast_constructor import ASTConstructionState
from .driver import DriverError, Driver
from .types import BaseType, TypeEnv, WithinContext
from .evaluation import Environment


class SamplingDriver(Driver):
    def __init__(self, rng, weights):
        self.rng = rng
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


def sample_type_environment(rng, types, max_size, variable_alphabet):
    size = rng.randint(1, 1 + max_size)
    weights = np.array([w for _, w in types])
    weights = weights / weights.sum()
    idx = rng.choice(weights.size, p=weights, size=size, replace=True)
    variables = rng.choice(sorted(variable_alphabet), size=size, replace=False)
    return TypeEnv(dict(zip(variables, [types[i][0] for i in idx])))


def weighted_choice(d):
    keys = sorted(d)
    weights = [d[k] for k in keys]
    weights = np.array(weights, dtype=np.float)
    weights = weights / weights.sum()
    idx = np.random.choice(weights.size, p=weights)
    return keys[idx]


def sample_datapoint(
    rng,
    *,
    sampler_spec,
    grammar,
    g_value,
    t_value,
    max_type_size,
    e_context,
    num_elements_dist
):
    num_elements = weighted_choice(num_elements_dist)
    type_environment = sample_type_environment(
        rng,
        t_value,
        max_size=max_type_size,
        variable_alphabet=grammar.possible_variables,
    )
    program = sample(dict(**sampler_spec, rng=rng), grammar, type_environment)
    environments = [
        sample(
            dict(**sampler_spec, rng=rng),
            g_value,
            TypeEnv({}),
            to_sample=type_environment,
        )
        for _ in range(num_elements)
    ]
    outputs = [
        Environment(e_context, e).evaluate(program).drawn_objects for e in environments
    ]

    return program, environments, outputs, type_environment


def sample_valid_datapoint(rng, *, minimal_objects, **kwargs):
    for sampled in count(1):
        p, i, o, te = sample_datapoint(rng, **kwargs)
        if validate_outputs(o, minimal_objects=minimal_objects):
            return dict(sampled=sampled, p=p, i=i, o=o, type_env=te)


def validate_outputs(outputs, **kwargs):
    return all(validate_output(output, **kwargs) for output in outputs)


def validate_output(output, *, minimal_objects):
    c = ValidatingCanvas()
    for x in output:
        try:
            x.draw(c)
        except DrawnObjectInvalidError as e:
            return False
    if len(c.lines + c.circles) < minimal_objects:
        return False
    return True
