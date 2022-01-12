import numpy as np


class SamplerError(Exception):
    pass


class Sampler:
    def __init__(self, seed, alphabet, distributions, do_not_assign_prob, retries):
        self.rng = np.random.RandomState(seed)
        self.alphabet = alphabet
        self.distributions = distributions
        self.do_not_assign_prob = do_not_assign_prob
        self.retries = retries

    def _sample(self, items):
        xs, ps = zip(*items)
        ps = np.array(ps, dtype=np.float32)
        ps /= ps.sum()
        return xs[self.rng.choice(ps.size, p=ps)]

    def sample_length(self):
        return self._sample(self.distributions.length)

    def sample_statement_type(self):
        return self._sample(self.distributions.statement_type)

    def sample_numeric_type(self):
        return self._sample(self.distributions.numeric_type)

    def sample_condition_type(self):
        return self._sample(self.distributions.condition_type)

    def sample_object_expression_type(self, typenv):
        type_dist = [
            (x, p)
            for x, p in self.distributions.expression_type
            if x.is_samplable(typenv)
        ]
        return self._sample(type_dist)

    def sample_fresh_variable(self, typenv, *, allow_none):
        if allow_none and self.rng.rand() < self.do_not_assign_prob:
            return None
        variables = sorted(
            set(self.alphabet.variables) - set(typenv.assigned_variables())
        )
        if len(variables) == 0:
            raise SamplerError("Cannot sample a fresh variable when none are left")
        return self.rng.choice(variables)

    def sample_variable(self, typenv, **kwargs):
        [x] = self.sample_distinct_variables(typenv, 1, **kwargs)

    def sample_distinct_variables(self, typenv, num, **kwargs):
        variables = sorted(typenv.assigned_variables(**kwargs))
        if num > len(variables):
            raise SamplerError(
                f"Cannot sample {num} variables from a population of {len(variables)}"
            )
        return self.rng.choice(variables, replace=False, size=num)

    def sample_constant_value(self):
        return self.rng.choice(self.alphabet.numbers)

    def condition_retries(self):
        return self.retries

    def expression_retries(self):
        return self.retries

    def sample_operation(self, elements):
        return self._sample(
            [(x, self.distributions.operation.get(x, 1)) for x in elements]
        )
