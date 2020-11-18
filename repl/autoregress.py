import functools

import attr

import torch

from repl.distribution import Distribution


def fcnet(*layer_sizes):
    modules = []
    for input_size, output_size in zip(layer_sizes, layer_sizes[1:]):
        modules.append(torch.nn.Linear(input_size, output_size))
        modules.append(torch.nn.ReLU())
    return torch.nn.Sequential(*modules)


class AutoRegressor(torch.nn.Module):
    def __init__(self, variables, *, hidden_size, n_layers):
        super().__init__()
        self.variables = variables
        hidden_sizes = [hidden_size] * (n_layers - 1)
        input_size = hidden_size
        predictor = {}
        for var_name, var_size in variables:
            predictor[var_name] = fcnet(input_size, *hidden_sizes, var_size)
            input_size += var_size
        self.predictor = torch.nn.ModuleDict(predictor)

    def predict(self, variable, variable_state, hidden_state):
        input_so_far = [hidden_state]
        for var_name, var_size in self.variables:
            if var_name == variable:
                break
            assert variable_state[var_name].shape[-1] == var_size
            input_so_far.append(variable_state[var_name])
        input_so_far = torch.cat(input_so_far, axis=-1)
        return self.predictor[variable](input_so_far)

    def rollout(self, hidden_state, extract_variable):
        variable_state = {}
        for var_name, _ in self.variables:
            out = self.predict(var_name, variable_state, hidden_state)
            variable_state[var_name] = extract_variable(var_name, out, variable_state)
        return variable_state


@attr.s
class AutoRegressDistribution(Distribution):
    auto_regressor = attr.ib()
    hidden_state = attr.ib()
    type = attr.ib()

    @property
    def _count(self):
        return self.hidden_state.shape[0]

    def _initialize(self, params_each):
        return [
            self.type(**{k: v[i].item() for k, v in params_each.items()})
            for i in range(self._count)
        ]

    def rollout(self, select_variable):
        outputs = {}

        def extract_variable(name, var_content, _previous):
            outputs[name] = select_variable(name, var_content)
            onehot = torch.zeros_like(var_content)
            onehot[:, outputs[name]] = 1
            return onehot

        self.auto_regressor.rollout(self.hidden_state, extract_variable)
        return self._initialize(outputs)

    def mle(self):
        def select_variable(_name, var_content):
            # This could be a beam search, but we'll just use greedy decoding
            return var_content.max(-1)[1]

        return self.rollout(select_variable)

    def log_probability(self, outcomes):
        logs = []

        def select_variable(name, var_content):
            attributes = [getattr(outcome, name) for outcome in outcomes]
            logs.append(var_content.log_softmax(-1)[range(len(attributes)), attributes])
            return torch.tensor(attributes)

        self.rollout(select_variable)

        return torch.stack(logs).sum(0)

    def sample(self, rng):
        def select_variable(_name, var_content):
            old_seed = torch.random.get_rng_state()
            torch.random.manual_seed(rng.randint(2 ** 32))
            selected = torch.distributions.Categorical(logits=var_content).sample()
            torch.random.set_rng_state(old_seed)

            return selected

        return self.rollout(select_variable)
