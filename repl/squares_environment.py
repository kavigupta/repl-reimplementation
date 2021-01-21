import attr

import numpy as np
import torch

from .utils.autoregress import AutoRegressor, AutoRegressDistribution
from .program import SequentialProgram
from .policy import Policy
from .specification import Spec
from .utils.distribution import IndependentDistribution


@attr.s(hash=True)
class Square:
    w = attr.ib()
    h = attr.ib()
    x = attr.ib()
    y = attr.ib()
    color = attr.ib()

    @staticmethod
    def sample(config, rng):
        x, y = rng.choice(config.size, size=2, replace=True)
        w, h = rng.choice(config.size // 2, size=2, replace=True) + 1
        c = rng.choice(config.num_colors)
        return Square(w, h, x, y, c)

    def paint(self, canvas):
        canvas[self.x : self.x + self.w, self.y : self.y + self.h] = 0
        canvas[self.x : self.x + self.w, self.y : self.y + self.h, self.color] = 1


@attr.s
class SquaresConfig:
    num_colors = attr.ib(default=3)
    max_num_squares = attr.ib(default=10)
    size = attr.ib(default=100)


@attr.s
class SquaresSpec(Spec):
    config = attr.ib()
    gold_program = attr.ib()

    program_class = SequentialProgram

    def partially_execute(self, program):
        canvas = np.zeros((self.config.size, self.config.size, self.config.num_colors))
        for square in program.tokens:
            square.paint(canvas)
        return canvas

    @property
    def output_pattern(self):
        return self.partially_execute(self.gold_program)

    def program_is_complete(self, program):
        return len(program.tokens) >= self.config.max_num_squares

    def program_is_correct(self, program):
        return np.all(self.partially_execute(program) == self.output_pattern)


def deduplicate_gold_program(config, gold, rng):
    ex = lambda g: SquaresSpec(config, SequentialProgram(g)).output_pattern

    gold = gold[:]
    safe = [False for _ in gold]
    while not all(safe):
        idx = rng.choice([i for i, s in enumerate(safe) if not s])
        if np.all(ex(gold) == ex(gold[:idx] + gold[idx + 1 :])):
            gold.pop(idx)
            safe.pop(idx)
        else:
            safe[idx] = True
    return gold


def sample(config, rng):
    gold = SequentialProgram(
        deduplicate_gold_program(
            config,
            [
                Square.sample(config, rng)
                for _ in range(rng.choice(config.max_num_squares) + 1)
            ],
            rng,
        )
    )
    spec = SquaresSpec(config, gold)
    return spec, gold


class StatePacker(torch.nn.Module):
    def __init__(self, config, channels, layers):
        super().__init__()
        self.initial_embed = torch.nn.Conv2d(config.num_colors * 2, channels, 1)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Conv2d(channels, channels, 3, padding=1) for _ in range(layers)]
        )
        self.attention = torch.nn.ReLU()

    def forward(self, states):
        def get_spp(state):
            [spp] = states[0].semantic_partial_programs
            return spp

        spps = np.array([get_spp(state) for state in states])
        goals = np.array([state.spec.output_pattern for state in states])
        embedded_state = np.concatenate([spps, goals], axis=3)
        embedded_state = np.ascontiguousarray(
            embedded_state.transpose((0, 3, 1, 2)).astype(np.float32)
        )
        embedded_state = torch.tensor(embedded_state).to(
            next(iter(self.parameters())).device
        )
        x = self.initial_embed(embedded_state)
        for layer in self.layers:
            x = layer(x)
            x = self.attention(x)
        return x


class SquaresPolicy(torch.nn.Module, Policy):
    def __init__(self, config, *, channels=100, last_layer_channels=10, batch_size=32):
        super().__init__()
        self.packer = StatePacker(config, channels, layers=5)
        self.pooler = torch.nn.Conv2d(channels, last_layer_channels, 1)

        self.autoregressor = AutoRegressor(
            list(
                dict(
                    w=config.size,
                    h=config.size,
                    x=config.size,
                    y=config.size,
                    color=config.num_colors,
                ).items()
            ),
            hidden_size=config.size ** 2 * last_layer_channels,
            n_layers=2,
        )
        self._batch_size = batch_size

    def forward(self, states):
        packed = self.packer(states)
        packed = self.pooler(packed)
        packed = packed.reshape(packed.shape[0], -1)
        return AutoRegressDistribution(self.autoregressor, packed, Square)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def initial_program_set(self):
        return [SequentialProgram([])]


class SquaresValue(torch.nn.Module):
    def __init__(self, config, channels=100, batch_size=32):
        super().__init__()
        self.packer = StatePacker(config, channels, 5)
        self.linear = torch.nn.Linear(channels, 1)

    def forward(self, states):
        x = self.packer(states)
        x = x.max(-1)[0].max(-1)[0]
        x = self.linear(x)
        return x.sigmoid().squeeze(-1)
