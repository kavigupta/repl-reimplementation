import attr

import numpy as np
import torch

from .program import SequentialProgram
from .policy import Policy
from .specification import Spec


@attr.s
class Block:
    color = attr.ib()
    length = attr.ib()

    @staticmethod
    def sample(config, rng):
        return Block(
            rng.choice(config.num_colors), rng.choice(config.max_block_length) + 1
        )


@attr.s
class BlockConfig:
    num_colors = attr.ib(default=3)
    overall_length = attr.ib(default=30)
    max_block_length = attr.ib(default=5)

    def pack_colors(self, color_indices, multipliers=None):
        result = np.zeros((self.overall_length, self.num_colors))
        if multipliers is None:
            multipliers = np.ones(len(color_indices))
        else:
            multipliers = np.array(multipliers)
        multipliers = multipliers[:, None]
        r = np.eye(self.num_colors)[color_indices] * multipliers
        result[: len(color_indices)] = r
        return result

    def pack_blocks(self, blocks):
        return self.pack_colors([b.color for b in blocks], [b.length for b in blocks])


@attr.s
class BlocksSpec(Spec):
    blocks = attr.ib()
    gold_program = attr.ib()

    program_class = SequentialProgram

    def partially_execute(self, program):
        result = []
        for i in program.tokens:
            if i < len(self.blocks):
                block = self.blocks[i]
                result += [block.color] * block.length
        return result[: sum(b.length for b in self.blocks)]

    @property
    def output_pattern(self):
        return self.partially_execute(self.gold_program)

    def program_is_complete(self, program):
        return len(program.tokens) >= len(self.blocks)

    def program_is_correct(self, program):
        return self.partially_execute(program) == self.output_pattern


def sample(config, rng):
    blocks = []
    total_length = 0
    while True:
        block = Block.sample(config, rng)
        if total_length + block.length >= config.overall_length:
            break
        total_length += block.length
        blocks.append(block)
    rng.shuffle(blocks)
    program = list(range(len(blocks)))
    rng.shuffle(program)
    gold = SequentialProgram(program)
    spec = BlocksSpec(blocks, gold)
    return spec, gold


class BlocksPolicy(torch.nn.Module, Policy):
    def __init__(self, config, batch_size=32, hidden_size=1000):
        torch.nn.Module.__init__(self)
        self._batch_size = batch_size
        self.config = config
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3 * config.overall_length * config.num_colors, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, config.overall_length),
            torch.nn.LogSoftmax(),
        )

    @property
    def batch_size(self):
        return self._batch_size

    def pack(self, states):
        epps = []
        spec_targets = []
        spec_blocks = []
        for state in states:
            [epp] = state.semantic_partial_programs
            epps.append(self.config.pack_colors(epp))
            spec_targets.append(self.config.pack_colors(state.spec.output_pattern))
            spec_blocks.append(self.config.pack_blocks(state.spec.blocks))

        epps = torch.tensor(epps)
        spec_targets = torch.tensor(spec_targets)
        spec_blocks = torch.tensor(spec_blocks)

        packed = torch.cat([epps, spec_targets, spec_blocks], axis=1)
        return packed.reshape(packed.shape[0], -1).type(torch.float32)

    def forward(self, states):
        packed = self.pack(states)
        result = self.net(packed)
        return result
