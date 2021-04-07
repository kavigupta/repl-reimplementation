import os
import shelve

import numpy as np
import attr

from karel_for_synthesis import execute, unparse, ExecutorRuntimeException

from ..dataset import Dataset
from ..spec import Pair, Specification
from ...repl.program import SequentialProgram
from .standard_karel import KarelDataset, get_one_hot, from_one_hot

ACTIONS = ["move", "turnRight", "turnLeft", "pickMarker", "putMarker"]
TOKENS = ACTIONS
TOKEN_TO_INDEX = {tok: idx for idx, tok in enumerate(TOKENS)}


def random_input(data, rng):
    spec, _ = data[rng.choice(len(data))]
    spec = spec.pairs + spec.test_pairs
    pair = spec[rng.choice(len(spec))]
    return pair.input


def random_compatible_input(data, program, rng, max_steps=100):
    for _ in range(max_steps):
        inp = random_input(data, rng)
        try:
            return Pair(inp, execute(program, inp).result)
        except ExecutorRuntimeException:
            pass


def is_end(token):
    return token == "END"


def remove_after_end(toks):
    new_toks = []
    for tok in toks:
        if is_end(tok):
            break
        new_toks.append(tok)
    return new_toks


def toks_to_program(toks):
    return unparse(dict(type="run", body=[dict(type=tok) for tok in toks]))


@attr.s
class NoRepeatsSampler:
    size = attr.ib()

    def __call__(self, rng):
        return rng.choice(ACTIONS, size=self.size, replace=True)



def random_sequential_program(size, rng):
    toks = rng.choice(TOKENS, size=size, replace=True)
    return SequentialProgram(tuple(toks)), toks_to_program(toks)


def _randomly_sample_spec_once(data, rng, *, program_sampler, train=5, test=1):
    toks = program_sampler(rng)
    toks, prog = SequentialProgram(tuple(toks)), toks_to_program(toks)

    pairs = []
    for _ in range(train + test):
        pair = random_compatible_input(data, prog, rng)
        if pair is None:
            return
        pairs.append(pair)
    return Specification(pairs[:train], pairs[train:]), toks


def randomly_sample_spec(*args, **kwargs):
    while True:
        res = _randomly_sample_spec_once(*args, **kwargs)
        if res is not None:
            return res


class KarelSequentialDataset(Dataset):
    def __init__(
        self, segment, sampler, path="data/karel_standard", limit=float("inf")
    ):
        super().__init__(segment)
        self.sampler = sampler
        self.underlying = KarelDataset(segment, path=path).datafile
        prefix = os.path.join(path, "karel_sequential")
        try:
            os.makedirs(prefix)
        except FileExistsError:
            pass
        self.shelf = shelve.open(os.path.join(prefix, f"{segment}_{sampler}"))
        self._limit = min(len(self.underlying), limit)

    def dataset(self, seed, pbar=lambda x: x):
        shuffled_idxs = list(range(self._limit))
        np.random.RandomState(seed).shuffle(shuffled_idxs)
        # just a standardized pseudorandom scheme
        seeds = np.random.RandomState(0).randint(2 ** 32, size=self._limit)
        for index, round_seed in pbar(list(zip(shuffled_idxs, seeds))):
            index = str(index)
            if index not in self.shelf:
                self.shelf[index] = self._pack(
                    *randomly_sample_spec(
                        self.underlying,
                        np.random.RandomState(round_seed),
                        sampler=self.sampler,
                    )
                )
            yield self._unpack(*self.shelf[index])

    def _pack(self, spec, p):
        packed_spec = spec.map_pairs(
            lambda x: Pair(from_one_hot(x.input), from_one_hot(x.output))
        )
        return packed_spec, p

    def _unpack(self, spec, p):
        unpacked_spec = spec.map_pairs(
            lambda x: Pair(get_one_hot(x.input), get_one_hot(x.output))
        )
        return unpacked_spec, p
