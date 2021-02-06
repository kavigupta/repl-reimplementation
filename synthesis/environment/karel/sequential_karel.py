import os
import shelve

import numpy as np

from karel_for_synthesis import execute, unparse, ExecutorRuntimeException

from ..dataset import Dataset
from ..spec import Pair, Specification
from ...repl.program import SequentialProgram
from .standard_karel import KarelDataset

TOKENS = ["move", "turnRight", "turnLeft", "pickMarker", "putMarker"]


def random_input(data, rng):
    spec, _ = data[rng.choice(len(data))]
    spec = spec.pairs + spec.test_pairs
    pair = spec[rng.choice(len(spec))]
    return pair.input


def random_compatible_input(data, program, rng, max_steps=100):
    for _ in range(max_steps):
        inp = random_input(data, rng)
        try:
            return Pair(inp, execute(program, inp))
        except ExecutorRuntimeException:
            pass


def toks_to_program(toks):
    return unparse(dict(type="run", body=[dict(type=tok) for tok in toks]))


def random_sequential_program(size, rng):
    toks = rng.choice(TOKENS, size=size, replace=True)
    return SequentialProgram(tuple(toks)), toks_to_program(toks)


def _randomly_sample_spec_once(data, rng, *, size, train=5, test=1):
    toks, prog = random_sequential_program(size, rng)

    pairs = []
    for _ in range(train + test):
        pair = random_compatible_input(data, prog, rng)
        if pair is None:
            return
        pairs.append(pair)
    return toks, Specification(pairs[:train], pairs[train:])


def randomly_sample_spec(*args, **kwargs):
    while True:
        res = _randomly_sample_spec_once(*args, **kwargs)
        if res is not None:
            return res


class KarelSequentialDataset(Dataset):
    def __init__(self, segment, size, path="data/karel_standard", limit=float("inf")):
        super().__init__(segment)
        self.size = size
        self.underlying = KarelDataset(segment, path=path).datafile
        prefix = os.path.join(path, "karel_sequential")
        try:
            os.makedirs(prefix)
        except FileExistsError:
            pass
        self.shelf = shelve.open(os.path.join(prefix, f"{segment}_{size}"))
        self.limit = min(len(self.underlying), limit)

    def dataset(self, seed, pbar=lambda x: x):
        shuffled_idxs = list(range(self.limit))
        np.random.RandomState(seed).shuffle(shuffled_idxs)
        # just a standardized pseudorandom scheme
        seeds = np.random.RandomState(0).randint(2 ** 32, size=self.limit)
        for index, round_seed in pbar(list(zip(shuffled_idxs, seeds))):
            index = str(index)
            if index not in self.shelf:
                self.shelf[index] = randomly_sample_spec(
                    self.underlying, np.random.RandomState(round_seed), size=16
                )
            p, spec = self.shelf[index]
            yield spec, p
