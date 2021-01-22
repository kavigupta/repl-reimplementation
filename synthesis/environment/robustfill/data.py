import shelve
import numpy as np

from ..dataset import Dataset
from ..spec import Pair, Specification
from ...utils.utils import split_indices
from ...repl.program import SequentialProgram


class RobustfillDataset(Dataset):
    def __init__(self, segment, split=0.1, path="data/robustfill/standard.s"):
        super().__init__(segment)
        self.split = split
        self.data = shelve.open(path, "c")

    def dataset(self, seed):
        indices = split_indices(self.segment, self.split, len(self.data))

        np.random.RandomState(seed).shuffle(indices)

        for i in indices:
            program, inputs, outputs = self.data[str(i)]
            spec = [Pair(inp, out) for inp, out in zip(inputs, outputs)]
            yield Specification(spec[:-1], spec[-1:]), SequentialProgram(program)
