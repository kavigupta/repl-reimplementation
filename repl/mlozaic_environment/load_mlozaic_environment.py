import os
import shelve

from more_itertools import chunked

import numpy as np

from mlozaic.grammar import BACKWARDS_ALPHABET
from mlozaic.renderer import render

from ..spec import Specification, Pair
from ..dataset import Dataset
from ..utils import split_indices


class MlozaicDataset(Dataset):
    def __init__(self, segment, split=0.1, path="data/mlozaic_standard"):
        super().__init__(segment)
        self.split = split
        if not os.path.exists(path):
            raise RuntimeError(
                f"Run 'standard_dataset {path} 200000' to generate the data"
            )
        self.data = shelve.open(os.path.join(path, "data"), "c")

    def dataset(self, seed):
        """
        Iterate through the dataset yielding spec, program pairs. The iteration
            order is determined by the seed, and the train/test split is determined
            deterministically by seed 0.

        Arguments:
            segment: the segment to use, must be 'train' or 'test'
            split: the percentage of the data to place in the test pool
        """
        indices = split_indices(self.segment, self.split, len(self.data))
        np.random.RandomState(seed).shuffle(indices)
        for i in indices:
            program, inputs = self.data[str(i)]
            inputs = inputs[:-1]
            spec = [
                Pair(
                    inp,
                    render(program.evaluate(inp), size=(50, 50), stretch=2, rgb=False),
                )
                for inp in inputs
            ]
            program = [BACKWARDS_ALPHABET[tok] + 2 for tok in program.code]
            yield Specification(spec), program
