import os
import shelve

from more_itertools import chunked

import numpy as np

from mlozaic.grammar import BACKWARDS_ALPHABET
from mlozaic.renderer import render

from .spec import MLozaicSpecification, MLozaicPair
from ..utils import split_indices


def standard_dataset(path="data/mlozaic_standard"):
    if not os.path.exists(path):
        raise RuntimeError(f"Run 'standard_dataset {path} 200000' to generate the data")
    return shelve.open(os.path.join(path, "data"), "c")


def batched_dataset_iter(*args, batch_size, **kwargs):
    for batch in chunked(dataset_iter(*args, **kwargs), batch_size):
        yield tuple(zip(*batch))


def dataset_iter(segment, split=0.1, dataset=standard_dataset, seed=0):
    """
    Iterate through the dataset yielding spec, program pairs. The iteration
        order is determined by the seed, and the train/test split is determined
        deterministically by seed 0.

    Arguments:
        segment: the segment to use, must be 'train' or 'test'
        split: the percentage of the data to place in the test pool
    """
    dataset = dataset()
    indices = split_indices(segment, split, len(dataset))
    np.random.RandomState(seed).shuffle(indices)
    for i in indices:
        program, inputs = dataset[str(i)]
        inputs = inputs[:-1]
        spec = [
            MLozaicPair(
                inp, render(program.evaluate(inp), size=(50, 50), stretch=2, rgb=False)
            )
            for inp in inputs
        ]
        program = [BACKWARDS_ALPHABET[tok] + 2 for tok in program.code]
        yield MLozaicSpecification(spec), program
