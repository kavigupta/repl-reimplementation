import os
import pickle
import struct

from more_itertools import chunked

import numpy as np

from ..spec import Specification, Pair

GRID_SIZE = (15, 18, 18)


def check_data(path):
    if not os.path.exists(path):
        error_message = f"""
        Data not found. Run

        mkdir -p {path}
        cd {path}
        wget https://s3.us-east-2.amazonaws.com/karel-dataset/karel.tar.gz
        tar xf karel.tar.gz
        cd -

        """
        raise RuntimeError(error_message)


def standard_dataset(segment, *, path="data/karel_standard"):
    check_data(path)
    root = os.path.join(path, "karel")
    path_prefix = os.path.join(
        root, {"train": "train", "test": "val"}[segment] + ".pkl"
    )
    return KarelData(path_prefix)


def read_vocab(root="data/karel_standard"):
    check_data(root)
    with open(os.path.join(root, "karel", "word.vocab")) as f:
        vocab = [line.split() for line in f]
    return {k: int(v) for k, v in vocab}


class KarelData:
    def __init__(self, path_prefix):
        self.indices = read_index(path_prefix + ".index")
        self.obj_file = open(path_prefix, "rb")

    def shuffle(self, seed):
        np.random.RandomState(seed).shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self.obj_file.seek(self.indices[i])
        object = pickle.load(self.obj_file, encoding="latin-1")
        pairs = [
            Pair(get_one_hot(eg["in"]), get_one_hot(eg["out"]))
            for eg in object["examples"]
        ]
        pairs = pairs[:-1]
        return Specification(pairs), object["code"]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def get_one_hot(indices):
    """
    Input: a list of indices which refer to positions marked "True" in a Karel grid.
    Output: a one-hot numpy array for the grid.
    """
    grid = np.zeros(GRID_SIZE, dtype=np.float32)
    grid.ravel()[indices] = 1

    return grid


def read_index(filename):
    index = []
    with open(filename, "rb") as index_file:
        while True:
            offset = index_file.read(8)
            if not offset:
                break
            (offset,) = struct.unpack("<Q", offset)
            index.append(offset)
    return index


def batched_dataset_iter(*args, batch_size, **kwargs):
    for batch in chunked(dataset_iter(*args, **kwargs), batch_size):
        yield tuple(zip(*batch))


def dataset_iter(segment, dataset=standard_dataset, seed=0):
    """
    Iterate through the dataset yielding spec, program pairs.

    Arguments:
        segment: the segment to use, must be 'train' or 'test'
    """
    ba = read_vocab()
    dataset = dataset(segment)
    dataset.shuffle(seed)
    for spec, program in dataset:
        program = [ba[tok] + 2 for tok in program]
        yield spec, program
