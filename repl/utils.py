import os

import attr

import torch

import numpy as np
from more_itertools import chunked


def load_model(folder, step=None, architecture=lambda: None):
    kwargs = {}
    if not torch.cuda.is_available():
        kwargs = dict(map_location=torch.device("cpu"))
    if os.path.isfile(folder):
        return None, torch.load(folder, **kwargs)
    model_dir = os.path.join(folder, "model")
    if not os.path.exists(model_dir):
        return 0, architecture()
    if step is None and os.listdir(model_dir):
        step = max(os.listdir(model_dir), key=int)
    path = os.path.join(model_dir, str(step))
    if not os.path.exists(path):
        return 0, architecture()
    return int(step), torch.load(path, **kwargs)


def save_model(model, folder, step):
    path = os.path.join(folder, "model", str(step))
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass
    torch.save(model, path)


def shuffle_chunks(data, chunk_size, rng=np.random):
    for chunk in chunked(data, chunk_size):
        chunk = list(chunk)
        rng.shuffle(chunk)
        yield from chunk


@attr.s
class JaggedEmbeddings:
    """
    Represents a jagged array of 2d embeddings.

    For example, if you want to represent [[a, b], [c, d, e]],
        you would represent it as

        InputEmbeddings([a, b, c, d, e], [[0, 1], [2, 3, 4]])
    """

    embeddings = attr.ib()
    indices_for_each = attr.ib()

    @property
    def _original_index(self):
        result = [None] * len(self.embeddings)
        for original_idx, indices in enumerate(self.indices_for_each):
            for idx in indices:
                result[idx] = original_idx
        return result

    def tile(self, extra):
        assert extra.shape[0] == len(self.indices_for_each)
        return extra[self._original_index]

    def cat(self, extra):
        assert extra.shape[0] == len(self.embeddings)
        return JaggedEmbeddings(
            torch.cat([self.embeddings, extra], dim=0), self.indices_for_each
        )

    def replace(self, new_embeddings):
        assert self.embeddings.shape[0] == new_embeddings.shape[0]
        return JaggedEmbeddings(new_embeddings, self.indices_for_each)

    def max_pool(self):
        """
        Pool everything in the same index class by max
        """
        return torch.stack(
            [self.embeddings[idxs].max(0)[0] for idxs in self.indices_for_each]
        )


@attr.s
class PaddedSequence:
    sequences = attr.ib()  # N x L
    mask = attr.ib()  # N x l

    @staticmethod
    def of(values, dtype):
        N = len(values)
        L = max(len(v) for v in values)
        sequences, mask = torch.zeros((N, L), dtype=dtype), torch.zeros(
            (N, L), dtype=torch.bool
        )
        for i, v in enumerate(values):
            sequences[i, : len(v)] = torch.tensor(v)
            mask[i, : len(v)] = 1
        return PaddedSequence(sequences, mask)

    @property
    def L(self):
        return self.mask.shape[1]
