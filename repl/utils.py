import os
import math

import attr

import torch
import torch.nn as nn

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

    def __getitem__(self, indices):
        new_indices = []
        gather = []
        for i in indices:
            each = self.indices_for_each[i]
            new_indices.append(list(range(len(gather), len(gather) + len(each))))
            gather.extend(each)
        return JaggedEmbeddings(
            embeddings=self.embeddings[gather], indices_for_each=new_indices
        )


@attr.s
class PaddedSequence:
    sequences = attr.ib()  # N x L
    mask = attr.ib()  # N x l

    @staticmethod
    def of(values, dtype, place_m):
        N = len(values)
        L = max(len(v) for v in values)
        sequences, mask = place(place_m, torch.zeros((N, L), dtype=dtype)), place(
            place_m, torch.zeros((N, L), dtype=torch.bool)
        )
        for i, v in enumerate(values):
            sequences[i, : len(v)] = place(place_m, torch.tensor(v))
            mask[i, : len(v)] = 1
        return PaddedSequence(sequences, mask)

    @property
    def L(self):
        return self.mask.shape[1]

    def map(self, f):
        return PaddedSequence(f(self.sequences), self.mask)


def split_indices(segment, split, dataset_size):
    test_indices = np.random.RandomState(0).rand(dataset_size) < split
    if segment == "test":
        indices_to_use = test_indices
    elif segment == "train":
        indices_to_use = ~test_indices
    else:
        raise RuntimeError("invalid segment, must be train or test")
    return np.arange(dataset_size, dtype=np.int)[indices_to_use]


class PositionalEncoding(nn.Module):
    # from the tutorial https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def place(m, x):
    x = x.to(next(m.parameters()).device)
    return x
