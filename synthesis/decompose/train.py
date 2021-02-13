import attr
from more_itertools import chunked

import torch
import numpy as np

from ..environment.dataset import Dataset
from ..train import train_generic


@attr.s
class DecomposerDataset(Dataset):
    underlying = attr.ib()
    oracle_decomposer = attr.ib()
    embedding_net = attr.ib()

    def __attrs_post_init__(self):
        super().__init__(self.underlying.segment)

    def dataset(self, seed):
        for spec, program in self.underlying.dataset(seed):
            first, _ = self.oracle_decomposer.split_spec(spec, program)
            yield [p.input for p in spec.pairs], [p.output for p in spec.pairs], [
                p.output for p in first.pairs
            ]


def train_decomposer(
    policy,
    decomposer_arch,
    data,
    rng,
    lr=1e-3,
    *,
    report_frequency=100,
    oracle_decomposer,
    batch_size,
    epochs,
    seed,
    model_path,
):
    optimizer = None

    def train_fn(decomposer, idx, chunk):
        nonlocal optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(decomposer.parameters(), lr=lr)
        with torch.no_grad():
            ins, outs, inters = [policy.embedding_net(grids) for grids in chunk]
        inters_pred = decomposer(ins, outs)
        assert inters.indices_for_each == inters_pred.indices_for_each
        loss = ((inters_pred.embeddings - inters.embeddings) ** 2).mean()
        loss.backward()
        optimizer.step()
        return loss.item()

    def report_fn(idx, output):
        losses = np.mean(output)
        return f"Loss: {np.mean(losses)}"

    train_generic(
        DecomposerDataset(
            data, oracle_decomposer, policy.embedding_net
        ).multiple_epochs_iter(batch_size=batch_size, epochs=epochs, seed=seed),
        train_fn,
        report_fn,
        [decomposer_arch],
        [model_path + "/decomposer"],
        save_frequency=report_frequency * 10,
        report_frequency=report_frequency,
    )
