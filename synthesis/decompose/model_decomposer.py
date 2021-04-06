import os

import numpy as np
import torch

from ..environment.spec import Specification, Pair


def run_decomposer(spec, decomposer_model):
    with torch.no_grad():
        ins = [p.input for p in spec.pairs]
        outs = [p.output for p in spec.pairs]
        return decomposer_model([ins], [outs]).embeddings.sigmoid().cpu().numpy()


def decompose(spec, decomposer_model):
    ins = [p.input for p in spec.pairs]
    inters = run_decomposer(spec, decomposer_model)
    inters = inters > decomposer_model.thresholds[None, :, None, None]
    return Specification([Pair(i, o) for i, o in zip(ins, inters)], [])


def condition_model(decomposer_model, dataset, path, limit=2000):
    if not os.path.exists(path):
        percentages = np.array(
            [
                [(p.input, p.output) for p in spec.pairs]
                for spec, _ in dataset.limit(limit).dataset(0)
            ]
        ).mean((0, 1, 2, 4, 5))
        inters = np.array(
            [
                run_decomposer(spec, decomposer_model)
                for spec, _ in dataset.limit(limit).dataset(0)
            ]
        )
        inters = inters.transpose((2, 0, 1, 3, 4)).reshape(15, -1)
        thresholds = np.array(
            [np.percentile(i, (1 - p) * 100) for i, p in zip(inters, percentages)]
        )
        np.save(path, thresholds)
    decomposer_model.thresholds = np.load(path)
