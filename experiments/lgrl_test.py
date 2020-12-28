import numpy as np
import torch

from repl.lgrl import LGRL
from repl.mlozaic_environment.spec_encoder import MLozaicSpecEncoder

from repl.mlozaic_environment.load_mlozaic_environment import batched_dataset_iter
from repl.train import train_generic


def train_fn(model, idx, chunk):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model.loss(*chunk)
    loss.backward()
    opt.step()
    return loss.item()


def report_fn(idx, outputs):
    return f"Loss: {np.mean(outputs)}"


def data():
    for seed in range(10):
        yield from batched_dataset_iter(segment="train", batch_size=32, seed=seed)


embedding_size = 64

train_generic(
    data=data(),
    train_fn=train_fn,
    report_fn=report_fn,
    architectures=[
        lambda: LGRL(
            MLozaicSpecEncoder(embedding_size=embedding_size),
            embedding_size=embedding_size,
        )
    ],
    paths=["logdirs/basic-lgrl"],
    save_frequency=20,
)
