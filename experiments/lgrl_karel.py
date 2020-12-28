import numpy as np
import torch

from repl.lgrl import LGRL
from repl.karel_environment.spec_encoder import KarelSpecEncoder

from repl.karel_environment.load_karel_environment import batched_dataset_iter
from repl.train import train_generic


def train_fn(model, idx, chunk):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = model.loss(*chunk)
    loss.backward()
    opt.step()
    return loss.item()


def report_fn(idx, outputs):
    return f"Loss: {np.mean(outputs)}"


def data():
    for seed in range(10):
        yield from batched_dataset_iter(segment="train", batch_size=32, seed=seed)


train_generic(
    data=data(),
    train_fn=train_fn,
    report_fn=report_fn,
    architectures=[
        lambda: LGRL(KarelSpecEncoder(embedding_size=64), embedding_size=64)
    ],
    paths=["logdirs/lgrl-karel-1e-4"],
    save_frequency=20,
)
