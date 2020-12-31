import numpy as np
import torch

from repl.lgrl import LGRL
from repl.karel_environment.spec_encoder import KarelSpecEncoder

from repl.karel_environment.load_karel_environment import KarelDataset
from repl.train import supervised_training


supervised_training(
    data=KarelDataset("train").multiple_epochs_iter(seed=0, batch_size=16, epochs=10),
    optimizer=lambda parameters: torch.optim.Adam(parameters, lr=1e-3),
    architectures=[
        lambda: LGRL(KarelSpecEncoder(embedding_size=64), embedding_size=64)
    ],
    paths=["logdirs/lgrl-karel"],
    save_frequency=20,
)
