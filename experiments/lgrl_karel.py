import numpy as np
import torch

from repl.lgrl import LGRL
from repl.karel_environment.spec_encoder import KarelSpecEncoder

from repl.karel_environment.load_karel_environment import KarelDataset
from repl.train import supervised_training

embedding_size = 256

supervised_training(
    data=KarelDataset("train").multiple_epochs_iter(seed=0, batch_size=16, epochs=10),
    optimizer=lambda parameters: torch.optim.Adam(parameters, lr=1e-4 * 16 / 128),
    architectures=[
        lambda: LGRL(
            KarelSpecEncoder(embedding_size=embedding_size),
            embedding_size=embedding_size,
        )
    ],
    paths=[f"logdirs/lgrl-karel-{embedding_size}"],
    save_frequency=200,
    report_frequency=20,
)
