import numpy as np
import torch

from synthesis.lgrl import LGRL
from synthesis.environment.karel.spec_encoder import KarelRecurrentSpecEncoder

from synthesis.environment.karel.load_karel_environment import KarelDataset
from synthesis.train import supervised_training


supervised_training(
    data=KarelDataset("train").multiple_epochs_iter(seed=0, batch_size=128, epochs=10),
    optimizer=lambda parameters: torch.optim.Adam(parameters, lr=1e-4),
    architectures=[
        lambda: LGRL(KarelRecurrentSpecEncoder(embedding_size=256), embedding_size=256)
    ],
    paths=["logdirs/lgrl-karel-recurrent"],
    save_frequency=200,
    report_frequency=20,
)
