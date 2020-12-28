import numpy as np
import torch

from repl.lgrl import LGRL
from repl.mlozaic_environment.spec_encoder import MLozaicSpecEncoder

from repl.mlozaic_environment.load_mlozaic_environment import MlozaicDataset
from repl.train import supervised_training


supervised_training(
    data=MlozaicDataset("train").multiple_epochs_iter(seed=0, batch_size=32, epochs=10),
    optimizer=lambda parameters: torch.optim.Adam(parameters, lr=1e-3),
    architectures=[
        lambda: LGRL(
            MLozaicSpecEncoder(embedding_size=64),
            embedding_size=64,
        )
    ],
    paths=["logdirs/basic-lgrl"],
    save_frequency=20,
)
