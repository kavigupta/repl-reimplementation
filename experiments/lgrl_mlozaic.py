import numpy as np
import torch

from synthesis.lgrl import LGRL
from synthesis.mlozaic_environment.spec_encoder import MLozaicSpecEncoder

from synthesis.mlozaic_environment.load_mlozaic_environment import MlozaicDataset
from synthesis.train import supervised_training


supervised_training(
    data=MlozaicDataset("train").multiple_epochs_iter(seed=0, batch_size=2, epochs=10),
    optimizer=lambda parameters: torch.optim.Adam(parameters, lr=1e-5),
    architectures=[
        lambda: LGRL(
            MLozaicSpecEncoder(embedding_size=64),
            embedding_size=64,
        )
    ],
    paths=["logdirs/lgrl-mlozaic"],
    save_frequency=200,
    report_frequency=20,
)
