import numpy as np

from repl.squares_environment import *
from repl.train import pretrain, finetune


config = SquaresConfig()
pretrain(SquaresPolicy(config, batch_size=16), lambda rng: sample(config, rng), np.random, n=10 ** 3)
finetune(
    SquaresPolicy(config),
    SquaresValue(config),
    lambda rng: sample(config, rng),
    np.random,
)
