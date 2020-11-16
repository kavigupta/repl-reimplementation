import numpy as np

from repl.squares_environment import *
from repl.train import pretrain, finetune


config = SquaresConfig()
p = SquaresPolicy(config, batch_size=16)
p = p.cuda()
pretrain(p, lambda rng: sample(config, rng), np.random, n=10 ** 3)
finetune(
    SquaresPolicy(config),
    SquaresValue(config),
    lambda rng: sample(config, rng),
    np.random,
)
