import numpy as np

from repl.blocks_environment import *
from repl.train import pretrain, finetune


config = BlockConfig()
pretrain(BlocksPolicy(config), lambda rng: sample(config, rng), np.random, n=10 ** 5)
finetune(
    BlocksPolicy(config),
    BlocksValue(config),
    lambda rng: sample(config, rng),
    np.random,
)
