import numpy as np

from repl.blocks_environment import *
from repl.train import pretrain


config = BlockConfig()
pretrain(BlocksPolicy(config), lambda rng: sample(config, rng), np.random)
