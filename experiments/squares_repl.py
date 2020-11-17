import numpy as np

from repl.squares_environment import *
from repl.train import pretrain, finetune

model_path = "logdirs/squares_repl_0"

config = SquaresConfig(size=10)
p = SquaresPolicy(config, batch_size=64).cuda()
v = SquaresValue(config).cuda()
rng = np.random.RandomState(0)
pretrain(
    p, lambda rng: sample(config, rng), rng, n=10 ** 6, lr=1e-4, model_path=model_path
)
finetune(p, v, lambda rng: sample(config, rng), rng, model_path=model_path, n=10 ** 6)
