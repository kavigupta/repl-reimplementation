import numpy as np

from repl.squares_environment import *
from repl.train import pretrain, finetune

model_path = "logdirs/squares_repl_1"
config = SquaresConfig(size=5, max_num_squares=4)

pa = lambda: SquaresPolicy(config, batch_size=64).cuda()
va = lambda: SquaresValue(config).cuda()
rng = np.random.RandomState(0)
pretrain(
    pa,
    lambda rng: sample(config, rng),
    rng,
    n=10 ** 6,
    lr=1e-4,
    model_path=model_path,
)
finetune(
    pa,
    va,
    lambda rng: sample(config, rng),
    rng,
    model_path=model_path,
    n=10 ** 6,
    lr=1e-5,
)
