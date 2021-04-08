import numpy as np

from synthesis.repl.train import pretrain, finetune
from synthesis.environment.karel.sequential_karel import KarelSequentialDataset, FlatRepeatsSampler
from synthesis.environment.karel.sequential_dynamics import KarelDynamics
from synthesis.environment.karel.sequential_model import (
    KarelSequentialPolicy,
    KarelSequentialValue,
    KarelSequentialDecomposer,
)
from synthesis.utils.utils import load_model

model_path = "logdirs/repl_karel_but_1"

max_length = 1000

data = KarelSequentialDataset("train", FlatRepeatsSampler(False), limit=2_000_000)
dynamics = KarelDynamics(max_length)

pa = lambda: KarelSequentialPolicy(max_length=max_length).cuda()
da = lambda: KarelSequentialDecomposer().cuda()
rng = np.random.RandomState(0)


batch_size = 32

pretrain(
    pa,
    dynamics,
    data,
    rng,
    lr=1e-5,
    model_path=model_path,
    batch_size=batch_size,
    epochs=2,
    seed=0,
)

va = lambda: KarelSequentialValue(pol).cuda()

finetune(
    va,
    data,
    rng,
    lr=1e-7,
    model_path=model_path,
    batch_size=batch_size,
    epochs=2,
    seed=1,
)
