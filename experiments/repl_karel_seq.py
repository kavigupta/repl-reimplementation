import numpy as np

from synthesis.repl.train import pretrain, finetune
from synthesis.environment.karel.sequential_karel import KarelSequentialDataset
from synthesis.environment.karel.sequential_dynamics import KarelDynamics
from synthesis.environment.karel.sequential_model import (
    KarelSequentialPolicy,
    KarelSequentialValue,
    KarelSequentialDecomposer,
)
from synthesis.utils.utils import load_model
from synthesis.decompose.train import train_decomposer
from synthesis.decompose.oracle_decomposer import (
    OracleDecomposer,
    half_split_sequential_program,
)

model_path = "logdirs/repl_karel_5"


data = KarelSequentialDataset("train", 8, limit=250_000)
dynamics = KarelDynamics()

pa = lambda: KarelSequentialPolicy().cuda()
da = lambda: KarelSequentialDecomposer().cuda()
rng = np.random.RandomState(0)


batch_size = 512

pretrain(
    pa,
    dynamics,
    data,
    rng,
    lr=1e-5,
    model_path=model_path,
    batch_size=batch_size,
    epochs=1,
    seed=0,
)


train_decomposer(
    da,
    data,
    rng,
    lr=1e-5,
    model_path=model_path,
    oracle_decomposer=OracleDecomposer(half_split_sequential_program, KarelDynamics()),
    batch_size=batch_size // 4,
    epochs=10,
    seed=1,
)

# va = lambda: KarelSequentialValue(pol).cuda()

# finetune(
#     va,
#     data,
#     rng,
#     lr=1e-7,
#     model_path=model_path,
#     batch_size=batch_size,
#     epochs=2,
#     seed=1,
# )
