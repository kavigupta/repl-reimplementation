import numpy as np

from synthesis.repl.train import pretrain, finetune
from synthesis.environment.karel.sequential_karel import KarelSequentialDataset
from synthesis.environment.karel.sequential_dynamics import KarelDynamics
from synthesis.environment.karel.sequential_model import KarelSequentialPolicy

model_path = "logdirs/repl_karel_1"


data = KarelSequentialDataset("train", 8, limit=250_000)
dynamics = KarelDynamics()

pa = lambda: KarelSequentialPolicy().cuda()
# va = lambda: RobustfillValue().cuda()
rng = np.random.RandomState(0)


batch_size = 1024

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
