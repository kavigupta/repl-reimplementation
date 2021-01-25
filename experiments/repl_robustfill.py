import numpy as np

from synthesis.environment.robustfill.data import RobustfillDataset
from synthesis.environment.robustfill.dynamics import RobustfillDynamics
from synthesis.environment.robustfill.model import RobustfillPolicy
from synthesis.repl.train import pretrain, finetune

model_path = "logdirs/repl_robustfill_1"


data = RobustfillDataset("train")
dynamics = RobustfillDynamics()

pa = lambda: RobustfillPolicy().cuda()
va = lambda: RobustfillValue().cuda()
rng = np.random.RandomState(0)


batch_size = 512

pretrain(
    pa,
    dynamics,
    data,
    rng,
    lr=1e-4,
    model_path=model_path,
    batch_size=batch_size, epochs=10, seed=0
)
