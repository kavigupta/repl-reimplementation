import numpy as np

from synthesis.environment.robustfill.data import RobustfillDataset
from synthesis.environment.robustfill.dynamics import RobustfillDynamics
from synthesis.environment.robustfill.model import RobustfillPolicy
from synthesis.repl.train import pretrain, finetune

model_path = "logdirs/repl_robustfill_3"


data = RobustfillDataset("train")
dynamics = RobustfillDynamics()

pa = lambda: RobustfillPolicy().cuda()
va = lambda: RobustfillValue().cuda()
rng = np.random.RandomState(0)


batch_size = 256

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
