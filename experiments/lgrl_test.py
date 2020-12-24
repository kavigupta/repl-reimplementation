import torch

from repl.lgrl import LGRL
from repl.mlozaic_environment.io_encoder import MLozaicIOEncoder

from repl.mlozaic_environment.load_mlozaic_environment import batched_dataset_iter
from repl.train import train_generic


def train_fn(model, idx, chunk):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model.loss(*chunk)
    loss.backward()
    opt.step()
    return loss.item()


def report_fn(outputs):
    return f"Loss: {np.mean(outputs)}"


train_generic(
    data=batched_dataset_iter(segment="train", batch_size=32),
    train_fn=train_fn,
    report_fn=report_fn,
    architectures=[
        lambda: LGRL(MLozaicIOEncoder(embedding_size=64), embedding_size=64)
    ],
    paths=["lgrl"],
    save_frequency=1,
)
