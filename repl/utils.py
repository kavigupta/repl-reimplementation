import os
import torch

import numpy as np
from more_itertools import chunked


def load_model(folder, step=None, architecture=lambda: None):
    kwargs = {}
    if not torch.cuda.is_available():
        kwargs = dict(map_location=torch.device("cpu"))
    if os.path.isfile(folder):
        return None, torch.load(folder, **kwargs)
    model_dir = os.path.join(folder, "model")
    if not os.path.exists(model_dir):
        return 0, architecture()
    if step is None and os.listdir(model_dir):
        step = max(os.listdir(model_dir), key=int)
    path = os.path.join(model_dir, str(step))
    if not os.path.exists(path):
        return 0, architecture()
    return int(step), torch.load(path, **kwargs)


def save_model(model, folder, step):
    path = os.path.join(folder, "model", str(step))
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass
    torch.save(model, path)


def shuffle_chunks(data, chunk_size, rng=np.random):
    for chunk in chunked(data, chunk_size):
        chunk = list(chunk)
        rng.shuffle(chunk)
        yield from chunk
