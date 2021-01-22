from datetime import datetime

import torch
import numpy as np

from .utils.utils import load_model, save_model


def train_generic(
    data,
    train_fn,
    report_fn,
    architectures,
    paths,
    save_frequency,
    report_frequency=None,
    gpu=True,
):
    if report_frequency is None:
        report_frequency = save_frequency
    models = []
    min_step = float("inf")
    for arch, path in zip(architectures, paths):
        step, model = load_model(path, architecture=arch)
        if gpu:
            model.cuda()
        else:
            model.cpu()
        models.append(model)
        min_step = min(step, min_step)

    outputs = []
    for idx, chunk in enumerate(data):
        if idx < step:
            continue
        outputs.append(train_fn(*models, idx, chunk))
        if (idx + 1) % save_frequency == 0:
            for model, path in zip(models, paths):
                save_model(model, path, idx)
        if (idx + 1) % report_frequency == 0:
            print(f"[{datetime.now()}]: s={idx}, {report_fn(idx, outputs)}")
            outputs = []
    return models


def supervised_training(optimizer, **kwargs):
    opt = None

    def train_fn(model, idx, chunk):
        nonlocal opt
        if opt is None:
            opt = optimizer(model.parameters())
        loss = model.loss(*chunk)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    def report_fn(idx, outputs):
        return f"Loss: {np.mean(outputs)}"

    return train_generic(report_fn=report_fn, train_fn=train_fn, **kwargs)
