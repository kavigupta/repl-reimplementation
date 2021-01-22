from datetime import datetime

from more_itertools import chunked

import torch
import numpy as np

from .repl.state import State
from .utils.utils import load_model, save_model, shuffle_chunks


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


def pretrain(policy_arch, data, rng, lr=1e-3, *, report_frequency=100, model_path):
    def train_fn(policy, idx, chunk):
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        states, actions = zip(*chunk)
        dist = policy(states)
        predictions = dist.mle()
        optimizer.zero_grad()
        loss = -dist.log_probability(actions).sum()
        loss.backward()
        optimizer.step()
        acc = np.mean([p == a for p, a in zip(predictions, actions)])
        return acc, loss

    def report_fn(outputs):
        accs, losses = np.array(outputs).T
        return f"Accuracy: {np.mean(accs) * 100:.02f}% Loss: {np.mean(losses)}"

    train_generic(
        data,
        train_fn,
        report_fn,
        [policy_arch],
        [model_path + "/p"],
        report_frequency,
    )


def finetune_step(policy, value, sampler, rng, n=1000, lr=1e-3):
    data = []

    specs = [sampler(rng)[0] for _ in range(n)]
    rewards = []
    for idx, chunk in enumerate(chunked(specs, policy.batch_size)):
        partials = policy.roll_forward(chunk, rng)
        for partial in partials:
            last_state, _ = partial[-1]
            reward = last_state.is_goal
            rewards.append(reward)
            for state, action in partial:
                if action is None:
                    continue
                data.append((state, action, reward))
    rng.shuffle(data)

    optimizer = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=lr)

    for chunk in chunked(data, policy.batch_size):
        states, actions, rewards = zip(*chunk)

        v = value(states)
        rewards = torch.tensor(rewards).float().to(next(value.parameters()).device)
        value_reward = (rewards * v.log() + (1 - rewards) * (1 - v).log()).sum()
        dist = policy(states)
        policy_reward = (rewards * dist.log_probability(actions)).sum()
        loss = -(value_reward + policy_reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return rewards


def finetune(policy, value, data, rng, *, model_path, **kwargs):
    train_generic(
        data=data,
        train_fn=lambda policy, value, idx, chunk: finetune_step(
            policy, value, chunk, rng, **kwargs, n=len(chunk)
        ),
        report_fn=lambda outs: f"Reward: {outs[0]}",
        architectures=[policy, value],
        paths=[model_path + "/pf", model_path + "/vf"],
        save_frequency=1,
    )


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