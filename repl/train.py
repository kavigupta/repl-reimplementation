from datetime import datetime

from more_itertools import chunked

import torch
import numpy as np

from .state import State
from .utils import load_model, save_model, shuffle_chunks


def train_generic(data, train_fn, report_fn, architectures, paths, save_frequency):
    models = []
    min_step = float("inf")
    for arch, path in zip(architectures, paths):
        step, model = load_model(path, architecture=arch)
        models.append(model)
        min_step = min(step, min_step)

    outputs = []
    for idx, chunk in enumerate(data):
        if idx < step:
            continue
        outputs.append(train_fn(*models, idx, chunk))
        if (idx + 1) % save_frequency == 0:
            save_model(*models, path, idx)
            print(f"[{datetime.now()}]: s={idx}, {report_fn(idx, outputs)}")
            outputs = []
    return models


def pretrain(
    policy_arch, sampler, rng, n=10000, lr=1e-3, *, report_frequency=100, model_path
):
    def data_iterator():
        for _ in range(n):
            spec, program = sampler(rng)
            for pp, a in program.partials:
                yield (State(pp, spec), a)

    def train_fn(policy, idx, chunk):
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        states, actions = zip(*chunk)
        dist = policy(states)
        predictions = dist.mle()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = np.mean([p == a for p, a in zip(predictions, actions)])
        loss = -dist.log_probability(actions).sum()
        return acc, loss

    def report_fn(outputs):
        accs, losses = np.array(outputs).T
        return f"Accuracy: {np.mean(accs) * 100:.02f}% Loss: {np.mean(losses)}"

    data = shuffle_chunks(data_iterator(), int(n ** (2 / 3)), rng=rng)

    train_generic(
        chunked(data, policy.batch_size),
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


def finetune(policy, value, sampler, rng, n=10000, *, model_path, **kwargs):
    train_generic(
        data=chunked(range(n), n_each),
        train_fn=lambda policy, value, idx, chunk: finetune_step(
            policy, value, sampler, rng, **kwargs, n=len(chunk)
        ),
        report_fn=lambda outs: f"Reward: {outs[0]}",
        architectures=[policy, value],
        paths=[model_path + "/pf", model_path + "/vf"],
        save_frequency=1,
    )


def supervised_training(optimizer, **kwargs):
    def train_fn(model, idx, chunk):
        opt = optimizer(model.parameters())
        loss = model.loss(*chunk)
        loss.backward()
        opt.step()
        return loss.item()

    def report_fn(idx, outputs):
        return f"Loss: {np.mean(outputs)}"

    return train_generic(report_fn=report_fn, train_fn=train_fn, **kwargs)
