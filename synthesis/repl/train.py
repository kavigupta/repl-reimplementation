import torch
import numpy as np

import attr
from more_itertools import chunked

from ..train import train_generic
from .state import ReplSearchState
from ..environment.dataset import Dataset


@attr.s
class PretrainDataset(Dataset):
    underlying = attr.ib()
    dynamics = attr.ib()

    def __attrs_post_init__(self):
        super().__init__(self.underlying.segment)

    def dataset(self, seed):
        for spec, program in self.underlying.dataset(seed):
            for pp, a in program.partials:
                yield (ReplSearchState(pp, spec, self.dynamics), a)


def pretrain(
    policy_arch,
    dynamics,
    data,
    rng,
    lr=1e-3,
    *,
    report_frequency=100,
    batch_size,
    epochs,
    seed,
    model_path,
):
    optimizer = None
    def train_fn(policy, idx, chunk):
        nonlocal optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        states, actions = chunk
        dist = policy(states)
        predictions = dist.mle()
        optimizer.zero_grad()
        loss = -dist.log_probability(actions).sum()
        loss.backward()
        optimizer.step()
        acc = np.mean([p == a for p, a in zip(predictions, actions)])
        return acc, loss.item()

    def report_fn(idx, outputs):
        accs, losses = np.array(outputs).T
        return f"Accuracy: {np.mean(accs) * 100:.02f}% Loss: {np.mean(losses)}"

    train_generic(
        PretrainDataset(data, dynamics)
        .shuffle_chunks(batch_size * 100)
        .multiple_epochs_iter(batch_size=batch_size, epochs=epochs, seed=seed),
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
