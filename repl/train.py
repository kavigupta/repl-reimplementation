from more_itertools import chunked

import torch
import numpy as np

from .state import State


def pretrain(policy, sampler, rng, n=10000, lr=1e-3):
    data = []
    for _ in range(n):
        spec, program = sampler(rng)
        for pp, a in program.partials:
            data.append((State(pp, spec), a))
    rng.shuffle(data)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for idx, chunk in enumerate(chunked(data, policy.batch_size)):
        states, actions = zip(*chunk)
        log_probs = policy(states)
        _, predictions = log_probs.max(1)
        acc = (predictions.numpy() == actions).mean()
        loss = -log_probs[range(log_probs.shape[0]), actions].sum()
        if idx % 100 == 0:
            print("Idx", idx, "Accuracy:", acc, "Loss:", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def finetune_step(policy, value, sampler, rng, n=1000, lr=1e-3):
    data = []

    specs = [sampler(rng)[0] for _ in range(n)]
    rewards = []
    for idx, chunk in enumerate(chunked(specs, policy.batch_size)):
        partials = policy.roll_forward(chunk)
        for partial in partials:
            last_state, _ = partial[-1]
            reward = last_state.is_goal
            rewards.append(reward)
            for state, action in partial:
                if action is None:
                    continue
                data.append((state, action, reward))
    print(f"Rewards: {np.mean(rewards):.2f}")
    rng.shuffle(data)

    optimizer = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=lr)

    for chunk in chunked(data, policy.batch_size):
        states, actions, rewards = zip(*chunk)

        v = value(states)
        rewards = torch.tensor(rewards).float()
        value_reward = (rewards * v.log() + (1 - rewards) * (1 - v).log()).sum()
        log_probs = policy(states)
        policy_reward = log_probs[range(log_probs.shape[0]), actions].sum()
        loss = -(value_reward + policy_reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def finetune(policy, value, sampler, rng, n=10000, **kwargs):
    n_each = policy.batch_size * 10
    for idxs in chunked(range(n), n_each):
        finetune_step(policy, value, sampler, rng, **kwargs, n=len(idxs))
