from more_itertools import chunked

import torch
import numpy as np

from .state import State
from .utils import save_model, shuffle_chunks


def pretrain(policy, sampler, rng, n=10000, lr=1e-3, *, model_path):
    def data_iterator():
        for _ in range(n):
            spec, program = sampler(rng)
            for pp, a in program.partials:
                yield (State(pp, spec), a)

    data = shuffle_chunks(data_iterator(), int(n ** (2 / 3)), rng=rng)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for idx, chunk in enumerate(chunked(data, policy.batch_size)):
        states, actions = zip(*chunk)
        dist = policy(states)
        predictions = dist.mle()
        acc = np.mean([p == a for p, a in zip(predictions, actions)])
        loss = -dist.log_probability(actions).sum()
        if idx % 100 == 0:
            print(f"Step {idx}, Accuracy: {acc * 100:.02f}% Loss: {loss.item()}")
            save_model(policy, model_path + "/p", policy.batch_size * idx)
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
        rewards = torch.tensor(rewards).float().to(next(value.parameters()).device)
        value_reward = (rewards * v.log() + (1 - rewards) * (1 - v).log()).sum()
        dist = policy(states)
        policy_reward = (rewards * dist.log_probability(actions)).sum()
        loss = -(value_reward + policy_reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def finetune(policy, value, sampler, rng, n=10000, *, model_path, **kwargs):
    n_each = policy.batch_size * 10
    step = 0
    for idxs in chunked(range(n), n_each):
        finetune_step(policy, value, sampler, rng, **kwargs, n=len(idxs))
        step += len(idxs)
        save_model(policy, model_path + "/pf", step)
        save_model(value, model_path + "/vf", step)
