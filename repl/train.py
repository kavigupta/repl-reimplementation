from more_itertools import chunked

import torch

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
        loss = -log_probs[range(log_probs.shape[0]), actions].mean()
        if idx % 100 == 0:
            print("Idx", idx, "Accuracy:", acc, "Loss:", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
