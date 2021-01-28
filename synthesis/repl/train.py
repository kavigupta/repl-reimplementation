import torch
import numpy as np

import attr
from more_itertools import chunked

from ..train import train_generic
from .state import ReplSearchState
from ..environment.dataset import Dataset
from ..utils.utils import load_model


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


@attr.s
class FinetuneDataset(Dataset):
    underlying = attr.ib()
    policy = attr.ib()
    batch_size = attr.ib(kw_only=True)

    def __attrs_post_init__(self):
        super().__init__(self.underlying.segment)

    def dataset(self, seed):
        rng = np.random.RandomState(seed)
        data = self.underlying.batched_dataset_iter(
            seed=rng.randint(2 ** 32), batch_size=self.batch_size
        )
        for specs, program in data:

            partials = self.policy.roll_forward(specs, rng)
            for partial in partials:
                is_goal, _, last_state_idx = max(
                    [(st.is_goal, not st.done, i) for i, (st, _) in enumerate(partial)]
                )
                reward = is_goal
                for state, action in partial[:last_state_idx]:
                    if action is None:
                        continue
                    yield (state, action, reward)


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
        save_frequency=report_frequency * 10,
        report_frequency=report_frequency,
    )


def finetune(
    value, data, rng, *, model_path, batch_size, lr, epochs, seed, report_frequency=100
):

    _, policy = load_model(model_path + "/p")

    finetune_data = FinetuneDataset(data, policy, batch_size=10)

    optimizer = None

    def train_fn(policy, value, idx, chunk):
        nonlocal optimizer

        if optimizer is None:
            optimizer = torch.optim.Adam(
                [*policy.parameters(), *value.parameters()], lr=lr
            )

        states, actions, rewards = chunk

        v = value(states)
        rewards = torch.tensor(rewards).float().to(v.device)
        value_reward = (rewards * v.log() + (1 - rewards) * (1 - v).log()).sum()

        dist = policy(states)
        policy_reward = (rewards * dist.log_probability(actions)).sum()
        loss = -(value_reward + policy_reward)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return rewards.mean().item(), loss.item()

    def report_fn(idx, outputs):
        accs, losses = np.array(outputs).T
        return f"Reward %: {np.mean(accs) * 100:.02f}% Loss: {np.mean(losses)}"

    train_generic(
        data=finetune_data.shuffle_chunks(batch_size * 100).multiple_epochs_iter(
            batch_size=batch_size, epochs=epochs, seed=seed
        ),
        train_fn=train_fn,
        report_fn=report_fn,
        architectures=[lambda: policy, value],
        paths=[model_path + "/pf", model_path + "/vf"],
        save_frequency=report_frequency * 10,
        report_frequency=report_frequency,
    )
