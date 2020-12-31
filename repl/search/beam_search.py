import attr

import torch

from ..utils import place
from .search import Search


@attr.s
class BeamSearch(Search):
    k = attr.ib()
    max_steps = attr.ib(default=100)

    def __call__(self, m, spec):
        weights = place(m, torch.zeros(1))
        beams = place(m, torch.zeros(self.k, 0, dtype=torch.long))
        finished = []
        state, preds = m.begin_inference([spec])
        for _ in range(self.max_steps):
            weights = weights[:, None] + preds
            good_finished = sum(1 for k, v in finished if k >= weights.max())
            values, indices = torch.topk(
                weights.flatten(), min(self.k - good_finished, len(weights.flatten()))
            )
            batch_idxs, tokens = flattened_meshgrid_like(weights)
            batch_idxs = batch_idxs[indices]
            tokens = tokens[indices]
            weights = weights.flatten()[indices]

            beams = torch.cat([beams[batch_idxs], tokens[:, None]], axis=1)

            dones = tokens == 1
            if dones.all():
                break
            finished.extend(zip(weights[dones], beams[dones]))
            beams = beams[~dones]
            batch_idxs = batch_idxs[~dones]
            tokens = tokens[~dones]
            weights = weights[~dones]

            state, preds = state.resample(batch_idxs).step(tokens)
        finished.extend(zip(weights, beams))
        return sorted(((w.item(), b.tolist()) for w, b in finished), reverse=True)


def flattened_meshgrid_like(tensor):
    return [
        x.flatten()
        for x in torch.meshgrid(
            *[torch.arange(s).to(tensor.device) for s in tensor.shape]
        )
    ]
