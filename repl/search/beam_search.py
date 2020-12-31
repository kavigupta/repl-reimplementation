import torch


def beam_search(m, spec, k, max_steps=100):
    weights = torch.zeros(1)
    beams = torch.zeros(k, 0, dtype=torch.long)
    finished = []
    state, preds = m.begin_inference([spec])
    for _ in range(max_steps):
        weights = weights[:, None] + preds
        good_finished = sum(1 for k, v in finished if k >= weights.max())
        values, indices = torch.topk(weights.flatten(), k - good_finished)
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
    return sorted(((w, b.tolist()) for w, b in finished), reverse=True)


def flattened_meshgrid_like(tensor):
    return [
        x.flatten() for x in torch.meshgrid(*[torch.arange(s) for s in tensor.shape])
    ]
