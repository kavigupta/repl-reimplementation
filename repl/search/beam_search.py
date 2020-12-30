def beam_search(m, spec, k, max_steps=100):
    weights = torch.zeros(1)
    beams = torch.zeros(k, 0)
    finished = []
    state, preds = m.begin_inference([spec])
    for _ in range(max_steps):
        values, indices = torch.topk(preds.flatten(), k)
        batch_idxs, tokens = flattened_meshgrid_like(preds)
        batch_idxs = batch_idxs[indices]
        tokens = tokens[indices]

        beams = torch.cat([beams, tokens[:, None]], axis=1)
        weights = weights[batch_idxs] + values

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
    return finished
