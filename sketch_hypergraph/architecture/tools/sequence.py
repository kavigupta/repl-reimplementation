import torch


def flat_to_batched(values, lengths):
    """
    Takes in a list of packed sequential values of the form (L1 + L2 + ... + Lk, ...) a list of lengths
        [L1, L2, ..., Lk] and returns a list of batched out values of the form (k, Lmax, ...).

    The remaining values are padded by 0s.
    """
    total_amount, *dims = values.shape
    assert total_amount == sum(lengths)
    result = torch.zeros(
        (len(lengths), max(lengths), *dims), dtype=values.dtype, device=values.device
    )
    batch_indices, seq_indices = [], []
    for i, length in enumerate(lengths):
        for j in range(length):
            batch_indices.append(i)
            seq_indices.append(j)
    result[batch_indices, seq_indices] = values
    return result


def sequence_mask_for(lengths):
    max_length = max(lengths)
    return torch.arange(max_length).expand(len(lengths), max_length) < torch.tensor(
        lengths
    ).unsqueeze(1)
