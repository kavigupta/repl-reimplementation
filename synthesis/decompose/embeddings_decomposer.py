"""
Decomposer models that assumes embeddings are used both input and output
"""

import torch
import torch.nn as nn

from ..utils.utils import JaggedEmbeddings


class TransformerEmbeddingsDecomposer(nn.Module):
    def __init__(self, e, **kwargs):
        super().__init__()
        self.transformer = nn.Transformer(d_model=e, **kwargs)
        self.projection = nn.Linear(e, e)

    def forward(self, spec_embeddings):
        sequences = spec_embeddings.to_padded_sequence()
        sequences = sequences.self_attn(self.transformer)
        sequences = sequences.map(self.projection)
        sequences = sequences.flatten()
        return JaggedEmbeddings(sequences, spec_embeddings.indices_for_each)
