"""
Decomposer models that assumes embeddings are used both input and output
"""

import torch
import torch.nn as nn

from ..utils.utils import JaggedEmbeddings


class TransformerEmbeddingsDecomposer(nn.Module):
    def __init__(self, e, **kwargs):
        super().__init__()
        self.transformer = nn.Transformer(d_model=2 * e, **kwargs)
        self.projection = nn.Linear(2 * e, e)

    def forward(self, ins, outs):
        embeddings = JaggedEmbeddings.cat_along_embedding(ins, outs)
        sequences = embeddings.to_padded_sequence()
        sequences = sequences.self_attn(self.transformer)
        sequences = sequences.map(self.projection)
        sequences = sequences.flatten()
        return JaggedEmbeddings(sequences, ins.indices_for_each)
