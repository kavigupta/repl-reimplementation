import math

import numpy as np
import torch
import torch.nn as nn

from mlozaic.colors import COLORS
from mlozaic.grammar import BACKWARDS_ALPHABET

from ..lgrl import SpecEncoder
from ..utils import JaggedEmbeddings


class MLozaicSpecEncoder(nn.Module, SpecEncoder):
    """
    Rough implementation of a vision transformer: https://arxiv.org/pdf/2010.11929.pdf

    Rather than dealing with multiple lengths, it just pads out extra variables with a special token

    The positional embedding is added to the image tokens but not to the variable tokens, which
        have no concept of position.

    A transformer encoder is used to encode the input, and then another transformer is used
        to autoregress over the previous sequential values and produce an output.

    The result is then transformed into alphabet space and max-pooled.
    """

    def __init__(
        self, *, image_size=(100, 100, len(COLORS)), patch_size=5, embedding_size
    ):
        super().__init__()
        self.e = embedding_size
        self.io_encoder = MLozaicIOEncoder(
            image_size=image_size, patch_size=patch_size, embedding_size=self.e
        )

        self.encode_attn = nn.Transformer(
            self.e, num_encoder_layers=2, num_decoder_layers=0
        )
        self.decode_attn = nn.Transformer(
            self.e, num_encoder_layers=0, num_decoder_layers=2
        )
        self.out = nn.Linear(self.e, 2 + len(BACKWARDS_ALPHABET))

    def encode(self, specifications):
        flat_specs = []
        indices = []
        for spec in specifications:
            indices.append(
                list(range(len(flat_specs), len(flat_specs) + len(spec.pairs)))
            )
            flat_specs += spec.pairs

        flat_specs = self.io_encoder(flat_specs)
        flat_specs = flat_specs.transpose(0, 1)
        encoding = self.encode_attn(flat_specs, flat_specs)
        encoding = encoding.transpose(0, 1)
        return JaggedEmbeddings(encoding, indices)

    def initial_hidden_state(self, encoded_io):
        n, _, e = encoded_io.embeddings.shape
        return torch.zeros(n, 0, e)

    def evolve_hidden_state(self, hidden_states, encodings, tokens):
        hidden_states = torch.cat(
            [encodings.tile(tokens).unsqueeze(1), hidden_states], axis=1
        )
        result = self.decode_attn(
            encodings.embeddings.transpose(0, 1), hidden_states.transpose(0, 1)
        )
        result = self.out(result)
        result = result[-1]
        return hidden_states, encodings.replace(result)


class MLozaicIOEncoder(nn.Module):
    """
    Performs the image, variable --> sequence part of the vision transformer.
    """

    def __init__(self, *, image_size, patch_size, embedding_size):
        super().__init__()
        self.w, self.h, self.c = image_size
        self.p = patch_size
        self.e = embedding_size

        assert self.w % self.p == self.h % self.p == 0
        assert self.e % 2 == 0

        self.patch_embedding = nn.Linear(self.p ** 2 * self.c, self.e)
        self.alphabet_embedding = nn.Embedding(1 + len(BACKWARDS_ALPHABET), self.e // 2)
        self.positional_encoding = PositionalEncoding(self.e)

    def forward(self, flat_specs):
        images = torch.tensor(
            [np.eye(self.c, dtype=np.float32)[spec.output] for spec in flat_specs]
        )
        n, w, h, c = images.shape
        p = self.p
        assert w == self.w and h == self.h and c == self.c
        images = images.reshape(n, w // p, p, h // p, p, c)
        images = images.transpose(2, 3)
        assert images.shape == (n, w // p, h // p, p, p, c)
        images = images.reshape(n, w // p, h // p, p * p * c)
        q = w // p * h // p
        images = images.reshape(n, q, p * p * c)
        # embedded_images : N x Q x E
        embedded_images = self.patch_embedding(images)
        max_num_variables = max(len(f.input) for f in flat_specs)
        variables = []
        for f in flat_specs:
            for k, v in f.input.items():
                variables += [BACKWARDS_ALPHABET[k] + 1, BACKWARDS_ALPHABET[str(v)] + 1]
            variables += [0, 0] * (max_num_variables - len(f.input))
        variables = torch.tensor(variables)
        embedded_variables = self.alphabet_embedding(variables)
        embedded_variables = embedded_variables.reshape(n, max_num_variables, self.e)
        embedded_images = self.positional_encoding(embedded_images)
        embeddings = torch.cat([embedded_images, embedded_variables], dim=1)
        return embeddings


class PositionalEncoding(nn.Module):
    # from the tutorial https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
