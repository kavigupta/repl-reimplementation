import numpy as np
import torch
import torch.nn as nn

from ..lgrl import AttentionalSpecEncoder
from ..utils import JaggedEmbeddings, PaddedSequence

from .load_karel_environment import GRID_SIZE


class KarelSpecEncoder(AttentionalSpecEncoder):
    """
    Rough implementation of a vision transformer: https://arxiv.org/pdf/2010.11929.pdf

    Rather than dealing with multiple lengths, it just pads out extra variables with a special token

    The positional embedding is added to the image tokens but not to the variable tokens, which
        have no concept of position.

    A transformer encoder is used to encode the input, and then another transformer is used
        to autoregress over the previous sequential values and produce an output.

    The result is then transformed into alphabet space and max-pooled.
    """

    def __init__(self, *, image_size=GRID_SIZE, embedding_size):
        super().__init__(embedding_size)
        self.e = embedding_size
        self.encoder = KarelTaskEncoder((image_size[0], *image_size[1:]))

    def encode(self, specifications):
        flat_specs = []
        indices = []
        for spec in specifications:
            indices.append(
                list(range(len(flat_specs), len(flat_specs) + len(spec.pairs)))
            )
            flat_specs += spec.pairs

        inputs = torch.tensor([fs.input for fs in flat_specs])
        outputs = torch.tensor([fs.output for fs in flat_specs])
        flat_specs = self.encoder(inputs, outputs)
        src_tgt = flat_specs.transpose(0, 1)
        # see entire_sequence_forward for note on mask
        encoding = self.encode_attn(src_tgt, src_tgt)
        encoding = encoding.transpose(0, 1)
        mask = torch.ones(encoding.shape[:-1], dtype=torch.bool)
        return JaggedEmbeddings(encoding, indices), mask


class KarelTaskEncoder(nn.Module):
    """Implements the encoder from:

    Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis
    https://openreview.net/forum?id=H1Xw62kRZ

    Instead of embedding the data, it just serializes it instead.
    """

    def __init__(self, image_size):
        super().__init__()

        c, w, h = image_size

        self.image_size = image_size

        self.input_encoder = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.output_encoder = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, input_grid, output_grid):
        assert self.image_size == input_grid.shape[-3:]
        batch_dims = input_grid.shape[:-3]
        input_grid = input_grid.contiguous().view(-1, *self.image_size)
        output_grid = output_grid.contiguous().view(-1, *self.image_size)

        input_enc = self.input_encoder(input_grid)
        output_enc = self.output_encoder(output_grid)
        enc = torch.cat([input_enc, output_enc], 1)
        enc = enc + self.block_1(enc)
        enc = enc + self.block_2(enc)

        enc = enc.view(*(batch_dims + (-1, 64)))
        return enc
