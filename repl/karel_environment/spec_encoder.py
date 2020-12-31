import numpy as np
import torch
import torch.nn as nn

from ..lgrl import AttentionalSpecEncoder
from ..utils import JaggedEmbeddings, PaddedSequence, PositionalEncoding

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
        self.encoder = KarelTaskEncoder(
            (image_size[0], *image_size[1:]), embedding_size
        )

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

    def __init__(self, image_size, embedding_size):
        super().__init__()

        self.e = embedding_size
        c, w, h = image_size
        assert self.e % 2 == 0

        self.image_size = image_size

        self.input_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=c, out_channels=self.e // 2, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )
        self.output_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=c, out_channels=self.e // 2, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.e, out_channels=self.e, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.e, out_channels=self.e, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.e, out_channels=self.e, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.e, out_channels=self.e, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.e, out_channels=self.e, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.e, out_channels=self.e, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )
        self.positional_encoding = PositionalEncoding(self.e)

    def forward(self, input_grid, output_grid):
        assert len(input_grid.shape) == 4
        assert self.image_size == input_grid.shape[-3:]

        input_enc = self.input_encoder(input_grid)
        output_enc = self.output_encoder(output_grid)
        enc = torch.cat([input_enc, output_enc], 1)
        enc = enc + self.block_1(enc)
        enc = enc + self.block_2(enc)

        enc = enc.view(input_grid.shape[0], -1, self.e)
        enc = enc.transpose(0, 1)
        enc = self.positional_encoding(enc)
        enc = enc.transpose(0, 1)
        return enc
