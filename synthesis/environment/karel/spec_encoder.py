import numpy as np
import torch
import torch.nn as nn

from ...lgrl import AttentionalSpecEncoder, RecurrentSpecEncoder
from ...utils.utils import JaggedEmbeddings, PaddedSequence, PositionalEncoding, place

from .standard_karel import GRID_SIZE


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

    def __init__(self, *, image_size=GRID_SIZE, embedding_size, **kwargs):
        super().__init__(embedding_size, **kwargs)
        self.e = embedding_size
        self.encoder = KarelTaskEncoder(
            (image_size[0], *image_size[1:]), embedding_size
        )
        self.positional_encoding = PositionalEncoding(self.e)

    def encode(self, specifications):
        enc, indices = self.encoder(specifications)
        enc = enc.view(enc.shape[0], -1, self.e)
        enc = enc.transpose(0, 1)
        enc = self.positional_encoding(enc)
        # see entire_sequence_forward for note on mask
        enc = self.encode_attn(enc, enc)
        enc = enc.transpose(0, 1)
        mask = place(self, torch.ones(enc.shape[:-1], dtype=torch.bool))
        return JaggedEmbeddings(enc, indices), mask


class KarelRecurrentSpecEncoder(RecurrentSpecEncoder):
    def __init__(self, *, image_size=GRID_SIZE, embedding_size, channels=64):
        super().__init__(embedding_size)
        self.e = embedding_size
        self.encoder = KarelTaskEncoder((image_size[0], *image_size[1:]), channels)
        _, w, h = image_size
        self.out = nn.Linear(w * h * channels, self.e * 2)

    def encode(self, specifications):
        enc, indices = self.encoder(specifications)
        enc = enc.view(enc.shape[0], -1)
        enc = self.out(enc)
        return JaggedEmbeddings(enc, indices)


class KarelTaskEncoder(nn.Module):
    """Implements the encoder from:

    Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis
    https://openreview.net/forum?id=H1Xw62kRZ

    Instead of embedding the data, it just serializes it instead.
    """

    def __init__(self, image_size, embedding_size, num_grids=2):
        super().__init__()

        self.e = embedding_size
        self.num_grids = num_grids
        c, w, h = image_size
        assert self.e % 2 == 0

        self.image_size = image_size

        self.grids_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=num_grids * c, out_channels=self.e, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )

        self.block_1 = karel_block(self.e)
        self.block_2 = karel_block(self.e)

    def forward(self, specifications):

        flat_specs = []
        indices = []
        for spec in specifications:
            indices.append(
                list(range(len(flat_specs), len(flat_specs) + len(spec.pairs)))
            )
            flat_specs += spec.pairs

        input_grid = [fs.input for fs in flat_specs]
        output_grid = [fs.output for fs in flat_specs]

        return self.run_on_grids(input_grid, output_grid)

    def run_on_grids(self, *grids):
        assert set(grid.shape[-3:] for grid in grids) == {self.image_size}
        assert len(grids) == self.num_grids

        grids = np.concatenate(grids, axis=-3)
        assert len(grids.shape) == 4

        grids = place(self, torch.tensor(grids))

        enc = self.grids_encoder(grids)

        enc = enc + self.block_1(enc)
        enc = enc + self.block_2(enc)

        return enc


def karel_block(e):
    return nn.Sequential(
        nn.Conv2d(in_channels=e, out_channels=e, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=e, out_channels=e, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=e, out_channels=e, kernel_size=3, padding=1),
        nn.ReLU(),
    )
