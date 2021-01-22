import numpy as np
import torch
import torch.nn as nn

from mlozaic.colors import COLORS
from mlozaic.grammar import BACKWARDS_ALPHABET

from ...lgrl import AttentionalSpecEncoder
from ...utils.utils import JaggedEmbeddings, PaddedSequence, PositionalEncoding, place


class MLozaicSpecEncoder(AttentionalSpecEncoder):
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
        super().__init__(embedding_size)
        self.e = embedding_size
        self.io_encoder = MLozaicIOEncoder(
            image_size=image_size, patch_size=patch_size, embedding_size=self.e
        )

    def encode(self, specifications):
        flat_specs = []
        indices = []
        for spec in specifications:
            indices.append(
                list(range(len(flat_specs), len(flat_specs) + len(spec.pairs)))
            )
            flat_specs += spec.pairs

        flat_specs = self.io_encoder(flat_specs)
        src_tgt = flat_specs.sequences.transpose(0, 1)
        # see entire_sequence_forward for note on mask
        encoding = self.encode_attn(
            src_tgt,
            src_tgt,
            src_key_padding_mask=~flat_specs.mask,
            tgt_key_padding_mask=~flat_specs.mask,
        )
        encoding = encoding.transpose(0, 1)
        return JaggedEmbeddings(encoding, indices), flat_specs.mask


class PatchEncoder(nn.Module):
    def __init__(self, image_size, patch_size, embedding_size):
        super().__init__()
        self.w, self.h, self.c = image_size
        self.p = patch_size
        self.e = embedding_size
        assert self.w % self.p == self.h % self.p == 0
        self.patch_embedding = nn.Linear(self.p ** 2 * self.c, self.e)
        self.positional_encoding = PositionalEncoding(self.e)

    def forward(self, images):
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
        images = self.patch_embedding(images)
        images = images.transpose(0, 1)
        images = self.positional_encoding(images)
        images = images.transpose(0, 1)
        return images


class MLozaicIOEncoder(nn.Module):
    """
    Performs the image, variable --> sequence part of the vision transformer.
    """

    def __init__(self, *, image_size, patch_size, embedding_size):
        super().__init__()
        self.e = embedding_size

        assert self.e % 2 == 0

        self.c = image_size[-1]
        self.alphabet_embedding = nn.Embedding(1 + len(BACKWARDS_ALPHABET), self.e // 2)
        self.patch_encoder = PatchEncoder(image_size, patch_size, embedding_size)

    def forward(self, flat_specs):
        images = place(
            self,
            torch.tensor(
                [np.eye(self.c, dtype=np.float32)[spec.output] for spec in flat_specs]
            ),
        )
        embedded_images = self.patch_encoder(images)
        max_num_variables = max(len(f.input) for f in flat_specs)
        variables = []
        paddings = []
        for f in flat_specs:
            for k, v in f.input.items():
                variables += [BACKWARDS_ALPHABET[k] + 1, BACKWARDS_ALPHABET[str(v)] + 1]
            padding_amount = max_num_variables - len(f.input)
            variables += [0, 0] * padding_amount
            paddings.append(padding_amount)
        variables = place(self, torch.tensor(variables))
        embedded_variables = self.alphabet_embedding(variables)
        embedded_variables = embedded_variables.reshape(
            images.shape[0], max_num_variables, self.e
        )
        embeddings = torch.cat([embedded_images, embedded_variables], dim=1)
        mask = place(self, torch.ones(embeddings.shape[:-1], dtype=torch.bool))
        for idx, padding in enumerate(paddings):
            mask[idx, mask.shape[1] - padding :] = False
        return PaddedSequence(embeddings, mask)
