import string

import torch
import torch.nn as nn
import numpy as np

from ...utils.utils import JaggedEmbeddings, PositionalEncoding
from ...utils.distribution import IndependentDistribution
from ...repl.policy import Policy
from ...repl.program import SequentialProgram
from .dynamics import RobustfillDynamics

from robustfill import TOKENS, TOKEN_TO_INDEX

ALPHABET_SIZE = len(string.printable)
NUM_TOKENS = len(TOKENS)


class RobustfillPolicy(nn.Module, Policy):
    def __init__(self, e=512, **kwargs):
        super().__init__()
        self.embedding = RobustfillEmbedding(e=e, **kwargs)
        self.output_layer = nn.Linear(e, NUM_TOKENS)

    @property
    def dynamics(self):
        return RobustfillDynamics()

    @property
    def initial_program_set(self):
        return [SequentialProgram(())]

    def forward(self, states):
        embedding = self.embedding(states)
        predictions = self.output_layer(embedding)
        predictions = predictions.log_softmax(-1)

        def get(token, attr):
            assert attr == "token_idx"
            return TOKEN_TO_INDEX[token]

        return IndependentDistribution(
            lambda token_idx: TOKENS[token_idx],
            dict(token_idx=predictions),
            getattr=get,
        )


class RobustfillValue(nn.Module):
    def __init__(self, e=512, **kwargs):
        super().__init__()
        self.embedding = RobustfillEmbedding(e=e, **kwargs)
        self.fcnet = nn.Sequential(
            nn.Linear(e, e),
            nn.ReLU(),
            nn.Linear(e, e),
            nn.ReLU(),
            nn.Linear(e, 1),
            nn.Sigmoid(),
        )

    def forward(self, states):
        embedding = self.embedding(states)
        predictions = self.fcnet(embedding)
        return predictions.squeeze(-1)


class RobustfillEmbedding(nn.Module):
    def __init__(self, e=512, nhead=8, channels=64, layers=4):
        super().__init__()
        self.alphabet_embedding = nn.Embedding(ALPHABET_SIZE, channels)
        self.masks_embedding = nn.Linear(7, channels)
        self.button_embedding = nn.Embedding(NUM_TOKENS + 1, e)
        self.block_1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.block_2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.spec_proj = nn.Linear(36 * channels, e)
        self.positional_button_encoding = PositionalEncoding(e)
        self.transformer = nn.Transformer(
            e,
            nhead,
            num_encoder_layers=layers // 2,
            num_decoder_layers=layers - layers // 2,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, states):
        all_characters, all_masks, all_buttons = [], [], []
        spec_lengths = []
        button_lengths = []
        for state in states:
            [[partial, _ignored_error]] = state.semantic_partial_programs
            (
                inputs,
                scratch,
                committed,
                outputs,
                masks,
                _,
                rendered_past_buttons,
            ) = partial.to_np(dict(render_scratch="yes", render_past_buttons="yes"))
            spec_lengths.append(len(state.specification.pairs))
            button_lengths.append(rendered_past_buttons.shape[0])
            all_characters.append(
                np.array([inputs, scratch, committed, outputs]).transpose(1, 2, 0)
            )
            all_masks.append(masks.transpose(0, 2, 1))
            all_buttons.append(rendered_past_buttons[:, None])

        all_characters = np.concatenate(all_characters, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        all_buttons = np.concatenate(all_buttons, axis=0)

        spec_embeddings = self.run_embed(self.alphabet_embedding, all_characters)
        all_masks = torch.tensor(all_masks).float().to(self.device)
        spec_embeddings = spec_embeddings + self.masks_embedding(all_masks)
        button_embeddings = self.run_embed(self.button_embedding, all_buttons)

        spec_embeddings = spec_embeddings.transpose(1, 2)
        spec_embeddings = spec_embeddings + self.block_1(spec_embeddings)
        spec_embeddings = spec_embeddings + self.block_2(spec_embeddings)
        spec_embeddings = spec_embeddings.transpose(1, 2)
        spec_embeddings = self.spec_proj(
            spec_embeddings.reshape(spec_embeddings.shape[0], -1)
        )

        button_embeddings = JaggedEmbeddings.consecutive(
            button_embeddings, button_lengths
        )
        spec_embeddings = JaggedEmbeddings.consecutive(spec_embeddings, spec_lengths)

        button_embeddings = button_embeddings.to_padded_sequence()
        button_embeddings = self.positional_button_encoding(button_embeddings)

        overall = button_embeddings.cat(spec_embeddings.to_padded_sequence())

        sequence = overall.sequences.transpose(0, 1)

        output = self.transformer(
            sequence,
            sequence,
            src_key_padding_mask=~overall.mask,
            tgt_key_padding_mask=~overall.mask,
        )

        output, _ = output.max(0)

        return output

    def run_embed(self, embedding, array):
        return embedding(torch.tensor(array).long().to(self.device)).sum(-2)
