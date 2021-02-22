import itertools

import torch
import torch.nn as nn
import numpy as np

from .standard_karel import GRID_SIZE

from ...repl.policy import Policy
from ...repl.program import SequentialProgram
from ...utils.utils import JaggedEmbeddings, place
from ...utils.distribution import IndependentDistribution

from .spec_encoder import KarelTaskEncoder, karel_block
from .sequential_dynamics import KarelDynamics
from .sequential_karel import TOKENS, TOKEN_TO_INDEX


class KarelSequentialEmbedding(nn.Module):
    def __init__(self, channels=64, e=512):
        super().__init__()
        self.embedding = KarelTaskEncoder(
            image_size=GRID_SIZE, embedding_size=channels, num_grids=2
        )
        self.project = nn.Linear(channels * GRID_SIZE[1] * GRID_SIZE[2], e)

    def embed(self, inputs, outputs):
        embedding = self.embedding.run_on_grids(inputs, outputs)
        embedding = embedding.reshape(embedding.shape[0], -1)
        embedding = self.project(embedding)
        return embedding

    def forward(self, states):
        outs, currents = [], []
        lengths = []
        for s in states:
            pairs = s.specification.pairs
            [current] = s.semantic_partial_programs
            lengths.append(len(pairs))
            outs += [p.output for p in pairs]
            currents += current

        currents, outs = np.array(currents), np.array(outs)
        embedding = self.embed(currents, outs)
        embedding = JaggedEmbeddings.consecutive(embedding, lengths).max_pool()
        return embedding


class KarelSequentialPolicy(nn.Module, Policy):
    def __init__(self, channels=64, e=512, *, max_length, **kwargs):
        super().__init__()
        self.sequential_embedding = KarelSequentialEmbedding(channels, e, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(e, e),
            nn.ReLU(),
            nn.Linear(e, e),
            nn.ReLU(),
            nn.Linear(e, e),
            nn.ReLU(),
            nn.Linear(e, len(TOKENS)),
        )

        self._max_length = max_length

    @property
    def dynamics(self):
        return KarelDynamics(self._max_length)

    @property
    def initial_program_set(self):
        return [SequentialProgram(())]

    def forward(self, states):
        embedding = self.sequential_embedding(states)
        predictions = self.net(embedding)
        predictions = predictions.log_softmax(-1)

        def get(token, attr):
            assert attr == "token_idx"
            return TOKEN_TO_INDEX[token]

        return IndependentDistribution(
            lambda token_idx: TOKENS[token_idx],
            dict(token_idx=predictions),
            getattr=get,
        )

    def embedding_net(self, pairs):
        inputs = np.array([p.input for ps in pairs for p in ps])
        outputs = np.array([p.output for ps in pairs for p in ps])
        embeddings = self.sequential_embedding.embed(inputs, outputs)
        return JaggedEmbeddings.consecutive(embeddings, [len(x) for x in pairs])


class KarelSequentialValue(nn.Module):
    def __init__(self, policy, e=512):
        super().__init__()
        self.sequential_embedding = policy.sequential_embedding
        self.network = nn.Sequential(
            nn.Linear(e, e),
            nn.ReLU(),
            nn.Linear(e, e),
            nn.ReLU(),
            nn.Linear(e, e),
            nn.ReLU(),
            nn.Linear(e, 1),
            nn.Sigmoid(),
        )

    def forward(self, states):
        embedding = self.sequential_embedding(states)
        return self.network(embedding)


class KarelSequentialDecomposer(nn.Module):
    def __init__(self, e=512, channels=64, **kwargs):
        super().__init__()
        self.sequential_embedding = KarelSequentialEmbedding(channels, e, **kwargs)
        self.process_spec = nn.Sequential(
            nn.Linear(e, e),
            nn.ReLU(),
            nn.Linear(e, e),
            nn.ReLU(),
            nn.Linear(e, e),
        )
        self.postprocess_pairs = nn.Sequential(
            nn.Linear(e, e),
            nn.ReLU(),
            nn.Linear(e, e),
            nn.ReLU(),
            nn.Linear(e, e),
        )
        self.project_to_grids = nn.Linear(e, GRID_SIZE[1] * GRID_SIZE[2] * channels)

        self.residual_in_outs = nn.Conv2d(GRID_SIZE[0] * 2, channels, 1)

        self.block_1 = karel_block(channels)
        self.block_2 = karel_block(channels)

        self.output_layer = nn.Conv2d(channels, GRID_SIZE[0], 1)

    def forward(self, ins, outs):
        lengths = [len(x) for x in ins]
        ins = np.array(list(itertools.chain(*ins)))
        outs = np.array(list(itertools.chain(*outs)))

        pairs = self.embed_pairs(ins, outs, lengths)
        in_outs = self.embed_in_outs(ins, outs)

        embeddings = pairs + in_outs

        embeddings = embeddings + self.block_1(embeddings)
        embeddings = embeddings + self.block_2(embeddings)

        embeddings = self.output_layer(embeddings)

        return JaggedEmbeddings.consecutive(embeddings, lengths)

    def cross_pollinate_pairs(self, pair_embeddings, lengths):
        pair_embeddings = JaggedEmbeddings.consecutive(pair_embeddings, lengths)
        spec_embeddings = pair_embeddings.max_pool()
        spec_embeddings = spec_embeddings + self.process_spec(spec_embeddings)
        tiled_spec_embeddings = pair_embeddings.tile(spec_embeddings)

        pair_embeddings = pair_embeddings.embeddings + tiled_spec_embeddings
        return pair_embeddings

    def embed_pairs(self, ins, outs, lengths):
        pair_embeddings = self.sequential_embedding.embed(ins, outs)

        pair_embeddings = self.cross_pollinate_pairs(pair_embeddings, lengths)
        pair_embeddings = pair_embeddings + self.postprocess_pairs(pair_embeddings)

        grid_embeddings = self.project_to_grids(pair_embeddings)
        grid_embeddings = grid_embeddings.reshape(
            grid_embeddings.shape[0], -1, GRID_SIZE[1], GRID_SIZE[2]
        )
        return grid_embeddings

    def embed_in_outs(self, ins, outs):
        in_outs = np.concatenate([ins, outs], axis=1)
        in_outs = place(self, torch.tensor(in_outs))
        return self.residual_in_outs(in_outs)
