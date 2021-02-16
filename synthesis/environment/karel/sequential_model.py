import torch
import torch.nn as nn
import numpy as np

from .standard_karel import GRID_SIZE

from ...repl.policy import Policy
from ...repl.program import SequentialProgram
from ...utils.utils import JaggedEmbeddings
from ...utils.distribution import IndependentDistribution

from .spec_encoder import KarelTaskEncoder
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
    def __init__(self, channels=64, e=512, **kwargs):
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

    @property
    def dynamics(self):
        return KarelDynamics()

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
