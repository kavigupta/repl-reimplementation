import itertools

import torch
import torch.nn as nn
import numpy as np

from karel_for_synthesis import STATE_DELTAS, compute_deltas, run_deltas

from .standard_karel import GRID_SIZE

from ...repl.policy import Policy
from ...repl.program import SequentialProgram
from ...utils.utils import JaggedEmbeddings, place, PaddedSequence
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


class DeltaLSTM(nn.Module):
    def __init__(self, e=512, channels=64, num_layers=2):
        super().__init__()

        self.backwards_map = {d: i + 1 for i, d in enumerate(STATE_DELTAS)}
        self.forwards_map = {i + 1: d for i, d in enumerate(STATE_DELTAS)}

        self.grid_embedding = KarelTaskEncoder(
            image_size=GRID_SIZE, embedding_size=channels, num_grids=1
        )
        self.grid_embedding_out = nn.Linear(GRID_SIZE[1] * GRID_SIZE[2] * channels, e)
        self.lstm = nn.LSTM(e, e, num_layers=num_layers)
        self.out = nn.Linear(e, len(STATE_DELTAS) + 1)

    def loss(self, embeddings, inputs, inters):
        # TODO actually regress against all, for now just shuffle

        deltas = []
        arrays = []
        for inp, inter in zip(inputs, inters):
            delta = compute_deltas(inp, inter)
            np.random.shuffle(delta)
            array = [inp]
            for d in delta:
                array.append(run_deltas([d], array[-1]))
            arrays.append(array)
            deltas.append([self.backwards_map[d] for d in delta] + [0])

        deltas = PaddedSequence.of(deltas, torch.long, self)
        flat_arrays = [x for array in arrays for x in array]
        flat_arrays = self.run_grid_embedding(flat_arrays)
        grid_embeddings = JaggedEmbeddings.consecutive(
            flat_arrays, [len(x) for x in arrays]
        )
        grid_embeddings.embeddings += grid_embeddings.tile(embeddings)
        grid_embeddings = grid_embeddings.to_padded_sequence()

        output_embeddings = grid_embeddings.sequences.transpose(0, 1)
        output_embeddings, _ = self.lstm(output_embeddings)
        output_embeddings = self.out(output_embeddings)
        output_embeddings = output_embeddings.transpose(0, 1)
        output_embeddings = output_embeddings.log_softmax(-1)

        output_embeddings = output_embeddings.reshape(-1, output_embeddings.shape[-1])
        delta_values = deltas.sequences.reshape(-1)
        selected_delta_values = output_embeddings[
            np.arange(delta_values.shape[0]), delta_values
        ]
        selected_delta_values = selected_delta_values.reshape(*deltas.mask.shape)
        return -(selected_delta_values * deltas.mask).sum() / deltas.mask.sum()

    def run_grid_embedding(self, flat_arrays):
        flat_arrays = self.grid_embedding.run_on_grids(np.array(flat_arrays))
        flat_arrays = flat_arrays.reshape(flat_arrays.shape[0], -1)
        flat_arrays = self.grid_embedding_out(flat_arrays)
        return flat_arrays

    def forward(self, embeddings, inputs, max_length=10):
        arrays = inputs
        done = [0] * len(arrays)
        state = None
        for _ in range(max_length):
            round_embeddings = self.run_grid_embedding(arrays) + embeddings
            out, state = self.lstm(round_embeddings.unsqueeze(1), state)
            out = self.out(out)
            _, operations = out.squeeze(1).max(-1)
            new_arrays = []
            for i, operation, array in zip(itertools.count(), operations, arrays):
                operation = operation.item()
                if done[i] or operation == 0:
                    new_arrays.append(array)
                    done[i] = True
                    continue
                new_arrays.append(run_deltas([self.forwards_map[operation]], array))
            if all(done):
                break
        return arrays


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

        self.delta_lstm = DeltaLSTM(e, channels=channels, **kwargs)

    def common_forward(self, ins, outs):
        lengths = [len(x) for x in ins]
        ins = np.array(list(itertools.chain(*ins)))
        outs = np.array(list(itertools.chain(*outs)))

        return lengths, ins, outs, self.embed_pairs(ins, outs, lengths)

    def loss(self, ins, outs, inters):
        lengths, ins, outs, embeddings = self.common_forward(ins, outs)
        inters = np.array(list(itertools.chain(*inters)))

        return self.delta_lstm.loss(embeddings, ins, inters)

    def forward(self, ins, outs):
        lengths, ins, outs, embeddings = self.common_forward(ins, outs)

        inters = self.delta_lstm(embeddings, ins)
        result = []
        start = 0
        for length in lengths:
            result.append(inters[start : start + length])
            start += length
        return result

    def embed_pairs(self, ins, outs, lengths):
        pair_embeddings = self.sequential_embedding.embed(ins, outs)

        pair_embeddings = self.cross_pollinate_pairs(pair_embeddings, lengths)
        return pair_embeddings + self.postprocess_pairs(pair_embeddings)

    def cross_pollinate_pairs(self, pair_embeddings, lengths):
        pair_embeddings = JaggedEmbeddings.consecutive(pair_embeddings, lengths)
        spec_embeddings = pair_embeddings.max_pool()
        spec_embeddings = spec_embeddings + self.process_spec(spec_embeddings)
        tiled_spec_embeddings = pair_embeddings.tile(spec_embeddings)

        pair_embeddings = pair_embeddings.embeddings + tiled_spec_embeddings
        return pair_embeddings
