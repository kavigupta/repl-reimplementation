from abc import ABC, abstractmethod

import attr

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlozaic.grammar import BACKWARDS_ALPHABET

from .utils import JaggedEmbeddings, PaddedSequence


class LGRL(nn.Module):
    def __init__(self, io_encoder, *, embedding_size):
        """
        Arguments:
            io_encoder: an IOEncoder object
            alphabet_size: the size of the alphabet to use. This should have
                0 for <s> and 1 for </s>
        """
        super().__init__()
        alphabet_size = 2 + len(BACKWARDS_ALPHABET)

        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(alphabet_size, embedding_size)
        self.io_encoder = io_encoder
        self.decoder_out = nn.Linear(embedding_size, embedding_size)
        self.syntax = nn.LSTMCell(input_size=embedding_size, hidden_size=embedding_size)
        self.syntax_out = nn.Linear(embedding_size, alphabet_size)

    def forward(self, inference_state, choices, normalize_logits=True):
        embedded_tokens = self.embedding(choices)
        new_decoder_state, decoder_out = self.io_encoder.evolve_hidden_state(
            inference_state.decoder_state, inference_state.io_embedding, embedded_tokens
        )
        new_syntax_state = self.syntax(embedded_tokens)

        decoder_out = decoder_out.max_pool()
        syntax_out = self.syntax_out(new_syntax_state[0])
        prediction_vector = decoder_out - torch.exp(syntax_out)
        if normalize_logits:
            prediction_vector = prediction_vector.log_softmax(-1)
        new_state = LGRLInferenceState(
            lgrl=inference_state.lgrl,
            io_embedding=inference_state.io_embedding,
            decoder_state=new_decoder_state,
            syntax_state=new_syntax_state,
        )
        return prediction_vector, new_state

    def begin_inference(self, specs, **kwargs):
        io_embedding = self.io_encoder.encode(specs)
        decoder_state = self.io_encoder.initial_hidden_state(io_embedding)
        syntax_state = torch.zeros(2, len(specs), self.embedding_size)
        state = LGRLInferenceState(
            lgrl=self,
            io_embedding=io_embedding,
            decoder_state=decoder_state,
            syntax_state=syntax_state,
        )
        return state.step(torch.zeros(len(specs), dtype=torch.long), **kwargs)

    def loss(self, specs, programs):
        # add in the end token and pad
        programs = PaddedSequence.of(
            [prog + [1] for prog in programs], dtype=torch.long
        )
        pred, state = self.begin_inference(specs, normalize_logits=False)
        losses = []
        for t in range(programs.L):
            print(t)
            actual = programs.sequences[:, t]
            losses.append(self._xe(pred, actual, programs.mask[:, t]))
            if t < programs.L - 1:
                pred, state = state.step(actual, normalize_logits=False)
        return torch.sum(torch.stack(losses))

    def _xe(self, pred, actual, mask):
        return F.cross_entropy(pred[mask], actual[mask])


class IOEncoder(ABC):
    @abstractmethod
    def encode(self, specs, alphabet_embedding):
        pass

    @abstractmethod
    def initial_hidden_state(self, encoded_io):
        pass

    @abstractmethod
    def evolve_hidden_state(self, hidden_states, encodings, tokens):
        """
        Returns (new_hidden_states, decoder_out) where
            new_hidden_states: can be passed into another call to this function
            encodings: is a JaggedEmbeddings object containing the embeddings
                per input
            tokens: the tokens chosen on the prior step
        """
        pass


@attr.s
class LGRLInferenceState:
    lgrl = attr.ib()
    io_embedding = attr.ib()
    decoder_state = attr.ib()
    syntax_state = attr.ib()

    def step(self, choices, **kwargs):
        return self.lgrl.forward(self, choices, **kwargs)
