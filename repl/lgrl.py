from abc import ABC, abstractmethod

import attr

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlozaic.grammar import BACKWARDS_ALPHABET

from .utils import JaggedEmbeddings, PaddedSequence


class LGRL(nn.Module):
    def __init__(self, spec_encoder, *, embedding_size):
        """
        Implementation of LGRL: https://arxiv.org/pdf/1805.04276.pdf

        One subtlety is that we allow for non-recurrent specification encoders. By using a conv + LSTM
            this should be exactly as described in the paper.

        Arguments:
            spec_encoder: an SpecEncoder object
            alphabet_size: the size of the alphabet to use. This should have
                0 for <s> and 1 for </s>
        """
        super().__init__()
        alphabet_size = 2 + len(BACKWARDS_ALPHABET)

        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(alphabet_size, embedding_size)
        self.spec_encoder = spec_encoder
        self.decoder_out = nn.Linear(embedding_size, embedding_size)
        self.syntax = nn.LSTM(input_size=embedding_size, hidden_size=embedding_size)
        self.syntax_out = nn.Linear(embedding_size, alphabet_size)

    def entire_sequence_forward(self, specs, programs, normalize_logits=True):
        inputs = PaddedSequence.of([[0] + prog for prog in programs], dtype=torch.long)
        outputs = PaddedSequence.of([prog + [1] for prog in programs], dtype=torch.long)

        embedded_inputs = inputs.map(self.embedding)
        spec_embedding = self.spec_encoder.encode(specs)
        decoder_out = self.spec_encoder.entire_sequence_forward(
            spec_embedding, embedded_inputs
        )
        decoder_out = decoder_out.max_pool()
        syntax_out, _ = self.syntax(embedded_inputs.sequences.transpose(0, 1))
        syntax_out = syntax_out.transpose(0, 1)
        syntax_out = self.syntax_out(syntax_out)
        prediction_vector = decoder_out - torch.exp(syntax_out)
        if normalize_logits:
            prediction_vector = prediction_vector.log_softmax(-1)
        return prediction_vector, outputs

    def resample_state(self, state, new_indices):
        """
        Update the given state with the new indices.
        """
        new_spec_embedding, new_decoder_state = self.spec_encoder.resample_state(
            state.spec_embedding, state.decoder_state, new_indices
        )
        new_syntax_state = [s[:, new_indices] for s in state.syntax_state]
        return LGRLInferenceState(
            lgrl=state.lgrl,
            spec_embedding=new_spec_embedding,
            decoder_state=new_decoder_state,
            syntax_state=new_syntax_state,
        )

    def forward(self, inference_state, choice, normalize_logits=True):
        if choice == 1:
            return None, None  # </s> token reached
        embedded_tokens = self.embedding(torch.tensor([choice]))
        new_decoder_state, decoder_out = self.spec_encoder.evolve_hidden_state(
            inference_state.decoder_state,
            inference_state.spec_embedding,
            embedded_tokens,
        )
        out, new_syntax_state = self.syntax(
            embedded_tokens.unsqueeze(0), inference_state.syntax_state
        )

        decoder_out = decoder_out.max_pool()
        syntax_out = self.syntax_out(out)
        prediction_vector = decoder_out - torch.exp(syntax_out)
        prediction_vector = prediction_vector.squeeze(0)[-1]
        if normalize_logits:
            prediction_vector = prediction_vector.log_softmax(-1)
        new_state = LGRLInferenceState(
            lgrl=inference_state.lgrl,
            spec_embedding=inference_state.spec_embedding,
            decoder_state=new_decoder_state,
            syntax_state=new_syntax_state,
        )
        return new_state, prediction_vector

    def begin_inference(self, spec, **kwargs):
        spec_embedding = self.spec_encoder.encode([spec])
        decoder_state = self.spec_encoder.initial_hidden_state(spec_embedding)
        syntax_state = [torch.zeros(1, 1, self.embedding_size) for _ in range(2)]
        state = LGRLInferenceState(
            lgrl=self,
            spec_embedding=spec_embedding,
            decoder_state=decoder_state,
            syntax_state=syntax_state,
        )
        return state.step(0, **kwargs)

    def loss(self, specs, programs):
        # add in the end token and pad
        predictions, outputs = self.entire_sequence_forward(
            specs, programs, normalize_logits=False
        )
        return self._xe(predictions, outputs.sequences, outputs.mask)

    def _xe(self, pred, actual, mask):
        return F.cross_entropy(pred[mask], actual[mask])


class SpecEncoder(ABC):
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

    @abstractmethod
    def entire_sequence_forward(self, encodings, tokens):
        """
        This could be implemented in terms of initial_hidden_state/evolve_hidden_state
        with a loop but can often be implemented more efficiently in a single pass.

        This can really only be used for training.

        Arguments
            encodings: as produced by self.encode
             tokens: a PaddedSequence object that represents the original sequence

        Output
            a PaddedSequence object representing the inferences on every element in
                the sequence.
        """
        pass

    @abstractmethod
    def resample_state(self, spec_embedding, decoder_state, new_indices):
        """
        Resample the spec embedding and the decoder state with the given indices
        """
        pass


class AttentionalSpecEncoder(nn.Module, SpecEncoder):
    def __init__(self, embedding_size, *, encoder_layers=2, decoder_layers=2):
        super().__init__()

        self.e = embedding_size

        self.encode_attn = nn.Transformer(
            self.e, num_encoder_layers=encoder_layers, num_decoder_layers=1
        )
        self.decode_attn = nn.Transformer(
            self.e, num_encoder_layers=1, num_decoder_layers=decoder_layers
        )
        self.out = nn.Linear(self.e, 2 + len(BACKWARDS_ALPHABET))

    def initial_hidden_state(self, encoded_io):
        n = len(encoded_io[0].indices_for_each)
        return torch.zeros(n, 0, self.e)

    def evolve_hidden_state(self, hidden_states, encodings, tokens):
        hidden_states = torch.cat([tokens.unsqueeze(1), hidden_states], axis=1)
        return hidden_states, self.entire_sequence_forward(
            encodings,
            PaddedSequence(
                hidden_states,
                torch.ones(
                    (hidden_states.shape[0], hidden_states.shape[1]), dtype=torch.bool
                ),
            ),
        )

    def entire_sequence_forward(self, encodings, tokens):
        encodings, encodings_mask = encodings
        tiled_tokens = encodings.tile(tokens.sequences)
        source = encodings.embeddings.transpose(0, 1)
        target = tiled_tokens.transpose(0, 1)
        # NOTE: the mask for the key padding is inverted. From the docs:
        #   "positions with the value of True will be ignored while the
        #       position with the value of False will be unchanged"
        result = self.decode_attn(
            source,
            target,
            tgt_mask=self.decode_attn.generate_square_subsequent_mask(target.shape[0]),
            src_key_padding_mask=~encodings_mask,
            tgt_key_padding_mask=~encodings.tile(tokens.mask),
        )
        result = result.transpose(0, 1)
        result = self.out(result)
        return encodings.replace(result)

    def resample_state(self, spec_embedding, decoder_state, new_indices):
        """
        Resample the spec embedding and the decoder state with the given indices
        """
        emb, mask = spec_embedding
        mask = JaggedEmbeddings(mask, emb.indices_for_each)[new_indices].embeddings
        emb = emb[new_indices]
        spec_embedding = emb, mask
        decoder_state = decoder_state[new_indices]
        return spec_embedding, decoder_state


@attr.s
class LGRLInferenceState:
    lgrl = attr.ib()
    spec_embedding = attr.ib()
    decoder_state = attr.ib()
    syntax_state = attr.ib()

    def step(self, choices, **kwargs):
        return self.lgrl.forward(self, choices, **kwargs)

    def resample(self, new_indices):
        return self.lgrl.resample_state(self, new_indices)
