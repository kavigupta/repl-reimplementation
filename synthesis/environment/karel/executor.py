import attr

import numpy as np

from karel_for_synthesis import (
    execute,
    ExecutorSyntaxException,
    ExecutorRuntimeException,
)

from ..executor import Executor, ExecutionError
from .standard_karel import read_vocab, number_to_token
from .sequential_karel import toks_to_program


@attr.s
class KarelExecutor(Executor):
    vocab = attr.ib(default={v: k for k, v in read_vocab().items()})

    def preprocess(self, program):
        return [number_to_token(self.vocab, tok) for tok in program]

    def execute(self, program, input):
        program = self.preprocess(program)
        try:
            return execute(program, input, record_trace=False).result
        except ExecutorSyntaxException:
            raise ExecutionError(syntax=True)
        except ExecutorRuntimeException:
            raise ExecutionError(syntax=False)

    def same(self, out1, out2):
        return np.all(out1 == out2)


class KarelSequentialExecutor(KarelExecutor):
    def preprocess(self, program):
        return toks_to_program(program.tokens)
