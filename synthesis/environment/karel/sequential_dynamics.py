import attr

import numpy as np

from karel_for_synthesis import execute, ExecutorRuntimeException

from ...repl.dynamics import Dynamics
from ...repl.program import SequentialProgram
from .sequential_karel import toks_to_program


class KarelDynamics(Dynamics):
    size = attr.ib()

    def partially_execute(self, program, spec):
        return super().partially_execute(program, spec)

    def partially_execute_pair(self, program, pair):
        try:
            return execute(toks_to_program(program.tokens), pair.input).result
        except ExecutorRuntimeException:
            return pair.input

    def program_is_correct(self, program, spec):
        executed = self.partially_execute(program, spec)
        return np.all(
            [np.all(e == pair.output) for e, pair in zip(executed, spec.pairs)]
        )

    def program_is_complete(self, program, spec):
        return len(program.tokens) == self.size

    @property
    def program_class(self):
        return SequentialProgram
