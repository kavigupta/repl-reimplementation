import attr

import numpy as np

from robustfill import RobState, interpret

from ..executor import Executor, ExecutionError


@attr.s
class RobustfillExecutor(Executor):
    def execute(self, program, input):
        state = RobState.new([input], [""])
        for tok in program.tokens:
            state, err = interpret(tok, state)
            if err:
                raise ExecutionError(syntax=False)
        return state.committed[0]

    def same(self, out1, out2):
        return out1 == out2
