from ...repl.dynamics import Dynamics
from ...repl.program import SequentialProgram

from robustfill import RobState, interpret


class RobustfillDynamics(Dynamics):
    def partially_execute(self, program, spec):
        out = []
        ins, outs = [p.input for p in spec.pairs], [p.output for p in spec.pairs]
        state = RobState.new(ins, outs)
        for tok in program.tokens:
            state = interpret(tok, state, strict=False)
        return state

    def program_is_correct(self, program, spec):
        executed = self.partially_execute(program, spec)
        return executed.committed == executed.outputs

    def program_is_complete(self, program, spec):
        return len(program.tokens) >= 20

    @property
    def program_class(self):
        return SequentialProgram
