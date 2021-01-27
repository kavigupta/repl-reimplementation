from ...repl.dynamics import Dynamics
from ...repl.program import SequentialProgram

from robustfill import RobState, interpret


class RobustfillDynamics(Dynamics):
    def partially_execute(self, program, spec):
        out = []
        err_overall = False
        ins, outs = [p.input for p in spec.pairs], [p.output for p in spec.pairs]
        state = RobState.new(ins, outs)
        for tok in program.tokens:
            state, err = interpret(tok, state)
            err_overall = err_overall or err
        return state, err_overall

    def program_is_correct(self, program, spec):
        executed, error = self.partially_execute(program, spec)
        return not error and executed.committed == executed.outputs

    def program_is_complete(self, program, spec):
        _, error = self.partially_execute(program, spec)
        return len(program.tokens) >= 20 or error

    @property
    def program_class(self):
        return SequentialProgram
