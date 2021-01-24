from ...repl.dynamics import Dynamics

from robustfill import RobState, interpret


class RobustfillDynamics(Dynamics):
    def partially_execute(self, program, spec):
        out = []
        ins, outs = [p.input for p in spec.pairs], [p.output for p in spec.pairs]
        state = RobState.new(ins, outs)
        for tok in program.tokens:
            state = interpret(tok, state)
        return state

    def program_is_correct(self, program):
        executed = self.partially_execute(program, spec)
        return executed.commited == executed.outputs

    def program_is_complete(self, program):
        return False
