import attr

from ..environment.spec import Specification, Pair
from ..repl.program import SequentialProgram


@attr.s
class OracleDecomposer:
    get_prefix = attr.ib()
    dyn = attr.ib()

    def _intermediates(self, prefix, pairs):
        inters = [self.dyn.partially_execute_pair(prefix, pair) for pair in pairs]
        firsts = [Pair(p.input, inter) for p, inter in zip(pairs, inters)]
        seconds = [Pair(inter, p.output) for p, inter in zip(pairs, inters)]
        return firsts, seconds

    def split_spec(self, spec, p):
        prefix = self.get_prefix(p)
        firsts, seconds = self._intermediates(prefix, spec.pairs)
        firsts_test, seconds_test = self._intermediates(prefix, spec.test_pairs)
        first_spec = Specification(firsts, firsts_test)
        second_spec = Specification(seconds, seconds_test)
        return first_spec, second_spec


def half_split_sequential_program(p):
    return SequentialProgram(p.tokens[: len(p.tokens) // 2])
