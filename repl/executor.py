from abc import ABC, abstractmethod

import attr
from permacache import permacache, stable_hash


class Executor(ABC):
    @abstractmethod
    def execute(self, program, input):
        pass

    @abstractmethod
    def same(self, out1, out2):
        pass


@attr.s
class ExecutionError(Exception):
    syntax = attr.ib()


@permacache(
    "repl/executor/evaluate_pair",
    key_function=dict(program=stable_hash, pair=stable_hash),
)
def evaluate_pair(executor, program, pair):
    try:
        if executor.same(executor.execute(program, pair.input), pair.output):
            return "correct"
        else:
            return "incorrect"
    except ExecutionError as e:
        if e.syntax:
            return "syntax-error"
        else:
            return "runtime-error"


TABLE_KEYS = "correct", "incorrect", "syntax-error", "runtime-error", "total"


def evaluate(executor, program, spec):
    outcome = {k: 0 for k in TABLE_KEYS}
    for pair in spec.pairs:
        outcome["total"] += 1
        outcome[evaluate_pair(executor, program, pair)] += 1
    return outcome
