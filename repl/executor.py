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


ERROR_KEYS = "syntax-error", "runtime-error", "incorrect"
TABLE_KEYS = "correct", *ERROR_KEYS, "total"


def evaluate(executor, program, spec, *, use_test):
    outcome = {k: 0 for k in TABLE_KEYS}
    outcome["each"] = []
    pairs = spec.pairs
    if use_test:
        pairs = pairs + spec.test_pairs
    for pair in pairs:
        outcome["total"] += 1
        res = evaluate_pair(executor, program, pair)
        outcome[res] += 1
        outcome["each"] += [res]
    return outcome


def evaluate_on_dataset(executor, specs, outputs, pbar=lambda x: x, **kwargs):
    results = {k: 0 for k in TABLE_KEYS}
    for output, spec in pbar(list(zip(outputs, specs))):
        _, program = output[0]
        outcome = evaluate(executor, program, spec, **kwargs)
        results["total"] += 1
        for err in ERROR_KEYS:
            if outcome[err] > 0:
                results[err] += 1
                break
        else:
            results["correct"] += 1
    return results
