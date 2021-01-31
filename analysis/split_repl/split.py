import hashlib

import numpy as np

from synthesis.repl.program import SequentialProgram
from synthesis.environment.spec import Specification, Pair
from synthesis.environment.robustfill.executor import RobustfillExecutor
from synthesis.environment.executor import evaluate
from synthesis.search.search import infer
from synthesis.search.particle_filter import ReplParticleFilter


def chunk_prog(p):
    progs = [[]]
    for tok in p.tokens:
        progs[-1].append(tok)
        if tok == "Commit":
            progs.append([])
    if progs[-1] == []:
        progs.pop()
    return progs


def chunks_to_split(progs, split_points):
    split_points = set(split_points)
    new_progs = [[]]
    for i, prog in enumerate(progs):
        if i in split_points:
            new_progs.append([])
        new_progs[-1] += prog

    return [SequentialProgram(tuple(p)) for p in new_progs]


def all_splits_prog(p):
    progs = chunk_prog(p)

    if len(progs) == 1:
        return [chunks_to_split(progs, set())]
    else:
        return [chunks_to_split(progs, {i}) for i in range(1, len(progs))]


def split_spec(sps, spec):

    split_specs = []
    inputs, inputs_test = [p.input for p in spec.pairs], [
        p.input for p in spec.test_pairs
    ]
    for sp in sps:
        exc = lambda u: RobustfillExecutor().execute(sp, u)
        outputs, outputs_test = list(map(exc, inputs)), list(map(exc, inputs_test))
        split_specs.append(
            Specification(
                [Pair(i, o) for i, o in zip(inputs, outputs)],
                [Pair(i, o) for i, o in zip(inputs_test, outputs_test)],
            )
        )
    return split_specs


def split_effectiveness(
    policy,
    value,
    prog,
    spec,
    n,
    split_strategy="proportional",
    choice_strategy="random",
    seed=0,
    evaluate_overall=True,
):
    def _eval(n, s):
        _, p = infer(ReplParticleFilter(n, seed=seed), (policy, value), s)[0]
        result = evaluate(RobustfillExecutor(), p, s, use_test=True)
        return result["correct"] == result["total"]

    sps = all_splits_prog(prog)

    if choice_strategy == "random":
        seed = int(hashlib.sha256(str(prog).encode("utf-8")).hexdigest(), 16) % 2 ** 32
        rng = np.random.RandomState(seed)
        sps = [sps[rng.choice(len(sps))]]
    elif choice_strategy == "all":
        pass
    elif isinstance(choice_strategy, list) and len(choice_strategy) == len(sps):
        sps = [sps[i] for i, b in enumerate(choice_strategy) if b]
    else:
        raise RuntimeError(f"Choice strategy {choice_strategy}")

    overall_result = _eval(n, spec) if evaluate_overall else None
    split_results = []

    for sps in sps:
        split_results_current = []
        sss = split_spec(sps, spec)

        for ss, sp in zip(sss, sps):

            n_sub = {
                "proportional": int(n * len(sp.tokens) / len(prog.tokens)),
                "same": n // len(sps),
            }[split_strategy]

            split_results_current.append(_eval(n_sub, ss))

        split_results.append(split_results_current)
    return overall_result, split_results
