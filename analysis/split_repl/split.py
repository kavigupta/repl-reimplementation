import hashlib

import numpy as np

from synthesis.repl.program import SequentialProgram
from synthesis.environment.spec import Specification, Pair
from synthesis.environment.robustfill.executor import RobustfillExecutor
from synthesis.environment.executor import evaluate
from synthesis.search.search import infer
from synthesis.search.particle_filter import ReplParticleFilter


def split_prog(p, *, min_segments, max_segments):
    seed = int(hashlib.sha256(str(p).encode("utf-8")).hexdigest(), 16) % 2 ** 32
    rng = np.random.RandomState(seed)
    progs = [[]]
    for tok in p.tokens:
        progs[-1].append(tok)
        if tok == "Commit" and rng.rand():
            progs.append([])
    if progs[-1] == []:
        progs.pop()

    num_splits = rng.choice(max_segments - min_segments + 1) + min_segments - 1

    split_points = list(range(1, len(progs)))
    num_splits = min(num_splits, len(split_points))

    chosen_split_points = set(rng.choice(split_points, size=num_splits, replace=False))

    new_progs = [[]]
    for i, prog in enumerate(progs):
        if i in chosen_split_points:
            new_progs.append([])
        new_progs[-1] += prog

    return [SequentialProgram(tuple(p)) for p in new_progs]


def split_spec(p, spec, **split_args):
    sps = split_prog(p, **split_args)

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
    return split_specs, sps


def split_effectiveness(
    policy, value, prog, spec, n, split_strategy="proportional", **split_args
):
    sss, sps = split_spec(prog, spec, **split_args)
    overall_result = evaluate(
        RobustfillExecutor(),
        infer(ReplParticleFilter(n), (policy, value), spec)[0][1],
        spec,
        use_test=True,
    )
    overall_result = overall_result["correct"] == overall_result["total"]
    split_results = []
    for ss, sp in zip(sss, sps):

        n_sub = {
            "proportional": int(n * len(sp.tokens) / len(prog.tokens)),
            "same": n // len(sps),
        }[split_strategy]

        result = evaluate(
            RobustfillExecutor(),
            infer(ReplParticleFilter(n_sub), (policy, value), ss,)[
                0
            ][1],
            ss,
            use_test=True,
        )
        split_results.append(result["correct"] == result["total"])
    return overall_result, split_results
