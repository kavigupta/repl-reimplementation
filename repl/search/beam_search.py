# copied from seq2struct project

import copy
import operator
import collections
import math

import attr


import torch


@attr.s
class Hypothesis:
    inference_state = attr.ib()
    next_choices = attr.ib()
    score = attr.ib(default=0)

    choice_history = attr.ib(factory=list)
    score_history = attr.ib(factory=list)


@attr.s
class Candidate:
    hypothesis = attr.ib()
    choice = attr.ib()
    choice_score = attr.ib()
    candidate_score = attr.ib()


def assume_single(all_equivalent_state_scores, candidate_score):
    assert len(all_equivalent_state_scores) == 1
    return all_equivalent_state_scores[0]


def maximal_score(all_equivalent_state_scores, candidate_score):
    return max(all_equivalent_state_scores)


def mean_score(all_equivalent_state_scores, candidate_score):
    return mean(all_equivalent_state_scores)


def mean_probability(all_equivalent_state_scores, candidate_score):
    return math.log(mean([math.exp(x) for x in all_equivalent_state_scores]))


def mean(lst):
    return sum(lst) / len(lst)


SCORE_AGGREGATORS = {
    "assume_single": assume_single,
    "maximal_score": maximal_score,
    "mean_score": mean_score,
    "mean_probability": mean_probability,
}


def beam_search(model, orig_item, beam_size, max_steps):
    return multi_beam_search(
        [model],
        [orig_item],
        beam_size,
        max_steps,
        agg_score=assume_single,
    )


def multi_beam_search(
    models,
    orig_items,
    beam_size,
    max_steps,
    agg_score,
):
    """
    Runs a multi-beam-search on the given models and inputs.

        models: models to use
        orig_items: inputs to use
        beam_size: maximal number of inferences to have at any point
        max_steps: maximum number of steps of beam search before giving up and returning fewer beams
        agg_score: how to calculate the aggregate score of a given candidate. Called as
            agg_score(all_equivalent_state_scores, candidate_score) where all_equivalent_state_scores
            includes candidate_score

    The options that involve combination assume that unique choices lead to unique
        states, which is fine for *2seq models.
    """
    num_model_input_pairs = len(models) * len(orig_items)
    total_num_beams = num_model_input_pairs * beam_size
    # beam[choice history as a tuple] = a list of Hypotheses that correspond to that sequence of choices
    beam = {
        (): [
            Hypothesis(*model.begin_inference(orig_item))
            for model in models
            for orig_item in orig_items
        ]
    }
    # list of finished beams
    finished = []

    for step in range(max_steps):
        # Check if all beams are finished
        if len(finished) == total_num_beams:
            break

        candidates_map = collections.defaultdict(list)

        # For each hypothesis, get possible expansions
        # Score each expansion
        for prefix, hyps_for_prefix in beam.items():
            # TODO: Assign zero or -inf probability to missing choices instead?
            for hyp in hyps_for_prefix:
                for choice, choice_score in enumerate(hyp.next_choices):
                    assert choice_score.shape == ()
                    candidate = Candidate(
                        hyp,
                        choice,
                        choice_score.item(),
                        hyp.score + choice_score.item(),
                    )
                    candidates_map[prefix + (choice,)].append(candidate)

        scores_by_prefix = {
            prefix: tuple(
                candidate.candidate_score for candidate in candidates_for_prefix
            )
            for prefix, candidates_for_prefix in candidates_map.items()
        }

        # annotated_candidiates[i] = (score, prefix, candidate)
        annotated_candidiates = [
            (
                agg_score(scores_by_prefix[prefix], candidate.candidate_score),
                prefix,
                candidate,
            )
            for prefix, candidates in sorted(candidates_map.items())
            for candidate in candidates
        ]

        # Keep the top K expansions
        annotated_candidiates.sort(key=operator.itemgetter(0), reverse=True)
        annotated_candidiates = annotated_candidiates[: total_num_beams - len(finished)]

        # Create the new hypotheses from the expansions
        beam = collections.defaultdict(list)
        for agg_score_value, prefix, candidate in annotated_candidiates:
            (
                next_inference_state,
                next_choices,
            ) = candidate.hypothesis.inference_state.step(candidate.choice)
            assert len(next_choices.shape) == 1
            if next_choices is None:
                finished.append(
                    Hypothesis(
                        next_inference_state,
                        None,
                        agg_score_value,
                        candidate.hypothesis.choice_history + [candidate.choice],
                        candidate.hypothesis.score_history + [candidate.choice_score],
                    )
                )
            else:
                beam[prefix].append(
                    Hypothesis(
                        next_inference_state,
                        next_choices,
                        agg_score_value,
                        candidate.hypothesis.choice_history + [candidate.choice],
                        candidate.hypothesis.score_history + [candidate.choice_score],
                    )
                )

    finished.sort(key=operator.attrgetter("score"), reverse=True)
    return finished
