import attr

import numpy as np
from operator import add, sub, mul, floordiv, mod, lt, le, gt, ge, eq

from .renderer import circle, square
from .transforms import TRANSFORMS


@attr.s
class Item:
    type = attr.ib()
    transform = attr.ib(default=np.eye(3))
    color = attr.ib(default=[0, 0, 0])


def check_form(length, tree):
    assert (
        len(tree) == length
    ), f"Expected {length} terms in form {tree} but got {len(tree)}"


def execute(tree, env):
    if tree == "null":
        return []
    if tree == "circle":
        return [Item(circle)]
    if tree == "square":
        return [Item(square)]
    if not isinstance(tree, list):
        raise SyntaxError(f"unexpected token {tree}")
    if tree[0] in TRANSFORMS:
        check_form(3, tree)
        shapes = execute(tree[2], env)
        return [
            Item(
                shape.type,
                TRANSFORMS[tree[0]](evaluate_number(tree[1], env)) @ shape.transform,
                shape.color,
            )
            for shape in shapes
        ]
    if tree[0] == "color":
        check_form(5, tree)
        color = (
            evaluate_number(tree[1], env),
            evaluate_number(tree[2], env),
            evaluate_number(tree[3], env),
        )
        shapes = execute(tree[4], env)
        return [Item(shape.type, shape.transform, color) for shape in shapes]
    if tree[0] == "combine":
        check_form(3, tree)
        shapes1, shapes2 = execute(tree[1], env), execute(tree[2], env)
        return shapes1 + shapes2
    if tree[0] == "repeat":
        check_form(5, tree)
        lower, upper = evaluate_number(tree[2], env), evaluate_number(tree[3], env)
        shape = []
        for i in range(int(lower), int(upper)):
            child_env = env.copy()
            child_env[tree[1]] = i
            shape += execute(tree[4], child_env)
        return shape
    if tree[0] == "if":
        return execute(["ife", *tree[1:], "null"])
    if tree[0] == "ife":
        if evaluate_cond(tree[1], env):
            return execute(tree[2], env)
        else:
            return execute(tree[3], env)
    raise SyntaxError(f"Unexpected form: {tree[0]}")


OPERATIONS = {
    "+": add,
    "-": sub,
    "*": mul,
    "/": floordiv,
    "%": mod,
}


def evaluate_number(tree, env):
    if tree[0] == "$":
        return env[tree]
    if isinstance(tree, str):
        try:
            return int(tree)
        except ValueError:
            raise SyntaxError(f"Expected numeric expression but received {tree}")
    if tree[0] not in OPERATIONS:
        raise SyntaxError(f"Unrecognized operation: {tree[0]}")
    return OPERATIONS[tree[0]](
        evaluate_number(tree[1], env), evaluate_number(tree[2], env)
    )


COMPARISONS = {
    "<": lt,
    "<=": le,
    ">": gt,
    ">=": ge,
    "=": eq,
}


def evaluate_cond(tree, env):
    if tree[0] in COMPARISONS:
        return COMPARISONS[tree[0]](
            evaluate_number(tree[1], env),
            evaluate_number(tree[2], env),
        )
    if tree[0] == "and":
        return evaluate_cond(tree[1], env) and evaluate_cond(tree[2], env)
    if tree[0] == "or":
        return evaluate_cond(tree[1], env) or evaluate_cond(tree[2], env)
    if tree[0] == "not":
        return not evaluate_cond(tree[1], env)

    raise SyntaxError(f"Unrecognized condition operation {tree[0]}")
