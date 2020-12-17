import attr

import numpy as np

from .leaves import LEAVES
from .transforms import TRANSFORMS
from .operations import OPERATIONS


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
    if isinstance(tree, str):
        return [Item(LEAVES[tree])]
    if tree[0] in TRANSFORMS:
        check_form(3, tree)
        shapes = execute(tree[2], env)
        return [
            Item(
                shape.type,
                TRANSFORMS[tree[0]](evaluate(tree[1], env)) @ shape.transform,
                shape.color,
            )
            for shape in shapes
        ]
    if tree[0] == "color":
        check_form(5, tree)
        color = (
            evaluate(tree[1], env),
            evaluate(tree[2], env),
            evaluate(tree[3], env),
        )
        shapes = execute(tree[4], env)
        return [Item(shape.type, shape.transform, color) for shape in shapes]
    if tree[0] == "combine":
        check_form(3, tree)
        shapes1, shapes2 = execute(tree[1], env), execute(tree[2], env)
        return shapes1 + shapes2
    if tree[0] == "repeat":
        check_form(5, tree)
        lower, upper = evaluate(tree[2], env), evaluate(tree[3], env)
        shape = []
        for i in range(int(lower), int(upper)):
            child_env = env.copy()
            child_env[tree[1]] = i
            shape += execute(tree[4], child_env)
        return shape
    if tree[0] == "if":
        return execute(["ife", *tree[1:], "null"], env)
    if tree[0] == "ife":
        if evaluate(tree[1], env):
            return execute(tree[2], env)
        else:
            return execute(tree[3], env)
    raise SyntaxError(f"Unexpected form: {tree[0]}")


def evaluate(tree, env):
    if tree[0] == "$":
        return env[tree]
    if isinstance(tree, str):
        try:
            return int(tree)
        except ValueError:
            raise SyntaxError(f"Expected expression but received {tree}")
    if tree[0] not in OPERATIONS:
        raise SyntaxError(f"Unrecognized operation: {tree[0]}")

    arguments = [evaluate(b, env) for b in tree[1:]]

    return OPERATIONS[tree[0]](*arguments)
