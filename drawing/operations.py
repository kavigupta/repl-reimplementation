from operator import add, sub, mul, floordiv, mod, lt, le, gt, ge, eq, and_, or_, not_

NUM_OPS = {
    "+": add,
    "-": sub,
    "*": mul,
    "/": floordiv,
    "%": mod,
}

COMPARISONS = {
    "<": lt,
    "<=": le,
    ">": gt,
    ">=": ge,
    "=": eq,
}

BOOL_OPS = {"and": and_, "or": or_, "~": not_}

OPERATIONS = {**NUM_OPS, **COMPARISONS, **BOOL_OPS}
