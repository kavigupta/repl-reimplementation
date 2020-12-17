from operator import add, sub, mul, floordiv, mod, lt, le, gt, ge, eq

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
