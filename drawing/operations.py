from operator import add, sub, mul, floordiv, mod, lt, le, gt, ge, eq

OPERATIONS = {
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
