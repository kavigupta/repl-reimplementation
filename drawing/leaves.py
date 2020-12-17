import numpy as np


def null(x, y):
    return np.zeros_like(x, dtype=np.bool)


def circle(x, y):
    return x * x + y * y <= 1


def square(x, y):
    return np.maximum(np.abs(x), np.abs(y)) <= 1


LEAVES = dict(circle=circle, square=square, null=null)
