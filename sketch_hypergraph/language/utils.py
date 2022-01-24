import numpy as np


def fit_circle(points):
    xs = np.array([p.x for p in points])
    ys = np.array([p.y for p in points])
    # 2ax + 2by + c = x^2 + y^2
    # (x - a)^2 + (x - b)^2 = c + a^2 + b^2
    A = np.array([2 * xs, 2 * ys, np.ones_like(xs)]).T
    b = xs ** 2 + ys ** 2
    (a, b, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r = (c + a ** 2 + b ** 2) ** 0.5
    circle = (a, b), r
    residuals = (xs - a) ** 2 + (ys - b) ** 2 - r ** 2
    return circle, residuals
