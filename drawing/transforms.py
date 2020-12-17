import numpy as np


def scale(cx, cy):
    return np.array([[cx, 0, 0], [0, cy, 0], [0, 0, 1]], dtype=np.float64)


def translate(cx, cy):
    return np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)


def rotate(theta):
    theta = theta * np.pi / 180
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


TRANSFORMS = {
    "scale": lambda c: scale(c, c),
    "scaleX": lambda c: scale(c, 1),
    "scaleY": lambda c: scale(1, c),
    "translateX": lambda c: translate(c, 0),
    "translateY": lambda c: translate(0, c),
    "rotate": rotate,
}
