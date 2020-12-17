import numpy as np


def invert_transform(t, coordinates):
    print(t)
    assert t.shape == (3, 3)
    A = t[:2, :2]
    b = t[:2, 2]

    shifted = coordinates - b[:, None, None]
    shifted = shifted.reshape(2, -1)
    rescaled = np.linalg.inv(A) @ shifted
    rescaled = rescaled.reshape(*coordinates.shape)
    return rescaled


def render(items, size=(100, 100)):
    w, h = size
    x, y = np.meshgrid(np.arange(w) - w / 2, h / 2 - np.arange(h))
    coords = np.array([x, y])
    image = np.zeros((*size, 3)) + 1
    for item in items:
        trans_coords = invert_transform(item.transform, coords)
        mask = item.type(*trans_coords)
        image[mask] = np.array([*item.color]) / 255
    return image
