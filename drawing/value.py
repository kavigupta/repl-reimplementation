import attr

import numpy as np


@attr.s
class Item:
    type = attr.ib()
    transform = attr.ib(default=np.eye(3))
    color = attr.ib(default=[0, 0, 0])
