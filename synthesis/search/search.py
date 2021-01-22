import torch

from abc import ABC, abstractmethod

from permacache import permacache, stable_hash


class Search(ABC):
    @abstractmethod
    def __call__(self, model, spec):
        pass


def frozen_hash(m):
    if not hasattr(m, "_frozen_hash"):
        m._frozen_hash = stable_hash(m.state_dict())
    return m._frozen_hash


@permacache("synthesis/search", key_function=dict(m=frozen_hash, spec=stable_hash))
def infer(search, m, spec):
    m.eval()
    with torch.no_grad():
        return search(m, spec)
