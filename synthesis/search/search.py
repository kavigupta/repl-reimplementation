import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from permacache import permacache, stable_hash


class Search(ABC):
    @abstractmethod
    def __call__(self, model, spec):
        pass


def traverse(m, model_leaf, other_leaf, combiner):
    if isinstance(m, nn.Module):
        return model_leaf(m)
    elif isinstance(m, (list, tuple)):
        return combiner([traverse(mo, model_leaf, other_leaf, combiner) for mo in m])
    elif m is None:
        return other_leaf(m)
    else:
        raise RuntimeError(f"Cannot interpret {type(m)} as a model")


def frozen_hash(m):
    def leaf(m):
        if not hasattr(m, "_frozen_hash"):
            m._frozen_hash = stable_hash(m.state_dict())
        return m._frozen_hash

    return traverse(m, leaf, stable_hash, stable_hash)


@permacache(
    "synthesis/search",
    key_function=dict(m=frozen_hash, spec=lambda x: stable_hash(x, fast_bytes=True)),
)
def infer(search, m, spec):
    traverse(m, lambda x: x.eval(), lambda x: None, lambda x: None)
    with torch.no_grad():
        return search(m, spec)
