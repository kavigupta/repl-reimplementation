from typing import Counter
import attr
from enum import Enum

from .driver import SamplerError


class BaseType(Enum):
    numeric = "numeric"
    bool = "bool"
    point = "point"
    range = "range"

    statement = "statement"
    block = "block"
    block_length = "block_length"


@attr.s
class FreeVariableType:
    pass


@attr.s
class WithinContext:
    type = attr.ib()
    typenv = attr.ib()


@attr.s
class TypeEnv:
    type_map = attr.ib()

    def assigned_variables(self, require_type=None):
        vars = self.type_map.keys()
        if require_type is not None:
            vars = [var for var in vars if self.type_map[var] == require_type]
        return vars

    def bind_all(self, mapping):
        type_map = self.type_map.copy()
        type_map.update(mapping)
        return TypeEnv(type_map)

    def bind(self, var, typ):
        if var is None:
            return self
        return self.bind_all({var: typ})

    def intersection(self, other):
        common_vars = set(self.type_map.keys()).intersection(other.type_map.keys())
        for var in common_vars:
            if self.type_map[var] != other.type_map[var]:
                raise SamplerError(
                    "Variable {} has incompatible types {} and {}".format(
                        var, self.type_map[var], other.type_map[var]
                    )
                )
        return TypeEnv({var: self.type_map[var] for var in common_vars})

    def contains_types(self, type_counts):
        actual_counts = Counter(self.type_map.values())
        return all(actual_counts[typ] >= count for typ, count in type_counts.items())
