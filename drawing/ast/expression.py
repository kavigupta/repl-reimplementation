from abc import ABC, abstractmethod

import attr

from .node import Atom, Form, Error
from ..constants import NUM_VARS, CONSTANTS_RANGE
from ..operations import NUM_OPS, COMPARISONS


@attr.s
class Constant(Atom):
    value = attr.ib()

    @classmethod
    def tags(cls):
        return [str(i) for i in range(-CONSTANTS_RANGE, CONSTANTS_RANGE + 1)]

    @classmethod
    def parse(cls, s):
        try:
            return cls(int(s))
        except (ValueError, TypeError):
            return Error()

    def evaluate(self, env):
        return self.value


@attr.s
class Variable(Atom):
    name = attr.ib()

    @classmethod
    def tags(cls):
        return [f"${i}" for i in range(NUM_VARS)]

    @classmethod
    def parse(cls, s):
        if not isinstance(s, str) or not s.startswith("$"):
            return Error()
        return cls(s)

    def evaluate(self, env):
        return env[self.name]


@attr.s
class Operation(Form):
    operation = attr.ib()
    operands = attr.ib()

    @classmethod
    def tags(cls):
        return list(cls.operations())

    @classmethod
    def parse(cls, tag, operands):
        return cls(tag, operands)

    @classmethod
    @abstractmethod
    def operations(cls):
        pass

    def evaluate(self, env):
        return self.operations()[self.operation](
            *[op.evaluate(env) for op in self.operands]
        )


class NumericOperation(Operation):
    @classmethod
    def operations(cls):
        return NUM_OPS


class Comparison(Operation):
    @classmethod
    def operations(cls):
        return COMPARISONS


class BooleanBinaryOp(Operation):
    @classmethod
    def operations(cls):
        return BOOL_BINARY_OPS


class BooleanUnaryOp(Operation):
    @classmethod
    def operations(cls):
        return BOOL_UNARY_OP
