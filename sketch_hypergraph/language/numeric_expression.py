from abc import ABC, abstractclassmethod
import attr

from operator import add, sub

from sketch_hypergraph.language.types import BaseType

NUMERIC_ARITHMETIC_OPERATION = {"+": add, "-": sub}


class NumericExpression(ABC):
    @abstractclassmethod
    def sample(self, sampler, typenv):
        typ = sampler.sample_numeric_type()
        return typ.sample(sampler, typenv)


@attr.s
class NumericVariable(NumericExpression):
    variable = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        variable = sampler.sample_variable(typenv, require_type=BaseType.numeric)
        return cls(variable)


@attr.s
class NumericConstant(NumericExpression):
    value = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        return cls(sampler.sample_constant_value())


@attr.s
class NumericBinOp(NumericExpression):
    operation = attr.ib()
    left = attr.ib()
    right = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        operation = sampler.sample_operation(NUMERIC_ARITHMETIC_OPERATION)
        left, right = [NumericExpression.sample(sampler, typenv) for _ in range(2)]
        return cls(operation, left, right)
