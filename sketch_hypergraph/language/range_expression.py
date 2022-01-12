import attr

from .numeric_expression import NumericExpression
from .types import BaseType

@attr.s
class RangeExpression:
    start = attr.ib()
    end = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        start = NumericExpression.sample(sampler, typenv)
        end = NumericExpression.sample(sampler, typenv)
        return cls(start, end), BaseType.numeric
