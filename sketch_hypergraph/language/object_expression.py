from abc import ABC, abstractclassmethod

import attr

from .sampler import SamplerError
from .types import BaseType
from .numeric_expression import NumericExpression


class ObjectExpression(ABC):
    @abstractclassmethod
    def sample(cls, sampler, typenv):
        typ = sampler.sample_object_expression_type(typenv)
        return typ.sample(sampler, typenv)

    @abstractclassmethod
    def is_samplable(cls, typenv):
        pass


@attr.s
class Point(ObjectExpression):
    x = attr.ib()
    y = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        x = NumericExpression.sample(sampler, typenv)
        y = NumericExpression.sample(sampler, typenv)
        return cls(x, y), BaseType.point

    @classmethod
    def is_samplable(cls, typenv):
        return True


@attr.s
class Line(ObjectExpression):
    p1 = attr.ib()
    p2 = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        p1, p2 = sampler.sample_distinct_variables(
            typenv, 2, require_type=BaseType.point
        )
        return cls(p1, p2), BaseType.compound_object

    @classmethod
    def is_samplable(cls, typenv):
        return typenv.contains_types({BaseType.point: 2})


@attr.s
class Circle(ObjectExpression):
    four_points = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        four_points = sampler.sample_distinct_variables(
            typenv, 4, require_type=BaseType.point
        )
        return cls(four_points), BaseType.compound_object

    @classmethod
    def is_samplable(cls, typenv):
        return typenv.contains_types({BaseType.point: 4})


@attr.s
class Arc(ObjectExpression):
    start = attr.ib()
    middle = attr.ib()
    end = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        start, middle, end = sampler.sample_distinct_variables(
            typenv, 3, require_type=BaseType.point
        )
        return cls(start, middle, end), BaseType.compound_object

    @classmethod
    def is_samplable(cls, typenv):
        return typenv.contains_types({BaseType.point: 3})
