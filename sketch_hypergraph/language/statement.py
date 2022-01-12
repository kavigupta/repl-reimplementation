from abc import ABC, abstractclassmethod

import attr

from .condition_expression import Condition
from .object_expression import ObjectExpression
from .range_expression import RangeExpression


class Statement(ABC):
    @abstractclassmethod
    def sample(cls, sampler, typenv):
        typ = sampler.sample_statement_type()
        return typ.sample(sampler, typenv)


@attr.s
class Assignment(Statement):
    expression = attr.ib()
    variable = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        expression, type = ObjectExpression.sample(sampler, typenv)
        variable = sampler.sample_fresh_variable(typenv, allow_none=True)
        return cls(expression, variable), typenv.bind(variable, type)


@attr.s
class Unassignment(Statement):
    variable = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        variable = sampler.sample_variable(typenv)
        return cls(variable), typenv


@attr.s
class If(Statement):
    condition = attr.ib()
    then_statement = attr.ib()
    else_statement = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        condition = Condition.sample(sampler, typenv)
        then_statement, typenv_1 = Statement.sample(sampler, typenv)
        else_statement, typenv_2 = Statement.sample(sampler, typenv)
        return cls(condition, then_statement, else_statement), typenv_1.intersection(
            typenv_2
        )


@attr.s
class For(Statement):
    variable = attr.ib()
    range_expression = attr.ib()
    statement = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        from .program import Program
        variable = sampler.sample_fresh_variable(typenv, allow_none=False)
        range_expression, content_type = RangeExpression.sample(sampler, typenv)
        statement, _ = Program.sample(sampler, typenv.bind(variable, content_type))
        return cls(variable, range_expression, statement), typenv
