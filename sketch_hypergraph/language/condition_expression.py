from abc import ABC, abstractclassmethod

import attr

from .numeric_expression import NumericExpression
from .sampler import SamplerError

from operator import eq, ne, le, lt, ge, gt, and_, or_

NUMERIC_CONDITION_OPERATION = {"==": eq, "!=": ne, "<=": le, "<": lt, ">=": ge, ">": gt}

BOOLEAN_CONDITION_OPERATION = {"&&": and_, "||": or_}


class Condition(ABC):
    @abstractclassmethod
    def sample(cls, sampler, typenv):
        for _ in range(sampler.condition_retries()):
            try:
                typ = sampler.sample_condition_type()
                return typ.sample(sampler, typenv)
            except SamplerError:
                pass
        raise SamplerError("Could not sample condition, too many retries.")


@attr.s
class BaseCondition(Condition):
    operation = attr.ib()
    operand_1 = attr.ib()
    operand_2 = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        operation = sampler.sample_operation(list(NUMERIC_CONDITION_OPERATION))
        operand_1, operand_2 = [
            NumericExpression.sample(sampler, typenv) for _ in range(2)
        ]
        return cls(operation, operand_1, operand_2)


@attr.s
class ConditionBinOp(Condition):
    operation = attr.ib()
    left = attr.ib()
    right = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        operation = sampler.sample_operation(list(BOOLEAN_CONDITION_OPERATION))
        left = Condition.sample(sampler, typenv)
        right = Condition.sample(sampler, typenv)
        return cls(operation, left, right)


@attr.s
class NegationCondition(Condition):
    condition = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        condition = Condition.sample(sampler, typenv)
        return cls(condition)
