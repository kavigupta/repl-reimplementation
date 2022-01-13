from abc import ABC, abstractmethod


import attr


from .ast import BuiltinSymbol, Constant, IfNode, Variable
from .ast_constructor import (
    PartiallyFinishedBlockNode,
    PartiallyFinishedForNode,
    PartiallyFinishedTreeNode,
)
from .types import BaseType, FreeVariableType, WithinContext


@attr.s
class Grammar:
    possible_variables = attr.ib()
    expansion_rules = attr.ib()

    def expand(self, target):
        assert isinstance(target, WithinContext), str(target)

        if isinstance(target.type, FreeVariableType):
            return sorted(
                Variable(x)
                for x in set(self.possible_variables)
                - set(target.typenv.assigned_variables())
            )
        assert isinstance(target.type, (BaseType, str))
        overall = []
        for rule in self.expansion_rules[target.type]:
            overall += rule.expand(target.typenv)
        return overall


class ExpansionRule(ABC):
    @abstractmethod
    def expand(self, typenv):
        pass


@attr.s
class ConstantExpansionRule(ExpansionRule):
    elements = attr.ib()

    def expand(self, typenv):
        return [Constant(x) for x in self.elements]


@attr.s
class BuiltinExpansionRule(ExpansionRule):
    elements = attr.ib()

    def expand(self, typenv):
        return [BuiltinSymbol(x) for x in self.elements]


@attr.s
class VariableExpansionRule(ExpansionRule):
    type = attr.ib()

    def expand(self, typenv):
        return [Variable(x) for x in typenv.assigned_variables(self.type)]


@attr.s
class NonTerminalExpansionRule(ExpansionRule):
    tag = attr.ib()
    types = attr.ib()

    def expand(self, typenv):
        return [
            PartiallyFinishedTreeNode(
                self.tag, [], [WithinContext(x, typenv) for x in self.types]
            )
        ]


@attr.s
class StatementExpansionRule(ExpansionRule):
    statement_signature = attr.ib()

    def expand(self, typenv):
        types = self.statement_signature.input_types + [FreeVariableType()] * len(
            self.statement_signature.output_types
        )
        types = [WithinContext(x, typenv) for x in types]
        return [
            PartiallyFinishedTreeNode(
                self.statement_signature.name,
                [],
                types,
            )
        ]


@attr.s
class BlockExpansionRule(ExpansionRule):

    evaluation_context = attr.ib()

    def expand(self, typenv):
        return [
            PartiallyFinishedBlockNode(
                evaluation_context=self.evaluation_context,
                typenv=typenv,
            )
        ]


@attr.s
class ForExpansionRule(ExpansionRule):
    def expand(self, typenv):
        return [
            PartiallyFinishedForNode(
                typenv=typenv,
            )
        ]


@attr.s
class IfExpansionRule(ExpansionRule):
    def expand(self, typenv):
        return [
            PartiallyFinishedTreeNode(
                "If",
                [],
                [
                    WithinContext(x, typenv)
                    for x in [BaseType.bool, BaseType.block, BaseType.block]
                ],
                ast_constructor=lambda tag, x: IfNode(*x),
            )
        ]
