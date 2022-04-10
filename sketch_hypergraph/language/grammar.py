from abc import ABC, abstractmethod


import attr


from .ast import BuiltinSymbol, Constant, IfNode, Variable
from .ast_constructor import (
    PartiallyFinishedBlockNode,
    PartiallyFinishedForNode,
    PartiallyFinishedTreeNode,
)
from .types import BaseType, FilteredType, FreeVariableType, WithinContext


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
        if isinstance(target.type, FilteredType):
            return [
                x
                for x in self.expand(target.map(lambda t: t.base_type))
                if x.node_summary() not in target.type.exclude_list
            ]

        assert isinstance(target.type, (BaseType, str))
        overall = []
        for rule in self.expansion_rules[target.type]:
            overall += rule.expand(target.typenv)
        return overall

    def token_alphabet(self):
        result = set()
        for variable in self.possible_variables:
            result.add(Variable(variable).node_summary())
        for rules in self.expansion_rules.values():
            for rule in rules:
                result.update(rule.token_alphabet())
        return sorted(result)


class ExpansionRule(ABC):
    @abstractmethod
    def expand(self, typenv):
        pass

    @abstractmethod
    def token_alphabet(self):
        pass


class ContextFreeExpansionRule(ExpansionRule):
    @abstractmethod
    def expand_context_free(self):
        pass

    def expand(self, typenv):
        return self.expand_context_free()

    def token_alphabet(self):
        return [x.node_summary() for x in self.expand_context_free()]


@attr.s
class ConstantExpansionRule(ContextFreeExpansionRule):
    elements = attr.ib()
    constructor = attr.ib(default=Constant)

    def expand_context_free(self):
        return [self.constructor(x) for x in self.elements]


@attr.s
class BuiltinExpansionRule(ContextFreeExpansionRule):
    elements = attr.ib()

    def expand_context_free(self):
        return [BuiltinSymbol(x) for x in self.elements]


@attr.s
class VariableExpansionRule(ExpansionRule):
    type = attr.ib()

    def expand(self, typenv):
        return [Variable(x) for x in typenv.assigned_variables(self.type)]

    def token_alphabet(self):
        return []  # no additional tokens created by this rule


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

    def token_alphabet(self):
        return [self.tag + "()"]


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
                require_distinct=self.statement_signature.require_distinct,
            )
        ]

    def token_alphabet(self):
        return [self.statement_signature.name + "()"]


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

    def token_alphabet(self):
        return ["Block()"]


@attr.s
class ForExpansionRule(ExpansionRule):
    def expand(self, typenv):
        return [
            PartiallyFinishedForNode(
                typenv=typenv,
            )
        ]

    def token_alphabet(self):
        return ["For()"]


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

    def token_alphabet(self):
        return ["If()"]


@attr.s
class ValueExpansionRule(ExpansionRule):
    constructor = attr.ib()
    types = attr.ib()

    def expand(self, typenv):
        return [
            PartiallyFinishedTreeNode(
                type(self.constructor).__name__,
                [],
                [WithinContext(x, typenv) for x in self.types],
                ast_constructor=lambda tag, x: self.constructor(*x),
            )
        ]

    def token_alphabet(self):
        return [type(self.constructor).__name__ + "()"]
