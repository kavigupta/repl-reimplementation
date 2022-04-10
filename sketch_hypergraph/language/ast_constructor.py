from abc import ABC, abstractmethod
import attr
from sketch_hypergraph.language.constructible import Constructible

from sketch_hypergraph.language.types import (
    BaseType,
    FilteredType,
    FreeVariableType,
    WithinContext,
)
from sketch_hypergraph.language.value import Object

from .ast import AST, ASTNode, BlockNode, Constant, ForNode, Variable


class PartiallyFinishedNode(Constructible):
    def node_params(self):
        return []

    @abstractmethod
    def next_type(self):
        pass

    @abstractmethod
    def replace(self, replacement):
        pass

    def serialize(self):
        raise NotImplementedError(
            "Cannot serialize a partially finished node as of now"
        )


@attr.s
class PartiallyFinishedTreeNode(PartiallyFinishedNode):
    tag = attr.ib()
    finished = attr.ib()
    to_finish = attr.ib()
    ast_constructor = attr.ib(kw_only=True, default=ASTNode)
    require_distinct = attr.ib(kw_only=True, default=False)

    def next_type(self):
        typ = self.to_finish[0]
        if self.require_distinct:
            return typ.map(
                lambda t: FilteredType(t, [x.node_summary() for x in self.finished])
            )
        return typ

    @to_finish.validator
    def _check_to_finish(self, attribute, value):
        assert value, "must have something left to finish"
        for element in value:
            assert isinstance(element, WithinContext), str(element)

    def replace(self, replacement):
        if len(self.to_finish) == 1:
            return self.ast_constructor(self.tag, self.finished + [replacement])
        return PartiallyFinishedTreeNode(
            self.tag,
            self.finished + [replacement],
            self.to_finish[1:],
            ast_constructor=self.ast_constructor,
            require_distinct=self.require_distinct,
        )

    def node_class(self):
        return self.tag


@attr.s
class PartiallyFinishedBlockNode(PartiallyFinishedNode):
    evaluation_context = attr.ib(kw_only=True)
    typenv = attr.ib(kw_only=True)
    length = attr.ib(default=None, kw_only=True)
    elements = attr.ib(default=[], kw_only=True)

    def next_type(self):
        if self.length is None:
            return WithinContext(BaseType.block_length, self.typenv)
        return WithinContext(BaseType.statement, self.typenv)

    def replace(self, replacement):
        if self.length is None:
            assert isinstance(replacement, Constant)
            replacement = replacement.value
            assert (
                isinstance(replacement, tuple)
                and len(replacement) == 2
                and replacement[0] == "block_length"
            )
            return PartiallyFinishedBlockNode(
                evaluation_context=self.evaluation_context,
                typenv=self.typenv,
                length=replacement[1],
            )
        new_elements = self.elements + [replacement]
        if len(new_elements) == self.length:
            return BlockNode(new_elements)
        return PartiallyFinishedBlockNode(
            evaluation_context=self.evaluation_context,
            typenv=self.evaluation_context.post_node_environment(
                replacement, self.typenv
            ),
            length=self.length,
            elements=new_elements,
        )

    def node_class(self):
        return "Block"


@attr.s
class PartiallyFinishedForNode(PartiallyFinishedNode):
    typenv = attr.ib(kw_only=True)
    variable = attr.ib(default=None, kw_only=True)
    range = attr.ib(default=None, kw_only=True)
    block = attr.ib(default=None, kw_only=True)

    def next_type(self):
        if self.variable is None:
            return WithinContext(FreeVariableType(), self.typenv)
        if self.range is None:
            return WithinContext(BaseType.range, self.typenv)
        return WithinContext(
            BaseType.block, self.typenv.bind(self.variable, BaseType.numeric)
        )

    def replace(self, replacement):
        if self.variable is None:
            assert isinstance(replacement, Variable)
            return PartiallyFinishedForNode(
                typenv=self.typenv,
                variable=replacement.name,
            )
        if self.range is None:
            return PartiallyFinishedForNode(
                typenv=self.typenv,
                variable=self.variable,
                range=replacement,
            )
        return ForNode(self.variable, self.range, replacement)

    def node_class(self):
        return "For"


class ASTConstructionState(ABC):
    @staticmethod
    def initialize(grammar, target):
        return ASTConstructionStatePartial(
            grammar, [PartiallyFinishedTreeNode("root", [], [target])]
        )

    @staticmethod
    def run_full_with_driver(grammar, driver, target):
        state = ASTConstructionState.initialize(grammar, target)
        while not state.is_done():
            state = state.step(driver)
        return state.ast

    @abstractmethod
    def step(self, driver, typenv):
        pass

    @abstractmethod
    def is_done(self):
        pass


@attr.s
class ASTConstructionStatePartial(ASTConstructionState):
    grammar = attr.ib()
    partially_finished_nodes = attr.ib()

    def step(self, driver):
        tag = self.partially_finished_nodes[-1].next_type()
        replacement = driver.select(self.grammar.expand(tag))
        return self.replace(replacement)

    def is_done(self):
        return False

    def replace(self, replacement):
        nodes = self.partially_finished_nodes[:]
        while nodes:
            if not isinstance(replacement, (AST, Object)):
                assert isinstance(replacement, PartiallyFinishedNode)
                return ASTConstructionStatePartial(self.grammar, nodes + [replacement])
            last_node = nodes.pop()
            replacement = last_node.replace(replacement)
        assert replacement.tag == "root"
        return ASTConstructionStateDone(replacement.children[0])


@attr.s
class ASTConstructionStateDone(ASTConstructionState):
    ast = attr.ib()

    def step(self, driver, typenv):
        raise Exception("done, no more steps")

    def is_done(self):
        return True
