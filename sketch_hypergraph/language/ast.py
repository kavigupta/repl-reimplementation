from abc import abstractmethod
import attr

from .constructible import Constructible


class AST(Constructible):
    @abstractmethod
    def s_exp(self):
        pass

    @abstractmethod
    def serialize(self):
        return [f"PLACEHOLDER [{self.node_summary()}]"]


@attr.s
class ASTNode(AST):
    tag = attr.ib()
    children = attr.ib()

    def node_class(self):
        return self.tag

    def node_params(self):
        return []

    def s_exp(self):
        return "(" + " ".join([self.tag] + [x.s_exp() for x in self.children]) + ")"

    def serialize(self):
        return [
            self.node_summary(),
            *[x for e in self.children for x in e.serialize()],
        ]


@attr.s
class BlockNode(AST):
    elements = attr.ib()

    def node_class(self):
        return "Block"

    def node_params(self):
        return []

    def s_exp(self):
        return "[" + "; ".join(x.s_exp() for x in self.elements) + ")"

    def serialize(self):
        return [
            self.node_summary(),
            *Constant(("block_length", len(self.elements))).serialize(),
            *[x for e in self.elements for x in e.serialize()],
        ]


@attr.s
class ForNode(AST):
    variable = attr.ib()
    range = attr.ib()
    body = attr.ib()

    def node_class(self):
        return "For"

    def node_params(self):
        return []

    def s_exp(self):
        return (
            "(For "
            + " ".join([self.variable] + [x.s_exp() for x in [self.range, self.body]])
            + ")"
        )


@attr.s
class IfNode(AST):
    condition = attr.ib()
    then_branch = attr.ib()
    else_branch = attr.ib()

    def node_class(self):
        return "If"

    def node_params(self):
        return []

    def s_exp(self):
        return (
            "("
            + " ".join(
                [
                    "If",
                    self.condition.s_exp(),
                    self.then_branch.s_exp(),
                    self.else_branch.s_exp(),
                ]
            )
            + ")"
        )


class ASTLeaf(AST):
    def node_class(self):
        return type(self).__name__

    def serialize(self):
        return [self.node_summary()]


@attr.s
class BuiltinSymbol(ASTLeaf):
    symbol = attr.ib()

    def node_params(self):
        return [self.symbol]

    def s_exp(self):
        return self.symbol


@attr.s
class Constant(ASTLeaf):
    value = attr.ib()

    def node_params(self):
        return [self.value]

    def s_exp(self):
        return str(self.value)


@attr.s
class Variable(ASTLeaf):
    name = attr.ib()

    def node_params(self):
        return [self.name]

    def s_exp(self):
        return self.name
