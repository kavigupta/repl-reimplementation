from abc import ABC, abstractmethod

import attr


@attr.s
class StatementSignature(ABC):
    name = attr.ib()
    input_types = attr.ib()
    output_types = attr.ib()

    def bind(self, node):
        assert node.tag == self.name
        assert len(node.children) == len(self.input_types) + len(self.output_types)

        inputs, outputs = (
            node.children[: len(self.input_types)],
            node.children[len(self.input_types) :],
        )
        return Statement(self, inputs, outputs)


class PrimitiveStatementSignature(StatementSignature):
    pass


@attr.s
class Statement:
    signature = attr.ib()
    inputs = attr.ib()
    outputs = attr.ib()

    def bind_outputs(self, input_environment):
        return input_environment.bind_all(
            {
                var.name: typ
                for var, typ in zip(self.outputs, self.signature.output_types)
            }
        )
