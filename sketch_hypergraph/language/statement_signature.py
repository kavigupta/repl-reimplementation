from abc import ABC, abstractmethod

import attr

from .evaluation import EvaluationResult
from .value import Arc, Circle, Line, Point


@attr.s
class StatementSignature(ABC):
    name = attr.ib()
    input_types = attr.ib()
    output_types = attr.ib()
    require_distinct = attr.ib()

    def bind(self, node):
        assert node.tag == self.name
        assert len(node.children) == len(self.input_types) + len(self.output_types)

        inputs, outputs = (
            node.children[: len(self.input_types)],
            node.children[len(self.input_types) :],
        )
        return Statement(self, inputs, outputs)

    @abstractmethod
    def evaluate(self, env, statement):
        pass


class PrimitiveStatementSignature(StatementSignature):
    def evaluate(self, env, statement):
        statement = self.bind(statement)
        evaluated_inputs = []
        drawn_objects = []
        for inp in statement.inputs:
            res = env.evaluate(inp)
            evaluated_inputs.append(res.value)
            drawn_objects += res.drawn_objects
            assert res.new_environment is env

        evaluated_result = dict(
            Point=Point.of, Line=Line.of, Circle=Circle.of, Arc=Arc.of
        )[self.name](*evaluated_inputs)

        assert len(self.output_types) <= 1
        if statement.outputs:
            [output] = statement.outputs
            env = env.bind(output.name, evaluated_result)
        drawn_objects += [evaluated_result]
        return EvaluationResult(
            value=evaluated_result, drawn_objects=drawn_objects, new_environment=env
        )


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
