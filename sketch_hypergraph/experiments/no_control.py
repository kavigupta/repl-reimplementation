import string

import attr

from ..language.evaluation import EvaluationContext
from ..language.sketch_hypergraph_grammar import standard_grammar, value_grammar
from ..language.statement_signature import PrimitiveStatementSignature
from ..language.types import BaseType

from .experimental_setting import ExperimentalSetting

primitive_context = EvaluationContext.of(
    PrimitiveStatementSignature(
        "Point", [BaseType.numeric] * 2, [BaseType.point], require_distinct=False
    ),
    PrimitiveStatementSignature(
        "Line", [BaseType.point] * 2, [], require_distinct=True
    ),
    PrimitiveStatementSignature("Arc", [BaseType.point] * 3, [], require_distinct=True),
    PrimitiveStatementSignature(
        "Circle", [BaseType.point] * 4, [], require_distinct=True
    ),
)


class NoControlExperiment(ExperimentalSetting):
    def __init__(
        self,
        max_type_size=5,
        num_elements_dist={4: 1, 5: 1, 6: 1, 7: 1},
        minimal_objects=4,
    ):
        super().__init__(
            max_type_size=max_type_size,
            num_elements_dist=num_elements_dist,
            minimal_objects=minimal_objects,
        )

    def variable_alphabet(self):
        return string.ascii_lowercase

    def grammar(self, context):
        return standard_grammar(
            self.variable_alphabet(),
            list(range(10)),
            10,
            context=context,
        )

    def value_grammar(self):
        return value_grammar(list(range(10)))

    def type_distribution(self):
        return [(BaseType.numeric, 1), (BaseType.point, 1)]

    def sampler_spec(self):
        return dict(
            type="SamplingDriver",
            weights={
                "Constant": 3,
                "Variable": 3,
                "NBinop": 1,
                "BNBinop": 8,
                "BBBinop": 1,
                "BUnop": 1,
                "Point": 0.5,
                "Circle": 1,
                "Line": 1,
                "Arc": 1,
                "For": 0,
                "If": 0,
            },
        )
