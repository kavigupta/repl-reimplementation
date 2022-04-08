import string

from ..language.evaluation import EvaluationContext
from ..language.sketch_hypergraph_grammar import standard_grammar, value_grammar
from ..language.statement_signature import PrimitiveStatementSignature
from ..language.types import BaseType

from .experimental_setting import ExperimentalSetting

_context = EvaluationContext.of(
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

no_control_experiment = ExperimentalSetting(
    context=_context,
    grammar=standard_grammar(
        string.ascii_lowercase,
        list(range(10)),
        10,
        context=_context,
    ),
    value_grammar=value_grammar(list(range(10))),
    sampler_spec=dict(
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
    ),
)
