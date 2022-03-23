import string
import unittest
from parameterized import parameterized
import numpy as np

from sketch_hypergraph.language.ast_constructor import ASTConstructionState
from sketch_hypergraph.language.evaluation import EvaluationContext

from sketch_hypergraph.language.sampler import sample_valid_datapoint
from sketch_hypergraph.language.sketch_hypergraph_grammar import standard_grammar, value_grammar
from sketch_hypergraph.language.statement_signature import PrimitiveStatementSignature
from sketch_hypergraph.language.types import BaseType, WithinContext
from sketch_hypergraph.language.deserializer import DeSerializingDriver


sampler_spec = dict(
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


class TestDeserializer(unittest.TestCase):
    @parameterized.expand([[x] for x in range(100)])
    def test_example(self, seed):

        context = EvaluationContext.of(
            PrimitiveStatementSignature(
                "Point",
                [BaseType.numeric] * 2,
                [BaseType.point],
                require_distinct=False,
            ),
            PrimitiveStatementSignature(
                "Line", [BaseType.point] * 2, [], require_distinct=True
            ),
            PrimitiveStatementSignature(
                "Arc", [BaseType.point] * 3, [], require_distinct=True
            ),
            PrimitiveStatementSignature(
                "Circle", [BaseType.point] * 4, [], require_distinct=True
            ),
        )

        grammar = standard_grammar(
            string.ascii_lowercase,
            list(range(10)),
            10,
            context=context,
        )

        res = sample_valid_datapoint(
            np.random.RandomState(seed),
            sampler_spec=sampler_spec,
            grammar=grammar,
            g_value=value_grammar(list(range(10))),
            t_value=[(BaseType.numeric, 1), (BaseType.point, 1)],
            max_type_size=5,
            e_context=context,
            num_elements=5,
            minimal_objects=4,
        )

        node = res["p"].serialize()
        created = ASTConstructionState.run_full_with_driver(
            grammar,
            DeSerializingDriver(node),
            WithinContext(BaseType.block, res["type_env"]),
        )
        self.assertEqual(res["p"].s_exp(), created.s_exp())
