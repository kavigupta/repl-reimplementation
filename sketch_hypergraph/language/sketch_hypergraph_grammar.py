from operator import add, sub, eq, ne, le, lt, ge, gt, and_, or_

from sketch_hypergraph.language.types import BaseType

from .grammar import (
    BlockExpansionRule,
    BuiltinExpansionRule,
    ForExpansionRule,
    Grammar,
    ConstantExpansionRule,
    IfExpansionRule,
    ValueExpansionRule,
    VariableExpansionRule,
    NonTerminalExpansionRule,
    StatementExpansionRule,
)
from .value import Point, Number

NUMERIC_ARITHMETIC_OPERATION = {"+": add, "-": sub}
BOOLEAN_ARITHMETIC_OPERATION = {
    "==": eq,
    "!=": ne,
    "<=": le,
    "<": lt,
    ">=": ge,
    ">": gt,
}
BOOLEAN_CONDITION_OPERATION = {"&&": and_, "||": or_}


def standard_grammar(alphabet, numbers, max_block_length, context):
    return Grammar(
        alphabet,
        {
            BaseType.numeric: [
                ConstantExpansionRule(numbers),
                VariableExpansionRule(BaseType.numeric),
                NonTerminalExpansionRule(
                    "NBinop", ["num_bin_op", BaseType.numeric, BaseType.numeric]
                ),
            ],
            "num_bin_op": [BuiltinExpansionRule(sorted(NUMERIC_ARITHMETIC_OPERATION))],
            BaseType.bool: [
                NonTerminalExpansionRule(
                    "BNBinop", ["bool_bin_op_num", BaseType.numeric, BaseType.numeric]
                ),
                NonTerminalExpansionRule(
                    "BBBinop", ["bool_bin_op_bool", BaseType.bool, BaseType.bool]
                ),
                NonTerminalExpansionRule("BUnop", ["bool_un_op", BaseType.bool]),
            ],
            "bool_bin_op_num": [
                BuiltinExpansionRule(sorted(BOOLEAN_ARITHMETIC_OPERATION))
            ],
            "bool_bin_op_bool": [
                BuiltinExpansionRule(sorted(BOOLEAN_ARITHMETIC_OPERATION))
            ],
            "bool_un_op": [BuiltinExpansionRule(["!"])],
            BaseType.point: [VariableExpansionRule(BaseType.point)],
            BaseType.statement: [
                StatementExpansionRule(signature)
                for signature in context.statement_signatures.values()
            ]
            + [ForExpansionRule(), IfExpansionRule()],
            BaseType.block_length: [
                ConstantExpansionRule(
                    [("block_length", i) for i in range(1, max_block_length + 1)]
                )
            ],
            BaseType.block: [BlockExpansionRule(context)],
            BaseType.range: [NonTerminalExpansionRule("Range", [BaseType.numeric] * 3)],
        },
    )


def value_grammar(numbers):
    return Grammar(
        "",
        {
            BaseType.numeric: [ConstantExpansionRule(numbers, constructor=Number)],
            BaseType.point: [ValueExpansionRule(Point.of, [BaseType.numeric] * 2)],
        },
    )
