from .ast import (
    Constant,
    Variable,
    NumericOperation,
    Comparison,
    BooleanBinaryOp,
    BooleanUnaryOp,
    Primitive,
    Transform,
    Color,
    Combine,
    Repeat,
    If,
    IfE,
)

grammar = {
    "N": [Constant, Variable, [NumericOperation, "N", "N"]],
    "C": [
        [Comparison, "N", "N"],
        [BooleanBinaryOp, "C", "C"],
        [BooleanUnaryOp, "C"],
    ],
    "D": [
        Primitive,
        [Transform, "N", "D"],
        [Color, "N", "N", "N", "D"],
        [Combine, "D", "D"],
        [Repeat, Variable, "N", "N", "D"],
        [If, "C", "D"],
        [IfE, "C", "D", "D"],
    ],
}
