from .grammar import grammar
from .ast.node import ParseError, Atom, Form

MATCHED_PARENS = {"(": ")", "[": "]", "{": "}"}
PARENS = set(MATCHED_PARENS.keys()) | set(MATCHED_PARENS.values())


def lex(x):
    tokens = []
    for c in x:
        if c.isspace():
            tokens.append("")
            continue
        if not tokens or c in PARENS:
            tokens.append(c)
            tokens.append("")
            continue
        tokens[-1] += c
    return [tok for tok in tokens if tok]


def _parse(tokens):
    if tokens[-1] in MATCHED_PARENS:
        last = MATCHED_PARENS[tokens.pop()]
        result = []
        while tokens[-1] != last:
            result.append(_parse(tokens))
        tokens.pop()
        return result
    return tokens.pop()


def parse_s_expression(x):
    return _parse(lex(x)[::-1])


def parse(x, grammar=grammar, production="D"):
    return parse_grammar(parse_s_expression(x), grammar, production)


def parse_grammar(s, grammar, production):
    if isinstance(production, str):
        rules = grammar[production]
        for rule in rules:
            try:
                return parse_grammar(s, grammar, rule)
            except ParseError:
                pass
        raise ParseError

    if isinstance(production, type) and issubclass(production, Atom):
        return production.parse(s)

    assert isinstance(production, list) and issubclass(production[0], Form), str(
        production
    )

    if len(production) != len(s):
        raise ParseError

    if s[0] not in production[0].tags():
        raise ParseError

    operands = [
        parse_grammar(b, grammar, prod) for b, prod in zip(s[1:], production[1:])
    ]

    return production[0].parse(s[0], operands)
