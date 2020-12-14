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


def parse(x):
    return _parse(lex(x)[::-1])


def _parse(tokens):
    if tokens[-1] in MATCHED_PARENS:
        last = MATCHED_PARENS[tokens.pop()]
        result = []
        while tokens[-1] != last:
            result.append(_parse(tokens))
        tokens.pop()
        return result
    return tokens.pop()
