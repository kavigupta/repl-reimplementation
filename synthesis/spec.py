import attr


@attr.s
class Specification:
    pairs = attr.ib()
    test_pairs = attr.ib()


@attr.s
class Pair:
    input = attr.ib()
    output = attr.ib()
