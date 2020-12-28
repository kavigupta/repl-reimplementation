import attr


@attr.s
class Specification:
    pairs = attr.ib()


@attr.s
class Pair:
    input = attr.ib()
    output = attr.ib()
