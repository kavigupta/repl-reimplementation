import attr


@attr.s
class MLozaicSpecification:
    pairs = attr.ib()


@attr.s
class MLozaicPair:
    input = attr.ib()
    output = attr.ib()
