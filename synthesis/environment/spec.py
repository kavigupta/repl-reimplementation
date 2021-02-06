import attr


@attr.s
class Specification:
    pairs = attr.ib()
    test_pairs = attr.ib()

    def map_pairs(self, fn):
        return Specification(
            [fn(p) for p in self.pairs], [fn(p) for p in self.test_pairs]
        )


@attr.s
class Pair:
    input = attr.ib()
    output = attr.ib()
