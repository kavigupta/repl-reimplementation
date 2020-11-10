import attr


@attr.s
class State:
    partial_programs = attr.ib()
    spec = attr.ib()

    @property
    def semantic_partial_programs(self):
        return [self.spec.partially_execute(p) for p in self.partial_programs]


@attr.s
class Action:
    production = attr.ib()
