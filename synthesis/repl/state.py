import attr


@attr.s
class ReplSearchState:
    partial_programs = attr.ib()
    specification = attr.ib()
    dynamics = attr.ib()

    @property
    def semantic_partial_programs(self):
        return [
            self.dynamics.partially_execute(p, self.specification)
            for p in self.partial_programs
        ]

    @property
    def done(self):
        return any(self.spec.program_is_complete(p) for p in self.partial_programs)

    @property
    def is_goal(self):
        return any(self.spec.program_is_correct(p) for p in self.partial_programs)

    def transition(self, action):
        return ReplSearchState(
            [self.spec.program_class.production(action, self.partial_programs)],
            self.spec,
            self.dynamics,
        )


@attr.s
class Action:
    production = attr.ib()
