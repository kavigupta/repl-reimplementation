import attr

from .specification import Spec


@attr.s
class ReplSearchState:
    partial_programs = attr.ib()
    spec: Spec = attr.ib()

    @property
    def semantic_partial_programs(self):
        return [self.spec.partially_execute(p) for p in self.partial_programs]

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
        )


@attr.s
class Action:
    production = attr.ib()
