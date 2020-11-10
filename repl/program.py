from abc import ABC, abstractmethod

import attr


class Program(ABC):
    @property
    @abstractmethod
    def partials(self):
        pass

    @classmethod
    @abstractmethod
    def production(cls, action, programs):
        pass


@attr.s
class SequentialProgram(Program):
    tokens = attr.ib()

    @property
    def partials(self):
        for t in range(len(self.tokens)):
            yield [SequentialProgram(self.tokens[:t])], self.tokens[t]

    @classmethod
    def production(self, action, programs):
        [program] = programs
        return SequentialProgram(tuple(program.tokens) + (action,))
