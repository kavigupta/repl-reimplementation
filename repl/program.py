from abc import ABC

import attr


class Program(ABC):
    @property
    @abstractmethod
    def partials(self):
        pass

    @abstractmethod
    def extend(self, production):
        pass


@attr.s
class SequentialProgram(Program):
    tokens = attr.ib()

    @property
    def partials(self):
        for t in range(len(self.tokens)):
            yield [Program(self.tokens[:t])], self.tokens[t]

    def extend(self, production):
        return SequentialProgram(tuple(self.tokens) + (production,))
