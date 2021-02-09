from abc import ABC, abstractmethod

from .program import Program
from ..environment.spec import Specification


class Dynamics(ABC):
    program_class = Program

    @abstractmethod
    def partially_execute(self, program, spec):
        return [self.partially_execute_pair(program, pair) for pair in spec.pairs]

    @abstractmethod
    def partially_execute_pair(self, program, pair):
        [result] = self.partially_execute(program, Specification([pair]))
        return result

    @abstractmethod
    def program_is_complete(self, program, spec):
        pass

    @abstractmethod
    def program_is_correct(self, program, spec):
        pass

    @property
    @abstractmethod
    def program_class(self):
        pass
