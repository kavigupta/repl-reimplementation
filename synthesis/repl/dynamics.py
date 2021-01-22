from abc import ABC, abstractmethod

from .program import Program


class Dynamics(ABC):
    program_class = Program

    @abstractmethod
    def partially_execute(self, program, spec):
        pass

    @abstractmethod
    def program_is_complete(self, program):
        pass

    @abstractmethod
    def program_is_correct(self, program):
        pass
