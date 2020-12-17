from abc import ABC, abstractmethod

import attr


class Node(ABC):
    @abstractmethod
    def evaluate(self, env):
        pass


class Atom(ABC):
    @classmethod
    @abstractmethod
    def parse(cls, s):
        pass


class Form(ABC):
    @classmethod
    @abstractmethod
    def tags(cls):
        pass

    @classmethod
    @abstractmethod
    def parse(cls, tag, operands):
        pass


@attr.s
class Error(Node):
    def evaluate(self, env):
        raise SyntaxError("error in parsing")
