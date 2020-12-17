from abc import ABC, abstractmethod

import attr


class Node(ABC):
    @classmethod
    @abstractmethod
    def tags(cls):
        pass

    @abstractmethod
    def evaluate(self, env):
        pass


class Atom(Node):
    @classmethod
    @abstractmethod
    def parse(cls, s):
        pass


class Form(Node):
    @classmethod
    @abstractmethod
    def parse(cls, tag, operands):
        pass


@attr.s
class Error(Node):
    @classmethod
    def tags(cls):
        raise SyntaxError("error in parsing")

    def evaluate(self, env):
        raise SyntaxError("error in parsing")
