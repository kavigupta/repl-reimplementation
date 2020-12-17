from abc import ABC, abstractmethod

import attr

from ..transforms import TRANSFORMS
from ..leaves import LEAVES
from ..value import Item
from .node import Atom, Form, Error
from .expression import Variable


@attr.s
class Primitive(Atom):
    tag = attr.ib()

    @classmethod
    def tags(cls):
        return list(LEAVES)

    @classmethod
    def parse(cls, s):
        if isinstance(s, str) and s in LEAVES:
            return cls(s)
        return Error()

    def evaluate(self, env):
        return [Item(LEAVES[self.tag])]


@attr.s
class Transform(Form):
    transform = attr.ib()
    operands = attr.ib()

    @classmethod
    def tags(cls):
        return list(TRANSFORMS)

    @classmethod
    def parse(cls, tag, operands):
        return cls(tag, operands)

    def evaluate(self, env):
        parameter, children = [op.evaluate(env) for op in self.operands]
        return [
            Item(
                child.type,
                TRANSFORMS[self.transform](parameter) @ child.transform,
                child.color,
            )
            for child in children
        ]


@attr.s
class SimpleForm(Form):
    operands = attr.ib()

    @classmethod
    def parse(cls, tag, operands):
        return cls(operands)


class Color(SimpleForm):
    @classmethod
    def tags(cls):
        return ["color"]

    def evaluate(self, env):
        r, g, b, children = [op.evaluate(env) for op in self.operands]
        return [Item(child.type, child.transform, [r, g, b]) for child in children]


class Combine(SimpleForm):
    @classmethod
    def tags(cls):
        return ["combine"]

    def evaluate(self, env):
        a, b = [op.evaluate(env) for op in self.operands]
        return a + b


class Repeat(SimpleForm):
    @classmethod
    def tags(cls):
        return ["repeat"]

    def evaluate(self, env):
        var, start, end, body = self.operands
        start, end = start.evaluate(env), end.evaluate(end)
        shape = []
        for i in range(int(start), int(end)):
            child_env = env.copy()
            child_env[var.name] = i
            shape += body.evaluate(child_env)
        return shape

    @classmethod
    def custom_sample(cls, sampler, variables):
        variable = Variable.fresh_variable(variables)
        start, end = sampler.sample(variables, production="N"), sampler.sample(
            variables, production="N"
        )
        body = sampler.sample(variables | {variable}, production="D")
        return cls.parse("repeat", (variable, start, end, body))


class If(SimpleForm):
    @classmethod
    def tags(cls):
        return ["if"]

    def evaluate(self, env):
        condition, consequent = [op.evaluate(env) for op in self.operands]
        if condition:
            return consequent
        else:
            return []


class IfE(SimpleForm):
    @classmethod
    def tags(cls):
        return ["ife"]

    def evaluate(self, env):
        condition, consequent, alternative = [op.evaluate(env) for op in self.operands]
        if condition:
            return consequent
        else:
            return alternative
