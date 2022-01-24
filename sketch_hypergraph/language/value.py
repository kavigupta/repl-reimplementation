from abc import ABC, abstractmethod

import attr
import numpy as np

from .constructible import Constructible
from .utils import fit_circle


class Object(Constructible):
    pass


@attr.s
class Number(Object):
    value = attr.ib()

    def node_class(self):
        return "Number"

    def node_params(self):
        return [self.value]


class DrawnObject(Object):
    @abstractmethod
    def draw(self, canvas):
        pass

    def node_class(self):
        return type(self).__name__

    def node_params(self):
        return []


class DrawnObjectInvalidError(Exception):
    pass


@attr.s
class Point(DrawnObject):
    x = attr.ib()
    y = attr.ib()

    @classmethod
    def of(cls, x, y):
        return cls(x.value, y.value)

    def draw(self, canvas):
        canvas.point(self.x, self.y)


@attr.s
class Line(DrawnObject):
    a = attr.ib()
    b = attr.ib()

    @classmethod
    def of(cls, a, b):
        return cls(a, b)

    def draw(self, canvas):
        if self.a == self.b:
            raise DrawnObjectInvalidError("Line is a point")
        canvas.draw_line(self.a, self.b)


@attr.s
class Arc(DrawnObject):
    a = attr.ib()
    b = attr.ib()
    c = attr.ib()

    @classmethod
    def of(cls, a, b, c):
        return cls(a, b, c)

    def draw(self, canvas):
        (center, r), residuals = fit_circle([self.a, self.b, self.c])
        if r == 0:
            raise DrawnObjectInvalidError("Arc is a point")
        if residuals.max() > 1e-6:
            raise DrawnObjectInvalidError("Arc is not a circle")
        theta_start = np.arctan2(self.a.y - center.y, self.a.x - center.x)
        theta_end = np.arctan2(self.a.y - center.y, self.a.x - center.x)
        canvas.draw_circle(center, r, theta_start, theta_end)


@attr.s
class Circle(DrawnObject):
    a = attr.ib()
    b = attr.ib()
    c = attr.ib()
    d = attr.ib()

    @classmethod
    def of(cls, a, b, c, d):
        return cls(a, b, c, d)

    def draw(self, canvas):
        (center, r), _ = fit_circle([self.a, self.b, self.c, self.d])
        if r == 0:
            raise DrawnObjectInvalidError("Circle is a point")
        canvas.draw_circle(center, r, 0, 2 * np.pi)
