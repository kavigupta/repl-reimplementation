from abc import ABC, abstractmethod

import attr
from PIL import Image, ImageDraw
import numpy as np
from .value import Point


class Canvas(ABC):
    @abstractmethod
    def draw_point(self, x, y):
        pass

    @abstractmethod
    def draw_line(self, a, b):
        pass

    @abstractmethod
    def draw_circle(self, center, r, theta_start, theta_end):
        pass


@attr.s
class ValidatingCanvas(Canvas):
    points = attr.ib(default=attr.Factory(list))
    lines = attr.ib(default=attr.Factory(list))
    circles = attr.ib(default=attr.Factory(list))

    def draw_point(self, x, y):
        self.points.append((x, y))

    def draw_line(self, a, b):
        self.lines.append((a, b))

    def draw_circle(self, center, r, theta_start, theta_end):
        self.circles.append((center, r, theta_start, theta_end))


class PillowCanvas(Canvas):
    def __init__(self, lower_left, size, pixels):
        self.image = Image.new("1", (pixels, pixels))
        self.draw = ImageDraw.Draw(self.image)

        self.lower_left = lower_left
        self.size = size
        self.pixels = pixels

    def _transform_point(self, p):
        return (
            (p.x - self.lower_left.x) / self.size * self.pixels,
            self.pixels - (p.y - self.lower_left.y) / self.size * self.pixels,
        )

    def draw_point(self, x, y):
        pass

    def draw_line(self, a, b):
        a = self._transform_point(a)
        b = self._transform_point(b)
        self.draw.line((*a, *b), fill=1)

    def draw_circle(self, center, r, theta_start, theta_end):
        top_left = Point(center.x - r, center.y + r)
        bottom_right = Point(center.x + r, center.y - r)
        bounding = self._transform_point(top_left) + self._transform_point(bottom_right)
        self.draw.arc(
            bounding,
            theta_start * 180 / np.pi,
            theta_end * 180 / np.pi,
            fill=1,
        )
