import unittest

from PIL import Image
import numpy as np

from drawing.parser import parse
from drawing.renderer import render

render_or_check = "render"


class TestParsing(unittest.TestCase):
    def render_or_check(self, name, program, env={}):
        image = render(parse(program).evaluate(env), stretch=1)

        image = (image * 255).astype(np.uint8)

        path = f"examples/{name}.png"

        if render_or_check == "render":
            image = Image.fromarray(image, "RGB")

            image.save(path)
        else:
            actual = Image.open(path)
            self.assertEqual(0, (np.array(actual) != image).sum())

    def test_in_check_mode(self):
        self.assertEqual(
            "check", render_or_check, "should be in check mode to check images"
        )

    def test_basic_circle(self):
        self.render_or_check("black-circle", "(scale 50 circle)")

    def test_color(self):
        self.render_or_check("red-circle", "(scale 50 (color 255 0 0 circle))")

    def test_square(self):
        self.render_or_check(
            "circle-and-square",
            "(combine (scale 30 (color 0 255 0 circle)) (scale 20 (color 0 0 255 square)))",
        )

    def test_line_of_circles(self):
        self.render_or_check(
            "line-of-circles",
            "(repeat $0 0 9 (translateX (- (* $0 10) 40) (scale 5 circle)))",
        )

    def test_rotated_square(self):
        self.render_or_check(
            "rotated-squares", "(repeat $0 0 3 (rotate (* $0 10) (scale 30 square)))"
        )

    def test_parabola(self):
        self.render_or_check(
            "parabola",
            "(repeat $0 -30 30 (translateY (/ (* $0 $0) 30) (translateX $0 circle)))",
        )

    def test_parabola(self):
        self.render_or_check(
            "star-field",
            """
            (combine
                (color 0 0 128 (scale 100 square))
                (repeat $0 -3 4
                    (repeat $1 -3 4
                        (translateX (* 12 $0) (translateY (* 12 $1)
                            (scale 5
                                (color 255 255 255 (ife (= (% (+ $0 $1) 2) 0) circle square))))))))
            """,
        )
