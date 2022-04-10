import matplotlib.pyplot as plt

from permacache import permacache

from .canvas import PillowCanvas
from .value import Point


def render(datapoint, size=5, dpi=100):
    _, axs = plt.subplots(
        1,
        len(datapoint["o"]),
        figsize=(size * len(datapoint["o"]), size),
        dpi=dpi,
        facecolor="white",
    )
    images = render_images(
        datapoint["o"], dict(lower_left=Point(-15, -15), size=40, pixels=400)
    )
    for i in range(len(datapoint["o"])):
        axs[i].imshow(images[i])
        axs[i].axis("off")
        axs[i].set_title(
            "; ".join(
                f"{k}={(v.x, v.y) if isinstance(v, Point) else v.value}"
                for k, v in sorted(datapoint["i"][i].items())
            )
        )
    plt.suptitle(datapoint["p"].s_exp())
    plt.show()


@permacache("sketch_hypergraph/language/renderer/render_images")
def render_images(outputs, canvas_spec):
    images = []
    for output in outputs:
        c = PillowCanvas(**canvas_spec)
        for o in output:
            o.draw(c)
        images.append(c.image)
    return images
