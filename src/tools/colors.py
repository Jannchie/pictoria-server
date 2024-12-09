from io import BufferedReader
from pathlib import Path

import numpy as np
from colorthief import ColorThief
from matplotlib import pyplot as plt

from utils import timer


@timer
def get_palette(image_path: Path, *, colors: int = 5) -> tuple[tuple[int, int, int], ...]:
    color_thief = ColorThief(image_path)
    return tuple(color_thief.get_palette(color_count=colors, quality=4))


def rgb2int(rgb: tuple[int, int, int]) -> int:
    return (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]


def int2rgb(i: int) -> tuple[int, int, int]:
    return (i >> 16, (i >> 8) & 0xFF, i & 0xFF)


def show_palette(palette: tuple[tuple[int, int, int]]) -> None:
    # Convert the list of tuples into a numpy array
    np_palette = np.array(palette, dtype=np.uint8)

    # Add an additional dimension to represent a row of colors
    np_palette = np_palette[np.newaxis, :, :]

    # Display the palette
    _, ax = plt.subplots()
    ax.imshow(np_palette)
    ax.axis("off")  # Turn off the axis
    plt.show()


def get_palette_ints(image: Path | BufferedReader, *, colors: int = 5) -> tuple[int, ...]:
    return tuple(rgb2int(rgb) for rgb in get_palette(image, colors=colors))


if __name__ == "__main__":
    p2 = get_palette(Path(R"E:\pictoria-server\demo\88f94319dd02b296c7658ae364009484.jpg"), colors=5)
    color_ints = [rgb2int(rgb) for rgb in p2]
    show_palette(p2)
