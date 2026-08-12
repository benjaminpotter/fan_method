"""
Show the rayleigh points as a binary image plot.

The Rust executable dumps raw uint8 data in row-major order, where each byte is
0 or 1 for a single pixel. This script reshapes that dump into the camera image
shape and displays it with matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROWS = 1024
COLS = 1224
EXPECTED_PIXELS = ROWS * COLS


def latest_dump() -> Path:
    dumps = sorted(Path.cwd().glob("rayleigh_point_*.bin"))
    if not dumps:
        raise FileNotFoundError(
            "no rayleigh_point_*.bin files found in current directory"
        )
    return dumps[-1]


def read_rayleigh_points(path: Path, rows: int, cols: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    expected_pixels = rows * cols

    if data.size != expected_pixels:
        raise ValueError(
            f"{path} contains {data.size} pixels, expected {expected_pixels} "
            f"for shape ({rows}, {cols})"
        )

    return data.reshape((rows, cols))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a rayleigh_point_*.bin dump file"
    )
    parser.add_argument(
        "dump",
        nargs="?",
        type=Path,
        help="Dump file to show. Defaults to the latest rayleigh_point_*.bin in cwd.",
    )
    parser.add_argument(
        "--rows", type=int, default=ROWS, help=f"Image rows, default {ROWS}"
    )
    parser.add_argument(
        "--cols", type=int, default=COLS, help=f"Image cols, default {COLS}"
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Save the plot to this path instead of only showing it",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not open an interactive plot window"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump = args.dump or latest_dump()

    image = read_rayleigh_points(dump, args.rows, args.cols)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(str(dump))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    n_rayleigh = int(np.count_nonzero(image))
    percent = 100.0 * n_rayleigh / image.size
    print(f"{dump}: {n_rayleigh}/{image.size} rayleigh points ({percent:.2f}%)")

    if args.save:
        fig.savefig(args.save, bbox_inches="tight", dpi=200)
        print(f"saved {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
