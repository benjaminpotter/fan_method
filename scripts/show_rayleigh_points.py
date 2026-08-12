"""
Show the rayleigh points, Rayleigh AoP, and measured AoP as image plots.

The Rust executable dumps raw uint8 rayleigh_point data in row-major order, where
each byte is 0 or 1 for a single pixel. It also dumps rayleigh_aop and measured
aop data as big-endian f64 radians in row-major order. This script reshapes the
dumps into the camera image shape and displays them with matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROWS = 1024
COLS = 1224
EXPECTED_PIXELS = ROWS * COLS


def latest_dump(pattern: str) -> Path:
    dumps = sorted(Path.cwd().glob(pattern))
    if not dumps:
        raise FileNotFoundError(f"no {pattern} files found in current directory")
    return dumps[-1]


def matching_rayleigh_aop_dump(rayleigh_dump: Path) -> Path:
    return rayleigh_dump.with_name(
        rayleigh_dump.name.replace("rayleigh_point_", "rayleigh_aop_", 1)
    )


def matching_measured_aop_dump(rayleigh_dump: Path) -> Path:
    return rayleigh_dump.with_name(
        rayleigh_dump.name.replace("rayleigh_point_", "aop_", 1)
    )


def read_dump(path: Path, dtype: np.dtype, rows: int, cols: int) -> np.ndarray:
    data = np.fromfile(path, dtype=dtype)
    expected_pixels = rows * cols

    if data.size != expected_pixels:
        raise ValueError(
            f"{path} contains {data.size} pixels, expected {expected_pixels} "
            f"for shape ({rows}, {cols})"
        )

    return data.reshape((rows, cols))


def read_rayleigh_points(path: Path, rows: int, cols: int) -> np.ndarray:
    return read_dump(path, np.dtype(np.uint8), rows, cols)


def read_aop(path: Path, rows: int, cols: int) -> np.ndarray:
    # src/main.rs writes f64::to_be_bytes(), so read as big-endian float64.
    return read_dump(path, np.dtype(">f8"), rows, cols)


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
        "--aop-dump",
        type=Path,
        help=(
            "rayleigh_aop_*.bin file to use. Defaults to the file matching the "
            "rayleigh_point frame, or the latest rayleigh_aop_*.bin if no match exists."
        ),
    )
    parser.add_argument(
        "--measured-aop-dump",
        type=Path,
        help=(
            "measured aop_*.bin file to use. Defaults to the file matching the "
            "rayleigh_point frame, or the latest aop_*.bin if no match exists."
        ),
    )
    parser.add_argument(
        "--aop-cmap", default="twilight", help="AoP heatmap colormap, default twilight"
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
    dump = args.dump or latest_dump("rayleigh_point_*.bin")
    aop_dump = args.aop_dump or matching_rayleigh_aop_dump(dump)
    if not aop_dump.exists():
        aop_dump = latest_dump("rayleigh_aop_*.bin")
    measured_aop_dump = args.measured_aop_dump or matching_measured_aop_dump(dump)
    if not measured_aop_dump.exists():
        measured_aop_dump = latest_dump("aop_*.bin")

    image = read_rayleigh_points(dump, args.rows, args.cols)
    aop = read_aop(aop_dump, args.rows, args.cols)
    measured_aop = read_aop(measured_aop_dump, args.rows, args.cols)
    aop_deg = np.rad2deg(aop)
    measured_aop_deg = np.rad2deg(measured_aop)
    finite_aop_deg = aop_deg[np.isfinite(aop_deg)]
    finite_measured_aop_deg = measured_aop_deg[np.isfinite(measured_aop_deg)]

    fig, (image_ax, aop_ax, measured_aop_ax) = plt.subplots(1, 3, figsize=(18, 5))
    image_ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    image_ax.set_title(str(dump))
    image_ax.set_xlabel("Column")
    image_ax.set_ylabel("Row")

    aop_im = aop_ax.imshow(
        aop_deg,
        cmap=args.aop_cmap,
        vmin=-90,
        vmax=90,
        interpolation="nearest",
    )
    aop_ax.set_title(str(aop_dump))
    aop_ax.set_xlabel("Column")
    aop_ax.set_ylabel("Row")
    fig.colorbar(aop_im, ax=aop_ax, label="Rayleigh AoP (deg)")

    measured_aop_im = measured_aop_ax.imshow(
        measured_aop_deg,
        cmap=args.aop_cmap,
        vmin=-90,
        vmax=90,
        interpolation="nearest",
    )
    measured_aop_ax.set_title(str(measured_aop_dump))
    measured_aop_ax.set_xlabel("Column")
    measured_aop_ax.set_ylabel("Row")
    fig.colorbar(measured_aop_im, ax=measured_aop_ax, label="Measured AoP (deg)")

    n_rayleigh = int(np.count_nonzero(image))
    percent = 100.0 * n_rayleigh / image.size
    print(f"{dump}: {n_rayleigh}/{image.size} rayleigh points ({percent:.2f}%)")
    print(
        f"{aop_dump}: Rayleigh AoP deg min={finite_aop_deg.min():.2f}, "
        f"max={finite_aop_deg.max():.2f}, mean={finite_aop_deg.mean():.2f}, "
        f"std={finite_aop_deg.std():.2f}"
    )
    print(
        f"{measured_aop_dump}: Measured AoP deg min={finite_measured_aop_deg.min():.2f}, "
        f"max={finite_measured_aop_deg.max():.2f}, "
        f"mean={finite_measured_aop_deg.mean():.2f}, "
        f"std={finite_measured_aop_deg.std():.2f}"
    )

    if args.save:
        fig.savefig(args.save, bbox_inches="tight", dpi=200)
        print(f"saved {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
