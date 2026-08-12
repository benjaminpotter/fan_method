"""
Show the rayleigh points, Rayleigh AoP, and measured AoP dumps as image plots.

The Rust executable dumps raw uint8 rayleigh_point data in row-major order, where
each byte is 0 or 1 for a single pixel. It also dumps rayleigh_aop_v, aop_s, and
aop_v data as big-endian f64 radians in row-major order. This script reshapes the
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


def matching_rayleigh_aop_v_dump(rayleigh_dump: Path) -> Path:
    return rayleigh_dump.with_name(
        rayleigh_dump.name.replace("rayleigh_point_", "rayleigh_aop_v_", 1)
    )


def matching_aop_s_dump(rayleigh_dump: Path) -> Path:
    return rayleigh_dump.with_name(
        rayleigh_dump.name.replace("rayleigh_point_", "aop_s_", 1)
    )


def matching_aop_v_dump(rayleigh_dump: Path) -> Path:
    return rayleigh_dump.with_name(
        rayleigh_dump.name.replace("rayleigh_point_", "aop_v_", 1)
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


def add_center_pixel_grid(ax: plt.Axes, rows: int, cols: int) -> None:
    center_row = rows // 2
    center_col = cols // 2

    ax.axvline(center_col, color="white", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.axhline(center_row, color="white", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.axvline(center_col, color="black", linestyle=":", linewidth=0.8, alpha=0.9)
    ax.axhline(center_row, color="black", linestyle=":", linewidth=0.8, alpha=0.9)


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
            "rayleigh_aop_v_*.bin file to use. Defaults to the file matching the "
            "rayleigh_point frame, or the latest rayleigh_aop_v_*.bin if no match exists."
        ),
    )
    parser.add_argument(
        "--aop-s-dump",
        type=Path,
        help=(
            "measured aop_s_*.bin sensor-frame file to use. Defaults to the file "
            "matching the rayleigh_point frame, or the latest aop_s_*.bin if no "
            "match exists."
        ),
    )
    parser.add_argument(
        "--aop-v-dump",
        type=Path,
        help=(
            "measured aop_v_*.bin v-frame file to use. Defaults to the file "
            "matching the rayleigh_point frame, or the latest aop_v_*.bin if no "
            "match exists."
        ),
    )
    parser.add_argument(
        "--aop-cmap", default="jet", help="AoP heatmap colormap, default twilight"
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
    aop_dump = args.aop_dump or matching_rayleigh_aop_v_dump(dump)
    if not aop_dump.exists():
        aop_dump = latest_dump("rayleigh_aop_v_*.bin")
    aop_s_dump = args.aop_s_dump or matching_aop_s_dump(dump)
    if not aop_s_dump.exists():
        aop_s_dump = latest_dump("aop_s_*.bin")
    aop_v_dump = args.aop_v_dump or matching_aop_v_dump(dump)
    if not aop_v_dump.exists():
        aop_v_dump = latest_dump("aop_v_*.bin")

    image = read_rayleigh_points(dump, args.rows, args.cols)
    rayleigh_aop_v = read_aop(aop_dump, args.rows, args.cols)
    aop_s = read_aop(aop_s_dump, args.rows, args.cols)
    aop_v = read_aop(aop_v_dump, args.rows, args.cols)
    rayleigh_aop_v_deg = np.rad2deg(rayleigh_aop_v)
    aop_s_deg = np.rad2deg(aop_s)
    aop_v_deg = np.rad2deg(aop_v)
    finite_rayleigh_aop_v_deg = rayleigh_aop_v_deg[np.isfinite(rayleigh_aop_v_deg)]
    finite_aop_s_deg = aop_s_deg[np.isfinite(aop_s_deg)]
    finite_aop_v_deg = aop_v_deg[np.isfinite(aop_v_deg)]

    fig, (image_ax, aop_ax, aop_s_ax, aop_v_ax) = plt.subplots(1, 4, figsize=(24, 5))
    image_ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    image_ax.set_title(str(dump))
    image_ax.set_xlabel("Column")
    image_ax.set_ylabel("Row")

    aop_im = aop_ax.imshow(
        rayleigh_aop_v_deg,
        cmap=args.aop_cmap,
        vmin=-90,
        vmax=90,
        interpolation="nearest",
    )
    aop_ax.set_title(str(aop_dump))
    aop_ax.set_xlabel("Column")
    aop_ax.set_ylabel("Row")
    fig.colorbar(aop_im, ax=aop_ax, label="Rayleigh AoP v-frame (deg)")

    aop_s_im = aop_s_ax.imshow(
        aop_s_deg,
        cmap=args.aop_cmap,
        vmin=-90,
        vmax=90,
        interpolation="nearest",
    )
    aop_s_ax.set_title(str(aop_s_dump))
    aop_s_ax.set_xlabel("Column")
    aop_s_ax.set_ylabel("Row")
    fig.colorbar(aop_s_im, ax=aop_s_ax, label="Measured AoP sensor-frame (deg)")

    aop_v_im = aop_v_ax.imshow(
        aop_v_deg,
        cmap=args.aop_cmap,
        vmin=-90,
        vmax=90,
        interpolation="nearest",
    )
    aop_v_ax.set_title(str(aop_v_dump))
    aop_v_ax.set_xlabel("Column")
    aop_v_ax.set_ylabel("Row")
    fig.colorbar(aop_v_im, ax=aop_v_ax, label="Measured AoP v-frame (deg)")

    for ax in (aop_ax, aop_s_ax, aop_v_ax):
        add_center_pixel_grid(ax, args.rows, args.cols)

    n_rayleigh = int(np.count_nonzero(image))
    percent = 100.0 * n_rayleigh / image.size
    print(f"{dump}: {n_rayleigh}/{image.size} rayleigh points ({percent:.2f}%)")
    print(
        f"{aop_dump}: Rayleigh AoP v-frame deg min={finite_rayleigh_aop_v_deg.min():.2f}, "
        f"max={finite_rayleigh_aop_v_deg.max():.2f}, "
        f"mean={finite_rayleigh_aop_v_deg.mean():.2f}, "
        f"std={finite_rayleigh_aop_v_deg.std():.2f}"
    )
    print(
        f"{aop_s_dump}: Measured AoP sensor-frame deg min={finite_aop_s_deg.min():.2f}, "
        f"max={finite_aop_s_deg.max():.2f}, "
        f"mean={finite_aop_s_deg.mean():.2f}, "
        f"std={finite_aop_s_deg.std():.2f}"
    )
    print(
        f"{aop_v_dump}: Measured AoP v-frame deg min={finite_aop_v_deg.min():.2f}, "
        f"max={finite_aop_v_deg.max():.2f}, "
        f"mean={finite_aop_v_deg.mean():.2f}, "
        f"std={finite_aop_v_deg.std():.2f}"
    )

    if args.save:
        fig.savefig(args.save, bbox_inches="tight", dpi=200)
        print(f"saved {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
