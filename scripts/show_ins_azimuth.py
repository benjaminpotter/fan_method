"""
Show the INS azimuth for each frame and the lat/lon trajectory.

The INS CSV is expected to use the same columns as src/dataset.rs:
- latitude: column 13
- longitude: column 14
- roll: column 19
- pitch: column 20
- azimuth: column 21

The trajectory plot is colorized by frame index.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


INS_CSV = Path(
    "/home/ben/git/research/polcam_dataset/2025-11-24/rmc/novatel_oem7_inspva/novatel_oem7_inspva.csv"
)
LAT_COL = 13
LON_COL = 14
ROLL_COL = 19
PITCH_COL = 20
AZIMUTH_COL = 21

AngleUnits = Literal["auto", "rad", "deg"]


def read_ins_csv(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat: list[float] = []
    lon: list[float] = []
    roll: list[float] = []
    pitch: list[float] = []
    azimuth: list[float] = []

    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                lat.append(float(row[LAT_COL]))
                lon.append(float(row[LON_COL]))
                roll.append(float(row[ROLL_COL]))
                pitch.append(float(row[PITCH_COL]))
                azimuth.append(float(row[AZIMUTH_COL]))
            except (IndexError, ValueError):
                # Skip headers or malformed rows, matching csv::Reader's header skip in Rust.
                continue

    if not azimuth:
        raise ValueError(f"no INS rows could be read from {path}")

    return (
        np.asarray(lat),
        np.asarray(lon),
        np.asarray(roll),
        np.asarray(pitch),
        np.asarray(azimuth),
    )


def angles_to_degrees(values: np.ndarray, units: AngleUnits) -> tuple[np.ndarray, str]:
    if units == "auto":
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return values, "unknown"
        units = "deg" if np.nanmax(np.abs(finite)) > 2 * np.pi else "rad"

    if units == "rad":
        return np.rad2deg(values), "rad"
    return values, "deg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot INS azimuth by frame and lat/lon trajectory."
    )
    parser.add_argument(
        "ins_csv",
        nargs="?",
        type=Path,
        default=INS_CSV,
        help=f"INS CSV path, default {INS_CSV}",
    )
    parser.add_argument(
        "--angle-units",
        choices=("auto", "rad", "deg"),
        default="auto",
        help="Units used by roll/pitch/azimuth columns. Auto guesses from value range.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Save the plot to this path instead of only showing it.",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not open an interactive plot window."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lat, lon, roll, pitch, azimuth = read_ins_csv(args.ins_csv)
    frames = np.arange(azimuth.size)
    azimuth_deg, detected_units = angles_to_degrees(azimuth, args.angle_units)

    fig, (azimuth_ax, trajectory_ax) = plt.subplots(1, 2, figsize=(14, 6))

    azimuth_ax.plot(frames, azimuth_deg, linewidth=1.0)
    azimuth_ax.set_title("INS azimuth by frame")
    azimuth_ax.set_xlabel("Frame")
    azimuth_ax.set_ylabel("Azimuth (deg, clockwise from north)")
    azimuth_ax.grid(True, alpha=0.35)

    scatter = trajectory_ax.scatter(lon, lat, c=frames, s=8, cmap="viridis")
    trajectory_ax.scatter(
        lon[0], lat[0], marker="o", s=60, c="lime", edgecolors="black", label="start"
    )
    trajectory_ax.scatter(
        lon[-1], lat[-1], marker="X", s=70, c="red", edgecolors="black", label="end"
    )
    trajectory_ax.set_title("INS lat/lon trajectory")
    trajectory_ax.set_xlabel("Longitude")
    trajectory_ax.set_ylabel("Latitude")
    trajectory_ax.grid(True, alpha=0.35)
    trajectory_ax.legend(loc="best")
    trajectory_ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(scatter, ax=trajectory_ax, label="Frame")

    fig.suptitle(str(args.ins_csv))
    fig.tight_layout()

    print(f"{args.ins_csv}: {frames.size} INS frames")
    print(f"angle units: {detected_units}")
    print(
        f"azimuth deg min={np.nanmin(azimuth_deg):.3f}, "
        f"max={np.nanmax(azimuth_deg):.3f}, "
        f"mean={np.nanmean(azimuth_deg):.3f}, "
        f"std={np.nanstd(azimuth_deg):.3f}"
    )
    print(
        f"lat min={np.nanmin(lat):.8f}, max={np.nanmax(lat):.8f}; "
        f"lon min={np.nanmin(lon):.8f}, max={np.nanmax(lon):.8f}"
    )

    if args.save:
        fig.savefig(args.save, bbox_inches="tight", dpi=200)
        print(f"saved {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
