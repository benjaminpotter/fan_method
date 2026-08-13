#!/usr/bin/env python3
"""Merge Fan-method SLURM shard CSVs into one frame-sorted CSV.

Example:
    scripts/merge_shards.py results/123456 --output results/123456/frame_results.csv

By default, this reads files matching ``frame_results_shard_*.csv`` in the input
path. If duplicate frame indices are present, the first occurrence in sorted file
order is kept.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable


DEFAULT_PATTERN = "frame_results_shard_*.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-shard Fan-method result CSVs, de-duplicate by frame index, and sort by frame index."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Directory containing shard CSVs, or one shard CSV file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("frame_results.csv"),
        help="Merged CSV path [default: frame_results.csv].",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob used when input is a directory [default: {DEFAULT_PATTERN}].",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write only a header if no data rows are found instead of failing.",
    )
    return parser.parse_args()


def shard_paths(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"input path does not exist: {input_path}")


def read_shards(paths: Iterable[Path]) -> tuple[list[str] | None, dict[int, list[str]]]:
    header: list[str] | None = None
    rows_by_index: dict[int, list[str]] = {}

    for path in paths:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            try:
                file_header = next(reader)
            except StopIteration:
                print(f"warning: skipping empty CSV: {path}", file=sys.stderr)
                continue

            if header is None:
                header = file_header
            elif file_header != header:
                raise ValueError(f"header mismatch in {path}")

            for line_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                try:
                    index = int(row[0])
                except (ValueError, IndexError) as e:
                    raise ValueError(
                        f"invalid frame index in {path}:{line_number}: {row!r}"
                    ) from e

                if index not in rows_by_index:
                    rows_by_index[index] = row
                else:
                    print(
                        f"warning: duplicate frame index {index} in {path}:{line_number}; keeping first occurrence",
                        file=sys.stderr,
                    )

    return header, rows_by_index


def main() -> int:
    args = parse_args()
    paths = shard_paths(args.input, args.pattern)
    if not paths:
        print(
            f"error: no shard CSVs found in {args.input} matching {args.pattern!r}",
            file=sys.stderr,
        )
        return 1

    header, rows_by_index = read_shards(paths)
    if header is None:
        print("error: no CSV headers found", file=sys.stderr)
        return 1
    if not rows_by_index and not args.allow_empty:
        print("error: no data rows found", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for index in sorted(rows_by_index):
            writer.writerow(rows_by_index[index])

    print(
        f"merged {len(paths)} shard file(s), wrote {len(rows_by_index)} frame row(s) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
