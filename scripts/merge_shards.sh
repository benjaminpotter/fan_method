#!/usr/bin/env bash
# Merge Fan-method SLURM shard CSVs into one frame-sorted CSV.
#
# This handles both layouts:
#   results/<jobid>/frame_results_shard_0_of_16.csv ...
# and array jobs where each shard ended up in its own result directory:
#   results/12073592/frame_results_shard_15_of_16.csv
#   results/12073593/frame_results_shard_0_of_16.csv
#   ...
#
# Usage:
#   scripts/merge_shards.sh RESULTS_ROOT [-o OUTPUT_CSV]
#
# Example:
#   scripts/merge_shards.sh results -o results/frame_results.csv

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/merge_shards.sh RESULTS_ROOT [-o OUTPUT_CSV]

Finds frame_results_shard_<i>_of_<n>.csv files recursively under RESULTS_ROOT,
verifies that exactly one complete shard set is present, validates matching CSV
headers, then writes one CSV sorted by frame index.

Options:
  -o, --output PATH   Output CSV path [default: RESULTS_ROOT/frame_results.csv]
  -h, --help          Show this help
EOF
}

RESULTS_ROOT=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      [[ $# -ge 2 ]] || { echo "error: missing value for $1" >&2; exit 1; }
      OUTPUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -* )
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -n "$RESULTS_ROOT" ]]; then
        echo "error: multiple RESULTS_ROOT arguments provided" >&2
        usage >&2
        exit 1
      fi
      RESULTS_ROOT="$1"
      shift
      ;;
  esac
done

if [[ -z "$RESULTS_ROOT" ]]; then
  echo "error: missing RESULTS_ROOT" >&2
  usage >&2
  exit 1
fi

if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "error: RESULTS_ROOT is not a directory: $RESULTS_ROOT" >&2
  exit 1
fi

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$RESULTS_ROOT/frame_results.csv"
fi

shopt -s nullglob

declare -A SHARD_FILES=()
declare -A SEEN_COUNTS=()
SHARD_COUNT=""

while IFS= read -r -d '' file; do
  base="$(basename "$file")"
  if [[ "$base" =~ ^frame_results_shard_([0-9]+)_of_([0-9]+)\.csv$ ]]; then
    shard_index="${BASH_REMATCH[1]}"
    shard_count="${BASH_REMATCH[2]}"

    if [[ -z "$SHARD_COUNT" ]]; then
      SHARD_COUNT="$shard_count"
    elif [[ "$SHARD_COUNT" != "$shard_count" ]]; then
      echo "error: mixed shard counts found: saw $SHARD_COUNT and $shard_count ($file)" >&2
      exit 1
    fi

    if [[ -n "${SHARD_FILES[$shard_index]:-}" ]]; then
      echo "error: duplicate shard $shard_index found:" >&2
      echo "  ${SHARD_FILES[$shard_index]}" >&2
      echo "  $file" >&2
      exit 1
    fi

    SHARD_FILES[$shard_index]="$file"
    SEEN_COUNTS[$shard_count]=1
  fi
done < <(find "$RESULTS_ROOT" -type f -name 'frame_results_shard_*_of_*.csv' -print0)

if [[ -z "$SHARD_COUNT" ]]; then
  echo "error: no shard CSVs found under $RESULTS_ROOT" >&2
  exit 1
fi

for ((i = 0; i < SHARD_COUNT; i++)); do
  if [[ -z "${SHARD_FILES[$i]:-}" ]]; then
    echo "error: missing shard $i of $SHARD_COUNT" >&2
    exit 1
  fi
done

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
DATA="$TMP_DIR/data.csv"
: > "$DATA"

HEADER=""
for ((i = 0; i < SHARD_COUNT; i++)); do
  file="${SHARD_FILES[$i]}"
  file_header="$(head -n 1 "$file")"

  if [[ -z "$HEADER" ]]; then
    HEADER="$file_header"
  elif [[ "$HEADER" != "$file_header" ]]; then
    echo "error: CSV header mismatch in $file" >&2
    exit 1
  fi

  # Append all non-header rows. Empty shard files with only a header are allowed.
  tail -n +2 "$file" >> "$DATA"
done

mkdir -p "$(dirname "$OUTPUT")"
{
  printf '%s\n' "$HEADER"
  # Sort by the first CSV field (frame index) and keep the first row for duplicate
  # indices. Frame result rows do not contain embedded newlines, and the first field
  # is an unquoted integer written by the Rust runner.
  sort -t, -k1,1n "$DATA" | awk -F, '!seen[$1]++'
} > "$OUTPUT"

row_count="$(tail -n +2 "$OUTPUT" | wc -l | tr -d ' ')"
echo "merged $SHARD_COUNT shard file(s), wrote $row_count frame row(s) to $OUTPUT"
