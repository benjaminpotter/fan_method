#!/usr/bin/env bash
#
# SLURM script for running the Rust Fan-method runner (`src/main.rs`) on an HPC
# cluster. Submit with, for example:
#
#   mkdir -p slurm_logs
#   TRAJECTORY_DIR=/path/to/trajectory sbatch driver.sh
#
# SLURM opens stdout/stderr before this script starts, so the slurm_logs/
# directory must exist before calling sbatch.
#
# By default this is an array job. Each array task processes a disjoint shard of
# every fifth frame (5 Hz input -> 1 Hz results) and writes its own checkpoint CSV.
# Merge the shard CSVs after the job completes if a single file is desired.

#SBATCH --job-name=fan-method
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-15%4
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

: "${TRAJECTORY_DIR:?Set TRAJECTORY_DIR to the trajectory directory passed to the runner}"

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results/${SLURM_JOB_ID}}"
FRAME_STRIDE="${FRAME_STRIDE:-5}"
START_FRAME="${START_FRAME:-0}"
MAX_FRAMES_ARG=()
if [[ -n "${MAX_FRAMES:-}" ]]; then
  MAX_FRAMES_ARG=(--max-frames "$MAX_FRAMES")
fi

mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/slurm_logs"
cd "$PROJECT_DIR"

export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

# Make Rust/Cargo available on clusters where non-interactive SLURM shells do not
# inherit your login-shell PATH. You may set RUST_MODULE, e.g.
#   sbatch --export=ALL,TRAJECTORY_DIR=/data/traj,RUST_MODULE=rust driver.sh
if [[ -n "${RUST_MODULE:-}" ]]; then
  module load "$RUST_MODULE"
elif [[ -f "${CARGO_ENV:-$HOME/.cargo/env}" ]]; then
  # rustup installs this file; it prepends $HOME/.cargo/bin to PATH.
  # shellcheck disable=SC1090
  source "${CARGO_ENV:-$HOME/.cargo/env}"
fi

BINARY="$PROJECT_DIR/target/release/fan_method"
if command -v cargo >/dev/null 2>&1; then
  cargo build --release --locked
elif [[ -x "$BINARY" ]]; then
  echo "warning: cargo not found; using existing binary at $BINARY" >&2
else
  echo "error: cargo not found and no executable exists at $BINARY" >&2
  echo "Install/load Rust on the compute node, set RUST_MODULE, source ~/.cargo/env, or pre-build target/release/fan_method before submitting." >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
OUTPUT_CSV="$OUTPUT_DIR/frame_results_shard_${TASK_ID}_of_${TASK_COUNT}.csv"

srun "$BINARY" \
  --trajectory-dir "$TRAJECTORY_DIR" \
  --output-csv "$OUTPUT_CSV" \
  --frame-stride "$FRAME_STRIDE" \
  --start-frame "$START_FRAME" \
  --shard-index "$TASK_ID" \
  --shard-count "$TASK_COUNT" \
  "${MAX_FRAMES_ARG[@]}"
