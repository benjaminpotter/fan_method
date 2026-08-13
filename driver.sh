#!/usr/bin/env bash
#
# SLURM script for running the Rust Fan-method runner (`src/main.rs`) on an HPC
# cluster. Submit with, for example:
#
#   TRAJECTORY_DIR=/path/to/trajectory sbatch driver.sh
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
#SBATCH --output=fan-method_%A_%a.out
#SBATCH --error=fan-method_%A_%a.err

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

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

cargo build --release --locked

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
OUTPUT_CSV="$OUTPUT_DIR/frame_results_shard_${TASK_ID}_of_${TASK_COUNT}.csv"

srun "$PROJECT_DIR/target/release/fan_method" \
  --trajectory-dir "$TRAJECTORY_DIR" \
  --output-csv "$OUTPUT_CSV" \
  --frame-stride "$FRAME_STRIDE" \
  --start-frame "$START_FRAME" \
  --shard-index "$TASK_ID" \
  --shard-count "$TASK_COUNT" \
  "${MAX_FRAMES_ARG[@]}"
