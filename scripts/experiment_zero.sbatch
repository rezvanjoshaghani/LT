#!/bin/bash
# SLURM template for Experiment Zero (PLAN Phase 3). Account and partition are
# never hard-coded; pass them at submit time:
#
#   sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
#          --array 0-17 scripts/experiment_zero.sbatch configs/experiment_zero.yaml
#
# Rerunning the array is safe: --resume skips scenes that already have results,
# and a finished scene's parquet is never overwritten.
#
# Environment:
#   LOT_ENV            micromamba env with torch and pyarrow (default lot-encode)
#   MAMBA_ROOT_PREFIX  micromamba root (default $HOME/micromamba)
#   MAMBA_EXE          micromamba binary (default: found on PATH or ~/.local/bin)
#   LOT_MODULES        optional space-separated modules to load
#
#SBATCH --job-name=lot-expzero
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch ... scripts/experiment_zero.sbatch <config.yaml>}"
LOT_ENV="${LOT_ENV:-lot-encode}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

if [ -n "${LOT_MODULES:-}" ] && command -v module >/dev/null 2>&1; then
    module load $LOT_MODULES
fi

MM="${MAMBA_EXE:-}"
if [ -z "$MM" ]; then
    if command -v micromamba >/dev/null 2>&1; then
        MM="$(command -v micromamba)"
    else
        MM="$HOME/.local/bin/micromamba"
    fi
fi

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# PROTOCOL 3.1 keeps the single-frame shape test and the batch-equality test in
# the suite permanently. Setting the variable alone does nothing: it only lifts
# a pytest skip, and this job runs a module, not pytest. So the job runs them,
# and refuses to do its own work if they fail. A cluster run is then a run that
# has checked the encoder contract rather than one that could have.
export LOT_ENCODER_SMOKE=1
"$MM" run -n "$LOT_ENV" python -m pytest tests/test_encoder_cache.py     -k "vggt_batching or grid_orientation or sees_one_frame" -q

"$MM" run -n "$LOT_ENV" python -m lot.evaluate \
    --config "$CONFIG" \
    --scene-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --resume
