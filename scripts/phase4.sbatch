#!/bin/bash
# SLURM template for Phase 4 evaluation (one task per scene). CPU shape, like
# Phase 3 evaluation: the job reads cached features and cached VGGT depth and
# never loads an encoder.
#
#   sbatch --account "${SLURM_ACCOUNT:-$SLURM_JOB_ACCOUNT}" \
#          --partition "${SLURM_PARTITION:-$SLURM_JOB_PARTITION}" \
#          --array 0-17 scripts/phase4.sbatch configs/phase4.yaml
#
# Run scripts/run_phase4.sh check, convention, and gates first; evaluation
# refuses to start without the convention report, and the gates are the 4.5
# stop condition that must pass before anything else is interpreted.
#
#SBATCH --job-name=lot-phase4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch ... scripts/phase4.sbatch <config.yaml> [gates|evaluate]}"
# Second argument picks what the array does. "gates" runs PROTOCOL 4.5's
# pure-rotation correctness gates for every alignment level and writes gate
# evidence without a parquet; "evaluate" is the full run. Same code, same
# per-scene shape, so the gates can be cleared at array speed before any
# result is produced.
MODE="${2:-evaluate}"
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

# Refuse a dirty worktree. Phase 4 rows must name the commit that made them.
if [ -n "$(git status --porcelain)" ]; then
    echo "refusing to run from a dirty worktree:" >&2
    git status --porcelain | head -20 >&2
    exit 1
fi
echo "clean worktree at commit $(git rev-parse HEAD)"

case "$MODE" in
gates)
    "$MM" run -n "$LOT_ENV" python -m lot.phase4 \
        --config "$CONFIG" \
        --scene-index "${SLURM_ARRAY_TASK_ID:-0}" \
        --gates-only
    ;;
evaluate)
    "$MM" run -n "$LOT_ENV" python -m lot.phase4 \
        --config "$CONFIG" \
        --scene-index "${SLURM_ARRAY_TASK_ID:-0}" \
        --resume
    ;;
*)
    echo "unknown mode '$MODE'; use gates or evaluate" >&2
    exit 1
    ;;
esac
