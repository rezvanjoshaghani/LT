#!/bin/bash
# SLURM template for the Phase 5 modes that are not per-(fold, seed) tasks:
# overfit, controls, and evaluate. Training has its own template.
#
#   sbatch --account "${SLURM_ACCOUNT:?}" --partition "${SLURM_PARTITION:?}" \
#          scripts/phase5_job.sbatch configs/phase5.yaml overfit
#   sbatch ... scripts/phase5_job.sbatch configs/phase5.yaml controls [--level L]
#   sbatch ... --array 0-17 scripts/phase5_job.sbatch configs/phase5.yaml evaluate [--level L]
#
# Under an array, evaluate runs one test scene per task, by its index in the
# fixed evaluation order. Each scene writes its own parquet once, so a failed
# task is rerun alone and a finished scene is never overwritten.
#
# scripts/run_phase5.sh submits these with the right array range and with the
# output captured under the run's evidence directory. Submitting by hand is
# equally safe: the entry point checks each mode's gate receipts before it
# reads anything, bound to this commit, configuration, and resolved inputs.
#
# GPU shape, like training. Evaluation and the controls run the predictor; the
# overfit gate trains it. None of them loads an encoder.
#
#SBATCH --job-name=lot-phase5
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

USAGE="usage: sbatch ... scripts/phase5_job.sbatch <config.yaml> <overfit|controls|evaluate> [--level L]"
CONFIG="${1:?$USAGE}"
MODE="${2:?$USAGE}"
shift 2
case "$MODE" in
    overfit|controls|evaluate) ;;
    *) echo "phase5_job.sbatch runs overfit, controls, or evaluate, not '$MODE'" >&2
       exit 1 ;;
esac

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

# Refuse a dirty worktree. Every receipt and parquet must name the commit that
# made it.
if [ -n "$(git status --porcelain)" ]; then
    echo "refusing to run from a dirty worktree:" >&2
    git status --porcelain | head -20 >&2
    exit 1
fi
echo "clean worktree at commit $(git rev-parse HEAD)"

ARGS=(--config "$CONFIG" --mode "$MODE" "$@")
if [ "$MODE" = evaluate ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ARGS+=(--scene-index "$SLURM_ARRAY_TASK_ID")
fi
"$MM" run -n "$LOT_ENV" python -m lot.phase5 "${ARGS[@]}"
