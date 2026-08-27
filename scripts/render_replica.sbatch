#!/bin/bash
# SLURM template for Replica rendering (PLAN Phase 1).
# Normally submitted by scripts/run_phase1_render.sh. Account and partition
# are never hard-coded; pass them at submit time:
#
#   sbatch --account "${SLURM_ACCOUNT:-$SLURM_JOB_ACCOUNT}" --partition "${SLURM_PARTITION:-$SLURM_JOB_PARTITION}" \
#          scripts/render_replica.sbatch configs/render_replica_pilot.yaml
#   sbatch --account "${SLURM_ACCOUNT:-$SLURM_JOB_ACCOUNT}" --partition "${SLURM_PARTITION:-$SLURM_JOB_PARTITION}" \
#          --array 0-17 scripts/render_replica.sbatch configs/render_replica_all.yaml
#
# Environment (defaults match scripts/setup_borah.sh):
#   LOT_ENV            micromamba env with habitat-sim (default lot-render)
#   MAMBA_ROOT_PREFIX  micromamba root (default $HOME/micromamba)
#   MAMBA_EXE          micromamba binary (default: found on PATH or ~/.local/bin)
#   LOT_MODULES        optional space-separated modules to load
#
#SBATCH --job-name=lot-render
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch ... scripts/render_replica.sbatch <config.yaml>}"
LOT_ENV="${LOT_ENV:-lot-render}"
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
"$MM" run -n "$LOT_ENV" python -m lot.render_replica \
    --config "$CONFIG" \
    --scene-index "${SLURM_ARRAY_TASK_ID:-0}"
