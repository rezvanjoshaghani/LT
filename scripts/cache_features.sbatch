#!/bin/bash
# SLURM template for frozen feature caching (PLAN Phase 2). Account and
# partition are never hard-coded; pass them at submit time:
#
#   sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
#          --array 0-17 scripts/cache_features.sbatch configs/cache_features_all.yaml
#
# Rerunning the array is safe: --resume skips scenes that already have a cache,
# and a finished cache is never overwritten.
#
# Environment:
#   LOT_ENV            micromamba env with torch (default lot-encode)
#   MAMBA_ROOT_PREFIX  micromamba root (default $HOME/micromamba)
#   MAMBA_EXE          micromamba binary (default: found on PATH or ~/.local/bin)
#   LOT_MODULES        optional space-separated modules to load
#   TORCH_HOME         torch.hub cache, so weights download once (default $PWD/cache/torch)
#   HF_HOME            huggingface cache for VGGT (default $PWD/cache/huggingface)
#
#SBATCH --job-name=lot-encode
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch ... scripts/cache_features.sbatch <config.yaml>}"
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
export TORCH_HOME="${TORCH_HOME:-$PWD/cache/torch}"
export HF_HOME="${HF_HOME:-$PWD/cache/huggingface}"
mkdir -p "$TORCH_HOME" "$HF_HOME"

"$MM" run -n "$LOT_ENV" python -m lot.encoders \
    --config "$CONFIG" \
    --scene-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --resume
