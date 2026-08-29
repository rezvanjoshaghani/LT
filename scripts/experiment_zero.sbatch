#!/bin/bash
# SLURM template for Experiment Zero (PLAN Phase 3). Account and partition are
# never hard-coded; pass them at submit time:
#
#   sbatch --account "${SLURM_ACCOUNT:-$SLURM_JOB_ACCOUNT}" --partition "${SLURM_PARTITION:-$SLURM_JOB_PARTITION}" \
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

# Runs are provenance or they are nothing. The Stream D records carry a -dirty
# marker because untracked SLURM logs shared the tree at run time; re-audit
# finding R-1 is that the marker cannot separate logs from source edits after
# the fact. The logs are gitignored now, so a clean tree is achievable, and
# this job refuses to produce evidence without one.
if [ -n "$(git status --porcelain)" ]; then
    echo "refusing to run from a dirty worktree:" >&2
    git status --porcelain | head -20 >&2
    exit 1
fi
echo "clean worktree at commit $(git rev-parse HEAD)"

# This job reads cached features and never loads an encoder, so it does not
# request a GPU. An earlier revision ran the real-weight encoder tests here to
# make PROTOCOL 3.1's permanent tests actually execute on the cluster. That was
# the wrong place: those tests fall back to CPU when CUDA is absent, so all 18
# array tasks would have loaded and run VGGT-1B on CPU before doing any
# evaluation. They belong in the caching job, which holds the GPU and is where
# the encoder contract is exercised, and they run there.
#
# The contract this job depends on is that the caches it reads carry the
# provenance saying which weights produced them. It needs no step here:
# evaluation validates each scene cache against its manifest before opening it,
# on metadata alone, for the one scene this task evaluates.

"$MM" run -n "$LOT_ENV" python -m lot.evaluate \
    --config "$CONFIG" \
    --scene-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --resume
