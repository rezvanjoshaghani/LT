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
# The weight caches are set before anything loads a model. The encoder tests
# below download the same checkpoints this job then encodes with, so setting
# these afterwards would have sent the test downloads to the default locations
# in the home directory: a different copy of the weights, on a filesystem with
# its own quota, fetched once per array task.
export TORCH_HOME="${TORCH_HOME:-$PWD/cache/torch}"
export HF_HOME="${HF_HOME:-$PWD/cache/huggingface}"
mkdir -p "$TORCH_HOME" "$HF_HOME"

# PROTOCOL 3.1 keeps the single-frame shape test and the batch-equality test in
# the suite permanently. Setting the variable alone does nothing: it only lifts
# a pytest skip, and this job runs a module, not pytest. So the job runs them,
# and refuses to do its own work if they fail. A cluster run is then a run that
# has checked the encoder contract rather than one that could have.
#
# They run here and not in the evaluation job. These tests load real weights and
# fall back to CPU when CUDA is absent, so they belong with the GPU allocation
# that is loading those weights anyway, not in an 18-task CPU array that never
# opens an encoder.
# The pins are required, not recorded-if-present. PROTOCOL locks the
# encoders, and an earlier revision of this job permitted unpinned runs while
# only a downstream warning noticed; the reporting layer now refuses unpinned
# provenance outright, so a cache built here without pins would be rebuilt
# anyway. Failing before the GPU allocation does any work is the cheap place.
# scripts/pin_encoder_revisions.py resolves both values.
for VAR in LOT_DINOV2_REVISION LOT_VGGT_REVISION; do
    VALUE="${!VAR:-}"
    if [ -z "$VALUE" ]; then
        echo "$VAR is not set. PROTOCOL locks the encoders; resolve the pins" >&2
        echo "first with: python scripts/pin_encoder_revisions.py" >&2
        exit 1
    fi
    if ! [[ "$VALUE" =~ ^[0-9a-f]{40}$ ]]; then
        echo "$VAR='$VALUE' is not a full 40-hex commit hash. A branch or tag" >&2
        echo "name is a moving ref, not a pin; resolve the commit with" >&2
        echo "  python scripts/pin_encoder_revisions.py" >&2
        exit 1
    fi
done
echo "LOT_DINOV2_REVISION=$LOT_DINOV2_REVISION"
echo "LOT_VGGT_REVISION=$LOT_VGGT_REVISION"

export LOT_ENCODER_SMOKE=1
"$MM" run -n "$LOT_ENV" python -m pytest tests/test_encoder_cache.py     -k "vggt_batching or grid_orientation or sees_one_frame" -q

"$MM" run -n "$LOT_ENV" python -m lot.encoders \
    --config "$CONFIG" \
    --scene-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --resume
