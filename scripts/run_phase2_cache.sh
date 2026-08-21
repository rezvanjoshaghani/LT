#!/bin/bash
# Submit the Phase 2 feature-caching experiment on Borah. Run from anywhere
# inside the repository after the lot-encode env exists (scripts/README.md).
#
# Usage:
#   ./scripts/run_phase2_cache.sh pilot              # cache the pilot scene
#   ./scripts/run_phase2_cache.sh all                # 18-scene batch (array job)
#   ./scripts/run_phase2_cache.sh vggt               # 18-scene batch with VGGT
#   ./scripts/run_phase2_cache.sh validate [config]  # validate caches, no GPU
#
# Required for pilot, all, and vggt (never hard-coded, per CLAUDE.md):
#   SLURM_ACCOUNT     SLURM account to charge
#   SLURM_PARTITION   GPU partition name
# Optional:
#   LOT_ENV           micromamba env name       (default lot-encode)
#   MAMBA_ROOT_PREFIX micromamba root           (default $HOME/micromamba)
#   LOT_GPU_GRES      gres string               (default gpu:1)
#   LOT_MODULES       modules the sbatch job should load, space separated
#   TORCH_HOME/HF_HOME  weight caches           (default cache/ in the repo)
#
# PLAN Phase 2 acceptance order: the pilot scene first, then the batch. This
# script enforces it. Scenes that already have a cache are never resubmitted,
# so a partially finished batch resumes instead of queueing jobs that no-op.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-pilot}"
LOT_ENV="${LOT_ENV:-lot-encode}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

if command -v micromamba >/dev/null 2>&1; then
    MM="$(command -v micromamba)"
else
    MM="$HOME/.local/bin/micromamba"
fi
[ -x "$MM" ] || { echo "micromamba not found; see scripts/README.md" >&2; exit 1; }
[ -d "$MAMBA_ROOT_PREFIX/envs/$LOT_ENV" ] || {
    echo "env $LOT_ENV not found under $MAMBA_ROOT_PREFIX." >&2
    echo "Create it per the Phase 2 section of scripts/README.md." >&2
    exit 1
}

PILOT_CONFIG=configs/cache_features_pilot.yaml
ALL_CONFIG=configs/cache_features_all.yaml
VGGT_CONFIG=configs/cache_features_vggt.yaml

require_slurm_env() {
    if [ -z "${SLURM_ACCOUNT:-}" ] || [ -z "${SLURM_PARTITION:-}" ]; then
        cat >&2 <<'EOF'
Set the SLURM account and GPU partition first, with the real names:
  export SLURM_ACCOUNT=myaccount SLURM_PARTITION=gpu
To find them:
  inside a running job:  echo "$SLURM_JOB_ACCOUNT $SLURM_JOB_PARTITION"
  your accounts:         sacctmgr -nP show assoc user=$USER format=account
  gpu partitions:        sinfo -o "%P %G" | grep -i gpu
EOF
        exit 1
    fi
}

run_lot() {
    PYTHONPATH="$REPO_ROOT/src" "$MM" run -n "$LOT_ENV" "$@"
}

# Encoder name and cache root come from the config, so the two never disagree.
config_field() {
    run_lot python -c 'import sys
from lot.encoders import load_cache_config
cfg = load_cache_config(sys.argv[1])
print(getattr(cfg, sys.argv[2]))' "$1" "$2"
}

cache_meta_path() {
    echo "$REPO_ROOT/$(config_field "$1" cache_root)/$(config_field "$1" encoder)/$2/meta.json"
}

# Indices of the config's scenes that have no cache yet, comma separated.
missing_indices() {
    local config="$1" root encoder
    root="$(config_field "$config" cache_root)"
    encoder="$(config_field "$config" encoder)"
    local out=""
    while read -r index scene; do
        [ -n "$scene" ] || continue
        if [ ! -f "$REPO_ROOT/$root/$encoder/$scene/meta.json" ]; then
            out="$out$index,"
        fi
    done < <(run_lot python -m lot.encoders --config "$config" --list-scenes)
    echo "${out%,}"
}

submit() {
    sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
        --gres "${LOT_GPU_GRES:-gpu:1}" \
        --export "ALL,LOT_ENV=$LOT_ENV,MAMBA_ROOT_PREFIX=$MAMBA_ROOT_PREFIX,MAMBA_EXE=$MM" \
        "$@"
}

submit_batch() {
    local config="$1" label="$2"
    require_slurm_env
    local pilot_meta
    pilot_meta="$(cache_meta_path "$PILOT_CONFIG" room_0)"
    if [ ! -f "$pilot_meta" ]; then
        echo "the pilot scene is not cached yet ($pilot_meta missing)." >&2
        echo "PLAN.md accepts the batch only after the pilot passes;" >&2
        echo "run './scripts/run_phase2_cache.sh pilot' first." >&2
        exit 1
    fi
    local missing
    missing="$(missing_indices "$config")"
    if [ -z "$missing" ]; then
        echo "every scene in $config is already cached; nothing to submit."
        echo "check them: ./scripts/run_phase2_cache.sh validate $config"
        exit 0
    fi
    submit --array "$missing" scripts/cache_features.sbatch "$config"
    cat <<EOF
$label submitted for scene indices: $missing
Afterwards:
  ./scripts/run_phase2_cache.sh validate $config
EOF
}

cd "$REPO_ROOT"
case "$MODE" in
pilot)
    require_slurm_env
    PILOT_META="$(cache_meta_path "$PILOT_CONFIG" room_0)"
    if [ -f "$PILOT_META" ]; then
        echo "pilot cache exists at $PILOT_META; caches are never overwritten." >&2
        echo "Delete that directory to re-encode." >&2
        exit 1
    fi
    submit scripts/cache_features.sbatch "$PILOT_CONFIG"
    cat <<EOF
Pilot submitted. When it finishes (squeue -u \$USER), check acceptance:
  1. The job log ends with 768 channels and a 37x37 grid for 518 px frames.
  2. Validation: ./scripts/run_phase2_cache.sh validate $PILOT_CONFIG
Then run the batch: ./scripts/run_phase2_cache.sh all
EOF
    ;;
all)
    submit_batch "$ALL_CONFIG" "Batch"
    ;;
vggt)
    submit_batch "$VGGT_CONFIG" "VGGT batch"
    ;;
validate)
    run_lot python -m lot.encoders --config "${2:-$ALL_CONFIG}" --validate-only
    ;;
*)
    echo "unknown mode '$MODE'; use pilot, all, vggt, or validate" >&2
    exit 1
    ;;
esac
