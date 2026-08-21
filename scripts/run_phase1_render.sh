#!/bin/bash
# Submit the Phase 1 rendering experiment on Borah. Run from anywhere inside
# the repository after scripts/setup_borah.sh has completed.
#
# Usage:
#   ./scripts/run_phase1_render.sh pilot              # render room_0 end to end
#   ./scripts/run_phase1_render.sh smoke              # tiny habitat pytest render
#   ./scripts/run_phase1_render.sh all                # 18-scene batch (array job)
#   ./scripts/run_phase1_render.sh validate [config]  # validate manifests, no GPU
#   ./scripts/run_phase1_render.sh qc [config]        # regenerate QC sheets, no GPU
#
# Required for pilot, smoke, and all (never hard-coded, per CLAUDE.md):
#   SLURM_ACCOUNT     SLURM account to charge
#   SLURM_PARTITION   GPU partition name
# Optional:
#   LOT_ENV           micromamba env name       (default lot-render)
#   MAMBA_ROOT_PREFIX micromamba root           (default $HOME/micromamba)
#   LOT_GPU_GRES      gres string               (default gpu:1)
#   LOT_MODULES       modules the sbatch job should load, space separated
#
# Phase 1 acceptance order (PLAN.md): pilot first, inspect the QC sheets and
# the depth-convention verdict, then the batch. This script enforces that
# order: 'all' refuses to submit until the pilot manifest exists.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-pilot}"
LOT_ENV="${LOT_ENV:-lot-render}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

if command -v micromamba >/dev/null 2>&1; then
    MM="$(command -v micromamba)"
else
    MM="$HOME/.local/bin/micromamba"
fi
[ -x "$MM" ] || { echo "micromamba not found; run scripts/setup_borah.sh first" >&2; exit 1; }
[ -d "$MAMBA_ROOT_PREFIX/envs/$LOT_ENV" ] || {
    echo "env $LOT_ENV not found under $MAMBA_ROOT_PREFIX; run scripts/setup_borah.sh" >&2
    exit 1
}

PILOT_CONFIG=configs/render_replica_pilot.yaml
ALL_CONFIG=configs/render_replica_all.yaml
PILOT_MANIFEST="$REPO_ROOT/data/replica_renders_pilot/room_0/manifest.json"

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

submit() {
    sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
        --gres "${LOT_GPU_GRES:-gpu:1}" \
        --export "ALL,LOT_ENV=$LOT_ENV,MAMBA_ROOT_PREFIX=$MAMBA_ROOT_PREFIX,MAMBA_EXE=$MM" \
        "$@"
}

cd "$REPO_ROOT"
case "$MODE" in
pilot)
    require_slurm_env
    if [ -f "$PILOT_MANIFEST" ]; then
        echo "pilot output exists at $PILOT_MANIFEST; outputs are never" >&2
        echo "overwritten. Delete data/replica_renders_pilot/room_0 to re-run." >&2
        exit 1
    fi
    submit scripts/render_replica.sbatch "$PILOT_CONFIG"
    cat <<EOF
Pilot submitted. When it finishes (squeue -u \$USER), check acceptance:
  1. QC sheets:      data/replica_renders_pilot/room_0/qc/qc_{rotation,translation,orbit}.png
  2. Depth verdict:  grep -A3 raw_verdict data/replica_renders_pilot/room_0/manifest.json
  3. Validation:     ./scripts/run_phase1_render.sh validate $PILOT_CONFIG
Then run the batch: ./scripts/run_phase1_render.sh all
EOF
    ;;
smoke)
    require_slurm_env
    REPLICA="$(cd "$REPO_ROOT/data/replica" 2>/dev/null && pwd -P)" || {
        echo "data/replica missing; run scripts/setup_borah.sh" >&2; exit 1; }
    submit --job-name lot-smoke --time 00:30:00 --cpus-per-task 2 --mem 8G \
        --output "slurm-lot-smoke-%j.out" \
        --wrap "cd '$REPO_ROOT' && ${LOT_MODULES:+module load $LOT_MODULES &&} \
REPLICA_ROOT='$REPLICA' PYTHONPATH='$REPO_ROOT/src' \
'$MM' run -n '$LOT_ENV' pytest tests/test_render_replica.py -k habitat -v"
    echo "smoke test submitted; see slurm-lot-smoke-<jobid>.out"
    ;;
all)
    require_slurm_env
    if [ ! -f "$PILOT_MANIFEST" ]; then
        echo "pilot has not run yet ($PILOT_MANIFEST missing)." >&2
        echo "PLAN.md accepts the batch only after the pilot QC passes;" >&2
        echo "run './scripts/run_phase1_render.sh pilot' first." >&2
        exit 1
    fi
    submit --array 0-17 scripts/render_replica.sbatch "$ALL_CONFIG"
    cat <<EOF
Batch submitted (18 array tasks, one scene each). Afterwards:
  ./scripts/run_phase1_render.sh validate
QC sheets land in data/replica_renders/<scene>/qc/.
EOF
    ;;
validate)
    CONFIG="${2:-$ALL_CONFIG}"
    PYTHONPATH="$REPO_ROOT/src" "$MM" run -n "$LOT_ENV" \
        python -m lot.render_replica --config "$CONFIG" --validate-only
    ;;
qc)
    CONFIG="${2:-$ALL_CONFIG}"
    PYTHONPATH="$REPO_ROOT/src" "$MM" run -n "$LOT_ENV" \
        python -m lot.render_replica --config "$CONFIG" --qc-only
    ;;
*)
    echo "unknown mode '$MODE'; use pilot, smoke, all, validate, or qc" >&2
    exit 1
    ;;
esac
