#!/bin/bash
# SLURM template for the path-agreement ledger (one task per scene), the same
# CPU shape as evaluation:
#
#   sbatch --account "${SLURM_ACCOUNT:-$SLURM_JOB_ACCOUNT}" \
#          --partition "${SLURM_PARTITION:-$SLURM_JOB_PARTITION}" \
#          --array 0-17 scripts/path_ledger.sbatch configs/experiment_zero.yaml
#
#SBATCH --job-name=lot-ledger
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
CONFIG="${1:?usage: sbatch ... scripts/path_ledger.sbatch <config.yaml>}"
LOT_ENV="${LOT_ENV:-lot-encode}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
MM="${MAMBA_EXE:-}"
if [ -z "$MM" ]; then
    if command -v micromamba >/dev/null 2>&1; then MM="$(command -v micromamba)"; else MM="$HOME/.local/bin/micromamba"; fi
fi
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$MM" run -n "$LOT_ENV" python -m lot.path_ledger \
    --config "$CONFIG" \
    --scene-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --out validation/evidence/path_agreement_ledger
