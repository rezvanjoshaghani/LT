#!/bin/bash
# SLURM template for Replica rendering (PLAN Phase 1).
# Account and partition are never hard-coded; pass them from the environment
# at submit time:
#
#   export SLURM_ACCOUNT=...   SLURM_PARTITION=...   LOT_CONDA_ENV=lot
#   # pilot, one scene:
#   sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
#          scripts/render_replica.sbatch configs/render_replica_pilot.yaml
#   # full batch, 18 scenes:
#   sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
#          --array 0-17 scripts/render_replica.sbatch configs/render_replica_all.yaml
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
: "${LOT_CONDA_ENV:?set LOT_CONDA_ENV to the conda env that has habitat-sim}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LOT_CONDA_ENV"

cd "$SLURM_SUBMIT_DIR"
python -m lot.render_replica \
    --config "$CONFIG" \
    --scene-index "${SLURM_ARRAY_TASK_ID:-0}"
