#!/bin/bash
# SLURM template for Phase 5 training: one task per (fold, seed).
#
# Three folds and three seeds is nine tasks, indexed 0 to 8 with
#   fold = index / 3, seed_index = index % 3
# computed inside lot.train's entry point from the frozen seed list, so the
# mapping lives with the frozen config rather than in this template.
#
#   sbatch --account "${SLURM_ACCOUNT:?}" \
#          --partition "${SLURM_PARTITION:?}" \
#          --array 0-8 scripts/phase5_train.sbatch configs/phase5.yaml
#
# Run scripts/run_phase5.sh check, then overfit, first. The tiny-subset overfit
# gate is a stop condition: a predictor that cannot fit eight pairs cannot
# support a scientific reading of its later underperformance, so training
# refuses to start until it has passed. This worker re-checks both receipts
# itself rather than trusting the launcher.
#
# GPU shape, unlike Phase 4's evaluation array. The job reads cached features
# and cached VGGT depth and never loads an encoder, but it does train.
#
#SBATCH --job-name=lot-phase5-train
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch ... scripts/phase5_train.sbatch <config.yaml>}"
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

# Refuse a dirty worktree. Every checkpoint must name the commit that made it.
if [ -n "$(git status --porcelain)" ]; then
    echo "refusing to run from a dirty worktree:" >&2
    git status --porcelain | head -20 >&2
    exit 1
fi
echo "clean worktree at commit $(git rev-parse HEAD)"

# The worker enforces the prerequisites itself rather than trusting that it was
# reached through run_phase5.sh. This file is documented as a direct sbatch
# target, so the launcher is not the real execution boundary; a job submitted by
# hand would otherwise train with neither gate having passed. Both receipts are
# checked here, bound to this commit, configuration, and resolved inputs.
#
# The directory is derived from the config this job was given, not hard-coded.
# A fixed path would read another run's receipts whenever the config named a
# different experiment, which is exactly the confusion the binding exists to
# prevent.
EVIDENCE_DIR="$("$MM" run -n "$LOT_ENV" python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'src')
from lot.phase5 import load_phase5_config
print(load_phase5_config(Path('$CONFIG')).evidence_dir)
")"
echo "receipts expected under $EVIDENCE_DIR"
GATE_RECEIPT="$EVIDENCE_DIR/integration_gate.json"
for receipt in "integration_gate:integration:the Borah integration gate" \
               "tiny_overfit:overfit:the tiny-subset overfit gate"; do
    name="${receipt%%:*}"
    rest="${receipt#*:}"
    kind="${rest%%:*}"
    label="${rest#*:}"
    file="$EVIDENCE_DIR/$name.json"
    if [ ! -f "$file" ]; then
        echo "$label has not been run; $file is absent." >&2
        echo "training is not permitted until both gates pass." >&2
        exit 1
    fi
    "$MM" run -n "$LOT_ENV" python -m lot.phase5_receipt \
        --receipt "$file" --config "$CONFIG" --label "$label" \
        --kind "$kind" --gate-receipt "$GATE_RECEIPT" || exit 1
done

# One task per (fold, seed). The task never sees a test scene: the fold object
# exposes train and val only, and lot.train asserts the seal inside its loop.
"$MM" run -n "$LOT_ENV" python -m lot.phase5 \
    --config "$CONFIG" \
    --mode train \
    --task-index "${SLURM_ARRAY_TASK_ID:-0}"
