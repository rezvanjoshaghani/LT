#!/bin/bash
# Stream D: the corrected Phase 3 re-run, on the cluster.
#
# Renders and cached features are untouched by the repair and are reused as
# they stand. What is regenerated is everything downstream of them:
# correspondence sampling, the nulls, the scoring, and the parquet.
#
#   ./scripts/run_stream_d.sh check     # env and suite, no data touched
#   ./scripts/run_stream_d.sh evaluate  # the re-run
#   ./scripts/run_stream_d.sh counts    # support counts only, no outcome values
#   ./scripts/run_stream_d.sh figures   # table and the four figures
#
# The order matters. PROTOCOL 3.4 permits support thresholds to be set from
# counts alone and never from outcome values, so `counts` comes before
# `figures` deliberately: it prints n and nothing else, and the threshold
# decision is meant to be made from that view before any margin is visible.
#
# Environment:
#   LOT_ENV   micromamba env with torch and pyarrow (default lot-encode)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-check}"
LOT_ENV="${LOT_ENV:-lot-encode}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

if command -v micromamba >/dev/null 2>&1; then
    MM="$(command -v micromamba)"
else
    MM="$HOME/.local/bin/micromamba"
fi
[ -x "$MM" ] || { echo "micromamba not found" >&2; exit 1; }

CONFIG="configs/experiment_zero.yaml"
RUN_DIR="outputs/experiment_zero"
EVAL_DIR="$RUN_DIR/eval"

run_lot() {
    PYTHONPATH="$REPO_ROOT/src" "$MM" run -n "$LOT_ENV" "$@"
}

cd "$REPO_ROOT"
case "$MODE" in
check)
    echo "== branch =="
    git rev-parse --abbrev-ref HEAD
    git log --oneline -1
    echo
    echo "== inputs present =="
    for path in data/replica_renders cache/features configs/analysis.yaml; do
        printf "  %-28s " "$path"
        [ -e "$path" ] && echo present || echo "MISSING"
    done
    echo
    echo "== frame stats sidecars (reused, not regenerated) =="
    ls -1 data/replica_renders/*/frame_stats.json 2>/dev/null | wc -l
    echo
    echo "== suite, with the permanent encoder tests ungated =="
    LOT_ENCODER_SMOKE=1 run_lot python -m pytest tests/ -q
    ;;
evaluate)
    if [ -d "$EVAL_DIR" ]; then
        echo "$EVAL_DIR exists. The schema changed, so the previous run cannot be" >&2
        echo "mixed with this one. Move it aside first, for example:" >&2
        echo "  mv $RUN_DIR ${RUN_DIR}_precorrection" >&2
        exit 1
    fi
    run_lot python -m lot.evaluate --config "$CONFIG" --resume
    echo
    echo "Per-scene run metadata is beside each parquet as <scene>.meta.json."
    ;;
counts)
    run_lot python -m lot.figures --eval-dir "$EVAL_DIR" --counts-only
    ;;
figures)
    run_lot python -m lot.figures --eval-dir "$EVAL_DIR"
    ;;
*)
    echo "unknown mode '$MODE'; use check, evaluate, counts, or figures" >&2
    exit 1
    ;;
esac
