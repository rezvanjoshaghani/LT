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
    echo "== cache provenance =="
    for ENCODER_CONFIG in configs/cache_features_all.yaml configs/cache_features_vggt.yaml; do
        run_lot python -m lot.encoders --config "$ENCODER_CONFIG" --validate-only
    done
    echo
    echo "== suite =="
    # LOT_ENCODER_SMOKE is deliberately not set. It ungates two tests that load
    # real weights, and they fall back to CPU when CUDA is absent, so on a login
    # node they would run VGGT-1B on the CPU. Those tests run inside the caching
    # job, which holds a GPU. The validation pass above is what this step needs:
    # it re-reads every cached array and checks the content hash recorded when
    # the cache was written.
    run_lot python -m pytest tests/ -q
    ;;
evaluate)
    # An existing directory is either this run's, interrupted, or an older run's.
    # The two need opposite treatment and only the evaluator can tell them apart:
    # it compares each parquet's stored run record against this run's config and
    # refuses a mismatch, so --resume continues an interruption and stops on a
    # mixed directory. Refusing here on existence alone made the documented
    # wrapper unable to resume the run it was invoking with --resume.
    if [ -d "$EVAL_DIR" ]; then
        echo "$EVAL_DIR exists. Finished scenes will be skipped and their run"
        echo "records checked against this run. If these results predate the"
        echo "repair, move them aside instead:"
        echo "  mv $RUN_DIR ${RUN_DIR}_precorrection"
        echo
    fi
    run_lot python -m lot.evaluate --config "$CONFIG" --resume
    echo
    echo "Per-scene run metadata is inside each parquet and beside it as <scene>.meta.json."
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
