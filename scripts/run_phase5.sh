#!/bin/bash
# Phase 5, rung 2: the learned-versus-explicit transformation limitation.
#
# The run sequence is rigid and each mode refuses to run out of order:
#
#   ./scripts/run_phase5.sh check       # Borah integration gate, no side effects
#   ./scripts/run_phase5.sh overfit     # tiny-subset adequacy gate, a stop condition
#   ./scripts/run_phase5.sh train       # nine (fold, seed) tasks, or use the array
#   ./scripts/run_phase5.sh controls    # pose and depth shuffle, validation only
#   ./scripts/run_phase5.sh evaluate    # test scenes, after checkpoints are locked
#   ./scripts/run_phase5.sh tables      # Stream AC tables
#   ./scripts/run_phase5.sh figures     # Stream AC figures
#   ./scripts/run_phase5.sh acceptance  # Stream AD, re-derived from artifacts
#
# check is a hard integration gate with no training side effects. It resolves
# and hashes the real artifacts, verifies the real schemas against what the
# frozen loaders assume, builds real scene inputs and one real example per
# regime, proves the leakage and support rules on real data, dry-runs one real
# batch through forward, loss, and backward, and completes the deferred pin.
# Nothing after it may run until it passes.
#
# The Phase 5 scientific design is frozen. No mode here changes a comparator
# definition, an information-access rule, a support, an estimand, a fold, the
# architecture, or a reporting rule. A genuine implementation or protocol
# mismatch found by check is reported as such and resolved deliberately, as an
# amendment with a written rationale, never by editing the design in place.
#
# Environment:
#   LOT_ENV            micromamba env with torch and pyarrow (default lot-encode)
#   SLURM_ACCOUNT      required only by the modes that submit jobs
#   SLURM_PARTITION    same

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-check}"
LOT_ENV="${LOT_ENV:-lot-encode}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

CONFIG="configs/phase5.yaml"
RUN_DIR="outputs/phase5_rung2"
EVIDENCE_DIR="$RUN_DIR/evidence"
FREEZE_COMMIT="d4ed1017bd2daca2871da28900b5b4a6a7ff92b6"
GATE_RECEIPT="$EVIDENCE_DIR/integration_gate.json"
OVERFIT_RECEIPT="$EVIDENCE_DIR/tiny_overfit.json"

if command -v micromamba >/dev/null 2>&1; then
    MM="$(command -v micromamba)"
else
    MM="$HOME/.local/bin/micromamba"
fi
[ -x "$MM" ] || { echo "micromamba not found" >&2; exit 1; }

run_lot() {
    PYTHONPATH="$REPO_ROOT/src" "$MM" run -n "$LOT_ENV" "$@"
}

require_clean_tree() {
    if [ -n "$(git status --porcelain)" ]; then
        echo "refusing to run from a dirty worktree:" >&2
        git status --porcelain | head -20 >&2
        exit 1
    fi
}

verify_freeze() {
    # The frozen artifacts must still hash to FREEZE.md's values at the freeze
    # commit. A mismatch is a stop before anything else runs.
    local fail=0
    for file in PROTOCOL.md AMENDMENTS.md configs/analysis.yaml VALIDATION.md; do
        want="$(grep -E "  $file\$" FREEZE.md | head -1 | awk '{print $1}')"
        got="$(git show "$FREEZE_COMMIT:$file" | sha256sum | cut -d' ' -f1)"
        if [ -z "$want" ]; then
            echo "FREEZE.md carries no hash for $file" >&2
            fail=1
        elif [ "$got" != "$want" ]; then
            echo "frozen blob mismatch for $file:" >&2
            echo "  FREEZE.md says $want" >&2
            echo "  freeze commit  $got" >&2
            fail=1
        fi
    done
    [ "$fail" -eq 0 ] || { echo "frozen artifacts do not verify; stopping" >&2; exit 1; }
    echo "frozen blobs verified against FREEZE.md at $FREEZE_COMMIT"
}

print_identity() {
    echo "commit          $(git rev-parse HEAD)"
    echo "branch          $(git rev-parse --abbrev-ref HEAD)"
    echo "mode            $MODE"
    echo "config          $CONFIG"
    echo "env             $LOT_ENV"
    echo "python          $(run_lot python -c 'import sys; print(sys.version.split()[0])')"
    echo "torch           $(run_lot python -c 'import torch; print(torch.__version__)')"
    run_lot python -m lot.phase5 --config "$CONFIG" --mode describe
}

require_gate_passed() {
    if [ ! -f "$GATE_RECEIPT" ]; then
        echo "the Borah integration gate has not been run." >&2
        echo "run './scripts/run_phase5.sh check' first; $GATE_RECEIPT is absent." >&2
        exit 1
    fi
    if ! run_lot python -c "
import json, sys
report = json.load(open('$GATE_RECEIPT'))
sys.exit(0 if report.get('passed') else 1)
"; then
        echo "the Borah integration gate did not pass. Training is not permitted." >&2
        echo "see $GATE_RECEIPT" >&2
        exit 1
    fi
    echo "integration gate: PASS (from $GATE_RECEIPT)"
}

require_overfit_passed() {
    if [ ! -f "$OVERFIT_RECEIPT" ]; then
        echo "the tiny-subset overfit gate has not been run." >&2
        echo "run './scripts/run_phase5.sh overfit' first." >&2
        exit 1
    fi
    if ! run_lot python -c "
import json, sys
report = json.load(open('$OVERFIT_RECEIPT'))
sys.exit(0 if report.get('passed') else 1)
"; then
        echo "the tiny-subset overfit gate did not pass." >&2
        echo "Phase 5 stops here: a predictor that cannot fit the tiny sample" >&2
        echo "cannot support a scientific reading of its underperformance." >&2
        exit 1
    fi
    echo "tiny-subset overfit gate: PASS (from $OVERFIT_RECEIPT)"
}

mkdir -p "$EVIDENCE_DIR"

case "$MODE" in

check)
    # A hard integration gate. No training side effects: the only writes are
    # the gate's own evidence and the completed pin.
    require_clean_tree
    verify_freeze
    print_identity
    echo
    echo "=== Phase 5 Borah integration gate ==="
    set +e
    run_lot python -m lot.phase5 --config "$CONFIG" --mode check \
        2>&1 | tee "$EVIDENCE_DIR/integration_gate.txt"
    # errexit and pipefail would abort the script at the pipeline itself,
    # before the status could be read, so each tee'd pipeline runs with
    # errexit off and its status is taken from PIPESTATUS deliberately.
    status="${PIPESTATUS[0]}"
    set -e
    if [ "$status" -ne 0 ]; then
        echo
        echo "integration gate failed; nothing further may run." >&2
        exit "$status"
    fi
    ;;

overfit)
    require_clean_tree
    verify_freeze
    require_gate_passed
    echo
    echo "=== Stream U step 14: tiny-subset overfit gate ==="
    set +e
    run_lot python -m lot.phase5 --config "$CONFIG" --mode overfit \
        2>&1 | tee "$EVIDENCE_DIR/tiny_overfit.txt"
    status="${PIPESTATUS[0]}"
    set -e
    exit "$status"
    ;;

train)
    require_clean_tree
    verify_freeze
    require_gate_passed
    require_overfit_passed
    : "${SLURM_ACCOUNT:?set SLURM_ACCOUNT before submitting}"
    : "${SLURM_PARTITION:?set SLURM_PARTITION before submitting}"
    echo "submitting nine (fold, seed) tasks"
    sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
        --array 0-8 scripts/phase5_train.sbatch "$CONFIG"
    ;;

controls)
    require_clean_tree
    require_gate_passed
    set +e
    run_lot python -m lot.phase5 --config "$CONFIG" --mode controls \
        2>&1 | tee "$EVIDENCE_DIR/input_use_controls.txt"
    status="${PIPESTATUS[0]}"
    set -e
    exit "$status"
    ;;

evaluate)
    # The only mode that touches test scenes, and it runs after checkpoints are
    # locked. Nothing it prints may be fed back into training or selection.
    require_clean_tree
    require_gate_passed
    require_overfit_passed
    set +e
    run_lot python -m lot.phase5 --config "$CONFIG" --mode evaluate \
        2>&1 | tee "$EVIDENCE_DIR/evaluate.txt"
    status="${PIPESTATUS[0]}"
    set -e
    exit "$status"
    ;;

tables|figures|acceptance)
    require_clean_tree
    require_gate_passed
    echo "mode '$MODE' is not implemented yet." >&2
    echo "Streams AC and AD are the remaining work; they are deliberately not" >&2
    echo "built before the real-data integration gate passes, because their" >&2
    echo "inputs are the evaluation records the gate exists to make trustworthy." >&2
    exit 2
    ;;

*)
    echo "unknown mode '$MODE'" >&2
    echo "use: check, overfit, train, controls, evaluate, tables, figures, acceptance" >&2
    exit 1
    ;;
esac
