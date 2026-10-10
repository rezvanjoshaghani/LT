#!/bin/bash
# Phase 5, rung 2: the learned-versus-explicit transformation limitation.
#
# The run sequence is rigid and each mode refuses to run out of order:
#
#   ./scripts/run_phase5.sh check       # Borah integration gate, no side effects
#   ./scripts/run_phase5.sh overfit     # tiny-subset adequacy gate, a stop condition
#   ./scripts/run_phase5.sh train       # nine (fold, seed) tasks, one array
#   ./scripts/run_phase5.sh controls    # pose and depth shuffle, validation only
#   ./scripts/run_phase5.sh lock        # bind checkpoints, records, controls by sha256
#   ./scripts/run_phase5.sh evaluate    # test scenes, one array task per scene
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
# overfit, train, controls, and evaluate are submitted to SLURM rather than run
# here, so the launcher itself never trains or evaluates on a login node. Each
# job's output lands under the run's evidence directory. The entry point checks
# every mode's gate receipts again inside the job, so a job cannot run past a
# gate that stopped standing while it waited in the queue.
#
# lock runs here. It loads and hashes the nine checkpoints, their training
# records, and the controls file, and reads no scene. Its receipt,
# checkpoint_lock_{level}.json, binds them all by sha256, and evaluate refuses
# to start without a lock that still matches the live files.
#
# Arguments after the mode are passed to the entry point. The one in use is
# --level, for a sensitivity or diagnostic alignment level, which the entry
# point refuses until the primary evaluation is complete:
#
#   ./scripts/run_phase5.sh lock --level affine
#   ./scripts/run_phase5.sh evaluate --level affine
#
# Environment:
#   LOT_ENV            micromamba env with torch and pyarrow (default lot-encode)
#   SLURM_ACCOUNT      required only by the modes that submit jobs
#   SLURM_PARTITION    same

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-check}"
[ "$#" -gt 0 ] && shift
# Passed through to the entry point. Expanded with the +-guard below because
# bash before 4.4 treats an empty array as unset under nounset.
EXTRA=("$@")
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

# A receipt is only evidence about the state that produced it. Checking that it
# says PASS, and nothing else, lets a later commit or an edited configuration
# inherit an older run's verdict: the gate would have verified artifacts and
# shapes for code that is no longer the code about to train. So every receipt is
# bound to the run identity, and a mismatch is refused with the field named.
require_receipt() {
    # kind selects which bindings apply. The integration receipt is bound to the
    # resolved inputs it hashed; the overfit receipt carries no such block and is
    # bound instead to the digest of the integration receipt that licensed it.
    # Any further arguments go to the verifier.
    local receipt="$1" label="$2" advice="$3" kind="$4"
    shift 4
    if [ ! -f "$receipt" ]; then
        echo "$label has not been run; $receipt is absent." >&2
        echo "$advice" >&2
        exit 1
    fi
    run_lot python -m lot.phase5_receipt \
        --receipt "$receipt" --config "$CONFIG" --label "$label" \
        --kind "$kind" --gate-receipt "$GATE_RECEIPT" "$@" || exit 1
}

require_gate_passed() {
    require_receipt "$GATE_RECEIPT" "the Borah integration gate" \
        "run './scripts/run_phase5.sh check' first." integration
}

require_overfit_passed() {
    require_receipt "$OVERFIT_RECEIPT" "the tiny-subset overfit gate" \
        "run './scripts/run_phase5.sh overfit' first." overfit
}

# The alignment level the entry point will run at, read through the entry
# point's own argument parser: --level when it is passed after the mode, the
# frozen primary level otherwise. A malformed argument fails here, as it would
# in the entry point.
resolved_level() {
    run_lot python -c '
import sys
from lot.phase5 import build_parser, load_phase5_config
args = build_parser().parse_args(sys.argv[1:])
print(args.level or load_phase5_config(args.config).primary_alignment_level)
' --config "$CONFIG" --mode describe ${EXTRA[@]+"${EXTRA[@]}"}
}

# The checkpoint lock is bound to both gate receipts and to every file it
# names, so the verifier hashes all of them again before anything is submitted.
require_lock_written() {
    local level="$1"
    require_receipt "$EVIDENCE_DIR/checkpoint_lock_${level}.json" \
        "the checkpoint lock for level $level" \
        "run './scripts/run_phase5.sh lock --level $level' first." lock \
        --overfit-receipt "$OVERFIT_RECEIPT" --level "$level"
}

require_slurm() {
    : "${SLURM_ACCOUNT:?set SLURM_ACCOUNT before submitting}"
    : "${SLURM_PARTITION:?set SLURM_PARTITION before submitting}"
}

# Submit one Phase 5 job. The first argument is its output file under the
# evidence directory, as an sbatch filename pattern: %j for a single job, %A_%a
# for an array. The rest go to sbatch, then the template and its arguments.
# Account and partition come from the environment only.
submit() {
    local output="$1"
    shift
    sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
        --output "$EVIDENCE_DIR/$output.txt" \
        "$@"
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
    require_slurm
    echo "=== Stream U step 14: tiny-subset overfit gate ==="
    submit tiny_overfit_%j scripts/phase5_job.sbatch "$CONFIG" overfit \
        ${EXTRA[@]+"${EXTRA[@]}"}
    echo "the verdict is written to $OVERFIT_RECEIPT and printed in the job's"
    echo "output under $EVIDENCE_DIR. A FAIL is a stop: nothing after it runs."
    ;;

train)
    require_clean_tree
    verify_freeze
    require_gate_passed
    require_overfit_passed
    require_slurm
    echo "submitting nine (fold, seed) tasks"
    submit train_%A_%a --array 0-8 scripts/phase5_train.sbatch "$CONFIG" \
        ${EXTRA[@]+"${EXTRA[@]}"}
    ;;

controls)
    # Validation scenes only, through every trained checkpoint. Run after all
    # nine training tasks have finished; a missing checkpoint is reported and
    # the job exits nonzero rather than writing a partial result as complete.
    require_clean_tree
    require_gate_passed
    require_overfit_passed
    require_slurm
    submit input_use_controls_%j scripts/phase5_job.sbatch "$CONFIG" controls \
        ${EXTRA[@]+"${EXTRA[@]}"}
    ;;

lock)
    # After the controls, before evaluate. reporting_rules.md section 7. It
    # refuses, listing every problem, unless each checkpoint loads through its
    # training record, each record names this run, and the controls file is
    # complete and ran on these checkpoints. Light enough to run here. Each run
    # keeps its own output file under the evidence directory.
    require_clean_tree
    require_gate_passed
    require_overfit_passed
    level="$(resolved_level)"
    log="$EVIDENCE_DIR/checkpoint_lock_${level}_$(date -u +%Y%m%dT%H%M%SZ).txt"
    echo "=== checkpoint lock, level $level ==="
    set +e
    run_lot python -m lot.phase5 --config "$CONFIG" --mode lock \
        ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$log"
    # The status is read from PIPESTATUS, as in check.
    status="${PIPESTATUS[0]}"
    set -e
    if [ "$status" -ne 0 ]; then
        echo
        echo "checkpoint lock refused; evaluate may not run. Output kept in $log" >&2
        exit "$status"
    fi
    echo "output kept in $log"
    ;;

evaluate)
    # The only mode that touches test scenes, and it runs only once the
    # checkpoints are locked: the lock of its level must still match the live
    # files. Nothing it prints may be fed back into training or selection.
    # One array task per test scene. The range is read from the frozen folds,
    # so it cannot disagree with the order the entry point indexes.
    require_clean_tree
    require_gate_passed
    require_overfit_passed
    level="$(resolved_level)"
    require_lock_written "$level"
    require_slurm
    last="$(run_lot python -c 'from lot.phase5_folds import frozen_folds
from lot.phase5_modes import evaluation_scenes
print(len(evaluation_scenes(frozen_folds())) - 1)')"
    echo "submitting $((last + 1)) test scenes"
    submit evaluate_%A_%a --array "0-$last" scripts/phase5_job.sbatch "$CONFIG" evaluate \
        ${EXTRA[@]+"${EXTRA[@]}"}
    ;;

tables|figures|acceptance)
    require_clean_tree
    require_gate_passed
    echo "mode '$MODE' is not implemented yet." >&2
    echo "It is part of Streams AC and AD. reporting_rules.md decision 1 requires" >&2
    echo "the reporting code, with the outcome and wording code in" >&2
    echo "src/lot/phase5_report.py, to be committed before the chain runs once at" >&2
    echo "one commit E. The chain, from check through evaluate, waits for it." >&2
    exit 2
    ;;

*)
    echo "unknown mode '$MODE'" >&2
    echo "use: check, overfit, train, controls, lock, evaluate, tables, figures, acceptance" >&2
    exit 1
    ;;
esac
