#!/bin/bash
# Phase 4: the estimated-geometry tax, on the cluster. PROTOCOL 4.x, executed
# in the order the execution plan freezes. Every evidence-producing mode
# refuses a dirty worktree and verifies the frozen artifacts first.
#
#   ./scripts/run_phase4.sh check       # pins, hashes, amendments, inputs, suite
#   ./scripts/run_phase4.sh inspect     # record installed VGGT depth semantics
#   ./scripts/run_phase4.sh convention  # PROTOCOL 4.1 depth-convention report
#   ./scripts/run_phase4.sh gates      # PROTOCOL 4.5 pure-rotation gates, all scenes
#   ./scripts/run_phase4.sh evaluate    # the full run, one process (or use the array)
#   ./scripts/run_phase4.sh figures     # ladder table and the four figures
#   ./scripts/run_phase4.sh smoke       # permanent VGGT real-weight tests, GPU node
#   ./scripts/run_phase4.sh validate23  # validator 2.3 against the real apartment_0
#
# The order matters: check before convention, convention before gates, gates
# before evaluate, evaluate before figures. Addendum E work (smoke, validate23)
# runs in the same cluster period and writes evidence without touching state.
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

CONFIG="configs/phase4.yaml"
RUN_DIR="outputs/phase4_rung1"
FREEZE_COMMIT="d4ed1017bd2daca2871da28900b5b4a6a7ff92b6"

run_lot() {
    PYTHONPATH="$REPO_ROOT/src" "$MM" run -n "$LOT_ENV" "$@"
}

require_clean_tree() {
    if [ -n "$(git status --porcelain)" ]; then
        echo "refusing to run from a dirty worktree:" >&2
        git status --porcelain | head -20 >&2
        exit 1
    fi
    echo "clean worktree at commit $(git rev-parse HEAD)"
}

verify_freeze() {
    # The frozen artifacts must still hash to FREEZE.md's values at the freeze
    # commit, and the live copies must be the frozen ones plus the enumerated
    # amendments. A frozen-blob mismatch is a stop before anything runs.
    local fail=0
    while read -r want file; do
        got="$(git show "$FREEZE_COMMIT:$file" | sha256sum | cut -d' ' -f1)"
        if [ "$got" != "$want" ]; then
            echo "FROZEN BLOB MISMATCH: $file $got != $want" >&2
            fail=1
        fi
    done <<'HASHES'
517cc4924f8b770c2d27c9f9fecb761c634a920fd21a77ab44e92fe28473eee4 PROTOCOL.md
bddd31e9d294778f10848028e4755ba56e6ea58f1e633b57eb4d09b29bbb4493 AMENDMENTS.md
91f3a82ff066bcb029b9df7b9ff9c3da4bba31ea9604d1d17cea48a660b440e6 configs/analysis.yaml
6e4897ff98b2061a4e8c544dfd30126e8b0b5cb561d4b8e306400f210a7c452e VALIDATION.md
HASHES
    [ "$fail" -eq 0 ] || exit 1
    echo "frozen blobs verified against FREEZE.md at $FREEZE_COMMIT"
    echo "post-freeze amendments in force:"
    grep -E '^### A[0-9]+' AMENDMENTS.md || echo "  none"
    echo "live sha256, recorded beside the frozen ones:"
    sha256sum PROTOCOL.md AMENDMENTS.md configs/analysis.yaml VALIDATION.md
}

cd "$REPO_ROOT"
case "$MODE" in
check)
    verify_freeze
    echo
    echo "== branch and commit =="
    git rev-parse --abbrev-ref HEAD
    git log --oneline -1
    require_clean_tree
    echo
    echo "== inputs present =="
    for path in data/replica_renders cache/features configs/analysis.yaml \
                outputs/experiment_zero/eval; do
        printf "  %-32s " "$path"
        [ -e "$path" ] && echo present || echo "MISSING"
    done
    echo
    echo "== cache provenance (features and depth, digests re-read) =="
    for ENCODER_CONFIG in configs/cache_features_all.yaml configs/cache_features_vggt.yaml; do
        run_lot python -m lot.encoders --config "$ENCODER_CONFIG" --validate-only
    done
    echo
    echo "== identities =="
    run_lot python - <<'PY'
from lot.analysis_config import load_analysis_config
from lot.phase4 import phase4_measurement_digest
cfg = load_analysis_config()
print("phase3 measurement digest:", cfg.measurement_digest())
print("phase4 measurement digest:", phase4_measurement_digest(cfg))
print("config digest:", cfg.digest())
PY
    echo
    echo "== suite (encoder smoke tests run in the smoke mode on a GPU node) =="
    run_lot python -m pytest tests/ -q
    ;;
inspect)
    require_clean_tree
    # Step 4's primary authority: what the installed VGGT source says its
    # depth head produces. The excerpt lands in evidence for the human
    # decision; pass the conclusion to convention as --doc-verdict.
    mkdir -p "$RUN_DIR/evidence"
    run_lot python - <<'PY' | tee "$RUN_DIR/evidence/vggt_source_inspection.txt"
import inspect, pathlib
try:
    import vggt
except ImportError:
    raise SystemExit("vggt is not installed in this environment")
root = pathlib.Path(vggt.__file__).parent
print("vggt package at", root)
for rel in ("heads/dpt_head.py", "heads/head_act.py", "utils/geometry.py"):
    path = root / rel
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    print(f"\n===== {rel} =====")
    for i, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        if any(k in low for k in ("depth", "unproject", "point_map", "pointmap", "ray")):
            print(f"{i:5d}: {line}")
PY
    echo "inspection -> $RUN_DIR/evidence/vggt_source_inspection.txt"
    ;;
convention)
    require_clean_tree
    # Pass the documented convention when the inspect step settled it:
    #   ./scripts/run_phase4.sh convention planar_z
    DOC="${2:-}"
    if [ -n "$DOC" ]; then
        run_lot python -m lot.phase4 --config "$CONFIG" --convention --doc-verdict "$DOC"
    else
        run_lot python -m lot.phase4 --config "$CONFIG" --convention
    fi
    ;;
gates)
    require_clean_tree
    run_lot python -m lot.phase4 --config "$CONFIG" --gates-only
    ;;
evaluate)
    require_clean_tree
    run_lot python -m lot.phase4 --config "$CONFIG" --resume
    ;;
figures)
    require_clean_tree
    run_lot python -m lot.phase4_report --eval-dir "$RUN_DIR/eval"
    ;;
smoke)
    require_clean_tree
    # The permanent real-weight encoder tests, PROTOCOL 3.1, on a GPU node.
    mkdir -p "$RUN_DIR/evidence"
    export LOT_ENCODER_SMOKE=1
    run_lot python -m pytest tests/test_encoder_cache.py \
        -k "vggt_batching or grid_orientation or sees_one_frame" -q \
        | tee "$RUN_DIR/evidence/encoder_smoke.txt"
    ;;
validate23)
    require_clean_tree
    # Stream F's remaining centerpiece: the independent pixels-to-rows
    # reproduction of apartment_0 against the corrected Phase 3 parquet.
    # Validator code; writes evidence only.
    mkdir -p validation/evidence/reaudit
    run_lot python validation/borah_check_2_3.py \
        --renders data/replica_renders --cache cache/features \
        --eval-dir outputs/experiment_zero/eval --scene apartment_0 \
        --out validation/evidence/reaudit/borah_check_2_3.json
    ;;
*)
    echo "unknown mode '$MODE'; use check, inspect, convention, gates, evaluate, figures, smoke, or validate23" >&2
    exit 1
    ;;
esac
