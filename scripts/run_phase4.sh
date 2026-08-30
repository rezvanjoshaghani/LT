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
import pathlib
try:
    import vggt
except ImportError:
    raise SystemExit("vggt is not installed in this environment")
# VGGT installs as a namespace package, so __file__ is None and the package
# directory has to come from __path__. Reading __file__ alone raised here.
roots = [pathlib.Path(p) for p in getattr(vggt, "__path__", [])]
if getattr(vggt, "__file__", None):
    roots.append(pathlib.Path(vggt.__file__).parent)
roots = sorted({r for r in roots if r.is_dir()})
if not roots:
    raise SystemExit("vggt imported but its package directory could not be located")
print("vggt package roots:", *[str(r) for r in roots], sep="\n  ")

# The decisive evidence is how VGGT itself turns predictions["depth"] into 3D
# points: a planar-z head assigns z = depth directly, while a ray-distance
# head must divide by the secant of the pixel angle first. Print those
# function bodies whole rather than grepping lines out of context.
DECISIVE = ("depth_to_cam_coords_points", "depth_to_world_coords_points")
KEYS = ("depth", "unproject", "pointmap", "point_map", "world_points", "ray")
for root in roots:
    for path in sorted(root.rglob("*.py")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for name in DECISIVE:
            for index, line in enumerate(lines):
                if line.startswith(f"def {name}"):
                    end = index + 1
                    while end < len(lines) and not lines[end].startswith("def "):
                        end += 1
                    print(f"\n===== {path.relative_to(root)}: {name} =====")
                    for offset, body in enumerate(lines[index:end], index + 1):
                        print(f"{offset:5d}: {body}")
        hits = [
            (i, l) for i, l in enumerate(lines, 1)
            if any(k in l.lower() for k in KEYS)
        ]
        if hits and path.name in ("vggt.py", "dpt_head.py", "head_act.py"):
            print(f"\n===== {path.relative_to(root)} ({len(hits)} matching lines) =====")
            for i, l in hits[:60]:
                print(f"{i:5d}: {l}")
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
    # All 18 scenes in one process. The forced-collision gate builds extra
    # transport plans per rotation pair, so this is the slow mode; on the
    # cluster prefer the array, which runs the same code one scene per task:
    #   sbatch --array 0-17 scripts/phase4.sbatch configs/phase4.yaml gates
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
    # The permanent real-weight encoder tests, PROTOCOL 3.1. These load VGGT-1B
    # and fall back to CPU when CUDA is absent, so on a login node they would
    # quietly run a billion-parameter trunk on the CPU. Refuse instead: run
    # this inside an allocation, for example
    #   srun --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
    #        --gres=gpu:1 --time=00:30:00 --pty ./scripts/run_phase4.sh smoke
    if ! run_lot python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        echo "no CUDA device visible. These tests load real VGGT weights, so" >&2
        echo "run them inside a GPU allocation rather than on a login node:" >&2
        echo "  srun --gres=gpu:1 --time=00:30:00 --pty ./scripts/run_phase4.sh smoke" >&2
        exit 1
    fi
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
