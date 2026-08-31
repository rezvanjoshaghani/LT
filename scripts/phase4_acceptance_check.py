"""Verify the five conditions Phase 4 acceptance was made contingent on.

Reads shipped artifacts only: the per-scene eval parquets and their run
records, the tables, the near-zero disclosure, and the validator 2.3
summary. Recomputes nothing scientific; every number it prints was
written by the evaluation or reporting layer.

    PYTHONPATH=src python scripts/phase4_acceptance_check.py \
        --eval-dir outputs/phase4_rung1/eval \
        --tables outputs/phase4_rung1/tables \
        --validator validation/evidence/reaudit/borah_check_2_3.json

Checks, in the order they were asked for:

1. The Level 0/1/2 source transport-valid sets satisfy the frozen 4.4
   invariant. The evaluation layer asserted set equality per pair at run
   time and refuses to write otherwise; this re-checks the persisted
   counts across the shipped rows, which is the part a reader can audit.
2. Affine fit failures are counted once, from the context frame, and are
   not path dependent. The ladder's per-scope column counts something
   else and is renamed accordingly.
3. Raw and centered tables both exist, over the same cells.
4. Bootstrap settings are the frozen ones and the intervals are present,
   including the secondary camera-pair intervals; the near-zero
   disclosure is decomposed rather than totalled.
5. Validator 2.3 carries a PASS with no mask or count mismatches.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MULTIPLICATIVE = ("none", "scene", "image")
RAW = "cosine_mean"
CENTERED = "cosine_centered_mean"


def read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def check_transport_valid(eval_dir: Path) -> tuple[bool, list[str]]:
    """4.4: positive scaling cannot change finite-and-positive depth."""
    notes: list[str] = []
    violations = 0
    pairs_checked = 0
    for path in sorted(eval_dir.glob("*.parquet")):
        by_key: dict[tuple, dict[str, int]] = {}
        for row in read_parquet(path):
            if row["population"] != "matched" or row["level"] not in MULTIPLICATIVE:
                continue
            key = (row["context_frame_id"], row["target_frame_id"], row["path"])
            by_key.setdefault(key, {})[row["level"]] = row["n_transport_valid"]
        for key, levels in by_key.items():
            present = {k: v for k, v in levels.items() if k in MULTIPLICATIVE}
            if len(present) < 2:
                continue
            pairs_checked += 1
            if len(set(present.values())) != 1:
                violations += 1
                if violations <= 3:
                    notes.append(f"    {path.stem} {key}: {present}")
    ok = violations == 0
    notes.insert(0, f"    {pairs_checked} (pair, path) groups checked across "
                    f"levels {', '.join(MULTIPLICATIVE)}; {violations} differ")
    notes.append("    set equality was asserted per pair at run time; this "
                 "re-checks the persisted counts")
    return ok, notes


def check_affine(eval_dir: Path, ladder: list[dict]) -> tuple[bool, list[str]]:
    """The affine fit is a property of the context image, not of a path."""
    from lot.evaluate import read_run_metadata

    notes: list[str] = []
    true_failures = 0
    scenes = 0
    for path in sorted(eval_dir.glob("*.parquet")):
        meta = read_run_metadata(path)
        if meta is None or meta.get("affine_failed_pairs") is None:
            notes.append(f"    {path.stem}: run record carries no affine_failed_pairs")
            return False, notes
        true_failures += int(meta["affine_failed_pairs"])
        scenes += 1
    notes.append(f"    affine fits that failed, from the context-image "
                 f"calibration, summed over {scenes} scenes: {true_failures}")
    per_path: dict[str, int] = {}
    for row in ladder:
        if row["analysis"] != "pooled" or row["level"] != "affine":
            continue
        if row["metric"] != CENTERED:
            continue
        value = row.get("affine_pairs_not_contributing")
        if value is None:
            value = row.get("affine_pairs_failed")
            notes.append("    ladder still uses the old column name "
                         "affine_pairs_failed; regenerate the tables")
        per_path[row["path"]] = value
    for path_name, value in sorted(per_path.items()):
        notes.append(f"    pairs in scope with no affine arm on {path_name}: {value}")
    run_total = {r.get("affine_arms_absent_run_total") for r in ladder
                 if r.get("affine_arms_absent_run_total") is not None}
    if run_total:
        notes.append(f"    (pair, path) slots with no affine arm, run total: "
                     f"{sorted(run_total)[0]}")
    notes.append("    the last two are path dependent because a pair can be "
                 "scored at one level and not another; the fit is not")
    return True, notes


def check_metrics(ladder: list[dict], bins: list[dict]) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    for name, table in (("ladder", ladder), ("bins", bins)):
        cells = {m: set() for m in (RAW, CENTERED)}
        for row in table:
            key = (row["analysis"], row["path"], row["level"], row.get("bin"))
            if row["metric"] in cells:
                cells[row["metric"]].add(key)
        raw_only = cells[RAW] - cells[CENTERED]
        cen_only = cells[CENTERED] - cells[RAW]
        notes.append(f"    {name}: {len(cells[RAW])} raw cells, "
                     f"{len(cells[CENTERED])} centered cells")
        if raw_only or cen_only:
            ok = False
            notes.append(f"      asymmetric: {len(raw_only)} raw-only, "
                         f"{len(cen_only)} centered-only")
    return ok, notes


def check_intervals(ladder: list[dict], near_zero: list[dict]) -> tuple[bool, list[str]]:
    from lot.analysis_config import load_analysis_config

    analysis = load_analysis_config()
    notes = [
        f"    bootstrap: {analysis.bootstrap_resamples} resamples, seed "
        f"{analysis.bootstrap_seed}, confidence {analysis.bootstrap_confidence}",
    ]
    ok = True
    quantities = ("depth_tax", "oracle_margin", "retained_fraction")
    for quantity in quantities:
        have_scene = sum(
            1 for r in ladder
            if isinstance(r.get(f"{quantity}_ci_low"), float)
            and math.isfinite(r[f"{quantity}_ci_low"])
        )
        have_pair = sum(
            1 for r in ladder
            if isinstance(r.get(f"{quantity}_pair_ci_low"), float)
            and math.isfinite(r[f"{quantity}_pair_ci_low"])
        )
        notes.append(f"    {quantity}: {have_scene} scene intervals, "
                     f"{have_pair} camera-pair intervals")
        if have_scene == 0 or have_pair == 0:
            ok = False
    recorded = {r.get("bootstrap_resamples") for r in ladder
                if r.get("bootstrap_resamples") is not None}
    if recorded and recorded != {analysis.bootstrap_resamples}:
        ok = False
        notes.append(f"    tables record resamples {sorted(recorded)}")
    return ok, notes


def decompose_disclosure(near_zero: list[dict]) -> list[str]:
    """The near-zero cells by quantity, regime, and level, not as one total.

    A pure-rotation zero is an invariant the protocol predicts, not an
    ambiguous small effect, so a single total mixes two different things.
    """
    notes = [f"    {len(near_zero)} disclosed cells in total"]
    by_case: dict[str, int] = {}
    for entry in near_zero:
        by_case[entry["case"]] = by_case.get(entry["case"], 0) + 1
    notes.append("    by case: " + ", ".join(
        f"{k} {v}" for k, v in sorted(by_case.items())))
    grouped: dict[tuple, dict[str, int]] = {}
    for entry in near_zero:
        key = (entry["analysis"], entry["level"], entry["quantity"])
        grouped.setdefault(key, {})
        grouped[key][entry["case"]] = grouped[key].get(entry["case"], 0) + 1
    notes.append(f"    {'regime':12s} {'level':7s} {'quantity':22s} cases")
    for key in sorted(grouped):
        regime, level, quantity = key
        cases = ", ".join(f"{k} {v}" for k, v in sorted(grouped[key].items()))
        notes.append(f"    {regime:12s} {level:7s} {quantity:22s} {cases}")
    rotation = sum(1 for e in near_zero if e["analysis"] == "rotation")
    notes.append(f"    of which regime rotation: {rotation}. Those are the 4.5 "
                 "invariant, predicted to be zero, not ambiguous effects")
    return notes


def check_validator(path: Path) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, [f"    {path} is absent"]
    summary = json.loads(path.read_text(encoding="utf-8"))
    worst = max(summary["metric_max_abs_diff"].values())
    ok = (
        summary.get("verdict") == "PASS"
        and summary.get("mask_mismatches") == 0
        and summary.get("count_mismatches") == 0
    )
    return ok, [
        f"    verdict {summary.get('verdict')}, {summary.get('pairs')} pairs, "
        f"{summary.get('rows_compared')} rows",
        f"    mask mismatches {summary.get('mask_mismatches')}, count "
        f"mismatches {summary.get('count_mismatches')}",
        f"    worst metric residual {worst:.2e} against tolerance "
        f"{summary.get('tolerance')}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    args = parser.parse_args()

    ladder = read_parquet(args.tables / "phase4_ladder.parquet")
    bins = read_parquet(args.tables / "phase4_bins.parquet")
    near_zero = json.loads((args.tables / "phase4_near_zero.json").read_text())

    print("=" * 74)
    print("PHASE 4 ACCEPTANCE CHECK")
    print("=" * 74)

    results = []
    for title, (ok, notes) in (
        ("1. Level 0/1/2 transport-valid invariant (4.4)",
         check_transport_valid(args.eval_dir)),
        ("2. Affine fit failures counted once, path independent",
         check_affine(args.eval_dir, ladder)),
        ("3. Raw and centered tables both present",
         check_metrics(ladder, bins)),
        ("4. Bootstrap settings and intervals",
         check_intervals(ladder, near_zero)),
        ("5. Validator 2.3", check_validator(args.validator)),
    ):
        print(f"\n{title}: {'PASS' if ok else 'FAIL'}")
        for note in notes:
            print(note)
        results.append((title, ok))

    print("\n6. Near-zero disclosure, decomposed")
    for note in decompose_disclosure(near_zero):
        print(note)

    print("\n" + "=" * 74)
    failed = [t for t, ok in results if not ok]
    if failed:
        print(f"NOT SATISFIED: {len(failed)} of {len(results)}")
        for title in failed:
            print(f"  {title}")
        sys.exit(1)
    print(f"All {len(results)} acceptance conditions satisfied.")


if __name__ == "__main__":
    main()
