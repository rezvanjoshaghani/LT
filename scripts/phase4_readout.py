"""Print the Phase 4 rung 1 tables as text, for reading and for pasting.

Reads only outputs/{experiment}/tables/*.parquet and the near-zero json,
which PROTOCOL requires every figure to be regenerable from. Computes
nothing: every number here was written by phase4_report.

    PYTHONPATH=src python scripts/phase4_readout.py \
        --tables outputs/phase4_rung1/tables
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LEVEL_ORDER = ("gt", "none", "scene", "image", "affine")
CENTERED = "cosine_centered_mean"
RAW = "cosine_mean"


def read(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def fmt(value, width=8, places=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return " " * (width - 1) + "."
    if isinstance(value, float):
        return f"{value:{width}.{places}f}"
    return f"{value:>{width}}"


def interval(row, name):
    low, high = row.get(f"{name}_ci_low"), row.get(f"{name}_ci_high")
    if low is None or (isinstance(low, float) and math.isnan(low)):
        return " " * 17
    return f"[{low:+7.4f},{high:+7.4f}]"


def level_key(row):
    order = LEVEL_ORDER.index(row["level"]) if row["level"] in LEVEL_ORDER else 9
    return (order, row["level"])


def ladder_block(ladder, scope, metric, path):
    rows = sorted(
        (r for r in ladder
         if r["analysis"] == scope and r["metric"] == metric and r["path"] == path),
        key=level_key,
    )
    if not rows:
        return
    print(f"\n  {scope} / {path} / {'centered' if metric == CENTERED else 'raw'}")
    print(f"    {'level':7s} {'pairs':>6s} {'ceiling':>8s} {'floor':>8s} "
          f"{'est':>8s} {'tax':>8s} {'tax 95% CI':>17s} {'retain':>8s} "
          f"{'trans.f':>8s} {'sel.diff':>9s} sup")
    for row in rows:
        print(
            f"    {row['level']:7s} {fmt(row.get('n_camera_pairs'), 6)} "
            f"{fmt(row.get('matched_ceiling'))} {fmt(row.get('matched_floor'))} "
            f"{fmt(row.get('estimated_score'))} {fmt(row.get('depth_tax'))} "
            f"{interval(row, 'depth_tax')} {fmt(row.get('retained_fraction'))} "
            f"{fmt(row.get('transported_fraction'))} "
            f"{fmt(row.get('selection_differential'), 9)} "
            f"{'y' if row.get('supported') else 'n'}"
        )
        if row["level"] == "affine" and row.get("affine_pairs_attempted") is not None:
            print(f"      affine fits: {row.get('affine_pairs_contributed')} of "
                  f"{row.get('affine_pairs_attempted')} contributed, "
                  f"{row.get('affine_pairs_failed')} failed")


def bins_block(bins, regime, axis, metric, path, level_filter=None):
    rows = [
        b for b in bins
        if b["analysis"] == regime and b["axis"] == axis
        and b["metric"] == metric and b["path"] == path
        and (level_filter is None or b["level"] in level_filter)
    ]
    if not rows:
        return
    labels = sorted({b["bin"] for b in rows})
    print(f"\n  {regime} by {axis} / {path} / "
          f"{'centered' if metric == CENTERED else 'raw'}: depth_tax")
    print(f"    {'level':7s} " + " ".join(f"{l[:14]:>16s}" for l in labels))
    for level in LEVEL_ORDER:
        present = {b["bin"]: b for b in rows if b["level"] == level}
        if not present:
            continue
        cells = []
        for label in labels:
            row = present.get(label)
            if row is None:
                cells.append(f"{'':>16s}")
                continue
            mark = "*" if not row.get("supported") else " "
            cell = f"{fmt(row.get('depth_tax'), 7)}{mark} n={row.get('n_camera_pairs')}"
            cells.append(f"{cell:>16s}")
        print(f"    {level:7s} " + " ".join(cells))
    print("    * unsupported cell (below the frozen 3.4 support thresholds)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--metric", choices=("centered", "raw"), default="centered")
    args = parser.parse_args()
    metric = CENTERED if args.metric == "centered" else RAW

    ladder = read(args.tables / "phase4_ladder.parquet")
    bins = read(args.tables / "phase4_bins.parquet")
    near_zero = json.loads((args.tables / "phase4_near_zero.json").read_text())

    print("=" * 78)
    print("PHASE 4 RUNG 1: THE ESTIMATED-GEOMETRY TAX")
    print("=" * 78)
    print(f"ladder rows {len(ladder)}, bin rows {len(bins)}")

    print("\n--- ALIGNMENT LADDER " + "-" * 56)
    for scope in ("pooled", "rotation", "translation", "orbit"):
        for path in ("per_point", "splat_pool"):
            ladder_block(ladder, scope, metric, path)

    print("\n--- PARALLAX AND ROTATION CURVES " + "-" * 44)
    bins_block(bins, "translation", "parallax_bin", metric, "per_point")
    bins_block(bins, "rotation", "rotation_bin", metric, "per_point")

    print("\n--- 4.5 CONTROL UNDER AMENDMENT A7 " + "-" * 42)
    rot = [r for r in ladder if r["analysis"] == "rotation"
           and r["path"] == "splat_pool" and r["metric"] == metric]
    for row in sorted(rot, key=level_key):
        print(f"    {row['level']:7s} forced identity gap raw "
              f"{fmt(row.get('forced_identity_gap_raw'), 11, 3):>11s}  centered "
              f"{fmt(row.get('forced_identity_gap_centered'), 11, 3):>11s}")

    print("\n--- LOCALIZATION CONTRASTS " + "-" * 51)
    for scope in ("translation", "orbit"):
        rows = [r for r in ladder if r["analysis"] == scope
                and r["path"] == "per_point" and r["metric"] == metric]
        for row in sorted(rows, key=level_key):
            if row["level"] == "gt":
                continue
            print(f"    {scope:12s} {row['level']:7s} "
                  f"boundary-interior {fmt(row.get('boundary_minus_interior_tax'))} "
                  f"{interval(row, 'boundary_minus_interior_tax')}  "
                  f"lowtex-hightex {fmt(row.get('lowtex_minus_hightex_tax'))} "
                  f"{interval(row, 'lowtex_minus_hightex_tax')}")

    print("\n--- NEAR-ZERO DISCLOSURE " + "-" * 53)
    cases: dict[str, int] = {}
    for entry in near_zero:
        cases[entry["case"]] = cases.get(entry["case"], 0) + 1
    for case, count in sorted(cases.items(), key=lambda kv: -kv[1]):
        print(f"    {case:28s} {count:>5d}")


if __name__ == "__main__":
    main()
