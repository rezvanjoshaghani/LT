"""Interpreted quantities under both evaluation paths, paired cell by cell.

PROTOCOL 3.9 gates the operator, not the claims. Its tolerance is a fixed
0.003, calibrated on the pilot's aggregate path agreement, and the corrected
run produces interpreted effects an order of magnitude smaller than that for
one encoder. Rather than re-derive the gate from the effects it is meant to be
independent of, this module reports every interpreted quantity under both
paths and restricts what may be claimed to what the two agree on.

The quantities, per encoder and reporting cell:

    oracle_margin      Oracle-Transport minus No-Warp-Copy
    localization_gap   Oracle-Transport minus Neighbor-Patch

Every term of every quantity is read from the cross-path common-valid cell
set, matched by sample_id, before any differencing. The evaluation layer
already scores each variant on that set and stores it as its own column, so a
difference taken here is a difference of two operators over one population,
never over two populations wearing the shape of an operator effect.

Order matters and is fixed. The paired support is established first, from the
pairs both paths scored. Scene-level contributions are computed on that fixed
support. Only then does the bootstrap run, resampling scenes once per
replicate and recomputing both paths and their difference from the same drawn
scenes. Support is never re-derived inside a replicate and the paths are never
resampled independently: a difference of two independently bootstrapped
intervals would carry variance the paired difference does not have.

Near-zero classification and the wording it licenses are in classify(). The
band is PROTOCOL 3.9's tolerance itself, so an effect no larger than the
operator gate is never described as small on the strength of one path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .analysis_config import DEFAULT_CONFIG_PATH, AnalysisConfig, load_analysis_config
from .evaluate import NEIGHBOR_PATCH, NO_WARP_COPY, ORACLE_TRANSPORT, PER_POINT, SPLAT_POOL
from .figures import (
    INTERSECT_METRICS,
    JOINT_REGIME,
    PRIMARY_REGIME,
    assign_bins,
    is_supported,
    read_eval_dir,
)

QUANTITIES = {
    "oracle_margin": (ORACLE_TRANSPORT, NO_WARP_COPY),
    "localization_gap": (ORACLE_TRANSPORT, NEIGHBOR_PATCH),
}

TABLE_NAME = "path_margin_differences.parquet"
SUMMARY_NAME = "path_margin_differences.txt"


def paired_pair_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per (pair, encoder, metric) carrying both paths' terms.

    A pair enters only when both paths scored it, because a cross-path
    difference is undefined otherwise. The scores are the intersection columns,
    which the evaluation layer computed on the cells both paths scored, so the
    three terms of a quantity already share one population before they are
    differenced here.
    """
    by_key: dict[tuple, dict[tuple, float]] = {}
    context: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = (row["scene"], row["context_frame_id"], row["target_frame_id"], row["encoder"])
        context.setdefault(key, row)
        for metric, column in INTERSECT_METRICS.items():
            by_key.setdefault(key, {})[(metric, row["path"], row["variant"])] = row[column]

    out: list[dict[str, Any]] = []
    for key, terms in by_key.items():
        row = context[key]
        for metric in INTERSECT_METRICS:
            values: dict[str, float] = {}
            complete = True
            for name, (high, low) in QUANTITIES.items():
                for path, suffix in ((PER_POINT, "pp"), (SPLAT_POOL, "sp")):
                    a = terms.get((metric, path, high))
                    b = terms.get((metric, path, low))
                    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
                        complete = False
                        break
                    values[f"{name}_{suffix}"] = a - b
                if not complete:
                    break
            if not complete:
                continue
            out.append(
                {
                    "scene": row["scene"],
                    "camera_pair": (row["context_frame_id"], row["target_frame_id"]),
                    "encoder": row["encoder"],
                    "metric": metric,
                    "regime": row["regime"],
                    "parallax_bin": row["parallax_bin"],
                    "rotation_bin": row["rotation_bin"],
                    **values,
                }
            )
    return out


def cell_key(record: dict[str, Any]) -> tuple[str, str] | None:
    """The reporting cell a record belongs to: (analysis, bin label).

    PROTOCOL 3.3 keeps each primary curve to the regime that holds the other
    axis at zero, and puts orbit in the joint view alone.
    """
    regime = record["regime"]
    for axis, primary in PRIMARY_REGIME.items():
        if regime == primary:
            return regime, record[axis]
    if regime == JOINT_REGIME:
        return regime, f"{record['rotation_bin']} x {record['parallax_bin']}"
    return None


def paired_bootstrap(
    by_scene: dict[str, list[dict[str, float]]],
    quantity: str,
    config: AnalysisConfig,
) -> dict[str, float]:
    """Point estimates and paired intervals for M_pp, M_sp and their difference.

    One draw of scenes per replicate serves both paths, and dM is recomputed
    inside the replicate rather than differenced afterwards, so the interval
    around it carries the covariance the two paths actually have. Bootstrapping
    the paths independently and subtracting their intervals would inflate it by
    exactly the correlation the pairing exists to keep.
    """
    scenes = sorted(by_scene)
    pp = {s: np.array([r[f"{quantity}_pp"] for r in by_scene[s]]) for s in scenes}
    sp = {s: np.array([r[f"{quantity}_sp"] for r in by_scene[s]]) for s in scenes}
    all_pp = np.concatenate([pp[s] for s in scenes])
    all_sp = np.concatenate([sp[s] for s in scenes])
    point = {
        "M_pp": float(all_pp.mean()),
        "M_sp": float(all_sp.mean()),
        "dM": float(all_pp.mean() - all_sp.mean()),
    }
    rng = np.random.default_rng(config.bootstrap_seed)
    draws: dict[str, list[float]] = {"M_pp": [], "M_sp": [], "dM": []}
    for _ in range(config.bootstrap_resamples):
        chosen = rng.integers(0, len(scenes), size=len(scenes))
        rep_pp = np.concatenate([pp[scenes[c]] for c in chosen])
        rep_sp = np.concatenate([sp[scenes[c]] for c in chosen])
        if rep_pp.size == 0:
            continue
        a, b = float(rep_pp.mean()), float(rep_sp.mean())
        draws["M_pp"].append(a)
        draws["M_sp"].append(b)
        draws["dM"].append(a - b)
    tail = (1.0 - config.bootstrap_confidence) / 2.0
    out = dict(point)
    for name, values in draws.items():
        out[f"{name}_ci_low"] = float(np.quantile(values, tail)) if values else float("nan")
        out[f"{name}_ci_high"] = (
            float(np.quantile(values, 1.0 - tail)) if values else float("nan")
        )
    out["replicates"] = len(draws["dM"])
    return out


def classify(quantity: str, stats: dict[str, float], band: float) -> dict[str, Any]:
    """The three-way reading, and the wording each case licenses.

    The band is PROTOCOL 3.9's own tolerance. An effect no larger than the
    operator gate is never called small on one path's authority, and the phrase
    "statistically significant but negligible" is never produced: a magnitude
    at the scale of evaluation-path choice is reported as such, not as a
    significant finding with a diminutive attached.
    """
    pp, sp = stats["M_pp"], stats["M_sp"]
    excludes_zero = (
        stats["M_pp_ci_low"] * stats["M_pp_ci_high"] > 0
        and stats["M_sp_ci_low"] * stats["M_sp_ci_high"] > 0
    )
    same_sign = (pp > 0) == (sp > 0)
    in_band = [abs(pp) <= band, abs(sp) <= band]
    direction = "positive" if pp > 0 else "negative"

    if same_sign and excludes_zero and all(in_band):
        case = "both_in_band"
        if quantity == "oracle_margin":
            sentence = (
                f"a small, sign-consistent {direction} transport margin under both "
                "evaluation paths, quantitatively close to the No-Warp floor"
            )
        else:
            sentence = (
                "Oracle transport adds only a small, sign-consistent improvement "
                "over Neighbor-Patch"
            )
    elif same_sign and excludes_zero and sum(in_band) == 1:
        case = "one_in_band"
        if quantity == "oracle_margin":
            sentence = (
                f"a {direction} transport margin under both evaluation paths, but "
                "its magnitude is path-sensitive"
            )
        else:
            sentence = (
                "Oracle improves over Neighbor-Patch under both paths, but the "
                "magnitude is path-sensitive"
            )
    elif same_sign and excludes_zero:
        case = "neither_in_band"
        sentence = ""
    else:
        case = "not_robust"
        sentence = (
            "no robust transport advantage; the measured effect is at the scale "
            "of evaluation-path choice"
        )
    # A ratio is descriptive and only meaningful above the band; below it the
    # denominator is itself at the scale of the operator difference.
    ratio = (
        abs(stats["dM"]) / abs(pp)
        if abs(pp) > band and abs(pp) > 0
        else float("nan")
    )
    return {
        "near_zero": bool(any(in_band)),
        "both_in_band": bool(all(in_band)),
        "same_sign": bool(same_sign),
        "both_ci_exclude_zero": bool(excludes_zero),
        "case": case,
        "sentence": sentence,
        "abs_dM_over_abs_M_pp": ratio,
    }


def build(eval_dir: Path, config: AnalysisConfig) -> list[dict[str, Any]]:
    """The whole table: support, then contributions, then the paired bootstrap."""
    rows = assign_bins(read_eval_dir(eval_dir, config), config)
    records = paired_pair_records(rows)

    # Support first, on the paired set, and fixed from here on.
    cells: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        key = cell_key(record)
        if key is None:
            continue
        cells.setdefault((record["encoder"], record["metric"], *key), []).append(record)

    out: list[dict[str, Any]] = []
    for key, members in sorted(cells.items(), key=repr):
        encoder, metric, analysis, label = key
        counts = {
            "n_scenes": len({r["scene"] for r in members}),
            "n_camera_pairs": len({r["camera_pair"] for r in members}),
            "n_feature_comparisons": 0,
        }
        supported = is_supported(counts, config)
        by_scene: dict[str, list[dict[str, float]]] = {}
        for record in members:
            by_scene.setdefault(record["scene"], []).append(record)
        for quantity in QUANTITIES:
            stats = paired_bootstrap(by_scene, quantity, config)
            verdict = classify(quantity, stats, config.path_agreement_tolerance)
            out.append(
                {
                    "encoder": encoder,
                    "metric": metric,
                    "analysis": analysis,
                    "bin": label,
                    "quantity": quantity,
                    "n_scenes": counts["n_scenes"],
                    "n_camera_pairs": counts["n_camera_pairs"],
                    "supported": supported,
                    "band": config.path_agreement_tolerance,
                    **stats,
                    **verdict,
                }
            )
    return out


def format_table(table: Sequence[dict[str, Any]], config: AnalysisConfig) -> str:
    """A readable summary: supported cells, then the near-zero ones in full."""
    lines = [
        "Interpreted quantities under both evaluation paths, paired by sample_id.",
        "Every term is read from the cross-path common-valid cell set before",
        "differencing. Support is established on the paired set and fixed; the",
        "bootstrap draws scenes once per replicate and serves both paths from the",
        f"same draw. Near-zero band is PROTOCOL 3.9's tolerance, "
        f"{config.path_agreement_tolerance}.",
        "",
        f"{'encoder':<16}{'metric':<22}{'analysis':<12}{'bin':<20}{'quantity':<18}"
        f"{'M_pp':>10}{'M_sp':>10}{'dM':>10}  case",
    ]
    supported = [r for r in table if r["supported"]]
    for row in supported:
        lines.append(
            f"{row['encoder']:<16}{row['metric']:<22}{row['analysis']:<12}"
            f"{row['bin']:<20}{row['quantity']:<18}"
            f"{row['M_pp']:>+10.4f}{row['M_sp']:>+10.4f}{row['dM']:>+10.4f}  {row['case']}"
        )
    flagged = [r for r in supported if r["near_zero"]]
    lines += ["", f"NEAR-ZERO CELLS ({len(flagged)} of {len(supported)} supported)", ""]
    for row in flagged:
        lines += [
            f"  {row['encoder']} / {row['metric']} / {row['analysis']} / {row['bin']}"
            f" / {row['quantity']}",
            f"    per_point  {row['M_pp']:+.5f}  "
            f"[{row['M_pp_ci_low']:+.5f}, {row['M_pp_ci_high']:+.5f}]",
            f"    splat_pool {row['M_sp']:+.5f}  "
            f"[{row['M_sp_ci_low']:+.5f}, {row['M_sp_ci_high']:+.5f}]",
            f"    dM         {row['dM']:+.5f}  "
            f"[{row['dM_ci_low']:+.5f}, {row['dM_ci_high']:+.5f}]",
            f"    case: {row['case']}",
            f"    claim: {row['sentence'] or '(no restricted wording; see case)'}",
            "",
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--out", type=Path, default=Path("validation/evidence")
    )
    args = parser.parse_args(argv)
    config = load_analysis_config(args.analysis_config)
    table = build(args.eval_dir, config)

    import pyarrow as pa
    import pyarrow.parquet as pq

    args.out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(table), args.out / TABLE_NAME)
    summary = format_table(table, config)
    (args.out / SUMMARY_NAME).write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\ntable   -> {args.out / TABLE_NAME}")
    print(f"summary -> {args.out / SUMMARY_NAME}")


if __name__ == "__main__":
    main()
