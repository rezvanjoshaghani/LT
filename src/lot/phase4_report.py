"""Phase 4 tables and figures, from the Phase 4 parquet and the config alone.

PROTOCOL 4.6 and 4.10. Every quantity is subset matched by construction: the
evaluation layer scored the estimated variant, the Oracle ceiling, and the
No-Warp floor on one persisted mask per (pair, path, level), so a difference
of the stored means is a paired difference on that mask. This layer only
groups, averages over camera pairs (the frozen estimand), bootstraps scenes,
and draws.

Quantities per (metric, path, level, cell), all unweighted means over pairs:

    matched_ceiling        Oracle-Transport on the surviving set
    estimated_score        the VGGT variant on the same set
    depth_tax              matched_ceiling minus estimated_score
    oracle_margin          matched_ceiling minus the matched No-Warp floor
    estimated_margin       estimated_score minus the matched No-Warp floor
    retained_fraction      estimated_margin over oracle_margin, suppressed
                           where the denominator is below epsilon_margin
    selection_differential full-population Oracle ceiling minus matched_ceiling
    transported_fraction   surviving fraction of the ground-truth co-visible set

Uncertainty is a paired scene bootstrap: one draw of scenes serves every term
of a quantity and the statistic is recomputed inside each replicate, so
ratios and differences carry the covariance the pairing gives them
(execution plan Addendum B). Support and greying follow the frozen 3.4 rules.
The near-zero disclosure of Addendum A applies to score-space quantities
whose magnitude is at most the frozen operator tolerance on either path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .analysis_config import DEFAULT_CONFIG_PATH, AnalysisConfig, load_analysis_config
from .datasets import bin_order, parallax_bin, rotation_bin
from .evaluate import MEAN_FEATURE, NO_WARP_COPY, ORACLE_TRANSPORT, PER_POINT, SPLAT_POOL
from .figures import CONTENT_DIGEST, is_pinned_revision, write_table
from .phase4 import GT_LEVEL, LEVELS, PHASE4_VERSION, POPULATION_FULL, POPULATION_MATCHED

RAW = "cosine_mean"
CENTERED = "cosine_centered_mean"
METRICS = (RAW, CENTERED)
SPLITS = ("boundary", "interior", "lowtex", "hightex")

PRIMARY_REGIME = {"parallax_bin": "translation", "rotation_bin": "rotation"}
JOINT_REGIME = "orbit"


# ---------------------------------------------------------------------------
# Reading and record building
# ---------------------------------------------------------------------------

def read_phase4_dir(eval_dir: Path, analysis: AnalysisConfig) -> list[dict[str, Any]]:
    """Read every per-scene Phase 4 parquet, checking they are one run."""
    import pyarrow.parquet as pq

    from .evaluate import read_run_metadata

    eval_dir = Path(eval_dir)
    files = sorted(eval_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {eval_dir}")
    rows: list[dict[str, Any]] = []
    metas: dict[str, dict[str, Any]] = {}
    for path in files:
        meta = read_run_metadata(path)
        if meta is None:
            raise ValueError(f"{path} carries no run record")
        metas[path.stem] = meta
        rows.extend(pq.read_table(path).to_pylist())
    for field in ("phase4_version", "seed", "phase4_measurement_digest",
                  "depth_weights_fingerprint", "depth_weights_revision",
                  "depth_code_revision", "run_scenes", "git_commit"):
        values = {json.dumps(m.get(field), sort_keys=True) for m in metas.values()}
        if len(values) > 1:
            raise ValueError(f"{eval_dir} mixes runs: {field} takes {sorted(values)[:3]}")
    any_meta = next(iter(metas.values()))
    if any_meta.get("phase4_version") != PHASE4_VERSION:
        raise ValueError(
            f"{eval_dir} was written by phase4_version {any_meta.get('phase4_version')}"
        )
    expected = set(any_meta.get("run_scenes") or ())
    if set(metas) != expected:
        raise ValueError(
            f"{eval_dir} holds {len(metas)} of {len(expected)} scenes; an "
            "incomplete directory is a different population"
        )
    fingerprint = any_meta.get("depth_weights_fingerprint")
    if not (isinstance(fingerprint, str) and CONTENT_DIGEST.fullmatch(fingerprint)):
        raise ValueError("depth weights fingerprint is missing or malformed")
    for field in ("depth_weights_revision", "depth_code_revision"):
        if not is_pinned_revision(field.replace("depth_", ""), any_meta.get(field)):
            raise ValueError(f"{field} {any_meta.get(field)!r} is not a pin")
    if str(any_meta.get("git_commit", "")).endswith("-dirty"):
        raise ValueError(
            "the Phase 4 run records carry a -dirty commit; the hardened "
            "launch templates should have made this impossible. Re-run from "
            "a clean tree rather than reporting on it"
        )
    return rows


def build_records(
    rows: Sequence[dict[str, Any]], analysis: AnalysisConfig
) -> list[dict[str, Any]]:
    """One record per (pair, path, metric, level) carrying every paired term."""
    by_key: dict[tuple, dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["scene"], row["context_frame_id"], row["target_frame_id"], row["path"])
        by_key.setdefault(key, {})[f"{row['level']}|{row['population']}|{row['variant']}"] = row

    records: list[dict[str, Any]] = []
    for key, slots in by_key.items():
        scene, context_frame_id, target_frame_id, path = key
        gt_oracle = slots.get(f"{GT_LEVEL}|{POPULATION_FULL}|{ORACLE_TRANSPORT}")
        if gt_oracle is None:
            continue
        context = gt_oracle
        pbin = parallax_bin(context["parallax"], analysis)
        rbin = rotation_bin(context["rotation_deg"], analysis)
        for level, variant_name in LEVELS:
            est = slots.get(f"{level}|{POPULATION_MATCHED}|{variant_name}")
            oracle_m = slots.get(f"{level}|{POPULATION_MATCHED}|{ORACLE_TRANSPORT}")
            nowarp_m = slots.get(f"{level}|{POPULATION_MATCHED}|{NO_WARP_COPY}")
            if est is None or oracle_m is None or nowarp_m is None:
                continue
            for metric in METRICS:
                if not all(
                    isinstance(r[metric], float) and math.isfinite(r[metric])
                    for r in (est, oracle_m, nowarp_m, gt_oracle)
                ):
                    continue
                record = {
                    "scene": scene,
                    "camera_pair": (context_frame_id, target_frame_id),
                    "regime": context["regime"],
                    "parallax": context["parallax"],
                    "rotation_deg": context["rotation_deg"],
                    "parallax_bin": pbin,
                    "rotation_bin": rbin,
                    "path": path,
                    "metric": metric,
                    "level": level,
                    "est": est[metric],
                    "oracle_m": oracle_m[metric],
                    "nowarp_m": nowarp_m[metric],
                    "oracle_full": gt_oracle[metric],
                    "n": est["n"],
                    "n_gt": est["n_gt"],
                    "transported_fraction": est["n"] / est["n_gt"] if est["n_gt"] else float("nan"),
                    "collision_tax_raw": est.get("collision_tax_raw", float("nan")),
                    "collision_tax_centered": est.get("collision_tax_centered", float("nan")),
                }
                for split in SPLITS:
                    est_s = slots.get(f"{level}|{split}|{variant_name}")
                    oracle_s = slots.get(f"{level}|{split}|{ORACLE_TRANSPORT}")
                    record[f"tax_{split}"] = (
                        oracle_s[metric] - est_s[metric]
                        if est_s is not None and oracle_s is not None
                        and math.isfinite(est_s[metric]) and math.isfinite(oracle_s[metric])
                        else float("nan")
                    )
                records.append(record)
    return records


# ---------------------------------------------------------------------------
# Statistics: unweighted pair means, paired scene bootstrap
# ---------------------------------------------------------------------------

def group_by(records: Sequence[dict], keys: Sequence[str]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        grouped.setdefault(tuple(record[k] for k in keys), []).append(record)
    return grouped


# The per-pair fields every reported quantity is built from. Each quantity is
# a closed-form function of these fields' unweighted means over camera pairs,
# which is what lets a replicate be summarized rather than re-pooled.
FIELDS = (
    "oracle_m", "est", "nowarp_m", "oracle_full", "transported_fraction",
    "tax_boundary", "tax_interior", "tax_lowtex", "tax_hightex",
)


def scene_aggregates(records: Sequence[dict]) -> dict[str, dict[str, tuple[float, int]]]:
    """Per scene, the sum and count of finite values of every field.

    A resampled mean is sum-of-sums over sum-of-counts, so a bootstrap
    replicate costs one pass over 18 scenes instead of one pass over every
    record in the cell. The value is identical to pooling the records: the
    statistic is still recomputed whole inside each replicate, from that
    replicate's own means, which is what Addendum B requires of a ratio.
    """
    out: dict[str, dict[str, list]] = {}
    for record in records:
        slot = out.setdefault(record["scene"], {field: [0.0, 0] for field in FIELDS})
        for field in FIELDS:
            value = record[field]
            if isinstance(value, (int, float)) and math.isfinite(value):
                slot[field][0] += float(value)
                slot[field][1] += 1
    return {
        scene: {field: (total, count) for field, (total, count) in fields.items()}
        for scene, fields in out.items()
    }


def pooled_means(
    aggregates: dict[str, dict[str, tuple[float, int]]], scenes: Sequence[str]
) -> dict[str, float]:
    """Field means over a (possibly resampled, possibly repeated) scene list."""
    means: dict[str, float] = {}
    for field in FIELDS:
        total, count = 0.0, 0
        for scene in scenes:
            scene_total, scene_count = aggregates[scene][field]
            total += scene_total
            count += scene_count
        means[field] = total / count if count else float("nan")
    return means


def quantity_formulas(analysis: AnalysisConfig) -> dict[str, Callable[[dict[str, float]], float]]:
    """Every reported quantity as a closed form over one replicate's means."""
    eps = analysis.epsilon_margin

    def retained(m: dict[str, float]) -> float:
        oracle_margin = m["oracle_m"] - m["nowarp_m"]
        if not math.isfinite(oracle_margin) or abs(oracle_margin) < eps:
            return float("nan")
        return (m["est"] - m["nowarp_m"]) / oracle_margin

    return {
        "matched_ceiling": lambda m: m["oracle_m"],
        "estimated_score": lambda m: m["est"],
        "matched_floor": lambda m: m["nowarp_m"],
        "depth_tax": lambda m: m["oracle_m"] - m["est"],
        "oracle_margin": lambda m: m["oracle_m"] - m["nowarp_m"],
        "estimated_margin": lambda m: m["est"] - m["nowarp_m"],
        "retained_fraction": retained,
        "selection_differential": lambda m: m["oracle_full"] - m["oracle_m"],
        "transported_fraction": lambda m: m["transported_fraction"],
        "boundary_minus_interior_tax": lambda m: m["tax_boundary"] - m["tax_interior"],
        "lowtex_minus_hightex_tax": lambda m: m["tax_lowtex"] - m["tax_hightex"],
    }


def support_counts(records: Sequence[dict]) -> dict[str, int]:
    return {
        "n_scenes": len({r["scene"] for r in records}),
        "n_camera_pairs": len({r["camera_pair"] for r in records}),
        "n_feature_comparisons": int(sum(r["n"] for r in records)),
    }


def is_supported(counts: dict[str, int], analysis: AnalysisConfig) -> bool:
    return (
        counts["n_scenes"] >= analysis.support_min_scenes
        and counts["n_camera_pairs"] >= analysis.support_min_camera_pairs
    )


def cell_summary(
    records: Sequence[dict],
    analysis: AnalysisConfig,
    formulas: dict[str, Callable[[dict[str, float]], float]],
    with_ci: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Point estimates, paired scene-bootstrap intervals, and support.

    One draw of scenes serves every quantity in the cell, so the intervals
    around a tax, its margins, and their ratio all come from the same
    resampled scenes and carry the covariance the pairing gives them. The
    replicate recomputes each quantity from its own means; nothing is
    assembled from independently bootstrapped components.
    """
    counts = support_counts(records)
    out: dict[str, Any] = {**counts, "supported": is_supported(counts, analysis)}
    aggregates = scene_aggregates(records)
    scenes = sorted(aggregates)
    point = pooled_means(aggregates, scenes)
    for name, formula in formulas.items():
        out[name] = formula(point)
    if not with_ci or not scenes:
        return out

    rng = np.random.default_rng(analysis.bootstrap_seed)
    draws: dict[str, list[float]] = {name: [] for name in with_ci}
    for _ in range(analysis.bootstrap_resamples):
        picked = [scenes[i] for i in rng.integers(0, len(scenes), size=len(scenes))]
        means = pooled_means(aggregates, picked)
        for name in with_ci:
            value = formulas[name](means)
            if math.isfinite(value):
                draws[name].append(value)
    tail = (1.0 - analysis.bootstrap_confidence) / 2.0
    for name in with_ci:
        values = draws[name]
        out[f"{name}_ci_low"] = float(np.quantile(values, tail)) if values else float("nan")
        out[f"{name}_ci_high"] = (
            float(np.quantile(values, 1.0 - tail)) if values else float("nan")
        )
        out[f"{name}_ci_replicates"] = len(values)
    out["bootstrap_resamples"] = analysis.bootstrap_resamples
    return out


# ---------------------------------------------------------------------------
# Table 1: the alignment ladder
# ---------------------------------------------------------------------------

LADDER_CI = ("depth_tax", "oracle_margin", "estimated_margin", "retained_fraction",
             "selection_differential", "boundary_minus_interior_tax",
             "lowtex_minus_hightex_tax")


def ladder_table(records: Sequence[dict], analysis: AnalysisConfig) -> list[dict[str, Any]]:
    """PROTOCOL 4.10 Table 1: per regime and pooled, per metric, path, level."""
    formulas = quantity_formulas(analysis)
    table: list[dict[str, Any]] = []
    scopes = [("pooled", None), ("rotation", "rotation"),
              ("translation", "translation"), ("orbit", "orbit")]
    for scope_name, regime in scopes:
        scoped = [r for r in records if regime is None or r["regime"] == regime]
        for key, cell in sorted(group_by(scoped, ("metric", "path", "level")).items()):
            metric, path, level = key
            summary = cell_summary(cell, analysis, formulas, with_ci=LADDER_CI)
            table.append({
                "analysis": scope_name, "metric": metric, "path": path,
                "level": level, **summary,
            })
    return table


def bin_table(records: Sequence[dict], analysis: AnalysisConfig) -> list[dict[str, Any]]:
    """Per-bin quantities behind Figures 1 to 4, one row per reported cell."""
    formulas = quantity_formulas(analysis)
    table: list[dict[str, Any]] = []
    for axis, regime in PRIMARY_REGIME.items():
        scoped = [r for r in records if r["regime"] == regime]
        for key, cell in sorted(group_by(scoped, ("metric", "path", "level", axis)).items()):
            metric, path, level, label = key
            summary = cell_summary(cell, analysis, formulas, with_ci=LADDER_CI)
            table.append({
                "analysis": regime, "axis": axis, "bin": label, "metric": metric,
                "path": path, "level": level, **summary,
            })
    joint = [r for r in records if r["regime"] == JOINT_REGIME]
    for key, cell in sorted(
        group_by(joint, ("metric", "path", "level", "rotation_bin", "parallax_bin")).items()
    ):
        metric, path, level, rlabel, plabel = key
        summary = cell_summary(cell, analysis, formulas, with_ci=("depth_tax",))
        table.append({
            "analysis": JOINT_REGIME, "axis": "rotation_bin x parallax_bin",
            "bin": f"{rlabel} x {plabel}", "metric": metric, "path": path,
            "level": level, **summary,
        })
    return table


# ---------------------------------------------------------------------------
# Addendum A: dual-path near-zero disclosure
# ---------------------------------------------------------------------------

DISCLOSED_QUANTITIES = ("depth_tax", "estimated_margin", "selection_differential")


def near_zero_disclosure(
    bin_rows: Sequence[dict[str, Any]], analysis: AnalysisConfig
) -> list[dict[str, Any]]:
    """Score-space effects at the scale of the operator gate, both paths shown.

    The band is the frozen 0.003 operator tolerance. It applies to score-space
    quantities only; retained and transported fractions are dimensionless and
    are governed by epsilon_margin instead (Addendum A). The two estimates are
    each path's own matched value; the frozen three-way reading restricts what
    may be claimed.
    """
    band = analysis.path_agreement_tolerance
    by_cell: dict[tuple, dict[str, dict]] = {}
    for row in bin_rows:
        key = (row["analysis"], row["axis"], row["bin"], row["metric"], row["level"])
        by_cell.setdefault(key, {})[row["path"]] = row
    out: list[dict[str, Any]] = []
    for key, paths in sorted(by_cell.items()):
        if PER_POINT not in paths or SPLAT_POOL not in paths:
            continue
        if not (paths[PER_POINT]["supported"] and paths[SPLAT_POOL]["supported"]):
            continue
        for quantity in DISCLOSED_QUANTITIES:
            m_pp = paths[PER_POINT][quantity]
            m_sp = paths[SPLAT_POOL][quantity]
            if not (math.isfinite(m_pp) and math.isfinite(m_sp)):
                continue
            in_band = [abs(m_pp) <= band, abs(m_sp) <= band]
            if not any(in_band):
                continue
            lo_pp = paths[PER_POINT].get(f"{quantity}_ci_low", float("nan"))
            hi_pp = paths[PER_POINT].get(f"{quantity}_ci_high", float("nan"))
            lo_sp = paths[SPLAT_POOL].get(f"{quantity}_ci_low", float("nan"))
            hi_sp = paths[SPLAT_POOL].get(f"{quantity}_ci_high", float("nan"))
            excludes = lo_pp * hi_pp > 0 and lo_sp * hi_sp > 0
            same_sign = (m_pp > 0) == (m_sp > 0)
            if same_sign and excludes and all(in_band):
                case = "both_in_band"
                sentence = (
                    "a small, sign-consistent effect under both evaluation paths"
                )
            elif same_sign and excludes and sum(in_band) == 1:
                case = "one_in_band"
                sentence = (
                    "an effect present under both evaluation paths whose "
                    "magnitude is path-sensitive"
                )
            elif same_sign and excludes:
                case = "neither_in_band"
                sentence = ""
            else:
                case = "not_robust"
                sentence = (
                    "no robust effect; the measurement is at the scale of "
                    "evaluation-path choice"
                )
            out.append({
                "analysis": key[0], "axis": key[1], "bin": key[2],
                "metric": key[3], "level": key[4], "quantity": quantity,
                "M_pp": m_pp, "M_sp": m_sp, "dM": m_pp - m_sp,
                "M_pp_ci": [lo_pp, hi_pp], "M_sp_ci": [lo_sp, hi_sp],
                "band": band, "case": case, "sentence": sentence,
            })
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _rows_for(bin_rows, **filters):
    out = []
    for row in bin_rows:
        if all(row.get(k) == v for k, v in filters.items()):
            out.append(row)
    return out


def _band(panel, positions):
    for position in positions:
        panel.axvspan(position - 0.5, position + 0.5, color="0.55", alpha=0.22, zorder=0)


def figure1_tax_vs_parallax(bin_rows, path_out: Path, analysis: AnalysisConfig) -> None:
    """Figure 1: the tax versus parallax, translation regime, one curve per level.

    Coverage is shown in the companion row so a better score cannot hide a
    shrinking transported population (PROTOCOL 4.4).
    """
    plt = _pyplot()
    order = [b for b in bin_order(analysis.parallax_edges())]
    figure, axes = plt.subplots(2, len(METRICS), figsize=(6.4 * len(METRICS), 8.0),
                                squeeze=False)
    for column, metric in enumerate(METRICS):
        tax_panel, coverage_panel = axes[0][column], axes[1][column]
        unsupported: set[int] = set()
        for level, variant_name in LEVELS:
            rows = _rows_for(bin_rows, analysis="translation", metric=metric,
                             path=PER_POINT, level=level)
            by_bin = {r["bin"]: r for r in rows}
            present = [b for b in order if b in by_bin]
            xs = [order.index(b) for b in present]
            ys = [by_bin[b]["depth_tax"] for b in present]
            low = [max(0.0, y - by_bin[b]["depth_tax_ci_low"]) for b, y in zip(present, ys)]
            high = [max(0.0, by_bin[b]["depth_tax_ci_high"] - y) for b, y in zip(present, ys)]
            tax_panel.errorbar(xs, ys, yerr=[low, high], fmt="-o", capsize=2,
                               markersize=4, label=variant_name)
            coverage_panel.plot(
                xs, [by_bin[b]["transported_fraction"] for b in present],
                "-s", markersize=4, label=variant_name,
            )
            unsupported |= {order.index(b) for b in present if not by_bin[b]["supported"]}
        for panel in (tax_panel, coverage_panel):
            _band(panel, sorted(unsupported))
            panel.set_xticks(range(len(order)))
            panel.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
            panel.grid(alpha=0.3)
            panel.legend(fontsize=6)
        tax_panel.axhline(0.0, color="black", linewidth=1)
        tax_panel.set_title(f"depth tax, {metric}, per-point", fontsize=9)
        tax_panel.set_ylabel("matched ceiling minus estimated score", fontsize=8)
        coverage_panel.set_title("transported fraction beside it", fontsize=9)
        coverage_panel.set_ylabel("surviving fraction of GT co-visible", fontsize=8)
        coverage_panel.set_xlabel("parallax bin", fontsize=9)
    figure.suptitle(
        "Figure 1: estimated-geometry tax versus parallax, translation regime",
        fontsize=11,
    )
    figure.tight_layout()
    path_out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_out, dpi=150)
    plt.close(figure)


def figure2_rotation_control(ladder_rows, gate_summary, path_out: Path) -> None:
    """Figure 2: the pure-rotation identity check and the collision-ordering tax.

    The overlap of every variant with the matched Oracle ceiling is the
    correctness control the 4.5 gates enforce; the unforced collision-ordering
    tax is shown beside it, small and separate.
    """
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    panel = axes[0]
    labels, oracle_values, est_values = [], [], []
    for row in ladder_rows:
        if (row["analysis"], row["metric"], row["path"]) != ("rotation", RAW, PER_POINT):
            continue
        labels.append(row["level"])
        oracle_values.append(row["matched_ceiling"])
        est_values.append(row["estimated_score"])
    xs = range(len(labels))
    panel.plot(xs, oracle_values, "o", markersize=9, fillstyle="none",
               label="matched Oracle-Transport")
    panel.plot(xs, est_values, "x", markersize=7, label="estimated depth")
    panel.set_xticks(list(xs))
    panel.set_xticklabels(labels)
    panel.set_title("pure rotation, per-point: the variants must overlap", fontsize=9)
    panel.set_ylabel("raw cosine", fontsize=9)
    panel.grid(alpha=0.3)
    panel.legend(fontsize=7)

    panel = axes[1]
    levels = list(gate_summary["collision_tax_raw_by_level"])
    values = [gate_summary["collision_tax_raw_by_level"][l] for l in levels]
    panel.bar(range(len(levels)), values)
    panel.set_xticks(range(len(levels)))
    panel.set_xticklabels(levels)
    panel.axhline(0.0, color="black", linewidth=1)
    panel.set_title(
        "unforced collision-ordering tax, raw, splat-and-pool only", fontsize=9
    )
    panel.set_ylabel("forced score minus ordinary score", fontsize=8)
    panel.grid(alpha=0.3)
    figure.suptitle(
        "Figure 2: pure-rotation correctness control under forced collision order",
        fontsize=11,
    )
    figure.tight_layout()
    path_out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_out, dpi=150)
    plt.close(figure)


def figure3_orbit(bin_rows, path_out: Path, analysis: AnalysisConfig,
                  metric: str = CENTERED) -> None:
    """Figure 3: orbit joint analysis, rotation by parallax, plotting the tax."""
    plt = _pyplot()
    rows = [b for b in bin_order(analysis.rotation_edges())]
    cols = [b for b in bin_order(analysis.parallax_edges())]
    level_names = [level for level, _ in LEVELS]
    figure, axes = plt.subplots(1, len(level_names),
                                figsize=(4.4 * len(level_names), 4.6), squeeze=False)
    populated = _rows_for(bin_rows, analysis=JOINT_REGIME, metric=metric, path=PER_POINT)
    extent = max(
        (abs(r["depth_tax"]) for r in populated if math.isfinite(r["depth_tax"])),
        default=1.0,
    )
    for column, level in enumerate(level_names):
        panel = axes[0][column]
        grid = np.full((len(rows), len(cols)), np.nan)
        cells = {r["bin"]: r for r in populated if r["level"] == level}
        for i, rlabel in enumerate(rows):
            for j, plabel in enumerate(cols):
                row = cells.get(f"{rlabel} x {plabel}")
                if row is None:
                    continue
                grid[i, j] = row["depth_tax"]
                if not row["supported"]:
                    panel.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                        edgecolor="0.85", linewidth=0.0,
                    ))
                panel.text(j, i, f"{row['depth_tax']:+.3f}\nn={row['n_camera_pairs']}",
                           ha="center", va="center", fontsize=5)
        image = panel.imshow(grid, cmap="RdBu_r", vmin=-extent, vmax=extent, aspect="auto")
        panel.set_xticks(range(len(cols)))
        panel.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        panel.set_yticks(range(len(rows)))
        panel.set_yticklabels(rows, fontsize=7)
        panel.set_title(dict(LEVELS)[level], fontsize=8)
        figure.colorbar(image, ax=panel, fraction=0.046)
    figure.suptitle(
        f"Figure 3: orbit interaction, depth tax, {metric}, per-point. Blank "
        "cell = combination the program cannot produce; hatched = below support.",
        fontsize=9,
    )
    figure.tight_layout()
    path_out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_out, dpi=150)
    plt.close(figure)


def figure4_localization(ladder_rows, path_out: Path) -> None:
    """Figure 4: error localization, depth boundary versus interior, texture beside."""
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for panel, quantity, title in (
        (axes[0], "boundary_minus_interior_tax",
         "depth-boundary tax minus interior tax"),
        (axes[1], "lowtex_minus_hightex_tax",
         "low-texture tax minus high-texture tax"),
    ):
        labels, values, low, high = [], [], [], []
        for row in ladder_rows:
            if (row["analysis"], row["metric"], row["path"]) != ("pooled", CENTERED, PER_POINT):
                continue
            value = row[quantity]
            if not math.isfinite(value):
                continue
            labels.append(row["level"])
            values.append(value)
            low.append(max(0.0, value - row[f"{quantity}_ci_low"]))
            high.append(max(0.0, row[f"{quantity}_ci_high"] - value))
        panel.errorbar(range(len(labels)), values, yerr=[low, high], fmt="o", capsize=3)
        panel.set_xticks(range(len(labels)))
        panel.set_xticklabels(labels)
        panel.axhline(0.0, color="black", linewidth=1)
        panel.set_title(title, fontsize=9)
        panel.grid(alpha=0.3)
    figure.suptitle(
        "Figure 4: error localization, centered, per-point, paired scene bootstrap",
        fontsize=11,
    )
    figure.tight_layout()
    path_out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_out, dpi=150)
    plt.close(figure)


def collision_summary(records: Sequence[dict]) -> dict[str, Any]:
    """The unforced collision-ordering tax per level, splat path, rotation only."""
    out: dict[str, Any] = {"collision_tax_raw_by_level": {},
                           "collision_tax_centered_by_level": {}}
    rotation = [
        r for r in records
        if r["regime"] == "rotation" and r["path"] == SPLAT_POOL and r["metric"] == RAW
    ]
    for level, _ in LEVELS:
        values_raw = [
            r["collision_tax_raw"] for r in rotation
            if r["level"] == level and math.isfinite(r["collision_tax_raw"])
        ]
        values_cen = [
            r["collision_tax_centered"] for r in rotation
            if r["level"] == level and math.isfinite(r["collision_tax_centered"])
        ]
        if values_raw:
            out["collision_tax_raw_by_level"][level] = float(np.mean(values_raw))
        if values_cen:
            out["collision_tax_centered_by_level"][level] = float(np.mean(values_cen))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4 tables and figures.")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    analysis = load_analysis_config(args.analysis_config)
    out_dir = args.out_dir or Path(args.eval_dir).parent

    rows = read_phase4_dir(args.eval_dir, analysis)
    records = build_records(rows, analysis)
    print(f"read {len(rows)} rows, {len(records)} paired records")

    ladder = ladder_table(records, analysis)
    bins = bin_table(records, analysis)
    disclosure = near_zero_disclosure(bins + ladder_with_axis(ladder), analysis)
    gates = collision_summary(records)

    write_table(Path(out_dir) / "tables" / "phase4_ladder.parquet", ladder)
    write_table(Path(out_dir) / "tables" / "phase4_bins.parquet", bins)
    (Path(out_dir) / "tables" / "phase4_near_zero.json").write_text(
        json.dumps(disclosure, indent=1)
    )
    figure1_tax_vs_parallax(bins, Path(out_dir) / "figures" / "phase4_figure1_tax_vs_parallax.png", analysis)
    figure2_rotation_control(ladder, gates, Path(out_dir) / "figures" / "phase4_figure2_rotation_control.png")
    figure3_orbit(bins, Path(out_dir) / "figures" / "phase4_figure3_orbit.png", analysis)
    figure4_localization(ladder, Path(out_dir) / "figures" / "phase4_figure4_localization.png")
    print(f"tables and figures -> {out_dir}")
    flagged = [d for d in disclosure if d["case"] != "neither_in_band"]
    print(f"near-zero disclosure: {len(flagged)} flagged cells -> phase4_near_zero.json")


def ladder_with_axis(ladder: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ladder rows in the shape the disclosure walker expects."""
    return [{**row, "axis": "overall", "bin": "overall"} for row in ladder]


if __name__ == "__main__":
    main()
