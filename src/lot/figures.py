"""Figures and tables, built from outputs/eval/*.parquet and the analysis config.

CLAUDE.md requires every figure be regenerable from the evaluation parquet
alone; PROTOCOL 3.2 requires the bin edges live in a committed analysis config
rather than in rows or in source. So this module reads exactly two things: the
parquet, and configs/analysis.yaml. It never touches a render or a cache.

Four things this layer is responsible for and the evaluation layer is not.

Binning. Rows carry continuous rotation_deg and parallax. The edges come from
the config, so changing a bin is a config edit and an amendment, never a source
change.

Regime discipline, PROTOCOL 3.3. In-place rotation is the sole source of the
primary rotation analysis and translation the sole source of the primary
parallax analysis, because each regime holds the other axis at exactly zero.
Orbit varies on both at once and appears only in the joint view. Orbit pairs on
a primary curve would silently mix an interaction into a marginal.

Support and uncertainty, PROTOCOL 3.4. Every bin reports how many scenes, how
many camera pairs, and how many feature comparisons stand behind it, and every
reported number carries a bootstrap interval resampled at the scene level.
Unsupported bins stay plotted, shaded, and labelled with their n; they are never
used for a headline.

Pairing, PROTOCOL 3.7. A margin is a difference measured on one pair, between
variants that scored the same records. The evaluation layer arranges that by
scoring every variant on the path's common valid set; this layer verifies it
from the persisted masks and refuses to subtract across a mismatch.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .analysis_config import DEFAULT_CONFIG_PATH, AnalysisConfig, load_analysis_config
from .datasets import bin_order, parallax_bin, rotation_bin
from .evaluate import (
    MEAN_FEATURE,
    NEIGHBOR_PATCH,
    NO_WARP_COPY,
    ORACLE_TRANSPORT,
    PER_POINT,
    RANDOM_PATCH,
    SPLAT_POOL,
)

PATH_ORDER = (PER_POINT, SPLAT_POOL)
# Ladder order: worst-case null first, correct answer last.
VARIANT_ORDER = (
    RANDOM_PATCH,
    MEAN_FEATURE,
    NO_WARP_COPY,
    NEIGHBOR_PATCH,
    ORACLE_TRANSPORT,
)

# PROTOCOL 3.3: which regime is the sole source of which primary analysis.
PRIMARY_REGIME = {"parallax_bin": "translation", "rotation_bin": "rotation"}
JOINT_REGIME = "orbit"

# What identifies one comparison, up to which method was used.
COMPARISON_KEYS = ("scene", "context_frame_id", "target_frame_id", "encoder", "path")

RAW = "cosine_mean"
CENTERED = "cosine_centered_mean"


def read_eval_dir(eval_dir: Path) -> list[dict[str, Any]]:
    """Read every per-scene parquet in a directory into one list of rows."""
    import pyarrow.parquet as pq

    eval_dir = Path(eval_dir)
    files = sorted(eval_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {eval_dir}")
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


# ---------------------------------------------------------------------------
# Binning, applied here and only here
# ---------------------------------------------------------------------------

def assign_bins(rows: Iterable[dict[str, Any]], config: AnalysisConfig) -> list[dict[str, Any]]:
    """Attach bin labels from the committed config to rows that carry no labels."""
    out = []
    for row in rows:
        if "parallax_bin" in row or "rotation_bin" in row:
            raise ValueError(
                "rows already carry bin labels; PROTOCOL 3.2 keeps labels out of "
                "rows so the analysis config is the only place edges live"
            )
        out.append(
            {
                **row,
                "parallax_bin": parallax_bin(row["parallax"], config),
                "rotation_bin": rotation_bin(row["rotation_deg"], config),
            }
        )
    return out


def restrict_to_regime(
    records: Sequence[dict[str, Any]], axis: str
) -> list[dict[str, Any]]:
    """PROTOCOL 3.3: keep only the regime that is the sole source of this axis.

    The other regimes hold this axis at exactly zero, or vary it jointly with
    the other axis. Either way their pairs are not points on this curve.
    """
    if axis not in PRIMARY_REGIME:
        raise ValueError(f"no primary regime defined for {axis!r}")
    regime = PRIMARY_REGIME[axis]
    return [r for r in records if r["regime"] == regime]


def assert_single_regime(records: Sequence[dict[str, Any]], regime: str) -> None:
    """Guard a primary curve against a pair from another regime."""
    found = {r["regime"] for r in records}
    if found - {regime}:
        raise ValueError(
            f"a primary {regime} analysis received pairs from {sorted(found - {regime})}; "
            "PROTOCOL 3.3 keeps orbit out of both marginals"
        )


# ---------------------------------------------------------------------------
# Pairing on sample identity
# ---------------------------------------------------------------------------

def paired_records(
    rows: Iterable[dict[str, Any]], metric: str = RAW
) -> tuple[list[dict[str, Any]], int]:
    """One record per comparison and variant, carrying its margin over the floor.

    PROTOCOL 3.7 makes a margin a difference between variants measured on the
    same records. The evaluation layer scores every variant of a path on that
    path's common valid set, so the masks within a comparison should be
    identical and the difference of the two means is then the paired difference.
    That is verified here rather than assumed: a comparison whose variant and
    floor carry different masks is excluded and counted, because its difference
    of means would be a difference of populations wearing the shape of a method
    effect.

    Returns (records, mask_mismatches).
    """
    grouped: dict[tuple, dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in COMPARISON_KEYS)
        grouped.setdefault(key, {})[row["variant"]] = row

    records: list[dict[str, Any]] = []
    mismatches = 0
    for variants in grouped.values():
        floor = variants.get(NO_WARP_COPY)
        if floor is None or not _finite(floor[metric]):
            continue
        for variant, row in variants.items():
            if not _finite(row[metric]):
                continue
            if row["sample_mask"] != floor["sample_mask"]:
                mismatches += 1
                continue
            records.append(
                {
                    "scene": row["scene"],
                    "split": row["split"],
                    "regime": row["regime"],
                    "camera_pair": (row["context_frame_id"], row["target_frame_id"]),
                    "parallax_bin": row["parallax_bin"],
                    "rotation_bin": row["rotation_bin"],
                    "parallax": row["parallax"],
                    "rotation_deg": row["rotation_deg"],
                    "encoder": row["encoder"],
                    "path": row["path"],
                    "metric": metric,
                    "variant": variant,
                    "value": row[metric],
                    "margin": row[metric] - floor[metric],
                    "n": row["n"],
                }
            )
    return records, mismatches


# ---------------------------------------------------------------------------
# Support and uncertainty
# ---------------------------------------------------------------------------

def support_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    """PROTOCOL 3.4's three counts for one cell."""
    return {
        "n_scenes": len({r["scene"] for r in records}),
        "n_camera_pairs": len({r["camera_pair"] for r in records}),
        "n_feature_comparisons": int(sum(r["n"] for r in records)),
    }


def is_supported(counts: dict[str, int], config: AnalysisConfig) -> bool:
    """The support decision rests on scenes and camera pairs, not raw comparisons."""
    return (
        counts["n_scenes"] >= config.support_min_scenes
        and counts["n_camera_pairs"] >= config.support_min_camera_pairs
    )


def mean_value(records: Sequence[dict[str, Any]]) -> float:
    """Point estimate of an absolute score."""
    values = [r["value"] for r in records]
    return float(np.mean(values)) if values else float("nan")


def mean_margin(records: Sequence[dict[str, Any]]) -> float:
    """Point estimate of a margin over the floor."""
    values = [r["margin"] for r in records]
    return float(np.mean(values)) if values else float("nan")


def cell_estimates(
    records: Sequence[dict[str, Any]],
    keys: Sequence[str],
    statistic: Callable[[Sequence[dict[str, Any]]], float],
) -> dict[tuple, float]:
    """The whole table of point estimates in one pass."""
    return {key: statistic(cell) for key, cell in group_by(records, keys).items()}


def bootstrap_cells(
    records: Sequence[dict[str, Any]],
    keys: Sequence[str],
    statistic: Callable[[Sequence[dict[str, Any]]], float],
    config: AnalysisConfig,
    unit: str = "scene",
) -> dict[tuple, tuple[float, float]]:
    """Resample whole units once per replicate and recompute the entire table.

    The loop is cells inside replicates, not replicates inside cells. Resampling
    the scene ids once and recomputing every cell from that one draw costs a
    thousand passes over the records in total rather than a thousand per cell,
    and it has a second property worth more than the speed: every cell in a
    replicate sees the same scenes, so the replicates carry the cross-cell
    correlation that simultaneous bands would need.

    The replicate calls the same function that produced the point estimate. For
    a mean that agrees with resampling precomputed per-unit values; it is
    written this way because Phase 4's retained fractions and selection
    differentials are ratio statistics, where resampling precomputed values is
    wrong, and one mechanism that is right everywhere beats two kept in step.
    """
    if not records:
        return {}
    by_unit: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        by_unit.setdefault(record[unit], []).append(record)
    units = sorted(by_unit, key=repr)
    rng = np.random.default_rng(config.bootstrap_seed)
    collected: dict[tuple, list[float]] = {}
    for _ in range(config.bootstrap_resamples):
        drawn = rng.integers(0, len(units), size=len(units))
        sample: list[dict[str, Any]] = []
        for position in drawn:
            sample.extend(by_unit[units[position]])
        for key, value in cell_estimates(sample, keys, statistic).items():
            if math.isfinite(value):
                collected.setdefault(key, []).append(value)
    tail = (1.0 - config.bootstrap_confidence) / 2.0
    return {
        key: (float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail)))
        for key, values in collected.items()
        if values
    }


def summaries_for(
    records: Sequence[dict[str, Any]],
    keys: Sequence[str],
    config: AnalysisConfig,
    statistic: Callable[[Sequence[dict[str, Any]]], float] = mean_margin,
) -> dict[tuple, dict[str, Any]]:
    """Point estimate, both intervals, and support for every cell, computed once."""
    grouped = group_by(records, keys)
    scene_ci = bootstrap_cells(records, keys, statistic, config, unit="scene")
    pair_ci = bootstrap_cells(records, keys, statistic, config, unit="camera_pair")
    nan = (float("nan"), float("nan"))
    out: dict[tuple, dict[str, Any]] = {}
    for key, cell in grouped.items():
        counts = support_counts(cell)
        low, high = scene_ci.get(key, nan)
        pair_low, pair_high = pair_ci.get(key, nan)
        out[key] = {
            **counts,
            "estimate": statistic(cell),
            "ci_low": low,
            "ci_high": high,
            "pair_ci_low": pair_low,
            "pair_ci_high": pair_high,
            "supported": is_supported(counts, config),
        }
    return out


def bootstrap_interval(
    records: Sequence[dict[str, Any]],
    statistic: Callable[[Sequence[dict[str, Any]]], float],
    config: AnalysisConfig,
    unit: str = "scene",
) -> tuple[float, float]:
    """Resample whole units and recompute the statistic inside every replicate.

    PROTOCOL 3.4 puts the primary interval at the scene level and the secondary
    at the camera-pair level; individual points and patches are never resampled,
    because records within a scene are not independent draws.

    The replicate calls the same function that produced the point estimate. For
    a mean that is more work than resampling precomputed per-scene values, and
    the two agree. It is written this way because Phase 4's retained fractions
    and selection differentials are ratio statistics, where resampling
    precomputed values is simply wrong, and one mechanism that is right
    everywhere beats two that must be kept in step.
    """
    intervals = bootstrap_cells(records, (), statistic, config, unit=unit)
    return intervals.get((), (float("nan"), float("nan")))


def cell_summary(
    records: Sequence[dict[str, Any]],
    config: AnalysisConfig,
    statistic: Callable[[Sequence[dict[str, Any]]], float] = mean_margin,
) -> dict[str, Any]:
    """Point estimate, both intervals, and the three support counts for one cell."""
    counts = support_counts(records)
    low, high = bootstrap_interval(records, statistic, config, unit="scene")
    pair_low, pair_high = bootstrap_interval(records, statistic, config, unit="camera_pair")
    return {
        **counts,
        "estimate": statistic(records),
        "ci_low": low,
        "ci_high": high,
        "pair_ci_low": pair_low,
        "pair_ci_high": pair_high,
        "supported": is_supported(counts, config),
    }


def group_by(
    records: Iterable[dict[str, Any]], keys: Sequence[str]
) -> dict[tuple, list[dict[str, Any]]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(tuple(record[k] for k in keys), []).append(record)
    return grouped


def matched_orbit_minus_translation(records: Sequence[dict[str, Any]]) -> float:
    """Orbit's margin minus translation's, within one parallax bin.

    Figure D exists to ask whether rotation adds anything once parallax is
    controlled, and the orbit band cannot answer that on its own: the circle
    ties baseline to rotation, so orbit's low-rotation cells can be empty for
    the same reason the band exists. Translation is the rotation-near-zero
    reference at every parallax, so the comparison that carries the claim is
    orbit against translation at matched parallax. Reading a colour gradient
    along the band is suggestive; this is the test.
    """
    orbit = [r for r in records if r["regime"] == JOINT_REGIME]
    translation = [r for r in records if r["regime"] == PRIMARY_REGIME["parallax_bin"]]
    if not orbit or not translation:
        return float("nan")
    return mean_margin(orbit) - mean_margin(translation)


# ---------------------------------------------------------------------------
# PROTOCOL 3.9: the operational transport check
# ---------------------------------------------------------------------------

def path_agreement(rows: Sequence[dict[str, Any]], config: AnalysisConfig) -> dict[str, Any]:
    """Compare the two paths on the cells both scored, per PROTOCOL 3.9.

    The evaluation layer scored each path on the cross-path intersection and
    emitted it as its own column, so this is a comparison of one population by
    two operators. Comparing the full-population scores instead would fold the
    coverage difference into the operator difference, and the coverage
    difference is exactly what is reported beside it rather than inside it.
    """
    by_key: dict[tuple, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["variant"] != ORACLE_TRANSPORT:
            continue
        key = (row["scene"], row["context_frame_id"], row["target_frame_id"], row["encoder"])
        by_key.setdefault(key, {})[row["path"]] = row

    differences: list[float] = []
    coverage: list[int] = []
    for paths in by_key.values():
        if set(paths) != set(PATH_ORDER):
            continue
        a = paths[PER_POINT]["cosine_intersect_mean"]
        b = paths[SPLAT_POOL]["cosine_intersect_mean"]
        if _finite(a) and _finite(b):
            differences.append(abs(a - b))
            coverage.append(
                int(paths[PER_POINT]["coverage_difference"])
                + int(paths[SPLAT_POOL]["coverage_difference"])
            )
    if not differences:
        return {"comparisons": 0, "max_abs_difference": float("nan"), "within_tolerance": False}
    return {
        "comparisons": len(differences),
        "mean_abs_difference": float(np.mean(differences)),
        "max_abs_difference": float(np.max(differences)),
        "tolerance": config.path_agreement_tolerance,
        "within_tolerance": bool(np.max(differences) <= config.path_agreement_tolerance),
        "mean_coverage_difference_cells": float(np.mean(coverage)),
        "max_coverage_difference_cells": int(np.max(coverage)),
    }


# ---------------------------------------------------------------------------
# PROTOCOL 3.10: the four required figures
# ---------------------------------------------------------------------------

def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


UNSUPPORTED_BAND = dict(color="0.55", alpha=0.22, zorder=0)


def _shade_unsupported(panel, positions: Sequence[int], labelled: bool = True) -> None:
    """Shade a band behind bins below the support threshold.

    A band rather than a greyed marker: a greyed point on a line is close to
    invisible, and a band still reads when several series share one axis.
    """
    for position in positions:
        panel.axvspan(position - 0.5, position + 0.5, **UNSUPPORTED_BAND)
    if positions and labelled:
        panel.plot([], [], color="0.55", linewidth=8, alpha=0.35, label="below support")


def _annotate_counts(panel, positions, counts, y) -> None:
    """PROTOCOL 3.4 asks for n shown; shown for every bin, not only shaded ones."""
    for position, count in zip(positions, counts):
        panel.annotate(
            f"n={count}",
            (position, y),
            fontsize=6,
            ha="center",
            va="bottom",
            rotation=90,
            color="0.35",
        )


def figure_a_null_ladder(
    records: Sequence[dict[str, Any]],
    path: Path,
    config: AnalysisConfig,
    omissions: dict[str, int] | None = None,
) -> None:
    """Figure A: the full null ladder per encoder, raw and centered.

    Mean-Feature appears in raw only. Its prediction is the mean vector, so
    centering sends it to the zero vector and its centered cosine is undefined;
    PROTOCOL 3.7 records that as not applicable rather than manufacturing a
    number, and a marker drawn at zero would be exactly that manufacture.
    """
    plt = _pyplot()
    encoders = sorted({r["encoder"] for r in records})
    metrics = [RAW, CENTERED]
    figure, axes = plt.subplots(1, len(metrics), figsize=(6.0 * len(metrics), 4.6), squeeze=False)
    ladder = summaries_for(
        [r for r in records if r["path"] == PER_POINT],
        ("metric", "encoder", "variant"),
        config,
        statistic=mean_value,
    )
    for column, metric in enumerate(metrics):
        panel = axes[0][column]
        subset = [r for r in records if r["metric"] == metric and r["path"] == PER_POINT]
        variants = [v for v in VARIANT_ORDER if any(r["variant"] == v for r in subset)]
        for offset, encoder in enumerate(encoders):
            xs, ys, low, high, counts = [], [], [], [], []
            for position, variant in enumerate(variants):
                summary = ladder.get((metric, encoder, variant))
                if summary is None:
                    continue
                xs.append(position + (offset - (len(encoders) - 1) / 2) * 0.12)
                ys.append(summary["estimate"])
                low.append(max(0.0, summary["estimate"] - summary["ci_low"]))
                high.append(max(0.0, summary["ci_high"] - summary["estimate"]))
                counts.append(summary["n_camera_pairs"])
            if xs:
                panel.errorbar(
                    xs, ys, yerr=[low, high], fmt="o", capsize=3, markersize=5, label=encoder
                )
        panel.set_xticks(range(len(variants)))
        panel.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
        panel.set_title(
            metric + ("" if metric == RAW else "   Mean-Feature not applicable"), fontsize=10
        )
        panel.set_ylabel("cosine", fontsize=9)
        panel.grid(alpha=0.3)
        panel.legend(fontsize=7)
    if omissions:
        figure.text(
            0.5,
            0.005,
            "per-variant omissions: " + ", ".join(f"{k} {v}" for k, v in sorted(omissions.items())),
            ha="center",
            fontsize=7,
            color="0.35",
        )
    figure.suptitle("Figure A: null ladder, per-point path", fontsize=11)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def figure_ceiling_and_floor(
    records: Sequence[dict[str, Any]],
    path: Path,
    config: AnalysisConfig,
    axis: str,
    title: str,
) -> None:
    """Figures B and C: the ceiling and the floor as absolute curves.

    PROTOCOL 3.10 calls the floor curve mandatory. A ceiling plotted alone, or a
    margin plotted with the floor implicit at zero, reproduces the raw-cosine
    mistake the floors exist to prevent: it shows a number moving without
    showing what a trivial answer would have scored beside it.
    """
    plt = _pyplot()
    regime = PRIMARY_REGIME[axis]
    subset = restrict_to_regime(records, axis)
    assert_single_regime(subset, regime)
    subset = [r for r in subset if r["path"] == PER_POINT]
    if not subset:
        raise ValueError(f"no {regime} records to plot")
    encoders = sorted({r["encoder"] for r in subset})
    edges = config.parallax_edges() if axis == "parallax_bin" else config.rotation_edges()
    order = [b for b in bin_order(edges) if any(r[axis] == b for r in subset)]
    metrics = [RAW, CENTERED]
    cells = summaries_for(
        subset, ("metric", "encoder", axis, "variant"), config, statistic=mean_value
    )

    figure, axes = plt.subplots(
        len(metrics),
        len(encoders),
        figsize=(5.4 * len(encoders), 4.2 * len(metrics)),
        squeeze=False,
    )
    for row_index, metric in enumerate(metrics):
        for column, encoder in enumerate(encoders):
            panel = axes[row_index][column]
            unsupported: list[int] = []
            counts: list[int] = []
            series = {ORACLE_TRANSPORT: [], NO_WARP_COPY: []}
            bands = {ORACLE_TRANSPORT: ([], []), NO_WARP_COPY: ([], [])}
            nan_summary = {
                "estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "supported": False, "n_camera_pairs": 0,
            }
            for position, label in enumerate(order):
                supported = True
                for variant in (ORACLE_TRANSPORT, NO_WARP_COPY):
                    summary = cells.get((metric, encoder, label, variant), nan_summary)
                    series[variant].append(summary["estimate"])
                    bands[variant][0].append(summary["ci_low"])
                    bands[variant][1].append(summary["ci_high"])
                    supported = supported and summary["supported"]
                    if variant == ORACLE_TRANSPORT:
                        counts.append(summary["n_camera_pairs"])
                if not supported:
                    unsupported.append(position)
            positions = list(range(len(order)))
            _shade_unsupported(panel, unsupported)
            panel.fill_between(
                positions,
                series[NO_WARP_COPY],
                series[ORACLE_TRANSPORT],
                alpha=0.18,
                color="tab:green",
                label="margin",
            )
            for variant, style in ((ORACLE_TRANSPORT, "-o"), (NO_WARP_COPY, "--s")):
                panel.plot(positions, series[variant], style, markersize=4, label=variant)
                panel.fill_between(positions, bands[variant][0], bands[variant][1], alpha=0.15)
            finite = [v for v in series[NO_WARP_COPY] if math.isfinite(v)]
            if finite:
                _annotate_counts(panel, positions, counts, min(finite))
            panel.set_xticks(positions)
            panel.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
            panel.set_title(f"{encoder}   {metric}", fontsize=9)
            panel.set_xlabel(axis.replace("_", " "), fontsize=9)
            if column == 0:
                panel.set_ylabel("cosine", fontsize=9)
            panel.grid(alpha=0.3)
            panel.legend(fontsize=6)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def figure_d_orbit_joint(
    records: Sequence[dict[str, Any]],
    path: Path,
    config: AnalysisConfig,
    metric: str = CENTERED,
) -> None:
    """Figure D: orbit as a rotation-by-parallax heatmap, never collapsed.

    Orbit moves on both axes at once and the two are tied together by the orbit
    radius, so either marginal would report an interaction as if it were a main
    effect. The populated cells therefore form a band rather than filling the
    grid, and that shape is the point: an empty cell is a combination the camera
    program cannot produce, which is a fact about the design and not missing
    data. Empty cells stay blank, cells below support are hatched, and every
    populated cell carries its margin and its n.
    """
    plt = _pyplot()
    subset = [
        r
        for r in records
        if r["regime"] == JOINT_REGIME
        and r["path"] == PER_POINT
        and r["metric"] == metric
        and r["variant"] == ORACLE_TRANSPORT
    ]
    if not subset:
        raise ValueError("no orbit records to plot")
    encoders = sorted({r["encoder"] for r in subset})
    rows = [
        b for b in bin_order(config.rotation_edges()) if any(r["rotation_bin"] == b for r in subset)
    ]
    cols = [
        b for b in bin_order(config.parallax_edges()) if any(r["parallax_bin"] == b for r in subset)
    ]

    cells = summaries_for(
        subset, ("encoder", "rotation_bin", "parallax_bin"), config, statistic=mean_margin
    )
    summaries: dict[tuple[str, int, int], dict[str, Any]] = {}
    grids: dict[str, np.ndarray] = {}
    for encoder in encoders:
        grid = np.full((len(rows), len(cols)), np.nan)
        for i, rot in enumerate(rows):
            for j, par in enumerate(cols):
                summary = cells.get((encoder, rot, par))
                if summary is None:
                    continue
                summaries[(encoder, i, j)] = summary
                grid[i, j] = summary["estimate"]
        grids[encoder] = grid
    pooled = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
    # Diverging and centred on zero: the sign of the margin is the anchor. A
    # sequential scale shared across encoders would spend its range on whichever
    # encoder has the wider spread and bury the other's crossing of zero, which
    # for a position-indexed representation is the whole story.
    extent = float(np.max(np.abs(pooled))) if pooled.size else 1.0
    vmin, vmax = -extent, extent

    # The matched control: orbit minus translation at the same parallax.
    joint = [
        r
        for r in records
        if r["path"] == PER_POINT
        and r["metric"] == metric
        and r["variant"] == ORACLE_TRANSPORT
        and r["regime"] in (JOINT_REGIME, PRIMARY_REGIME["parallax_bin"])
    ]
    matched = summaries_for(
        joint, ("encoder", "parallax_bin"), config, statistic=matched_orbit_minus_translation
    )

    figure, axes = plt.subplots(
        2,
        len(encoders),
        figsize=(1.6 * max(len(cols), 3) * len(encoders) + 2, 2.1 * max(len(rows), 3) + 3.2),
        squeeze=False,
    )
    for column, encoder in enumerate(encoders):
        panel = axes[0][column]
        image = panel.imshow(grids[encoder], cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        for (owner, i, j), summary in summaries.items():
            if owner != encoder:
                continue
            if not summary["supported"]:
                panel.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="0.85",
                        linewidth=0.0,
                    )
                )
            midpoint = (vmin + vmax) / 2
            panel.text(
                j,
                i,
                f"{summary['estimate']:+.3f}\nn={summary['n_camera_pairs']}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if summary["estimate"] < midpoint else "black",
            )
        panel.set_xticks(range(len(cols)))
        panel.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        panel.set_yticks(range(len(rows)))
        panel.set_yticklabels(rows, fontsize=8)
        panel.set_xlabel("parallax bin", fontsize=9)
        if column == 0:
            panel.set_ylabel("rotation bin", fontsize=9)
        panel.set_title(encoder, fontsize=9)
        figure.colorbar(image, ax=panel, fraction=0.046)

        # Row two: the matched control the band cannot supply for itself.
        control = axes[1][column]
        present = [c for c in cols if (encoder, c) in matched]
        xs = [cols.index(c) for c in present]
        ys = [matched[(encoder, c)]["estimate"] for c in present]
        low = [max(0.0, y - matched[(encoder, c)]["ci_low"]) for c, y in zip(present, ys)]
        high = [max(0.0, matched[(encoder, c)]["ci_high"] - y) for c, y in zip(present, ys)]
        unsupported = [
            cols.index(c) for c in present if not matched[(encoder, c)]["supported"]
        ]
        _shade_unsupported(control, unsupported)
        control.axhline(0.0, color="black", linewidth=1)
        if xs:
            control.errorbar(xs, ys, yerr=[low, high], fmt="-o", capsize=3, markersize=4)
            _annotate_counts(
                control,
                xs,
                [matched[(encoder, c)]["n_camera_pairs"] for c in present],
                min(ys),
            )
        control.set_xticks(range(len(cols)))
        control.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        control.set_xlabel("parallax bin", fontsize=9)
        if column == 0:
            control.set_ylabel("orbit margin minus translation margin", fontsize=8)
        control.grid(alpha=0.3)
        control.set_title("matched control: rotation's effect at equal parallax", fontsize=8)
    figure.suptitle(
        f"Figure D: orbit joint analysis, margin over No-Warp-Copy, {metric}. "
        "Blank cell = combination the program cannot produce; hatched = below support. "
        "Lower row is the matched comparison against translation, the rotation-near-zero "
        "reference at each parallax.",
        fontsize=8,
    )
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Table and entrypoint
# ---------------------------------------------------------------------------

def summary_table(
    records: Sequence[dict[str, Any]], config: AnalysisConfig
) -> list[dict[str, Any]]:
    """One row per reported cell, with support and both intervals on every number."""
    table: list[dict[str, Any]] = []
    for axis, regime in PRIMARY_REGIME.items():
        edges = config.parallax_edges() if axis == "parallax_bin" else config.rotation_edges()
        scoped = [r for r in records if r["regime"] == regime]
        keys = ("encoder", "metric", "path", axis, "variant")
        values = summaries_for(scoped, keys, config, statistic=mean_value)
        margins = summaries_for(scoped, keys, config, statistic=mean_margin)
        for key in values:
            encoder, metric, path, label, variant = key
            summary = values[key]
            margin = margins[key]
            table.append(
                {
                    "analysis": regime,
                    "axis": axis,
                    "bin": label,
                    "bin_index": bin_order(edges).index(label),
                    "encoder": encoder,
                    "metric": metric,
                    "path": path,
                    "variant": variant,
                    "value": summary["estimate"],
                    "value_ci_low": summary["ci_low"],
                    "value_ci_high": summary["ci_high"],
                    "margin": margin["estimate"],
                    "margin_ci_low": margin["ci_low"],
                    "margin_ci_high": margin["ci_high"],
                    "margin_pair_ci_low": margin["pair_ci_low"],
                    "margin_pair_ci_high": margin["pair_ci_high"],
                    "n_scenes": summary["n_scenes"],
                    "n_camera_pairs": summary["n_camera_pairs"],
                    "n_feature_comparisons": summary["n_feature_comparisons"],
                    "supported": summary["supported"],
                }
            )
    # The matched control for the orbit interaction claim, as table rows so it
    # can be read without the figure.
    joint = [
        r
        for r in records
        if r["variant"] == ORACLE_TRANSPORT
        and r["regime"] in (JOINT_REGIME, PRIMARY_REGIME["parallax_bin"])
    ]
    matched = summaries_for(
        joint,
        ("encoder", "metric", "path", "parallax_bin"),
        config,
        statistic=matched_orbit_minus_translation,
    )
    for (encoder, metric, path, label), summary in matched.items():
        if not math.isfinite(summary["estimate"]):
            continue
        table.append(
            {
                "analysis": "orbit_minus_translation",
                "axis": "parallax_bin",
                "bin": label,
                "bin_index": bin_order(config.parallax_edges()).index(label),
                "encoder": encoder,
                "metric": metric,
                "path": path,
                "variant": ORACLE_TRANSPORT,
                "value": float("nan"),
                "value_ci_low": float("nan"),
                "value_ci_high": float("nan"),
                "margin": summary["estimate"],
                "margin_ci_low": summary["ci_low"],
                "margin_ci_high": summary["ci_high"],
                "margin_pair_ci_low": summary["pair_ci_low"],
                "margin_pair_ci_high": summary["pair_ci_high"],
                "n_scenes": summary["n_scenes"],
                "n_camera_pairs": summary["n_camera_pairs"],
                "n_feature_comparisons": summary["n_feature_comparisons"],
                "supported": summary["supported"],
            }
        )
    table.sort(key=lambda r: (r["analysis"], r["encoder"], r["metric"], r["path"],
                              r["bin_index"], r["variant"]))
    return table


def write_table(path: Path, table: Sequence[dict[str, Any]]) -> None:
    """Write the summary table as parquet. Refuses to overwrite."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; delete it to regenerate.")
    if not table:
        raise ValueError("no table rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(table)), path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the Experiment Zero figures and table from eval parquet."
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    config = load_analysis_config(args.analysis_config)
    out_dir = args.out_dir or Path(args.eval_dir).parent

    rows = assign_bins(read_eval_dir(args.eval_dir), config)
    records: list[dict[str, Any]] = []
    mismatches = 0
    for metric in (RAW, CENTERED):
        part, count = paired_records(rows, metric=metric)
        records.extend(part)
        mismatches += count
    print(f"read {len(rows)} rows, {len(records)} paired records")
    if mismatches:
        print(f"WARNING: {mismatches} comparisons excluded for mask mismatch")

    agreement = path_agreement(rows, config)
    print(
        f"PROTOCOL 3.9 path agreement on the cross-path intersection: "
        f"max |per_point - splat_pool| = {agreement['max_abs_difference']:.5f} "
        f"over {agreement['comparisons']} comparisons, tolerance "
        f"{agreement.get('tolerance', float('nan'))}, "
        f"within = {agreement['within_tolerance']}"
    )
    print(
        f"  coverage difference beside it: mean "
        f"{agreement.get('mean_coverage_difference_cells', float('nan')):.2f} cells, "
        f"max {agreement.get('max_coverage_difference_cells', 0)}"
    )

    table_path = Path(out_dir) / "tables" / "experiment_zero.parquet"
    write_table(table_path, summary_table(records, config))
    print(f"table  -> {table_path}")

    figures = Path(out_dir) / "figures"
    plan = [
        ("figure_a_null_ladder.png", lambda p: figure_a_null_ladder(records, p, config)),
        (
            "figure_b_parallax_translation.png",
            lambda p: figure_ceiling_and_floor(
                records, p, config, "parallax_bin",
                "Figure B: ceiling and floor versus parallax, translation regime",
            ),
        ),
        (
            "figure_c_rotation_inplace.png",
            lambda p: figure_ceiling_and_floor(
                records, p, config, "rotation_bin",
                "Figure C: ceiling and floor versus rotation angle, in-place rotation regime",
            ),
        ),
        ("figure_d_orbit_joint.png", lambda p: figure_d_orbit_joint(records, p, config)),
    ]
    for name, build in plan:
        target = figures / name
        try:
            build(target)
        except ImportError as error:
            print(f"figures skipped: {error}. Install matplotlib and rerun; the table is written.")
            break
        except ValueError as error:
            print(f"{name} skipped: {error}")
        else:
            print(f"figure -> {target}")


if __name__ == "__main__":
    main()
