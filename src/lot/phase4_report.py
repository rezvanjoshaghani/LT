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
from .phase4 import (
    GT_LEVEL,
    LEVELS,
    PHASE4_VERSION,
    POPULATION_FULL,
    POPULATION_MATCHED,
    phase4_measurement_digest,
)

RAW = "cosine_mean"
CENTERED = "cosine_centered_mean"
METRICS = (RAW, CENTERED)
# The cross-path common-valid columns the evaluator stored per row.
INTERSECT_OF = {RAW: "cosine_intersect_mean",
                CENTERED: "cosine_centered_intersect_mean"}
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
    # The config reading a run must be the config that produced it, in every
    # value that decided what the rows contain. Checking only that the files
    # agree with one another binds nothing: a whole directory measured under a
    # different confidence or validity rule agrees with itself perfectly, and
    # would be reported as this experiment.
    expected = any_meta.get("phase4_measurement_digest")
    active = phase4_measurement_digest(analysis)
    if expected != active:
        raise ValueError(
            f"{eval_dir} was evaluated under Phase 4 measurement config "
            f"{expected}, this analysis carries {active}. Those values decide "
            "what the rows contain, not how they are read; re-run the "
            "evaluation or analyse with the config it used"
        )
    if not any_meta.get("phase3_pairs_reconciled"):
        raise ValueError(
            f"{eval_dir} carries no record of its pair population having been "
            "reconciled against Phase 3; PROTOCOL 4.1 inherits those pairs"
        )
    return rows


def build_records(
    rows: Sequence[dict[str, Any]], analysis: AnalysisConfig
) -> list[dict[str, Any]]:
    """One record per (pair, path, metric, level) carrying every paired term."""
    by_key: dict[tuple, dict[str, dict[str, Any]]] = {}
    duplicates: list[tuple] = []
    for row in rows:
        key = (row["scene"], row["context_frame_id"], row["target_frame_id"], row["path"])
        slot = f"{row['level']}|{row['population']}|{row['variant']}"
        bucket = by_key.setdefault(key, {})
        if slot in bucket:
            duplicates.append((*key, slot))
        bucket[slot] = row
    if duplicates:
        # Assigning over a duplicate silently keeps whichever row was read
        # last, so a directory holding two runs would produce a table drawn
        # from a population nobody chose. Phase 3's analysis layer refuses the
        # same way.
        raise ValueError(
            f"{len(duplicates)} duplicate (comparison, level, population, "
            f"variant) rows, first {duplicates[0]}. The evaluation directory "
            "holds more than one run for these comparisons"
        )

    records: list[dict[str, Any]] = []
    # Arms whose masks disagree are excluded and counted rather than averaged,
    # and affine levels that failed their fit are counted rather than dropped
    # in silence: PROTOCOL 4.3 requires a nonpositive-slope failure to be
    # reported. build_records returns both counts alongside the records.
    mask_mismatches: list[tuple] = []
    affine_absent: set[tuple] = set()
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
                # An affine level that failed its fit contributes no matched
                # arm. Counted below rather than vanishing.
                if level == "affine":
                    affine_absent.add((scene, context_frame_id, target_frame_id, path))
                continue
            # PROTOCOL 4.6 makes the tax a subset-matched quantity: the
            # estimated score, the matched ceiling, and the matched floor must
            # be the same records. The evaluation layer arranges that; this
            # verifies it from the persisted masks rather than trusting it,
            # because a difference of means over different populations wears
            # exactly the shape of a method effect.
            arms = (est, oracle_m, nowarp_m)
            if len({bytes(arm["sample_mask"]) for arm in arms}) != 1 or len(
                {(arm["n"], arm["n_intersect"]) for arm in arms}
            ) != 1:
                mask_mismatches.append(
                    (scene, context_frame_id, target_frame_id, path, level)
                )
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
                    # PROTOCOL 3.9 compares the two paths on the cells both
                    # scored. The evaluation layer computed those scores and
                    # stored them per row; the ordinary means are over each
                    # path's own matched set and are not comparable across
                    # paths.
                    "est_intersect": est[INTERSECT_OF[metric]],
                    "oracle_intersect": oracle_m[INTERSECT_OF[metric]],
                    "nowarp_intersect": nowarp_m[INTERSECT_OF[metric]],
                    "n_intersect": est["n_intersect"],
                    # PROTOCOL 4.6's representation reference ceiling, carried
                    # through so it can be reported rather than only used to
                    # form the selection differential.
                    "forced_oracle_raw": est.get("forced_oracle_raw", float("nan")),
                    "forced_estimated_raw": est.get("forced_estimated_raw", float("nan")),
                    "forced_oracle_centered": est.get(
                        "forced_oracle_centered", float("nan")),
                    "forced_estimated_centered": est.get(
                        "forced_estimated_centered", float("nan")),
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
                # A contrast is a difference between two arms, so a pair
                # contributes to it only when it has both. Leaving one arm
                # populated would let the two sides average over different
                # pairs and call the difference a localization effect.
                for left, right in (("boundary", "interior"), ("lowtex", "hightex")):
                    if not (math.isfinite(record[f"tax_{left}"])
                            and math.isfinite(record[f"tax_{right}"])):
                        record[f"tax_{left}"] = float("nan")
                        record[f"tax_{right}"] = float("nan")
                records.append(record)
    return records, {
        "mask_mismatched_arms": len(mask_mismatches),
        "mask_mismatch_examples": [list(m) for m in mask_mismatches[:5]],
        "affine_arms_absent": len(affine_absent),
    }


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
    "forced_oracle_raw", "forced_estimated_raw",
    "forced_oracle_centered", "forced_estimated_centered",
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
        # PROTOCOL 4.6: the full-population Phase 3 ceiling is the
        # representation reference ceiling. It is reported in its own
        # right, not only consumed to form the selection differential,
        # and it is never subtracted from a score on another population.
        "reference_ceiling_phase3": lambda m: m["oracle_full"],
        "matched_ceiling": lambda m: m["oracle_m"],
        "estimated_score": lambda m: m["est"],
        "matched_floor": lambda m: m["nowarp_m"],
        "depth_tax": lambda m: m["oracle_m"] - m["est"],
        "oracle_margin": lambda m: m["oracle_m"] - m["nowarp_m"],
        "estimated_margin": lambda m: m["est"] - m["nowarp_m"],
        "retained_fraction": retained,
        "selection_differential": lambda m: m["oracle_full"] - m["oracle_m"],
        "transported_fraction": lambda m: m["transported_fraction"],
        # The localization contrasts are differences between two arms, so the
        # arms must be the same pairs. scene_aggregates would otherwise average
        # each arm over whatever pairs happened to populate it and present the
        # difference as a contrast. build_records nulls both arms of a split
        # whenever either is missing, so the means below are over one
        # population by construction, and each arm's support is reported.
        "boundary_minus_interior_tax": lambda m: m["tax_boundary"] - m["tax_interior"],
        "lowtex_minus_hightex_tax": lambda m: m["tax_lowtex"] - m["tax_hightex"],
        # PROTOCOL 4.10 Figure 2 is the forced-collision-order identity
        # check, so the forced scores are reported quantities rather than
        # gate internals. Nonfinite outside the rotation regime.
        "forced_oracle_raw": lambda m: m["forced_oracle_raw"],
        "forced_estimated_raw": lambda m: m["forced_estimated_raw"],
        "forced_oracle_centered": lambda m: m["forced_oracle_centered"],
        "forced_estimated_centered": lambda m: m["forced_estimated_centered"],
        "forced_identity_gap_raw": lambda m: (
            m["forced_oracle_raw"] - m["forced_estimated_raw"]),
        "forced_identity_gap_centered": lambda m: (
            m["forced_oracle_centered"] - m["forced_estimated_centered"]),
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


def ladder_table(
    records: Sequence[dict],
    analysis: AnalysisConfig,
    affine_absent: int = 0,
) -> list[dict[str, Any]]:
    """PROTOCOL 4.10 Table 1: per regime and pooled, per metric, path, level.

    The affine sensitivity row carries its own accounting. PROTOCOL 4.3 marks
    an image whose fitted scale is nonpositive as a failed affine row and
    requires the failure to be reported rather than silently skipped, so every
    affine row states how many pairs were attempted, how many contributed, and
    how many failed their fit.
    """
    formulas = quantity_formulas(analysis)
    table: list[dict[str, Any]] = []
    scopes = [("pooled", None), ("rotation", "rotation"),
              ("translation", "translation"), ("orbit", "orbit")]
    attempted = {
        key: len({r["camera_pair"] for r in cell})
        for key, cell in group_by(
            [r for r in records if r["level"] == "none"], ("metric", "path")
        ).items()
    }
    for scope_name, regime in scopes:
        scoped = [r for r in records if regime is None or r["regime"] == regime]
        for key, cell in sorted(group_by(scoped, ("metric", "path", "level")).items()):
            metric, path, level = key
            summary = cell_summary(cell, analysis, formulas, with_ci=LADDER_CI)
            row = {
                "analysis": scope_name, "metric": metric, "path": path,
                "level": level, **summary,
            }
            if level == "affine":
                contributed = len({r["camera_pair"] for r in cell})
                total = attempted.get((metric, path), contributed)
                row["affine_pairs_attempted"] = total
                row["affine_pairs_contributed"] = contributed
                row["affine_pairs_failed"] = max(0, total - contributed)
                row["affine_failure_reason"] = (
                    "fitted scale nonpositive or calibration population too small"
                )
            table.append(row)
    if affine_absent:
        for row in table:
            if row["level"] == "affine":
                row["affine_arms_absent_run_total"] = affine_absent
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


def reporting_cell(record: dict[str, Any]) -> tuple[str, str] | None:
    """The reporting cell a record belongs to, per PROTOCOL 3.3.

    Each primary curve takes only the regime that holds the other axis at
    zero; orbit appears in the joint view alone.
    """
    regime = record["regime"]
    for axis, primary in PRIMARY_REGIME.items():
        if regime == primary:
            return regime, record[axis]
    if regime == JOINT_REGIME:
        return regime, f"{record['rotation_bin']} x {record['parallax_bin']}"
    return None


def cross_path_disclosure(
    records: Sequence[dict[str, Any]], analysis: AnalysisConfig
) -> list[dict[str, Any]]:
    """Score-space effects at the scale of the operator gate, both paths shown.

    PROTOCOL 3.9 compares the two paths on the cells both scored, so every
    term here is read from the intersection columns the evaluator computed on
    the cross-path common-valid set, and a pair enters only when both paths
    scored it. Comparing each path's ordinary matched mean would difference
    two different populations and call the result an operator effect.

    The interval on the difference comes from a paired scene bootstrap: one
    draw of scenes serves both paths and dM is recomputed inside the
    replicate, so it carries the covariance the pairing gives it rather than
    the inflation of two independently bootstrapped intervals.

    The band is the frozen operator tolerance and applies to score-space
    quantities only; dimensionless ratios are governed by epsilon_margin.
    """
    band = analysis.path_agreement_tolerance
    quantities = {
        "depth_tax": ("oracle_intersect", "est_intersect"),
        "estimated_margin": ("est_intersect", "nowarp_intersect"),
    }
    paired: dict[tuple, dict[str, dict]] = {}
    for record in records:
        key = (record["scene"], record["camera_pair"], record["level"], record["metric"])
        paired.setdefault(key, {})[record["path"]] = record

    cells: dict[tuple, list[dict[str, Any]]] = {}
    for (scene, camera_pair, level, metric), paths in paired.items():
        if PER_POINT not in paths or SPLAT_POOL not in paths:
            continue
        per_point, splat = paths[PER_POINT], paths[SPLAT_POOL]
        if per_point["n_intersect"] != splat["n_intersect"]:
            continue
        cell = reporting_cell(per_point)
        if cell is None:
            continue
        values: dict[str, float] = {}
        complete = True
        for name, (high, low) in quantities.items():
            for row, suffix in ((per_point, "pp"), (splat, "sp")):
                a, b = row[high], row[low]
                if not (math.isfinite(a) and math.isfinite(b)):
                    complete = False
                    break
                values[f"{name}_{suffix}"] = a - b
            if not complete:
                break
        if not complete:
            continue
        cells.setdefault((level, metric, *cell), []).append(
            {"scene": scene, "camera_pair": camera_pair, **values}
        )

    out: list[dict[str, Any]] = []
    for key, members in sorted(cells.items(), key=repr):
        level, metric, analysis_name, label = key
        counts = {
            "n_scenes": len({m["scene"] for m in members}),
            "n_camera_pairs": len({m["camera_pair"] for m in members}),
            "n_feature_comparisons": 0,
        }
        if not is_supported(counts, analysis):
            continue
        by_scene: dict[str, list[dict[str, float]]] = {}
        for member in members:
            by_scene.setdefault(member["scene"], []).append(member)
        scenes = sorted(by_scene)
        for quantity in quantities:
            pp = {s: np.array([m[f"{quantity}_pp"] for m in by_scene[s]]) for s in scenes}
            sp = {s: np.array([m[f"{quantity}_sp"] for m in by_scene[s]]) for s in scenes}
            m_pp = float(np.concatenate([pp[s] for s in scenes]).mean())
            m_sp = float(np.concatenate([sp[s] for s in scenes]).mean())
            rng = np.random.default_rng(analysis.bootstrap_seed)
            draws = {"pp": [], "sp": [], "dM": []}
            for _ in range(analysis.bootstrap_resamples):
                picked = rng.integers(0, len(scenes), size=len(scenes))
                rep_pp = np.concatenate([pp[scenes[i]] for i in picked])
                rep_sp = np.concatenate([sp[scenes[i]] for i in picked])
                if not rep_pp.size:
                    continue
                a, b = float(rep_pp.mean()), float(rep_sp.mean())
                draws["pp"].append(a)
                draws["sp"].append(b)
                draws["dM"].append(a - b)
            tail = (1.0 - analysis.bootstrap_confidence) / 2.0
            interval = {
                name: (
                    (float(np.quantile(values, tail)),
                     float(np.quantile(values, 1.0 - tail)))
                    if values else (float("nan"), float("nan"))
                )
                for name, values in draws.items()
            }
            in_band = [abs(m_pp) <= band, abs(m_sp) <= band]
            if not any(in_band):
                continue
            lo_pp, hi_pp = interval["pp"]
            lo_sp, hi_sp = interval["sp"]
            excludes = lo_pp * hi_pp > 0 and lo_sp * hi_sp > 0
            same_sign = (m_pp > 0) == (m_sp > 0)
            if same_sign and excludes and all(in_band):
                case = "both_in_band"
                sentence = "a small, sign-consistent effect under both evaluation paths"
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
                "analysis": analysis_name, "bin": label, "metric": metric,
                "level": level, "quantity": quantity,
                "M_pp": m_pp, "M_sp": m_sp, "dM": m_pp - m_sp,
                "M_pp_ci": list(interval["pp"]), "M_sp_ci": list(interval["sp"]),
                "dM_ci": list(interval["dM"]), "replicates": len(draws["dM"]),
                "band": band, "case": case, "sentence": sentence,
                **counts,
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
    """Figure 2: the pure-rotation identity check under forced collision order.

    PROTOCOL 4.10 asks for the identity check with all variants overlapping,
    and the check is defined under forced collision ordering: only with the
    winner map fixed is agreement a true invariant, because discrete
    collisions can legitimately flip which sample wins a cell. Plotting the
    ordinary matched scores instead would show a control the protocol did not
    ask for. The unforced collision-ordering tax is shown beside it, small and
    separate, and is a splat-path diagnostic with no per-point counterpart.
    """
    plt = _pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.4))

    for panel, metric, label in (
        (axes[0], "raw", "raw cosine"), (axes[1], "centered", "centered cosine"),
    ):
        oracle_key = f"forced_oracle_{metric}"
        est_key = f"forced_estimated_{metric}"
        labels, oracle_values, est_values, gaps = [], [], [], []
        for row in ladder_rows:
            if (row["analysis"], row["path"]) != ("rotation", SPLAT_POOL):
                continue
            if row["metric"] != (RAW if metric == "raw" else CENTERED):
                continue
            if not math.isfinite(row.get(oracle_key, float("nan"))):
                continue
            labels.append(row["level"])
            oracle_values.append(row[oracle_key])
            est_values.append(row[est_key])
            gaps.append(row[f"forced_identity_gap_{metric}"])
        positions = range(len(labels))
        panel.plot(positions, oracle_values, "o", markersize=11, fillstyle="none",
                   label="matched Oracle-Transport, forced order")
        panel.plot(positions, est_values, "x", markersize=8,
                   label="estimated depth, forced order")
        panel.set_xticks(list(positions))
        panel.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        panel.set_title(
            f"forced collision order, {label}: the variants must overlap", fontsize=9
        )
        panel.set_ylabel(label, fontsize=9)
        panel.grid(alpha=0.3)
        panel.legend(fontsize=7)
        for position, gap in zip(positions, gaps):
            panel.annotate(f"{gap:+.1e}", (position, oracle_values[position]),
                           textcoords="offset points", xytext=(0, 12),
                           ha="center", fontsize=6, color="0.35")

    panel = axes[2]
    levels = list(gate_summary["collision_tax_raw_by_level"])
    width = 0.38
    offsets = [p - width / 2 for p in range(len(levels))]
    panel.bar(offsets, [gate_summary["collision_tax_raw_by_level"][l] for l in levels],
              width=width, label="raw")
    panel.bar([p + width for p in offsets],
              [gate_summary["collision_tax_centered_by_level"].get(l, float("nan"))
               for l in levels], width=width, label="centered")
    panel.set_xticks(range(len(levels)))
    panel.set_xticklabels(levels, rotation=20, ha="right", fontsize=8)
    panel.axhline(0.0, color="black", linewidth=1)
    panel.set_title("unforced collision-ordering tax, splat path only", fontsize=9)
    panel.set_ylabel("forced score minus ordinary score", fontsize=8)
    panel.grid(alpha=0.3)
    panel.legend(fontsize=7)

    figure.suptitle(
        "Figure 2: pure-rotation correctness control under forced collision order. "
        "Annotated values are the forced-order identity gap, which the 4.5 gate bounds.",
        fontsize=10,
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
        labels, values, low, high, unsupported = [], [], [], [], []
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
            # Support is the contrast's own, not the level's overall support:
            # a pair contributes to a contrast only when it carries both arms,
            # so the arms can be far thinner than the matched population and a
            # cell can pass overall while the contrast rests on almost nothing.
            if not row["supported"] or row[f"{quantity}_ci_replicates"] == 0:
                unsupported.append(len(labels) - 1)
        _band(panel, unsupported)
        panel.errorbar(range(len(labels)), values, yerr=[low, high], fmt="o", capsize=3)
        panel.set_xticks(range(len(labels)))
        panel.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
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
    records, exclusions = build_records(rows, analysis)
    print(f"read {len(rows)} rows, {len(records)} paired records")
    if exclusions["mask_mismatched_arms"]:
        raise SystemExit(
            f"{exclusions['mask_mismatched_arms']} matched arms carry "
            "different sample masks, so their differences would be "
            "differences of populations. PROTOCOL 4.6 makes every tax "
            f"subset matched; first {exclusions['mask_mismatch_examples'][:1]}"
        )
    if exclusions["affine_arms_absent"]:
        print(f"affine arms absent (failed fits): "
              f"{exclusions['affine_arms_absent']}")

    ladder = ladder_table(records, analysis, exclusions["affine_arms_absent"])
    bins = bin_table(records, analysis)
    disclosure = cross_path_disclosure(records, analysis)
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
