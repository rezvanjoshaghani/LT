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
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .analysis_config import DEFAULT_CONFIG_PATH, AnalysisConfig, load_analysis_config
from .datasets import bin_order, parallax_bin, rotation_bin
from .evaluate import (
    EVAL_VERSION,
    RUN_METADATA_KEY,
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


def read_eval_dir(eval_dir: Path, config: AnalysisConfig | None = None) -> list[dict[str, Any]]:
    """Read every per-scene parquet in a directory, checking they are one run.

    Globbing a directory and concatenating whatever is there treats the file
    system as the population definition. That is wrong in three ways at once,
    and every one of them produces a table that looks finished.

    A parquet from an earlier version is not comparable with a current one: the
    repair changed sample identity, so every hash-derived null draws
    differently, and scoring moved to the path's common valid set. Rows from
    either side of that would be averaged together.

    A missing scene is invisible. Each sidecar records the full scene list of
    the run that wrote it, so any one file declares what a complete directory
    should hold, and a failed array task shows up as an absent scene rather
    than as a smaller population nobody counted.

    Two runs at different commits, or with a different seed, sampling depth, or
    encoder set, differ in what they measured. They are refused rather than
    concatenated.
    """
    import pyarrow.parquet as pq

    eval_dir = Path(eval_dir)
    files = sorted(eval_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {eval_dir}")

    # Provenance comes from inside the parquet, so CLAUDE.md's rule that every
    # figure be regenerable from outputs/eval/*.parquet alone still holds. The
    # sidecar is a convenience for reading a run record without pyarrow, and is
    # used only for a parquet written before the record moved inside.
    tables = {path.stem: pq.read_table(path) for path in files}
    sidecars: dict[str, dict[str, Any]] = {}
    for stem, table in tables.items():
        raw = (table.schema.metadata or {}).get(RUN_METADATA_KEY)
        if raw is not None:
            sidecars[stem] = json.loads(raw.decode("utf-8"))
            continue
        meta_path = eval_dir / f"{stem}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{eval_dir / (stem + '.parquet')} carries no run record and has "
                f"no {meta_path.name} beside it. Its provenance is unknown, so it "
                "cannot be shown to belong with the others"
            )
        sidecars[stem] = json.loads(meta_path.read_text(encoding="utf-8"))

    required = (
        "eval_version", "encoders", "seed", "max_pairs_per_stratum",
        "git_commit", "run_scenes", "analysis_config_digest", "cache_provenance",
    )
    for field in required:
        # Present, then equal. Comparing only what happens to be there makes a
        # field that no file carries agree with itself: every value is null, the
        # set has one element, and a directory written before the field existed
        # passes the check that was added to catch it.
        absent = sorted(scene for scene, meta in sidecars.items() if meta.get(field) is None)
        if absent:
            raise ValueError(
                f"{eval_dir}: {len(absent)} run records carry no {field}, first "
                f"{absent[0]}. They predate this analysis and cannot be shown to "
                "describe the same measurement"
            )
        found = {json.dumps(meta.get(field), sort_keys=True) for meta in sidecars.values()}
        if field == "cache_provenance":
            continue  # per-scene by construction; the weights are checked below
        if len(found) > 1:
            raise ValueError(
                f"{eval_dir} mixes runs: {field} takes {len(found)} values "
                f"{sorted(found)[:3]}. These scenes did not measure the same thing"
            )

    any_meta = next(iter(sidecars.values()))
    if any_meta.get("eval_version") != EVAL_VERSION:
        raise ValueError(
            f"{eval_dir} was written by eval_version {any_meta.get('eval_version')}, "
            f"this analysis expects {EVAL_VERSION}. Re-run the evaluation"
        )
    expected = set(any_meta.get("run_scenes") or ())
    if not expected:
        raise ValueError(f"{eval_dir} sidecars declare no run_scenes")
    present = set(sidecars)
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise ValueError(
            f"{eval_dir} holds {len(present)} of the {len(expected)} scenes the "
            f"run declares. Missing: {missing}. Unexpected: {extra}. An "
            "incomplete directory is a different population, not a smaller one"
        )
    if str(any_meta.get("git_commit", "")).endswith("-dirty"):
        print(
            f"warning: results were produced at {any_meta['git_commit']}, from a "
            "worktree with uncommitted changes. The analysis config is bound by "
            f"content ({any_meta.get('analysis_config_digest')}), so an edit to it "
            "cannot hide here, but source edits are not covered"
        )
    # Provenance must cover the run before its contents can mean anything.
    # The required-fields check above asks that cache_provenance exists, and an
    # empty mapping exists: every loop below would then iterate nothing and
    # refuse nothing, so a sidecar with cache_provenance: {} sailed past the
    # whole pin apparatus. Each sidecar must name exactly the encoders its run
    # declares, no more, no fewer.
    for stem, meta in sidecars.items():
        declared = set(meta.get("encoders") or ())
        if not declared:
            raise ValueError(
                f"{eval_dir}: {stem} declares no encoders; an empty run cannot "
                "be shown to be the run the directory claims"
            )
        covered = set((meta.get("cache_provenance") or {}))
        if covered != declared:
            raise ValueError(
                f"{eval_dir}: {stem} carries cache provenance for "
                f"{sorted(covered) or 'no encoders'} but the run declares "
                f"{sorted(declared)}. Provenance that does not cover the run "
                "gates nothing"
            )
    # Every provenance entry must carry all four fields, correctly formed,
    # before anything compares identities. The identity tuple was assembled
    # with entry.get(...), so an entry missing weights_fingerprint compared as
    # None against None and matched across scenes: a whole field could be
    # absent, consistently, and pass. For DINOv2 the fingerprint is the only
    # evidence the unversioned checkpoint bytes did not change, so its absence
    # is precisely the case that must not slide.
    for stem, meta in sidecars.items():
        for encoder, entry in (meta.get("cache_provenance") or {}).items():
            for field in ("features_digest", "weights_fingerprint"):
                value = entry.get(field)
                if not (isinstance(value, str) and CONTENT_DIGEST.fullmatch(value)):
                    raise ValueError(
                        f"{eval_dir}: {stem} carries no well-formed {field} for "
                        f"{encoder} (got {value!r}; expected 32 hex characters). "
                        "Without it the cache these rows came from cannot be "
                        "identified, and for DINOv2 the fingerprint is the only "
                        "evidence the unversioned checkpoint bytes did not change"
                    )
            for field in ("weights_revision", "code_revision"):
                if field not in entry:
                    raise ValueError(
                        f"{eval_dir}: {stem} carries no {field} for {encoder}"
                    )
    # Every scene must have been evaluated from the same encoder, and an
    # encoder's identity is the whole triple: the weight bytes, the recorded
    # weights revision, and the inference implementation. Comparing the
    # fingerprint alone accepted two scenes with identical weights run through
    # different VGGT inference commits, which produce different features from
    # the same state dict; that is a mixture wearing one fingerprint.
    by_encoder: dict[str, set] = {}
    for meta in sidecars.values():
        for encoder, entry in (meta.get("cache_provenance") or {}).items():
            by_encoder.setdefault(encoder, set()).add(
                (
                    entry.get("weights_fingerprint"),
                    entry.get("weights_revision"),
                    entry.get("code_revision"),
                )
            )
    mixed = {encoder: sorted(map(str, v)) for encoder, v in by_encoder.items() if len(v) > 1}
    if mixed:
        raise ValueError(
            f"scenes were evaluated from different encoder identities: {mixed}. "
            "One encoder is one frozen representation: same weights, same "
            "revisions, same inference code. A cross-scene aggregate over two "
            "of these is a mixture"
        )
    # PROTOCOL locks the encoders, so unpinned provenance refuses the report
    # rather than annotating it. The rule is positive: a pin is a full commit
    # hash, or for weights alone the explicit "unpinnable: ..." declaration of
    # a loader that pinned everything it could. An earlier version listed the
    # bad strings instead, and a blocklist accepts everything it did not think
    # of: "main" passed it, and a branch name is a moving ref, which is the
    # opposite of a pin wearing the shape of one.
    not_pinned = sorted(
        {
            f"{encoder} ({field}={value!r})"
            for meta in sidecars.values()
            for encoder, entry in (meta.get("cache_provenance") or {}).items()
            for field, value in (
                ("weights_revision", entry.get("weights_revision")),
                ("code_revision", entry.get("code_revision")),
            )
            if not is_pinned_revision(field, value)
        }
    )
    if not_pinned:
        raise ValueError(
            "results were produced from encoders without a full pin: "
            + ", ".join(not_pinned)
            + ". PROTOCOL locks the encoders, so what a run used must be "
            "retrievable before it can be reported. A pin is a full commit "
            "hash; a branch or tag name is a moving ref and is not one. "
            "DINOv2's checkpoint bytes are declared unpinnable and that "
            "declaration is accepted; its hub ref and both VGGT revisions are "
            "pinnable and required. Resolve them with "
            "scripts/pin_encoder_revisions.py and re-cache"
        )

    # The config that reads a run must be the config that produced it, in every
    # value that decided what the rows contain. Loading an arbitrary config and
    # applying it bound nothing: a different co-visibility tolerance, sampling
    # cap, or manifest bound would produce a different report from the same
    # parquet with no complaint, and none of those are recoverable from the rows.
    #
    # Reporting values are deliberately not bound. PROTOCOL 3.4 has the support
    # thresholds set from realized counts after the run, so requiring the whole
    # config to match would forbid the documented workflow. What that edit
    # changes is recorded instead, and the report carries both digests.
    if config is not None:
        expected = any_meta.get("analysis_measurement_digest")
        if expected is None:
            raise ValueError(
                f"{eval_dir}: run records carry no analysis_measurement_digest, "
                "so the config that produced them cannot be identified"
            )
        if expected != config.measurement_digest():
            raise ValueError(
                f"{eval_dir} was evaluated under measurement config {expected}, "
                f"this analysis carries {config.measurement_digest()}. Those "
                "values decide what the rows contain, not how they are read; "
                "re-run the evaluation or analyse with the config it used"
            )
        if any_meta.get("analysis_reporting_digest") != config.reporting_digest():
            print(
                "note: reporting config differs from the run's "
                f"({any_meta.get('analysis_reporting_digest')} -> "
                f"{config.reporting_digest()}). Bin edges, support thresholds, "
                "bootstrap settings or gate tolerances have been edited since "
                "the evaluation, which PROTOCOL 3.4 permits from counts alone."
            )

    # The rows must be the population the metadata declares, checked before
    # anything aggregates them. Concatenating whatever the files held meant a
    # whole encoder could be absent, a file could carry another scene's rows,
    # and complete camera pairs could vanish, all without leaving the partial
    # comparisons the later completeness checks look for: a deleted pair is
    # not incomplete, it is gone.
    expected_variants = set(VARIANT_ORDER)
    rows: list[dict[str, Any]] = []
    for stem in sorted(tables):
        file_rows = tables[stem].to_pylist()
        meta = sidecars[stem]
        counters = {}
        for key in ("pairs_scored_both_paths", "pairs_scored_per_point_only",
                    "pairs_scored_splat_only"):
            if meta.get(key) is None:
                raise ValueError(
                    f"{eval_dir}: {stem} records no {key}, so its row population "
                    "cannot be reconciled against what the evaluator scored"
                )
            counters[key] = int(meta[key])
        scenes_present = {row["scene"] for row in file_rows}
        if scenes_present != {meta["scene"]} or stem != meta["scene"]:
            raise ValueError(
                f"{eval_dir}: {stem}.parquet holds rows for {sorted(scenes_present)} "
                f"while its record says scene {meta['scene']!r}; a file carrying "
                "another scene's rows is a mixed directory however its name reads"
            )
        declared_encoders = set(meta["encoders"])
        encoders_present = {row["encoder"] for row in file_rows}
        if encoders_present != declared_encoders:
            raise ValueError(
                f"{eval_dir}: {stem} declares encoders {sorted(declared_encoders)} "
                f"but its rows carry {sorted(encoders_present)}. An encoder with "
                "no rows disappeared without leaving a partial comparison behind"
            )
        # Per encoder and per path, the pair population must equal what the
        # evaluator counted. Per path, not combined: a combined count cannot
        # see a balanced error, where a lost per-point comparison and a
        # spurious splat one leave the sum where it was.
        for encoder in sorted(declared_encoders):
            paths_by_pair: dict[tuple, set] = {}
            variants: dict[tuple, set] = {}
            for row in file_rows:
                if row["encoder"] != encoder:
                    continue
                pair = (row["context_frame_id"], row["target_frame_id"])
                paths_by_pair.setdefault(pair, set()).add(row["path"])
                variants.setdefault((pair, row["path"]), set()).add(row["variant"])
            both = sum(1 for p in paths_by_pair.values() if len(p) == 2)
            pp_only = sum(1 for p in paths_by_pair.values() if p == {PER_POINT})
            sp_only = sum(1 for p in paths_by_pair.values() if p == {SPLAT_POOL})
            found = {
                "pairs_scored_both_paths": both,
                "pairs_scored_per_point_only": pp_only,
                "pairs_scored_splat_only": sp_only,
            }
            if found != counters:
                raise ValueError(
                    f"{eval_dir}: {stem} / {encoder}: the evaluator scored "
                    f"{counters} but the rows hold {found}. Camera pairs have "
                    "been added or removed since the run wrote them"
                )
            short = {
                key: sorted(missing := expected_variants - present)
                for key, present in variants.items()
                if present != expected_variants
            }
            if short:
                first = next(iter(short.items()))
                raise ValueError(
                    f"{eval_dir}: {stem} / {encoder}: {len(short)} comparisons "
                    f"do not carry all {len(expected_variants)} variants, first "
                    f"{first[0]} missing {first[1]}"
                )
        rows.extend(file_rows)
    return rows


FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
# blake2b with digest_size 16, the format every content digest in this
# repository uses: features, depth, the mean vector, the analysis config.
CONTENT_DIGEST = re.compile(r"^[0-9a-f]{32}$")


def is_pinned_revision(field: str, value: Any) -> bool:
    """Whether a recorded revision is actually a pin.

    A pin is a full 40-hex commit hash: immutable, so the thing it names can be
    retrieved. A branch or tag name resolves to something today and something
    else after a push, which is the opposite property. weights_revision may
    instead carry the "unpinnable: ..." declaration, made by a loader that
    pinned everything it could about an unversioned checkpoint URL.
    """
    if not isinstance(value, str):
        return False
    if FULL_COMMIT_SHA.fullmatch(value):
        return True
    return field == "weights_revision" and value.startswith("unpinnable:")


def neighbor_omitted_total(rows: Sequence[dict[str, Any]]) -> int:
    """Neighbor-Patch samples omitted across the run, from the rows.

    The quantity is per pair, and the column repeats it into every encoder,
    path, and variant row of that pair, so summing the column over-counts by
    their product. Deduplicating by pair recovers it, and it comes from the
    parquet rather than a sidecar so the analysis stays regenerable from the
    parquet alone.
    """
    per_pair: dict[tuple, int] = {}
    for row in rows:
        key = (row["scene"], row["context_frame_id"], row["target_frame_id"])
        per_pair[key] = int(row.get("neighbor_omitted", 0))
    return sum(per_pair.values())


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

    A duplicate row raises rather than counting, unlike a mask mismatch. A
    mismatch is a property of one comparison and the analysis can proceed
    without it; a repeated (comparison, variant) means the parquet directory
    holds two runs, or one run written twice, and every aggregate over it is
    then drawn from a population nobody chose. Assigning into the dictionary
    silently kept the last row and reported nothing.

    Returns (records, mask_mismatches).
    """
    grouped: dict[tuple, dict[str, dict[str, Any]]] = {}
    duplicates: list[tuple] = []
    for row in rows:
        key = tuple(row[k] for k in COMPARISON_KEYS)
        slot = grouped.setdefault(key, {})
        if row["variant"] in slot:
            duplicates.append((*key, row["variant"]))
        slot[row["variant"]] = row
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} duplicate (comparison, variant) rows, first "
            f"{duplicates[0]}. The evaluation directory holds more than one run "
            "for these comparisons; analysing it would silently keep whichever "
            "row was read last"
        )

    # PROTOCOL 3.7 gives every variant of a path the same records, so a
    # comparison carries all five or the evaluation did not do what it says.
    # Skipping the incomplete ones removed the population without counting it:
    # a method that failed on the hardest pairs, or a truncated parquet, would
    # take those pairs out of every aggregate and leave a table that reads as if
    # they had never been sampled.
    incomplete = [key for key, variants in grouped.items() if set(variants) != set(VARIANT_ORDER)]
    if incomplete:
        missing = sorted(set(VARIANT_ORDER) - set(grouped[incomplete[0]]))
        raise ValueError(
            f"{len(incomplete)} comparisons do not carry all {len(VARIANT_ORDER)} "
            f"variants, first {incomplete[0]} missing {missing}. PROTOCOL 3.7 "
            "scores every variant on the path's common valid set, so a partial "
            "comparison means the evaluation is incomplete, not that the "
            "analysis should proceed on what survived"
        )

    # Nonfinite is permitted for exactly one cell of the table. Mean-Feature has
    # no centered value by construction: the vector subtracted is the prediction
    # itself, so the centered prediction is the zero vector and its cosine is
    # undefined. Everywhere else a nonfinite is a failed method.
    nonfinite = [
        (key, variant)
        for key, variants in grouped.items()
        for variant, row in variants.items()
        if not _finite(row[metric]) and not (metric == CENTERED and variant == MEAN_FEATURE)
    ]
    if nonfinite:
        raise ValueError(
            f"{len(nonfinite)} nonfinite {metric} values outside centered "
            f"Mean-Feature, first {nonfinite[0]}. PROTOCOL 3.7 permits a "
            "nonfinite there and nowhere else"
        )

    records: list[dict[str, Any]] = []
    mismatches = 0
    for variants in grouped.values():
        floor = variants[NO_WARP_COPY]
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
                    "neighbor_omitted": row.get("neighbor_omitted", 0),
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


def comparison_weighted(records: Sequence[dict[str, Any]], field: str) -> float:
    """The same cell estimate with each record weighted by its comparison count.

    Reported as a diagnostic column, never as a headline number. The estimand is
    the unweighted mean above, where the camera pair is the unit. PROTOCOL 3.4
    fixes that unit twice over: support "depends primarily on independent camera
    pairs and scene coverage, not raw comparison counts", and the bootstrap is
    over scenes and pairs with points and patches excluded by name. Weighting a
    pair by how many correspondences survived in it would make the point the
    unit of the point estimate while the interval around it kept the pair, and
    the two would then describe different quantities.

    It is worth computing because the weighting is not neutral. A pair's
    comparison count is largely set by how much of the target the context still
    sees, so within a bin the weight rises with the easier geometry, and the
    weighted number is expected to sit above the unweighted one. The gap
    measures how much of any reported margin rides on that.
    """
    values = np.array([r[field] for r in records], dtype=np.float64)
    weights = np.array([r["n"] for r in records], dtype=np.float64)
    total = weights.sum()
    if values.size == 0 or total <= 0:
        return float("nan")
    return float(np.dot(values, weights) / total)


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
    # The replicate count travels with the interval. A cell whose statistic is
    # undefined in a replicate contributes nothing to that draw, and there is no
    # honest way to make it: the quantiles are then over the draws in which the
    # cell existed, which is a different and narrower distribution. That is
    # tolerable and has to be visible, because a quantile over three values and
    # a quantile over a thousand print identically. A cell backed by one scene
    # is the extreme case: every replicate that contains it gives exactly the
    # same estimate, so the interval has width zero and reads as certainty.
    tail = (1.0 - config.bootstrap_confidence) / 2.0
    return {
        key: (
            float(np.quantile(values, tail)),
            float(np.quantile(values, 1.0 - tail)),
            len(values),
        )
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
    nan = (float("nan"), float("nan"), 0)
    out: dict[tuple, dict[str, Any]] = {}
    for key, cell in grouped.items():
        counts = support_counts(cell)
        low, high, scene_draws = scene_ci.get(key, nan)
        pair_low, pair_high, pair_draws = pair_ci.get(key, nan)
        out[key] = {
            **counts,
            "estimate": statistic(cell),
            "ci_low": low,
            "ci_high": high,
            "ci_replicates": scene_draws,
            "pair_ci_low": pair_low,
            "pair_ci_high": pair_high,
            "pair_ci_replicates": pair_draws,
            "bootstrap_resamples": config.bootstrap_resamples,
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
    low, high, _ = intervals.get((), (float("nan"), float("nan"), 0))
    return low, high


def bootstrap_interval_with_count(
    records: Sequence[dict[str, Any]],
    statistic: Callable[[Sequence[dict[str, Any]]], float],
    config: AnalysisConfig,
    unit: str = "scene",
) -> tuple[float, float, int]:
    """The interval and how many replicates produced a value for it."""
    intervals = bootstrap_cells(records, (), statistic, config, unit=unit)
    return intervals.get((), (float("nan"), float("nan"), 0))


def cell_summary(
    records: Sequence[dict[str, Any]],
    config: AnalysisConfig,
    statistic: Callable[[Sequence[dict[str, Any]]], float] = mean_margin,
) -> dict[str, Any]:
    """Point estimate, both intervals, and the three support counts for one cell."""
    counts = support_counts(records)
    low, high, draws = bootstrap_interval_with_count(records, statistic, config, unit="scene")
    pair_low, pair_high, pair_draws = bootstrap_interval_with_count(
        records, statistic, config, unit="camera_pair"
    )
    return {
        **counts,
        "estimate": statistic(records),
        "ci_low": low,
        "ci_high": high,
        "ci_replicates": draws,
        "pair_ci_low": pair_low,
        "pair_ci_high": pair_high,
        "pair_ci_replicates": pair_draws,
        "bootstrap_resamples": config.bootstrap_resamples,
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


def matched_summaries(
    records: Sequence[dict[str, Any]], keys: Sequence[str], config: AnalysisConfig
) -> dict[tuple, dict[str, Any]]:
    """Summaries for the orbit-minus-translation difference, supported per arm.

    Support has to hold for each arm separately. Counting the pooled records
    would let two arms that each fail the threshold add up to a cell that
    passes, and a difference resting on an unsupported arm is unsupported no
    matter how many records the other arm brings.
    """
    summaries = summaries_for(
        records, keys, config, statistic=matched_orbit_minus_translation
    )
    grouped = group_by(records, keys)
    for key, summary in summaries.items():
        cell = grouped[key]
        arms = {
            regime: support_counts([r for r in cell if r["regime"] == regime])
            for regime in (JOINT_REGIME, PRIMARY_REGIME["parallax_bin"])
        }
        summary["supported"] = all(is_supported(counts, config) for counts in arms.values())
        summary["n_camera_pairs"] = min(counts["n_camera_pairs"] for counts in arms.values())
        summary["n_scenes"] = min(counts["n_scenes"] for counts in arms.values())
        summary["arm_support"] = {
            regime: counts["n_camera_pairs"] for regime, counts in arms.items()
        }
    return summaries


# ---------------------------------------------------------------------------
# PROTOCOL 3.9: the operational transport check
# ---------------------------------------------------------------------------

INTERSECT_METRICS = {
    RAW: "cosine_intersect_mean",
    CENTERED: "cosine_centered_intersect_mean",
}


def path_agreement(rows: Sequence[dict[str, Any]], config: AnalysisConfig) -> dict[str, Any]:
    """Compare the two paths on the cells both scored, per PROTOCOL 3.9.

    The evaluation layer scored each path on the cross-path intersection and
    emitted it as its own column, so this is a comparison of one population by
    two operators. Comparing the full-population scores instead would fold the
    coverage difference into the operator difference, and the coverage
    difference is exactly what is reported beside it rather than inside it.

    The gated quantity is the mean of the per-pair absolute differences. Two
    earlier choices were both wrong, in opposite directions. Gating the largest
    single pair applies a tolerance established on pooled scores to a statistic
    it was never measured against. Gating the absolute difference of the two
    pooled means fixes that but destroys the thing being measured: a pair where
    the per-point path reads high and a pair where it reads low cancel, so two
    pairs disagreeing by 0.2 in opposite directions produce an aggregate of
    zero and a passing gate. Taking absolute values per pair before aggregating
    keeps the comparison paired, keeps it an aggregate, and cannot cancel.

    Both metrics are gated. Centering subtracts a fixed vector from both sides,
    but it does not act equally on the two paths: the splat path's pooled output
    is a weighted mean over a cell and the per-point path's is a single sample,
    so a disagreement invisible in raw cosine can appear once the shared
    component is removed. A gate that checked only the raw metric would certify
    a centered table it never looked at, and the centered metric is the one the
    VGGT reading rests on.
    """
    by_key: dict[tuple, dict[str, dict[str, Any]]] = {}
    duplicates = 0
    for row in rows:
        if row["variant"] != ORACLE_TRANSPORT:
            continue
        key = (row["scene"], row["context_frame_id"], row["target_frame_id"], row["encoder"])
        slot = by_key.setdefault(key, {})
        if row["path"] in slot:
            duplicates += 1
        slot[row["path"]] = row

    per_metric: dict[str, list[float]] = {name: [] for name in INTERSECT_METRICS}
    coverage: list[int] = []
    complete = 0
    incomplete = 0
    for paths in by_key.values():
        if set(paths) != set(PATH_ORDER):
            # A pair one path could not score is not a pair the gate can compare,
            # but it is exactly the pair whose coverage differs most, so it is
            # counted and its coverage is reported rather than disappearing from
            # both the gate and the diagnostic beside it.
            incomplete += 1
            coverage.extend(int(row["coverage_difference"]) for row in paths.values())
            continue
        complete += 1
        for name, column in INTERSECT_METRICS.items():
            a = paths[PER_POINT][column]
            b = paths[SPLAT_POOL][column]
            if _finite(a) and _finite(b):
                per_metric[name].append(abs(a - b))
        coverage.append(
            int(paths[PER_POINT]["coverage_difference"])
            + int(paths[SPLAT_POOL]["coverage_difference"])
        )

    tolerance = config.path_agreement_tolerance
    result: dict[str, Any] = {
        "comparisons": complete,
        "single_path_pairs": incomplete,
        "duplicate_rows": duplicates,
        "tolerance": tolerance,
        "within_tolerance": complete > 0 and duplicates == 0,
    }
    for name, differences in per_metric.items():
        if not differences:
            result[f"{name}_mean_abs_difference"] = float("nan")
            result[f"{name}_max_abs_difference"] = float("nan")
            result[f"{name}_pairs_over_tolerance"] = 0
            result[f"{name}_n"] = 0
            result["within_tolerance"] = False
            continue
        values = np.asarray(differences, dtype=np.float64)
        mean_abs = float(values.mean())
        result[f"{name}_mean_abs_difference"] = mean_abs
        result[f"{name}_median_abs_difference"] = float(np.median(values))
        result[f"{name}_max_abs_difference"] = float(values.max())
        result[f"{name}_pairs_over_tolerance"] = int((values > tolerance).sum())
        result[f"{name}_n"] = int(values.size)
        if mean_abs > tolerance:
            result["within_tolerance"] = False
    if coverage:
        result["mean_coverage_difference_cells"] = float(np.mean(coverage))
        result["max_coverage_difference_cells"] = int(np.max(coverage))
    else:
        result["mean_coverage_difference_cells"] = float("nan")
        result["max_coverage_difference_cells"] = 0
    return result


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
    matched = matched_summaries(joint, ("encoder", "parallax_bin"), config)

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
        cells = group_by(scoped, keys)
        for key in values:
            encoder, metric, path, label, variant = key
            summary = values[key]
            margin = margins[key]
            cell = cells[key]
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
                    "value_pair_ci_low": summary["pair_ci_low"],
                    "value_pair_ci_high": summary["pair_ci_high"],
                    "value_ci_replicates": summary["ci_replicates"],
                    "value_pair_ci_replicates": summary["pair_ci_replicates"],
                    "margin": margin["estimate"],
                    "margin_ci_low": margin["ci_low"],
                    "margin_ci_high": margin["ci_high"],
                    "margin_pair_ci_low": margin["pair_ci_low"],
                    "margin_pair_ci_high": margin["pair_ci_high"],
                    # How many of the bootstrap_resamples draws produced a value
                    # for this cell. A quantile over three replicates and one
                    # over a thousand print identically without it.
                    "margin_ci_replicates": margin["ci_replicates"],
                    "margin_pair_ci_replicates": margin["pair_ci_replicates"],
                    "bootstrap_resamples": margin["bootstrap_resamples"],
                    # Diagnostics, not headline numbers. See comparison_weighted.
                    "value_comparison_weighted": comparison_weighted(cell, "value"),
                    "margin_comparison_weighted": comparison_weighted(cell, "margin"),
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
    matched = matched_summaries(joint, ("encoder", "metric", "path", "parallax_bin"), config)
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
                # The matched difference is NaN in any replicate whose resample
                # drops an arm, so of every statistic in this table it is the
                # one whose interval is most likely to rest on a subset of the
                # draws. Its counts are correspondingly the least optional.
                "margin_ci_replicates": summary["ci_replicates"],
                "margin_pair_ci_replicates": summary["pair_ci_replicates"],
                "bootstrap_resamples": summary["bootstrap_resamples"],
                "n_scenes": summary["n_scenes"],
                "n_camera_pairs": summary["n_camera_pairs"],
                "n_feature_comparisons": summary["n_feature_comparisons"],
                "supported": summary["supported"],
            }
        )
    table.sort(key=lambda r: (r["analysis"], r["encoder"], r["metric"], r["path"],
                              r["bin_index"], r["variant"]))
    return table


def counts_table(
    records: Sequence[dict[str, Any]], config: AnalysisConfig
) -> list[dict[str, Any]]:
    """Support counts per reported cell, and nothing else.

    PROTOCOL 3.4 permits bin edges and support thresholds to be set from counts
    alone, never from outcome values, before the freeze locks them. Counts are
    design facts; margins are results. This view exists so that decision can be
    made without the margins in front of you, which is the difference between
    choosing a threshold and choosing which claims survive it.
    """
    out: list[dict[str, Any]] = []
    for axis, regime in PRIMARY_REGIME.items():
        edges = config.parallax_edges() if axis == "parallax_bin" else config.rotation_edges()
        scoped = [
            r
            for r in records
            if r["regime"] == regime
            and r["variant"] == ORACLE_TRANSPORT
            and r["metric"] == RAW
        ]
        for key, cell in group_by(scoped, ("encoder", "path", axis)).items():
            encoder, path, label = key
            counts = support_counts(cell)
            out.append(
                {
                    "analysis": regime,
                    "axis": axis,
                    "bin": label,
                    "bin_index": bin_order(edges).index(label),
                    "encoder": encoder,
                    "path": path,
                    **counts,
                    "supported_at_current_threshold": is_supported(counts, config),
                }
            )
    joint = [
        r
        for r in records
        if r["regime"] == JOINT_REGIME
        and r["variant"] == ORACLE_TRANSPORT
        and r["metric"] == RAW
    ]
    for key, cell in group_by(
        joint, ("encoder", "path", "rotation_bin", "parallax_bin")
    ).items():
        encoder, path, rotation_label, parallax_label = key
        counts = support_counts(cell)
        out.append(
            {
                "analysis": JOINT_REGIME,
                "axis": "rotation_bin x parallax_bin",
                "bin": f"{rotation_label} x {parallax_label}",
                "bin_index": bin_order(config.rotation_edges()).index(rotation_label),
                "encoder": encoder,
                "path": path,
                **counts,
                "supported_at_current_threshold": is_supported(counts, config),
            }
        )
    out.sort(
        key=lambda r: (r["analysis"], r["encoder"], r["path"], r["bin_index"], r["bin"])
    )
    return out


def format_counts(table: Sequence[dict[str, Any]], config: AnalysisConfig) -> str:
    lines = [
        "Support counts only. PROTOCOL 3.4 permits thresholds to be set from "
        "these and never from outcome values.",
        "Both paths are listed. A pair can be scorable on one path and not the "
        "other, so per-bin counts differ by path, and a threshold chosen from "
        "the per-point counts alone can leave splat cells unsupported unseen.",
        f"current thresholds: scenes >= {config.support_min_scenes}, "
        f"camera pairs >= {config.support_min_camera_pairs}",
        "",
        f"{'analysis':<12} {'encoder':<18} {'path':<11} {'bin':<22} {'scenes':>7} "
        f"{'pairs':>7} {'comparisons':>12}  supported",
    ]
    for row in table:
        lines.append(
            f"{row['analysis']:<12} {row['encoder']:<18} {row['path']:<11} "
            f"{row['bin']:<22} {row['n_scenes']:>7} {row['n_camera_pairs']:>7} "
            f"{row['n_feature_comparisons']:>12}  {row['supported_at_current_threshold']}"
        )
    return chr(10).join(lines)


def write_table(path: Path, table: Sequence[dict[str, Any]], replace: bool = False) -> None:
    """Write a table as parquet, atomically. Refuses to overwrite unless asked.

    replace is for the counts view alone, and it is not a loosening of the
    no-overwrite rule so much as a consequence of what that view is for. The
    runbook has the user read counts, edit the support thresholds, and read them
    again, so a write-once counts file would keep the
    supported_at_current_threshold column from the configuration that has just
    been replaced and contradict the verdict printed beside it. The counts
    themselves are a function of the parquet alone. Results are never replaced.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(f"{path} exists; delete it to regenerate.")
    if not table:
        raise ValueError("no table rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.partial")
    pq.write_table(pa.Table.from_pylist(list(table)), tmp)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the Experiment Zero figures and table from eval parquet."
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="print support counts and exit, with no outcome values, so a "
        "threshold can be chosen from design facts alone",
    )
    args = parser.parse_args(argv)
    config = load_analysis_config(args.analysis_config)
    out_dir = args.out_dir or Path(args.eval_dir).parent

    rows = assign_bins(read_eval_dir(args.eval_dir, config), config)
    records: list[dict[str, Any]] = []
    mismatches = 0
    for metric in (RAW, CENTERED):
        part, count = paired_records(rows, metric=metric)
        records.extend(part)
        mismatches += count
    print(f"read {len(rows)} rows, {len(records)} paired records")
    if mismatches:
        # Not a warning. Every variant of a path is scored on that path's common
        # valid set, so identical masks are an invariant of the evaluation, and a
        # mismatch means the run that produced this parquet did not hold it. Any
        # table built on top would be subtracting across populations.
        raise SystemExit(
            f"{mismatches} comparisons carry mismatched masks between a variant "
            "and its floor. PROTOCOL 3.7 makes every difference paired, so this "
            "parquet cannot be reported on; re-run the evaluation."
        )

    if args.counts_only:
        table = counts_table(records, config)
        print(format_counts(table, config))
        # Rewritten every time, unlike the results table. The runbook has the
        # user read counts, edit the support thresholds, and read them again,
        # so a write-once artifact would keep the supported_at_current_threshold
        # column from the configuration that has just been replaced, and the
        # file would contradict the verdict printed beside it. The counts
        # themselves are a function of the parquet alone; only the threshold
        # verdict moves, and re-deriving it is the point.
        counts_path = Path(out_dir) / "tables" / "support_counts.parquet"
        write_table(counts_path, table, replace=True)
        print(f"{chr(10)}counts -> {counts_path}")
        return

    agreement = path_agreement(rows, config)
    print(
        f"PROTOCOL 3.9 path agreement on the cross-path intersection, over "
        f"{agreement['comparisons']} comparisons, tolerance "
        f"{agreement['tolerance']}:"
    )
    for name in INTERSECT_METRICS:
        print(
            f"  gated, {name}: mean per-pair |difference| = "
            f"{agreement[f'{name}_mean_abs_difference']:.5f} over "
            f"{agreement[f'{name}_n']} pairs"
        )
        print(
            f"    diagnostic: median "
            f"{agreement.get(f'{name}_median_abs_difference', float('nan')):.5f}, max "
            f"{agreement[f'{name}_max_abs_difference']:.5f}, "
            f"{agreement[f'{name}_pairs_over_tolerance']} pairs above the tolerance"
        )
    print(
        f"  coverage difference beside it: mean "
        f"{agreement['mean_coverage_difference_cells']:.2f} cells, "
        f"max {agreement['max_coverage_difference_cells']}"
    )
    if agreement["duplicate_rows"]:
        raise SystemExit(
            f"{agreement['duplicate_rows']} duplicate path rows in the evaluation "
            "parquet. One comparison scored twice means the directory mixes runs, "
            "and the second silently replaces the first."
        )
    if not agreement["within_tolerance"]:
        # PROTOCOL 3.9 states the two paths must agree within the tolerance. A
        # run that fails it has a transport operator that does not reach the
        # representational ceiling, which changes what every later rung means,
        # so it stops here rather than producing a table that reads as if it had.
        failed = [
            f"{name} at {agreement[f'{name}_mean_abs_difference']:.5f}"
            for name in INTERSECT_METRICS
            if not (agreement[f"{name}_mean_abs_difference"] <= agreement["tolerance"])
        ]
        raise SystemExit(
            "PROTOCOL 3.9 path agreement failed: mean per-pair absolute "
            f"difference exceeds the {agreement['tolerance']} tolerance for "
            f"{', '.join(failed) or 'no scored comparisons'}. Do not report "
            "these results; find the difference between the two paths first."
        )

    # PROTOCOL 3.7 reports per-variant omission counts beside Figure A in place
    # of differing n, since the common-valid rule gives every variant one n.
    #
    # Read from the run sidecars, where evaluation accumulates it once per pair.
    # Summing the row column instead counted each pair once per encoder and once
    # per path, and the divisor it was corrected by, the number of metrics, has
    # no relation to that duplication: rows are wide in metric and long in
    # encoder and path. With two encoders and two paths it reported twice the
    # true count.
    omissions = {NEIGHBOR_PATCH: neighbor_omitted_total(rows)}
    table_path = Path(out_dir) / "tables" / "experiment_zero.parquet"
    if table_path.exists():
        raise SystemExit(
            f"{table_path} exists; outputs are never overwritten. Delete it, or "
            "the run directory, to rebuild"
        )

    # Everything is built into a staging directory and moved into place only
    # once all of it succeeded. Writing the table first and the figures after
    # left a directory holding a table and no figures when a figure failed, and
    # the retry then stopped because the table was already there: a state that
    # was neither complete nor re-runnable.
    #
    # The staging directory is per-process, not a fixed shared name. A fixed one
    # is deleted on entry, so a second invocation would remove the first's
    # half-built figures out from under it.
    staging = Path(out_dir) / f".partial.{uuid.uuid4().hex}"
    if staging.exists():
        shutil.rmtree(staging)
    figures = staging / "figures"
    plan = [
        (
            "figure_a_null_ladder.png",
            lambda p: figure_a_null_ladder(records, p, config, omissions=omissions),
        ),
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
    failures: list[str] = []
    for name, build in plan:
        target = figures / name
        try:
            build(target)
        except ImportError as error:
            raise SystemExit(
                f"matplotlib is required to produce the four figures PROTOCOL "
                f"3.10 requires: {error}"
            )
        except ValueError as error:
            failures.append(f"{name}: {error}")
        else:
            pass
    if failures:
        # PROTOCOL 3.10 requires four figures. One that cannot be built is a
        # fact about the run that has to surface, not a line in a log above a
        # successful exit.
        shutil.rmtree(staging, ignore_errors=True)
        raise SystemExit(
            "required figures could not be produced: " + "; ".join(failures)
        )

    write_table(staging / "tables" / "experiment_zero.parquet", summary_table(records, config))

    # Publication order: refuse first, then move the figures, then the table.
    #
    # Every destination is checked before anything moves, so a name that already
    # exists stops the run while the output directory is still untouched rather
    # than after half the figures have been replaced. Outputs are never
    # overwritten, and a figure is an output.
    #
    # The table goes last because it is what a retry trips over. If a move fails
    # partway, the absent table lets the next run proceed; the reverse order
    # left a table standing over a partial figure set and no way to rebuild it.
    final_figures = Path(out_dir) / "figures"
    built = sorted(figures.glob("*.png"))
    clashes = [final_figures / p.name for p in built if (final_figures / p.name).exists()]
    if clashes:
        shutil.rmtree(staging, ignore_errors=True)
        raise SystemExit(
            "these figures already exist and are never overwritten: "
            + ", ".join(str(p) for p in clashes)
            + ". Delete them, or the run directory, to rebuild"
        )
    final_figures.mkdir(parents=True, exist_ok=True)
    for source in built:
        source.replace(final_figures / source.name)
        print(f"figure -> {final_figures / source.name}")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    (staging / "tables" / "experiment_zero.parquet").replace(table_path)
    print(f"table  -> {table_path}")
    shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
