"""Phase 5 tables: Stream AC, under reporting_rules.md sections 6 and 9.

The tables mode reads one evaluated run and writes its tables once. It is
licensed by the run's own provenance, never by receipts at the reporting
commit: lot.phase5_provenance.require_evaluated_run checks the run before a
row is read. The outcome and wording rules are not here. They live in
lot.phase5_outcomes, which decision 1 freezes at commit E with the estimand
layer, and this module only applies them.

Record preparation, every failure a stop:

1. Each scene's rows are read alone, through read_scene_rows, which hashes the
   file before and after the read. Rows are Python values, never pandas.
2. The rows must be the run record's: one row per (pair, seed, region), the
   record's scene, level, and fold, and as many pairs as its audit evaluated.
3. Seeds are collapsed to one record per pair and region. Predictor fields
   become the seed mean. Every other field must be identical across seeds,
   NaN-aware. The failure count is summed and kept per seed.
4. The region splits must sum to the whole support, every score on a counted
   population must be finite, and every pair must sit inside its regime's
   frozen definition.

Cells, reporting_rules.md section 6. One row per regime, plus a pooled row
labelled a summary. Rotation pairs are binned by the frozen rotation edges,
translation pairs by the frozen parallax edges, both closed on the right as
PROTOCOL 3.4 fixes. Orbit pairs appear in joint rotation by parallax cells
only, never marginalized. Every row carries its visibility bucket, which is
co-visible. A pair that cannot be binned stays in its regime and pooled rows
and is counted. If it holds a sample, that is an anomaly and a stop.

A cell's estimate is the unweighted mean over camera pairs, with the paired
scene bootstrap as its interval and the camera-pair bootstrap beside it, each
with its replicate count, and the comparison-weighted value as a diagnostic.
A region contrast is a cell too. Seed spread has its own columns and never
widens the interval.

A cell below the frozen support thresholds makes no claim, PROTOCOL 3.4. Its
counts, estimate, interval, and disclosure terms are shown. Its near-zero flag
is None and its wording is lot.phase5_outcomes.WORDING_BELOW_SUPPORT, in every
table and in phase5_near_zero.json.

The tables, each a parquet carrying its run record:

- phase5_primary: the headline, delta_learn_pp, with both methods, the
  No-Warp-Copy and Mean-Feature floors, both margins, the disclosure terms,
  seed spread, failures, the read deficit with its flags at the primary
  level, and the outcome columns of decision 2;
- phase5_formulation: the information/formulation diagnostic and its floors;
- phase5_splat_pool: the secondary operational comparison;
- phase5_cross_path: PROTOCOL 3.9's terms on the cells both paths share;
- phase5_landing_offset: the pre-registered diagnostic, primary level only,
  in the headline's cells;
- phase5_regions: boundary and interior, low and high texture, with the
  paired contrasts;
- phase5_per_seed: composite estimates per seed index;
- phase5_l2: every L2 companion, with no near-zero wording;
- phase5_adequacy and phase5_validation_history: from the training records,
  controls, and overfit verdict each run record embeds;
- phase5_references: the Phase 3 ceiling and the Phase 4 depth tax, read from
  the Phase 4 evaluation parquets, beside delta_learn_pp and the floors, three
  rungs kept apart;
- phase5_measured_outcome: decision 2's measured outcome, per regime.

Beside them: phase5_near_zero.json, every interpreted-effect cell's full
disclosure; phase5_accounting.json, what happened to every pair; and
MANIFEST.json, written last, with every file's sha256 and the run record. All
of it is built in a staging directory and published by one rename. An earlier
output is moved aside, never deleted. Every run record names the level's role,
primary, sensitivity, or diagnostic. A non-primary level is tabulated only
beside the primary level's verified tables, and only when its run shares the
primary run's commit E, digests, and gate receipts, reporting_rules.md
decision 4.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .analysis_config import AnalysisConfig
from .datasets import (
    assert_translation_parallax_floor,
    parallax_bin,
    parallax_bin_order,
    rotation_bin,
    rotation_bin_order,
)
from .phase5_check import sha256_file
from .phase5_estimands import (
    CL_TRANSPORT,
    CROSS_PATH,
    FORMULATION,
    INTERPRETED_EFFECTS,
    INTERSECTION_FIELDS,
    L2_METRICS,
    L2_QUANTITY_POPULATION,
    N_OFFSET_BINS,
    OFFSET_BINS,
    OFFSET_WHOLE,
    PER_POINT,
    PREDICT_WITH_DEPTH,
    PRIMARY_FIELDS,
    REGION_CONTRASTS,
    SPLAT_FIELDS,
    SPLAT_POOL,
    SPLAT_TRANSPORT,
    TL_REFERENCE,
    CellResult,
    HeadlineSubstitutionError,
    PathEstimate,
    SupportCounts,
    assert_not_target_lift_headline,
    comparison_weighted,
    contrast_field,
    disclosure_terms,
    evaluate_contrast,
    evaluate_quantity,
    l2_quantity_formulas,
    offset_count_field,
    offset_population,
    pivot_regions,
    population_count_field,
    population_fields,
    quantity_formulas,
    quantity_interval,
    quantity_population,
    seed_sensitivity,
    three_rung_decomposition,
)
from .phase5_modes import REGIONS, fold_seed_key
from .phase5_outcomes import (
    METRICS,
    POOLED_LABEL,
    POOLED_SCOPE,
    REGIME_SCOPES,
    SCOPES,
    control_label,
    gap_outcome,
    landing_offset_flags,
    measured_outcome,
    metric_sensitive,
    outcome_49,
    shown_near_zero,
    stable_validation_curve,
)
from .phase5_provenance import (
    ProvenanceError,
    output_run_record,
    read_scene_rows,
    require_evaluated_run,
    require_primary_chain,
    staged_output,
    write_parquet_with_record,
)

# The layout of the tables. A reader names the version it reads.
REPORT_VERSION = 1

TABLES_DIR = "tables"
PRIMARY_TABLE = "phase5_primary.parquet"
FORMULATION_TABLE = "phase5_formulation.parquet"
SPLAT_TABLE = "phase5_splat_pool.parquet"
CROSS_PATH_TABLE = "phase5_cross_path.parquet"
LANDING_OFFSET_TABLE = "phase5_landing_offset.parquet"
REGION_TABLE = "phase5_regions.parquet"
PER_SEED_TABLE = "phase5_per_seed.parquet"
L2_TABLE = "phase5_l2.parquet"
ADEQUACY_TABLE = "phase5_adequacy.parquet"
HISTORY_TABLE = "phase5_validation_history.parquet"
REFERENCE_TABLE = "phase5_references.parquet"
MEASURED_OUTCOME_TABLE = "phase5_measured_outcome.parquet"
# Every table the primary level writes, in writing order. A non-primary level
# writes all but the landing-offset table.
TABLE_FILES = (
    PRIMARY_TABLE, FORMULATION_TABLE, SPLAT_TABLE, CROSS_PATH_TABLE,
    LANDING_OFFSET_TABLE, REGION_TABLE, PER_SEED_TABLE, L2_TABLE, ADEQUACY_TABLE,
    HISTORY_TABLE, REFERENCE_TABLE, MEASURED_OUTCOME_TABLE,
)
NEAR_ZERO_FILE = "phase5_near_zero.json"
ACCOUNTING_FILE = "phase5_accounting.json"
MANIFEST_FILE = "MANIFEST.json"
TABLE_KIND = "phase5_table"

TABLE_LABELS = {
    PRIMARY_TABLE: "primary per-point learned-versus-explicit comparison, the headline table",
    FORMULATION_TABLE: (
        "information/formulation diagnostic, not the learned-versus-explicit estimand"
    ),
    SPLAT_TABLE: "secondary operational target-grid comparison",
    CROSS_PATH_TABLE: (
        "PROTOCOL 3.9 disclosure terms, on the cells both evaluation paths share"
    ),
    LANDING_OFFSET_TABLE: (
        "Context-Lift Oracle-Transport: model-free reference condition for the landing read"
    ),
    REGION_TABLE: (
        "error regions: ground-truth depth boundary and interior, low and high texture"
    ),
    PER_SEED_TABLE: "per-seed composite estimates over every test scene",
    L2_TABLE: "L2 companions on unit-normalized features; lower is closer",
    ADEQUACY_TABLE: "training adequacy",
    HISTORY_TABLE: "validation history of every training run",
    REFERENCE_TABLE: (
        "historical/reference estimators; distinct from the Phase 5 causal comparator"
    ),
    MEASURED_OUTCOME_TABLE: (
        "the measured outcome: one outcome per regime row and metric, plus the pooled "
        "summary"
    ),
}

# The headline's declared composition. Every build passes each method through
# the estimand layer's guard, so the Phase 4 target-lift score can never be
# built into the headline gap.
HEADLINE_METHODS = (CL_TRANSPORT, PREDICT_WITH_DEPTH)
HEADLINE_DEFINITION = (
    f"delta_learn_pp is {CL_TRANSPORT} minus {PREDICT_WITH_DEPTH}, on the primary "
    "per-point support V_P5_pp"
)
VISIBILITY_BUCKET = "co-visible"
MEAN_FEATURE_REPORTED = "raw cosine floor, PROTOCOL 3.7"
MEAN_FEATURE_NOT_APPLICABLE = "not applicable under centering, PROTOCOL 3.7"
READ_DEPTH_NOTE = (
    "read_deficit uses ground-truth context depth; delta_learn_pp uses aligned context depth"
)
LANDING_OFFSET_REPORTED = "reported beside the headline, at the primary level"
LANDING_OFFSET_PRIMARY_ONLY = (
    "the landing-offset diagnostic is reported once, from the primary level's records"
)
OFFSET_READING_NOTE = (
    "read the curve across bins for its shape only; the bins are not randomized, and "
    "offset can covary with image position and content"
)
OFFSET_BOUNDS_NOTE = (
    "patch units, closed on the right; the first bin starts at zero and the last ends "
    "at the cell corner"
)
PER_SEED_NOTE = (
    "seed s pools each fold's seed-s model over every test scene; the pairing of seed "
    "indices across folds is arbitrary"
)
CONTRAST_NOTE = (
    "paired over the camera pairs present in both regions; no near-zero wording applies"
)
BEST_VALIDATION_NOTE = "model-selection statistic, not an estimand"
CONTROLS_NOTE = "input-use controls are diagnostics, never thresholds"

# Field roles. The predictor's columns and its failure count depend on the
# seed. Every other column evaluate_scene writes is explicit and must not.
SEED_DEPENDENT_PREFIXES = ("predict_", "sp_predict_", "x_predict_", "x_sp_predict_")
SEED_DEPENDENT_COUNT = "n_predict_nonfinite"
PAIR_FIELDS = ("scene", "context_frame_id", "target_frame_id")
# Levels whose aligned context depth cannot fail. lot.phase4.aligned_depth
# returns no map only for a failed affine fit. Native depth passes through, and
# an invalid image scale raises rather than returning nothing.
LEVELS_WITHOUT_FAILED_ARMS = ("image", "none")
# Each region split of the whole support, from the registered contrasts.
REGION_SPLITS = tuple(REGION_CONTRASTS.values())
# Counts each region split partitions, and counts only the whole support holds.
SPLIT_COUNTS = ("n_primary", "n_splat", "n_intersect", SEED_DEPENDENT_COUNT)
WHOLE_ONLY_COUNTS = ("n_formulation",) + tuple(
    offset_count_field(label) for label in OFFSET_BINS + (OFFSET_WHOLE,)
)
# The populations every record's finiteness is checked on.
CHECKED_POPULATIONS = (PER_POINT, FORMULATION, SPLAT_POOL, CROSS_PATH) + tuple(
    offset_population(label) for label in OFFSET_BINS + (OFFSET_WHOLE,)
)
# A pair that cannot be binned must hold none of these.
UNBINNABLE_COUNTS = (
    "n_primary", "n_formulation", "n_splat", "n_intersect", offset_count_field(OFFSET_WHOLE),
)
IDENTITY_FIELDS = (
    "scene", "camera_pair", "context_frame_id", "target_frame_id", "regime",
    "rotation_deg", "parallax", "fold", "level", "region",
)
# What a per-seed record keeps: identity, the two per-pair populations it is
# read on, and their counts.
PER_SEED_FIELDS = IDENTITY_FIELDS + (
    "n_primary", "n_splat", SEED_DEPENDENT_COUNT,
) + PRIMARY_FIELDS + SPLAT_FIELDS
# What a region record keeps. Formulation and offset cells exist on the whole
# support only, so a region record carries none.
REGION_FIELDS = IDENTITY_FIELDS + (
    "n_primary", "n_splat", "n_intersect", SEED_DEPENDENT_COUNT,
) + PRIMARY_FIELDS + SPLAT_FIELDS + INTERSECTION_FIELDS

# The quantities of each table, in column order.
PRIMARY_QUANTITIES = (
    "cl_transport", "predict_with_depth", "no_warp_copy", "mean_feature",
    "cl_margin", "predict_margin", "delta_learn_pp",
)
PRIMARY_EFFECTS = ("cl_margin", "predict_margin", "delta_learn_pp")
PRIMARY_SEEDED = ("predict_with_depth", "predict_margin", "delta_learn_pp")
FORMULATION_QUANTITIES = (
    "tl_reference", "cl_on_formulation_support", "no_warp_copy_form", "mean_feature_form",
    "delta_formulation",
)
SPLAT_QUANTITIES = (
    "sp_transport", "sp_predict", "sp_no_warp_copy", "sp_mean_feature",
    "sp_transport_margin", "sp_predict_margin", "delta_learn_sp",
)
SPLAT_EFFECTS = ("sp_transport_margin", "sp_predict_margin", "delta_learn_sp")
SPLAT_SEEDED = ("sp_predict", "sp_predict_margin", "delta_learn_sp")
CROSS_PATH_QUANTITIES = (
    "x_delta_learn_pp", "x_delta_learn_sp", "path_difference_learn",
    "x_cl_margin", "x_sp_transport_margin", "path_difference_cl_margin",
    "x_predict_margin", "x_sp_predict_margin", "path_difference_predict_margin",
    "x_mean_feature", "x_sp_mean_feature",
)
OFFSET_QUANTITIES = ("cl_oracle_offset", "nowarp_offset", "cl_oracle_offset_margin",
                     "meanfeat_offset")
REGION_PATHS = (PER_POINT, SPLAT_POOL)
REGION_QUANTITIES = {
    PER_POINT: ("cl_transport", "predict_with_depth", "no_warp_copy", "mean_feature",
                "delta_learn_pp"),
    SPLAT_POOL: ("sp_transport", "sp_predict", "sp_no_warp_copy", "sp_mean_feature",
                 "delta_learn_sp"),
}
REGION_GAP = {PER_POINT: "delta_learn_pp", SPLAT_POOL: "delta_learn_sp"}
CONTRAST_QUANTITIES = {
    PER_POINT: ("cl_transport", "predict_with_depth", "delta_learn_pp"),
    SPLAT_POOL: ("sp_transport", "sp_predict", "delta_learn_sp"),
}
# The Phase 5 levels the reference table shows beside the Phase 3 and Phase 4
# rungs, with both floors, and the headline gap.
REFERENCE_QUANTITIES = ("cl_transport", "predict_with_depth", "no_warp_copy", "mean_feature",
                        "delta_learn_pp")
L2_POPULATIONS = (PER_POINT, SPLAT_POOL, FORMULATION)
L2_METRIC_ORDER = ("l2_centered", "l2_raw")
# The Mean-Feature floor's cells, each with a status column. PROTOCOL 3.7
# defines it under raw metrics only.
MEAN_FEATURE_NAMES = (
    "mean_feature", "mean_feature_form", "sp_mean_feature", "x_mean_feature",
    "x_sp_mean_feature", "meanfeat_offset", "l2_mean_feature", "l2_mean_feature_form",
    "l2_sp_mean_feature",
)
# Phase 4's names for the two cosine metrics, lot.phase4_report.
PHASE4_METRIC = {"centered": "cosine_centered_mean", "raw": "cosine_mean"}
# Collapsing seeds and pooling them agree up to float summation order. The
# mean of the per-seed estimates must meet the collapsed estimate this closely.
SEED_MEAN_TOLERANCE = 1e-9

SCOPE_AXIS = "all"
ROTATION_AXIS = "rotation_bin"
PARALLAX_AXIS = "parallax_bin"
JOINT_AXIS = "rotation_bin x parallax_bin"
_SCOPE_GROUP = {POOLED_SCOPE: 0, "rotation": 1, "translation": 2, "orbit": 3}
_BIN_GROUP = {"rotation": 4, "translation": 5, "orbit": 6}


class TablesStop(ValueError):
    """The evaluated records cannot be reported as they are. A stop, not a result."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _python(value: Any) -> Any:
    """A plain Python value. A numpy integer would fail the estimand layer's
    isinstance filters and silently empty a cell."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _same(a: Any, b: Any) -> bool:
    """Identical value and type, with two NaNs identical."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return _number(value) and math.isfinite(value)


def _positive(value: Any) -> bool:
    return _finite(value) and value > 0


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _float(value: Any) -> float:
    return float(value) if _number(value) else float("nan")


def _int(value: Any) -> int | None:
    return int(value) if _number(value) else None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _seed_tuple(seeds: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(_python(seed)) for seed in seeds)
    if not out or len(set(out)) != len(out):
        raise ValueError(f"the seeds must be distinct and at least one: {seeds}")
    return out


def is_seed_dependent(field: str) -> bool:
    """Whether a record field depends on the training seed."""
    return field == SEED_DEPENDENT_COUNT or field.startswith(SEED_DEPENDENT_PREFIXES)


def seed_count_field(seed: int) -> str:
    """One seed's failure count, as a collapsed record names it."""
    return f"{SEED_DEPENDENT_COUNT}_seed{seed}"


def camera_pair_key(row: Mapping[str, Any]) -> str:
    """The camera pair a record belongs to: scene, context frame, target frame."""
    return "|".join(str(row[field]) for field in PAIR_FIELDS)


def _region_order(region: Any) -> tuple[int, str]:
    return (REGIONS.index(region) if region in REGIONS else len(REGIONS), str(region))


# ---------------------------------------------------------------------------
# Record preparation
# ---------------------------------------------------------------------------

def check_scene_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    scene: str,
    level: str,
    fold: int,
    seeds: Sequence[int],
    audit: Mapping[str, Any],
) -> None:
    """One scene's rows are the rows its run record describes, or a stop.

    Every row names the scene, level, and fold of its run record, a seed of
    the run, a region of lot.phase5_modes.REGIONS, and a regime. Each pair
    holds exactly one row per seed and region. The audit's evaluated pairs
    are the pairs the rows hold, evaluated plus no_arm is the audit's pairs,
    and no_arm_pairs lists the no_arm pairs, none of them evaluated. At a
    level whose aligned depth cannot fail, no pair may lack an arm.
    """
    seeds = _seed_tuple(seeds)
    problems: list[str] = []

    def note(field: str, message: str) -> None:
        if len(problems) < 25:
            problems.append(f"{field}: {message}")

    seen: set[tuple] = set()
    slots: dict[tuple, set] = {}
    for index, row in enumerate(rows):
        for field, want in (("scene", scene), ("level", level), ("fold", fold)):
            have = _python(row.get(field))
            if have != want:
                note(field, f"row {index} names {field} {have!r}, not {want!r}")
        seed = _python(row.get("seed"))
        region = row.get("region")
        if seed not in seeds:
            note("seed", f"row {index} names seed {seed!r}, not one of {list(seeds)}")
        if region not in REGIONS:
            note("region", f"row {index} names region {region!r}, not one of {list(REGIONS)}")
        if row.get("regime") not in REGIME_SCOPES:
            note("regime", f"row {index} names regime {row.get('regime')!r}")
        pair = (row.get("context_frame_id"), row.get("target_frame_id"))
        key = pair + (seed, region)
        if key in seen:
            note("rows", f"a duplicate row for pair {pair}, seed {seed}, region {region}")
        seen.add(key)
        slots.setdefault(pair, set()).add((seed, region))
    expected = {(seed, region) for seed in seeds for region in REGIONS}
    for pair, held in slots.items():
        if held != expected:
            note("rows", f"pair {pair} holds {len(held & expected)} of its "
                         f"{len(expected)} seed and region rows")

    missing = [field for field in ("pairs", "evaluated", "no_arm", "no_arm_pairs")
               if field not in audit]
    if missing:
        note("audit", f"the run record's audit lacks {missing}")
    else:
        evaluated, no_arm, total = audit["evaluated"], audit["no_arm"], audit["pairs"]
        listed = audit["no_arm_pairs"] or []
        if evaluated != len(slots):
            note("evaluated", f"the audit counts {evaluated} evaluated pairs, and the rows "
                              f"hold {len(slots)}")
        if evaluated + no_arm != total:
            note("no_arm", f"evaluated {evaluated} plus no_arm {no_arm} is not the audit's "
                           f"{total} pairs")
        if len(listed) != no_arm:
            note("no_arm_pairs", f"lists {len(listed)} pairs for no_arm {no_arm}")
        both = {(entry[0], entry[1]) for entry in listed} & set(slots)
        if both:
            note("no_arm_pairs", f"pairs {sorted(both)[:3]} have no arm and rows both")
        if no_arm and level in LEVELS_WITHOUT_FAILED_ARMS:
            note("no_arm", f"at level {level} an arm cannot be absent, because aligned "
                           f"depth never fails there, so {no_arm} pairs with no arm is an "
                           "anomaly")
        if len(rows) != evaluated * len(seeds) * len(REGIONS):
            note("rows", f"{len(rows)} rows, not {evaluated} pairs by {len(seeds)} seeds by "
                         f"{len(REGIONS)} regions")
    if problems:
        raise TablesStop(
            f"{scene}: the evaluation rows do not match their run record:\n  - "
            + "\n  - ".join(problems)
        )


def _seed_mean(values: Sequence[Any], field: str, where: str) -> float:
    """A predictor field's mean over seeds: finite for every seed, or for none."""
    if not all(_number(value) for value in values):
        raise TablesStop(f"{where}: {field} holds {values}, not a number for every seed")
    finite = [math.isfinite(value) for value in values]
    if all(finite):
        return float(sum(float(value) for value in values) / len(values))
    if not any(finite) and all(math.isnan(value) for value in values):
        return float("nan")
    raise TablesStop(
        f"{where}: {field} is {values} across seeds. A predictor field is finite for "
        "every seed or for none, because the support it is read on does not depend on "
        "the seed"
    )


def collapse_seeds(rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> list[dict]:
    """One record per (pair, region), from that pair and region's seed rows.

    reporting_rules.md section 6. Predictor fields, the four prefixes of
    SEED_DEPENDENT_PREFIXES, are the mean over seeds. The failure count is
    summed, and each seed's count is kept as n_predict_nonfinite_seed{s}.
    Every other field must be identical across seeds, NaN-aware, or the
    collapse stops naming it. Exactly one row per seed is required: a missing
    seed, a duplicate row, or a seed outside seeds stops it. The seed field is
    dropped, and camera_pair is added. Every value is a plain Python value.
    Records come sorted by pair, then in the order of REGIONS.
    """
    seeds = _seed_tuple(seeds)
    groups: dict[tuple, dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(_python(row[field]) for field in PAIR_FIELDS) + (row["region"],)
        seed = _python(row["seed"])
        where = f"{key[0]} {key[1]} -> {key[2]} region {key[3]}"
        if seed not in seeds:
            raise TablesStop(f"{where}: seed {seed} is not one of the run's seeds {list(seeds)}")
        slot = groups.setdefault(key, {})
        if seed in slot:
            raise TablesStop(f"{where}: a duplicate row for seed {seed}")
        slot[seed] = row

    out: list[dict] = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], _region_order(k[3]))):
        slot = groups[key]
        where = f"{key[0]} {key[1]} -> {key[2]} region {key[3]}"
        missing = [seed for seed in seeds if seed not in slot]
        if missing:
            raise TablesStop(f"{where}: missing seed(s) {missing}")
        members = [slot[seed] for seed in seeds]
        names = list(members[0])
        for seed, member in zip(seeds[1:], members[1:]):
            if set(member) != set(names):
                differ = sorted(set(member) ^ set(names))
                raise TablesStop(f"{where}: seed {seed} carries different fields {differ}")
        record: dict[str, Any] = {}
        for name in names:
            if name == "seed":
                continue
            values = [_python(member[name]) for member in members]
            if name == SEED_DEPENDENT_COUNT:
                if not all(_count(value) for value in values):
                    raise TablesStop(f"{where}: {name} holds {values}, not counts")
                record[name] = int(sum(values))
                for seed, value in zip(seeds, values):
                    record[seed_count_field(seed)] = int(value)
            elif is_seed_dependent(name):
                record[name] = _seed_mean(values, name, where)
            else:
                first = values[0]
                for seed, value in zip(seeds[1:], values[1:]):
                    if not _same(first, value):
                        raise TablesStop(
                            f"{where}: the explicit field {name} is {first!r} for seed "
                            f"{seeds[0]} and {value!r} for seed {seed}. An explicit field "
                            "does not depend on the seed"
                        )
                record[name] = first
        record["camera_pair"] = camera_pair_key(record)
        out.append(record)
    return out


def select_seed(
    rows: Sequence[Mapping[str, Any]], seed: int, region: str = "all",
    fields: Sequence[str] = PER_SEED_FIELDS,
) -> list[dict]:
    """One seed's records of one region, kept to the fields a per-seed cell reads.

    Each record gains camera_pair, and every value is a plain Python value.
    Records come sorted by pair.
    """
    out = []
    for row in rows:
        if _python(row["seed"]) != seed or row["region"] != region:
            continue
        record = {field: _python(row[field]) for field in fields if field in row}
        record["camera_pair"] = camera_pair_key(record)
        out.append(record)
    return sorted(out, key=lambda record: record["camera_pair"])


def check_region_partition(collapsed: Sequence[Mapping[str, Any]]) -> None:
    """The region splits of every pair sum to its whole support, or a stop.

    Boundary plus interior, and low plus high texture, each equal the whole
    support in n_primary, n_splat, n_intersect, and every failure count. The
    formulation and offset counts exist on the whole support only, so they
    are zero on every region record.
    """
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in collapsed:
        by_pair.setdefault(record["camera_pair"], {})[record["region"]] = record
    problems: list[str] = []
    for pair, regions in by_pair.items():
        missing = [region for region in REGIONS if region not in regions]
        if missing:
            problems.append(f"{pair}: no record for region(s) {missing}")
            continue
        whole = regions["all"]
        counts = SPLIT_COUNTS + tuple(
            name for name in whole if name.startswith(f"{SEED_DEPENDENT_COUNT}_seed")
        )
        for count in counts:
            for left, right in REGION_SPLITS:
                values = (regions[left].get(count), regions[right].get(count), whole.get(count))
                if not all(_count(value) for value in values):
                    problems.append(f"{pair}: {count} is {values} on {left}, {right}, all")
                elif values[0] + values[1] != values[2]:
                    problems.append(
                        f"{pair}: {count} of {left} and {right}, {values[0]} + {values[1]}, "
                        f"is not the whole support's {values[2]}"
                    )
        for region in REGIONS:
            if region == "all":
                continue
            for count in WHOLE_ONLY_COUNTS:
                if regions[region].get(count) != 0:
                    problems.append(
                        f"{pair}: {count} is {regions[region].get(count)!r} on region "
                        f"{region}, where only the whole support holds it"
                    )
    if problems:
        raise TablesStop("the region records do not partition the whole support:\n  - "
                         + "\n  - ".join(problems[:25]))


def check_population_finiteness(collapsed: Sequence[Mapping[str, Any]]) -> None:
    """Every score on a counted population is finite, or a stop.

    On each record whose count on a population is above zero, every field of
    that population must be finite. A NaN there would silently unpair a
    difference, because the bootstrap skips a missing value field by field.
    Counts must be non-negative integers, no seed may fail on more samples
    than the support holds, and the offset bins must partition the whole
    diagnostic support.
    """
    problems: list[str] = []
    for record in collapsed:
        where = f"{record.get('camera_pair')} region {record.get('region')}"
        for population in CHECKED_POPULATIONS:
            count_field = population_count_field(population)
            if count_field not in record and record.get("region") != "all":
                continue
            count = record.get(count_field)
            if not _count(count):
                problems.append(f"{where}: {count_field} is {count!r}, not a count")
                continue
            if count > 0:
                for field in population_fields(population):
                    if not _finite(record.get(field)):
                        problems.append(
                            f"{where}: {field} is {record.get(field)!r} on a record whose "
                            f"{count_field} is {count}"
                        )
        n_primary = record.get("n_primary")
        for name, value in record.items():
            if name.startswith(f"{SEED_DEPENDENT_COUNT}_seed") and _count(n_primary):
                if not _count(value) or value > n_primary:
                    problems.append(f"{where}: {name} is {value!r}, more than n_primary "
                                    f"{n_primary} or not a count")
        bins = [record.get(offset_count_field(label)) for label in OFFSET_BINS]
        whole = record.get(offset_count_field(OFFSET_WHOLE))
        if all(_count(value) for value in bins) and _count(whole) and sum(bins) != whole:
            problems.append(f"{where}: the offset bins hold {sum(bins)} samples, and "
                            f"{offset_count_field(OFFSET_WHOLE)} is {whole}")
    if problems:
        raise TablesStop("a counted population holds a value that cannot be scored:\n  - "
                         + "\n  - ".join(problems[:25]))


def check_regime_geometry(records: Sequence[Mapping[str, Any]], analysis: AnalysisConfig) -> None:
    """Every pair sits inside its regime's frozen definition, or a stop.

    PROTOCOL 3.3 and 3.4. Rotation magnitude is finite and non-negative, and
    parallax, where defined, is non-negative. A rotation pair has parallax
    below zero_parallax_tol, so the rotation curve carries no parallax. A
    translation pair rotates less than zero_rotation_tol_deg, so the parallax
    curve carries no rotation, and its parallax honours the programme's design
    floor. Orbit pairs vary both and may sit anywhere.
    """
    problems: list[str] = []
    for record in records:
        where = f"{record.get('scene')} {record.get('camera_pair')}"
        regime = record.get("regime")
        rotation = record.get("rotation_deg")
        parallax = record.get("parallax")
        if not _finite(rotation) or rotation < 0:
            problems.append(f"{where}: rotation_deg {rotation!r} is not a finite, "
                            "non-negative angle")
        if _finite(parallax) and parallax < 0:
            problems.append(f"{where}: parallax {parallax!r} is negative")
        if regime == "rotation" and _finite(parallax) and parallax >= analysis.zero_parallax_tol:
            problems.append(
                f"{where}: a rotation pair has parallax {parallax:g}, at or above "
                f"zero_parallax_tol {analysis.zero_parallax_tol:g}"
            )
        if regime == "translation":
            if _finite(rotation) and rotation >= analysis.zero_rotation_tol_deg:
                problems.append(
                    f"{where}: a translation pair rotates {rotation:g} degrees, at or above "
                    f"zero_rotation_tol_deg {analysis.zero_rotation_tol_deg:g}"
                )
            try:
                assert_translation_parallax_floor(
                    regime, parallax if _finite(parallax) else float("nan"), analysis, where
                )
            except ValueError as error:
                problems.append(str(error))
    if problems:
        raise TablesStop("a pair lies outside its regime's frozen definition:\n  - "
                         + "\n  - ".join(problems[:25]))


@dataclasses.dataclass
class PreparedScene:
    """One scene's records after every check: collapsed, per seed, accounted."""

    scene: str
    collapsed: list[dict]
    per_seed: dict[int, list[dict]]
    accounting: dict[str, Any]


def _scene_accounting(scene: str, fold: int, rows: Sequence[Mapping[str, Any]],
                      whole: Sequence[Mapping[str, Any]], audit: Mapping[str, Any],
                      seeds: Sequence[int]) -> dict[str, Any]:
    pairs_by_regime = {regime: 0 for regime in REGIME_SCOPES}
    empty: dict[str, dict[str, int]] = {
        count: {regime: 0 for regime in REGIME_SCOPES} for count in UNBINNABLE_COUNTS
    }
    for record in whole:
        pairs_by_regime[record["regime"]] += 1
        for count in UNBINNABLE_COUNTS:
            if not _positive(record.get(count)):
                empty[count][record["regime"]] += 1
    failures = {f"seed{seed}": sum(int(r.get(seed_count_field(seed), 0)) for r in whole)
                for seed in seeds}
    failures["total"] = sum(failures.values())
    return {
        "scene": scene,
        "fold": fold,
        "pairs": int(audit["pairs"]),
        "evaluated": int(audit["evaluated"]),
        "no_arm": int(audit["no_arm"]),
        "no_arm_pairs": [[scene, *[str(item) for item in entry]]
                         for entry in (audit.get("no_arm_pairs") or [])],
        "worst_per_point_residual": _float(audit.get("worst_per_point_residual")),
        "worst_splat_residual": _float(audit.get("worst_splat_residual")),
        "rows": len(rows),
        "pairs_by_regime": pairs_by_regime,
        "empty": empty,
        "n_predict_nonfinite": failures,
    }


def prepare_scene(
    rows: Sequence[Mapping[str, Any]],
    *,
    scene: str,
    level: str,
    fold: int,
    seeds: Sequence[int],
    audit: Mapping[str, Any],
    analysis: AnalysisConfig,
) -> PreparedScene:
    """Check one scene's rows, collapse its seeds, and keep what the tables read."""
    seeds = _seed_tuple(seeds)
    check_scene_rows(rows, scene=scene, level=level, fold=fold, seeds=seeds, audit=audit)
    collapsed = collapse_seeds(rows, seeds)
    check_region_partition(collapsed)
    check_population_finiteness(collapsed)
    whole = [record for record in collapsed if record["region"] == "all"]
    check_regime_geometry(whole, analysis)
    per_seed = {seed: select_seed(rows, seed) for seed in seeds}
    kept = [
        record if record["region"] == "all"
        else {field: record[field] for field in REGION_FIELDS + tuple(
            name for name in record if name.startswith(f"{SEED_DEPENDENT_COUNT}_seed")
        ) if field in record}
        for record in collapsed
    ]
    return PreparedScene(scene, kept, per_seed,
                         _scene_accounting(scene, fold, rows, whole, audit, seeds))


@dataclasses.dataclass
class PreparedRun:
    """Every scene's prepared records, and what the tables need to read them."""

    level: str
    primary_level: str
    seeds: tuple[int, ...]
    offset_edges: tuple[float, ...]
    collapsed: list[dict]
    per_seed: dict[int, list[dict]]
    scenes: dict[str, dict[str, Any]]


def prepare_run(
    sources: Iterable[tuple[str, Callable[[], Sequence[Mapping[str, Any]]], Mapping[str, Any]]],
    analysis: AnalysisConfig,
    *,
    level: str,
    primary_level: str,
    seeds: Sequence[int],
    offset_edges: Sequence[float],
) -> PreparedRun:
    """Prepare every scene, reading one at a time.

    sources yields, per scene, its name, a function that reads its rows, and
    its run record, which names its fold and audit. Each scene is read,
    checked, and collapsed before the next is read, so the uncollapsed rows of
    only one scene are ever held. offset_edges are the landing-offset upper
    edges, lot.phase5.landing_offset_edges.
    """
    seeds = _seed_tuple(seeds)
    edges = tuple(float(edge) for edge in offset_edges)
    if len(edges) + 1 != N_OFFSET_BINS:
        raise ValueError(f"{len(edges)} offset edges make {len(edges) + 1} bins, not "
                         f"{N_OFFSET_BINS}")
    collapsed: list[dict] = []
    per_seed: dict[int, list[dict]] = {seed: [] for seed in seeds}
    scenes: dict[str, dict[str, Any]] = {}
    for scene, load, record in sources:
        if scene in scenes:
            raise TablesStop(f"scene {scene} is read twice")
        rows = load()
        prepared = prepare_scene(rows, scene=scene, level=level, fold=record["fold"],
                                 seeds=seeds, audit=record["audit"], analysis=analysis)
        del rows
        collapsed.extend(prepared.collapsed)
        for seed in seeds:
            per_seed[seed].extend(prepared.per_seed[seed])
        scenes[scene] = prepared.accounting
    return PreparedRun(level, primary_level, seeds, edges, collapsed, per_seed, scenes)


# ---------------------------------------------------------------------------
# Reporting cells
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CellKey:
    """One reporting cell: a regime or pooled scope, or one bin of a regime.

    analysis is pooled, rotation, translation, or orbit. axis is all for a
    scope row, rotation_bin for a rotation bin, parallax_bin for a translation
    bin, and the joint axis for an orbit cell. bin_index orders bins along the
    frozen edges, and group orders the kinds of row.
    """

    analysis: str
    axis: str
    bin: str
    bin_index: int
    group: int
    rotation_bin: str | None = None
    parallax_bin: str | None = None

    @property
    def summary(self) -> bool:
        return self.analysis == POOLED_SCOPE

    @property
    def order(self) -> tuple[int, int]:
        return (self.group, self.bin_index)

    def columns(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis,
            "axis": self.axis,
            "bin": self.bin,
            "bin_index": self.bin_index,
            "rotation_bin": self.rotation_bin,
            "parallax_bin": self.parallax_bin,
            "row_label": POOLED_LABEL if self.summary else self.analysis,
            "summary": self.summary,
            "visibility_bucket": VISIBILITY_BUCKET,
        }


def scope_key(scope: str) -> CellKey:
    """The row of one regime, or of the pooled summary."""
    return CellKey(scope, SCOPE_AXIS, "all", 0, _SCOPE_GROUP[scope])


def _label(binner: Callable[[float, AnalysisConfig], str], value: float,
           analysis: AnalysisConfig, where: str) -> str:
    try:
        return binner(value, analysis)
    except ValueError as error:
        raise TablesStop(f"{where}: {error}") from None


def _bin_key(record: Mapping[str, Any], analysis: AnalysisConfig, rotation_order: list[str],
             parallax_order: list[str]) -> tuple[CellKey | None, str | None]:
    """The bin a record belongs to, or None with the reason it has none."""
    regime = record["regime"]
    where = f"{record.get('scene')} {record.get('camera_pair')}"
    rotation, parallax = record.get("rotation_deg"), record.get("parallax")
    if regime in ("rotation", "orbit") and not _finite(rotation):
        return None, "rotation_deg is not finite"
    if regime in ("translation", "orbit") and not _finite(parallax):
        return None, "parallax is not finite"
    if regime == "rotation":
        label = _label(rotation_bin, rotation, analysis, where)
        return CellKey("rotation", ROTATION_AXIS, label, rotation_order.index(label),
                       _BIN_GROUP["rotation"], rotation_bin=label), None
    if regime == "translation":
        label = _label(parallax_bin, parallax, analysis, where)
        return CellKey("translation", PARALLAX_AXIS, label, parallax_order.index(label),
                       _BIN_GROUP["translation"], parallax_bin=label), None
    r_label = _label(rotation_bin, rotation, analysis, where)
    p_label = _label(parallax_bin, parallax, analysis, where)
    index = rotation_order.index(r_label) * len(parallax_order) + parallax_order.index(p_label)
    return CellKey("orbit", JOINT_AXIS, f"{r_label} x {p_label}", index, _BIN_GROUP["orbit"],
                   rotation_bin=r_label, parallax_bin=p_label), None


def _cell_members(records: Sequence[Mapping[str, Any]], analysis: AnalysisConfig
                  ) -> tuple[dict[CellKey, list], list[dict[str, Any]]]:
    rotation_order = rotation_bin_order(analysis)
    parallax_order = parallax_bin_order(analysis)
    members: dict[CellKey, list] = {}
    unbinnable: list[dict[str, Any]] = []
    for record in records:
        regime = record.get("regime")
        if regime not in REGIME_SCOPES:
            raise TablesStop(f"{record.get('camera_pair')}: unknown regime {regime!r}")
        keys = [scope_key(POOLED_SCOPE), scope_key(regime)]
        key, reason = _bin_key(record, analysis, rotation_order, parallax_order)
        if key is None:
            held = [count for count in UNBINNABLE_COUNTS if _positive(record.get(count))]
            if held:
                raise TablesStop(
                    f"{record.get('camera_pair')}: the {regime} pair cannot be binned, "
                    f"because its {reason}, yet it holds samples on {held}. A pair with "
                    "no co-visible point holds none, so this is an anomaly"
                )
            unbinnable.append({
                "scene": record.get("scene"),
                "camera_pair": record.get("camera_pair"),
                "regime": regime,
                "rotation_deg": record.get("rotation_deg")
                if _finite(record.get("rotation_deg")) else None,
                "parallax": record.get("parallax") if _finite(record.get("parallax")) else None,
                "reason": reason,
            })
        else:
            keys.append(key)
        for cell in keys:
            members.setdefault(cell, []).append(record)
    return members, unbinnable


def reporting_cells(
    records: Sequence[Mapping[str, Any]], analysis: AnalysisConfig
) -> tuple[list[tuple[CellKey, list]], dict[str, Any]]:
    """Every reporting cell that holds a record, with its records, in order.

    The pooled summary, then each regime's row, then the rotation bins, the
    translation bins, and the orbit joint cells, each along the frozen edges.
    A pair joins the pooled row, its regime's row, and the one bin its regime
    is binned on. A pair that cannot be binned joins the first two only and is
    listed under unbinnable in the accounting returned beside the cells. If it
    holds a sample, that stops the tables.
    """
    members, unbinnable = _cell_members(records, analysis)
    cells = sorted(members.items(), key=lambda item: item[0].order)
    return cells, {"unbinnable": unbinnable, "cells": len(cells)}


def scope_cells(records: Sequence[Mapping[str, Any]]) -> list[tuple[CellKey, list]]:
    """The pooled row and each regime's row that holds a record, in order."""
    members: dict[CellKey, list] = {}
    for record in records:
        regime = record.get("regime")
        if regime not in REGIME_SCOPES:
            raise TablesStop(f"{record.get('camera_pair')}: unknown regime {regime!r}")
        for key in (scope_key(POOLED_SCOPE), scope_key(regime)):
            members.setdefault(key, []).append(record)
    return sorted(members.items(), key=lambda item: item[0].order)


def offset_bounds(edges: Sequence[float]) -> dict[str, tuple[float, float]]:
    """Each landing-offset bin's lower and upper bound, in patch units.

    Closed on the right, as lot.phase5_score.offset_bins assigns them. The
    first bin starts at zero and the last ends at the cell corner, sqrt(2)/2.
    """
    corner = math.sqrt(2.0) / 2.0
    uppers = list(edges) + [corner]
    lowers = [0.0] + list(edges)
    bounds = {label: (float(lo), float(hi))
              for label, lo, hi in zip(OFFSET_BINS, lowers, uppers)}
    bounds[OFFSET_WHOLE] = (0.0, corner)
    return bounds


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

def _defined(quantity: str, metric: str) -> bool:
    forms = l2_quantity_formulas(metric) if metric in L2_METRICS else quantity_formulas(metric)
    return quantity in forms


def _scene_columns(name: str, estimate: Any, lo: Any, hi: Any, replicates: Any
                   ) -> dict[str, Any]:
    return {
        name: _float(estimate),
        f"{name}_ci_low": _float(lo),
        f"{name}_ci_high": _float(hi),
        f"{name}_ci_replicates": int(replicates or 0),
    }


def _pair_columns(name: str, interval: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{name}_pair_ci_low": _float(interval["lo"]),
        f"{name}_pair_ci_high": _float(interval["hi"]),
        f"{name}_pair_ci_replicates": int(interval["n_replicates"] or 0),
    }


def _support_columns(counts: SupportCounts, analysis: AnalysisConfig, prefix: str = ""
                     ) -> dict[str, Any]:
    return {
        f"{prefix}n_scenes": int(counts.n_scenes),
        f"{prefix}n_camera_pairs": int(counts.n_camera_pairs),
        f"{prefix}n_feature_comparisons": int(counts.n_feature_comparisons),
        f"{prefix}supported": bool(counts.is_supported(analysis)),
    }


def _cell_counts(cell: CellResult) -> SupportCounts:
    return SupportCounts(cell.n_scenes, cell.n_camera_pairs, cell.n_feature_comparisons)


def _n_folds(records: Sequence[Mapping[str, Any]], count_field: str) -> int:
    return len({record.get("fold") for record in records if _positive(record.get(count_field))})


def _rectangular(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every row with every column, missing ones None, one value type per column.

    pyarrow names a table's columns from its first row, so a column the first
    row lacks would be dropped in silence, and it widens an integer column to
    float beside a float. Both are refused here instead.
    """
    columns = list(dict.fromkeys(name for row in rows for name in row))
    out = [{name: row.get(name) for name in columns} for row in rows]
    for name in columns:
        kinds = {type(row[name]).__name__ for row in out if row[name] is not None}
        if len(kinds) > 1:
            raise TablesStop(f"column {name} mixes value types {sorted(kinds)}")
    return out


def _assert_no_target_lift(rows: Sequence[Mapping[str, Any]]) -> None:
    """Stream AD: nothing of the Phase 4 target-lift score enters the headline."""
    for row in rows:
        for name, value in row.items():
            if name.startswith("tl_") or (
                isinstance(value, str) and (TL_REFERENCE in value or "Target-Lift" in value)
            ):
                raise HeadlineSubstitutionError(
                    f"the headline table holds {name}={value!r}; the Phase 4 target-lift "
                    "score is never part of the learned-versus-explicit headline"
                )


def _require_once(records: Sequence[Mapping[str, Any]], keys: Sequence[str], what: str) -> None:
    """Each record's key fields name it once among records, or a stop."""
    seen: set[tuple] = set()
    repeated: list[tuple] = []
    for record in records:
        key = tuple(record.get(field) for field in keys)
        if key in seen:
            repeated.append(key)
        seen.add(key)
    if repeated:
        raise TablesStop(
            f"{what} must hold each {', '.join(keys)} once, and {len(repeated)} repeat, "
            f"first {repeated[0]}. Raw seed rows would count each pair once per seed"
        )


class _Intervals:
    """Every interval the tables read, computed once per records, quantity, and metric.

    group names the records a value is computed on, so a cell shared by two
    tables is bootstrapped once.
    """

    def __init__(self, analysis: AnalysisConfig):
        self.analysis = analysis
        self._memo: dict[tuple, Any] = {}

    def _get(self, key: tuple, compute: Callable[[], Any]) -> Any:
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    def cell(self, group: tuple, records: Sequence[dict], quantity: str, metric: str
             ) -> CellResult:
        return self._get(("cell", group, quantity, metric), lambda: evaluate_quantity(
            records, quantity, metric, self.analysis))

    def interval(self, group: tuple, records: Sequence[dict], quantity: str, metric: str,
                 unit: str) -> tuple[dict[str, Any], SupportCounts]:
        return self._get(("interval", group, quantity, metric, unit), lambda: quantity_interval(
            records, quantity, metric, self.analysis, unit))

    def terms(self, group: tuple, records: Sequence[dict], quantity: str, metric: str
              ) -> dict[str, Any]:
        return self._get(("terms", group, quantity, metric), lambda: disclosure_terms(
            records, quantity, metric, self.analysis))

    def weighted(self, group: tuple, records: Sequence[dict], quantity: str, metric: str
                 ) -> float:
        return self._get(("weighted", group, quantity, metric), lambda: comparison_weighted(
            records, quantity, metric))


def _seed_spread(name: str, estimates: Mapping[int, float], interpreted: bool, band: float
                 ) -> dict[str, Any]:
    stats = seed_sensitivity(dict(estimates))
    columns: dict[str, Any] = {
        f"{name}_seed{seed}": _float(value) for seed, value in sorted(estimates.items())
    }
    columns.update({
        f"{name}_seed_mean": _float(stats["mean"]),
        f"{name}_seed_min": _float(stats["min"]),
        f"{name}_seed_max": _float(stats["max"]),
        f"{name}_seed_range": _float(stats["range"]),
        f"{name}_seed_n": int(stats["n_seeds"]),
    })
    if interpreted:
        finite = [value for value in estimates.values() if math.isfinite(value)]
        columns[f"{name}_seed_crosses_zero"] = (
            None if not finite
            else not (all(v > 0.0 for v in finite) or all(v < 0.0 for v in finite))
        )
        columns[f"{name}_seed_enters_band"] = (
            None if not finite else any(abs(v) <= band for v in finite)
        )
    return columns


def contrast_weighted(pivot: Sequence[Mapping[str, Any]], contrast: str, quantity: str,
                      metric: str) -> float:
    """PROTOCOL 3.4's comparison-weighted diagnostic for one region contrast.

    pivot is pivot_regions' output on the quantity's population, so it holds
    the pairs present in both regions. Each side is the quantity's
    comparison-weighted value over those pairs, read from that region's own
    fields and weighted by that region's own counts. The contrast is left
    minus right, as evaluate_contrast takes it. It is a diagnostic beside the
    contrast's estimate, never the estimate. An empty pivot gives NaN.
    """
    population = quantity_population(quantity, metric)
    count_field = population_count_field(population)
    fields = population_fields(population)

    def side(region: str) -> list[dict[str, Any]]:
        return [{name: record.get(contrast_field(name, region))
                 for name in (count_field,) + tuple(fields)} for record in pivot]

    left, right = REGION_CONTRASTS[contrast]
    return float(comparison_weighted(side(left), quantity, metric)
                 - comparison_weighted(side(right), quantity, metric))


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TableSet:
    """Every table's rows, in writing order, with the two JSON outputs."""

    tables: dict[str, list[dict]]
    near_zero: list[dict]
    accounting: dict[str, Any]
    labels: dict[str, str]


class _Builder:
    def __init__(self, prepared: PreparedRun, analysis: AnalysisConfig):
        self.prepared = prepared
        self.analysis = analysis
        self.level = prepared.level
        self.primary = prepared.level == prepared.primary_level
        self.seeds = prepared.seeds
        self.band = analysis.path_agreement_tolerance
        self.ev = _Intervals(analysis)
        self.near_zero: list[dict] = []
        # Support counts records as camera pairs, so raw seed rows would count
        # each pair once per seed. Each pair is held once per region, and once
        # per seed, or the tables stop.
        _require_once(prepared.collapsed, ("camera_pair", "region"), "the collapsed records")
        for seed in self.seeds:
            _require_once(prepared.per_seed[seed], ("camera_pair",), f"seed {seed}'s records")
        whole = [record for record in prepared.collapsed if record["region"] == "all"]
        self.cells, self.cell_accounting = reporting_cells(whole, analysis)
        keys = [key for key, _ in self.cells]
        self.seed_cells: dict[int, dict[CellKey, list]] = {}
        for seed in self.seeds:
            members, _ = _cell_members(prepared.per_seed[seed], analysis)
            if set(members) != set(keys):
                raise TablesStop(
                    f"seed {seed}'s records fall in other cells than the collapsed records"
                )
            self.seed_cells[seed] = members
        self.regions = {
            region: [record for record in prepared.collapsed if record["region"] == region]
            for region in REGIONS if region != "all"
        }

    # -- shared pieces -----------------------------------------------------

    def _common(self, key: CellKey, metric: str, table: str, region: str = "all"
                ) -> dict[str, Any]:
        return {"level": self.level, "metric": metric, **key.columns(), "region": region,
                "table_label": TABLE_LABELS[table]}

    def _quantity(self, group: tuple, records: Sequence[dict], quantity: str, metric: str,
                  *, name: str | None = None, pair: bool = True, weighted: bool = True
                  ) -> dict[str, Any]:
        name = name or quantity
        defined = _defined(quantity, metric)
        nan = float("nan")
        if not defined:
            columns = _scene_columns(name, nan, nan, nan, 0)
            if pair:
                columns.update(_pair_columns(name, {"lo": nan, "hi": nan, "n_replicates": 0}))
            if weighted:
                columns[f"{name}_comparison_weighted"] = nan
        else:
            if metric in METRICS:
                cell = self.ev.cell(group, records, quantity, metric)
                columns = _scene_columns(name, cell.estimate, cell.lo, cell.hi, cell.n_replicates)
            else:
                interval, _ = self.ev.interval(group, records, quantity, metric, "scene")
                columns = _scene_columns(name, interval["estimate"], interval["lo"],
                                         interval["hi"], interval["n_replicates"])
            if pair:
                columns.update(_pair_columns(
                    name, self.ev.interval(group, records, quantity, metric, "camera_pair")[0]))
            if weighted:
                columns[f"{name}_comparison_weighted"] = _float(
                    self.ev.weighted(group, records, quantity, metric))
        if name in MEAN_FEATURE_NAMES:
            columns[f"{name}_status"] = (
                MEAN_FEATURE_REPORTED if defined else MEAN_FEATURE_NOT_APPLICABLE)
        return columns

    def _terms(self, group: tuple, records: Sequence[dict], effect: str, metric: str
               ) -> tuple[CellResult, dict[str, Any]]:
        cell = self.ev.cell(group, records, effect, metric)
        terms = self.ev.terms(group, records, effect, metric)
        checks = [(terms["reported"][k], getattr(cell, attr)) for k, attr in (
            ("estimate", "estimate"), ("lo", "lo"), ("hi", "hi"),
            ("n_replicates", "n_replicates"))]
        for path in ("per_point", "splat_pool", "path_difference"):
            if (terms[path] is None) != (cell.disclosure.get(path) is None):
                raise TablesStop(f"{effect}: the disclosure and its terms disagree on {path}")
            if terms[path] is not None:
                checks += [(terms[path][k], cell.disclosure[path][k]) for k in ("estimate", "lo", "hi")]
        if not all(_same(float(a), float(b)) for a, b in checks):
            raise TablesStop(f"{effect} under {metric}: the disclosure terms are not the "
                             "cell's own")
        return cell, terms

    def _disclosure(self, group: tuple, records: Sequence[dict], effect: str, metric: str
                    ) -> dict[str, Any]:
        cell, terms = self._terms(group, records, effect, metric)
        # A cell below support makes no claim, PROTOCOL 3.4: its terms are
        # shown, and its flag and wording are lot.phase5_outcomes' marker.
        shown = shown_near_zero(cell)
        columns: dict[str, Any] = {
            f"{effect}_near_zero": shown["near_zero"],
            f"{effect}_near_zero_wording": shown["wording"],
        }
        if terms["splat_pool"] is not None:
            for path in ("per_point", "splat_pool", "path_difference"):
                term = terms[path]
                columns.update(_scene_columns(term["quantity"], term["estimate"], term["lo"],
                                              term["hi"], term["n_replicates"]))
        return columns

    def _cross_support(self, group: tuple, records: Sequence[dict], effect: str, metric: str
                       ) -> dict[str, Any]:
        support = self._terms(group, records, effect, metric)[1]["cross_path_support"]
        return {f"cross_path_{name}": value for name, value in support.items()}

    def _entry(self, table: str, key: CellKey, region: str, group: tuple,
               records: Sequence[dict], effect: str, metric: str,
               seed_crosses_zero: bool | None = None, path: str | None = None) -> None:
        cell, terms = self._terms(group, records, effect, metric)
        disclosure = cell.disclosure
        shown = shown_near_zero(cell)
        reported = terms["reported"]["estimate"]
        # The terms, the band, sign agreement, and clearance describe the
        # terms and are kept for every cell. The flag and the wording are a
        # claim, so a cell below support shows the marker instead.
        self.near_zero.append({
            "table": table,
            "level": self.level,
            "metric": metric,
            "region": region,
            "path": path,
            "analysis": key.analysis,
            "axis": key.axis,
            "bin": key.bin,
            "bin_index": key.bin_index,
            "quantity": effect,
            "supported": bool(cell.supported),
            "reported": terms["reported"],
            "per_point": terms["per_point"],
            "splat_pool": terms["splat_pool"],
            "path_difference": terms["path_difference"],
            "cross_path_support": terms["cross_path_support"],
            "band": disclosure.get("band"),
            "paths_agree_in_sign": disclosure.get("paths_agree_in_sign"),
            "both_intervals_exclude_zero": disclosure.get("both_intervals_exclude_zero"),
            "near_zero": shown["near_zero"],
            "wording": shown["wording"],
            "reported_in_band": bool(_finite(reported) and abs(reported) <= self.band),
            "seed_crosses_zero": seed_crosses_zero,
        })

    def _seeds(self, key: CellKey, records: Sequence[dict], quantities: Sequence[str],
               metric: str) -> dict[str, Any]:
        columns: dict[str, Any] = {}
        for quantity in quantities:
            estimates = {
                seed: self.ev.interval(("seed", seed, key), self.seed_cells[seed].get(key, []),
                                       quantity, metric, "scene")[0]["estimate"]
                for seed in self.seeds
            }
            spread = _seed_spread(quantity, estimates, quantity in INTERPRETED_EFFECTS,
                                  self.band)
            estimate = self.ev.cell(("all", key), records, quantity, metric).estimate
            mean = spread[f"{quantity}_seed_mean"]
            if (math.isfinite(estimate) and math.isfinite(mean)
                    and abs(mean - estimate) > SEED_MEAN_TOLERANCE):
                raise TablesStop(
                    f"{quantity} under {metric} in {key.analysis} {key.bin}: the mean of the "
                    f"per-seed estimates, {mean!r}, is not the collapsed estimate "
                    f"{estimate!r}. Collapsing seeds and pooling them must agree"
                )
            columns.update(spread)
        return columns

    def _failures(self, records: Sequence[dict]) -> dict[str, Any]:
        contributing = [r for r in records if _positive(r.get("n_primary"))]
        columns = {
            seed_count_field(seed): sum(int(r.get(seed_count_field(seed), 0))
                                        for r in contributing)
            for seed in self.seeds
        }
        columns[f"{SEED_DEPENDENT_COUNT}_total"] = sum(columns.values())
        return columns

    # -- T1 ----------------------------------------------------------------

    def primary_table(self) -> list[dict]:
        rows = []
        for metric in METRICS:
            for key, records in self.cells:
                group = ("all", key)
                delta = self.ev.cell(group, records, "delta_learn_pp", metric)
                row = self._common(key, metric, PRIMARY_TABLE)
                row["headline_definition"] = HEADLINE_DEFINITION
                row.update(_support_columns(_cell_counts(delta), self.analysis))
                row["n_folds"] = _n_folds(records, "n_primary")
                row["n_pairs_in_cell"] = len(records)
                for quantity in PRIMARY_QUANTITIES:
                    row.update(self._quantity(group, records, quantity, metric))
                for effect in PRIMARY_EFFECTS:
                    row.update(self._disclosure(group, records, effect, metric))
                row.update(self._cross_support(group, records, "delta_learn_pp", metric))
                row.update(self._seeds(key, records, PRIMARY_SEEDED, metric))
                row.update(self._failures(records))
                if self.primary:
                    deficit = self.ev.cell(group, records, "read_deficit", metric)
                    row.update(self._quantity(group, records, "read_deficit", metric,
                                              weighted=False))
                    row.update(_support_columns(_cell_counts(deficit), self.analysis,
                                                prefix="read_deficit_"))
                    row.update(landing_offset_flags(delta, deficit))
                    row["read_depth_note"] = READ_DEPTH_NOTE
                    row["landing_offset_status"] = LANDING_OFFSET_REPORTED
                else:
                    row["landing_offset_status"] = LANDING_OFFSET_PRIMARY_ONLY
                called = gap_outcome(delta)
                splat = gap_outcome(self.ev.cell(group, records, "delta_learn_sp", metric))
                row.update({
                    "delta_learn_pp_outcome": called["outcome"],
                    "delta_learn_pp_outcome_wording": called["outcome_wording"],
                    "delta_learn_pp_qualifier": called["qualifier"],
                    "delta_learn_sp_outcome": splat["outcome"],
                    "outcome_49": outcome_49(called["outcome"], splat["outcome"]),
                    "metric_sensitive": metric_sensitive(
                        self.ev.cell(group, records, "delta_learn_pp", "centered"),
                        self.ev.cell(group, records, "delta_learn_pp", "raw"),
                    ),
                })
                rows.append(row)
                for effect in PRIMARY_EFFECTS:
                    self._entry(PRIMARY_TABLE, key, "all", group, records, effect, metric,
                                row.get(f"{effect}_seed_crosses_zero"), PER_POINT)
        _assert_no_target_lift(rows)
        return rows

    # -- T2 ----------------------------------------------------------------

    def formulation_table(self) -> list[dict]:
        rows = []
        for metric in METRICS:
            for key, records in self.cells:
                group = ("all", key)
                effect = self.ev.cell(group, records, "delta_formulation", metric)
                row = self._common(key, metric, FORMULATION_TABLE)
                row["compared_methods"] = f"{TL_REFERENCE} minus {CL_TRANSPORT}"
                row.update(_support_columns(_cell_counts(effect), self.analysis))
                row["n_folds"] = _n_folds(records, "n_formulation")
                row["n_pairs_in_cell"] = len(records)
                for quantity in FORMULATION_QUANTITIES:
                    row.update(self._quantity(group, records, quantity, metric))
                row.update(self._disclosure(group, records, "delta_formulation", metric))
                rows.append(row)
                self._entry(FORMULATION_TABLE, key, "all", group, records, "delta_formulation",
                            metric, None, PER_POINT)
        return rows

    # -- T3 ----------------------------------------------------------------

    def splat_table(self) -> list[dict]:
        rows = []
        for metric in METRICS:
            for key, records in self.cells:
                group = ("all", key)
                gap = self.ev.cell(group, records, "delta_learn_sp", metric)
                row = self._common(key, metric, SPLAT_TABLE)
                row["explicit_method"] = SPLAT_TRANSPORT
                row.update(_support_columns(_cell_counts(gap), self.analysis))
                row["n_folds"] = _n_folds(records, "n_splat")
                row["n_pairs_in_cell"] = len(records)
                for quantity in SPLAT_QUANTITIES:
                    row.update(self._quantity(group, records, quantity, metric))
                for effect in SPLAT_EFFECTS:
                    row.update(self._disclosure(group, records, effect, metric))
                row.update(self._cross_support(group, records, "delta_learn_sp", metric))
                row.update(self._seeds(key, records, SPLAT_SEEDED, metric))
                called = gap_outcome(gap)
                row.update({
                    "delta_learn_sp_outcome": called["outcome"],
                    "delta_learn_sp_qualifier": called["qualifier"],
                    "metric_sensitive": metric_sensitive(
                        self.ev.cell(group, records, "delta_learn_sp", "centered"),
                        self.ev.cell(group, records, "delta_learn_sp", "raw"),
                    ),
                })
                rows.append(row)
                for effect in SPLAT_EFFECTS:
                    self._entry(SPLAT_TABLE, key, "all", group, records, effect, metric,
                                row.get(f"{effect}_seed_crosses_zero"), SPLAT_POOL)
        return rows

    # -- T4 ----------------------------------------------------------------

    def cross_path_table(self) -> list[dict]:
        rows = []
        for metric in METRICS:
            for key, records in self.cells:
                group = ("all", key)
                anchor = self.ev.cell(group, records, "x_delta_learn_pp", metric)
                row = self._common(key, metric, CROSS_PATH_TABLE)
                row.update(_support_columns(_cell_counts(anchor), self.analysis))
                row["n_folds"] = _n_folds(records, "n_intersect")
                row["n_pairs_in_cell"] = len(records)
                for quantity in CROSS_PATH_QUANTITIES:
                    row.update(self._quantity(group, records, quantity, metric))
                rows.append(row)
        return rows

    # -- T5 ----------------------------------------------------------------

    def landing_offset_table(self) -> list[dict]:
        bounds = offset_bounds(self.prepared.offset_edges)
        rows = []
        for metric in METRICS:
            for key, records in self.cells:
                group = ("all", key)
                for label in OFFSET_BINS + (OFFSET_WHOLE,):
                    anchor = self.ev.cell(group, records, f"cl_oracle_offset_{label}", metric)
                    row = self._common(key, metric, LANDING_OFFSET_TABLE)
                    lo, hi = bounds[label]
                    row.update({"offset_label": label, "offset_lo": lo, "offset_hi": hi,
                                "offset_bounds_note": OFFSET_BOUNDS_NOTE})
                    row.update(_support_columns(_cell_counts(anchor), self.analysis))
                    row["n_folds"] = _n_folds(records, offset_count_field(label))
                    row["n_pairs_in_cell"] = len(records)
                    for base in OFFSET_QUANTITIES:
                        row.update(self._quantity(group, records, f"{base}_{label}", metric,
                                                  name=base))
                    row["reading_note"] = OFFSET_READING_NOTE
                    row["read_depth_note"] = READ_DEPTH_NOTE
                    rows.append(row)
                deficit = self.ev.cell(group, records, "read_deficit", metric)
                row = self._common(key, metric, LANDING_OFFSET_TABLE)
                row.update({"offset_label": "deficit", "offset_lo": None, "offset_hi": None,
                            "offset_bounds_note": OFFSET_BOUNDS_NOTE})
                row.update(_support_columns(_cell_counts(deficit), self.analysis))
                row["n_folds"] = _n_folds(records, offset_count_field(OFFSET_BINS[0]))
                row["n_pairs_in_cell"] = len(records)
                row.update(self._quantity(group, records, "read_deficit", metric,
                                          weighted=False))
                row["reading_note"] = OFFSET_READING_NOTE
                row["read_depth_note"] = READ_DEPTH_NOTE
                rows.append(row)
        return rows

    # -- T6 ----------------------------------------------------------------

    def region_table(self) -> list[dict]:
        scopes = {
            region: dict(scope_cells(records)) for region, records in self.regions.items()
        }
        keys = sorted({key for cells in scopes.values() for key in cells},
                      key=lambda key: key.order)
        rows = []
        for metric in METRICS:
            for key in keys:
                for path in REGION_PATHS:
                    count_field = population_count_field(path)
                    gap_name = REGION_GAP[path]
                    for region in self.regions:
                        records = scopes[region].get(key, [])
                        group = ("region", region, key)
                        gap = self.ev.cell(group, records, gap_name, metric)
                        row = self._common(key, metric, REGION_TABLE, region)
                        row.update({"row_kind": "region", "path": path,
                                    "contrast_left": None, "contrast_right": None})
                        row.update(_support_columns(_cell_counts(gap), self.analysis))
                        row["n_folds"] = _n_folds(records, count_field)
                        row["n_pairs_in_cell"] = len(records)
                        for quantity in REGION_QUANTITIES[path]:
                            row.update(self._quantity(group, records, quantity, metric))
                        row.update(self._disclosure(group, records, gap_name, metric))
                        rows.append(row)
                        self._entry(REGION_TABLE, key, region, group, records, gap_name,
                                    metric, None, path)
                    for contrast, (left, right) in REGION_CONTRASTS.items():
                        records = scopes[left].get(key, []) + scopes[right].get(key, [])
                        pivot = pivot_regions(records, contrast, path)
                        row = self._common(key, metric, REGION_TABLE, contrast)
                        row.update({"row_kind": "contrast", "path": path,
                                    "contrast_left": left, "contrast_right": right})
                        gap = None
                        for quantity in CONTRAST_QUANTITIES[path]:
                            scene = evaluate_contrast(pivot, contrast, quantity, metric,
                                                      self.analysis, "scene")
                            pair = evaluate_contrast(pivot, contrast, quantity, metric,
                                                     self.analysis, "camera_pair")
                            name = f"contrast_{quantity}"
                            row.update(_scene_columns(name, scene["estimate"], scene["lo"],
                                                      scene["hi"], scene["n_replicates"]))
                            row.update(_pair_columns(name, pair))
                            # A contrast is a cell, so the weighted value sits
                            # beside its estimate, PROTOCOL 3.4.
                            row[f"{name}_comparison_weighted"] = _float(
                                contrast_weighted(pivot, contrast, quantity, metric))
                            if quantity == gap_name:
                                gap = scene
                        row.update(_support_columns(
                            SupportCounts(gap["n_scenes"], gap["n_camera_pairs"],
                                          gap["n_feature_comparisons"]), self.analysis))
                        paired = {record["camera_pair"] for record in pivot}
                        row["n_folds"] = len({record.get("fold") for record in records
                                              if record["camera_pair"] in paired})
                        row["n_pairs_in_cell"] = len({record["camera_pair"]
                                                      for record in records})
                        row["in_band"] = bool(_finite(gap["estimate"])
                                              and abs(gap["estimate"]) <= self.band)
                        row["contrast_note"] = CONTRAST_NOTE
                        rows.append(row)
        return rows

    # -- T7 ----------------------------------------------------------------

    def per_seed_table(self) -> list[dict]:
        rows = []
        for metric in METRICS:
            for key, _ in self.cells:
                for population, quantities in ((PER_POINT, PRIMARY_SEEDED),
                                               (SPLAT_POOL, SPLAT_SEEDED)):
                    for quantity in quantities:
                        for seed in self.seeds:
                            records = self.seed_cells[seed].get(key, [])
                            interval, counts = self.ev.interval(
                                ("seed", seed, key), records, quantity, metric, "scene")
                            row = self._common(key, metric, PER_SEED_TABLE)
                            row.update({"population": population, "quantity": quantity,
                                        "seed": seed})
                            row.update({
                                "estimate": _float(interval["estimate"]),
                                "ci_low": _float(interval["lo"]),
                                "ci_high": _float(interval["hi"]),
                                "ci_replicates": int(interval["n_replicates"]),
                            })
                            row.update(_support_columns(counts, self.analysis))
                            row["note"] = PER_SEED_NOTE
                            rows.append(row)
        return rows

    # -- T8 ----------------------------------------------------------------

    def l2_table(self) -> list[dict]:
        rows = []
        for metric in L2_METRIC_ORDER:
            for key, records in self.cells:
                group = ("all", key)
                for population in L2_POPULATIONS:
                    quantities = [quantity for quantity, home in L2_QUANTITY_POPULATION.items()
                                  if home == population]
                    anchor = next(q for q in quantities if _defined(q, metric))
                    _, counts = self.ev.interval(group, records, anchor, metric, "scene")
                    row = self._common(key, metric, L2_TABLE)
                    row["population"] = population
                    row.update(_support_columns(counts, self.analysis))
                    row["n_folds"] = _n_folds(records, population_count_field(population))
                    row["n_pairs_in_cell"] = len(records)
                    for quantity in quantities:
                        row.update(self._quantity(group, records, quantity, metric))
                    rows.append(row)
        return rows

    # -- T11 ---------------------------------------------------------------

    def reference_table(self, phase4: Mapping[str, Any]) -> list[dict]:
        level = phase4.get("level", self.level)
        path = phase4.get("path", PER_POINT)
        index: dict[tuple, Mapping[str, Any]] = {}
        for row in phase4.get("ladder") or []:
            if row.get("level") == level and row.get("path") == path:
                index[(row["metric"], row["analysis"], SCOPE_AXIS, "all")] = row
        for row in phase4.get("bins") or []:
            if row.get("level") == level and row.get("path") == path:
                index[(row["metric"], row["analysis"], row["axis"], row["bin"])] = row
        rows = []
        for metric in METRICS:
            for key, records in self.cells:
                group = ("all", key)
                delta = self.ev.cell(group, records, "delta_learn_pp", metric)
                match = index.get((PHASE4_METRIC[metric], key.analysis, key.axis, key.bin))
                row = self._common(key, metric, REFERENCE_TABLE)
                row.update({"phase4_level": level, "phase4_path": path,
                            "phase4_metric": PHASE4_METRIC[metric],
                            "phase4_matched": match is not None})
                # The floors sit beside the levels, CLAUDE.md and section 6:
                # No-Warp-Copy under both metrics, Mean-Feature under raw only.
                for quantity in REFERENCE_QUANTITIES:
                    row.update(self._quantity(group, records, quantity, metric, pair=False,
                                              weighted=False))
                row["supported"] = bool(delta.supported)
                reference = {
                    "phase3_reference_ceiling": None, "phase4_matched_ceiling": None,
                    "phase4_matched_floor": None,
                    "phase4_estimated_score": None, "phase4_depth_tax": None,
                    "phase4_depth_tax_ci_low": None, "phase4_depth_tax_ci_high": None,
                    "phase4_depth_tax_ci_replicates": None, "phase4_n_scenes": None,
                    "phase4_n_camera_pairs": None, "phase4_n_feature_comparisons": None,
                    "phase4_supported": None,
                }
                rungs: dict[str, Any] = {name: None for name in _RUNG_COLUMNS}
                if match is not None:
                    reference.update({
                        "phase3_reference_ceiling": _float(match.get("reference_ceiling_phase3")),
                        "phase4_matched_ceiling": _float(match.get("matched_ceiling")),
                        # Phase 4's own No-Warp-Copy floor on its matched set, as
                        # lot.phase4_report reports it beside the matched ceiling.
                        "phase4_matched_floor": _float(match.get("matched_floor")),
                        "phase4_estimated_score": _float(match.get("estimated_score")),
                        "phase4_depth_tax": _float(match.get("depth_tax")),
                        "phase4_depth_tax_ci_low": _float(match.get("depth_tax_ci_low")),
                        "phase4_depth_tax_ci_high": _float(match.get("depth_tax_ci_high")),
                        "phase4_depth_tax_ci_replicates": _int(
                            match.get("depth_tax_ci_replicates")),
                        "phase4_n_scenes": _int(match.get("n_scenes")),
                        "phase4_n_camera_pairs": _int(match.get("n_camera_pairs")),
                        "phase4_n_feature_comparisons": _int(match.get("n_feature_comparisons")),
                        "phase4_supported": bool(match.get("supported")),
                    })
                    decomposition = three_rung_decomposition(
                        reference["phase3_reference_ceiling"],
                        PathEstimate(reference["phase4_depth_tax"],
                                     reference["phase4_depth_tax_ci_low"],
                                     reference["phase4_depth_tax_ci_high"]),
                        delta,
                    )
                    rungs = {f"{rung}_{field}": value
                             for rung, values in decomposition.items()
                             for field, value in values.items()}
                row.update(reference)
                row.update(rungs)
                rows.append(row)
        return rows

    # -- the measured outcome ----------------------------------------------

    def measured_outcome_table(self) -> list[dict]:
        scopes = {key.analysis: (key, records) for key, records in self.cells
                  if key.axis == SCOPE_AXIS}
        missing = [scope for scope in SCOPES if scope not in scopes]
        if missing:
            raise TablesStop(f"no records for scope(s) {missing}, so the measured outcome, "
                             "one row per regime, cannot be formed")

        def cells(quantity: str) -> dict[tuple[str, str], CellResult]:
            return {(scope, metric): self.ev.cell(("all", scopes[scope][0]), scopes[scope][1],
                                                  quantity, metric)
                    for scope in SCOPES for metric in METRICS}

        rows = measured_outcome(cells("delta_learn_pp"), cells("delta_learn_sp"),
                                cells("read_deficit") if self.primary else None)
        out = []
        for row in rows:
            key, records = scopes[row["scope"]]
            group, metric = ("all", key), row["metric"]
            # Every interval carries its replicate count, PROTOCOL 3.4. The
            # measured outcome shows the cross-path terms and the read deficit
            # without theirs, so they are added beside them here.
            terms = self._terms(group, records, "delta_learn_pp", metric)[1]
            extra = {f"{terms[path]['quantity']}_ci_replicates": int(terms[path]["n_replicates"])
                     for path in ("per_point", "splat_pool", "path_difference")}
            if self.primary:
                extra["read_deficit_ci_replicates"] = int(
                    self.ev.cell(group, records, "read_deficit", metric).n_replicates)
            # Each row is a headline scope row, so it names that row's axis and
            # bin, and its visibility bucket, reporting_rules.md section 6.
            out.append({"level": self.level, **row, "axis": key.axis, "bin": key.bin,
                        "visibility_bucket": VISIBILITY_BUCKET, **extra,
                        "table_label": TABLE_LABELS[MEASURED_OUTCOME_TABLE]})
        return out

    # -- accounting --------------------------------------------------------

    def accounting(self, tables: Mapping[str, list[dict]],
                   adequacy: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
        scenes = self.prepared.scenes
        totals = {field: sum(entry[field] for entry in scenes.values())
                  for field in ("pairs", "evaluated", "no_arm", "rows")}
        no_arm: dict[str, dict[str, Any]] = {regime: {"count": 0, "pairs": []}
                                             for regime in REGIME_SCOPES}
        for entry in scenes.values():
            for pair in entry["no_arm_pairs"]:
                regime = pair[3] if len(pair) > 3 else "unknown"
                slot = no_arm.setdefault(regime, {"count": 0, "pairs": []})
                slot["count"] += 1
                slot["pairs"].append(pair)
        unbinnable = self.cell_accounting["unbinnable"]
        by_regime = {regime: sum(1 for item in unbinnable if item["regime"] == regime)
                     for regime in REGIME_SCOPES}
        empty = {
            count: {regime: sum(entry["empty"][count][regime] for entry in scenes.values())
                    for regime in REGIME_SCOPES}
            for count in UNBINNABLE_COUNTS
        }
        failures: dict[str, int] = {}
        for entry in scenes.values():
            for name, value in entry["n_predict_nonfinite"].items():
                failures[name] = failures.get(name, 0) + value
        return {
            "level": self.level,
            "primary_level": self.prepared.primary_level,
            "report_version": REPORT_VERSION,
            "seeds": list(self.seeds),
            "scenes": scenes,
            "totals": totals,
            "no_arm_pairs": no_arm,
            "unbinnable_pairs": {"count": len(unbinnable), "by_regime": by_regime,
                                 "pairs": unbinnable},
            "empty_supports": empty,
            "n_predict_nonfinite": failures,
            "cells": {"reporting_cells": len(self.cells)},
            "tables": {
                name: {"rows": len(rows),
                       "supported_rows": sum(1 for row in rows if row.get("supported") is True)}
                for name, rows in tables.items()
            },
            "training_superseded": {
                fold_seed_key(entry["fold"].index, entry["seed"]):
                    entry["training"].get("superseded")
                for entry in (adequacy or [])
            },
        }


# The three rungs' columns in the reference table, as three_rung_decomposition
# names them, flattened.
_RUNG_COLUMNS = (
    "representation_limitation_oracle_phase",
    "representation_limitation_oracle_estimator",
    "representation_limitation_oracle_estimate",
    "estimated_geometry_limitation_phase",
    "estimated_geometry_limitation_estimator",
    "estimated_geometry_limitation_estimate",
    "estimated_geometry_limitation_lo",
    "estimated_geometry_limitation_hi",
    "learned_vs_explicit_limitation_phase",
    "learned_vs_explicit_limitation_estimator",
    "learned_vs_explicit_limitation_population",
    "learned_vs_explicit_limitation_estimate",
    "learned_vs_explicit_limitation_lo",
    "learned_vs_explicit_limitation_hi",
    "learned_vs_explicit_limitation_n_replicates",
    "learned_vs_explicit_limitation_supported",
)


# ---------------------------------------------------------------------------
# Training adequacy and validation history, from the embedded evidence
# ---------------------------------------------------------------------------

def adequacy_entries(
    records_by_scene: Mapping[str, Mapping[str, Any]],
    folds: Sequence[Any],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """One entry per (fold, seed), from the evidence each run record embeds.

    Every scene of a fold embeds the same training records and controls, and
    every scene the same overfit verdict, or this stops. lot.phase5_provenance
    has already checked that each is the locked or licensed file's content.
    Entries come in (fold, seed) order.
    """
    seeds = _seed_tuple(seeds)
    by_index = {fold.index: fold for fold in folds}
    by_fold: dict[int, list[tuple[str, Mapping[str, Any]]]] = {}
    for scene, record in records_by_scene.items():
        by_fold.setdefault(record["fold"], []).append((scene, record))
    verdicts = {_canonical(record.get("overfit_verdict")) for record in records_by_scene.values()}
    if len(verdicts) != 1:
        raise TablesStop("the run records embed different overfit verdicts")
    entries = []
    for fold_index in sorted(by_fold):
        if fold_index not in by_index:
            raise TablesStop(f"fold {fold_index} is not one of the folds {sorted(by_index)}")
        (scene, first), *others = by_fold[fold_index]
        for other_scene, other in others:
            for field in ("training_record_contents", "controls"):
                if _canonical(other.get(field)) != _canonical(first.get(field)):
                    raise TablesStop(f"fold {fold_index}: {other_scene} and {scene} embed "
                                     f"different {field}")
        contents = first.get("training_record_contents") or {}
        controls = first.get("controls") or {}
        for seed in seeds:
            key = fold_seed_key(fold_index, seed)
            training = contents.get(str(seed))
            result = (controls.get("results") or {}).get(key)
            if not isinstance(training, Mapping) or not isinstance(result, Mapping):
                raise TablesStop(f"{key}: the run records embed no training record or no "
                                 "controls result for it")
            entries.append({
                "fold": by_index[fold_index],
                "seed": seed,
                "training": training,
                "controls": result,
                "controls_checkpoint": (controls.get("checkpoints") or {}).get(key),
                "controls_sha256": controls.get("sha256"),
                "overfit": first.get("overfit_verdict") or {},
            })
    return entries


def _scene_list(value: Any) -> list[str]:
    return [str(item) for item in (value or [])]


def adequacy_table(
    entries: Sequence[Mapping[str, Any]],
    analysis: AnalysisConfig,
    *,
    level: str,
    primary_level: str,
    training: Mapping[str, Any] | None = None,
) -> list[dict]:
    """Specification step 44, one row per (fold, seed).

    The scene roles are checked against the fold, and the scenes each role
    planned from must be inside it. Decision 5's control labels and stable
    validation curve are applied with the frozen 0.003, path_agreement_tolerance.
    The validation score is a model-selection statistic and is labelled one.
    """
    tolerance = analysis.path_agreement_tolerance
    training = training or {}
    rows = []
    for entry in entries:
        fold, seed = entry["fold"], entry["seed"]
        record, controls, verdict = entry["training"], entry["controls"], entry["overfit"]
        key = fold_seed_key(fold.index, seed)
        problems = []
        for field, want in (("fold", fold.index), ("seed", seed), ("level", level)):
            if record.get(field) != want:
                problems.append(f"{field} is {record.get(field)!r}, not {want!r}")
        for role in ("train", "val", "test"):
            if _scene_list(record.get(f"{role}_scenes")) != list(getattr(fold, role)):
                problems.append(f"{role}_scenes are not fold {fold.index}'s")
        for role in ("train", "val"):
            planned = set(_scene_list(record.get(f"{role}_scenes_planned")))
            if not planned <= set(getattr(fold, role)):
                problems.append(f"{role}_scenes_planned reach outside the {role} role: "
                                f"{sorted(planned - set(getattr(fold, role)))}")
        if problems:
            raise TablesStop(f"{key}: the training record does not describe its run: "
                             + "; ".join(problems))
        history = [[_int(step), _float(score)] for step, score in (record.get("history") or [])]
        scores = [score for _, score in history]
        last = scores[-1] if scores else float("nan")
        best_seen = max(scores) if scores and all(math.isfinite(s) for s in scores) else float("nan")
        census = record.get("census") or []
        stopped_early = record.get("stopped_early") is True
        primary = level == primary_level
        scope = (
            f"a single gate for every training run: run once on fold {verdict.get('fold')}, "
            f"seed {verdict.get('seed')}, at the primary level {verdict.get('level')}"
            if primary else
            f"not rerun at this level: the gate runs once, at the primary level "
            f"{verdict.get('level')}, and this level reuses its verdict"
        )
        row: dict[str, Any] = {
            "level": level,
            "fold": int(fold.index),
            "seed": int(seed),
            "train_scenes": ",".join(_scene_list(record.get("train_scenes"))),
            "val_scenes": ",".join(_scene_list(record.get("val_scenes"))),
            "test_scenes": ",".join(_scene_list(record.get("test_scenes"))),
            "train_scenes_planned": ",".join(_scene_list(record.get("train_scenes_planned"))),
            "val_scenes_planned": ",".join(_scene_list(record.get("val_scenes_planned"))),
            "n_train_examples": _int(record.get("n_train_examples")),
            "n_val_examples": _int(record.get("n_val_examples")),
            "n_pairs": sum(int(item.get("n_pairs", 0)) for item in census),
            "n_planned": sum(int(item.get("n_planned", 0)) for item in census),
            "n_no_arm": sum(int(item.get("n_no_arm", 0)) for item in census),
            "n_empty_support": sum(int(item.get("n_empty_support", 0)) for item in census),
            "tiny_overfit_passed": verdict.get("passed") is True,
            "tiny_overfit_reached_centered_cosine": _float(
                verdict.get("reached_centered_cosine")),
            "tiny_overfit_threshold": _float(verdict.get("threshold")),
            "tiny_overfit_steps": _int(verdict.get("steps")),
            "tiny_overfit_n_pairs": _int(verdict.get("n_pairs")),
            "tiny_overfit_regimes": ",".join(_scene_list(verdict.get("regimes"))),
            "tiny_overfit_fold": _int(verdict.get("fold")),
            "tiny_overfit_seed": _int(verdict.get("seed")),
            "tiny_overfit_level": verdict.get("level"),
            "tiny_overfit_receipt_sha256": verdict.get("sha256"),
            "tiny_overfit_scope": scope,
            "selected_checkpoint": record.get("checkpoint"),
            "checkpoint_sha256": record.get("checkpoint_sha256"),
            "best_step": _int(record.get("best_step")),
            "best_validation_centered_cosine": _float(
                record.get("best_validation_centered_cosine")),
            "best_validation_note": BEST_VALIDATION_NOTE,
            "n_validations": len(history),
            "last_validation_centered_cosine": _float(last),
            "validation_last_minus_best": _float(last - best_seen),
            "steps_run": _int(record.get("steps_run")),
            "max_steps": _int(training.get("max_steps")),
            "stopped_early": stopped_early,
            "validation_every_steps": _int(training.get("validation_every_steps")),
            "early_stopping_patience": _int(training.get("early_stopping_patience")),
            "stable_validation_curve": stable_validation_curve(history, stopped_early,
                                                               tolerance),
            "parameter_count": _int(record.get("parameter_count")),
            "training_config_digest": record.get("training_config_digest"),
            "config_digest": record.get("config_digest"),
            "training_commit": record.get("commit"),
            "training_written_utc": record.get("written_utc"),
            "controls_n_pairs": _int(controls.get("n_pairs")),
            "controls_val_scenes_planned": ",".join(
                _scene_list(controls.get("val_scenes_planned"))),
            "controls_checkpoint_sha256": entry.get("controls_checkpoint"),
            "controls_sha256": entry.get("controls_sha256"),
            "controls_note": CONTROLS_NOTE,
            "table_label": TABLE_LABELS[ADEQUACY_TABLE],
        }
        for name, prefix in (("pose_shuffle", "pose"), ("depth_shuffle", "depth")):
            block = controls.get(name) or {}
            degradation = _float(block.get("degradation"))
            row.update({
                f"{prefix}_baseline": _float(block.get("baseline_centered_cosine")),
                f"{prefix}_shuffled": _float(block.get("shuffled_centered_cosine")),
                f"{prefix}_degradation": degradation,
                f"{prefix}_n_samples": _int(block.get("n_samples")),
                f"{prefix}_n_unchanged": _int(block.get("n_unchanged")),
                f"{prefix}_label": control_label(degradation, tolerance),
            })
        rows.append(row)
    return rows


def validation_history_table(entries: Sequence[Mapping[str, Any]], *, level: str) -> list[dict]:
    """Every validation of every training run, one row each, in recorded order."""
    rows = []
    for entry in entries:
        record = entry["training"]
        best_step = _int(record.get("best_step"))
        for index, (step, score) in enumerate(record.get("history") or []):
            rows.append({
                "level": level,
                "fold": int(entry["fold"].index),
                "seed": int(entry["seed"]),
                "validation_index": index,
                "step": int(step),
                "validation_centered_cosine": _float(score),
                "is_best": int(step) == best_step,
                "table_label": TABLE_LABELS[HISTORY_TABLE],
            })
    return rows


# ---------------------------------------------------------------------------
# Building every table
# ---------------------------------------------------------------------------

def build_tables(
    prepared: PreparedRun,
    analysis: AnalysisConfig,
    *,
    adequacy: Sequence[Mapping[str, Any]] | None = None,
    training: Mapping[str, Any] | None = None,
    phase4: Mapping[str, Any] | None = None,
) -> TableSet:
    """Every table from prepared records, in writing order, with the JSON outputs.

    adequacy is adequacy_entries' output, and training names max_steps,
    validation_every_steps, and early_stopping_patience. Without adequacy the
    adequacy and history tables are not built. phase4 holds the Phase 4 cells
    phase4_reference_cells reads. Without it the reference table is not built.
    The landing-offset table and the read-deficit columns are built at the
    primary level only.
    """
    for method in HEADLINE_METHODS:
        assert_not_target_lift_headline("delta_learn_pp", method)
    builder = _Builder(prepared, analysis)
    tables: dict[str, list[dict]] = {
        PRIMARY_TABLE: builder.primary_table(),
        FORMULATION_TABLE: builder.formulation_table(),
        SPLAT_TABLE: builder.splat_table(),
        CROSS_PATH_TABLE: builder.cross_path_table(),
    }
    if builder.primary:
        tables[LANDING_OFFSET_TABLE] = builder.landing_offset_table()
    tables[REGION_TABLE] = builder.region_table()
    tables[PER_SEED_TABLE] = builder.per_seed_table()
    tables[L2_TABLE] = builder.l2_table()
    if adequacy is not None:
        tables[ADEQUACY_TABLE] = adequacy_table(
            adequacy, analysis, level=prepared.level, primary_level=prepared.primary_level,
            training=training)
        tables[HISTORY_TABLE] = validation_history_table(adequacy, level=prepared.level)
    if phase4 is not None:
        tables[REFERENCE_TABLE] = builder.reference_table(phase4)
    tables[MEASURED_OUTCOME_TABLE] = builder.measured_outcome_table()
    return TableSet(
        tables=tables,
        near_zero=builder.near_zero,
        accounting=builder.accounting(tables, adequacy),
        labels={name: TABLE_LABELS[name] for name in tables},
    )


# ---------------------------------------------------------------------------
# The Phase 4 references, from the Phase 4 evaluation parquets
# ---------------------------------------------------------------------------

def phase4_reference_cells(cfg: Any, analysis: AnalysisConfig, level: str, identity: Any
                           ) -> dict[str, Any]:
    """The Phase 3 ceiling and the Phase 4 depth tax per cell, at one level.

    Read from the Phase 4 evaluation parquets through lot.phase4_report's
    verified reader, which checks they are one accepted run under this
    analysis. Each parquet must be the one evaluation reconciled against, by
    the sha256 every Phase 5 run record names, and the Phase 4 run records
    must name the Phase 4 commit the evaluated run names. The files are hashed
    again after the read. Only the per-point path at the given level is kept,
    and its ladder and bin rows come from phase4_report as it computes them.
    """
    from . import phase4_report
    from .evaluate import PER_POINT as PHASE4_PER_POINT
    from .evaluate import read_run_metadata

    eval_dir = Path(cfg.phase4_dir) / "eval"
    files = sorted(path for path in eval_dir.glob("*.parquet") if path.is_file())
    if not files:
        raise TablesStop(f"no Phase 4 evaluation parquets under {eval_dir}")
    inputs = {f"phase4/eval/{path.name}": sha256_file(path) for path in files}
    problems = []
    for scene in identity.scenes:
        want = identity.records[scene].get("phase4_parquet_sha256")
        have = inputs.get(f"phase4/eval/{scene}.parquet")
        if have != want:
            problems.append(f"{scene}: the Phase 4 parquet has sha256 {have}, but evaluation "
                            f"reconciled against {want}")
    commits = sorted({str((read_run_metadata(path) or {}).get("git_commit")) for path in files})
    if commits != [str(identity.phase4_commit)]:
        problems.append(f"phase4_commit: the Phase 4 run records name {commits}, and the "
                        f"evaluated run names {identity.phase4_commit}")
    if problems:
        raise TablesStop("the Phase 4 references are not the run evaluation read:\n  - "
                         + "\n  - ".join(problems))
    rows = phase4_report.read_phase4_dir(eval_dir, analysis)
    records, exclusions = phase4_report.build_records(rows, analysis)
    del rows
    if exclusions.get("mask_mismatched_arms"):
        raise TablesStop(f"{exclusions['mask_mismatched_arms']} Phase 4 matched arms carry "
                         "different sample masks")
    chosen = [r for r in records if r.get("level") == level and r.get("path") == PHASE4_PER_POINT]
    del records
    ladder = [row for row in phase4_report.ladder_table(chosen, analysis)
              if row.get("level") == level]
    bins = [row for row in phase4_report.bin_table(chosen, analysis) if row.get("level") == level]
    after = {f"phase4/eval/{path.name}": sha256_file(path) for path in files}
    if after != inputs:
        raise TablesStop("the Phase 4 parquets changed while they were read")
    return {"ladder": ladder, "bins": bins, "inputs": inputs, "level": level,
            "path": PHASE4_PER_POINT, "phase4_commit": identity.phase4_commit}


# ---------------------------------------------------------------------------
# The levels
# ---------------------------------------------------------------------------

# What a level is in the frozen configuration. reporting_rules.md decision 4:
# the primary level's result decides Rung 2, and every other level is
# reported beside it, never in its place.
PRIMARY_ROLE = "primary"
SENSITIVITY_ROLE = "sensitivity"
DIAGNOSTIC_ROLE = "diagnostic"


def level_role(cfg: Any, level: str) -> str:
    """Whether level is the primary level, a sensitivity level, or a diagnostic one.

    Read from the frozen configuration. An undeclared level raises ValueError.
    """
    if level == cfg.primary_alignment_level:
        return PRIMARY_ROLE
    if level in cfg.sensitivity_alignment_levels:
        return SENSITIVITY_ROLE
    if level in cfg.diagnostic_alignment_levels:
        return DIAGNOSTIC_ROLE
    raise ValueError(f"level {level!r} is not declared in the configuration")


# ---------------------------------------------------------------------------
# Writing, once
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: Any) -> None:
    """Write a JSON file once and whole, through a temporary name and a rename.

    Floats are written by repr, so values are exact. A NaN is written as NaN,
    which Python's json reads back.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; outputs are written once")
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def write_tables(directory: Path, tableset: TableSet, identity: Any,
                 phase4_inputs: Mapping[str, str] | None = None, *,
                 role: str) -> list[str]:
    """Write every table, then the JSON outputs, then MANIFEST.json, last.

    Each parquet carries its run record, from output_run_record, with the
    table's name and label. The reference table's record names the Phase 4
    parquets it read. MANIFEST.json names every other file by sha256 beside
    the run record. Every run record names the level's role, level_role's, so
    an output read on its own says whether it is the primary result. Returns
    the names written, in order.
    """
    directory = Path(directory)
    phase4_inputs = dict(phase4_inputs or {})
    written: list[str] = []
    files: dict[str, str] = {}
    for name, rows in tableset.tables.items():
        if not rows:
            raise TablesStop(f"{name} has no rows")
        record = output_run_record(
            identity, TABLE_KIND,
            inputs=phase4_inputs if name == REFERENCE_TABLE else None,
            extra={"table": name, "table_label": tableset.labels[name],
                   "report_version": REPORT_VERSION, "level_role": role},
        )
        write_parquet_with_record(directory / name, _rectangular(rows), record)
        files[name] = sha256_file(directory / name)
        written.append(name)
    _write_json(directory / NEAR_ZERO_FILE, {
        "run_record": output_run_record(
            identity, "phase5_near_zero",
            extra={"report_version": REPORT_VERSION, "level_role": role}),
        "entries": tableset.near_zero,
    })
    files[NEAR_ZERO_FILE] = sha256_file(directory / NEAR_ZERO_FILE)
    written.append(NEAR_ZERO_FILE)
    _write_json(directory / ACCOUNTING_FILE, {
        "run_record": output_run_record(
            identity, "phase5_accounting",
            extra={"report_version": REPORT_VERSION, "level_role": role}),
        **tableset.accounting,
    })
    files[ACCOUNTING_FILE] = sha256_file(directory / ACCOUNTING_FILE)
    written.append(ACCOUNTING_FILE)
    _write_json(directory / MANIFEST_FILE, {
        "kind": "phase5_tables_manifest",
        "report_version": REPORT_VERSION,
        "files": dict(sorted(files.items())),
        "run_record": output_run_record(
            identity, "phase5_tables_manifest", inputs=phase4_inputs,
            extra={"report_version": REPORT_VERSION, "tables": list(tableset.tables),
                   "level_role": role},
        ),
    })
    written.append(MANIFEST_FILE)
    return written


def require_primary_tables(directory: Path, *, primary_level: str) -> Any:
    """The primary level's published tables, verified, or a refusal.

    A non-primary level is reported only beside the primary level's published
    tables, reporting_rules.md decision 4. They are read through
    lot.phase5_figures.read_published_tables, as the figures mode reads them.
    So MANIFEST.json must be the tables mode's, of this report version and of
    primary_level, with its run record. Every file it names must still have
    its sha256, and every table must carry the run record MANIFEST.json names.

    A missing MANIFEST.json raises FileNotFoundError, and any other refusal
    raises ProvenanceError naming each problem's field. Returns the verified
    tables, whose run_record names the primary run, for
    lot.phase5_provenance.require_primary_chain.
    """
    from .phase5_figures import read_published_tables

    directory = Path(directory)
    if not (directory / MANIFEST_FILE).is_file():
        raise FileNotFoundError(
            f"no primary tables at {directory}; a non-primary level is reported only "
            "after the primary level's tables are published"
        )
    try:
        return read_published_tables(directory, level=primary_level)
    except ProvenanceError as error:
        raise ProvenanceError(
            error.problems,
            f"the primary level's tables at {directory} cannot stand beside a non-primary "
            "level",
        ) from None


def run_tables(
    cfg: Any,
    analysis: AnalysisConfig,
    config_path: Path,
    level: str,
    supersede: bool = False,
    *,
    expected_scenes: Sequence[str] | None = None,
    folds: Sequence[Any] | None = None,
    repo_root: Path | None = None,
    check_code: bool = True,
    phase4_reference: Callable[..., Mapping[str, Any]] | None = None,
    tables_root: Path | None = None,
) -> dict[str, Any]:
    """The tables mode: one evaluated level's tables, written once.

    An existing output is refused before any work, unless supersede is given,
    and a non-primary level requires the primary level's tables, verified
    through require_primary_tables. The run is licensed by
    require_evaluated_run. A non-primary level's run must then be the primary
    run's in all but its level, lot.phase5_provenance.require_primary_chain,
    reporting_rules.md decision 4. Each scene is read and prepared in turn,
    every table is built, and everything is written into
    tables/{level}.partial.<id>/ with MANIFEST.json last, then published by
    one rename. An earlier output is moved aside and nothing is deleted.

    expected_scenes, folds, repo_root, and check_code pass to
    require_evaluated_run, and phase4_reference replaces the Phase 4 reader,
    for tests. A reporting mode passes none of them. Returns where the tables
    were published, where an earlier output went, and the files written.
    """
    from .phase5 import landing_offset_edges
    from .phase5_folds import frozen_folds
    from .train import training_config_from

    root = Path(tables_root) if tables_root is not None else Path(cfg.run_dir) / TABLES_DIR
    final = root / level
    if final.exists() and not supersede:
        raise FileExistsError(
            f"{final} exists; outputs are written once. Supersede it to rebuild, which "
            "moves it aside and deletes nothing"
        )
    primary = cfg.primary_alignment_level
    role = level_role(cfg, level)
    primary_tables = (require_primary_tables(root / primary, primary_level=primary)
                      if level != primary else None)
    folds = list(folds) if folds is not None else frozen_folds()
    identity = require_evaluated_run(cfg, analysis, config_path, level,
                                     expected_scenes=expected_scenes, folds=folds,
                                     repo_root=repo_root, check_code=check_code)
    if primary_tables is not None:
        require_primary_chain(primary_tables.run_record, identity)
    reader = phase4_reference if phase4_reference is not None else phase4_reference_cells
    phase4 = reader(cfg, analysis, level, identity)
    sources = [
        (scene, (lambda scene=scene: read_scene_rows(identity, scene)), identity.records[scene])
        for scene in identity.scenes
    ]
    prepared = prepare_run(sources, analysis, level=level, primary_level=primary,
                           seeds=identity.seeds, offset_edges=landing_offset_edges(cfg))
    settings = training_config_from(cfg.training)
    tableset = build_tables(
        prepared, analysis,
        adequacy=adequacy_entries(identity.records, folds, identity.seeds),
        training={"max_steps": settings.max_steps,
                  "validation_every_steps": settings.validation_every_steps,
                  "early_stopping_patience": settings.early_stopping_patience},
        phase4=phase4,
    )
    with staged_output(final, supersede=supersede) as staged:
        written = write_tables(staged.path, tableset, identity,
                               phase4_inputs=(phase4 or {}).get("inputs"), role=role)
    return {**(staged.outcome or {}), "written": written}
