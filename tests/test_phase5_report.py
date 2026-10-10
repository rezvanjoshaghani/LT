"""Phase 5 tables: Stream AC under reporting_rules.md sections 6 and 9.

The records are synthetic and written in the layout lot.phase5_modes.
evaluate_scene writes: one row per (pair, seed, region), explicit arms
repeated across seeds, predictor columns per seed, formulation and offset
columns on region all rows only. Three scenes, one per fold, carry twelve
pairs per regime, so every regime row is supported and the outcome rules,
the flags, and the disclosure actually run. Every bin is thinner and stays
unsupported, so the unsupported branches run too.

The values are chosen so the expected outcomes are known in advance:

- rotation: Context-Lift leads by 0.05 under centered cosine and trails by
  0.0035 under raw cosine, so the cell is metric-sensitive;
- translation: Context-Lift leads by 0.04 on the per-point path while the
  splat-pool gap changes sign by scene, so outcome 49 is flagged;
- orbit: Predict-with-Depth leads by 0.03 on both paths;
- each region differs from the whole support by a constant shift, so every
  region contrast has an exact hand value.

The tables are built with a shortened bootstrap. The number of resamples is
a reporting value, and nothing here depends on the full thousand except the
test that compares pooling with collapsing, which uses the frozen analysis.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

import lot.phase5_report as report
from lot.analysis_config import load_analysis_config
from lot.evaluate import read_rows, read_run_metadata, write_rows
from lot.phase5_estimands import (
    ALL_FIELDS,
    CL_TRANSPORT,
    INTERPRETED_EFFECTS,
    OFFSET_BINS,
    OFFSET_WHOLE,
    PREDICT_WITH_DEPTH,
    TL_REFERENCE,
    WORDING_NOT_ESTIMABLE,
    WORDING_ONLY_PATH_CLEAR,
    WORDING_ONLY_PATH_INCLUDES_ZERO,
    WORDING_OUTSIDE_BAND,
    WORDING_OUTSIDE_ON_COMMON_CELLS,
    WORDING_PATH_SENSITIVE,
    WORDING_SMALL_SIGN_CONSISTENT,
    WORDING_VETO,
    evaluate_quantity,
    offset_count_field,
    offset_fields,
)
from lot.phase5_folds import frozen_folds
from lot.phase5_modes import REGIONS, fold_seed_key
from lot.phase5_outcomes import (
    ESSENTIALLY_NO_EFFECT,
    OUTCOME_WORDING,
    POOLED_LABEL,
)
from lot.phase5_report import (
    TablesStop,
    adequacy_entries,
    build_tables,
    camera_pair_key,
    check_population_finiteness,
    check_region_partition,
    check_regime_geometry,
    check_scene_rows,
    collapse_seeds,
    is_seed_dependent,
    prepare_run,
    reporting_cells,
    select_seed,
)

ANALYSIS = load_analysis_config()
# The interval settings are reporting values. Forty resamples keep the suite
# fast, and every outcome below is far from its interval's edge.
FAST = dataclasses.replace(ANALYSIS, bootstrap_resamples=40)
NAN = float("nan")
FOLDS = frozen_folds()
# One test scene of each fold, so every fold's evidence is read.
SCENES = tuple(fold.test[0] for fold in FOLDS)
LEVEL = "image"
SEEDS = (0, 1, 2)
EDGES = (0.1, 0.2, 0.3, 0.4, 0.5)
REGIMES = ("rotation", "translation", "orbit")
PER_REGIME = 12
REPO = Path(__file__).resolve().parents[1]

SCENE_LEVEL = {SCENES[0]: 0.0, SCENES[1]: 0.05, SCENES[2]: -0.03}
# delta_learn_pp each regime carries, by metric.
GAP_PP = {
    ("rotation", "centered"): 0.05, ("rotation", "raw"): -0.0035,
    ("translation", "centered"): 0.04, ("translation", "raw"): 0.03,
    ("orbit", "centered"): -0.03, ("orbit", "raw"): -0.03,
}
# delta_learn_sp each regime carries. Translation changes sign by scene.
GAP_SP = {"rotation": 0.03, "orbit": -0.02}
TRANSLATION_SP = {SCENES[0]: 0.004, SCENES[1]: -0.004, SCENES[2]: 0.001}
# Predictor seed s moves every predictor score by (s - 1) times this.
SEED_SHIFT = 0.002
# region: shifts of (Context-Lift, Predict-with-Depth, Transport-Only splat,
# Predict-with-Depth splat) relative to the whole support.
REGION_SHIFTS = {
    "all": (0.0, 0.0, 0.0, 0.0),
    "boundary": (-0.04, -0.01, -0.03, -0.01),
    "interior": (0.01, 0.005, 0.008, 0.003),
    "low_texture": (-0.02, -0.02, -0.01, -0.01),
    "high_texture": (0.01, 0.01, 0.005, 0.005),
}
# region: (n_primary, n_splat, n_intersect). Each split sums to all.
REGION_COUNTS = {
    "all": (200, 60, 50), "boundary": (60, 20, 15), "interior": (140, 40, 35),
    "low_texture": (90, 25, 20), "high_texture": (110, 35, 30),
}
OFFSET_COUNTS = (40, 30, 20, 10, 5, 0)
DEFICIT = 0.004
# The pair whose boundary holds no per-point sample, and the pair whose
# predictor fails twice under seed 1.
BOUNDARY_EMPTY = (SCENES[0], "rotation", 0)
FAILING = (SCENES[1], "translation", 0)


# ---------------------------------------------------------------------------
# Synthetic records in the layout evaluate_scene writes
# ---------------------------------------------------------------------------

def geometry(regime: str, index: int) -> tuple[float, float]:
    """(rotation_deg, parallax). 10 degrees and 0.05 sit exactly on an edge."""
    if regime == "rotation":
        return (5.0, 10.0, 15.0)[index % 3], 2e-7
    if regime == "translation":
        return 0.0, (0.03, 0.05, 0.15)[index % 3]
    return (5.0, 25.0)[index % 2], (0.01, 0.3)[(index // 2) % 2]


def _arm(prefix: str, centered: float, raw: float) -> dict:
    return {f"{prefix}_raw": raw, f"{prefix}_centered": centered,
            f"{prefix}_l2_raw": 1.0 - raw, f"{prefix}_l2_centered": 1.0 - centered}


def _floor(prefix: str, raw: float) -> dict:
    return {f"{prefix}_raw": raw, f"{prefix}_l2_raw": 1.0 - raw}


def _blank(fields) -> dict:
    return {field: NAN for field in fields}


def pair_rows(scene: str, fold: int, regime: str, index: int, *, level: str = LEVEL,
              seeds=SEEDS, empty: bool = False) -> list[dict]:
    """One pair's rows: every seed, every region."""
    from lot.phase5_estimands import (
        FORMULATION_FIELDS, INTERSECTION_FIELDS, PRIMARY_FIELDS, SPLAT_FIELDS,
    )

    rng = np.random.default_rng([SCENES.index(scene), REGIMES.index(regime), index])
    level_shift = SCENE_LEVEL[scene]
    rotation, parallax = geometry(regime, index)
    if empty:
        parallax = NAN
    base = {
        "scene": scene, "split": "test", "viewpoint": 0, "regime": regime,
        "context_frame_id": f"{regime}_{index:02d}_c",
        "target_frame_id": f"{regime}_{index:02d}_t",
        "baseline_m": 0.0 if regime == "rotation" else 0.2,
        "context_median_depth_m": 2.0, "rotation_deg": rotation, "parallax": parallax,
        "covisible_fraction": 0.8, "level": level, "fold": fold,
    }
    # Pair noise, the same in every region of the pair, so each region
    # contrast stays exact while no interval is degenerate.
    noise = {(arm, m): float(rng.normal(0.0, 1e-4))
             for arm in ("cl", "predict", "sp_predict") for m in ("centered", "raw")}
    sp_gap = TRANSLATION_SP[scene] if regime == "translation" else GAP_SP[regime]
    boundary_empty = (scene, regime, index) == BOUNDARY_EMPTY
    failing = (scene, regime, index) == FAILING

    rows = []
    for seed in seeds:
        seed_shift = (seed - 1) * SEED_SHIFT
        for region in REGIONS:
            n_primary, n_splat, n_intersect = REGION_COUNTS[region]
            if boundary_empty and region in ("boundary", "interior"):
                n_primary = 0 if region == "boundary" else 200
                n_intersect = 0 if region == "boundary" else 50
            if empty:
                n_primary = n_splat = n_intersect = 0
            s_cl, s_p, s_spt, s_spp = REGION_SHIFTS[region]
            cl = {m: (0.62 if m == "centered" else 0.85) + level_shift + noise[("cl", m)]
                  for m in ("centered", "raw")}
            predict = {m: cl[m] - GAP_PP[(regime, m)] + noise[("predict", m)] + seed_shift
                       for m in ("centered", "raw")}
            nowarp = {"centered": 0.40 + level_shift, "raw": 0.70 + level_shift}
            meanfeat = 0.30 + level_shift
            sp_t = {"centered": 0.66 + level_shift, "raw": 0.86 + level_shift}
            sp_p = {m: sp_t[m] - sp_gap + noise[("sp_predict", m)] + seed_shift
                    for m in ("centered", "raw")}
            sp_nowarp = {"centered": 0.45 + level_shift, "raw": 0.65 + level_shift}
            sp_meanfeat = 0.32 + level_shift

            row = {**base, "seed": seed, "region": region, "n_primary": n_primary}
            if n_primary:
                row.update(_arm("cl", cl["centered"] + s_cl, cl["raw"] + s_cl))
                row.update(_arm("predict", predict["centered"] + s_p, predict["raw"] + s_p))
                row.update(_arm("nowarp", nowarp["centered"], nowarp["raw"]))
                row.update(_floor("meanfeat", meanfeat))
            else:
                row.update(_blank(PRIMARY_FIELDS))
            fails = 2 if failing and seed == 1 and region in ("all", "boundary", "low_texture") else 0
            row["n_predict_nonfinite"] = fails
            if region == "all" and not empty:
                row["n_formulation"] = 80
                row.update(_arm("tl_form", cl["centered"] + 0.02, cl["raw"] + 0.02))
                row.update(_arm("cl_form", cl["centered"] - 0.001, cl["raw"] - 0.001))
                row.update(_arm("nowarp_form", nowarp["centered"], nowarp["raw"]))
                row.update(_floor("meanfeat_form", meanfeat))
            else:
                row["n_formulation"] = 0
                row.update(_blank(FORMULATION_FIELDS))
            row["n_splat"] = n_splat
            if n_splat:
                row.update(_arm("sp_transport", sp_t["centered"] + s_spt, sp_t["raw"] + s_spt))
                row.update(_arm("sp_predict", sp_p["centered"] + s_spp, sp_p["raw"] + s_spp))
                row.update(_arm("sp_nowarp", sp_nowarp["centered"], sp_nowarp["raw"]))
                row.update(_floor("sp_meanfeat", sp_meanfeat))
            else:
                row.update(_blank(SPLAT_FIELDS))
            row["n_intersect"] = n_intersect
            if n_intersect:
                row.update(_arm("x_cl", cl["centered"] + s_cl + 0.001, cl["raw"] + s_cl + 0.001))
                row.update(_arm("x_predict", predict["centered"] + s_p + 0.001,
                                predict["raw"] + s_p + 0.001))
                row.update(_arm("x_nowarp", nowarp["centered"], nowarp["raw"]))
                row.update(_floor("x_meanfeat", meanfeat))
                row.update(_arm("x_sp_transport", sp_t["centered"] + s_spt,
                                sp_t["raw"] + s_spt))
                row.update(_arm("x_sp_predict", sp_p["centered"] + s_spp, sp_p["raw"] + s_spp))
                row.update(_arm("x_sp_nowarp", sp_nowarp["centered"], sp_nowarp["raw"]))
                row.update(_floor("x_sp_meanfeat", sp_meanfeat))
            else:
                row.update(_blank(INTERSECTION_FIELDS))
            for k, label in enumerate(OFFSET_BINS + (OFFSET_WHOLE,)):
                count = (OFFSET_COUNTS[k] if label != OFFSET_WHOLE else sum(OFFSET_COUNTS))
                if region != "all" or empty:
                    count = 0
                row[offset_count_field(label)] = count
                if not count:
                    row.update(_blank(offset_fields(label)))
                    continue
                oracle = 0.70 + level_shift + (DEFICIT - 0.002 * k if label != OFFSET_WHOLE else 0.0)
                for m, lift in (("centered", 0.0), ("raw", 0.2)):
                    row[f"offset_cl_oracle_{m}_{label}"] = oracle + lift
                    row[f"offset_cl_oracle_l2_{m}_{label}"] = 1.0 - oracle - lift
                    row[f"offset_nowarp_{m}_{label}"] = 0.40 + level_shift + lift
                    row[f"offset_nowarp_l2_{m}_{label}"] = 0.60 - level_shift - lift
                row[f"offset_meanfeat_raw_{label}"] = meanfeat
                row[f"offset_meanfeat_l2_raw_{label}"] = 1.0 - meanfeat
            rows.append(row)
    return rows


def scene_rows(scene: str, fold: int, *, level: str = LEVEL, seeds=SEEDS,
               unbinnable: bool = False) -> tuple[list[dict], dict]:
    rows = []
    for regime in REGIMES:
        for index in range(PER_REGIME):
            rows += pair_rows(scene, fold, regime, index, level=level, seeds=seeds)
    if unbinnable:
        rows += pair_rows(scene, fold, "translation", 99, level=level, seeds=seeds, empty=True)
    pairs = len({(r["context_frame_id"], r["target_frame_id"]) for r in rows})
    audit = {"pairs": pairs, "evaluated": pairs, "no_arm": 0,
             "worst_per_point_residual": 0.0, "worst_splat_residual": 0.0,
             "no_arm_pairs": []}
    return rows, audit


def all_scene_rows(level: str = LEVEL) -> dict[str, tuple[list[dict], dict]]:
    return {scene: scene_rows(scene, fold, level=level, unbinnable=fold == 0)
            for fold, scene in enumerate(SCENES)}


def prepare(level: str = LEVEL, primary: str = LEVEL, analysis=FAST):
    sources = []
    for fold, (scene, (rows, audit)) in enumerate(all_scene_rows(level).items()):
        sources.append((scene, (lambda rows=rows: rows), {"fold": fold, "audit": audit}))
    return prepare_run(sources, analysis, level=level, primary_level=primary, seeds=SEEDS,
                       offset_edges=EDGES)


# The embedded evidence each evaluation run record carries, for the adequacy
# and history tables.

def training_record(fold, seed: int, level: str = LEVEL) -> dict:
    history = [[500 * i, 0.50 + 0.01 * i] for i in range(1, 9)]
    stopped_early = True
    if (fold.index, seed) == (2, 2):
        # Ran out of steps with the last score well below the best: unstable.
        history = [[500 * i, 0.50 + 0.01 * i] for i in range(1, 6)] + [[3000, 0.53]]
        stopped_early = False
    best = max(history, key=lambda item: item[1])
    return {
        "fold": fold.index, "seed": seed,
        "train_scenes": list(fold.train), "val_scenes": list(fold.val),
        "test_scenes": list(fold.test),
        "steps_run": history[-1][0], "best_step": best[0],
        "best_validation_centered_cosine": best[1],
        "parameter_count": 16680960, "training_config_digest": "7" * 64,
        "stopped_early": stopped_early, "history": history,
        "level": level, "checkpoint": f"outputs/checkpoints/{level}/fold{fold.index}_seed{seed}.pt",
        "checkpoint_sha256": f"{fold.index}{seed}" * 32,
        "n_train_examples": 120, "n_val_examples": 30,
        "census": [{"scene": scene, "level": level, "n_pairs": 10, "n_planned": 9,
                    "n_no_arm": 0, "n_empty_support": 1} for scene in fold.train],
        "superseded": {"checkpoint": None, "record": None},
        "config_digest": "8" * 64, "commit": "e" * 40, "licence": {},
        "train_scenes_planned": list(fold.train), "val_scenes_planned": list(fold.val),
        "written_utc": "2026-10-10T08:00:00.000000+00:00",
    }


def control_result(fold, seed: int) -> dict:
    return {
        "n_pairs": 40, "val_scenes_planned": list(fold.val),
        "pose_shuffle": {"baseline_centered_cosine": 0.6, "shuffled_centered_cosine": 0.4,
                         "degradation": 0.2, "n_samples": 900, "n_unchanged": 0},
        "depth_shuffle": {"baseline_centered_cosine": 0.6,
                          "shuffled_centered_cosine": 0.599, "degradation": 0.001,
                          "n_samples": 900, "n_unchanged": 3},
    }


OVERFIT = {"sha256": "f" * 64, "passed": True, "reached_centered_cosine": 0.9917,
           "threshold": 0.98, "steps": 1460, "n_pairs": 8,
           "regimes": ["rotation", "translation", "orbit"], "subset": [], "fold": 0,
           "level": LEVEL, "seed": 0}
TRAINING = {"max_steps": 20000, "validation_every_steps": 500, "early_stopping_patience": 10}


def evidence_records(level: str = LEVEL) -> dict[str, dict]:
    out = {}
    for fold, scene in zip(FOLDS, SCENES):
        keys = [fold_seed_key(fold.index, seed) for seed in SEEDS]
        out[scene] = {
            "fold": fold.index,
            "training_record_contents": {str(s): training_record(fold, s, level)
                                         for s in SEEDS},
            "controls": {"sha256": "c" * 64, "written_utc": "2026-10-10T08:30:00.000000+00:00",
                         "results": {key: control_result(fold, s) for key, s in zip(keys, SEEDS)},
                         "checkpoints": {key: "d" * 64 for key in keys}},
            "overfit_verdict": dict(OVERFIT),
        }
    return out


def phase4_rows(level: str = LEVEL) -> dict:
    """Phase 4 cells as lot.phase4_report's ladder and bin tables give them.

    Every scope row is present, and of the bins only rotation 0-10."""
    def cell(**keys):
        return {**keys, "path": "per_point", "level": level,
                "reference_ceiling_phase3": 0.70, "matched_ceiling": 0.68,
                "matched_floor": 0.60,
                "estimated_score": 0.64, "depth_tax": 0.04, "depth_tax_ci_low": 0.03,
                "depth_tax_ci_high": 0.05, "depth_tax_ci_replicates": 1000,
                "n_scenes": 18, "n_camera_pairs": 900, "n_feature_comparisons": 90000,
                "supported": True}

    metrics = ("cosine_mean", "cosine_centered_mean")
    ladder = [cell(analysis=scope, metric=metric)
              for scope in ("pooled", "rotation", "translation", "orbit") for metric in metrics]
    bins = [cell(analysis="rotation", axis="rotation_bin", bin="0-10", metric=metric)
            for metric in metrics]
    return {"ladder": ladder, "bins": bins, "level": level, "path": "per_point",
            "inputs": {"phase4/eval/x.parquet": "9" * 64}}


def build(level: str = LEVEL, primary: str = LEVEL, analysis=FAST, phase4=True):
    prepared = prepare(level=level, primary=primary, analysis=analysis)
    return build_tables(
        prepared, analysis,
        adequacy=adequacy_entries(evidence_records(level), FOLDS, SEEDS), training=TRAINING,
        phase4=phase4_rows(level) if phase4 else None,
    )


@pytest.fixture(scope="module")
def built():
    return build()


def rows_of(built, name: str) -> list[dict]:
    return built.tables[name]


def by_cell(rows, **filters) -> dict:
    """Rows keyed by (metric, analysis, axis, bin), after filtering."""
    out = {}
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            key = (row["metric"], row["analysis"], row["axis"], row["bin"])
            assert key not in out, key
            out[key] = row
    return out


def same(a, b) -> bool:
    """Equality that treats two NaNs as equal, through lists and dicts."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


# ---------------------------------------------------------------------------
# Seeds collapse to one record per pair and region
# ---------------------------------------------------------------------------

def _one_pair(**kwargs) -> list[dict]:
    return pair_rows(SCENES[0], 0, "translation", 3, **kwargs)


def test_collapse_seeds_means_the_predictor_and_keeps_the_explicit_fields():
    rows = _one_pair()
    collapsed = collapse_seeds(rows, SEEDS)
    assert len(collapsed) == len(REGIONS)
    assert [r["region"] for r in collapsed] == list(REGIONS)
    whole = collapsed[0]
    seed_rows = [r for r in rows if r["region"] == "all"]
    for field in ("predict_centered", "sp_predict_raw", "x_predict_l2_centered",
                  "x_sp_predict_centered"):
        assert whole[field] == pytest.approx(sum(r[field] for r in seed_rows) / 3, abs=1e-15)
    for field in ("cl_centered", "nowarp_raw", "tl_form_centered", "offset_cl_oracle_raw_b0",
                  "rotation_deg", "n_primary", "regime", "fold"):
        assert whole[field] == seed_rows[0][field]
    assert "seed" not in whole
    assert whole["camera_pair"] == camera_pair_key(whole) == (
        f"{SCENES[0]}|translation_03_c|translation_03_t")


def test_collapse_seeds_compares_explicit_fields_nan_aware():
    """A field that is NaN for every seed is identical across seeds."""
    rows = _one_pair()
    collapsed = collapse_seeds(rows, SEEDS)
    boundary = [r for r in collapsed if r["region"] == "boundary"][0]
    assert math.isnan(boundary["tl_form_centered"]) and boundary["n_formulation"] == 0


@pytest.mark.parametrize("field", ["cl_centered", "n_splat", "regime", "parallax",
                                   "offset_nowarp_raw_b2"])
def test_collapse_seeds_stops_on_an_explicit_field_that_differs_across_seeds(field):
    rows = _one_pair()
    target = [r for r in rows if r["seed"] == 2 and r["region"] == "all"][0]
    value = target[field]
    target[field] = value + 1 if isinstance(value, (int, float)) else "orbit"
    with pytest.raises(TablesStop, match=field):
        collapse_seeds(rows, SEEDS)


def test_collapse_seeds_stops_on_a_missing_seed():
    rows = [r for r in _one_pair() if not (r["seed"] == 1 and r["region"] == "interior")]
    with pytest.raises(TablesStop, match="missing seed"):
        collapse_seeds(rows, SEEDS)


def test_collapse_seeds_stops_on_a_duplicate_row():
    rows = _one_pair()
    with pytest.raises(TablesStop, match="duplicate"):
        collapse_seeds(rows + [dict(rows[4])], SEEDS)


def test_collapse_seeds_stops_on_a_seed_it_was_not_told_of():
    with pytest.raises(TablesStop, match="seed 2"):
        collapse_seeds(_one_pair(), (0, 1))


def test_collapse_seeds_sums_the_failure_count_and_keeps_each_seed():
    rows = pair_rows(*FAILING[:1], 1, FAILING[1], FAILING[2])
    collapsed = {r["region"]: r for r in collapse_seeds(rows, SEEDS)}
    for region, expected in (("all", 2), ("boundary", 2), ("interior", 0)):
        record = collapsed[region]
        assert record["n_predict_nonfinite"] == expected
        assert (record["n_predict_nonfinite_seed0"], record["n_predict_nonfinite_seed1"],
                record["n_predict_nonfinite_seed2"]) == (0, expected, 0)


def test_collapse_seeds_stops_when_a_predictor_field_is_finite_for_some_seeds_only():
    rows = _one_pair()
    [r for r in rows if r["seed"] == 0 and r["region"] == "all"][0]["predict_raw"] = NAN
    with pytest.raises(TablesStop, match="predict_raw"):
        collapse_seeds(rows, SEEDS)


def test_collapse_seeds_emits_python_types():
    """A numpy integer fails the isinstance filters of the estimand layer and
    silently empties a cell, so the records carry Python types only."""
    rows = _one_pair()
    for row in rows:
        row["n_primary"] = np.int64(row["n_primary"])
        row["cl_raw"] = np.float64(row["cl_raw"])
        row["predict_raw"] = np.float32(row["predict_raw"])
        row["fold"] = np.int64(row["fold"])
    for record in collapse_seeds(rows, SEEDS):
        for key, value in record.items():
            assert type(value) in (int, float, str, bool, type(None)), (key, type(value))
    for record in select_seed(rows, 1):
        for key, value in record.items():
            assert type(value) in (int, float, str, bool, type(None)), (key, type(value))


def test_every_record_field_is_classified_by_seed():
    """The predictor's columns and its failure count depend on the seed. Every
    other column the evaluation writes is explicit and must not."""
    dependent = {field for field in ALL_FIELDS if is_seed_dependent(field)}
    assert dependent == {
        field for field in ALL_FIELDS
        if field.startswith(("predict_", "sp_predict_", "x_predict_", "x_sp_predict_"))
    } | {"n_predict_nonfinite"}
    for field in ("cl_raw", "tl_form_centered", "sp_transport_raw", "x_cl_centered",
                  "offset_cl_oracle_raw_b0", "n_primary", "rotation_deg", "seed"):
        assert not is_seed_dependent(field), field


def test_pooling_seed_rows_and_collapsing_them_agree_but_count_pairs_differently():
    """Pooling the three seed rows of every pair gives the collapsed estimate
    and interval, to float rounding, because every formula is linear in field
    means. But support counts pairs as records, so pooled seed rows would
    triple n_camera_pairs. That is why seeds are collapsed first."""
    rows = []
    for fold, scene in enumerate(SCENES):
        rows += [r for r in scene_rows(scene, fold)[0] if r["region"] == "all"]
    collapsed = collapse_seeds(rows, SEEDS)
    for quantity in ("delta_learn_pp", "predict_margin", "delta_learn_sp", "sp_predict"):
        for metric in ("centered", "raw"):
            pooled = evaluate_quantity(rows, quantity, metric, ANALYSIS)
            single = evaluate_quantity(collapsed, quantity, metric, ANALYSIS)
            assert single.estimate == pytest.approx(pooled.estimate, abs=1e-12)
            assert single.lo == pytest.approx(pooled.lo, abs=1e-12)
            assert single.hi == pytest.approx(pooled.hi, abs=1e-12)
            assert single.n_replicates == pooled.n_replicates
            assert pooled.n_camera_pairs == 3 * single.n_camera_pairs
            assert pooled.n_scenes == single.n_scenes


# ---------------------------------------------------------------------------
# Each scene's records are checked before any cell is computed
# ---------------------------------------------------------------------------

def _scene(scene=SCENES[0], fold=0):
    rows, audit = scene_rows(scene, fold)
    return rows, audit


def test_the_tables_refuse_records_that_were_not_collapsed():
    """Raw seed rows would count each pair three times as camera pairs, so a
    cell built on them would overstate its support. The builder refuses them."""
    prepared = prepare()
    raw = [r for r in scene_rows(SCENES[0], 0)[0]]
    for row in raw:
        row["camera_pair"] = camera_pair_key(row)
    broken = dataclasses.replace(prepared, collapsed=prepared.collapsed + raw[:3])
    with pytest.raises(TablesStop, match="once"):
        build_tables(broken, FAST)
    doubled = dataclasses.replace(
        prepared, per_seed={**prepared.per_seed, 1: prepared.per_seed[1] * 2})
    with pytest.raises(TablesStop, match="once"):
        build_tables(doubled, FAST)


def test_per_seed_records_that_do_not_pool_to_the_collapsed_records_stop():
    """Seed 1's predictor scores move after the seeds were collapsed. The mean of
    the per-seed estimates is then not the collapsed estimate, and the tables
    stop, reporting_rules.md section 6."""
    prepared = prepare()
    shifted = [{**record, "predict_centered": record["predict_centered"] + 0.01,
                "predict_raw": record["predict_raw"] + 0.01}
               for record in prepared.per_seed[1]]
    broken = dataclasses.replace(prepared, per_seed={**prepared.per_seed, 1: shifted})
    with pytest.raises(TablesStop, match="Collapsing seeds and pooling them must agree"):
        build_tables(broken, FAST)


@pytest.mark.parametrize("second", [1.5, NAN], ids=["float", "nan"])
def test_a_column_that_mixes_int_and_float_stops(second):
    """pyarrow would widen the integer to a float in silence, so the write
    refuses the column instead."""
    rows = [{"cell": "a", "count": 3}, {"cell": "b", "count": second}]
    with pytest.raises(TablesStop, match="count mixes value types"):
        report._rectangular(rows)


def test_a_complete_scene_passes_its_checks():
    rows, audit = _scene()
    check_scene_rows(rows, scene=SCENES[0], level=LEVEL, fold=0, seeds=SEEDS, audit=audit)
    collapsed = collapse_seeds(rows, SEEDS)
    check_region_partition(collapsed)
    check_population_finiteness(collapsed)
    check_regime_geometry([r for r in collapsed if r["region"] == "all"], ANALYSIS)


@pytest.mark.parametrize("change, match", [
    (lambda audit: audit.update(evaluated=audit["evaluated"] + 1), "evaluated"),
    (lambda audit: audit.update(pairs=audit["pairs"] + 1), "no_arm"),
    (lambda audit: audit.update(pairs=audit["pairs"] + 1, no_arm=1,
                                no_arm_pairs=[["c", "t", "rotation"]]), "level image"),
])
def test_a_scene_whose_audit_does_not_match_its_rows_stops(change, match):
    rows, audit = _scene()
    change(audit)
    with pytest.raises(TablesStop, match=match):
        check_scene_rows(rows, scene=SCENES[0], level=LEVEL, fold=0, seeds=SEEDS, audit=audit)


@pytest.mark.parametrize("field, value, match", [
    ("scene", SCENES[1], "scene"),
    ("level", "affine", "level"),
    ("fold", 2, "fold"),
    ("region", "edges", "region"),
])
def test_a_row_that_does_not_belong_to_its_scene_stops(field, value, match):
    rows, audit = _scene()
    rows[7][field] = value
    with pytest.raises(TablesStop, match=match):
        check_scene_rows(rows, scene=SCENES[0], level=LEVEL, fold=0, seeds=SEEDS, audit=audit)


@pytest.mark.parametrize("count", ["n_primary", "n_splat", "n_intersect",
                                   "n_predict_nonfinite_seed1"])
def test_a_region_split_that_does_not_sum_to_the_whole_stops(count):
    collapsed = collapse_seeds(_one_pair(), SEEDS)
    [r for r in collapsed if r["region"] == "interior"][0][count] += 1
    with pytest.raises(TablesStop, match=count):
        check_region_partition(collapsed)


def test_a_region_row_with_formulation_or_offset_cells_stops():
    collapsed = collapse_seeds(_one_pair(), SEEDS)
    [r for r in collapsed if r["region"] == "boundary"][0]["offset_n_b1"] = 3
    with pytest.raises(TablesStop, match="offset_n_b1"):
        check_region_partition(collapsed)


@pytest.mark.parametrize("field", ["cl_raw", "predict_centered", "tl_form_centered",
                                   "sp_nowarp_raw", "x_sp_predict_l2_raw",
                                   "offset_nowarp_centered_b3"])
def test_a_nonfinite_score_on_its_own_population_stops(field):
    """A NaN on a counted record would silently unpair a difference."""
    collapsed = collapse_seeds(_one_pair(), SEEDS)
    collapsed[0][field] = NAN
    with pytest.raises(TablesStop, match=field):
        check_population_finiteness(collapsed)


def test_more_failures_than_samples_stops():
    collapsed = collapse_seeds(_one_pair(), SEEDS)
    collapsed[0]["n_predict_nonfinite_seed0"] = collapsed[0]["n_primary"] + 1
    with pytest.raises(TablesStop, match="n_predict_nonfinite"):
        check_population_finiteness(collapsed)


def test_offset_bins_that_do_not_partition_the_whole_support_stop():
    collapsed = collapse_seeds(_one_pair(), SEEDS)
    collapsed[0]["offset_n_all"] += 1
    with pytest.raises(TablesStop, match="offset_n_all"):
        check_population_finiteness(collapsed)


@pytest.mark.parametrize("regime, rotation, parallax, match", [
    # A rotation pair with a baseline would put parallax into the rotation curve.
    ("rotation", 15.0, 0.01, "rotation"),
    # A translation pair that rotates would put rotation into the parallax curve.
    ("translation", 1.0, 0.1, "translation"),
    # PROTOCOL 3.4's design floor: (0, 0.025) is empty for translation pairs.
    ("translation", 0.0, 0.01, "0.025"),
    ("orbit", -1.0, 0.1, "rotation_deg"),
])
def test_a_pair_outside_its_regime_definition_stops(regime, rotation, parallax, match):
    record = {"scene": SCENES[0], "camera_pair": "x", "regime": regime,
              "rotation_deg": rotation, "parallax": parallax}
    with pytest.raises(TablesStop, match=match):
        check_regime_geometry([record], ANALYSIS)


def test_orbit_pairs_may_sit_below_the_translation_floor():
    check_regime_geometry([{"scene": "s", "camera_pair": "x", "regime": "orbit",
                            "rotation_deg": 5.0, "parallax": 0.01}], ANALYSIS)


# ---------------------------------------------------------------------------
# Reporting cells: right-closed bins, orbit in joint cells only
# ---------------------------------------------------------------------------

def _record(regime, rotation, parallax, **counts):
    return {"scene": "s", "camera_pair": f"{regime}|{rotation}|{parallax}", "regime": regime,
            "rotation_deg": rotation, "parallax": parallax, "n_primary": 1,
            "n_formulation": 0, "n_splat": 0, "n_intersect": 0, "offset_n_all": 0, **counts}


@pytest.mark.parametrize("regime, rotation, parallax, axis, label", [
    ("rotation", 10.0, 0.0, "rotation_bin", "0-10"),
    ("rotation", 10.000001, 0.0, "rotation_bin", "10-20"),
    ("rotation", 50.0, 0.0, "rotation_bin", "40-50"),
    ("rotation", 50.5, 0.0, "rotation_bin", "50+"),
    ("rotation", 0.0, 0.0, "rotation_bin", "zero"),
    ("translation", 0.0, 0.05, "parallax_bin", "0.025-0.05"),
    ("translation", 0.0, 0.0500001, "parallax_bin", "0.05-0.1"),
    ("translation", 0.0, 0.4, "parallax_bin", "0.2-0.4"),
    ("translation", 0.0, 0.41, "parallax_bin", "0.4+"),
    ("orbit", 20.0, 0.025, "rotation_bin x parallax_bin", "10-20 x 0-0.025"),
    ("orbit", 25.0, 0.3, "rotation_bin x parallax_bin", "20-30 x 0.2-0.4"),
])
def test_bins_are_closed_on_the_right_at_the_frozen_edges(regime, rotation, parallax,
                                                         axis, label):
    """PROTOCOL 3.4: a value equal to an edge belongs to the lower bin."""
    cells, _ = reporting_cells([_record(regime, rotation, parallax)], ANALYSIS)
    keys = [key for key, _ in cells]
    assert [(k.analysis, k.axis, k.bin) for k in keys] == [
        ("pooled", "all", "all"), (regime, "all", "all"), (regime, axis, label)]
    assert keys[0].summary and not any(k.summary for k in keys[1:])


def test_bins_take_their_order_from_the_frozen_edges():
    records = [_record("rotation", angle, 0.0) for angle in (55.0, 5.0, 25.0)]
    records += [_record("translation", 0.0, p) for p in (0.3, 0.03)]
    records += [_record("orbit", 25.0, 0.3), _record("orbit", 5.0, 0.01)]
    cells, _ = reporting_cells(records, ANALYSIS)
    names = [(k.analysis, k.axis, k.bin) for k, _ in cells]
    assert names == [
        ("pooled", "all", "all"), ("rotation", "all", "all"), ("translation", "all", "all"),
        ("orbit", "all", "all"),
        ("rotation", "rotation_bin", "0-10"), ("rotation", "rotation_bin", "20-30"),
        ("rotation", "rotation_bin", "50+"),
        ("translation", "parallax_bin", "0.025-0.05"),
        ("translation", "parallax_bin", "0.2-0.4"),
        ("orbit", "rotation_bin x parallax_bin", "0-10 x 0-0.025"),
        ("orbit", "rotation_bin x parallax_bin", "20-30 x 0.2-0.4"),
    ]
    joint = [k for k, _ in cells if k.analysis == "orbit" and k.axis != "all"]
    assert [(k.rotation_bin, k.parallax_bin) for k in joint] == [
        ("0-10", "0-0.025"), ("20-30", "0.2-0.4")]


def test_orbit_is_never_marginalized(built):
    """PROTOCOL 3.3: orbit appears in its own regime row, the pooled summary,
    and the joint cells, and never on the rotation or parallax curve."""
    prepared = prepare()
    records = [r for r in prepared.collapsed if r["region"] == "all"]
    cells, _ = reporting_cells(records, FAST)
    for key, members in cells:
        regimes = {r["regime"] for r in members}
        if key.axis == "rotation_bin":
            assert regimes == {"rotation"}, key
        elif key.axis == "parallax_bin":
            assert regimes == {"translation"}, key
        elif key.axis == "rotation_bin x parallax_bin":
            assert regimes == {"orbit"}, key
        elif key.analysis == "pooled":
            assert regimes == set(REGIMES)
        else:
            assert regimes == {key.analysis}, key
    for name in ("phase5_primary.parquet", "phase5_splat_pool.parquet",
                 "phase5_formulation.parquet", "phase5_landing_offset.parquet"):
        for row in rows_of(built, name):
            if row["axis"] in ("rotation_bin", "parallax_bin"):
                assert row["analysis"] == {"rotation_bin": "rotation",
                                           "parallax_bin": "translation"}[row["axis"]]
            if row["analysis"] == "orbit":
                assert row["axis"] in ("all", "rotation_bin x parallax_bin")


def test_an_unbinnable_pair_is_accounted_and_stays_in_its_regime_row(built):
    """A translation pair with no co-visible point has no parallax. It cannot
    be binned, it holds no sample, and it is counted rather than dropped."""
    prepared = prepare()
    records = [r for r in prepared.collapsed if r["region"] == "all"]
    cells, accounting = reporting_cells(records, FAST)
    assert accounting["unbinnable"] == [{
        "scene": SCENES[0], "camera_pair": f"{SCENES[0]}|translation_99_c|translation_99_t",
        "regime": "translation", "rotation_deg": 0.0, "parallax": None,
        "reason": "parallax is not finite",
    }]
    sizes = {(k.analysis, k.axis, k.bin): len(m) for k, m in cells}
    assert sizes[("translation", "all", "all")] == 3 * PER_REGIME + 1
    assert sizes[("pooled", "all", "all")] == 9 * PER_REGIME + 1
    binned = sum(n for (a, axis, _), n in sizes.items()
                 if a == "translation" and axis == "parallax_bin")
    assert binned == 3 * PER_REGIME
    primary = by_cell(rows_of(built, "phase5_primary.parquet"))
    row = primary[("centered", "translation", "all", "all")]
    assert row["n_pairs_in_cell"] == 3 * PER_REGIME + 1
    assert row["n_camera_pairs"] == 3 * PER_REGIME
    assert built.accounting["unbinnable_pairs"]["count"] == 1


def test_an_unbinnable_pair_that_holds_samples_stops():
    record = _record("translation", 0.0, NAN, n_primary=12)
    with pytest.raises(TablesStop, match="parallax"):
        reporting_cells([record], ANALYSIS)


# ---------------------------------------------------------------------------
# The headline table and the outcomes
# ---------------------------------------------------------------------------

def test_the_regime_rows_are_supported_and_the_bins_are_not(built):
    primary = rows_of(built, "phase5_primary.parquet")
    for row in primary:
        scope = row["axis"] == "all"
        assert row["supported"] is scope, (row["analysis"], row["axis"], row["bin"])
        assert row["visibility_bucket"] == "co-visible"
        assert row["region"] == "all" and row["level"] == LEVEL
    pooled = by_cell(primary)[("centered", "pooled", "all", "all")]
    assert pooled["row_label"] == POOLED_LABEL and pooled["summary"] is True
    assert (pooled["n_scenes"], pooled["n_camera_pairs"]) == (3, 9 * PER_REGIME)
    assert pooled["n_feature_comparisons"] == 9 * PER_REGIME * 200
    assert pooled["n_folds"] == 3
    assert [r["metric"] for r in primary[: len(primary) // 2]] == (
        ["centered"] * (len(primary) // 2))


def test_the_outcome_columns(built):
    primary = by_cell(rows_of(built, "phase5_primary.parquet"))

    def row(metric, scope):
        return primary[(metric, scope, "all", "all")]

    rotation, translation, orbit = (row("centered", s) for s in REGIMES)
    assert rotation["delta_learn_pp"] == pytest.approx(0.05, abs=1e-3)
    assert rotation["delta_learn_pp_outcome"] == "46"
    assert rotation["delta_learn_pp_outcome_wording"] == OUTCOME_WORDING["46"]
    assert rotation["delta_learn_sp_outcome"] == "46" and rotation["outcome_49"] is False
    # Translation: per-point explicit wins, the splat gap is no measurable gap.
    assert translation["delta_learn_pp_outcome"] == "46"
    assert translation["delta_learn_sp_outcome"] == "47"
    assert translation["outcome_49"] is True
    # A large headline gap is not engaged, whatever its splat term does.
    assert translation["delta_learn_pp_near_zero"] is False
    assert translation["delta_learn_pp_qualifier"] is None
    assert orbit["delta_learn_pp_outcome"] == "48"
    assert orbit["delta_learn_pp_outcome_wording"] == OUTCOME_WORDING["48"]
    assert orbit["outcome_49"] is False

    # Raw cosine: rotation turns over, so the cell is metric-sensitive.
    assert row("raw", "rotation")["delta_learn_pp_outcome"] == "48"
    assert row("raw", "rotation")["metric_sensitive"] is True
    assert row("centered", "rotation")["metric_sensitive"] is True
    assert row("centered", "translation")["metric_sensitive"] is False
    # The pooled raw gap sits in the band: its wording travels as a qualifier
    # and the outcome is still called from its own interval.
    pooled_raw = row("raw", "pooled")
    assert abs(pooled_raw["delta_learn_pp"]) <= FAST.path_agreement_tolerance
    assert pooled_raw["delta_learn_pp_near_zero"] is True
    assert pooled_raw["delta_learn_pp_qualifier"] == pooled_raw["delta_learn_pp_near_zero_wording"]
    assert pooled_raw["delta_learn_pp_outcome"] == "48"

    # Unsupported cells are shown with their counts and are not classified.
    for (metric, analysis, axis, _), cell in primary.items():
        if axis == "all":
            continue
        assert cell["delta_learn_pp_outcome"] is None
        assert cell["delta_learn_pp_outcome_wording"] is None
        assert cell["delta_learn_pp_qualifier"] is None
        assert cell["outcome_49"] is None and cell["metric_sensitive"] is None
        assert cell["n_camera_pairs"] > 0
        assert math.isfinite(cell["delta_learn_pp"])
    splat = by_cell(rows_of(built, "phase5_splat_pool.parquet"))
    assert splat[("centered", "translation", "all", "all")]["delta_learn_sp_outcome"] == "47"
    assert "delta_learn_sp_outcome_wording" not in splat[("centered", "translation", "all", "all")]


def test_the_measured_outcome_is_the_headline_tables_scope_rows(built):
    measured = rows_of(built, "phase5_measured_outcome.parquet")
    assert len(measured) == 8
    primary = by_cell(rows_of(built, "phase5_primary.parquet"))
    for row in measured:
        cell = primary[(row["metric"], row["scope"], "all", "all")]
        assert row["outcome"] == cell["delta_learn_pp_outcome"]
        assert row["qualifier"] == cell["delta_learn_pp_qualifier"]
        assert row["outcome_49"] == cell["outcome_49"]
        assert row["metric_sensitive"] == cell["metric_sensitive"]
        assert row["delta_learn_pp"] == cell["delta_learn_pp"]
        assert row["lead_within_read"] == cell["lead_within_read"]
        assert row["level"] == LEVEL
        # reporting_rules.md section 6: every row names its visibility bucket,
        # and the measured outcome's rows are the headline's scope rows.
        assert row["visibility_bucket"] == cell["visibility_bucket"] == "co-visible"
        assert (row["axis"], row["bin"]) == (cell["axis"], cell["bin"]) == ("all", "all")
    assert {r["scope"]: r["summary"] for r in measured} == {
        "rotation": False, "translation": False, "orbit": False, "pooled": True}


def test_mean_feature_is_reported_under_raw_cosine_only(built):
    """PROTOCOL 3.7: Mean-Feature predicts the centering vector, so its
    centered score is not applicable. Centered rows say so and hold no number."""
    cells = {
        "phase5_primary.parquet": ["mean_feature"],
        "phase5_formulation.parquet": ["mean_feature_form"],
        "phase5_splat_pool.parquet": ["sp_mean_feature"],
        "phase5_cross_path.parquet": ["x_mean_feature", "x_sp_mean_feature"],
        "phase5_landing_offset.parquet": ["meanfeat_offset"],
    }
    for name, quantities in cells.items():
        for row in rows_of(built, name):
            for quantity in quantities:
                if name == "phase5_landing_offset.parquet" and row["offset_label"] == "deficit":
                    continue
                if row["metric"] == "centered":
                    assert math.isnan(row[quantity]), (name, quantity)
                    assert math.isnan(row[f"{quantity}_ci_low"])
                    assert row[f"{quantity}_ci_replicates"] == 0
                    assert row[f"{quantity}_status"] == report.MEAN_FEATURE_NOT_APPLICABLE
                else:
                    assert row[f"{quantity}_status"] == report.MEAN_FEATURE_REPORTED
                    if row["supported"]:
                        assert math.isfinite(row[quantity]), (name, quantity)
    for row in rows_of(built, "phase5_regions.parquet"):
        if row["row_kind"] == "region":
            name = "mean_feature" if row["path"] == "per_point" else "sp_mean_feature"
            assert math.isnan(row[name]) is (row["metric"] == "centered")
    l2 = [r for r in rows_of(built, "phase5_l2.parquet") if r["population"] == "per_point"]
    for row in l2:
        assert math.isnan(row["l2_mean_feature"]) is (row["metric"] == "l2_centered")
    assert "not applicable" in report.MEAN_FEATURE_NOT_APPLICABLE


def test_the_landing_offset_cells_mirror_the_headline_cells(built):
    primary = {(r["metric"], r["analysis"], r["axis"], r["bin"])
               for r in rows_of(built, "phase5_primary.parquet")}
    offsets = rows_of(built, "phase5_landing_offset.parquet")
    labels = OFFSET_BINS + (OFFSET_WHOLE, "deficit")
    for label in labels:
        mirrored = {(r["metric"], r["analysis"], r["axis"], r["bin"])
                    for r in offsets if r["offset_label"] == label}
        assert mirrored == primary, label
    bounds = {r["offset_label"]: (r["offset_lo"], r["offset_hi"]) for r in offsets}
    assert bounds["b0"] == (0.0, 0.1) and bounds["b4"] == (0.4, 0.5)
    assert bounds["b5"] == (0.5, pytest.approx(math.sqrt(2) / 2))
    assert bounds[OFFSET_WHOLE] == (0.0, pytest.approx(math.sqrt(2) / 2))
    rotation = by_cell(offsets, offset_label="b0")[("centered", "rotation", "all", "all")]
    assert rotation["cl_oracle_offset"] == pytest.approx(0.70 + DEFICIT + 0.02 / 3, abs=1e-12)
    assert rotation["n_feature_comparisons"] == 3 * PER_REGIME * OFFSET_COUNTS[0]
    empty = by_cell(offsets, offset_label="b5")[("centered", "rotation", "all", "all")]
    assert empty["n_camera_pairs"] == 0 and empty["supported"] is False
    assert math.isnan(empty["cl_oracle_offset"])
    deficit = by_cell(offsets, offset_label="deficit")[("raw", "pooled", "all", "all")]
    assert deficit["read_deficit"] == pytest.approx(DEFICIT, abs=1e-12)
    for row in offsets:
        assert "near_zero" not in " ".join(row)
        assert row["table_label"] == report.TABLE_LABELS["phase5_landing_offset.parquet"]


def test_the_read_deficit_and_its_flags_sit_beside_the_headline(built):
    primary = by_cell(rows_of(built, "phase5_primary.parquet"))
    rotation_raw = primary[("raw", "rotation", "all", "all")]
    assert rotation_raw["read_deficit"] == pytest.approx(DEFICIT, abs=1e-12)
    assert rotation_raw["read_deficit_supported"] is True
    assert rotation_raw["read_deficit_n_camera_pairs"] == 3 * PER_REGIME
    # Predict-with-Depth leads by 0.0035, inside the deficit's 0.004.
    assert rotation_raw["lead_within_read"] is True
    assert rotation_raw["cl_lead_understated"] is False
    rotation = primary[("centered", "rotation", "all", "all")]
    assert rotation["lead_within_read"] is None and rotation["cl_lead_understated"] is True
    assert primary[("centered", "orbit", "all", "all")]["lead_within_read"] is False
    for row in primary.values():
        assert row["read_deficit_anomaly"] in (False, None)
        assert row["read_depth_note"] == report.READ_DEPTH_NOTE
        assert "read_deficit_comparison_weighted" not in row
        if not row["supported"]:
            assert row["lead_within_read"] is None
            assert row["cl_lead_understated"] is None
            assert row["read_deficit_anomaly"] is None


def test_seed_spread_has_its_own_columns(built):
    primary = by_cell(rows_of(built, "phase5_primary.parquet"))
    for row in primary.values():
        for quantity in ("predict_with_depth", "predict_margin", "delta_learn_pp"):
            seeds = [row[f"{quantity}_seed{s}"] for s in SEEDS]
            assert row[f"{quantity}_seed_n"] == 3
            assert row[f"{quantity}_seed_mean"] == pytest.approx(row[quantity], abs=1e-9)
            assert row[f"{quantity}_seed_range"] == pytest.approx(2 * SEED_SHIFT, abs=1e-9)
            assert row[f"{quantity}_seed_min"] == min(seeds)
            assert row[f"{quantity}_seed_max"] == max(seeds)
        assert "predict_with_depth_seed_crosses_zero" not in row
    pooled_raw = primary[("raw", "pooled", "all", "all")]
    assert pooled_raw["delta_learn_pp_seed_crosses_zero"] is True
    assert pooled_raw["delta_learn_pp_seed_enters_band"] is True
    rotation = primary[("centered", "rotation", "all", "all")]
    assert rotation["delta_learn_pp_seed_crosses_zero"] is False
    assert rotation["delta_learn_pp_seed_enters_band"] is False
    # The seed spread never widens the scene interval: the interval is the
    # collapsed records' own, which the seed columns sit beside.
    assert rotation["delta_learn_pp_ci_high"] - rotation["delta_learn_pp_ci_low"] < 0.004
    translation = primary[("centered", "translation", "all", "all")]
    assert (translation["n_predict_nonfinite_seed0"], translation["n_predict_nonfinite_seed1"],
            translation["n_predict_nonfinite_seed2"],
            translation["n_predict_nonfinite_total"]) == (0, 2, 0, 2)
    assert primary[("centered", "rotation", "all", "all")]["n_predict_nonfinite_total"] == 0
    splat = by_cell(rows_of(built, "phase5_splat_pool.parquet"))
    row = splat[("centered", "rotation", "all", "all")]
    assert row["delta_learn_sp_seed_mean"] == pytest.approx(row["delta_learn_sp"], abs=1e-9)


def test_the_per_seed_table_holds_composite_estimates_per_seed_index(built):
    per_seed = rows_of(built, "phase5_per_seed.parquet")
    primary = by_cell(rows_of(built, "phase5_primary.parquet"))
    cells = len(primary)
    assert len(per_seed) == cells * 3 * 6
    for row in per_seed:
        assert row["note"] == report.PER_SEED_NOTE
        assert row["n_scenes"] == primary[(row["metric"], row["analysis"], row["axis"],
                                           row["bin"])]["n_scenes"]
        if row["quantity"] in ("predict_with_depth", "predict_margin", "delta_learn_pp"):
            cell = primary[(row["metric"], row["analysis"], row["axis"], row["bin"])]
            assert row["estimate"] == cell[f"{row['quantity']}_seed{row['seed']}"]
            assert row["population"] == "per_point"
        else:
            assert row["population"] == "splat_pool"
    rotation = [r for r in per_seed if (r["metric"], r["analysis"], r["axis"],
                                        r["quantity"]) == ("centered", "rotation", "all",
                                                           "delta_learn_pp")]
    by_seed = {r["seed"]: r["estimate"] for r in rotation}
    assert by_seed[0] - by_seed[2] == pytest.approx(2 * SEED_SHIFT, abs=1e-9)


def test_every_interval_carries_its_replicate_count(built):
    for name, rows in built.tables.items():
        for row in rows:
            for column in row:
                if column.endswith("_ci_low"):
                    stem = column[: -len("_ci_low")]
                    assert f"{stem}_ci_high" in row, (name, column)
                    assert f"{stem}_ci_replicates" in row, (name, column)


def test_every_scene_interval_has_a_camera_pair_interval_and_a_weighted_value(built):
    for row in rows_of(built, "phase5_primary.parquet"):
        for quantity in ("cl_transport", "predict_with_depth", "no_warp_copy", "cl_margin",
                         "predict_margin", "delta_learn_pp", "read_deficit"):
            assert f"{quantity}_pair_ci_low" in row
            assert f"{quantity}_pair_ci_replicates" in row
        for quantity in ("cl_transport", "delta_learn_pp", "mean_feature"):
            assert f"{quantity}_comparison_weighted" in row
        if row["supported"]:
            assert row["delta_learn_pp_pair_ci_replicates"] == FAST.bootstrap_resamples
            weighted = row["delta_learn_pp_comparison_weighted"]
            assert weighted == pytest.approx(row["delta_learn_pp"], abs=1e-3)
    # A region contrast is a cell too, PROTOCOL 3.4: each of its quantities
    # carries the camera-pair interval and the weighted value beside its own.
    contrasts = [row for row in rows_of(built, "phase5_regions.parquet")
                 if row["row_kind"] == "contrast"]
    assert contrasts
    for row in contrasts:
        for quantity in report.CONTRAST_QUANTITIES[row["path"]]:
            name = f"contrast_{quantity}"
            assert f"{name}_pair_ci_low" in row and f"{name}_pair_ci_replicates" in row
            assert isinstance(row[f"{name}_comparison_weighted"], float), (name, row)


def test_the_disclosure_is_shown_beside_each_interpreted_effect(built):
    """A supported cell shows its near-zero flag and wording beside its terms. A
    cell below support makes no claim, PROTOCOL 3.4, so it shows its terms with
    no flag, and a marker in place of the wording."""
    from lot.phase5_outcomes import WORDING_BELOW_SUPPORT

    supported = unsupported = 0
    for row in rows_of(built, "phase5_primary.parquet"):
        for effect, terms in (
            ("delta_learn_pp", ("x_delta_learn_pp", "x_delta_learn_sp", "path_difference_learn")),
            ("cl_margin", ("x_cl_margin", "x_sp_transport_margin", "path_difference_cl_margin")),
            ("predict_margin", ("x_predict_margin", "x_sp_predict_margin",
                                "path_difference_predict_margin")),
        ):
            if row["supported"]:
                assert isinstance(row[f"{effect}_near_zero"], bool)
                assert row[f"{effect}_near_zero_wording"] not in (None, WORDING_BELOW_SUPPORT)
                supported += 1
            else:
                assert row[f"{effect}_near_zero"] is None
                assert row[f"{effect}_near_zero_wording"] == WORDING_BELOW_SUPPORT
                unsupported += 1
            for term in terms:
                assert f"{term}_ci_replicates" in row
        assert row["cross_path_n_camera_pairs"] >= 0
    assert supported and unsupported


def test_the_near_zero_json_holds_every_interpreted_effect_cell(built):
    from lot.phase5_outcomes import WORDING_BELOW_SUPPORT

    entries = built.near_zero
    primary = rows_of(built, "phase5_primary.parquet")
    splat = rows_of(built, "phase5_splat_pool.parquet")
    formulation = rows_of(built, "phase5_formulation.parquet")
    regions = [r for r in rows_of(built, "phase5_regions.parquet") if r["row_kind"] == "region"]
    assert len(entries) == 3 * len(primary) + 3 * len(splat) + len(formulation) + len(regions)
    licensed = {WORDING_VETO, WORDING_SMALL_SIGN_CONSISTENT, WORDING_PATH_SENSITIVE,
                WORDING_OUTSIDE_ON_COMMON_CELLS, WORDING_ONLY_PATH_CLEAR,
                WORDING_ONLY_PATH_INCLUDES_ZERO}
    for entry in entries:
        assert entry["quantity"] in INTERPRETED_EFFECTS
        assert entry["level"] == LEVEL and entry["band"] == FAST.path_agreement_tolerance
        for term in ("reported", "per_point"):
            assert {"estimate", "lo", "hi", "n_replicates", "n_units"} <= set(entry[term])
        if entry["quantity"] == "delta_formulation":
            assert entry["splat_pool"] is None and entry["cross_path_support"] is None
        else:
            assert entry["splat_pool"]["n_replicates"] >= 0
            assert {"n_scenes", "n_camera_pairs", "n_feature_comparisons", "supported"} == set(
                entry["cross_path_support"])
        if not entry["supported"]:
            # Below support: the terms are kept, and nothing is claimed.
            assert entry["near_zero"] is None and entry["wording"] == WORDING_BELOW_SUPPORT
        elif entry["near_zero"]:
            assert entry["wording"] in licensed
        else:
            assert entry["wording"] in (WORDING_OUTSIDE_BAND, WORDING_NOT_ESTIMABLE)
        assert entry["reported_in_band"] == (
            math.isfinite(entry["reported"]["estimate"])
            and abs(entry["reported"]["estimate"]) <= FAST.path_agreement_tolerance)
    headline = [e for e in entries if e["table"] == "phase5_primary.parquet"
                and e["quantity"] == "delta_learn_pp"]
    rows = by_cell(primary)
    for entry in headline:
        row = rows[(entry["metric"], entry["analysis"], entry["axis"], entry["bin"])]
        assert entry["near_zero"] == row["delta_learn_pp_near_zero"]
        assert entry["wording"] == row["delta_learn_pp_near_zero_wording"]
        assert entry["reported"]["estimate"] == row["delta_learn_pp"]
        assert entry["seed_crosses_zero"] == row["delta_learn_pp_seed_crosses_zero"]
    assert "equivalen" not in json.dumps(entries).lower()


def test_a_cell_below_support_carries_no_near_zero_wording(monkeypatch):
    """PROTOCOL 3.4 keeps a cell below support out of every claim, and section 9
    gives it no qualifier.

    Rotation's per-point gap is set to 0.001 and its splat-pool gap to 0.002, so
    both cross-path terms sit inside the band, agree in sign, and clear zero.
    The supported rotation row then engages the small, sign-consistent wording
    as its qualifier. The rotation bins hold three scenes and fewer than thirty
    pairs, so they are below support. Their terms engage the same wording, and
    yet no table row and no disclosure entry may print it."""
    from lot.phase5_outcomes import WORDING_BELOW_SUPPORT

    monkeypatch.setitem(GAP_PP, ("rotation", "centered"), 0.001)
    monkeypatch.setitem(GAP_SP, "rotation", 0.002)
    tables = build(phase4=False)
    primary = by_cell(tables.tables["phase5_primary.parquet"])
    rotation = primary[("centered", "rotation", "all", "all")]
    assert rotation["supported"] is True and rotation["delta_learn_pp_outcome"] == "46"
    assert rotation["delta_learn_pp_near_zero"] is True
    assert rotation["delta_learn_pp_near_zero_wording"] == WORDING_SMALL_SIGN_CONSISTENT
    assert rotation["delta_learn_pp_qualifier"] == WORDING_SMALL_SIGN_CONSISTENT

    effects = {
        "phase5_primary.parquet": lambda row: report.PRIMARY_EFFECTS,
        "phase5_formulation.parquet": lambda row: ("delta_formulation",),
        "phase5_splat_pool.parquet": lambda row: report.SPLAT_EFFECTS,
        "phase5_regions.parquet": lambda row: (report.REGION_GAP[row["path"]],),
    }
    below = 0
    for name, named in effects.items():
        for row in tables.tables[name]:
            if row.get("row_kind") == "contrast" or row["supported"] is not False:
                continue
            for effect in named(row):
                assert row[f"{effect}_near_zero"] is None, (name, effect, row["bin"])
                assert row[f"{effect}_near_zero_wording"] == WORDING_BELOW_SUPPORT
                below += 1
    assert below

    engaged_terms = 0
    for entry in tables.near_zero:
        if entry["supported"]:
            continue
        assert entry["near_zero"] is None and entry["wording"] == WORDING_BELOW_SUPPORT
        assert {"estimate", "lo", "hi", "n_replicates"} <= set(entry["reported"])
        assert {"estimate", "lo", "hi", "n_replicates"} <= set(entry["per_point"])
        if (entry["quantity"] == "delta_learn_pp" and entry["analysis"] == "rotation"
                and entry["metric"] == "centered" and entry["reported_in_band"]
                and entry["paths_agree_in_sign"] and entry["both_intervals_exclude_zero"]):
            engaged_terms += 1
    # The case the rule is for: below support, with terms that would engage.
    assert engaged_terms


def test_the_headline_table_carries_no_target_lift(built):
    """Stream AD: the Phase 4 target-lift score never enters the headline."""
    for row in rows_of(built, "phase5_primary.parquet"):
        for key, value in row.items():
            assert not key.startswith("tl_"), key
            if isinstance(value, str):
                assert TL_REFERENCE not in value and "Target-Lift" not in value, key
        assert row["headline_definition"] == report.HEADLINE_DEFINITION
        assert CL_TRANSPORT in row["headline_definition"]
        assert PREDICT_WITH_DEPTH in row["headline_definition"]
        assert row["delta_learn_pp"] == pytest.approx(
            row["cl_transport"] - row["predict_with_depth"], abs=1e-12)


def test_the_headline_guard_refuses_a_target_lift_headline(monkeypatch):
    """The guard runs on the headline's declared composition every build."""
    monkeypatch.setattr(report, "HEADLINE_METHODS", (TL_REFERENCE, PREDICT_WITH_DEPTH))
    from lot.phase5_estimands import HeadlineSubstitutionError

    with pytest.raises(HeadlineSubstitutionError):
        build(phase4=False)


def test_the_formulation_and_splat_tables_carry_their_labels_and_floors(built):
    formulation = rows_of(built, "phase5_formulation.parquet")
    label = "information/formulation diagnostic, not the learned-versus-explicit estimand"
    for row in formulation:
        assert row["table_label"] == label
        assert {"tl_reference", "cl_on_formulation_support", "no_warp_copy_form",
                "mean_feature_form", "delta_formulation"} <= set(row)
        assert not any(key.endswith(("_seed0", "_outcome")) for key in row)
    pooled = by_cell(formulation)[("centered", "pooled", "all", "all")]
    assert pooled["delta_formulation"] == pytest.approx(0.021, abs=1e-12)
    assert pooled["n_feature_comparisons"] == 9 * PER_REGIME * 80
    splat = rows_of(built, "phase5_splat_pool.parquet")
    for row in splat:
        assert row["table_label"] == "secondary operational target-grid comparison"
        assert {"sp_transport", "sp_predict", "sp_no_warp_copy", "sp_transport_margin",
                "sp_predict_margin", "delta_learn_sp"} <= set(row)
    cross = rows_of(built, "phase5_cross_path.parquet")
    for row in cross:
        assert not any("wording" in key for key in row)
        assert row["path_difference_learn"] == pytest.approx(
            row["x_delta_learn_pp"] - row["x_delta_learn_sp"], abs=1e-12)


def test_region_contrasts_on_synthetic_values(built):
    """Each region differs from the whole by a constant, so each contrast has
    an exact value. The pair with no boundary sample is excluded from the
    per-point boundary contrast but kept in the interior row."""
    regions = rows_of(built, "phase5_regions.parquet")
    expected = {
        ("per_point", "boundary_minus_interior"): {
            "contrast_cl_transport": -0.05, "contrast_predict_with_depth": -0.015,
            "contrast_delta_learn_pp": -0.035},
        ("per_point", "low_minus_high_texture"): {
            "contrast_cl_transport": -0.03, "contrast_predict_with_depth": -0.03,
            "contrast_delta_learn_pp": 0.0},
        ("splat_pool", "boundary_minus_interior"): {
            "contrast_sp_transport": -0.038, "contrast_sp_predict": -0.013,
            "contrast_delta_learn_sp": -0.025},
        ("splat_pool", "low_minus_high_texture"): {
            "contrast_sp_transport": -0.015, "contrast_sp_predict": -0.015,
            "contrast_delta_learn_sp": 0.0},
    }
    contrasts = [r for r in regions if r["row_kind"] == "contrast"]
    assert len(contrasts) == 2 * 4 * 2 * 2
    for row in contrasts:
        for column, value in expected[(row["path"], row["region"])].items():
            assert row[column] == pytest.approx(value, abs=1e-12), (row["region"], column)
            assert row[f"{column}_ci_low"] == pytest.approx(value, abs=1e-12)
            assert row[f"{column}_pair_ci_replicates"] == FAST.bootstrap_resamples
            # Each region's count is the same for every pair it shares, so
            # weighting by counts leaves the exact hand value.
            assert row[f"{column}_comparison_weighted"] == pytest.approx(value, abs=1e-12), (
                row["region"], column)
        assert row["contrast_note"] == report.CONTRAST_NOTE
        assert not any("wording" in key for key in row)
    pooled = {(r["metric"], r["path"], r["region"]): r for r in contrasts
              if r["analysis"] == "pooled"}
    boundary = pooled[("centered", "per_point", "boundary_minus_interior")]
    assert boundary["n_camera_pairs"] == 9 * PER_REGIME - 1
    assert boundary["supported"] is True
    assert boundary["contrast_left"] == "boundary" and boundary["contrast_right"] == "interior"
    assert pooled[("centered", "splat_pool", "boundary_minus_interior")]["n_camera_pairs"] == (
        9 * PER_REGIME)
    assert boundary["in_band"] is False
    assert pooled[("centered", "per_point", "low_minus_high_texture")]["in_band"] is True
    rows = {(r["metric"], r["analysis"], r["path"], r["region"]): r for r in regions
            if r["row_kind"] == "region"}
    assert rows[("centered", "pooled", "per_point", "boundary")]["n_camera_pairs"] == (
        9 * PER_REGIME - 1)
    assert rows[("centered", "pooled", "per_point", "interior")]["n_camera_pairs"] == (
        9 * PER_REGIME)
    interior = rows[("centered", "rotation", "per_point", "interior")]
    assert interior["delta_learn_pp"] == pytest.approx(0.05 + 0.005, abs=1e-3)
    assert isinstance(interior["delta_learn_pp_near_zero"], bool)


def test_the_contrast_weighted_value_weights_each_region_by_its_own_counts():
    """PROTOCOL 3.4's diagnostic for a region contrast. Each side is weighted by
    its own region's counts, over the pairs both regions hold, and the contrast
    is left minus right. With unequal counts it differs from the estimate, the
    unweighted mean over pairs, and matches a hand computation."""
    from lot.phase5_estimands import PER_POINT, pivot_regions

    def record(pair, region, n, cl, predict):
        return {"scene": "s", "camera_pair": pair, "region": region, "n_primary": n,
                "cl_centered": cl, "predict_centered": predict}

    records = [
        record("p1", "boundary", 10, 0.5, 0.4), record("p1", "interior", 30, 0.7, 0.5),
        record("p2", "boundary", 30, 0.6, 0.6), record("p2", "interior", 10, 0.9, 0.6),
        # A pair with no boundary sample is not in the contrast at all.
        record("p3", "boundary", 0, NAN, NAN), record("p3", "interior", 50, 0.1, 0.1),
    ]
    contrast = "boundary_minus_interior"
    pivot = pivot_regions(records, contrast, PER_POINT)
    assert [entry["camera_pair"] for entry in pivot] == ["p1", "p2"]
    # (10 * 0.5 + 30 * 0.6) / 40 minus (30 * 0.7 + 10 * 0.9) / 40.
    weighted = report.contrast_weighted(pivot, contrast, "cl_transport", "centered")
    assert weighted == pytest.approx(0.575 - 0.75, abs=1e-12)
    # The unweighted contrast is (0.5 + 0.6) / 2 minus (0.7 + 0.9) / 2.
    assert weighted != pytest.approx(0.55 - 0.8, abs=1e-3)
    # The gap: Context-Lift 0.575 and Predict-with-Depth 0.55 on the left,
    # 0.75 and 0.525 on the right.
    gap = report.contrast_weighted(pivot, contrast, "delta_learn_pp", "centered")
    assert gap == pytest.approx((0.575 - 0.55) - (0.75 - 0.525), abs=1e-12)
    assert math.isnan(report.contrast_weighted([], contrast, "cl_transport", "centered"))


def test_the_l2_table_holds_every_companion_without_wording(built):
    l2 = rows_of(built, "phase5_l2.parquet")
    primary = rows_of(built, "phase5_primary.parquet")
    assert {r["metric"] for r in l2} == {"l2_raw", "l2_centered"}
    assert len(l2) == len(primary) * 3
    for row in l2:
        assert not any("wording" in key or "near_zero" in key for key in row)
    pooled = {(r["metric"], r["population"]): r for r in l2
              if (r["analysis"], r["axis"]) == ("pooled", "all")}
    row = pooled[("l2_centered", "per_point")]
    centered = by_cell(primary)[("centered", "pooled", "all", "all")]
    # With L2 = 1 - cosine in the synthetic records, the L2 difference named
    # for its sign equals the cosine gap.
    assert row["l2_predict_minus_cl"] == pytest.approx(centered["delta_learn_pp"], abs=1e-12)
    assert row["n_camera_pairs"] == centered["n_camera_pairs"]
    assert pooled[("l2_raw", "splat_pool")]["l2_sp_predict_minus_transport"] == (
        pytest.approx(by_cell(rows_of(built, "phase5_splat_pool.parquet"))[
            ("raw", "pooled", "all", "all")]["delta_learn_sp"], abs=1e-12))
    assert pooled[("l2_raw", "formulation")]["l2_cl_form_minus_tl_form"] == pytest.approx(
        0.021, abs=1e-12)


def test_the_adequacy_and_validation_history_tables(built):
    adequacy = rows_of(built, "phase5_adequacy.parquet")
    assert [(r["fold"], r["seed"]) for r in adequacy] == [
        (f, s) for f in range(3) for s in SEEDS]
    for row in adequacy:
        fold = FOLDS[row["fold"]]
        assert row["train_scenes"] == ",".join(fold.train)
        assert row["val_scenes"] == ",".join(fold.val)
        assert row["test_scenes"] == ",".join(fold.test)
        assert row["tiny_overfit_passed"] is True and row["tiny_overfit_threshold"] == 0.98
        assert row["tiny_overfit_regimes"] == "rotation,translation,orbit"
        assert row["parameter_count"] == 16680960
        assert row["max_steps"] == 20000 and row["validation_every_steps"] == 500
        assert row["pose_label"] is None
        assert row["depth_label"] == ESSENTIALLY_NO_EFFECT
        assert row["depth_n_unchanged"] == 3
        assert row["n_pairs"] == 10 * len(fold.train)
        assert row["n_empty_support"] == len(fold.train)
        assert row["best_validation_note"] == "model-selection statistic, not an estimand"
        assert row["stable_validation_curve"] is ((row["fold"], row["seed"]) != (2, 2))
    history = rows_of(built, "phase5_validation_history.parquet")
    assert len(history) == 8 * 8 + 6
    best = [r for r in history if r["is_best"]]
    assert len(best) == 9


def test_scenes_of_one_fold_that_embed_different_evidence_stop():
    """Every scene of a fold embeds the same training records and controls."""
    records = evidence_records()
    other = json.loads(json.dumps(records[SCENES[0]]))
    other["controls"]["results"][fold_seed_key(0, 0)]["n_pairs"] = 41
    records[FOLDS[0].test[1]] = other
    with pytest.raises(TablesStop, match="controls"):
        adequacy_entries(records, FOLDS, SEEDS)
    records = evidence_records()
    records[SCENES[2]]["overfit_verdict"]["passed"] = False
    with pytest.raises(TablesStop, match="overfit"):
        adequacy_entries(records, FOLDS, SEEDS)


@pytest.mark.parametrize("field, change, match", [
    ("train_scenes_planned", lambda value: value + [FOLDS[1].test[0]], "train_scenes_planned"),
    ("val_scenes", lambda value: value[:-1], "val_scenes"),
    ("seed", lambda value: value + 1, "seed"),
    ("level", lambda value: "affine", "level"),
])
def test_a_training_record_that_does_not_describe_its_run_stops(field, change, match):
    records = evidence_records()
    record = records[SCENES[1]]["training_record_contents"]["2"]
    record[field] = change(record[field])
    entries = adequacy_entries(records, FOLDS, SEEDS)
    with pytest.raises(TablesStop, match=match):
        report.adequacy_table(entries, FAST, level=LEVEL, primary_level=LEVEL,
                              training=TRAINING)


def test_the_reference_table_keeps_the_three_rungs_separate(built):
    references = rows_of(built, "phase5_references.parquet")
    primary = by_cell(rows_of(built, "phase5_primary.parquet"))
    assert len(references) == len(primary)
    for row in references:
        assert row["table_label"] == (
            "historical/reference estimators; distinct from the Phase 5 causal comparator")
        cell = primary[(row["metric"], row["analysis"], row["axis"], row["bin"])]
        assert row["learned_vs_explicit_limitation_estimate"] in (cell["delta_learn_pp"], None)
        matched = row["phase4_matched"]
        if row["axis"] == "all" or row["bin"] == "0-10" and row["analysis"] == "rotation":
            assert matched is True
            assert row["representation_limitation_oracle_estimate"] == 0.70
            assert row["estimated_geometry_limitation_estimate"] == 0.04
            assert row["estimated_geometry_limitation_lo"] == 0.03
            assert row["learned_vs_explicit_limitation_estimate"] == cell["delta_learn_pp"]
            assert row["phase4_metric"] == {"centered": "cosine_centered_mean",
                                            "raw": "cosine_mean"}[row["metric"]]
        else:
            assert matched is False
            assert row["estimated_geometry_limitation_estimate"] is None
        # Nothing is computed across populations.
        assert not any("minus" in key for key in row)
        # The floors sit beside every level the table reports, CLAUDE.md and
        # reporting_rules.md section 6: No-Warp-Copy under both metrics, and
        # Mean-Feature under raw cosine only. They are the headline cell's own.
        assert same(row["no_warp_copy"], cell["no_warp_copy"])
        assert same(row["no_warp_copy_ci_low"], cell["no_warp_copy_ci_low"])
        if row["metric"] == "raw":
            assert same(row["mean_feature"], cell["mean_feature"])
            assert row["mean_feature_status"] == report.MEAN_FEATURE_REPORTED
        else:
            assert math.isnan(row["mean_feature"])
            assert row["mean_feature_status"] == report.MEAN_FEATURE_NOT_APPLICABLE
        # The Phase 4 rung keeps its own floor beside its matched ceiling.
        assert row["phase4_matched_floor"] == (0.60 if matched else None)


def test_building_twice_gives_identical_tables():
    first, second = build(), build()
    assert first.tables.keys() == second.tables.keys()
    for name in first.tables:
        assert same(first.tables[name], second.tables[name]), name
    assert same(first.near_zero, second.near_zero)
    assert same(first.accounting, second.accounting)


def test_no_string_holds_an_em_dash_or_a_letter_label(built):
    strings: list[str] = []

    def walk(value):
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                strings.append(str(key))
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(built.tables)
    walk(built.near_zero)
    walk(built.accounting)
    walk(report.TABLE_LABELS)
    letter = re.compile(r"(?<![A-Za-z0-9_])[ABC](?![A-Za-z0-9_])")
    for text in strings:
        assert "\u2014" not in text, text
        assert not letter.search(text), text
        assert not re.search(r"\bmethod [ABC]\b", text, re.IGNORECASE), text
    for module in ("phase5_report.py", "phase5_outcomes.py", "phase5_estimands.py"):
        source = (REPO / "src" / "lot" / module).read_text(encoding="utf-8")
        assert "\u2014" not in source, module
        assert not re.search(r"\bmethod [ABC]\b", source), module


def test_a_non_primary_level_has_no_landing_offset_cells():
    """The diagnostic is reported once, from the primary level's records."""
    tables = build(level="affine", primary=LEVEL, phase4=False)
    assert "phase5_landing_offset.parquet" not in tables.tables
    assert "phase5_references.parquet" not in tables.tables
    for row in tables.tables["phase5_primary.parquet"]:
        assert row["level"] == "affine"
        assert not any(key.startswith("read_deficit") or key == "lead_within_read"
                       for key in row)
        assert row["landing_offset_status"] == report.LANDING_OFFSET_PRIMARY_ONLY
    for row in tables.tables["phase5_adequacy.parquet"]:
        assert "primary level" in row["tiny_overfit_scope"]


# ---------------------------------------------------------------------------
# The tables mode on disk: licensed by the evaluated run, written once
# ---------------------------------------------------------------------------

from test_phase5_provenance import build_run, eval_path  # noqa: E402


def _stage_synthetic_rows(run, analysis) -> None:
    """Replace each scene's rows with the synthetic records, keeping its run
    record, its scene, and its fold. The record's audit and reporting digest
    are rewritten to match."""
    rows_by_scene = all_scene_rows()
    for scene, (rows, audit) in rows_by_scene.items():
        path = eval_path(run.run_dir, scene)
        meta = read_run_metadata(path)
        meta["audit"] = audit
        meta["analysis_reporting_digest"] = analysis.reporting_digest()
        path.unlink()
        path.with_suffix(".meta.json").unlink()
        write_rows(path, rows, meta)


def _fake_phase4(cfg, analysis, level, identity):
    return phase4_rows(level)


def run_tables(run, **kwargs):
    arguments = {"expected_scenes": list(SCENES), "folds": FOLDS, "check_code": False,
                 "phase4_reference": _fake_phase4, **kwargs}
    return report.run_tables(run.cfg, FAST, run.config, LEVEL, **arguments)


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    patcher = pytest.MonkeyPatch()
    try:
        run = build_run(tmp_path_factory.mktemp("published"), patcher)
        _stage_synthetic_rows(run, FAST)
        outcome = run_tables(run)
        yield run, outcome
    finally:
        patcher.undo()


def test_the_tables_mode_publishes_every_table_with_its_run_record(published):
    run, outcome = published
    final = Path(run.run_dir) / "tables" / LEVEL
    assert outcome["published"] == str(final) and outcome["superseded"] is None
    names = sorted(p.name for p in final.iterdir())
    assert names == sorted(set(report.TABLE_FILES) | {
        "phase5_near_zero.json", "phase5_accounting.json", "MANIFEST.json"})
    assert outcome["written"][-1] == "MANIFEST.json"
    manifest = json.loads((final / "MANIFEST.json").read_text(encoding="utf-8"))
    from lot.phase5_check import sha256_file

    assert set(manifest["files"]) == set(names) - {"MANIFEST.json"}
    for name, sha in manifest["files"].items():
        assert sha256_file(final / name) == sha, name
    assert manifest["run_record"]["evaluation_commit"] == run.commit
    assert manifest["run_record"]["kind"] == "phase5_tables_manifest"
    assert manifest["run_record"]["inputs"]["phase4/eval/x.parquet"] == "9" * 64
    # Every output names what its level is, so a file read on its own says
    # whether it is the primary result. reporting_rules.md decision 4.
    assert manifest["run_record"]["level_role"] == "primary"
    for name in report.TABLE_FILES:
        record = read_run_metadata(final / name)
        assert record["kind"] == "phase5_table" and record["table"] == name
        assert record["evaluation_commit"] == run.commit
        assert record["report_version"] == report.REPORT_VERSION
        assert record["bootstrap"]["resamples"] == FAST.bootstrap_resamples
        assert record["table_label"] == report.TABLE_LABELS[name]
        assert record["level_role"] == "primary"
    near_zero = json.loads((final / "phase5_near_zero.json").read_text(encoding="utf-8"))
    assert near_zero["run_record"]["kind"] == "phase5_near_zero"
    assert near_zero["run_record"]["level_role"] == "primary"
    assert near_zero["entries"]
    accounting = json.loads((final / "phase5_accounting.json").read_text(encoding="utf-8"))
    assert accounting["run_record"]["level_role"] == "primary"
    assert accounting["totals"]["evaluated"] == sum(
        audit["evaluated"] for _, audit in all_scene_rows().values())
    assert not [p for p in final.parent.iterdir() if ".partial." in p.name]


def test_the_published_tables_are_the_tables_built_in_memory(published):
    run, _ = published
    final = Path(run.run_dir) / "tables" / LEVEL
    memory = build()
    for name, rows in memory.tables.items():
        if name in ("phase5_adequacy.parquet", "phase5_validation_history.parquet"):
            # Built from the run records' embedded evidence, which the disk run
            # writes differently from evidence_records.
            continue
        on_disk = read_rows(final / name)
        assert len(on_disk) == len(rows), name
        for written, built_row in zip(on_disk, rows):
            for key, value in built_row.items():
                assert same(written[key], value), (name, key)


def test_an_existing_output_is_refused_before_any_work(published, monkeypatch):
    run, _ = published
    monkeypatch.setattr(report, "require_evaluated_run",
                        lambda *a, **k: pytest.fail("the run was read"))
    with pytest.raises(FileExistsError, match="written once"):
        run_tables(run)


def test_supersede_moves_the_earlier_output_aside_and_deletes_nothing(tmp_path, monkeypatch,
                                                                    built):
    run = build_run(tmp_path, monkeypatch)
    _stage_synthetic_rows(run, FAST)
    monkeypatch.setattr(report, "build_tables", lambda *a, **k: built)
    first = run_tables(run)
    final = Path(first["published"])
    before = {p.name: p.read_bytes() for p in final.iterdir()}
    second = run_tables(run, supersede=True)
    archived = Path(second["superseded"])
    assert archived.name == f"{LEVEL}.superseded.1"
    assert {p.name: p.read_bytes() for p in archived.iterdir()} == before
    assert sorted(p.name for p in final.iterdir()) == sorted(before)
    for name in report.TABLE_FILES:
        assert same(read_rows(final / name), read_rows(archived / name)), name


def test_a_failed_write_leaves_no_output_and_no_staging(tmp_path, monkeypatch, built):
    run = build_run(tmp_path, monkeypatch)
    _stage_synthetic_rows(run, FAST)
    monkeypatch.setattr(report, "build_tables", lambda *a, **k: built)
    monkeypatch.setattr(report, "_write_json", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("disk full")))
    with pytest.raises(RuntimeError, match="disk full"):
        run_tables(run)
    tables = Path(run.run_dir) / "tables"
    assert not tables.exists() or list(tables.iterdir()) == []


def test_a_non_primary_level_requires_the_primary_tables(tmp_path, monkeypatch):
    run = build_run(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="primary"):
        report.run_tables(run.cfg, FAST, run.config, "affine", expected_scenes=list(SCENES),
                          folds=FOLDS, check_code=False, phase4_reference=_fake_phase4)


def test_a_primary_table_changed_after_publishing_is_refused(published, tmp_path):
    run, _ = published
    copy = tmp_path / "tables"
    shutil.copytree(Path(run.run_dir) / "tables" / LEVEL, copy / LEVEL)
    (copy / LEVEL / "phase5_primary.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="phase5_primary.parquet"):
        report.require_primary_tables(copy / LEVEL, primary_level=LEVEL)
    verified = report.require_primary_tables(Path(run.run_dir) / "tables" / LEVEL,
                                             primary_level=LEVEL)
    assert verified.run_record["evaluation_commit"] == run.commit
    assert verified.run_record["level"] == LEVEL


def test_primary_tables_of_another_kind_layout_or_level_are_refused(published, tmp_path):
    """The primary tables are read as the figures mode reads them, so a manifest
    that only lists hashes, one of another kind, one without its run record, or
    tables of another level, never license a non-primary level."""
    run, _ = published
    copy = tmp_path / "tables" / LEVEL
    shutil.copytree(Path(run.run_dir) / "tables" / LEVEL, copy)
    manifest_path = copy / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for broken in ({}, {"files": manifest["files"]}, {**manifest, "kind": "something_else"},
                   {**manifest, "run_record": None},
                   {**manifest, "report_version": report.REPORT_VERSION + 1}):
        manifest_path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match="manifest"):
            report.require_primary_tables(copy, primary_level=LEVEL)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="level"):
        report.require_primary_tables(copy, primary_level="affine")


class _PastTheChain(Exception):
    """The tables mode passed the chain check and went on to read the run."""


def _level_identity(run, level: str = "affine", **changes):
    """The primary run's identity, as a level run under the same chain carries it.

    Only the lock differs: each level has its own. The gate receipts, E, and
    every digest are the primary run's.
    """
    from lot.phase5_modes import checkpoint_lock_path
    from lot.phase5_provenance import require_evaluated_run

    identity = require_evaluated_run(run.cfg, FAST, run.config, LEVEL,
                                     expected_scenes=list(SCENES), folds=FOLDS, check_code=False)
    licence = {stem: sha for stem, sha in identity.licence.items()
               if stem in ("integration_gate", "tiny_overfit")}
    licence[checkpoint_lock_path(Path("."), level).stem] = "1" * 64
    return dataclasses.replace(identity, level=level, licence=licence, **changes)


def _tables_at(run, level: str, identity, monkeypatch, tmp_path):
    """run_tables at level, with its run licensed as identity, its output beside."""
    monkeypatch.setattr(report, "require_evaluated_run", lambda *a, **k: identity)

    def past(*args, **kwargs):
        raise _PastTheChain()

    return report.run_tables(run.cfg, FAST, run.config, level, expected_scenes=list(SCENES),
                             folds=FOLDS, check_code=False, phase4_reference=past,
                             tables_root=Path(run.run_dir) / "tables")


@pytest.mark.parametrize("change, field", [
    ({"commit": "0" * 40, "training_commit": "0" * 40}, "evaluation_commit"),
    ({"config_digest": "0" * 64}, "config_digest"),
    ({"measurement_digest": "0" * 64}, "measurement_digest"),
    ({"seeds": (0, 1)}, "seeds"),
    ({"code_checked": True}, "code_checked"),
    ("tiny_overfit", "licence.tiny_overfit"),
    ("integration_gate", "licence.integration_gate"),
], ids=["commit", "config", "measurement", "seeds", "code", "overfit", "gate"])
def test_a_non_primary_level_under_another_chain_is_refused(published, monkeypatch, tmp_path,
                                                            change, field):
    """reporting_rules.md decision 4: the affine level trains under the same
    chain, at E, and the overfit gate runs once, at the primary level. A level
    run at another commit, under other digests or seeds, or licensed by other
    gate receipts, is not reported beside the primary result."""
    from lot.phase5_provenance import ProvenanceError

    run, _ = published
    if isinstance(change, str):
        identity = _level_identity(run)
        identity = dataclasses.replace(identity, licence={**identity.licence, change: "0" * 64})
    else:
        identity = _level_identity(run, **change)
    with pytest.raises(ProvenanceError) as error:
        _tables_at(run, "affine", identity, monkeypatch, tmp_path)
    assert field in error.value.fields, str(error.value)
    assert not (Path(run.run_dir) / "tables" / "affine").exists()


def test_a_non_primary_level_under_the_primary_chain_is_tabulated(published, monkeypatch,
                                                                  tmp_path):
    run, _ = published
    with pytest.raises(_PastTheChain):
        _tables_at(run, "affine", _level_identity(run), monkeypatch, tmp_path)
    assert not (Path(run.run_dir) / "tables" / "affine").exists()


def test_the_role_of_each_level_is_read_from_the_configuration():
    cfg = load_phase5_config()
    assert report.level_role(cfg, cfg.primary_alignment_level) == "primary"
    for level in cfg.sensitivity_alignment_levels:
        assert report.level_role(cfg, level) == "sensitivity"
    for level in cfg.diagnostic_alignment_levels:
        assert report.level_role(cfg, level) == "diagnostic"
    with pytest.raises(ValueError, match="not declared"):
        report.level_role(cfg, "scene")


def test_the_tables_read_each_scene_once_through_the_verified_reader(tmp_path, monkeypatch,
                                                                    built):
    run = build_run(tmp_path, monkeypatch)
    _stage_synthetic_rows(run, FAST)
    monkeypatch.setattr(report, "build_tables", lambda *a, **k: built)
    seen = []
    original = report.read_scene_rows

    def spy(identity, scene):
        seen.append(scene)
        return original(identity, scene)

    monkeypatch.setattr(report, "read_scene_rows", spy)
    run_tables(run)
    assert seen == list(SCENES)


# ---------------------------------------------------------------------------
# The Phase 4 references are read from the Phase 4 evaluation parquets
# ---------------------------------------------------------------------------

def test_the_phase4_reader_binds_the_parquets_evaluation_reconciled_against(
    tmp_path, monkeypatch
):
    import lot.phase4_report as phase4_report
    from lot.phase5_check import sha256_file

    phase4 = tmp_path / "phase4" / "eval"
    phase4.mkdir(parents=True)
    records = {}
    for scene in SCENES:
        path = phase4 / f"{scene}.parquet"
        write_rows(path, [{"scene": scene}], {"git_commit": "4" * 40})
        records[scene] = {"phase4_parquet_sha256": sha256_file(path)}
    cfg = dataclasses.replace(load_phase5_config(), phase4_dir=str(tmp_path / "phase4"))
    identity = type("Identity", (), {"scenes": SCENES, "records": records,
                                     "phase4_commit": "4" * 40})()
    calls = {}

    def fake_read(eval_dir, analysis):
        calls["dir"] = Path(eval_dir)
        return [{"row": 1}]

    def fake_build(rows, analysis):
        return [{"level": "image", "path": "per_point"}, {"level": "affine",
                                                           "path": "per_point"}], {
            "mask_mismatched_arms": 0}

    monkeypatch.setattr(phase4_report, "read_phase4_dir", fake_read)
    monkeypatch.setattr(phase4_report, "build_records", fake_build)
    monkeypatch.setattr(phase4_report, "ladder_table", lambda records, analysis: [
        {**r, "analysis": "pooled"} for r in records])
    monkeypatch.setattr(phase4_report, "bin_table", lambda records, analysis: [])
    out = report.phase4_reference_cells(cfg, ANALYSIS, "image", identity)
    assert calls["dir"] == phase4
    assert out["ladder"] == [{"level": "image", "path": "per_point", "analysis": "pooled"}]
    assert set(out["inputs"]) == {f"phase4/eval/{scene}.parquet" for scene in SCENES}

    records[SCENES[1]]["phase4_parquet_sha256"] = "0" * 64
    with pytest.raises(TablesStop, match=SCENES[1]):
        report.phase4_reference_cells(cfg, ANALYSIS, "image", identity)
    records[SCENES[1]]["phase4_parquet_sha256"] = sha256_file(phase4 / f"{SCENES[1]}.parquet")
    identity.phase4_commit = "5" * 40
    with pytest.raises(TablesStop, match="phase4_commit"):
        report.phase4_reference_cells(cfg, ANALYSIS, "image", identity)


def load_phase5_config():
    from lot.phase5 import load_phase5_config as load

    return load(REPO / "configs" / "phase5.yaml")
