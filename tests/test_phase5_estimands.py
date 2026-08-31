"""Streams V and W: the estimand arithmetic, its pairing, and its populations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lot.analysis_config import load_analysis_config
from lot.phase5_estimands import (
    CL_TRANSPORT,
    FORMULATION,
    PER_POINT,
    QUANTITY_POPULATION,
    SPLAT_POOL,
    TL_REFERENCE,
    HeadlineSubstitutionError,
    assert_not_target_lift_headline,
    evaluate_quantity,
    near_zero_disclosure,
    quantity_formulas,
    seed_sensitivity,
    support_counts,
    three_rung_decomposition,
)

ANALYSIS = load_analysis_config()


def make_records(
    n_scenes: int = 8,
    pairs_per_scene: int = 6,
    cl: float = 0.62,
    predict: float = 0.55,
    nowarp: float = 0.40,
    scene_spread: float = 0.10,
    noise: float = 1e-3,
    seed: int = 4,
) -> list[dict]:
    """Pairs whose two methods share a large scene effect, as real scenes do."""
    rng = np.random.default_rng(seed)
    records = []
    for s in range(n_scenes):
        level = rng.normal(0.0, scene_spread)
        for p in range(pairs_per_scene):
            records.append({
                "scene": f"scene_{s}",
                "pair": f"{s}_{p}",
                "regime": "translation",
                "cl_raw": cl + level + rng.normal(0, noise),
                "cl_centered": cl + level + rng.normal(0, noise),
                "predict_raw": predict + level + rng.normal(0, noise),
                "predict_centered": predict + level + rng.normal(0, noise),
                "nowarp_raw": nowarp + level + rng.normal(0, noise),
                "nowarp_centered": nowarp + level + rng.normal(0, noise),
                "tl_form_raw": cl + 0.02 + level, "tl_form_centered": cl + 0.02 + level,
                "cl_form_raw": cl + level, "cl_form_centered": cl + level,
                "sp_transport_raw": 0.7 + level, "sp_transport_centered": 0.7 + level,
                "sp_predict_raw": 0.69 + level, "sp_predict_centered": 0.69 + level,
                "sp_nowarp_raw": 0.5 + level, "sp_nowarp_centered": 0.5 + level,
                "n_primary": 900, "n_formulation": 700, "n_splat": 800,
                "n_predict_nonfinite": 0,
            })
    return records


# ---------------------------------------------------------------------------
# The formulas
# ---------------------------------------------------------------------------

def test_headline_gap_is_context_lift_minus_predictor():
    forms = quantity_formulas("centered")
    means = {"cl_centered": 0.62, "predict_centered": 0.55, "nowarp_centered": 0.40}
    assert forms["delta_learn_pp"](means) == pytest.approx(0.07)
    assert forms["cl_margin"](means) == pytest.approx(0.22)
    assert forms["predict_margin"](means) == pytest.approx(0.15)


def test_formulation_gap_is_target_lift_minus_context_lift():
    forms = quantity_formulas("raw")
    means = {"tl_form_raw": 0.64, "cl_form_raw": 0.62}
    assert forms["delta_formulation"](means) == pytest.approx(0.02)


def test_operational_gap_uses_the_splat_columns():
    forms = quantity_formulas("centered")
    means = {"sp_transport_centered": 0.70, "sp_predict_centered": 0.69,
             "sp_nowarp_centered": 0.50}
    assert forms["delta_learn_sp"](means) == pytest.approx(0.01)
    assert forms["sp_transport_margin"](means) == pytest.approx(0.20)


def test_every_quantity_declares_its_population():
    for metric in ("raw", "centered"):
        for name in quantity_formulas(metric):
            assert name in QUANTITY_POPULATION, name
    assert QUANTITY_POPULATION["delta_learn_pp"] == PER_POINT
    assert QUANTITY_POPULATION["delta_formulation"] == FORMULATION
    assert QUANTITY_POPULATION["delta_learn_sp"] == SPLAT_POOL


def test_metric_must_be_raw_or_centered():
    with pytest.raises(ValueError, match="raw or centered"):
        quantity_formulas("cosine")


# ---------------------------------------------------------------------------
# Stream AD: the target-lift substitution guard
# ---------------------------------------------------------------------------

def test_target_lift_cannot_become_the_headline_gap():
    assert_not_target_lift_headline("delta_learn_pp", CL_TRANSPORT)
    assert_not_target_lift_headline("delta_formulation", TL_REFERENCE)
    with pytest.raises(HeadlineSubstitutionError, match="information asymmetry"):
        assert_not_target_lift_headline("delta_learn_pp", TL_REFERENCE)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def test_headline_interval_is_paired_and_survives_a_large_scene_effect():
    """The gap is nearly constant across scenes while the levels are not.

    A paired interval sees the constant gap; subtracting two independent
    intervals would report an uncertainty dominated by the scene effect that
    cancels. This is the single arithmetic property the headline depends on.
    """
    records = make_records(scene_spread=0.15, noise=1e-4)
    gap = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    assert gap.estimate == pytest.approx(0.07, abs=2e-3)
    assert gap.hi - gap.lo < 0.01, (gap.lo, gap.hi)
    assert gap.lo > 0.0, "the interval should exclude zero here"

    level = evaluate_quantity(records, "cl_transport", "centered", ANALYSIS)
    assert level.hi - level.lo > 0.05, "each level alone must be wide"


def test_negative_gap_is_reported_as_negative():
    """A predictor that wins must be reported directly, not folded into zero."""
    records = make_records(cl=0.50, predict=0.60, scene_spread=0.05, noise=1e-4)
    gap = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    assert gap.estimate == pytest.approx(-0.10, abs=3e-3)
    assert gap.hi < 0.0


# ---------------------------------------------------------------------------
# Populations and support
# ---------------------------------------------------------------------------

def test_a_cell_uses_only_pairs_that_contributed_to_its_population():
    records = make_records(n_scenes=6, pairs_per_scene=5)
    for record in records[:10]:
        record["n_formulation"] = 0
    primary = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    formulation = evaluate_quantity(records, "delta_formulation", "centered", ANALYSIS)
    assert primary.n_camera_pairs == 30
    assert formulation.n_camera_pairs == 20
    assert formulation.population == FORMULATION


def test_support_thresholds_come_from_the_frozen_config():
    plenty = make_records(n_scenes=8, pairs_per_scene=6)
    assert evaluate_quantity(plenty, "delta_learn_pp", "centered", ANALYSIS).supported

    thin = make_records(n_scenes=2, pairs_per_scene=4)
    result = evaluate_quantity(thin, "delta_learn_pp", "centered", ANALYSIS)
    assert not result.supported
    assert result.n_scenes == 2 < ANALYSIS.support_min_scenes
    # An unsupported cell is still reported with its n, never dropped.
    assert math.isfinite(result.estimate)


def test_support_counts_sum_the_population_specific_count_field():
    records = make_records(n_scenes=3, pairs_per_scene=2)
    counts = support_counts(records, "n_primary")
    assert counts.n_scenes == 3
    assert counts.n_camera_pairs == 6
    assert counts.n_feature_comparisons == 6 * 900


def test_empty_population_yields_a_reported_empty_cell():
    records = make_records(n_scenes=4, pairs_per_scene=3)
    for record in records:
        record["n_splat"] = 0
    result = evaluate_quantity(records, "delta_learn_sp", "centered", ANALYSIS)
    assert result.n_camera_pairs == 0
    assert not result.supported
    assert math.isnan(result.estimate)


# ---------------------------------------------------------------------------
# Stream AB: the near-zero disclosure
# ---------------------------------------------------------------------------

def test_near_zero_flags_inside_the_frozen_band():
    out = near_zero_disclosure("delta_learn_pp", 0.0012, 0.0009, 0.0015, ANALYSIS)
    assert out["near_zero"]
    assert out["band"] == ANALYSIS.path_agreement_tolerance == 0.003
    assert out["interval_excludes_zero"]
    assert out["wording"] == "small, sign-consistent effect"


def test_near_zero_with_an_interval_containing_zero_claims_nothing():
    out = near_zero_disclosure("delta_learn_pp", 0.0004, -0.0020, 0.0028, ANALYSIS)
    assert out["near_zero"]
    assert not out["interval_excludes_zero"]
    assert out["wording"] == "no measurable difference at the reported scale"
    assert "equivalen" not in out["wording"], "equivalence is never claimed"


def test_effects_outside_the_band_are_not_flagged():
    out = near_zero_disclosure("delta_learn_pp", 0.031, 0.026, 0.036, ANALYSIS)
    assert not out["near_zero"]
    assert out["wording"] == "effect outside the operator band"


def test_the_band_is_exactly_at_the_boundary():
    assert near_zero_disclosure("delta_learn_pp", 0.003, 0.002, 0.004, ANALYSIS)["near_zero"]
    assert not near_zero_disclosure(
        "delta_learn_pp", 0.0030001, 0.002, 0.004, ANALYSIS
    )["near_zero"]


def test_the_band_does_not_apply_to_dimensionless_quantities():
    out = near_zero_disclosure("n_primary", 0.001, 0.0, 0.002, ANALYSIS)
    assert not out["applicable"]
    assert not out["near_zero"]


def test_a_nonfinite_estimate_is_not_flagged_as_near_zero():
    out = near_zero_disclosure("delta_learn_pp", float("nan"), float("nan"),
                               float("nan"), ANALYSIS)
    assert out["applicable"]
    assert not out["near_zero"]


def test_cell_row_carries_the_disclosure_flat():
    records = make_records(cl=0.500, predict=0.4995, scene_spread=0.02, noise=1e-5)
    row = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS).as_row()
    assert row["near_zero"] is True
    assert row["quantity"] == "delta_learn_pp"
    assert row["population"] == PER_POINT
    assert "disclosure" not in row


# ---------------------------------------------------------------------------
# Seeds and the decomposition
# ---------------------------------------------------------------------------

def test_seed_sensitivity_reports_spread_separately():
    out = seed_sensitivity({0: 0.070, 1: 0.074, 2: 0.068})
    assert out["mean"] == pytest.approx(0.070666, abs=1e-5)
    assert out["range"] == pytest.approx(0.006, abs=1e-9)
    assert out["n_seeds"] == 3


def test_seed_sensitivity_with_nothing_finite():
    out = seed_sensitivity({0: float("nan")})
    assert out["n_seeds"] == 0
    assert math.isnan(out["mean"])


def test_three_rungs_stay_separate():
    rungs = three_rung_decomposition(
        phase3_oracle=0.80, phase4_transport_only=0.77,
        cl_transport=0.75, predict_with_depth=0.70,
    )
    assert rungs["representation_limitation_oracle"] == pytest.approx(0.80)
    assert rungs["estimated_geometry_limitation"] == pytest.approx(0.03)
    assert rungs["learned_vs_explicit_limitation"] == pytest.approx(0.05)
    # The Phase 5 rung is built from the information-symmetric comparator only.
    assert rungs["learned_vs_explicit_limitation"] != pytest.approx(
        0.77 - 0.70
    ), "the Phase 4 target-lift score leaked into the Phase 5 rung"
