"""Streams V and W: the estimand arithmetic, its pairing, and its populations."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from lot.analysis_config import load_analysis_config
from lot.phase5_estimands import (
    CL_TRANSPORT,
    CROSS_PATH,
    FORMULATION,
    PER_POINT,
    QUANTITY_POPULATION,
    SPLAT_POOL,
    TL_REFERENCE,
    HeadlineSubstitutionError,
    PathEstimate,
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
    meanfeat: float = 0.30,
    scene_spread: float = 0.10,
    noise: float = 1e-3,
    seed: int = 4,
) -> list[dict]:
    """Pairs whose two methods share a large scene effect, as real scenes do.

    The Mean-Feature columns draw no noise, so adding them left every other
    column of every record exactly as it was.
    """
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
                # Mean-Feature is defined under raw metrics only, per PROTOCOL 3.7.
                "meanfeat_raw": meanfeat + level, "meanfeat_l2_raw": 1.0 - meanfeat - level,
                "tl_form_raw": cl + 0.02 + level, "tl_form_centered": cl + 0.02 + level,
                "cl_form_raw": cl + level, "cl_form_centered": cl + level,
                "sp_transport_raw": 0.7 + level, "sp_transport_centered": 0.7 + level,
                "sp_predict_raw": 0.69 + level, "sp_predict_centered": 0.69 + level,
                "sp_nowarp_raw": 0.5 + level, "sp_nowarp_centered": 0.5 + level,
                "sp_meanfeat_raw": 0.32 + level, "sp_meanfeat_l2_raw": 0.68 - level,
                # The cross-path common-valid columns. Both paths re-scored on
                # the cells they share, which is the population PROTOCOL 3.9's
                # disclosure is computed on.
                **_intersection(cl + level, predict + level, nowarp + level,
                                0.7 + level, 0.69 + level, 0.5 + level,
                                meanfeat + level, 0.32 + level),
                "n_primary": 900, "n_formulation": 700, "n_splat": 800,
                "n_intersect": 640, "n_predict_nonfinite": 0,
            })
    return records


def _intersection(cl, predict, nowarp, sp_transport, sp_predict, sp_nowarp,
                  meanfeat, sp_meanfeat) -> dict:
    """All six cross-path arms with all four columns, cosines set and L2 finite.

    The Mean-Feature floor on each path carries its two raw columns only.
    """
    values = {
        "cl": cl, "predict": predict, "nowarp": nowarp,
        "sp_transport": sp_transport, "sp_predict": sp_predict, "sp_nowarp": sp_nowarp,
    }
    out = {}
    for arm, v in values.items():
        out[f"x_{arm}_raw"] = v
        out[f"x_{arm}_centered"] = v
        out[f"x_{arm}_l2_raw"] = 1.0 - v
        out[f"x_{arm}_l2_centered"] = 1.0 - v
    for arm, v in (("meanfeat", meanfeat), ("sp_meanfeat", sp_meanfeat)):
        out[f"x_{arm}_raw"] = v
        out[f"x_{arm}_l2_raw"] = 1.0 - v
    return out


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

def _pe(estimate, lo, hi):
    return PathEstimate(estimate, lo, hi)


def test_both_paths_inside_the_band_and_clear_of_zero_license_the_strong_wording():
    """PROTOCOL 3.9's strongest near-zero licence, and its exact preconditions."""
    out = near_zero_disclosure(
        "delta_learn_pp", _pe(0.0012, 0.0009, 0.0015), _pe(0.0018, 0.0011, 0.0024),
        ANALYSIS,
    )
    assert out["near_zero"]
    assert out["band"] == ANALYSIS.path_agreement_tolerance == 0.003
    assert out["paths_agree_in_sign"] and out["both_intervals_exclude_zero"]
    assert out["wording"] == "small, sign-consistent effect"


def test_a_disagreeing_splat_path_withdraws_the_strong_wording():
    """The over-licensing the single-path version allowed, now impossible.

    Same per-point cell as above. The operational path reverses sign, so 3.9
    licenses no claim of advantage, where the earlier one-path function would
    have called this a small, sign-consistent effect.
    """
    out = near_zero_disclosure(
        "delta_learn_pp", _pe(0.0012, 0.0009, 0.0015), _pe(-0.0016, -0.0022, -0.0009),
        ANALYSIS,
    )
    assert not out["paths_agree_in_sign"]
    assert out["wording"] == (
        "no claim of advantage; the effect is at the scale of "
        "evaluation-path choice"
    )
    assert "equivalen" not in out["wording"]


def test_an_interval_containing_zero_withdraws_the_strong_wording():
    out = near_zero_disclosure(
        "delta_learn_pp", _pe(0.0012, -0.0004, 0.0028), _pe(0.0018, 0.0011, 0.0024),
        ANALYSIS,
    )
    assert not out["both_intervals_exclude_zero"]
    assert out["wording"].startswith("no claim of advantage")


def test_exactly_one_path_inside_the_band_is_reported_as_path_sensitive():
    """Only when the veto does not fire: same sign, both intervals clear."""
    out = near_zero_disclosure(
        "delta_learn_pp", _pe(0.0012, 0.0009, 0.0015), _pe(0.031, 0.026, 0.036),
        ANALYSIS,
    )
    assert out["near_zero"]
    assert "path-sensitive" in out["wording"]


def test_the_sign_veto_outranks_the_exactly_one_in_band_branch():
    """PROTOCOL 3.9's third clause is unconditional, so it is a veto.

    Exactly one estimate sits inside the band, which taken alone would read as
    path-sensitive. The paths reverse sign, and 3.9 says that licenses no claim
    of advantage. The veto has to be tested first or this cell is over-reported.
    """
    out = near_zero_disclosure(
        "delta_learn_pp", _pe(0.001, 0.0005, 0.0015), _pe(-0.020, -0.030, -0.010),
        ANALYSIS,
    )
    assert not out["paths_agree_in_sign"]
    assert out["wording"] == (
        "no claim of advantage; the effect is at the scale of "
        "evaluation-path choice"
    )
    assert "path-sensitive" not in out["wording"]


def test_the_interval_veto_outranks_the_exactly_one_in_band_branch():
    out = near_zero_disclosure(
        "delta_learn_pp", _pe(0.001, -0.0004, 0.0024), _pe(0.031, 0.026, 0.036),
        ANALYSIS,
    )
    assert not out["both_intervals_exclude_zero"]
    assert out["wording"].startswith("no claim of advantage")


def test_both_paths_outside_the_band_are_not_flagged():
    out = near_zero_disclosure(
        "delta_learn_pp", _pe(0.031, 0.026, 0.036), _pe(0.028, 0.022, 0.034), ANALYSIS
    )
    assert not out["near_zero"]
    assert out["wording"] == "effect outside the operator band"


def test_the_band_is_exactly_at_the_boundary():
    inside = near_zero_disclosure(
        "delta_learn_pp", _pe(0.003, 0.002, 0.004), _pe(0.003, 0.002, 0.004), ANALYSIS
    )
    assert inside["near_zero"]
    outside = near_zero_disclosure(
        "delta_learn_pp", _pe(0.0030001, 0.002, 0.004), _pe(0.0030001, 0.002, 0.004),
        ANALYSIS,
    )
    assert not outside["near_zero"]


def test_a_missing_second_path_can_never_reach_the_strong_wording():
    """A quantity with no counterpart is disclosed at the weakest licence.

    Updated for reporting_rules.md decision 2. The interval here, 0.0009 to
    0.0015, excludes zero. The previous code printed "no measurable
    difference" for it, which the interval contradicts. The single-path
    sentence now follows the reported interval.
    """
    out = near_zero_disclosure("delta_learn_pp", _pe(0.0012, 0.0009, 0.0015), None,
                               ANALYSIS)
    assert out["near_zero"]
    assert out["splat_pool"] is None
    assert out["wording"] == (
        "within the operator band on its only path; its size is not "
        "certified by a second path"
    )
    assert out["wording"] != "small, sign-consistent effect"


def test_the_band_does_not_apply_to_dimensionless_quantities():
    out = near_zero_disclosure("n_primary", _pe(0.001, 0.0, 0.002), None, ANALYSIS)
    assert not out["applicable"]
    assert not out["near_zero"]


def test_a_nonfinite_estimate_is_not_flagged_as_near_zero():
    nan = float("nan")
    out = near_zero_disclosure("delta_learn_pp", _pe(nan, nan, nan), None, ANALYSIS)
    assert out["applicable"]
    assert not out["near_zero"]


def test_no_branch_ever_claims_equivalence():
    """No frozen equivalence region exists, so no wording may imply one."""
    cases = [
        (_pe(0.001, 0.0005, 0.0015), _pe(0.001, 0.0005, 0.0015)),
        (_pe(0.001, -0.001, 0.003), _pe(0.001, 0.0005, 0.0015)),
        (_pe(0.001, 0.0005, 0.0015), _pe(-0.001, -0.0015, -0.0005)),
        (_pe(0.001, 0.0005, 0.0015), _pe(0.04, 0.03, 0.05)),
        (_pe(0.04, 0.03, 0.05), _pe(0.04, 0.03, 0.05)),
        (_pe(0.001, 0.0005, 0.0015), None),
    ]
    for per_point, splat in cases:
        wording = near_zero_disclosure("delta_learn_pp", per_point, splat, ANALYSIS)
        assert "equivalen" not in wording["wording"].lower()


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


def _delta_learn_cell(quantity: str = "delta_learn_pp", supported: bool = True):
    from lot.phase5_estimands import CellResult

    return CellResult(
        quantity=quantity, metric="centered",
        population=QUANTITY_POPULATION[quantity],
        estimate=0.0500, lo=0.0400, hi=0.0600,
        n_scenes=18, n_camera_pairs=900, n_feature_comparisons=720_000,
        n_replicates=1000, supported=supported, disclosure={},
    )


def test_three_rungs_stay_separate():
    """reporting_rules.md section 6: each rung under its own phase's estimator.

    The inputs are chosen so that the Phase 3 ceiling minus the Phase 4
    estimated score, 0.6000 - 0.5721 = 0.0279, differs from Phase 4's own
    depth tax, 0.0300. That difference is the selection differential between
    the full Phase 3 population and Phase 4's matched one. A helper that
    subtracted across phases would report 0.0279 for rung 1. No rung is
    computed from another here: each is passed through as its phase reported it.
    """
    from lot.phase5_estimands import PER_POINT

    depth_tax = PathEstimate(0.0300, 0.0250, 0.0350)
    rungs = three_rung_decomposition(
        phase3_reference_ceiling=0.6000,
        phase4_depth_tax=depth_tax,
        delta_learn_pp=_delta_learn_cell(),
    )
    assert rungs["representation_limitation_oracle"] == {
        "phase": 3, "estimator": "reference_ceiling_phase3", "estimate": 0.6000,
    }
    assert rungs["estimated_geometry_limitation"] == {
        "phase": 4, "estimator": "depth_tax",
        "estimate": 0.0300, "lo": 0.0250, "hi": 0.0350,
    }
    assert rungs["estimated_geometry_limitation"]["estimate"] != pytest.approx(
        0.6000 - 0.5721
    ), "rung 1 must be Phase 4's own depth tax, not a difference across phases"
    assert rungs["learned_vs_explicit_limitation"] == {
        "phase": 5, "estimator": "delta_learn_pp", "population": PER_POINT,
        "estimate": 0.0500, "lo": 0.0400, "hi": 0.0600,
        "n_replicates": 1000, "supported": True,
    }
    json.dumps(rungs)


@pytest.mark.parametrize("quantity", ["delta_learn_sp", "delta_formulation", "tl_reference"])
def test_the_phase5_rung_is_delta_learn_pp_only(quantity):
    """The Phase 5 rung is defined only through the information-symmetric
    comparison. Any other cell in its place is refused."""
    with pytest.raises(HeadlineSubstitutionError, match="delta_learn_pp"):
        three_rung_decomposition(
            phase3_reference_ceiling=0.6000,
            phase4_depth_tax=PathEstimate(0.0300, 0.0250, 0.0350),
            delta_learn_pp=_delta_learn_cell(quantity),
        )


# ---------------------------------------------------------------------------
# PROTOCOL 3.9: the disclosure is computed on the cross-path common-valid set
# ---------------------------------------------------------------------------

def test_the_disclosure_reads_the_intersection_columns_not_the_own_population_ones():
    """Comparing V_P5_pp against V_sp would mix operator with selection.

    The per-point and splat columns are given a large, opposite-signed gap here
    while the intersection columns carry a small agreeing one. A disclosure
    built on the own-population columns would report the former; the protocol
    requires the latter.
    """
    records = make_records(n_scenes=8, pairs_per_scene=5, noise=1e-5)
    for record in records:
        # Own-population columns: a large gap the disclosure must NOT read.
        record["cl_centered"] = 0.80
        record["predict_centered"] = 0.40
        record["sp_transport_centered"] = 0.40
        record["sp_predict_centered"] = 0.80
        # Intersection columns: a small, agreeing gap, which is what 3.9 wants.
        record["x_cl_centered"] = 0.6010
        record["x_predict_centered"] = 0.6000
        record["x_sp_transport_centered"] = 0.5012
        record["x_sp_predict_centered"] = 0.5000

    cell = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    disclosure = cell.disclosure
    assert disclosure["per_point"]["estimate"] == pytest.approx(0.0010, abs=1e-6)
    assert disclosure["splat_pool"]["estimate"] == pytest.approx(0.0012, abs=1e-6)
    assert disclosure["paths_agree_in_sign"]
    # Updated for reporting_rules.md decision 2. The reported effect is 0.40,
    # outside the band, so the near-zero wording is not engaged. The previous
    # code engaged on the in-band path terms alone and printed the
    # sign-consistent sentence. The terms are still shown beside the cell.
    # test_the_engaged_wording_reads_the_intersection_terms keeps the wording
    # check, with a reported effect inside the band.
    assert not disclosure["near_zero"]
    assert disclosure["wording"] == "effect outside the operator band"
    # The headline estimand itself still comes from the primary support.
    assert cell.estimate == pytest.approx(0.40, abs=1e-6)


def test_the_engaged_wording_reads_the_intersection_terms():
    """The same check, with a reported effect inside the band so it engages.

    The splat path's own-population gap is large and of opposite sign. Read
    from there, the veto would fire. Read from the intersection, as 3.9
    requires, both terms are small and agree.
    """
    records = make_records(n_scenes=8, pairs_per_scene=5, noise=1e-5)
    for record in records:
        # Own-population columns. The per-point gap is the reported effect.
        record["cl_centered"] = 0.5015
        record["predict_centered"] = 0.5000
        record["sp_transport_centered"] = 0.40
        record["sp_predict_centered"] = 0.80
        # Intersection columns: a small, agreeing gap.
        record["x_cl_centered"] = 0.6010
        record["x_predict_centered"] = 0.6000
        record["x_sp_transport_centered"] = 0.5012
        record["x_sp_predict_centered"] = 0.5000

    cell = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    disclosure = cell.disclosure
    assert cell.estimate == pytest.approx(0.0015, abs=1e-6)
    assert disclosure["near_zero"]
    assert disclosure["per_point"]["estimate"] == pytest.approx(0.0010, abs=1e-6)
    assert disclosure["splat_pool"]["estimate"] == pytest.approx(0.0012, abs=1e-6)
    assert disclosure["wording"] == "small, sign-consistent effect"


def test_the_path_difference_carries_a_paired_interval():
    """3.9 requires an interval for the difference, from one scene draw."""
    records = make_records(n_scenes=8, pairs_per_scene=5, scene_spread=0.2,
                           noise=1e-5)
    cell = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    difference = cell.disclosure["path_difference"]
    assert math.isfinite(difference["lo"]) and math.isfinite(difference["hi"])
    assert difference["lo"] <= difference["estimate"] <= difference["hi"]
    # The scene effect is shared by both paths and cancels in the difference, so
    # the paired interval is tight where each path's own level is not.
    assert difference["hi"] - difference["lo"] < 0.01


def test_the_difference_equals_the_two_terms_it_is_built_from():
    records = make_records(n_scenes=6, pairs_per_scene=4, noise=1e-6)
    cell = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    d = cell.disclosure
    assert d["path_difference"]["estimate"] == pytest.approx(
        d["per_point"]["estimate"] - d["splat_pool"]["estimate"], abs=1e-9
    )


def test_the_disclosure_terms_share_one_scene_draw():
    """Paired means the same resample serves both paths and the difference.

    The bootstrap seed and unit list are identical across the three calls, so the
    draw is the same stream; this pins that the difference's interval is not
    wider than a subtraction of two independently drawn ones would allow it to
    look, by checking it against the same-draw construction directly.
    """
    from lot.paired_bootstrap import paired_interval
    from lot.phase5_estimands import INTERSECTION_FIELDS, quantity_formulas

    records = make_records(n_scenes=8, pairs_per_scene=4, scene_spread=0.2,
                           noise=1e-5)
    forms = quantity_formulas("centered")
    direct = paired_interval(
        records, INTERSECTION_FIELDS, forms["path_difference_learn"],
        resamples=ANALYSIS.bootstrap_resamples, seed=ANALYSIS.bootstrap_seed,
        confidence=ANALYSIS.bootstrap_confidence,
    )
    cell = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    assert cell.disclosure["path_difference"]["estimate"] == pytest.approx(
        direct["estimate"], abs=1e-12
    )
    assert cell.disclosure["path_difference"]["lo"] == pytest.approx(
        direct["lo"], abs=1e-12
    )


def test_cross_path_quantities_declare_their_population():
    from lot.phase5_estimands import CROSS_PATH

    for name in (
        "x_delta_learn_pp", "x_delta_learn_sp", "x_cl_margin", "x_sp_transport_margin",
        "x_predict_margin", "x_sp_predict_margin", "path_difference_learn",
        "path_difference_cl_margin", "path_difference_predict_margin",
    ):
        assert QUANTITY_POPULATION[name] == CROSS_PATH


def test_an_empty_intersection_reports_an_empty_disclosure():
    records = make_records(n_scenes=5, pairs_per_scene=4)
    for record in records:
        record["n_intersect"] = 0
    cell = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    assert math.isnan(cell.disclosure["per_point"]["estimate"])
    assert not cell.disclosure["near_zero"]


# ---------------------------------------------------------------------------
# Round-three finding 4: every interpreted effect is disclosed against its path
# ---------------------------------------------------------------------------

def test_a_small_margin_is_disclosed_against_the_splat_margin():
    """A per-point margin inside the band must consult the operational margin."""
    from lot.phase5_estimands import DISCLOSURE_PAIR, INTERPRETED_EFFECTS

    assert "cl_margin" in DISCLOSURE_PAIR and "predict_margin" in DISCLOSURE_PAIR
    records = make_records(n_scenes=8, pairs_per_scene=5, noise=1e-6)
    for r in records:
        # Updated for reporting_rules.md decision 2. Only the reported effect
        # engages the near-zero wording, so the reported margin is set inside
        # the band here. The previous code engaged on the in-band per-point
        # term alone, while the reported margin was 0.22.
        r["cl_centered"], r["nowarp_centered"] = 0.5020, 0.5000
        # Per-point margin tiny and positive; splat margin large and negative.
        r["x_cl_centered"], r["x_nowarp_centered"] = 0.5020, 0.5000
        r["x_sp_transport_centered"], r["x_sp_nowarp_centered"] = 0.40, 0.45
    cell = evaluate_quantity(records, "cl_margin", "centered", ANALYSIS)
    d = cell.disclosure
    assert d["near_zero"]
    assert d["splat_pool"] is not None, "the counterpart was not consulted"
    assert d["per_point"]["estimate"] == pytest.approx(0.002, abs=1e-6)
    assert d["splat_pool"]["estimate"] == pytest.approx(-0.05, abs=1e-6)
    assert d["wording"].startswith("no claim of advantage")
    assert "single path" not in d["wording"]


def test_the_operational_gap_is_paired_with_the_same_disclosure_as_the_headline():
    records = make_records(n_scenes=8, pairs_per_scene=5, noise=1e-6)
    pp = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS).disclosure
    sp = evaluate_quantity(records, "delta_learn_sp", "centered", ANALYSIS).disclosure
    assert sp["splat_pool"] is not None
    assert sp["per_point"]["estimate"] == pytest.approx(pp["per_point"]["estimate"])
    assert sp["splat_pool"]["estimate"] == pytest.approx(pp["splat_pool"]["estimate"])
    assert sp["wording"] == pp["wording"]


def test_every_interpreted_effect_has_a_pair_or_is_declared_single_path():
    from lot.phase5_estimands import (
        DISCLOSURE_PAIR, INTERPRETED_EFFECTS, SINGLE_PATH_BY_CONSTRUCTION,
    )

    for name in INTERPRETED_EFFECTS:
        assert name in DISCLOSURE_PAIR or name in SINGLE_PATH_BY_CONSTRUCTION, name
    assert SINGLE_PATH_BY_CONSTRUCTION == {"delta_formulation"}


def test_an_interpreted_effect_without_a_pair_raises_rather_than_going_single_path(monkeypatch):
    from lot import phase5_estimands as m

    monkeypatch.setattr(m, "INTERPRETED_EFFECTS", m.INTERPRETED_EFFECTS | {"cl_transport"})
    records = make_records(n_scenes=6, pairs_per_scene=4)
    with pytest.raises(ValueError, match="no disclosure pair"):
        evaluate_quantity(records, "cl_transport", "centered", ANALYSIS)


def test_the_formulation_diagnostic_names_its_single_path_as_design():
    """Updated for reporting_rules.md decision 2.

    The single-path sentence now follows the reported interval. A constant gap
    of 0.001 has an interval clear of zero. The previous code still printed
    "no measurable difference ... by construction" for it. A gap whose sign
    alternates by scene has an interval that includes zero, and it keeps the
    sentence that names the single path as design.
    """
    records = make_records(n_scenes=8, pairs_per_scene=5, noise=1e-6)
    for r in records:
        r["tl_form_centered"], r["cl_form_centered"] = 0.6010, 0.6000
    cell = evaluate_quantity(records, "delta_formulation", "centered", ANALYSIS)
    assert cell.disclosure["splat_pool"] is None
    assert cell.disclosure["near_zero"]
    assert cell.lo > 0.0
    assert cell.disclosure["wording"] == (
        "within the operator band on its only path; its size is not "
        "certified by a second path"
    )
    assert cell.disclosure["wording"] != "small, sign-consistent effect"

    for r in records:
        sign = 1.0 if int(r["scene"].split("_")[1]) % 2 == 0 else -1.0
        r["tl_form_centered"] = 0.6000 + sign * 0.002
    cell = evaluate_quantity(records, "delta_formulation", "centered", ANALYSIS)
    assert cell.disclosure["near_zero"]
    assert cell.lo <= 0.0 <= cell.hi
    assert cell.disclosure["wording"] == (
        "no measurable difference at the reported scale; this quantity "
        "has no second evaluation path by construction"
    )


def test_margin_path_differences_are_their_own_paired_quantities():
    records = make_records(n_scenes=8, pairs_per_scene=5, scene_spread=0.2, noise=1e-6)
    for name in ("cl_margin", "predict_margin"):
        d = evaluate_quantity(records, name, "centered", ANALYSIS).disclosure
        diff = d["path_difference"]
        assert math.isfinite(diff["lo"]) and math.isfinite(diff["hi"])
        assert diff["estimate"] == pytest.approx(
            d["per_point"]["estimate"] - d["splat_pool"]["estimate"], abs=1e-9
        )


# ---------------------------------------------------------------------------
# The near-zero trigger is the reported effect, and the rule's scope
# ---------------------------------------------------------------------------

def test_a_reported_effect_inside_the_band_is_disclosed_even_when_the_terms_are_not():
    """The defect this guards: the trigger read the intersection terms, so a
    headline gap squarely inside the band could print with no disclosure."""
    out = near_zero_disclosure(
        "delta_learn_pp",
        _pe(0.012, 0.010, 0.014),      # per-point term, outside the band
        _pe(0.011, 0.009, 0.013),      # splat term, outside the band
        ANALYSIS,
        reported=_pe(0.0015, 0.0010, 0.0020),   # the effect the cell reports
    )
    assert out["near_zero"], "an effect inside the band was not disclosed"
    assert out["reported"]["estimate"] == pytest.approx(0.0015)
    # The wording still comes from the two path terms, as 3.9 requires.
    assert out["per_point"]["estimate"] == pytest.approx(0.012)


def test_the_reported_effect_defaults_to_the_per_point_term():
    out = near_zero_disclosure(
        "delta_learn_pp", _pe(0.0012, 0.0009, 0.0015), None, ANALYSIS
    )
    assert out["reported"]["estimate"] == pytest.approx(0.0012)


def test_the_cell_disclosure_carries_the_estimate_the_cell_reports():
    records = make_records(n_scenes=8, pairs_per_scene=5, noise=1e-6)
    for r in records:
        # Headline gap inside the band; intersection terms far outside it.
        r["cl_centered"], r["predict_centered"] = 0.5015, 0.5000
        r["x_cl_centered"], r["x_predict_centered"] = 0.62, 0.55
        r["x_sp_transport_centered"], r["x_sp_predict_centered"] = 0.61, 0.54
    cell = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    assert cell.estimate == pytest.approx(0.0015, abs=1e-5)
    assert cell.disclosure["reported"]["estimate"] == pytest.approx(cell.estimate)
    assert cell.disclosure["near_zero"], "the reported effect is inside the band"


def test_the_near_zero_rule_applies_to_interpreted_effects_only():
    """Its three licensed sentences are all claims about an advantage.

    The defect this guards: SCORE_SPACE_QUANTITIES was frozenset of every
    quantity, so path_difference_learn, which is itself the difference between
    the two paths, was told it had no second evaluation path by construction.
    """
    from lot.phase5_estimands import INTERPRETED_EFFECTS, SCORE_SPACE_QUANTITIES

    assert SCORE_SPACE_QUANTITIES == INTERPRETED_EFFECTS
    for name in ("path_difference_learn", "x_delta_learn_pp", "x_sp_predict_margin",
                 "cl_transport", "sp_transport", "tl_reference"):
        out = near_zero_disclosure(name, _pe(0.001, 0.0005, 0.0015), None, ANALYSIS)
        assert not out["applicable"], name
        assert not out["near_zero"], name


def test_a_disclosure_term_never_receives_the_single_path_sentence():
    records = make_records(n_scenes=6, pairs_per_scene=4, noise=1e-6)
    for name in ("path_difference_learn", "x_delta_learn_pp"):
        cell = evaluate_quantity(records, name, "centered", ANALYSIS)
        assert not cell.disclosure["applicable"]
        assert "by construction" not in cell.disclosure.get("wording", "")


# ---------------------------------------------------------------------------
# reporting_rules.md decision 2: what engages the near-zero wording, and which
# sentence it prints. Every branch is pinned here by its exact words.
# ---------------------------------------------------------------------------

VETO = "no claim of advantage; the effect is at the scale of evaluation-path choice"
SMALL = "small, sign-consistent effect"
PATH_SENSITIVE = (
    "effect licensed but its size is path-sensitive; the two evaluation "
    "paths do not agree about whether it sits inside the operator band"
)
SUPPORT_SENSITIVE = (
    "effect licensed; inside the operator band on its own support but "
    "outside it on both paths' common cells"
)
OUTSIDE = "effect outside the operator band"
ONLY_PATH_CLEAR = (
    "within the operator band on its only path; its size is not certified "
    "by a second path"
)
ONLY_PATH_ZERO = (
    "no measurable difference at the reported scale; this quantity has no "
    "second evaluation path by construction"
)
NOT_ESTIMABLE = "not estimable on this cell"
# Decision 2's six sentences. Only these may be printed with near_zero True.
ENGAGED_WORDINGS = (
    VETO, SMALL, PATH_SENSITIVE, SUPPORT_SENSITIVE, ONLY_PATH_CLEAR, ONLY_PATH_ZERO,
)
# Two labels carried over from the code at cc20e3e. They are not decision 2
# wording. Each marks a cell that is not engaged.
NOT_ENGAGED_LABELS = (OUTSIDE, NOT_ESTIMABLE)
ALL_WORDINGS = ENGAGED_WORDINGS + NOT_ENGAGED_LABELS
DISCLOSURE_KEYS = {
    "reported", "per_point", "splat_pool", "path_difference",
    "paths_agree_in_sign", "both_intervals_exclude_zero", "band", "applicable",
    "wording", "near_zero",
}
NAN = float("nan")


def test_a_large_reported_effect_with_a_near_zero_splat_term_is_not_engaged():
    """The defect decision 2 removes.

    A headline gap of 0.05, interval clear of zero, has a splat term that sits
    near zero with an interval that includes it. The previous code engaged on
    that term and printed the veto, which made outcome 49 unreachable. The
    reported effect is far outside the band, so nothing engages.
    """
    out = near_zero_disclosure(
        "delta_learn_pp",
        _pe(0.048, 0.030, 0.066),
        _pe(0.001, -0.002, 0.004),
        ANALYSIS,
        difference=_pe(0.047, 0.029, 0.065),
        reported=_pe(0.05, 0.03, 0.07),
    )
    assert not out["near_zero"]
    assert out["wording"] == OUTSIDE
    assert "no claim of advantage" not in out["wording"]
    # The terms and their paired difference are shown, engaged or not.
    assert out["applicable"]
    assert out["reported"]["estimate"] == pytest.approx(0.05)
    assert out["splat_pool"]["estimate"] == pytest.approx(0.001)
    assert out["path_difference"]["estimate"] == pytest.approx(0.047)
    assert out["paths_agree_in_sign"] is True
    assert out["both_intervals_exclude_zero"] is False


@pytest.mark.parametrize(
    "per_point, splat",
    [
        # Per-point term inside the band, splat term outside it.
        (_pe(0.001, 0.0005, 0.0015), _pe(0.030, 0.020, 0.040)),
        # Splat term inside the band, per-point term outside it.
        (_pe(0.030, 0.020, 0.040), _pe(0.001, 0.0005, 0.0015)),
        # Both terms inside the band.
        (_pe(0.001, 0.0005, 0.0015), _pe(0.0012, 0.0006, 0.0018)),
        # Both terms inside the band and disagreeing in sign.
        (_pe(0.001, 0.0005, 0.0015), _pe(-0.001, -0.0015, -0.0005)),
    ],
)
def test_a_path_term_alone_never_engages_the_wording(per_point, splat):
    """Engagement reads the reported effect only. The terms never engage it."""
    out = near_zero_disclosure(
        "delta_learn_pp", per_point, splat, ANALYSIS,
        reported=_pe(0.020, 0.015, 0.025),
    )
    assert not out["near_zero"]
    assert out["wording"] == OUTSIDE


def test_a_large_per_point_term_with_a_near_zero_splat_term_is_not_engaged():
    """The same defect with reported left to default to the per-point term."""
    out = near_zero_disclosure(
        "delta_learn_pp", _pe(0.05, 0.03, 0.07), _pe(0.001, -0.002, 0.004), ANALYSIS,
    )
    assert out["reported"]["estimate"] == pytest.approx(0.05)
    assert not out["near_zero"]
    assert out["wording"] == OUTSIDE


@pytest.mark.parametrize(
    "per_point, splat, expected",
    [
        # 1. The veto: the terms differ in sign. Neither is in the band, so
        #    the veto must outrank wording 4.
        (_pe(0.012, 0.010, 0.014), _pe(-0.011, -0.013, -0.009), VETO),
        # 1. The veto: an interval includes zero, with neither term in band.
        (_pe(0.012, -0.001, 0.025), _pe(0.011, 0.009, 0.013), VETO),
        # 1. The veto outranks both terms sitting in the band.
        (_pe(0.001, -0.0005, 0.0025), _pe(0.0012, 0.0006, 0.0018), VETO),
        # 2. Both terms inside the band.
        (_pe(0.001, 0.0005, 0.0015), _pe(0.0012, 0.0006, 0.0018), SMALL),
        # 2. The same, negative.
        (_pe(-0.001, -0.0015, -0.0005), _pe(-0.0012, -0.0018, -0.0006), SMALL),
        # 3. Exactly one term inside the band: the per-point one.
        (_pe(0.001, 0.0005, 0.0015), _pe(0.030, 0.020, 0.040), PATH_SENSITIVE),
        # 3. Exactly one term inside the band: the splat one.
        (_pe(0.030, 0.020, 0.040), _pe(0.001, 0.0005, 0.0015), PATH_SENSITIVE),
        # 4. Neither term inside the band. The paths agree with each other.
        (_pe(0.012, 0.010, 0.014), _pe(0.011, 0.009, 0.013), SUPPORT_SENSITIVE),
    ],
)
def test_the_engaged_two_path_wordings_in_order(per_point, splat, expected):
    out = near_zero_disclosure(
        "delta_learn_pp", per_point, splat, ANALYSIS,
        reported=_pe(0.0015, 0.0010, 0.0020),
    )
    assert out["near_zero"]
    assert out["wording"] == expected


def test_neither_term_in_band_does_not_say_the_paths_disagree():
    """The previous code printed the path-sensitive sentence here.

    That sentence says the two paths disagree about the band. Both are outside
    it, so they agree. Decision 2's wording 4 says what is true instead.
    """
    out = near_zero_disclosure(
        "delta_learn_pp", _pe(0.012, 0.010, 0.014), _pe(0.011, 0.009, 0.013),
        ANALYSIS, reported=_pe(0.0015, 0.0010, 0.0020),
    )
    assert out["wording"] == SUPPORT_SENSITIVE
    assert "path-sensitive" not in out["wording"]
    assert "do not agree" not in out["wording"]


@pytest.mark.parametrize(
    "reported, near_zero, expected",
    [
        # Inside the band, interval clear of zero.
        (_pe(0.0012, 0.0009, 0.0015), True, ONLY_PATH_CLEAR),
        # Inside the band, negative, interval clear of zero.
        (_pe(-0.0012, -0.0015, -0.0009), True, ONLY_PATH_CLEAR),
        # Inside the band, interval including zero.
        (_pe(0.0012, -0.0004, 0.0028), True, ONLY_PATH_ZERO),
        # Interval touching zero is not clear of it.
        (_pe(0.0012, 0.0, 0.0024), True, ONLY_PATH_ZERO),
        # Outside the band.
        (_pe(0.012, 0.009, 0.015), False, OUTSIDE),
    ],
)
def test_the_single_path_wordings_follow_the_reported_interval(
    reported, near_zero, expected
):
    out = near_zero_disclosure("delta_formulation", reported, None, ANALYSIS)
    assert out["near_zero"] is near_zero
    assert out["wording"] == expected
    assert out["splat_pool"] is None


def test_the_single_path_interval_is_the_reported_one():
    """The single-path wording reads the reported interval, not the term's."""
    out = near_zero_disclosure(
        "delta_formulation", _pe(0.0012, -0.0004, 0.0028), None, ANALYSIS,
        reported=_pe(0.0012, 0.0009, 0.0015),
    )
    assert out["wording"] == ONLY_PATH_CLEAR


@pytest.mark.parametrize(
    "per_point, splat",
    [
        # The cross-path set is empty, so both terms are undefined.
        (_pe(NAN, NAN, NAN), _pe(NAN, NAN, NAN)),
        # Only the per-point term is undefined.
        (_pe(NAN, NAN, NAN), _pe(0.001, 0.0005, 0.0015)),
        # Only the splat term is undefined. The quantity still has a second
        # path, so the "by construction" sentence would be false here.
        (_pe(0.001, 0.0005, 0.0015), _pe(NAN, NAN, NAN)),
    ],
)
def test_nonfinite_terms_with_a_reported_effect_in_band_get_the_veto(per_point, splat):
    """The previous code returned "not estimable" with near_zero False, even
    when the reported effect sat inside the band. Decision 2 engages on the
    reported effect, so that cell is disclosed.

    The sentence is 3.9's veto. PROTOCOL 3.9 restricts the interpretation of
    such an effect to content the two paths share, and here they share none.
    Of decision 2's six sentences, the veto is the only one that is true of
    this cell. Sign agreement and clearance stay None, because they are
    undefined rather than failed."""
    inside = near_zero_disclosure(
        "delta_learn_pp", per_point, splat, ANALYSIS,
        reported=_pe(0.0015, 0.0010, 0.0020),
    )
    assert inside["near_zero"]
    assert inside["wording"] == VETO
    assert inside["paths_agree_in_sign"] is None
    assert inside["both_intervals_exclude_zero"] is None
    outside = near_zero_disclosure(
        "delta_learn_pp", per_point, splat, ANALYSIS,
        reported=_pe(0.05, 0.03, 0.07),
    )
    assert not outside["near_zero"]
    assert outside["wording"] == OUTSIDE


@pytest.mark.parametrize("splat", [_pe(0.0012, 0.0006, 0.0018), None])
def test_a_nonfinite_reported_effect_is_never_engaged(splat):
    """Even with both terms finite and inside the band."""
    out = near_zero_disclosure(
        "delta_learn_pp", _pe(0.001, 0.0005, 0.0015), splat, ANALYSIS,
        reported=_pe(NAN, NAN, NAN),
    )
    assert out["applicable"]
    assert not out["near_zero"]
    assert out["wording"] == NOT_ESTIMABLE


@pytest.mark.parametrize(
    "per_point, splat, reported",
    [
        (_pe(0.001, 0.0005, 0.0015), _pe(0.0012, 0.0006, 0.0018), None),
        (_pe(0.05, 0.03, 0.07), _pe(0.001, -0.002, 0.004), None),
        (_pe(0.012, 0.010, 0.014), _pe(0.011, 0.009, 0.013),
         _pe(0.0015, 0.0010, 0.0020)),
        (_pe(0.0012, 0.0009, 0.0015), None, None),
        (_pe(0.012, 0.009, 0.015), None, None),
        (_pe(NAN, NAN, NAN), _pe(NAN, NAN, NAN), _pe(0.0015, 0.0010, 0.0020)),
        (_pe(NAN, NAN, NAN), None, None),
        (_pe(0.001, 0.0005, 0.0015), _pe(0.001, 0.0005, 0.0015),
         _pe(NAN, NAN, NAN)),
    ],
)
def test_every_applicable_branch_returns_every_disclosure_key(
    per_point, splat, reported
):
    out = near_zero_disclosure(
        "delta_learn_pp", per_point, splat, ANALYSIS, reported=reported,
    )
    assert DISCLOSURE_KEYS <= set(out), DISCLOSURE_KEYS - set(out)
    assert out["band"] == ANALYSIS.path_agreement_tolerance
    assert out["wording"] in ALL_WORDINGS


def test_the_flag_is_exactly_the_reported_effect_in_band():
    """near_zero is engagement, and engagement is the reported effect alone."""
    terms = [
        _pe(0.001, 0.0005, 0.0015), _pe(-0.001, -0.0015, -0.0005),
        _pe(0.030, 0.020, 0.040), _pe(0.001, -0.001, 0.003),
    ]
    for reported in (_pe(0.0015, 0.001, 0.002), _pe(0.020, 0.015, 0.025),
                     _pe(-0.003, -0.004, -0.002), _pe(-0.0031, -0.004, -0.002)):
        for per_point in terms:
            for splat in terms + [None]:
                out = near_zero_disclosure(
                    "delta_learn_pp", per_point, splat, ANALYSIS, reported=reported,
                )
                assert out["near_zero"] is (abs(reported.estimate) <= 0.003)


def test_no_new_wording_claims_equivalence():
    for wording in ALL_WORDINGS:
        assert "equivalen" not in wording.lower()


def test_a_large_headline_gap_with_a_null_splat_term_is_not_engaged_end_to_end():
    """Decision 2's motivating case, through evaluate_quantity.

    The headline gap is 0.07 and clearly nonzero. The cross-path splat term is
    zero on average with an interval that includes zero. The previous code
    vetoed this cell. Under decision 2 it is outside the band.
    """
    records = make_records(n_scenes=8, pairs_per_scene=5, noise=1e-6)
    for r in records:
        sign = 1.0 if int(r["scene"].split("_")[1]) % 2 == 0 else -1.0
        r["x_sp_transport_centered"] = 0.5000 + sign * 0.001
        r["x_sp_predict_centered"] = 0.5000
    cell = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    d = cell.disclosure
    assert cell.estimate == pytest.approx(0.07, abs=1e-4)
    assert cell.lo > 0.0
    assert abs(d["splat_pool"]["estimate"]) <= ANALYSIS.path_agreement_tolerance
    assert d["splat_pool"]["lo"] <= 0.0 <= d["splat_pool"]["hi"]
    assert not d["near_zero"]
    assert d["wording"] == OUTSIDE
    assert cell.as_row()["near_zero"] is False


def test_the_two_gaps_engage_on_their_own_reported_effects():
    """delta_learn_pp and delta_learn_sp share their terms, not their trigger."""
    records = make_records(n_scenes=8, pairs_per_scene=5, noise=1e-6)
    for r in records:
        # Headline gap inside the band. The operational gap stays at 0.01.
        r["cl_centered"], r["predict_centered"] = 0.5015, 0.5000
    pp = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS).disclosure
    sp = evaluate_quantity(records, "delta_learn_sp", "centered", ANALYSIS).disclosure
    assert pp["per_point"] == sp["per_point"]
    assert pp["splat_pool"] == sp["splat_pool"]
    assert pp["near_zero"] and not sp["near_zero"]
    assert sp["wording"] == OUTSIDE
    assert pp["wording"] != OUTSIDE


def test_an_empty_intersection_with_a_reported_effect_in_band_is_disclosed():
    """A cell with no cross-path set is still engaged, and gets 3.9's veto.

    Its two paths share no cell, so no shared content licenses a claim.
    """
    records = make_records(n_scenes=6, pairs_per_scene=4, noise=1e-6)
    for r in records:
        r["cl_centered"], r["predict_centered"] = 0.5015, 0.5000
        r["n_intersect"] = 0
    cell = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    assert math.isnan(cell.disclosure["per_point"]["estimate"])
    assert cell.disclosure["near_zero"]
    assert cell.disclosure["wording"] == VETO
    assert cell.as_row()["near_zero"] is True
    assert cell.as_row()["near_zero_wording"] == VETO


@pytest.mark.parametrize(
    "undefined",
    [
        # The estimate is not finite, so the term is not finite.
        _pe(NAN, NAN, NAN),
        # The estimate is finite and its interval is not.
        _pe(0.001, NAN, NAN),
    ],
    ids=["nan estimate", "nan interval"],
)
@pytest.mark.parametrize("side", ["per_point", "splat"])
def test_an_undefined_term_always_gives_the_veto_when_engaged(undefined, side):
    """A term with an undefined estimate and a term with an undefined interval
    both leave the veto as the only licensed sentence. An interval that is not
    there does not clear zero."""
    defined = _pe(0.0012, 0.0006, 0.0018)
    per_point, splat = (undefined, defined) if side == "per_point" else (defined, undefined)
    out = near_zero_disclosure(
        "delta_learn_pp", per_point, splat, ANALYSIS,
        reported=_pe(0.0015, 0.0010, 0.0020),
    )
    assert out["near_zero"]
    assert out["wording"] == VETO


# ---------------------------------------------------------------------------
# The engaged sentences are a closed set: decision 2's six, verbatim
# ---------------------------------------------------------------------------

REPORTING_RULES = (
    Path(__file__).resolve().parents[1]
    / "validation" / "evidence" / "phase5" / "reporting_rules.md"
)


def _decision_two_sentences() -> tuple[str, ...]:
    """The six sentences decision 2 quotes under "The wording, when engaged".

    Read from the user's recorded decisions, with whitespace collapsed, so the
    code is held to the file rather than to a copy of it.
    """
    text = " ".join(REPORTING_RULES.read_text(encoding="utf-8").split())
    start = text.index("### The wording, when engaged")
    end = text.index("### How outcomes are called")
    return tuple(re.findall(r'"([^"]+)"', text[start:end]))


def test_the_engaged_sentences_are_decision_twos_six_verbatim():
    sentences = _decision_two_sentences()
    assert len(sentences) == 6, sentences
    assert set(sentences) == set(ENGAGED_WORDINGS)


def test_the_module_holds_decision_twos_sentences_and_two_carried_labels_only():
    """Every WORDING_ constant is either one of decision 2's six sentences or one
    of the two labels carried over from cc20e3e, which mark a cell that is not
    engaged. A sentence added outside decision 2 fails here."""
    import lot.phase5_estimands as estimands

    constants = {
        name: value for name, value in vars(estimands).items()
        if name.startswith("WORDING_")
    }
    carried = {"WORDING_OUTSIDE_BAND": OUTSIDE, "WORDING_NOT_ESTIMABLE": NOT_ESTIMABLE}
    decision_two = {name: value for name, value in constants.items() if name not in carried}
    assert set(decision_two.values()) == set(_decision_two_sentences()), decision_two
    assert len(decision_two) == 6, sorted(decision_two)
    assert {name: constants.get(name) for name in carried} == carried


_SWEEP_TERMS = (
    _pe(0.0012, 0.0006, 0.0018),      # in band, clear of zero
    _pe(-0.0012, -0.0018, -0.0006),   # in band, negative, clear of zero
    _pe(0.0012, -0.0004, 0.0028),     # in band, interval includes zero
    _pe(0.030, 0.020, 0.040),         # outside the band, clear of zero
    _pe(-0.030, -0.040, -0.020),      # outside the band, negative
    _pe(0.0, -0.001, 0.001),          # exactly zero
    _pe(NAN, NAN, NAN),               # undefined estimate
    _pe(0.0012, NAN, NAN),            # undefined interval
)
_SWEEP_REPORTED = _SWEEP_TERMS + (None,)


@pytest.mark.parametrize("quantity", ["delta_learn_pp", "cl_margin", "delta_formulation"])
def test_every_engaged_wording_is_one_of_decision_twos_six(quantity):
    """Swept over reported effects and both terms, NaN estimates and NaN
    intervals included. Engaged, the wording is one of decision 2's six.
    Not engaged, it is one of the two carried labels."""
    sentences = set(_decision_two_sentences())
    splats = (None,) if quantity == "delta_formulation" else _SWEEP_TERMS
    for reported in _SWEEP_REPORTED:
        for per_point in _SWEEP_TERMS:
            for splat in splats:
                out = near_zero_disclosure(
                    quantity, per_point, splat, ANALYSIS, reported=reported,
                )
                if out["near_zero"]:
                    assert out["wording"] in sentences, (reported, per_point, splat)
                else:
                    assert out["wording"] in NOT_ENGAGED_LABELS, (reported, per_point, splat)


# ---------------------------------------------------------------------------
# The Mean-Feature floor, beside No-Warp-Copy on every record, per CLAUDE.md
# ---------------------------------------------------------------------------

# Each Mean-Feature cell, the field it reads, and its population.
MEAN_FEATURE_CELLS = {
    "mean_feature": ("meanfeat_raw", PER_POINT),
    "sp_mean_feature": ("sp_meanfeat_raw", SPLAT_POOL),
    "x_mean_feature": ("x_meanfeat_raw", CROSS_PATH),
    "x_sp_mean_feature": ("x_sp_meanfeat_raw", CROSS_PATH),
}


def test_the_registry_carries_mean_feature_under_raw_metrics_only():
    """PROTOCOL 3.7: Mean-Feature predicts the centering vector, so its centered
    score is not applicable. The columns do not exist rather than being filled."""
    from lot.phase5_estimands import (
        ALL_FIELDS, INTERSECTION_FIELDS, PRIMARY_FIELDS, SPLAT_FIELDS,
    )

    assert {"meanfeat_raw", "meanfeat_l2_raw"} <= set(PRIMARY_FIELDS)
    assert {"sp_meanfeat_raw", "sp_meanfeat_l2_raw"} <= set(SPLAT_FIELDS)
    assert {"x_meanfeat_raw", "x_meanfeat_l2_raw",
            "x_sp_meanfeat_raw", "x_sp_meanfeat_l2_raw"} <= set(INTERSECTION_FIELDS)
    assert not [f for f in ALL_FIELDS if "meanfeat" in f and "centered" in f]
    assert len(set(ALL_FIELDS)) == len(ALL_FIELDS)


def test_each_mean_feature_cell_reads_its_own_records_floor():
    forms = quantity_formulas("raw")
    means = {"meanfeat_raw": 0.31, "sp_meanfeat_raw": 0.33,
             "x_meanfeat_raw": 0.35, "x_sp_meanfeat_raw": 0.37}
    for quantity, (field, population) in MEAN_FEATURE_CELLS.items():
        assert forms[quantity](means) == pytest.approx(means[field]), quantity
        assert QUANTITY_POPULATION[quantity] == population, quantity


def test_the_mean_feature_cells_exist_under_raw_cosine_only():
    centered = quantity_formulas("centered")
    for quantity in MEAN_FEATURE_CELLS:
        assert quantity not in centered, quantity


def test_a_mean_feature_cell_is_a_floor_and_never_gets_near_zero_wording():
    """A floor is an absolute level, not a claim of one method over another. Its
    value here sits inside the 0.003 band, and the near-zero rule still does not
    apply to it."""
    from lot.phase5_estimands import INTERPRETED_EFFECTS

    records = make_records(n_scenes=8, pairs_per_scene=5)
    for record in records:
        for field, _ in MEAN_FEATURE_CELLS.values():
            record[field] = 0.001
    for quantity, (_, population) in MEAN_FEATURE_CELLS.items():
        assert quantity not in INTERPRETED_EFFECTS, quantity
        cell = evaluate_quantity(records, quantity, "raw", ANALYSIS)
        assert cell.estimate == pytest.approx(0.001), quantity
        assert cell.population == population, quantity
        assert cell.n_camera_pairs == 40, quantity
        assert not cell.disclosure["applicable"], quantity
        assert not cell.disclosure["near_zero"], quantity
        assert cell.as_row()["near_zero_wording"] == "", quantity


# ---------------------------------------------------------------------------
# The two floors on the formulation support, reporting_rules.md section 6
# ---------------------------------------------------------------------------
#
# No-Warp-Copy under both metrics and Mean-Feature under raw metrics only, each
# a level on V_form. They are not margins and add no interpreted effect.

# The formulation record's columns before the floors were added, in order.
FORMULATION_BASE_FIELDS = (
    "tl_form_raw", "tl_form_centered", "tl_form_l2_raw", "tl_form_l2_centered",
    "cl_form_raw", "cl_form_centered", "cl_form_l2_raw", "cl_form_l2_centered",
)
FORMULATION_FLOOR_FIELDS = (
    "nowarp_form_raw", "nowarp_form_centered",
    "nowarp_form_l2_raw", "nowarp_form_l2_centered",
    "meanfeat_form_raw", "meanfeat_form_l2_raw",
)
# Each formulation floor cell, and the column it reads under each metric.
FORMULATION_FLOOR_CELLS = {
    "no_warp_copy_form": {"raw": "nowarp_form_raw", "centered": "nowarp_form_centered"},
    "mean_feature_form": {"raw": "meanfeat_form_raw"},
}
# Every quantity on the formulation population. A margin over a floor here
# would be a new interpreted effect, which reporting_rules.md does not add.
FORMULATION_QUANTITIES = {
    "delta_formulation", "tl_reference", "cl_on_formulation_support",
    "no_warp_copy_form", "mean_feature_form",
}


def _with_formulation_floors(records: list[dict], nowarp: float, meanfeat: float) -> list[dict]:
    """Add the floor columns as constants. No other column of any record moves."""
    for r in records:
        r.update({
            "nowarp_form_raw": nowarp, "nowarp_form_centered": nowarp - 0.1,
            "nowarp_form_l2_raw": 1.0 - nowarp, "nowarp_form_l2_centered": 1.1 - nowarp,
            "meanfeat_form_raw": meanfeat, "meanfeat_form_l2_raw": 1.0 - meanfeat,
        })
    return records


def test_the_formulation_floors_are_appended_to_the_formulation_record():
    from lot.phase5_estimands import ALL_FIELDS, FORMULATION_FIELDS

    assert FORMULATION_FIELDS == FORMULATION_BASE_FIELDS + FORMULATION_FLOOR_FIELDS
    assert not [f for f in FORMULATION_FIELDS if "meanfeat" in f and "centered" in f]
    assert set(FORMULATION_FLOOR_FIELDS) <= set(ALL_FIELDS)
    assert len(set(ALL_FIELDS)) == len(ALL_FIELDS)


def test_each_formulation_floor_cell_reads_its_own_column():
    means = {"nowarp_form_raw": 0.41, "nowarp_form_centered": 0.21,
             "meanfeat_form_raw": 0.33}
    for metric in ("raw", "centered"):
        forms = quantity_formulas(metric)
        for quantity, columns in FORMULATION_FLOOR_CELLS.items():
            if metric not in columns:
                assert quantity not in forms, (quantity, metric)
                continue
            assert forms[quantity](means) == means[columns[metric]], (quantity, metric)
            assert QUANTITY_POPULATION[quantity] == FORMULATION, quantity


def test_the_formulation_floors_are_levels_and_add_no_interpreted_effect():
    from lot.phase5_estimands import (
        DISCLOSURE_PAIR, INTERPRETED_EFFECTS, SINGLE_PATH_BY_CONSTRUCTION,
    )

    on_formulation = {q for q, p in QUANTITY_POPULATION.items() if p == FORMULATION}
    assert on_formulation == FORMULATION_QUANTITIES
    for quantity in FORMULATION_FLOOR_CELLS:
        assert quantity not in INTERPRETED_EFFECTS, quantity
        assert quantity not in DISCLOSURE_PAIR, quantity
        assert quantity not in SINGLE_PATH_BY_CONSTRUCTION, quantity
    assert INTERPRETED_EFFECTS & on_formulation == {"delta_formulation"}


def test_a_formulation_floor_cell_is_a_level_on_the_formulation_support():
    """Inside the 0.003 band, and still no near-zero wording, because a floor is
    an absolute level and not a claim of one method over another."""
    records = _with_formulation_floors(
        make_records(n_scenes=6, pairs_per_scene=5), nowarp=0.002, meanfeat=0.001
    )
    for record in records[:10]:
        record["n_formulation"] = 0
    for quantity, columns in FORMULATION_FLOOR_CELLS.items():
        for metric, column in columns.items():
            cell = evaluate_quantity(records, quantity, metric, ANALYSIS)
            assert cell.population == FORMULATION, quantity
            assert cell.estimate == pytest.approx(records[0][column]), (quantity, metric)
            assert cell.n_camera_pairs == 20, quantity
            assert cell.n_feature_comparisons == 20 * 700, quantity
            assert not cell.disclosure["applicable"], quantity
            assert not cell.disclosure["near_zero"], quantity
            assert cell.as_row()["near_zero_wording"] == "", quantity


def test_appending_the_floors_moves_no_existing_formulation_cell():
    """Each pre-existing formulation cell equals the paired interval over the
    original eight fields, with or without the floor columns on the records.

    The point estimate is exact, because each field's mean is pooled on its own.
    The interval is held to 1e-12 rather than to the bit, because a replicate's
    means are a matrix product whose width grew, and BLAS may round it
    differently.
    """
    from lot.paired_bootstrap import paired_interval

    bare = make_records(n_scenes=8, pairs_per_scene=6)
    floored = _with_formulation_floors(
        make_records(n_scenes=8, pairs_per_scene=6), nowarp=0.45, meanfeat=0.31
    )
    for metric in ("raw", "centered"):
        forms = quantity_formulas(metric)
        for quantity in ("delta_formulation", "tl_reference", "cl_on_formulation_support"):
            base = paired_interval(
                bare, FORMULATION_BASE_FIELDS, forms[quantity],
                resamples=ANALYSIS.bootstrap_resamples, seed=ANALYSIS.bootstrap_seed,
                confidence=ANALYSIS.bootstrap_confidence,
            )
            for records in (bare, floored):
                cell = evaluate_quantity(records, quantity, metric, ANALYSIS)
                assert cell.estimate == base["estimate"], (quantity, metric)
                assert cell.lo == pytest.approx(base["lo"], abs=1e-12), (quantity, metric)
                assert cell.hi == pytest.approx(base["hi"], abs=1e-12), (quantity, metric)
                assert cell.n_replicates == base["n_replicates"], (quantity, metric)


# ---------------------------------------------------------------------------
# No-Warp-Copy on the splat-pool support, reporting_rules.md section 6
# ---------------------------------------------------------------------------
#
# "No-Warp-Copy sits beside every reported metric." The splat-pool table reports
# the two arms as levels, so the floor is a level beside them. Specification
# step 38 lists it in that table. The two splat-pool margins already subtract
# the same column, so the level adds no new record field and no interpreted
# effect.

# Every quantity on the splat-pool population. The first six existed before
# the floor's level was added.
SPLAT_POOL_QUANTITIES = {
    "delta_learn_sp", "sp_transport", "sp_predict", "sp_transport_margin",
    "sp_predict_margin", "sp_mean_feature", "sp_no_warp_copy",
}


def test_the_splat_pool_no_warp_copy_level_reads_the_floor_column():
    from lot.phase5_estimands import (
        DISCLOSURE_PAIR, INTERPRETED_EFFECTS, SINGLE_PATH_BY_CONSTRUCTION,
    )

    means = {"sp_nowarp_raw": 0.51, "sp_nowarp_centered": 0.31}
    for metric in ("raw", "centered"):
        forms = quantity_formulas(metric)
        assert forms["sp_no_warp_copy"](means) == means[f"sp_nowarp_{metric}"], metric
    assert QUANTITY_POPULATION["sp_no_warp_copy"] == SPLAT_POOL
    assert "sp_no_warp_copy" not in INTERPRETED_EFFECTS
    assert "sp_no_warp_copy" not in DISCLOSURE_PAIR
    assert "sp_no_warp_copy" not in SINGLE_PATH_BY_CONSTRUCTION
    on_splat = {q for q, p in QUANTITY_POPULATION.items() if p == SPLAT_POOL}
    assert on_splat == SPLAT_POOL_QUANTITIES


def test_the_splat_pool_floor_is_a_level_on_its_own_support():
    """Inside the 0.003 band, and still no near-zero wording, because a floor is
    an absolute level and not a claim of one method over another."""
    records = make_records(n_scenes=6, pairs_per_scene=5)
    for record in records:
        record["sp_nowarp_raw"] = record["sp_nowarp_centered"] = 0.002
    for record in records[:10]:
        record["n_splat"] = 0
    for metric in ("raw", "centered"):
        cell = evaluate_quantity(records, "sp_no_warp_copy", metric, ANALYSIS)
        assert cell.estimate == pytest.approx(0.002), metric
        assert cell.population == SPLAT_POOL
        assert cell.n_camera_pairs == 20
        assert cell.n_feature_comparisons == 20 * 800
        assert math.isfinite(cell.lo) and math.isfinite(cell.hi)
        assert not cell.disclosure["applicable"]
        assert not cell.disclosure["near_zero"]
        assert cell.as_row()["near_zero_wording"] == ""


def test_the_splat_pool_floor_is_what_both_splat_margins_subtract():
    records = make_records(n_scenes=8, pairs_per_scene=5)
    for metric in ("raw", "centered"):
        def estimate(quantity):
            return evaluate_quantity(records, quantity, metric, ANALYSIS).estimate

        floor = estimate("sp_no_warp_copy")
        assert floor == pytest.approx(
            estimate("sp_transport") - estimate("sp_transport_margin"), abs=1e-12
        )
        assert floor == pytest.approx(
            estimate("sp_predict") - estimate("sp_predict_margin"), abs=1e-12
        )


def test_adding_the_splat_pool_floor_moves_no_existing_splat_cell():
    """Each pre-existing splat-pool cell equals the paired interval over the
    unchanged splat-pool fields, to the bit. The level adds no field."""
    from lot.paired_bootstrap import paired_interval
    from lot.phase5_estimands import SPLAT_FIELDS

    records = make_records(n_scenes=8, pairs_per_scene=6)
    for metric in ("raw", "centered"):
        forms = quantity_formulas(metric)
        existing = sorted(q for q in SPLAT_POOL_QUANTITIES - {"sp_no_warp_copy"} if q in forms)
        assert existing, metric
        for quantity in existing:
            base = paired_interval(
                records, SPLAT_FIELDS, forms[quantity],
                resamples=ANALYSIS.bootstrap_resamples, seed=ANALYSIS.bootstrap_seed,
                confidence=ANALYSIS.bootstrap_confidence,
            )
            cell = evaluate_quantity(records, quantity, metric, ANALYSIS)
            assert cell.estimate == base["estimate"], (quantity, metric)
            assert cell.lo == base["lo"], (quantity, metric)
            assert cell.hi == base["hi"], (quantity, metric)
            assert cell.n_replicates == base["n_replicates"], (quantity, metric)


def test_every_support_carries_a_no_warp_copy_level_under_each_metric():
    levels = {"no_warp_copy": PER_POINT, "no_warp_copy_form": FORMULATION,
              "sp_no_warp_copy": SPLAT_POOL}
    for metric in ("raw", "centered"):
        forms = quantity_formulas(metric)
        for quantity, population in levels.items():
            assert quantity in forms, (quantity, metric)
            assert QUANTITY_POPULATION[quantity] == population, quantity
