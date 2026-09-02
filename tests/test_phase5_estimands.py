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
                # The cross-path common-valid columns. Both paths re-scored on
                # the cells they share, which is the population PROTOCOL 3.9's
                # disclosure is computed on.
                **_intersection(cl + level, predict + level, nowarp + level,
                                0.7 + level, 0.69 + level, 0.5 + level),
                "n_primary": 900, "n_formulation": 700, "n_splat": 800,
                "n_intersect": 640, "n_predict_nonfinite": 0,
            })
    return records


def _intersection(cl, predict, nowarp, sp_transport, sp_predict, sp_nowarp) -> dict:
    """All six cross-path arms with all four columns, cosines set and L2 finite."""
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
    """A quantity with no counterpart is disclosed at the weakest licence."""
    out = near_zero_disclosure("delta_learn_pp", _pe(0.0012, 0.0009, 0.0015), None,
                               ANALYSIS)
    assert out["near_zero"]
    assert out["splat_pool"] is None
    assert out["wording"] == (
        "no measurable difference at the reported scale; this quantity "
        "has no second evaluation path by construction"
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
    assert disclosure["wording"] == "small, sign-consistent effect"
    # The headline estimand itself still comes from the primary support.
    assert cell.estimate == pytest.approx(0.40, abs=1e-6)


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
        # Per-point margin tiny and positive; splat margin large and negative.
        r["x_cl_centered"], r["x_nowarp_centered"] = 0.5020, 0.5000
        r["x_sp_transport_centered"], r["x_sp_nowarp_centered"] = 0.40, 0.45
    cell = evaluate_quantity(records, "cl_margin", "centered", ANALYSIS)
    d = cell.disclosure
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
    records = make_records(n_scenes=8, pairs_per_scene=5, noise=1e-6)
    for r in records:
        r["tl_form_centered"], r["cl_form_centered"] = 0.6010, 0.6000
    cell = evaluate_quantity(records, "delta_formulation", "centered", ANALYSIS)
    assert cell.disclosure["splat_pool"] is None
    assert "by construction" in cell.disclosure["wording"]
    assert cell.disclosure["wording"] != "small, sign-consistent effect"


def test_margin_path_differences_are_their_own_paired_quantities():
    records = make_records(n_scenes=8, pairs_per_scene=5, scene_spread=0.2, noise=1e-6)
    for name in ("cl_margin", "predict_margin"):
        d = evaluate_quantity(records, name, "centered", ANALYSIS).disclosure
        diff = d["path_difference"]
        assert math.isfinite(diff["lo"]) and math.isfinite(diff["hi"])
        assert diff["estimate"] == pytest.approx(
            d["per_point"]["estimate"] - d["splat_pool"]["estimate"], abs=1e-9
        )
