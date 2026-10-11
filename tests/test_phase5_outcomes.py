"""reporting_rules.md decision 2: how Phase 5 outcomes are called.

Truth tables for the outcome of a cell, the qualifier it carries, the outcome
49 flag, the metric-sensitive flag, the landing-offset flags, and the measured
outcome per regime. The cells are built by hand, so every branch is reached on
purpose, and then once through lot.phase5_estimands.evaluate_quantity, so the
rules are known to read what the estimand layer produces.

The code lives in lot.phase5_outcomes, which decision 1 freezes at commit E.
These tests moved with it from tests/test_phase5_report.py. Decision 5's
control label and stable validation curve, and section 9's rule that a
supported estimate outside its own interval stops, are pinned here too.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from lot.analysis_config import load_analysis_config
from lot.phase5_estimands import (
    QUANTITY_POPULATION,
    WORDING_SMALL_SIGN_CONSISTENT,
    WORDING_VETO,
    CellResult,
    evaluate_quantity,
)
from lot.phase5_outcomes import (
    METRIC_SENSITIVE,
    METRICS,
    OUTCOME_EXPLICIT_WINS,
    OUTCOME_LEARNED_WINS,
    OUTCOME_NO_MEASURABLE_GAP,
    OUTCOME_WORDING,
    POOLED_LABEL,
    POOLED_SCOPE,
    REGIME_SCOPES,
    SCOPES,
    OutcomeAnomaly,
    call_outcome,
    gap_outcome,
    landing_offset_flags,
    measured_outcome,
    metric_sensitive,
    outcome_49,
)

from test_phase5_estimands import make_records

ANALYSIS = load_analysis_config()
NAN = float("nan")
REPORTING_RULES = (
    Path(__file__).resolve().parents[1]
    / "validation" / "evidence" / "phase5" / "reporting_rules.md"
)


def cell(quantity="delta_learn_pp", metric="centered", estimate=0.05, lo=0.03, hi=0.07,
         supported=True, near_zero=False, wording="effect outside the operator band",
         disclosure=None):
    """One CellResult as evaluate_quantity returns it, built by hand."""
    if disclosure is None:
        disclosure = {"applicable": True, "near_zero": near_zero, "wording": wording,
                      "per_point": {"estimate": 0.011, "lo": 0.009, "hi": 0.013},
                      "splat_pool": {"estimate": 0.004, "lo": -0.001, "hi": 0.009},
                      "path_difference": {"estimate": 0.007, "lo": 0.002, "hi": 0.012}}
    return CellResult(
        quantity=quantity, metric=metric, population=QUANTITY_POPULATION[quantity],
        estimate=estimate, lo=lo, hi=hi, n_scenes=18 if supported else 2,
        n_camera_pairs=900 if supported else 4, n_feature_comparisons=640_000,
        n_replicates=1000, supported=supported, disclosure=disclosure,
    )


# ---------------------------------------------------------------------------
# The outcome of one cell, specification steps 46 to 48
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("estimate, lo, hi, expected", [
    # The interval excludes zero and the estimate is positive.
    (0.05, 0.03, 0.07, OUTCOME_EXPLICIT_WINS),
    (0.0004, 0.0001, 0.0007, OUTCOME_EXPLICIT_WINS),
    # The interval excludes zero and the estimate is negative.
    (-0.05, -0.07, -0.03, OUTCOME_LEARNED_WINS),
    (-0.0004, -0.0007, -0.0001, OUTCOME_LEARNED_WINS),
    # The interval includes zero.
    (0.01, -0.01, 0.03, OUTCOME_NO_MEASURABLE_GAP),
    (-0.01, -0.03, 0.01, OUTCOME_NO_MEASURABLE_GAP),
    (0.0, -0.01, 0.01, OUTCOME_NO_MEASURABLE_GAP),
    # An endpoint exactly at zero is not clear of it.
    (0.02, 0.0, 0.04, OUTCOME_NO_MEASURABLE_GAP),
    (-0.02, -0.04, 0.0, OUTCOME_NO_MEASURABLE_GAP),
])
def test_a_supported_cell_is_called_from_its_own_interval(estimate, lo, hi, expected):
    assert call_outcome(estimate, lo, hi, supported=True) == expected


@pytest.mark.parametrize("estimate, lo, hi", [
    (0.05, 0.03, 0.07), (0.01, -0.01, 0.03), (NAN, NAN, NAN), (0.05, NAN, NAN),
])
def test_an_unsupported_cell_is_never_classified(estimate, lo, hi):
    """It is shown with its counts and not classified, whatever its values."""
    assert call_outcome(estimate, lo, hi, supported=False) is None


@pytest.mark.parametrize("estimate, lo, hi", [
    (NAN, 0.03, 0.07),
    (0.05, NAN, 0.07),
    (0.05, 0.03, NAN),
    (math.inf, 0.03, 0.07),
])
def test_a_supported_cell_without_a_finite_interval_is_an_anomaly(estimate, lo, hi):
    with pytest.raises(OutcomeAnomaly, match="finite"):
        call_outcome(estimate, lo, hi, supported=True)


@pytest.mark.parametrize("estimate, lo, hi", [
    # The interval sits above zero and the estimate below it.
    (-0.001, 0.002, 0.004),
    # The interval sits below zero and the estimate above it.
    (0.001, -0.004, -0.002),
    # The interval excludes zero and the estimate is zero.
    (0.0, 0.002, 0.004),
])
def test_an_estimate_on_the_other_side_of_its_interval_is_an_anomaly(estimate, lo, hi):
    """Neither 46 nor 48 is true of it. CLAUDE.md: stop and report it."""
    with pytest.raises(OutcomeAnomaly, match="other side"):
        call_outcome(estimate, lo, hi, supported=True)


def test_an_inverted_interval_is_an_anomaly():
    with pytest.raises(OutcomeAnomaly, match="lower end"):
        call_outcome(0.01, 0.03, -0.01, supported=True)


def test_the_outcome_wordings_are_decision_twos_and_never_claim_equivalence():
    text = " ".join(REPORTING_RULES.read_text(encoding="utf-8").split())
    section = text[text.index("## Decision 2."):text.index("## Decision 3.")]
    assert set(OUTCOME_WORDING) == {"46", "47", "48"}
    for code, wording in OUTCOME_WORDING.items():
        assert f"outcome {code}, {wording}" in section, (code, wording)
        assert "equivalen" not in wording.lower()
    assert f'"{METRIC_SENSITIVE}"' in section


# ---------------------------------------------------------------------------
# The qualifier travels with the cell and never changes the outcome
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("estimate, lo, hi, expected", [
    (0.002, 0.001, 0.003, OUTCOME_EXPLICIT_WINS),
    (-0.002, -0.003, -0.001, OUTCOME_LEARNED_WINS),
    (0.001, -0.001, 0.003, OUTCOME_NO_MEASURABLE_GAP),
])
def test_an_engaged_wording_is_a_qualifier_and_never_changes_the_outcome(
    estimate, lo, hi, expected
):
    plain = gap_outcome(cell(estimate=estimate, lo=lo, hi=hi))
    engaged = gap_outcome(cell(estimate=estimate, lo=lo, hi=hi, near_zero=True,
                               wording=WORDING_VETO))
    assert plain["outcome"] == engaged["outcome"] == expected
    assert plain["outcome_wording"] == engaged["outcome_wording"] == OUTCOME_WORDING[expected]
    assert plain["qualifier"] is None
    assert engaged["qualifier"] == WORDING_VETO


def test_an_unsupported_cell_carries_no_outcome_and_no_qualifier():
    out = gap_outcome(cell(supported=False, near_zero=True,
                           wording=WORDING_SMALL_SIGN_CONSISTENT))
    assert out == {"outcome": None, "outcome_wording": None, "qualifier": None}


def test_the_operational_gap_is_called_by_the_same_rule_and_carries_its_code_only():
    """Decision 2 words outcomes 46 to 48 for delta_learn_pp. The splat-pool
    explicit arm is Transport-Only, not Context-Lift, so the operational gap
    carries the code alone."""
    for estimate, lo, hi, code in ((0.05, 0.03, 0.07, "46"), (0.001, -0.001, 0.003, "47"),
                                   (-0.05, -0.07, -0.03, "48")):
        out = gap_outcome(cell("delta_learn_sp", estimate=estimate, lo=lo, hi=hi))
        assert out["outcome"] == code
        assert out["outcome_wording"] is None


@pytest.mark.parametrize("quantity", ["cl_margin", "delta_formulation", "read_deficit"])
def test_only_the_two_learned_gaps_are_called(quantity):
    with pytest.raises(ValueError, match="delta_learn"):
        gap_outcome(cell(quantity))


# ---------------------------------------------------------------------------
# Outcome 49, metric-sensitive cells, and the landing-offset flags
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("per_point, splat, expected", [
    ("46", "47", True),
    ("46", "48", True),
    ("46", "46", False),
    ("47", "46", False), ("47", "47", False), ("47", "48", False),
    ("48", "46", False), ("48", "47", False), ("48", "48", False),
    # The operational cell is unsupported, so whether 49 holds is unknown.
    ("46", None, None),
    # The per-point cell is unsupported.
    (None, "47", None), (None, None, None),
    # The per-point cell is not 46, so 49 cannot hold whatever the splat cell.
    ("47", None, False), ("48", None, False),
])
def test_outcome_49_truth_table(per_point, splat, expected):
    assert outcome_49(per_point, splat) is expected


def test_outcome_49_refuses_a_code_that_is_not_an_outcome():
    with pytest.raises(ValueError, match="outcome"):
        outcome_49("49", "47")


@pytest.mark.parametrize("centered, raw, expected", [
    (0.010, 0.020, False),
    (-0.010, -0.020, False),
    (0.010, -0.001, True),
    (-0.010, 0.001, True),
    (0.0, 0.010, True),
])
def test_a_sign_disagreement_between_metrics_is_metric_sensitive(centered, raw, expected):
    out = metric_sensitive(cell(estimate=centered, lo=centered - 0.05, hi=centered + 0.05),
                           cell(metric="raw", estimate=raw, lo=raw - 0.05, hi=raw + 0.05))
    assert out is expected


def test_metric_sensitivity_is_read_only_in_supported_cells():
    centered = cell(estimate=0.01)
    raw = cell(metric="raw", estimate=-0.01, lo=-0.02, hi=0.0)
    assert metric_sensitive(centered, raw) is True
    assert metric_sensitive(cell(estimate=0.01, supported=False), raw) is None
    assert metric_sensitive(centered, cell(metric="raw", estimate=-0.01, supported=False)) is None


def test_metric_sensitivity_compares_one_quantity_under_its_two_metrics():
    with pytest.raises(ValueError, match="centered"):
        metric_sensitive(cell(metric="raw"), cell(metric="raw"))
    with pytest.raises(ValueError, match="quantity"):
        metric_sensitive(cell(), cell("delta_learn_sp", metric="raw"))


def deficit(estimate=0.004, lo=0.002, hi=0.006, supported=True, metric="centered"):
    return cell("read_deficit", metric=metric, estimate=estimate, lo=lo, hi=hi,
                supported=supported, disclosure={"near_zero": False, "applicable": False})


@pytest.mark.parametrize("gap, read, expected", [
    # Predict-with-Depth leads by 0.003. The deficit's upper end, 0.006,
    # covers the lead, so the lead is within the size of the landing read.
    (cell(estimate=-0.003, lo=-0.005, hi=-0.001), deficit(),
     {"lead_within_read": True, "cl_lead_understated": False, "read_deficit_anomaly": False}),
    # A lead of 0.010 is larger than the deficit's upper end.
    (cell(estimate=-0.010, lo=-0.012, hi=-0.008), deficit(),
     {"lead_within_read": False, "cl_lead_understated": False, "read_deficit_anomaly": False}),
    # An upper end exactly the size of the lead covers it.
    (cell(estimate=-0.006, lo=-0.008, hi=-0.004), deficit(),
     {"lead_within_read": True, "cl_lead_understated": False, "read_deficit_anomaly": False}),
    # Context-Lift leads. The read works against it, so its lead is if
    # anything understated. Whether a lead is within the read is not asked.
    (cell(estimate=0.020, lo=0.010, hi=0.030), deficit(),
     {"lead_within_read": None, "cl_lead_understated": True, "read_deficit_anomaly": False}),
    # A negative deficit contradicts the mechanism and is an anomaly.
    (cell(estimate=0.020, lo=0.010, hi=0.030), deficit(-0.002, -0.004, 0.0),
     {"lead_within_read": None, "cl_lead_understated": True, "read_deficit_anomaly": True}),
    # A gap of exactly zero: neither method leads.
    (cell(estimate=0.0, lo=-0.002, hi=0.002), deficit(),
     {"lead_within_read": None, "cl_lead_understated": False, "read_deficit_anomaly": False}),
])
def test_the_landing_offset_flags(gap, read, expected):
    assert landing_offset_flags(gap, read) == expected


@pytest.mark.parametrize("gap_supported, read_supported", [
    (False, True), (True, False), (False, False),
])
def test_the_landing_offset_flags_need_both_cells_supported(gap_supported, read_supported):
    out = landing_offset_flags(
        cell(estimate=-0.003, lo=-0.005, hi=-0.001, supported=gap_supported),
        deficit(-0.002, -0.004, 0.0, supported=read_supported),
    )
    assert out == {"lead_within_read": None, "cl_lead_understated": None,
                   "read_deficit_anomaly": None}


def test_the_landing_offset_flags_read_one_cell_of_each_kind_under_one_metric():
    with pytest.raises(ValueError, match="read_deficit"):
        landing_offset_flags(cell(), cell())
    with pytest.raises(ValueError, match="delta_learn_pp"):
        landing_offset_flags(cell("delta_learn_sp"), deficit())
    with pytest.raises(ValueError, match="metric"):
        landing_offset_flags(cell(), deficit(metric="raw"))


# ---------------------------------------------------------------------------
# The measured outcome: one outcome per regime row and metric, plus a pooled
# summary row
# ---------------------------------------------------------------------------

def _cells(quantity, values, supported=None):
    """One cell per (scope, metric). values maps a scope to (estimate, lo, hi)."""
    out = {}
    for scope in SCOPES:
        for metric in METRICS:
            estimate, lo, hi = values[scope]
            if metric == "raw":
                estimate, lo, hi = values.get((scope, "raw"), (estimate, lo, hi))
            ok = True if supported is None else supported.get(scope, True)
            out[(scope, metric)] = cell(quantity, metric=metric, estimate=estimate,
                                        lo=lo, hi=hi, supported=ok)
    return out


PER_POINT_VALUES = {
    "rotation": (0.050, 0.030, 0.070),        # 46
    "translation": (0.040, 0.020, 0.060),     # 46
    "orbit": (-0.030, -0.050, -0.010),        # 48
    "pooled": (0.020, -0.005, 0.045),         # 47
    # Raw disagrees in sign on rotation.
    ("rotation", "raw"): (-0.004, -0.010, 0.002),
}
SPLAT_VALUES = {
    "rotation": (0.030, 0.010, 0.050),        # 46: no 49
    "translation": (0.001, -0.004, 0.006),    # 47: outcome 49 when per-point is 46
    "orbit": (-0.020, -0.030, -0.010),        # 48
    "pooled": (0.010, 0.002, 0.018),          # 46
}


def test_the_measured_outcome_is_one_row_per_regime_and_metric_plus_a_pooled_summary():
    rows = measured_outcome(_cells("delta_learn_pp", PER_POINT_VALUES),
                            _cells("delta_learn_sp", SPLAT_VALUES))
    assert [(r["metric"], r["scope"]) for r in rows] == [
        (metric, scope) for metric in METRICS for scope in SCOPES
    ]
    assert METRICS[0] == "centered"
    by = {(r["scope"], r["metric"]): r for r in rows}
    for scope in REGIME_SCOPES:
        assert by[(scope, "centered")]["row_label"] == scope
        assert by[(scope, "centered")]["summary"] is False
    pooled = by[(POOLED_SCOPE, "centered")]
    assert pooled["summary"] is True
    assert pooled["row_label"] == POOLED_LABEL
    assert "summary" in POOLED_LABEL
    assert all(r["primary_metric"] is (r["metric"] == "centered") for r in rows)

    assert by[("rotation", "centered")]["outcome"] == "46"
    assert by[("rotation", "raw")]["outcome"] == "47"
    assert by[("translation", "centered")]["outcome"] == "46"
    assert by[("orbit", "centered")]["outcome"] == "48"
    assert by[("pooled", "centered")]["outcome"] == "47"
    assert by[("orbit", "centered")]["outcome_wording"] == "the learned transformation wins"

    # Outcome 49 only where per-point is 46 and the operational gap is 47 or 48.
    assert by[("translation", "centered")]["delta_learn_sp_outcome"] == "47"
    assert by[("translation", "centered")]["outcome_49"] is True
    assert by[("rotation", "centered")]["outcome_49"] is False
    assert by[("orbit", "centered")]["outcome_49"] is False
    assert by[("pooled", "centered")]["outcome_49"] is False

    # The two statements are never collapsed: both gaps keep their own numbers,
    # and the cross-path terms sit beside them.
    row = by[("translation", "centered")]
    assert (row["delta_learn_pp"], row["delta_learn_pp_ci_low"],
            row["delta_learn_pp_ci_high"]) == (0.040, 0.020, 0.060)
    assert (row["delta_learn_sp"], row["delta_learn_sp_ci_low"],
            row["delta_learn_sp_ci_high"]) == (0.001, -0.004, 0.006)
    assert row["x_delta_learn_pp"] == 0.011
    assert (row["x_delta_learn_sp"], row["x_delta_learn_sp_ci_low"],
            row["x_delta_learn_sp_ci_high"]) == (0.004, -0.001, 0.009)
    assert row["path_difference_learn"] == 0.007

    # Metric sensitivity is a property of the cell, on both of its rows.
    assert by[("rotation", "centered")]["metric_sensitive"] is True
    assert by[("rotation", "raw")]["metric_sensitive"] is True
    assert by[("translation", "centered")]["metric_sensitive"] is False
    # No row claims equivalence, under any outcome.
    assert "equivalen" not in json.dumps(rows).lower()


def test_an_unsupported_regime_row_is_shown_with_its_counts_and_not_classified():
    rows = measured_outcome(
        _cells("delta_learn_pp", PER_POINT_VALUES, supported={"orbit": False}),
        _cells("delta_learn_sp", SPLAT_VALUES),
    )
    for row in rows:
        if row["scope"] != "orbit":
            continue
        assert row["supported"] is False
        assert row["outcome"] is None and row["outcome_wording"] is None
        assert row["outcome_49"] is None and row["metric_sensitive"] is None
        assert (row["n_scenes"], row["n_camera_pairs"]) == (2, 4)
        assert row["delta_learn_pp"] == -0.030


def test_a_table_shows_the_near_zero_wording_of_a_supported_cell_only():
    """PROTOCOL 3.4 keeps a cell below support out of every claim, and section 9
    gives it no qualifier. So a table shows no flag beside it, and a marker
    that claims nothing in place of the wording. The disclosure itself is left
    as the estimand layer computed it."""
    import lot.phase5_estimands as estimands
    from lot.phase5_outcomes import WORDING_BELOW_SUPPORT, shown_near_zero

    engaged = cell(estimate=0.001, lo=0.0005, hi=0.0015, near_zero=True,
                   wording=WORDING_SMALL_SIGN_CONSISTENT)
    assert shown_near_zero(engaged) == {"near_zero": True,
                                        "wording": WORDING_SMALL_SIGN_CONSISTENT}
    assert shown_near_zero(cell()) == {"near_zero": False,
                                       "wording": "effect outside the operator band"}
    thin = cell(estimate=0.001, lo=0.0005, hi=0.0015, supported=False, near_zero=True,
                wording=WORDING_SMALL_SIGN_CONSISTENT)
    assert shown_near_zero(thin) == {"near_zero": None, "wording": WORDING_BELOW_SUPPORT}
    assert thin.disclosure["near_zero"] is True
    assert thin.disclosure["wording"] == WORDING_SMALL_SIGN_CONSISTENT
    sentences = {value for name, value in vars(estimands).items() if name.startswith("WORDING_")}
    assert WORDING_SMALL_SIGN_CONSISTENT in sentences
    assert WORDING_BELOW_SUPPORT not in sentences
    assert "equivalen" not in WORDING_BELOW_SUPPORT and "\u2014" not in WORDING_BELOW_SUPPORT


def test_a_regime_row_below_support_shows_no_near_zero_flag():
    """An orbit cell below support whose disclosure engages the wording: its
    measured-outcome rows keep their terms and counts, and show no flag and no
    qualifier. The supported rows keep theirs."""
    per_point = _cells("delta_learn_pp", PER_POINT_VALUES)
    for metric in METRICS:
        per_point[("orbit", metric)] = cell(
            "delta_learn_pp", metric=metric, estimate=0.001, lo=0.0005, hi=0.0015,
            supported=False, near_zero=True, wording=WORDING_SMALL_SIGN_CONSISTENT)
    rows = measured_outcome(per_point, _cells("delta_learn_sp", SPLAT_VALUES))
    for row in rows:
        if row["scope"] == "orbit":
            assert row["supported"] is False and row["outcome"] is None
            assert row["near_zero"] is None and row["qualifier"] is None
            assert (row["x_delta_learn_pp"], row["path_difference_learn"]) == (0.011, 0.007)
            assert (row["n_scenes"], row["n_camera_pairs"]) == (2, 4)
        else:
            assert row["near_zero"] is False and row["qualifier"] is None


def test_the_measured_outcome_carries_the_landing_flags_when_the_deficit_is_given():
    read = {(scope, metric): deficit(metric=metric) for scope in SCOPES for metric in METRICS}
    rows = measured_outcome(_cells("delta_learn_pp", PER_POINT_VALUES),
                            _cells("delta_learn_sp", SPLAT_VALUES), read_deficit=read)
    by = {(r["scope"], r["metric"]): r for r in rows}
    orbit = by[("orbit", "centered")]
    # Predict-with-Depth leads by 0.030, beyond the deficit's upper end.
    assert orbit["lead_within_read"] is False
    assert orbit["cl_lead_understated"] is False
    assert by[("rotation", "centered")]["cl_lead_understated"] is True
    assert orbit["read_deficit"] == 0.004 and orbit["read_deficit_supported"] is True

    without = measured_outcome(_cells("delta_learn_pp", PER_POINT_VALUES),
                               _cells("delta_learn_sp", SPLAT_VALUES))
    assert all("lead_within_read" not in row for row in without)


def test_the_measured_outcome_refuses_missing_or_misplaced_cells():
    per_point = _cells("delta_learn_pp", PER_POINT_VALUES)
    splat = _cells("delta_learn_sp", SPLAT_VALUES)
    missing = {k: v for k, v in per_point.items() if k != ("orbit", "raw")}
    with pytest.raises(ValueError, match="orbit"):
        measured_outcome(missing, splat)
    with pytest.raises(ValueError, match="delta_learn_sp"):
        measured_outcome(per_point, per_point)
    swapped = dict(per_point)
    swapped[("rotation", "centered")] = per_point[("rotation", "raw")]
    with pytest.raises(ValueError, match="metric"):
        measured_outcome(swapped, splat)


# ---------------------------------------------------------------------------
# Through the estimand layer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cl, predict, expected", [
    (0.62, 0.55, "46"),
    (0.50, 0.60, "48"),
])
def test_cells_from_evaluate_quantity_are_called_by_their_own_interval(cl, predict, expected):
    records = make_records(cl=cl, predict=predict, scene_spread=0.05, noise=1e-4)
    result = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    assert result.supported
    out = gap_outcome(result)
    assert out["outcome"] == expected
    assert out["qualifier"] is None


def test_a_gap_that_alternates_by_scene_is_no_measurable_gap_with_its_qualifier():
    records = make_records(n_scenes=8, pairs_per_scene=5, noise=1e-6)
    for r in records:
        sign = 1.0 if int(r["scene"].split("_")[1]) % 2 == 0 else -1.0
        r["cl_centered"] = r["predict_centered"] + sign * 0.002
    result = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    assert result.supported and result.lo <= 0.0 <= result.hi
    assert result.disclosure["near_zero"]
    out = gap_outcome(result)
    assert out["outcome"] == "47"
    assert out["outcome_wording"] == "no measurable gap at the reported scale"
    assert out["qualifier"] == result.disclosure["wording"]


def test_a_thin_cell_from_evaluate_quantity_is_not_classified():
    records = make_records(n_scenes=2, pairs_per_scene=4)
    result = evaluate_quantity(records, "delta_learn_pp", "centered", ANALYSIS)
    assert not result.supported
    assert gap_outcome(result)["outcome"] is None


def test_the_module_text_uses_no_em_dash_and_no_letter_method_labels():
    import lot.phase5_outcomes as outcomes

    source = Path(outcomes.__file__).read_text(encoding="utf-8")
    assert "\u2014" not in source
    assert not re.search(r"\bmethod [ABC]\b", source)


# ---------------------------------------------------------------------------
# reporting_rules.md section 9: an estimate outside its own interval stops
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("estimate, lo, hi", [
    # Above an interval that excludes zero, on the same side.
    (0.08, 0.03, 0.07),
    # Below an interval that excludes zero, on the same side.
    (0.0295, 0.03, 0.07),
    (-0.08, -0.07, -0.03),
    # Outside an interval that includes zero.
    (0.012, -0.01, 0.01),
    (-0.012, -0.01, 0.01),
])
def test_a_supported_estimate_outside_its_own_interval_is_an_anomaly(estimate, lo, hi):
    """Section 9: "an estimate outside its own interval stops the tables rather
    than being called". It fits no outcome, whichever side of zero it is on."""
    with pytest.raises(OutcomeAnomaly, match="outside its own interval"):
        call_outcome(estimate, lo, hi, supported=True)
    assert call_outcome(estimate, lo, hi, supported=False) is None


@pytest.mark.parametrize("estimate, lo, hi, expected", [
    # An estimate on an endpoint is inside its interval.
    (0.03, 0.03, 0.07, OUTCOME_EXPLICIT_WINS),
    (0.07, 0.03, 0.07, OUTCOME_EXPLICIT_WINS),
    (-0.07, -0.07, -0.03, OUTCOME_LEARNED_WINS),
    (0.01, -0.01, 0.01, OUTCOME_NO_MEASURABLE_GAP),
])
def test_an_estimate_on_an_endpoint_is_inside_its_interval(estimate, lo, hi, expected):
    assert call_outcome(estimate, lo, hi, supported=True) == expected


def test_section_nine_states_the_stop_the_code_applies():
    text = " ".join(REPORTING_RULES.read_text(encoding="utf-8").split())
    assert ("A supported cell with an undefined interval, or an estimate outside its "
            "own interval, stops the tables rather than being called.") in text


# ---------------------------------------------------------------------------
# reporting_rules.md decision 5: the control label and the stable curve
# ---------------------------------------------------------------------------

from lot.phase5_outcomes import (  # noqa: E402
    ESSENTIALLY_NO_EFFECT,
    control_label,
    stable_validation_curve,
)

TOLERANCE = ANALYSIS.path_agreement_tolerance


def test_decision_five_names_the_label_and_the_tolerance_verbatim():
    text = " ".join(REPORTING_RULES.read_text(encoding="utf-8").split())
    section = text[text.index("## Decision 5."):text.index("## 6.")]
    assert f'is labelled "{ESSENTIALLY_NO_EFFECT}"' in section
    assert "magnitude at most 0.003" in section
    assert "within 0.003 of the best" in section
    # The tolerance the tables pass in is that number, read from the frozen
    # analysis configuration rather than written into the code.
    assert TOLERANCE == 0.003


@pytest.mark.parametrize("degradation, expected", [
    (0.0, ESSENTIALLY_NO_EFFECT),
    (0.003, ESSENTIALLY_NO_EFFECT),
    (-0.003, ESSENTIALLY_NO_EFFECT),
    (0.0029, ESSENTIALLY_NO_EFFECT),
    (-0.001, ESSENTIALLY_NO_EFFECT),
    (0.0031, None),
    (-0.0031, None),
    (0.2, None),
    (NAN, None),
    (math.inf, None),
])
def test_a_control_with_a_tiny_degradation_is_labelled_essentially_no_effect(
    degradation, expected
):
    """Descriptive only. A control is never a threshold anything must pass."""
    assert control_label(degradation, TOLERANCE) == expected


@pytest.mark.parametrize("history, stopped_early, expected", [
    # Early stopping fired.
    ([[500, 0.50], [1000, 0.55], [1500, 0.54]], True, True),
    # The last score is the best.
    ([[500, 0.50], [1000, 0.55]], False, True),
    # The last score is within the tolerance of the best.
    ([[500, 0.50], [1000, 0.555], [1500, 0.5521]], False, True),
    # The last score fell further than the tolerance below the best.
    ([[500, 0.50], [1000, 0.555], [1500, 0.5519]], False, False),
    # One validation is not a curve, even when early stopping fired.
    ([[500, 0.50]], True, False),
    ([], True, False),
    # A score that is not finite makes the curve unstable.
    ([[500, 0.50], [1000, NAN], [1500, 0.55]], True, False),
    ([[500, 0.50], [1000, math.inf]], True, False),
])
def test_the_stable_validation_curve(history, stopped_early, expected):
    assert stable_validation_curve(history, stopped_early, TOLERANCE) is expected


def test_the_stable_curve_reads_the_score_not_the_step():
    """A history is (step, score) pairs, as lot.train records them."""
    assert stable_validation_curve(((500, 0.50), (1000, 0.40)), False, TOLERANCE) is False
    assert stable_validation_curve(((500, 0.40), (1000, 0.50)), False, TOLERANCE) is True


# ---------------------------------------------------------------------------
# One home for the frozen outcome code
# ---------------------------------------------------------------------------

def test_the_outcome_code_has_one_home_in_the_frozen_class():
    """Decision 1 freezes the outcome and wording code at E. It lives in
    lot.phase5_outcomes, which FROZEN_AT_E names, and the reporting module
    defines none of it again."""
    import ast

    import lot.phase5_outcomes as outcomes
    import lot.phase5_report as report
    from lot.phase5_provenance import FROZEN_AT_E

    assert "src/lot/phase5_outcomes.py" in FROZEN_AT_E
    frozen = {
        node.name for node in ast.parse(Path(outcomes.__file__).read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert {"call_outcome", "gap_outcome", "outcome_49", "metric_sensitive",
            "landing_offset_flags", "measured_outcome", "control_label",
            "stable_validation_curve", "OutcomeAnomaly"} <= frozen
    defined = {
        node.name for node in ast.parse(Path(report.__file__).read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert not frozen & defined, sorted(frozen & defined)
    for name in frozen:
        if hasattr(report, name):
            assert getattr(report, name) is getattr(outcomes, name), name
