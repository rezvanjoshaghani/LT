"""Phase 5 outcome and wording code: reporting_rules.md decisions 2 and 5.

Decision 1 freezes this module at commit E, with lot.phase5_estimands,
lot.paired_bootstrap, and reporting_rules.md itself. Any difference in it
between E and the reporting commit fails acceptance, which is why the tables
in lot.phase5_report import it rather than define it. It must therefore exist
before the chain runs.

Every function here is pure. It reads cells that lot.phase5_estimands has
already computed and never computes an estimate or an interval itself. A cell
is a CellResult from evaluate_quantity, with its paired scene interval, its
support, and its near-zero disclosure.

The rules, from decision 2:

- A supported cell is classified from the reported gap's own paired scene
  interval. Clear of zero with a positive estimate is outcome 46, explicit
  context-lift wins. Clear of zero with a negative estimate is outcome 48,
  the learned transformation wins, reported directly. An interval that
  includes zero is outcome 47, no measurable gap at the reported scale. It
  is never called equivalence.
- An unsupported cell is shown with its counts and is not classified.
- An engaged near-zero wording travels with the cell as a qualifier. It never
  changes the outcome. PROTOCOL 3.4 keeps an unsupported cell out of every
  claim, so only a classified cell carries a qualifier here. The wording
  itself stays in the cell's own disclosure either way. A table shows it
  beside a supported cell only. Beside a cell below support it shows no flag,
  and WORDING_BELOW_SUPPORT in place of the wording, with the terms kept.
- Outcome 49 is flagged when the per-point cell is 46 and the operational
  delta_learn_sp cell, classified on its own population by the same rule, is
  47 or 48. The cross-path terms are shown beside it. The two statements are
  never collapsed.
- A sign disagreement between centered and raw cosine in a supported cell is
  flagged "metric-sensitive". It is descriptive.
- The landing-offset flags of landing_offset_diagnostic.md are reported only
  where both the delta_learn_pp cell and the read_deficit cell are supported.
- The measured outcome is one outcome per regime row and metric, plus the
  pooled row, which is labelled a summary. Claims are made per regime.

Trends with rotation angle or parallax are described from the bin estimates.
No slope test is registered, so none is computed here.

An anomaly is a stop, per CLAUDE.md and section 9. A supported cell without a
finite interval, or one whose estimate lies outside its own interval, raises
OutcomeAnomaly rather than being called.

Decision 5 defines two more things, kept here so they freeze with the rest:

- A shuffle control whose degradation has magnitude at most 0.003 is
  labelled "essentially no effect". The label is descriptive. A control is
  never a threshold anything must pass.
- A training run's validation curve is stable when every validation score is
  finite, at least two validations ran, and early stopping fired or the last
  score is within 0.003 of the best. All nine runs must be stable for
  acceptance.

Decision 5's 0.003 is the frozen path_agreement_tolerance of
configs/analysis.yaml. Nothing in src/ carries a normative constant as a
literal, so the caller passes it in as tolerance.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .phase5_estimands import CellResult

# The outcomes of specification steps 46 to 48, keyed by step number, in
# decision 2's words. Decision 2 words them for delta_learn_pp.
OUTCOME_EXPLICIT_WINS = "46"
OUTCOME_NO_MEASURABLE_GAP = "47"
OUTCOME_LEARNED_WINS = "48"
OUTCOME_WORDING = {
    OUTCOME_EXPLICIT_WINS: "explicit context-lift wins",
    OUTCOME_NO_MEASURABLE_GAP: "no measurable gap at the reported scale",
    OUTCOME_LEARNED_WINS: "the learned transformation wins",
}
# The flag decision 2 names for a sign disagreement between the two metrics.
METRIC_SENSITIVE = "metric-sensitive"

# The two learned-versus-explicit gaps the rule classifies. The headline is
# delta_learn_pp. delta_learn_sp is classified by the same rule for outcome 49.
GAP_QUANTITIES = ("delta_learn_pp", "delta_learn_sp")

# The rows of the measured outcome. Each regime is a separate experimental
# control, PROTOCOL 3.3, so claims are made per regime. The pooled row is a
# summary and is labelled as one.
REGIME_SCOPES = ("rotation", "translation", "orbit")
POOLED_SCOPE = "pooled"
SCOPES = REGIME_SCOPES + (POOLED_SCOPE,)
POOLED_LABEL = "pooled over regimes (summary)"
# Centered cosine is primary, with raw cosine beside it.
PRIMARY_METRIC = "centered"
METRICS = ("centered", "raw")

# The landing-offset flags, landing_offset_diagnostic.md.
LANDING_OFFSET_FLAGS = ("lead_within_read", "cl_lead_understated", "read_deficit_anomaly")

# Decision 5's label for a shuffle control with essentially no effect.
ESSENTIALLY_NO_EFFECT = "essentially no effect"

# What a table shows in place of the near-zero wording of a cell below the
# frozen support thresholds. PROTOCOL 3.4 keeps such a cell out of every claim,
# and section 9 of reporting_rules.md gives it no qualifier, so it shows its
# terms beside this marker. It is none of decision 2's sentences, and it
# claims nothing.
WORDING_BELOW_SUPPORT = "below support; not interpreted"


class OutcomeAnomaly(ValueError):
    """A supported cell whose interval cannot be called. A stop, not a result."""


def call_outcome(estimate: float, lo: float, hi: float, supported: bool) -> str | None:
    """The outcome of one cell from its own paired scene interval.

    Returns None for an unsupported cell, which is not classified. Otherwise
    "46" when the interval excludes zero and the estimate is positive, "48"
    when it excludes zero and the estimate is negative, and "47" when it
    includes zero. An interval with an endpoint at zero includes it, as
    PathEstimate.excludes_zero reads it.

    A supported cell must have a finite estimate and a finite, ordered
    interval, and the estimate must lie inside its own interval, endpoints
    included. Section 9 of reporting_rules.md makes each failure a stop: it
    raises OutcomeAnomaly. An estimate on the other side of zero from an
    interval that excludes zero is reported as such, because it fits neither
    46 nor 48.
    """
    if not supported:
        return None
    if not all(math.isfinite(x) for x in (estimate, lo, hi)):
        raise OutcomeAnomaly(
            f"a supported cell has no finite estimate and interval: estimate "
            f"{estimate!r}, interval [{lo!r}, {hi!r}]"
        )
    if lo > hi:
        raise OutcomeAnomaly(
            f"the interval's lower end {lo!r} is above its upper end {hi!r}"
        )
    excludes_zero = lo > 0.0 or hi < 0.0
    if excludes_zero and not ((lo > 0.0 and estimate > 0.0) or (hi < 0.0 and estimate < 0.0)):
        raise OutcomeAnomaly(
            f"the interval [{lo!r}, {hi!r}] excludes zero, and the estimate "
            f"{estimate!r} sits on the other side of zero or at it"
        )
    if not lo <= estimate <= hi:
        raise OutcomeAnomaly(
            f"the estimate {estimate!r} lies outside its own interval [{lo!r}, {hi!r}]"
        )
    if lo > 0.0:
        return OUTCOME_EXPLICIT_WINS
    if hi < 0.0:
        return OUTCOME_LEARNED_WINS
    return OUTCOME_NO_MEASURABLE_GAP


def gap_outcome(cell: CellResult) -> dict[str, Any]:
    """The outcome of one learned-versus-explicit gap cell, with its qualifier.

    outcome is call_outcome on the cell's own estimate, interval, and support.
    outcome_wording is decision 2's sentence for that outcome. Decision 2
    words the outcomes for delta_learn_pp. The splat-pool explicit arm is
    Transport-Only, not Context-Lift, so a delta_learn_sp cell carries its
    code alone. qualifier is the cell's near-zero wording when the wording is
    engaged and the cell is classified, and None otherwise. It never changes
    the outcome.
    """
    if cell.quantity not in GAP_QUANTITIES:
        raise ValueError(
            f"outcomes are called for {GAP_QUANTITIES} only, not {cell.quantity!r}"
        )
    outcome = call_outcome(cell.estimate, cell.lo, cell.hi, cell.supported)
    engaged = bool(cell.disclosure.get("near_zero", False))
    worded = outcome is not None and cell.quantity == "delta_learn_pp"
    return {
        "outcome": outcome,
        "outcome_wording": OUTCOME_WORDING[outcome] if worded else None,
        "qualifier": (
            cell.disclosure.get("wording") if outcome is not None and engaged else None
        ),
    }


def shown_near_zero(cell: CellResult) -> dict[str, Any]:
    """The near-zero flag and wording a table shows beside one cell.

    A supported cell shows its disclosure's own flag and wording. A cell below
    support makes no claim, PROTOCOL 3.4, and carries no qualifier, section 9
    of reporting_rules.md. So its flag is None and its wording is
    WORDING_BELOW_SUPPORT. Its disclosure terms are shown either way, and the
    disclosure itself is left as the estimand layer computed it.
    """
    if not cell.supported:
        return {"near_zero": None, "wording": WORDING_BELOW_SUPPORT}
    return {
        "near_zero": bool(cell.disclosure.get("near_zero", False)),
        "wording": cell.disclosure.get("wording"),
    }


def outcome_49(per_point_outcome: str | None, splat_outcome: str | None) -> bool | None:
    """Specification step 49, from the two gaps' own outcomes.

    True when the per-point cell is 46 and the operational cell is 47 or 48.
    False when the per-point cell is classified and that does not hold. None
    when it cannot be told: the per-point cell is unclassified, or it is 46
    and the operational cell is unclassified.
    """
    for code in (per_point_outcome, splat_outcome):
        if code is not None and code not in OUTCOME_WORDING:
            raise ValueError(f"{code!r} is not an outcome of steps 46 to 48")
    if per_point_outcome is None:
        return None
    if per_point_outcome != OUTCOME_EXPLICIT_WINS:
        return False
    if splat_outcome is None:
        return None
    return splat_outcome in (OUTCOME_NO_MEASURABLE_GAP, OUTCOME_LEARNED_WINS)


def _sign(value: float) -> int:
    return 0 if value == 0.0 else (1 if value > 0.0 else -1)


def metric_sensitive(centered: CellResult, raw: CellResult) -> bool | None:
    """Whether one cell's estimates under centered and raw cosine differ in sign.

    Descriptive. Read only in a supported cell, so None unless both are
    supported. A supported cell without a finite estimate raises
    OutcomeAnomaly, as call_outcome does.
    """
    if (centered.metric, raw.metric) != ("centered", "raw"):
        raise ValueError(
            f"compare a cell under centered and raw cosine, not "
            f"{centered.metric!r} and {raw.metric!r}"
        )
    if centered.quantity != raw.quantity:
        raise ValueError(
            f"one quantity under two metrics is compared, not {centered.quantity!r} "
            f"and {raw.quantity!r}"
        )
    if not (centered.supported and raw.supported):
        return None
    for cell in (centered, raw):
        if not math.isfinite(cell.estimate):
            raise OutcomeAnomaly(
                f"a supported {cell.quantity} cell under {cell.metric} has no "
                f"finite estimate"
            )
    return _sign(centered.estimate) != _sign(raw.estimate)


def landing_offset_flags(
    delta_learn_pp: CellResult, read_deficit: CellResult
) -> dict[str, bool | None]:
    """The interpretation flags of landing_offset_diagnostic.md for one cell.

    The read deficit sits in the same cell as delta_learn_pp, under the same
    metric. Both cells must be supported, or every flag is None.

    - lead_within_read: Predict-with-Depth leads, so delta_learn_pp is below
      zero, and the read deficit's upper interval end is at least the size of
      that lead. Such a lead is not attributed to learning the
      transformation. None where Predict-with-Depth does not lead.
    - cl_lead_understated: Context-Lift leads, so delta_learn_pp is above
      zero. The read works against Context-Lift, so its lead is if anything
      understated.
    - read_deficit_anomaly: the read deficit is below zero, which contradicts
      the mechanism the diagnostic measures. It is reported and not
      interpreted.
    """
    if delta_learn_pp.quantity != "delta_learn_pp":
        raise ValueError(f"the gap cell must be delta_learn_pp, not {delta_learn_pp.quantity!r}")
    if read_deficit.quantity != "read_deficit":
        raise ValueError(f"the deficit cell must be read_deficit, not {read_deficit.quantity!r}")
    if delta_learn_pp.metric != read_deficit.metric:
        raise ValueError(
            f"the two cells are under different metrics: {delta_learn_pp.metric!r} "
            f"and {read_deficit.metric!r}"
        )
    if not (delta_learn_pp.supported and read_deficit.supported):
        return {flag: None for flag in LANDING_OFFSET_FLAGS}
    for cell in (delta_learn_pp, read_deficit):
        if not all(math.isfinite(x) for x in (cell.estimate, cell.lo, cell.hi)):
            raise OutcomeAnomaly(
                f"a supported {cell.quantity} cell has no finite estimate and interval"
            )
    gap = delta_learn_pp.estimate
    return {
        "lead_within_read": read_deficit.hi >= abs(gap) if gap < 0.0 else None,
        "cl_lead_understated": gap > 0.0,
        "read_deficit_anomaly": read_deficit.estimate < 0.0,
    }


def _interval_columns(name: str, estimate: Any, lo: Any, hi: Any) -> dict[str, Any]:
    return {name: estimate, f"{name}_ci_low": lo, f"{name}_ci_high": hi}


def _term_columns(name: str, term: Any) -> dict[str, Any]:
    """A cross-path term from a disclosure, NaN when the disclosure lacks it."""
    term = term if isinstance(term, Mapping) else {}
    nan = float("nan")
    return _interval_columns(
        name, term.get("estimate", nan), term.get("lo", nan), term.get("hi", nan)
    )


def _require_cells(
    cells: Mapping[tuple[str, str], CellResult], quantity: str, label: str
) -> None:
    """The cells of one measured-outcome input: every scope and metric, nothing else."""
    expected = {(scope, metric) for scope in SCOPES for metric in METRICS}
    missing = sorted(expected - set(cells))
    extra = sorted(set(cells) - expected, key=repr)
    if missing or extra:
        raise ValueError(
            f"the {label} cells must be one per (scope, metric) of {SCOPES} by "
            f"{METRICS}; missing cells {missing}, unexpected cells {extra}"
        )
    for (scope, metric), cell in cells.items():
        if cell.quantity != quantity:
            raise ValueError(
                f"the {label} cell for {scope}, {metric} holds {cell.quantity!r}, "
                f"not {quantity!r}"
            )
        if cell.metric != metric:
            raise ValueError(
                f"the {label} cell for {scope}, {metric} is under metric {cell.metric!r}"
            )


def measured_outcome(
    per_point: Mapping[tuple[str, str], CellResult],
    splat_pool: Mapping[tuple[str, str], CellResult],
    read_deficit: Mapping[tuple[str, str], CellResult] | None = None,
) -> list[dict[str, Any]]:
    """The verdict's measured outcome: one row per regime and metric, plus pooled.

    per_point maps each (scope, metric) to its delta_learn_pp cell, and
    splat_pool to its delta_learn_sp cell, for every scope in SCOPES and
    every metric in METRICS. read_deficit, given at the primary level, maps
    the same keys to read_deficit cells, and adds the landing-offset flags.

    Rows come centered first, the primary metric, then raw, each in the order
    of SCOPES. A row carries:

    - its scope and label, whether it is the pooled summary, and whether its
      metric is the primary one;
    - delta_learn_pp with its interval, replicate count, and support counts;
    - the outcome, its wording, the near-zero flag, and the qualifier. The
      flag is shown_near_zero's, so a cell below support shows None;
    - delta_learn_sp with its interval and support, and its own outcome;
    - the outcome 49 flag, with the three cross-path terms beside it, from
      the delta_learn_pp cell's disclosure;
    - whether the cell is metric-sensitive, the same on both metric rows;
    - with read_deficit given, the deficit, its interval and support, and the
      three landing-offset flags.
    """
    _require_cells(per_point, "delta_learn_pp", "per-point")
    _require_cells(splat_pool, "delta_learn_sp", "splat-pool")
    if read_deficit is not None:
        _require_cells(read_deficit, "read_deficit", "read-deficit")

    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        for scope in SCOPES:
            pp = per_point[(scope, metric)]
            sp = splat_pool[(scope, metric)]
            called = gap_outcome(pp)
            sp_outcome = gap_outcome(sp)["outcome"]
            disclosure = pp.disclosure
            row: dict[str, Any] = {
                "scope": scope,
                "row_label": POOLED_LABEL if scope == POOLED_SCOPE else scope,
                "summary": scope == POOLED_SCOPE,
                "metric": metric,
                "primary_metric": metric == PRIMARY_METRIC,
                **_interval_columns("delta_learn_pp", pp.estimate, pp.lo, pp.hi),
                "delta_learn_pp_ci_replicates": pp.n_replicates,
                "n_scenes": pp.n_scenes,
                "n_camera_pairs": pp.n_camera_pairs,
                "n_feature_comparisons": pp.n_feature_comparisons,
                "supported": pp.supported,
                "outcome": called["outcome"],
                "outcome_wording": called["outcome_wording"],
                "near_zero": shown_near_zero(pp)["near_zero"],
                "qualifier": called["qualifier"],
                **_interval_columns("delta_learn_sp", sp.estimate, sp.lo, sp.hi),
                "delta_learn_sp_ci_replicates": sp.n_replicates,
                "delta_learn_sp_supported": sp.supported,
                "delta_learn_sp_outcome": sp_outcome,
                "outcome_49": outcome_49(called["outcome"], sp_outcome),
                **_term_columns("x_delta_learn_pp", disclosure.get("per_point")),
                **_term_columns("x_delta_learn_sp", disclosure.get("splat_pool")),
                **_term_columns("path_difference_learn", disclosure.get("path_difference")),
                "metric_sensitive": metric_sensitive(
                    per_point[(scope, "centered")], per_point[(scope, "raw")]
                ),
            }
            if read_deficit is not None:
                deficit = read_deficit[(scope, metric)]
                row.update(_interval_columns(
                    "read_deficit", deficit.estimate, deficit.lo, deficit.hi
                ))
                row["read_deficit_supported"] = deficit.supported
                row.update(landing_offset_flags(pp, deficit))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Decision 5: the controls label and the stable validation curve
# ---------------------------------------------------------------------------

def control_label(degradation: float, tolerance: float) -> str | None:
    """Decision 5's label for one shuffle control, or None.

    "essentially no effect" when the degradation is finite and its magnitude
    is at most tolerance, which the caller passes as the frozen 0.003. The
    label is descriptive and travels with the control's count of exchanges
    that changed nothing. A control is never a threshold anything must pass,
    so no other label exists, and an undefined degradation gets none.
    """
    if isinstance(degradation, bool) or not isinstance(degradation, (int, float)):
        return None
    if not math.isfinite(degradation):
        return None
    return ESSENTIALLY_NO_EFFECT if abs(degradation) <= tolerance else None


def stable_validation_curve(
    history: Sequence[Sequence[float]], stopped_early: bool, tolerance: float
) -> bool:
    """Decision 5: whether one training run's validation curve is stable.

    history is the run's (step, validation centered cosine) pairs, in the
    order lot.train records them. The curve is stable when every score is
    finite, at least two validations ran, and early stopping fired or the
    last score is within tolerance of the best. tolerance is the frozen 0.003.
    """
    scores = [entry[1] for entry in history]
    if len(scores) < 2:
        return False
    if not all(
        isinstance(score, (int, float)) and not isinstance(score, bool)
        and math.isfinite(score)
        for score in scores
    ):
        return False
    if stopped_early is True:
        return True
    return max(scores) - scores[-1] <= tolerance
