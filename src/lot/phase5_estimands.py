"""Phase 5 estimands: Streams V and W.

This module holds the arithmetic of the phase and nothing else. It does not read
a cache, a manifest, or a parquet; it takes per-pair scored records and turns
them into the reported quantities with their paired intervals. Keeping it free
of I/O is what makes the headline arithmetic testable without the cluster.

Three populations, kept separate on purpose and never silently merged.

    V_P5_pp    the primary support. The GT-evaluable samples Context-Lift
               Transport-Only can validly transport with context-image-scaled
               context depth. CL-Transport, Predict-with-Depth, and
               No-Warp-Copy are all scored on exactly this set.

    V_form     the formulation-diagnostic support, the intersection of V_P5_pp
               with the accepted Phase 4 target-lift set, on the target patch
               cell both estimators share. Only TL-Reference and CL-Transport
               are compared here, and this population never redefines the
               headline one.

    V_sp       the accepted Phase 4 context-image-scale splat-pool scored-cell
               support, used unchanged for the secondary operational
               comparison after its context-side symmetry was verified.

The headline estimand is

    delta_learn_pp = CL-Transport - Predict-with-Depth   on V_P5_pp

and it is the only quantity that may be called the learned-versus-explicit
transformation limitation. The Phase 4 target-lift score is never substituted
into it; a helper below raises if a caller tries.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Sequence

from .analysis_config import AnalysisConfig
from .paired_bootstrap import paired_interval

# Method names, verbatim in every row, table, and figure, per CLAUDE.md.
CL_TRANSPORT = "Context-Lift Transport-Only"
PREDICT_WITH_DEPTH = "Predict-with-Depth"
NO_WARP_COPY = "No-Warp-Copy"
TL_REFERENCE = "Phase4 Target-Lift Reference"
SPLAT_TRANSPORT = "Transport-Only"

PER_POINT = "per_point"
SPLAT_POOL = "splat_pool"
FORMULATION = "formulation"
# The cross-path common-valid population of PROTOCOL 3.9. Both paths are
# re-scored on the target cells they share, so the near-zero disclosure compares
# two operators rather than two selections.
CROSS_PATH = "cross_path"

# Per-pair score fields. Each is the pair's mean score over its own supported
# samples, so a cell's estimand is the unweighted mean over camera pairs that
# PROTOCOL 3.4 fixes as the unit.
# Each method carries all four PROTOCOL 3.7 columns. The reported quantities are
# built on the cosines, as Phase 4's are; the L2 companions are aggregated
# alongside so the shipped tables are schema-complete against the frozen
# protocol rather than cosine-only.
PRIMARY_FIELDS = (
    "cl_raw", "cl_centered", "cl_l2_raw", "cl_l2_centered",
    "predict_raw", "predict_centered", "predict_l2_raw", "predict_l2_centered",
    "nowarp_raw", "nowarp_centered", "nowarp_l2_raw", "nowarp_l2_centered",
)
FORMULATION_FIELDS = (
    "tl_form_raw", "tl_form_centered", "tl_form_l2_raw", "tl_form_l2_centered",
    "cl_form_raw", "cl_form_centered", "cl_form_l2_raw", "cl_form_l2_centered",
)
SPLAT_FIELDS = (
    "sp_transport_raw", "sp_transport_centered",
    "sp_transport_l2_raw", "sp_transport_l2_centered",
    "sp_predict_raw", "sp_predict_centered",
    "sp_predict_l2_raw", "sp_predict_l2_centered",
    "sp_nowarp_raw", "sp_nowarp_centered",
    "sp_nowarp_l2_raw", "sp_nowarp_l2_centered",
)
# Counts and diagnostics. Never bootstrapped as scores, and the near-zero rule
# does not apply to them because they are not in score space.
# PROTOCOL 3.9's cross-path common-valid columns. Both paths live in one field
# tuple deliberately: a single bootstrap draw then serves both, which is what
# makes the path difference paired rather than a subtraction of two independent
# intervals.
INTERSECTION_FIELDS = (
    "x_cl_raw", "x_cl_centered",
    "x_predict_raw", "x_predict_centered",
    "x_sp_transport_raw", "x_sp_transport_centered",
    "x_sp_predict_raw", "x_sp_predict_centered",
)
COUNT_FIELDS = (
    "n_primary", "n_formulation", "n_splat", "n_intersect", "n_predict_nonfinite",
)

ALL_FIELDS = (
    PRIMARY_FIELDS + FORMULATION_FIELDS + SPLAT_FIELDS
    + INTERSECTION_FIELDS + COUNT_FIELDS
)


def quantity_formulas(metric: str) -> dict[str, Callable[[dict[str, float]], float]]:
    """Every reported quantity as a closed form over one replicate's field means.

    Expressing them this way is what keeps the bootstrap paired: the replicate
    draws scenes once, all field means come from that one draw, and each
    quantity is then recomputed whole. Nothing here subtracts two intervals.

    metric is 'raw' or 'centered' and selects which columns the forms read.
    """
    if metric not in ("raw", "centered"):
        raise ValueError(f"metric must be raw or centered, got {metric!r}")
    m = metric

    return {
        # Absolutes, primary per-point path.
        "cl_transport": lambda v: v[f"cl_{m}"],
        "predict_with_depth": lambda v: v[f"predict_{m}"],
        "no_warp_copy": lambda v: v[f"nowarp_{m}"],
        # Stream W step 25: floor-relative margins on the fair matched support.
        "cl_margin": lambda v: v[f"cl_{m}"] - v[f"nowarp_{m}"],
        "predict_margin": lambda v: v[f"predict_{m}"] - v[f"nowarp_{m}"],
        # Stream W step 22: the headline learned-versus-explicit gap.
        "delta_learn_pp": lambda v: v[f"cl_{m}"] - v[f"predict_{m}"],
        # Stream W step 24: the formulation/information diagnostic, on its own
        # population, never interpreted as a learned-model effect.
        "delta_formulation": lambda v: v[f"tl_form_{m}"] - v[f"cl_form_{m}"],
        "tl_reference": lambda v: v[f"tl_form_{m}"],
        "cl_on_formulation_support": lambda v: v[f"cl_form_{m}"],
        # Stream W step 23: the secondary operational gap on the splat path.
        "delta_learn_sp": lambda v: v[f"sp_transport_{m}"] - v[f"sp_predict_{m}"],
        "sp_transport": lambda v: v[f"sp_transport_{m}"],
        "sp_predict": lambda v: v[f"sp_predict_{m}"],
        "sp_transport_margin": lambda v: v[f"sp_transport_{m}"] - v[f"sp_nowarp_{m}"],
        "sp_predict_margin": lambda v: v[f"sp_predict_{m}"] - v[f"sp_nowarp_{m}"],
        # PROTOCOL 3.9's disclosure terms, every one recomputed on the
        # cross-path common-valid cell set before differencing.
        "x_delta_learn_pp": lambda v: v[f"x_cl_{m}"] - v[f"x_predict_{m}"],
        "x_delta_learn_sp": (
            lambda v: v[f"x_sp_transport_{m}"] - v[f"x_sp_predict_{m}"]
        ),
        # The quantity 3.9 requires an interval for: the difference between what
        # the two evaluation paths say about the same effect on the same cells.
        "path_difference": lambda v: (
            (v[f"x_cl_{m}"] - v[f"x_predict_{m}"])
            - (v[f"x_sp_transport_{m}"] - v[f"x_sp_predict_{m}"])
        ),
    }


# Which population each quantity is defined on. A quantity computed on the wrong
# population is the single most damaging mistake available in this phase, so the
# mapping is explicit and asserted rather than implied by naming.
QUANTITY_POPULATION = {
    "cl_transport": PER_POINT,
    "predict_with_depth": PER_POINT,
    "no_warp_copy": PER_POINT,
    "cl_margin": PER_POINT,
    "predict_margin": PER_POINT,
    "delta_learn_pp": PER_POINT,
    "delta_formulation": FORMULATION,
    "tl_reference": FORMULATION,
    "cl_on_formulation_support": FORMULATION,
    "delta_learn_sp": SPLAT_POOL,
    "sp_transport": SPLAT_POOL,
    "sp_predict": SPLAT_POOL,
    "sp_transport_margin": SPLAT_POOL,
    "sp_predict_margin": SPLAT_POOL,
    "x_delta_learn_pp": CROSS_PATH,
    "x_delta_learn_sp": CROSS_PATH,
    "path_difference": CROSS_PATH,
}

# Quantities that live in score space, and to which the frozen 0.003 near-zero
# disclosure applies. Counts and fractions are excluded by Stream AB step 35's
# instruction not to apply the operator band to dimensionless quantities.
SCORE_SPACE_QUANTITIES = frozenset(QUANTITY_POPULATION)


class HeadlineSubstitutionError(RuntimeError):
    """A caller tried to build the headline gap from the target-lift reference."""


def assert_not_target_lift_headline(quantity: str, method: str) -> None:
    """Stream AD: the Phase 4 target-lift score is never the headline gap.

    The confound that motivated the whole Phase 5 redesign would reappear the
    moment TL-Reference were subtracted from Predict-with-Depth and the result
    called the learned-versus-explicit limitation. Making that an exception
    rather than a convention means it cannot happen quietly.
    """
    if quantity == "delta_learn_pp" and method == TL_REFERENCE:
        raise HeadlineSubstitutionError(
            "delta_learn_pp is defined as Context-Lift Transport-Only minus "
            "Predict-with-Depth. Substituting the Phase 4 target-lift reference "
            "reintroduces the information asymmetry Phase 5 exists to remove."
        )


@dataclasses.dataclass(frozen=True)
class SupportCounts:
    """The support a cell rests on, per PROTOCOL 3.4."""

    n_scenes: int
    n_camera_pairs: int
    n_feature_comparisons: int

    def is_supported(self, analysis: AnalysisConfig) -> bool:
        return (
            self.n_scenes >= analysis.support_min_scenes
            and self.n_camera_pairs >= analysis.support_min_camera_pairs
        )


def support_counts(records: Sequence[dict], count_field: str) -> SupportCounts:
    scenes = {r["scene"] for r in records}
    comparisons = sum(
        int(r.get(count_field, 0) or 0)
        for r in records
        if isinstance(r.get(count_field), (int, float))
        and math.isfinite(r.get(count_field, float("nan")))
    )
    return SupportCounts(
        n_scenes=len(scenes),
        n_camera_pairs=len(records),
        n_feature_comparisons=comparisons,
    )


@dataclasses.dataclass(frozen=True)
class PathEstimate:
    """One evaluation path's estimate and paired interval for a quantity."""

    estimate: float
    lo: float
    hi: float

    @property
    def finite(self) -> bool:
        return math.isfinite(self.estimate)

    def within(self, band: float) -> bool:
        return self.finite and abs(self.estimate) <= band

    @property
    def excludes_zero(self) -> bool:
        return math.isfinite(self.lo) and math.isfinite(self.hi) and (
            self.lo > 0.0 or self.hi < 0.0
        )

    @property
    def sign(self) -> int:
        if not self.finite or self.estimate == 0.0:
            return 0
        return 1 if self.estimate > 0.0 else -1


def near_zero_disclosure(
    quantity: str,
    per_point: PathEstimate,
    splat_pool: PathEstimate | None,
    analysis: AnalysisConfig,
    difference: PathEstimate | None = None,
) -> dict[str, Any]:
    """The frozen 0.003 rule, carried forward verbatim from PROTOCOL 3.9.

    The rule is explicitly two-path. Its exact words: both estimates within the
    band with one sign and both intervals clear of zero licenses a claim of a
    small, sign-consistent effect; exactly one within the band licenses the
    effect but not a claim about its size, which is reported as path-sensitive;
    a difference in sign, or an interval that includes zero, licenses no claim
    of advantage and the effect is reported as being at the scale of
    evaluation-path choice.

    An earlier version of this function saw one path and could therefore issue
    the strongest of those three wordings while the other path disagreed or
    reversed sign. That is precisely the over-licensing 3.9 exists to prevent,
    so the second path is now a required argument. Passing None is permitted
    only for a quantity that has no second path, and it can never reach the
    sign-consistent wording.

    Applied only to score-space quantities. Counts and fractions are not in the
    metric's units and the band means nothing for them.
    """
    if quantity not in SCORE_SPACE_QUANTITIES:
        return {"near_zero": False, "applicable": False}

    band = analysis.path_agreement_tolerance
    if not per_point.finite:
        # An undefined term still reports both sides, so a reader can see that
        # the cell was empty rather than that the effect was large.
        return {
            "near_zero": False,
            "applicable": True,
            "band": band,
            "per_point": dataclasses.asdict(per_point),
            "splat_pool": (
                dataclasses.asdict(splat_pool) if splat_pool is not None else None
            ),
            "paths_agree_in_sign": None,
            "both_intervals_exclude_zero": None,
            "wording": "not estimable on this cell",
        }

    pp_in = per_point.within(band)
    if splat_pool is None or not splat_pool.finite:
        # No comparable second path. The band still flags the cell, but the only
        # claim available is the weakest one: 3.9's licence for a sign-consistent
        # effect is conditional on both paths, and one path cannot supply it.
        return {
            "near_zero": bool(pp_in),
            "applicable": True,
            "band": band,
            "per_point": dataclasses.asdict(per_point),
            "splat_pool": None,
            "paths_agree_in_sign": None,
            "both_intervals_exclude_zero": None,
            "wording": (
                "no measurable difference at the reported scale, single path"
                if pp_in else "effect outside the operator band"
            ),
        }

    sp_in = splat_pool.within(band)
    same_sign = per_point.sign == splat_pool.sign and per_point.sign != 0
    both_clear = per_point.excludes_zero and splat_pool.excludes_zero

    # PROTOCOL 3.9's clauses are not three alternatives to be tried in the order
    # the sentence lists them. Its third clause, "a difference in sign, or an
    # interval that includes zero, licenses no claim of advantage", is stated
    # unconditionally, so it is a veto over the other two rather than a fallback
    # after them. Testing the exactly-one-in-band branch first, as an earlier
    # version did, let a cell whose two paths reverse sign be reported as merely
    # path-sensitive, which is a claim the protocol withholds.
    #
    # The band question is asked first only to decide whether the near-zero
    # discipline is engaged at all: 3.9 scopes it to an effect "no larger than
    # this tolerance", so a cell outside the band on both paths is not a
    # near-zero cell and the veto has nothing to act on.
    if not (pp_in or sp_in):
        wording = "effect outside the operator band"
    elif not same_sign or not both_clear:
        wording = (
            "no claim of advantage; the effect is at the scale of "
            "evaluation-path choice"
        )
    elif pp_in and sp_in:
        wording = "small, sign-consistent effect"
    else:
        wording = (
            "effect licensed but its size is path-sensitive; the two evaluation "
            "paths do not agree about whether it sits inside the operator band"
        )

    return {
        "near_zero": bool(pp_in or sp_in),
        "applicable": True,
        "band": band,
        "per_point": dataclasses.asdict(per_point),
        "splat_pool": dataclasses.asdict(splat_pool),
        # The difference carries its own paired interval, from the same scene
        # draw, rather than being a bare subtraction of two point estimates.
        "path_difference": (
            dataclasses.asdict(difference) if difference is not None
            else {"estimate": per_point.estimate - splat_pool.estimate,
                  "lo": float("nan"), "hi": float("nan")}
        ),
        "paths_agree_in_sign": bool(same_sign),
        "both_intervals_exclude_zero": bool(both_clear),
        # Equivalence is never among the licensed wordings, under any branch,
        # because no equivalence region was ever frozen.
        "wording": wording,
    }


@dataclasses.dataclass(frozen=True)
class CellResult:
    quantity: str
    metric: str
    population: str
    estimate: float
    lo: float
    hi: float
    n_scenes: int
    n_camera_pairs: int
    n_feature_comparisons: int
    n_replicates: int
    supported: bool
    disclosure: dict[str, Any]

    def as_row(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        disclosure = row.pop("disclosure")
        row["near_zero"] = disclosure.get("near_zero", False)
        row["near_zero_wording"] = disclosure.get("wording", "")
        return row


def _fields_for(population: str) -> tuple[str, ...]:
    if population == PER_POINT:
        return PRIMARY_FIELDS
    if population == FORMULATION:
        return FORMULATION_FIELDS
    if population == SPLAT_POOL:
        return SPLAT_FIELDS
    if population == CROSS_PATH:
        return INTERSECTION_FIELDS
    raise ValueError(f"unknown population {population!r}")


def _count_field_for(population: str) -> str:
    return {
        PER_POINT: "n_primary",
        FORMULATION: "n_formulation",
        SPLAT_POOL: "n_splat",
        CROSS_PATH: "n_intersect",
    }[population]


# Which splat-pool quantity is the same effect measured on the operational
# path. PROTOCOL 3.9's near-zero rule compares an effect across both evaluation
# paths, so a quantity that has a counterpart must be disclosed against it.
# The formulation quantities have none: they exist only on the per-point path,
# and a cell without a second path can never reach the sign-consistent wording.
# The disclosure pair for each quantity, both terms living on the CROSS_PATH
# population so they are recomputed on the cells the two paths share. Comparing
# a quantity on V_P5_pp against its counterpart on V_sp would mix an operator
# difference with a selection difference, which is exactly what PROTOCOL 3.9
# forbids, so the mapping points at the intersection columns and not at the
# own-population ones.
DISCLOSURE_PAIR = {
    "delta_learn_pp": ("x_delta_learn_pp", "x_delta_learn_sp"),
}


def _interval_for(
    records: Sequence[dict],
    quantity: str,
    metric: str,
    analysis: AnalysisConfig,
    unit: str,
) -> tuple[dict[str, Any], SupportCounts]:
    """One quantity's paired interval on its own population, without disclosure."""
    population = QUANTITY_POPULATION[quantity]
    fields = _fields_for(population)
    count_field = _count_field_for(population)
    formula = quantity_formulas(metric)[quantity]
    contributing = [
        r for r in records
        if isinstance(r.get(count_field), (int, float)) and (r.get(count_field) or 0) > 0
    ]
    interval = paired_interval(
        contributing, fields, formula,
        resamples=analysis.bootstrap_resamples,
        seed=analysis.bootstrap_seed,
        confidence=analysis.bootstrap_confidence,
        unit=unit,
    )
    return interval, support_counts(contributing, count_field)


def evaluate_quantity(
    records: Sequence[dict],
    quantity: str,
    metric: str,
    analysis: AnalysisConfig,
    unit: str = "scene",
) -> CellResult:
    """One reported cell: the estimate, its paired interval, support, disclosure.

    Every difference is paired by construction, because the interval is built by
    resampling scenes once and recomputing the whole quantity inside each
    replicate from that replicate's own means.
    """
    population = QUANTITY_POPULATION[quantity]
    interval, counts = _interval_for(records, quantity, metric, analysis, unit)

    # PROTOCOL 3.9's disclosure is two-path and is computed on the cells the two
    # paths share. Both terms and their difference come from the CROSS_PATH
    # population, whose fields sit in one tuple, so one scene draw serves all
    # three and the difference is recomputed inside each replicate.
    pair = DISCLOSURE_PAIR.get(quantity)
    counterpart = None
    difference = None
    per_point_term = PathEstimate(interval["estimate"], interval["lo"], interval["hi"])
    if pair is not None:
        pp_name, sp_name = pair
        pp_x, _ = _interval_for(records, pp_name, metric, analysis, unit)
        sp_x, _ = _interval_for(records, sp_name, metric, analysis, unit)
        diff, _ = _interval_for(records, "path_difference", metric, analysis, unit)
        per_point_term = PathEstimate(pp_x["estimate"], pp_x["lo"], pp_x["hi"])
        counterpart = PathEstimate(sp_x["estimate"], sp_x["lo"], sp_x["hi"])
        difference = PathEstimate(diff["estimate"], diff["lo"], diff["hi"])

    disclosure = near_zero_disclosure(
        quantity, per_point_term, counterpart, analysis, difference
    )
    return CellResult(
        quantity=quantity,
        metric=metric,
        population=population,
        estimate=interval["estimate"],
        lo=interval["lo"],
        hi=interval["hi"],
        n_scenes=counts.n_scenes,
        n_camera_pairs=counts.n_camera_pairs,
        n_feature_comparisons=counts.n_feature_comparisons,
        n_replicates=interval["n_replicates"],
        supported=counts.is_supported(analysis),
        disclosure=disclosure,
    )


def seed_sensitivity(per_seed: dict[int, float]) -> dict[str, float]:
    """Spread of a quantity across training seeds, reported beside every estimate.

    Stream U step 16 keeps the scene as the primary independent unit and forbids
    ensembling predictions before scoring, so seeds are averaged at the level of
    the scene statistic and their spread is reported separately rather than
    folded into the interval. A wide spread does not widen the scene interval;
    it is a different source of uncertainty and is named as one.
    """
    values = [v for v in per_seed.values() if math.isfinite(v)]
    if not values:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"),
                "range": float("nan"), "n_seeds": 0}
    mean = sum(values) / len(values)
    return {
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "n_seeds": len(values),
    }


def three_rung_decomposition(
    phase3_oracle: float,
    phase4_transport_only: float,
    cl_transport: float,
    predict_with_depth: float,
) -> dict[str, float]:
    """Stream X step 27, with each rung labelled by the estimator that defines it.

    The rungs are not collapsed into one end-to-end residual, and the Phase 5
    rung is defined only through the information-symmetric comparison. The
    Phase 4 rung keeps its own accepted estimator, because the Phase 5 redesign
    does not retroactively alter Phase 4.
    """
    return {
        # Rung 0, Phase 3: how far exact geometry falls short of the target's
        # own features. Reported as the oracle level, not as a difference.
        "representation_limitation_oracle": phase3_oracle,
        # Rung 1, Phase 4, under the accepted Phase 4 estimator.
        "estimated_geometry_limitation": phase3_oracle - phase4_transport_only,
        # Rung 2, Phase 5, information symmetric by construction.
        "learned_vs_explicit_limitation": cl_transport - predict_with_depth,
    }
