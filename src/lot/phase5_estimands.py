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

# Per-pair score fields. Each is the pair's mean score over its own supported
# samples, so a cell's estimand is the unweighted mean over camera pairs that
# PROTOCOL 3.4 fixes as the unit.
PRIMARY_FIELDS = (
    "cl_raw", "cl_centered",
    "predict_raw", "predict_centered",
    "nowarp_raw", "nowarp_centered",
)
FORMULATION_FIELDS = (
    "tl_form_raw", "tl_form_centered",
    "cl_form_raw", "cl_form_centered",
)
SPLAT_FIELDS = (
    "sp_transport_raw", "sp_transport_centered",
    "sp_predict_raw", "sp_predict_centered",
    "sp_nowarp_raw", "sp_nowarp_centered",
)
# Counts and diagnostics. Never bootstrapped as scores, and the near-zero rule
# does not apply to them because they are not in score space.
COUNT_FIELDS = (
    "n_primary", "n_formulation", "n_splat", "n_predict_nonfinite",
)

ALL_FIELDS = PRIMARY_FIELDS + FORMULATION_FIELDS + SPLAT_FIELDS + COUNT_FIELDS


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


def near_zero_disclosure(
    quantity: str, estimate: float, lo: float, hi: float, analysis: AnalysisConfig
) -> dict[str, Any]:
    """The frozen 0.003 rule, carried forward unchanged from PROTOCOL 3.9.

    An effect no larger than the operator band is not certified by that band.
    The exact estimate and its paired interval are printed, the cell is flagged,
    and the wording discipline applies: no equivalence is claimed, because no
    equivalence region was ever frozen.

    Applied only to score-space quantities. Counts and fractions are not in the
    metric's units and the band means nothing for them.
    """
    if quantity not in SCORE_SPACE_QUANTITIES:
        return {"near_zero": False, "applicable": False}
    band = analysis.path_agreement_tolerance
    if not math.isfinite(estimate):
        return {"near_zero": False, "applicable": True}
    flagged = abs(estimate) <= band
    interval_excludes_zero = (
        math.isfinite(lo) and math.isfinite(hi) and (lo > 0.0 or hi < 0.0)
    )
    return {
        "near_zero": bool(flagged),
        "applicable": True,
        "band": band,
        "interval_excludes_zero": bool(interval_excludes_zero),
        # The only claim a flagged cell licenses on its own. Equivalence is not
        # among them and is never asserted from a near-zero estimate.
        "wording": (
            "no measurable difference at the reported scale"
            if flagged and not interval_excludes_zero
            else "small, sign-consistent effect"
            if flagged
            else "effect outside the operator band"
        ),
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
    raise ValueError(f"unknown population {population!r}")


def _count_field_for(population: str) -> str:
    return {
        PER_POINT: "n_primary",
        FORMULATION: "n_formulation",
        SPLAT_POOL: "n_splat",
    }[population]


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
    fields = _fields_for(population)
    count_field = _count_field_for(population)
    formula = quantity_formulas(metric)[quantity]

    # Only pairs that actually contributed to this population may enter its cell.
    contributing = [
        r for r in records
        if isinstance(r.get(count_field), (int, float)) and (r.get(count_field) or 0) > 0
    ]
    counts = support_counts(contributing, count_field)
    interval = paired_interval(
        contributing, fields, formula,
        resamples=analysis.bootstrap_resamples,
        seed=analysis.bootstrap_seed,
        confidence=analysis.bootstrap_confidence,
        unit=unit,
    )
    disclosure = near_zero_disclosure(
        quantity, interval["estimate"], interval["lo"], interval["hi"], analysis
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
