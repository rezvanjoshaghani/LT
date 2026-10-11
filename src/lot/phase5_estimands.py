"""Phase 5 estimands: Streams V and W.

This module holds the arithmetic of the phase and nothing else. It does not read
a cache, a manifest, or a parquet; it takes per-pair scored records and turns
them into the reported quantities with their paired intervals. Keeping it free
of I/O is what makes the headline arithmetic testable without the cluster.

Three populations, kept separate on purpose and never silently merged.

    V_P5_pp    the primary support. The GT-evaluable samples Context-Lift
               Transport-Only can validly transport with context-image-scaled
               context depth. CL-Transport, Predict-with-Depth, No-Warp-Copy,
               and Mean-Feature are all scored on exactly this set.

    V_form     the formulation-diagnostic support, the intersection of V_P5_pp
               with the accepted Phase 4 target-lift set, on the target patch
               cell both estimators share. Only TL-Reference and CL-Transport
               are compared here, and this population never redefines the
               headline one. No-Warp-Copy and Mean-Feature sit beside them as
               levels, with no margin built on either.

    V_sp       the accepted Phase 4 context-image-scale splat-pool scored-cell
               support, used unchanged for the secondary operational
               comparison after its context-side symmetry was verified.
               No-Warp-Copy and Mean-Feature sit beside the two arms as
               levels.

The headline estimand is

    delta_learn_pp = CL-Transport - Predict-with-Depth   on V_P5_pp

and it is the only quantity that may be called the learned-versus-explicit
transformation limitation. The Phase 4 target-lift score is never substituted
into it; a helper below raises if a caller tries.

The end of the module holds what the reporting tables read, added before
commit E so it freezes with the rest: the public interval with its replicate
count, the disclosure terms with theirs, the comparison-weighted diagnostic,
the region contrasts, and the L2 companions. None of them changes a quantity,
a population, or a wording.
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
# The Mean-Feature floor sits beside No-Warp-Copy on each support, as CLAUDE.md
# requires of every metric. PROTOCOL 3.7 defines it under raw cosine only, so it
# carries its two raw columns and no centered ones.
PRIMARY_FIELDS = (
    "cl_raw", "cl_centered", "cl_l2_raw", "cl_l2_centered",
    "predict_raw", "predict_centered", "predict_l2_raw", "predict_l2_centered",
    "nowarp_raw", "nowarp_centered", "nowarp_l2_raw", "nowarp_l2_centered",
    "meanfeat_raw", "meanfeat_l2_raw",
)
FORMULATION_FIELDS = (
    "tl_form_raw", "tl_form_centered", "tl_form_l2_raw", "tl_form_l2_centered",
    "cl_form_raw", "cl_form_centered", "cl_form_l2_raw", "cl_form_l2_centered",
    # The two floors on V_form, appended by reporting_rules.md section 6.
    # No-Warp-Copy carries all four columns, Mean-Feature its two raw ones.
    "nowarp_form_raw", "nowarp_form_centered",
    "nowarp_form_l2_raw", "nowarp_form_l2_centered",
    "meanfeat_form_raw", "meanfeat_form_l2_raw",
)
SPLAT_FIELDS = (
    "sp_transport_raw", "sp_transport_centered",
    "sp_transport_l2_raw", "sp_transport_l2_centered",
    "sp_predict_raw", "sp_predict_centered",
    "sp_predict_l2_raw", "sp_predict_l2_centered",
    "sp_nowarp_raw", "sp_nowarp_centered",
    "sp_nowarp_l2_raw", "sp_nowarp_l2_centered",
    "sp_meanfeat_raw", "sp_meanfeat_l2_raw",
)
# Counts and diagnostics. Never bootstrapped as scores, and the near-zero rule
# does not apply to them because they are not in score space.
# PROTOCOL 3.9's cross-path common-valid columns. Both paths live in one field
# tuple deliberately: a single bootstrap draw then serves both, which is what
# makes the path difference paired rather than a subtraction of two independent
# intervals.
INTERSECTION_FIELDS = tuple(
    f"x_{arm}_{column}"
    for arm in ("cl", "predict", "nowarp", "sp_transport", "sp_predict", "sp_nowarp")
    for column in ("raw", "centered", "l2_raw", "l2_centered")
) + tuple(
    # The Mean-Feature floor on each path, under raw metrics only.
    f"x_{arm}_{column}"
    for arm in ("meanfeat", "sp_meanfeat")
    for column in ("raw", "l2_raw")
)
COUNT_FIELDS = (
    "n_primary", "n_formulation", "n_splat", "n_intersect", "n_predict_nonfinite",
)

# ---------------------------------------------------------------------------
# The landing-offset diagnostic. Pre-registered 2026-10-09, before any Phase 5
# result on real data existed, in
# validation/evidence/phase5/landing_offset_diagnostic.md. It sizes the
# landing read on real features. It is model free, it is not an estimand of the
# phase, and it is never subtracted from the headline.
# ---------------------------------------------------------------------------

# Context-Lift Transport-Only with ground-truth context depth. A reference
# condition, as Oracle-Transport is, and never a method under comparison.
CL_ORACLE = "Context-Lift Oracle-Transport"

# Five frozen upper edges in configs/phase5.yaml, closed on the right, and one
# open bin above the last edge. lot.phase5.landing_offset_edges binds the
# config to this count, so a record can never carry bins this layer does not
# read.
N_OFFSET_BINS = 6
OFFSET_BINS = tuple(f"b{k}" for k in range(N_OFFSET_BINS))
OFFSET_WHOLE = "all"
OFFSET_ARMS = ("cl_oracle", "nowarp")
METRIC_COLUMNS = ("raw", "centered", "l2_raw", "l2_centered")
# Mean-Feature's prediction is the centering vector itself, so PROTOCOL 3.7
# defines it under raw cosine only. Its centered columns do not exist rather
# than being filled with a manufactured value.
MEAN_FEATURE_COLUMNS = ("raw", "l2_raw")
LANDING_OFFSET = "landing_offset"
# The read deficit's population: pairs with at least one near-grid sample. It
# reads the near-grid bin and the whole support of the same pairs, so one scene
# draw serves both and the difference is paired.
READ_DEFICIT = "landing_offset_deficit"


def offset_count_field(label: str) -> str:
    return f"offset_n_{label}"


def offset_fields(label: str) -> tuple[str, ...]:
    """The score columns of one offset bin, or of the whole diagnostic support.

    Context-Lift with ground-truth depth and the No-Warp-Copy floor carry all
    four PROTOCOL 3.7 columns; the Mean-Feature floor carries its raw ones.
    """
    return tuple(
        f"offset_{arm}_{column}_{label}"
        for arm in OFFSET_ARMS
        for column in METRIC_COLUMNS
    ) + tuple(f"offset_meanfeat_{column}_{label}" for column in MEAN_FEATURE_COLUMNS)


def landing_offset_record_fields() -> tuple[str, ...]:
    """Every per-pair column the diagnostic writes, in record order."""
    return tuple(
        name
        for label in OFFSET_BINS + (OFFSET_WHOLE,)
        for name in (offset_count_field(label), *offset_fields(label))
    )


def offset_population(label: str) -> str:
    return f"{LANDING_OFFSET}_{label}"


ALL_FIELDS = (
    PRIMARY_FIELDS + FORMULATION_FIELDS + SPLAT_FIELDS
    + INTERSECTION_FIELDS + COUNT_FIELDS + landing_offset_record_fields()
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

    forms = {
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
        # No-Warp-Copy on the formulation support, reporting_rules.md section
        # 6. A level beside the two arms. No margin is built on it.
        "no_warp_copy_form": lambda v: v[f"nowarp_form_{m}"],
        # Stream W step 23: the secondary operational gap on the splat path.
        "delta_learn_sp": lambda v: v[f"sp_transport_{m}"] - v[f"sp_predict_{m}"],
        "sp_transport": lambda v: v[f"sp_transport_{m}"],
        "sp_predict": lambda v: v[f"sp_predict_{m}"],
        # No-Warp-Copy on the splat-pool support, reporting_rules.md section 6
        # and specification step 38. A level beside the two arms. The two
        # margins below subtract the same column.
        "sp_no_warp_copy": lambda v: v[f"sp_nowarp_{m}"],
        "sp_transport_margin": lambda v: v[f"sp_transport_{m}"] - v[f"sp_nowarp_{m}"],
        "sp_predict_margin": lambda v: v[f"sp_predict_{m}"] - v[f"sp_nowarp_{m}"],
        # PROTOCOL 3.9's disclosure terms, every one recomputed on the
        # cross-path common-valid cell set before differencing. Each interpreted
        # effect has a per-point term, a splat term, and a path difference, and
        # all three read one field tuple so one scene draw serves them.
        "x_delta_learn_pp": lambda v: v[f"x_cl_{m}"] - v[f"x_predict_{m}"],
        "x_delta_learn_sp": (
            lambda v: v[f"x_sp_transport_{m}"] - v[f"x_sp_predict_{m}"]
        ),
        "x_cl_margin": lambda v: v[f"x_cl_{m}"] - v[f"x_nowarp_{m}"],
        "x_sp_transport_margin": (
            lambda v: v[f"x_sp_transport_{m}"] - v[f"x_sp_nowarp_{m}"]
        ),
        "x_predict_margin": lambda v: v[f"x_predict_{m}"] - v[f"x_nowarp_{m}"],
        "x_sp_predict_margin": (
            lambda v: v[f"x_sp_predict_{m}"] - v[f"x_sp_nowarp_{m}"]
        ),
        "path_difference_learn": lambda v: (
            (v[f"x_cl_{m}"] - v[f"x_predict_{m}"])
            - (v[f"x_sp_transport_{m}"] - v[f"x_sp_predict_{m}"])
        ),
        "path_difference_cl_margin": lambda v: (
            (v[f"x_cl_{m}"] - v[f"x_nowarp_{m}"])
            - (v[f"x_sp_transport_{m}"] - v[f"x_sp_nowarp_{m}"])
        ),
        "path_difference_predict_margin": lambda v: (
            (v[f"x_predict_{m}"] - v[f"x_nowarp_{m}"])
            - (v[f"x_sp_predict_{m}"] - v[f"x_sp_nowarp_{m}"])
        ),
    }

    # The Mean-Feature floor on each record's own support. PROTOCOL 3.7 defines
    # it under raw cosine only, so these cells do not exist under centering.
    # Each is the absolute level of a floor, not an interpreted effect, so the
    # near-zero rule does not apply to it.
    if m == "raw":
        forms["mean_feature"] = lambda v: v["meanfeat_raw"]
        forms["mean_feature_form"] = lambda v: v["meanfeat_form_raw"]
        forms["sp_mean_feature"] = lambda v: v["sp_meanfeat_raw"]
        forms["x_mean_feature"] = lambda v: v["x_meanfeat_raw"]
        forms["x_sp_mean_feature"] = lambda v: v["x_sp_meanfeat_raw"]

    # The landing-offset diagnostic: the oracle level, the floor, and the margin
    # in every offset bin and on the whole diagnostic support, then the paired
    # read deficit, which is the near-grid bin minus the whole support.
    for label in OFFSET_BINS + (OFFSET_WHOLE,):
        cl = f"offset_cl_oracle_{m}_{label}"
        nowarp = f"offset_nowarp_{m}_{label}"
        forms[f"cl_oracle_offset_{label}"] = lambda v, cl=cl: v[cl]
        forms[f"nowarp_offset_{label}"] = lambda v, nowarp=nowarp: v[nowarp]
        forms[f"cl_oracle_offset_margin_{label}"] = (
            lambda v, cl=cl, nowarp=nowarp: v[cl] - v[nowarp]
        )
        if m == "raw":
            meanfeat = f"offset_meanfeat_raw_{label}"
            forms[f"meanfeat_offset_{label}"] = lambda v, meanfeat=meanfeat: v[meanfeat]
    near = f"offset_cl_oracle_{m}_{OFFSET_BINS[0]}"
    whole = f"offset_cl_oracle_{m}_{OFFSET_WHOLE}"
    forms["read_deficit"] = lambda v: v[near] - v[whole]
    return forms


# Which population each quantity is defined on. A quantity computed on the wrong
# population is the single most damaging mistake available in this phase, so the
# mapping is explicit and asserted rather than implied by naming.
QUANTITY_POPULATION = {
    "cl_transport": PER_POINT,
    "predict_with_depth": PER_POINT,
    "no_warp_copy": PER_POINT,
    "mean_feature": PER_POINT,
    "cl_margin": PER_POINT,
    "predict_margin": PER_POINT,
    "delta_learn_pp": PER_POINT,
    "delta_formulation": FORMULATION,
    "tl_reference": FORMULATION,
    "cl_on_formulation_support": FORMULATION,
    "no_warp_copy_form": FORMULATION,
    "mean_feature_form": FORMULATION,
    "delta_learn_sp": SPLAT_POOL,
    "sp_transport": SPLAT_POOL,
    "sp_predict": SPLAT_POOL,
    "sp_no_warp_copy": SPLAT_POOL,
    "sp_transport_margin": SPLAT_POOL,
    "sp_predict_margin": SPLAT_POOL,
    "sp_mean_feature": SPLAT_POOL,
    "x_delta_learn_pp": CROSS_PATH,
    "x_delta_learn_sp": CROSS_PATH,
    "x_cl_margin": CROSS_PATH,
    "x_sp_transport_margin": CROSS_PATH,
    "x_predict_margin": CROSS_PATH,
    "x_sp_predict_margin": CROSS_PATH,
    "path_difference_learn": CROSS_PATH,
    "path_difference_cl_margin": CROSS_PATH,
    "path_difference_predict_margin": CROSS_PATH,
    "x_mean_feature": CROSS_PATH,
    "x_sp_mean_feature": CROSS_PATH,
    "read_deficit": READ_DEFICIT,
}
for _label in OFFSET_BINS + (OFFSET_WHOLE,):
    for _name in ("cl_oracle_offset", "nowarp_offset", "cl_oracle_offset_margin",
                  "meanfeat_offset"):
        QUANTITY_POPULATION[f"{_name}_{_label}"] = offset_population(_label)
del _label, _name

# Quantities that live in score space, and to which the frozen 0.003 near-zero
# disclosure applies. Counts and fractions are excluded by Stream AB step 35's
# instruction not to apply the operator band to dimensionless quantities.
# The near-zero rule is scoped by PROTOCOL 3.9 to "an interpreted effect", which
# is a claim about one method relative to another. It is not every quantity that
# happens to be measured in score units. The set is therefore INTERPRETED_EFFECTS
# and is defined below, next to the disclosure registry it must stay consistent
# with; an earlier version took frozenset(QUANTITY_POPULATION), which swept in
# the cross-path disclosure terms and the path differences and then told
# path_difference_learn, which is itself the difference between the two paths,
# that it had no second evaluation path by construction.
#
# Absolute levels (cl_transport, sp_transport, tl_reference) are excluded for the
# same reason: a level near zero is not a near-zero *effect*, and the three
# licensed wordings are all statements about an advantage.


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
    # quantity: (per-point term, splat term, path difference), all on CROSS_PATH.
    "delta_learn_pp": ("x_delta_learn_pp", "x_delta_learn_sp", "path_difference_learn"),
    # The operational gap is the same effect read from the other side, so it is
    # disclosed against the same pair with the roles kept in path order.
    "delta_learn_sp": ("x_delta_learn_pp", "x_delta_learn_sp", "path_difference_learn"),
    "cl_margin": ("x_cl_margin", "x_sp_transport_margin", "path_difference_cl_margin"),
    "sp_transport_margin": (
        "x_cl_margin", "x_sp_transport_margin", "path_difference_cl_margin"
    ),
    "predict_margin": (
        "x_predict_margin", "x_sp_predict_margin", "path_difference_predict_margin"
    ),
    "sp_predict_margin": (
        "x_predict_margin", "x_sp_predict_margin", "path_difference_predict_margin"
    ),
}

# The quantities PROTOCOL 3.9's near-zero wording applies to: the interpreted
# effects, meaning differences a reader would take as a claim about one method
# relative to another. Every one either has a disclosure pair above or is
# declared here as having no second path by construction. A quantity in this
# set with neither is a programming error and evaluate_quantity raises, so the
# single-path wording can never be reached by omission.
INTERPRETED_EFFECTS = frozenset({
    "delta_learn_pp", "delta_learn_sp",
    "cl_margin", "sp_transport_margin",
    "predict_margin", "sp_predict_margin",
    "delta_formulation",
})
# The formulation diagnostic compares two per-point estimators and has no splat
# counterpart at all, so its single-path disclosure is a fact of the design
# rather than a fallback.
SINGLE_PATH_BY_CONSTRUCTION = frozenset({"delta_formulation"})

# PROTOCOL 3.9's near-zero discipline applies exactly to the interpreted
# effects. Everything else in score space is reported with its interval and
# no near-zero wording, because none of the three licensed sentences would be
# true of it.
SCORE_SPACE_QUANTITIES = INTERPRETED_EFFECTS


# Decision 2 of reporting_rules.md quotes six sentences, copied here verbatim.
# They are the only sentences printed with near_zero True.
# WORDING_VETO, WORDING_SMALL_SIGN_CONSISTENT, WORDING_PATH_SENSITIVE and
# WORDING_OUTSIDE_ON_COMMON_CELLS are its wordings 1 to 4.
# WORDING_ONLY_PATH_CLEAR and WORDING_ONLY_PATH_INCLUDES_ZERO are its two
# single-path sentences.
# Two labels are not decision 2 wording. They are carried over from the code
# at cc20e3e and are printed only with near_zero False. WORDING_OUTSIDE_BAND
# marks a cell that is not engaged. WORDING_NOT_ESTIMABLE marks a reported
# effect that is not finite.
# Equivalence is never among these sentences, under any branch, because no
# equivalence region was ever frozen.
WORDING_NOT_ESTIMABLE = "not estimable on this cell"
WORDING_OUTSIDE_BAND = "effect outside the operator band"
WORDING_VETO = (
    "no claim of advantage; the effect is at the scale of evaluation-path choice"
)
WORDING_SMALL_SIGN_CONSISTENT = "small, sign-consistent effect"
WORDING_PATH_SENSITIVE = (
    "effect licensed but its size is path-sensitive; the two evaluation paths "
    "do not agree about whether it sits inside the operator band"
)
WORDING_OUTSIDE_ON_COMMON_CELLS = (
    "effect licensed; inside the operator band on its own support but outside "
    "it on both paths' common cells"
)
WORDING_ONLY_PATH_CLEAR = (
    "within the operator band on its only path; its size is not certified by a "
    "second path"
)
WORDING_ONLY_PATH_INCLUDES_ZERO = (
    "no measurable difference at the reported scale; this quantity has no "
    "second evaluation path by construction"
)


def near_zero_disclosure(
    quantity: str,
    per_point: PathEstimate,
    splat_pool: PathEstimate | None,
    analysis: AnalysisConfig,
    difference: PathEstimate | None = None,
    reported: PathEstimate | None = None,
) -> dict[str, Any]:
    """PROTOCOL 3.9's near-zero rule, as reporting_rules.md decision 2 states it.

    The trigger. PROTOCOL 3.9 scopes its discipline to "an interpreted effect
    no larger than this tolerance". Decision 2 reads that scope literally. The
    wording is engaged when, and only when, the reported effect is finite and
    its estimate has magnitude at most path_agreement_tolerance, 0.003. The
    reported effect is the one the cell's table shows. `reported` defaults to
    `per_point` for callers that have only one number. The path terms never
    engage the wording.

    The terms. Once engaged, the wording is decided by the two cross-path
    estimates, per_point and splat_pool. Both are recomputed on the cells the
    two paths share, with paired intervals. They and their paired difference
    are returned beside the wording, engaged or not.

    The two-path wordings, when engaged, in order:

    1. The terms differ in sign, or either interval includes zero: no claim of
       advantage. This is 3.9's veto. Its clause is stated unconditionally,
       so it is tested before the band questions, never after them.
    2. Both terms inside the band: a small, sign-consistent effect.
    3. Exactly one term inside the band: licensed, but path-sensitive.
    4. Neither term inside the band: licensed, inside the band on its own
       support but outside it on both paths' common cells.

    A quantity with no second path by construction passes splat_pool=None.
    Its wording follows the reported interval. Clear of zero, it is within the
    band on its only path and its size is not certified. Including zero, it
    is no measurable difference at the reported scale.

    A quantity that has a second path, but whose cross-path terms are not
    finite, gets neither single-path sentence. "By construction" would be
    false of it. When the reported effect is in the band, the cell is
    engaged and gets 3.9's veto. PROTOCOL 3.9 restricts the interpretation
    of such an effect to content the two paths share, and here they share
    none. The veto is the only decision 2 sentence that is true of this
    cell. Sign agreement and clearance are returned as None, because they
    are undefined rather than failed. A reported effect that is not finite
    is not estimable on this cell and is never engaged.

    What the previous code did, and why it changed. It engaged when the
    reported effect or either path term fell in the band. A clearly nonzero
    headline gap whose splat term sat near zero then printed "no claim of
    advantage". That made the specification's outcome 49 unreachable, and
    3.9's scope sentence does not reach such an effect. It printed wording 3
    when neither term was in the band, which said the paths disagree when
    they agree. Its single-path sentence claimed no measurable difference for
    an interval that excludes zero. It returned "not estimable" whenever the
    per-point term was undefined, even with the reported effect in the band.
    Decision 2 names the first three and replaces them. The fourth follows
    from its trigger, which engages on the reported effect alone, and the
    veto now covers that cell.

    An earlier version still saw only one path. It could issue the strongest
    wording while the other path reversed sign. The second path has been a
    required argument since, and None is reserved for a quantity without one.

    Applied only to the interpreted effects. Counts, fractions, absolute
    levels, and the cross-path disclosure terms are not claims of advantage,
    and none of the licensed sentences would be true of them.
    """
    if quantity not in SCORE_SPACE_QUANTITIES:
        return {"near_zero": False, "applicable": False}

    band = analysis.path_agreement_tolerance
    if reported is None:
        reported = per_point

    if splat_pool is None:
        path_difference = None
    elif difference is not None:
        # The difference carries its own paired interval, from the same scene
        # draw, rather than being a bare subtraction of two point estimates.
        path_difference = dataclasses.asdict(difference)
    else:
        path_difference = {
            "estimate": per_point.estimate - splat_pool.estimate,
            "lo": float("nan"), "hi": float("nan"),
        }

    def disclosure(
        near_zero: bool,
        wording: str,
        paths_agree_in_sign: bool | None = None,
        both_intervals_exclude_zero: bool | None = None,
    ) -> dict[str, Any]:
        # Every applicable branch returns the same keys, so a reader can always
        # see both terms and their difference beside the wording.
        return {
            "near_zero": bool(near_zero),
            "applicable": True,
            "band": band,
            "reported": dataclasses.asdict(reported),
            "per_point": dataclasses.asdict(per_point),
            "splat_pool": (
                dataclasses.asdict(splat_pool) if splat_pool is not None else None
            ),
            "path_difference": path_difference,
            "paths_agree_in_sign": paths_agree_in_sign,
            "both_intervals_exclude_zero": both_intervals_exclude_zero,
            "wording": wording,
        }

    if not reported.finite:
        # An undefined effect still reports both terms, so a reader can see
        # that the cell was empty rather than that the effect was large.
        return disclosure(False, WORDING_NOT_ESTIMABLE)

    # Decision 2's trigger: the reported effect, and nothing else.
    engaged = reported.within(band)

    if splat_pool is None:
        # No second path by construction. 3.9's licence for a sign-consistent
        # effect is conditional on both paths, and one path cannot supply it.
        if not engaged:
            wording = WORDING_OUTSIDE_BAND
        elif reported.excludes_zero:
            wording = WORDING_ONLY_PATH_CLEAR
        else:
            wording = WORDING_ONLY_PATH_INCLUDES_ZERO
        return disclosure(engaged, wording)

    if not (per_point.finite and splat_pool.finite):
        # A second path exists, but the cells both paths share gave no finite
        # term. PROTOCOL 3.9 restricts the effect's interpretation to content
        # the two paths share, and here they share none. The veto is the only
        # decision 2 sentence that is true of this cell. The branch stays
        # explicit so that sign agreement and clearance are None, because they
        # are undefined rather than failed.
        return disclosure(engaged, WORDING_VETO if engaged else WORDING_OUTSIDE_BAND)

    pp_in = per_point.within(band)
    sp_in = splat_pool.within(band)
    same_sign = per_point.sign == splat_pool.sign and per_point.sign != 0
    both_clear = per_point.excludes_zero and splat_pool.excludes_zero

    if not engaged:
        wording = WORDING_OUTSIDE_BAND
    elif not same_sign or not both_clear:
        # 3.9's veto, tested first. Testing an in-band branch first, as an
        # earlier version did, let a cell whose two paths reverse sign be
        # reported as merely path-sensitive.
        wording = WORDING_VETO
    elif pp_in and sp_in:
        wording = WORDING_SMALL_SIGN_CONSISTENT
    elif pp_in or sp_in:
        wording = WORDING_PATH_SENSITIVE
    else:
        wording = WORDING_OUTSIDE_ON_COMMON_CELLS
    return disclosure(engaged, wording, bool(same_sign), bool(both_clear))


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
    if population == READ_DEFICIT:
        return offset_fields(OFFSET_BINS[0]) + offset_fields(OFFSET_WHOLE)
    for label in OFFSET_BINS + (OFFSET_WHOLE,):
        if population == offset_population(label):
            return offset_fields(label)
    raise ValueError(f"unknown population {population!r}")


def _count_field_for(population: str) -> str:
    if population == READ_DEFICIT:
        return offset_count_field(OFFSET_BINS[0])
    for label in OFFSET_BINS + (OFFSET_WHOLE,):
        if population == offset_population(label):
            return offset_count_field(label)
    return {
        PER_POINT: "n_primary",
        FORMULATION: "n_formulation",
        SPLAT_POOL: "n_splat",
        CROSS_PATH: "n_intersect",
    }[population]


def _population_interval(
    records: Sequence[dict],
    population: str,
    formula: Callable[[dict[str, float]], float],
    analysis: AnalysisConfig,
    unit: str,
) -> tuple[dict[str, Any], SupportCounts]:
    """A statistic's paired interval over the records that hold its population.

    A record contributes when its population count is above zero. The
    statistic reads that population's score fields, so one scene draw serves
    every field it reads.
    """
    fields = _fields_for(population)
    count_field = _count_field_for(population)
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


def _interval_for(
    records: Sequence[dict],
    quantity: str,
    metric: str,
    analysis: AnalysisConfig,
    unit: str,
) -> tuple[dict[str, Any], SupportCounts]:
    """One quantity's paired interval on its own population, without disclosure."""
    population = QUANTITY_POPULATION[quantity]
    formula = quantity_formulas(metric)[quantity]
    return _population_interval(records, population, formula, analysis, unit)


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
    if (
        quantity in INTERPRETED_EFFECTS
        and pair is None
        and quantity not in SINGLE_PATH_BY_CONSTRUCTION
    ):
        raise ValueError(
            f"{quantity} is an interpreted effect with no disclosure pair and is "
            "not declared single-path by construction; PROTOCOL 3.9 does not "
            "permit falling back to a one-path wording when a counterpart exists"
        )
    counterpart = None
    difference = None
    per_point_term = PathEstimate(interval["estimate"], interval["lo"], interval["hi"])
    if pair is not None:
        pp_name, sp_name, diff_name = pair
        pp_x, _ = _interval_for(records, pp_name, metric, analysis, unit)
        sp_x, _ = _interval_for(records, sp_name, metric, analysis, unit)
        diff, _ = _interval_for(records, diff_name, metric, analysis, unit)
        per_point_term = PathEstimate(pp_x["estimate"], pp_x["lo"], pp_x["hi"])
        counterpart = PathEstimate(sp_x["estimate"], sp_x["lo"], sp_x["hi"])
        difference = PathEstimate(diff["estimate"], diff["lo"], diff["hi"])

    disclosure = near_zero_disclosure(
        quantity, per_point_term, counterpart, analysis, difference,
        reported=PathEstimate(interval["estimate"], interval["lo"], interval["hi"]),
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
    phase3_reference_ceiling: float,
    phase4_depth_tax: PathEstimate,
    delta_learn_pp: CellResult,
) -> dict[str, dict[str, Any]]:
    """Stream X step 27, with each rung labelled by the estimator that defines it.

    reporting_rules.md section 6. The Phase 3 Oracle-Transport ceiling, the
    Phase 4 estimated-geometry result under its own estimator, and
    delta_learn_pp are reported separately and never folded into one
    residual.

    Each rung is taken as its own phase estimated it, and nothing is computed
    across rungs. No rung is a difference across phases. Phase 4's own
    estimator is its depth tax, the matched ceiling minus the estimated
    score on Phase 4's matched population. The Phase 3 ceiling is measured on
    another population, so subtracting a Phase 4 score from it would mix a
    selection difference into rung 1. The Phase 5 redesign does not
    retroactively alter Phase 4, so Phase 4's rung keeps its own estimator.

    phase3_reference_ceiling is the Phase 3 Oracle-Transport ceiling of the
    cell, as Phase 4's tables carry it in reference_ceiling_phase3.
    phase4_depth_tax is Phase 4's depth tax for the same cell with its
    interval, passed through unchanged. delta_learn_pp is the Phase 5 cell
    from evaluate_quantity. The Phase 5 rung is defined only through the
    information-symmetric comparison, so any other quantity in its place
    raises HeadlineSubstitutionError.
    """
    if delta_learn_pp.quantity != "delta_learn_pp":
        raise HeadlineSubstitutionError(
            f"the Phase 5 rung is delta_learn_pp, Context-Lift Transport-Only minus "
            f"Predict-with-Depth on its own support; {delta_learn_pp.quantity!r} "
            "cannot stand in for it"
        )
    return {
        # Rung 0, Phase 3: how far exact geometry falls short of the target's
        # own features. Reported as the ceiling level, not as a difference.
        "representation_limitation_oracle": {
            "phase": 3,
            "estimator": "reference_ceiling_phase3",
            "estimate": phase3_reference_ceiling,
        },
        # Rung 1, Phase 4, under the accepted Phase 4 estimator.
        "estimated_geometry_limitation": {
            "phase": 4,
            "estimator": "depth_tax",
            **dataclasses.asdict(phase4_depth_tax),
        },
        # Rung 2, Phase 5, information symmetric by construction.
        "learned_vs_explicit_limitation": {
            "phase": 5,
            "estimator": "delta_learn_pp",
            "population": delta_learn_pp.population,
            "estimate": delta_learn_pp.estimate,
            "lo": delta_learn_pp.lo,
            "hi": delta_learn_pp.hi,
            "n_replicates": delta_learn_pp.n_replicates,
            "supported": delta_learn_pp.supported,
        },
    }


# ---------------------------------------------------------------------------
# What the reporting tables read. Added before commit E, so they freeze with
# the rest of this module. None of them changes a quantity, a population, or
# a wording above. Each is a public form of what evaluate_quantity already
# computes, or a quantity the tables need beside it.
# ---------------------------------------------------------------------------

def population_fields(population: str) -> tuple[str, ...]:
    """The score fields every interval on one population reads."""
    return _fields_for(population)


def population_count_field(population: str) -> str:
    """The count field that decides which records hold one population.

    A record contributes to a cell on the population when this count is
    above zero. The read deficit's population is its near-grid bin's.
    """
    return _count_field_for(population)


# The L2 companions. PROTOCOL 3.7 pairs every cosine with an L2 distance on
# unit-normalized features, and CLAUDE.md requires every reported metric to
# carry both. Lower is closer. Each difference is named for its sign: positive
# means the first-named method is farther from the target. So
# l2_predict_minus_cl has the sign of delta_learn_pp, and l2_nowarp_minus_cl
# the sign of cl_margin.
#
# The L2 quantities live in their own registry, apart from the cosine one.
# None is an interpreted effect, and evaluate_quantity, the only path to
# near-zero wording, refuses an L2 metric. PROTOCOL 3.9 calibrates the 0.003
# band on cosine, so no L2 cell is worded.
L2_METRICS = ("l2_raw", "l2_centered")


def l2_quantity_formulas(metric: str) -> dict[str, Callable[[dict[str, float]], float]]:
    """Every L2 companion as a closed form over one replicate's field means.

    metric is 'l2_raw' or 'l2_centered'. The Mean-Feature floor exists under
    l2_raw only, as its cosine exists under raw only, per PROTOCOL 3.7.
    """
    if metric not in L2_METRICS:
        raise ValueError(f"an L2 metric is l2_raw or l2_centered, got {metric!r}")
    m = metric
    forms = {
        # Primary per-point support, V_P5_pp.
        "l2_cl_transport": lambda v: v[f"cl_{m}"],
        "l2_predict_with_depth": lambda v: v[f"predict_{m}"],
        "l2_no_warp_copy": lambda v: v[f"nowarp_{m}"],
        "l2_predict_minus_cl": lambda v: v[f"predict_{m}"] - v[f"cl_{m}"],
        "l2_nowarp_minus_cl": lambda v: v[f"nowarp_{m}"] - v[f"cl_{m}"],
        "l2_nowarp_minus_predict": lambda v: v[f"nowarp_{m}"] - v[f"predict_{m}"],
        # The formulation diagnostic's support, V_form.
        "l2_tl_reference": lambda v: v[f"tl_form_{m}"],
        "l2_cl_on_formulation_support": lambda v: v[f"cl_form_{m}"],
        "l2_no_warp_copy_form": lambda v: v[f"nowarp_form_{m}"],
        "l2_cl_form_minus_tl_form": lambda v: v[f"cl_form_{m}"] - v[f"tl_form_{m}"],
        # The splat-pool support, V_sp.
        "l2_sp_transport": lambda v: v[f"sp_transport_{m}"],
        "l2_sp_predict": lambda v: v[f"sp_predict_{m}"],
        "l2_sp_no_warp_copy": lambda v: v[f"sp_nowarp_{m}"],
        "l2_sp_predict_minus_transport": (
            lambda v: v[f"sp_predict_{m}"] - v[f"sp_transport_{m}"]
        ),
        "l2_sp_nowarp_minus_transport": (
            lambda v: v[f"sp_nowarp_{m}"] - v[f"sp_transport_{m}"]
        ),
        "l2_sp_nowarp_minus_predict": lambda v: v[f"sp_nowarp_{m}"] - v[f"sp_predict_{m}"],
    }
    if m == "l2_raw":
        forms["l2_mean_feature"] = lambda v: v["meanfeat_l2_raw"]
        forms["l2_mean_feature_form"] = lambda v: v["meanfeat_form_l2_raw"]
        forms["l2_sp_mean_feature"] = lambda v: v["sp_meanfeat_l2_raw"]
    return forms


# Which population each L2 companion is defined on, in table order.
L2_QUANTITY_POPULATION = {
    "l2_cl_transport": PER_POINT,
    "l2_predict_with_depth": PER_POINT,
    "l2_no_warp_copy": PER_POINT,
    "l2_mean_feature": PER_POINT,
    "l2_predict_minus_cl": PER_POINT,
    "l2_nowarp_minus_cl": PER_POINT,
    "l2_nowarp_minus_predict": PER_POINT,
    "l2_tl_reference": FORMULATION,
    "l2_cl_on_formulation_support": FORMULATION,
    "l2_no_warp_copy_form": FORMULATION,
    "l2_mean_feature_form": FORMULATION,
    "l2_cl_form_minus_tl_form": FORMULATION,
    "l2_sp_transport": SPLAT_POOL,
    "l2_sp_predict": SPLAT_POOL,
    "l2_sp_no_warp_copy": SPLAT_POOL,
    "l2_sp_mean_feature": SPLAT_POOL,
    "l2_sp_predict_minus_transport": SPLAT_POOL,
    "l2_sp_nowarp_minus_transport": SPLAT_POOL,
    "l2_sp_nowarp_minus_predict": SPLAT_POOL,
}


def quantity_population(quantity: str, metric: str) -> str:
    """The population a quantity is defined on, under one metric.

    Cosine quantities are read from QUANTITY_POPULATION and L2 companions
    from L2_QUANTITY_POPULATION. A quantity that is not defined under the
    metric's registry raises ValueError naming it.
    """
    if metric in L2_METRICS:
        registry = L2_QUANTITY_POPULATION
    elif metric in ("raw", "centered"):
        registry = QUANTITY_POPULATION
    else:
        raise ValueError(f"unknown metric {metric!r}")
    if quantity not in registry:
        raise ValueError(f"{quantity!r} is not a quantity under metric {metric!r}")
    return registry[quantity]


def _formula_for(quantity: str, metric: str) -> Callable[[dict[str, float]], float]:
    forms = l2_quantity_formulas(metric) if metric in L2_METRICS else quantity_formulas(metric)
    if quantity not in forms:
        raise ValueError(f"{quantity!r} is not defined under metric {metric!r}")
    return forms[quantity]


def quantity_interval(
    records: Sequence[dict],
    quantity: str,
    metric: str,
    analysis: AnalysisConfig,
    unit: str = "scene",
) -> tuple[dict[str, Any], SupportCounts]:
    """One quantity's paired interval on its own population, and its support.

    The public form of the interval evaluate_quantity reports, without the
    near-zero disclosure. unit 'scene' is PROTOCOL 3.4's primary interval and
    'camera_pair' its secondary one. For the camera-pair unit every record
    carries a camera_pair key. Under an L2 metric the quantity is an L2
    companion, read through l2_quantity_formulas.

    The interval holds the estimate, lo, hi, n_units, and n_replicates, the
    number of draws in which the quantity was defined.
    """
    if metric in L2_METRICS:
        population = quantity_population(quantity, metric)
        return _population_interval(
            records, population, _formula_for(quantity, metric), analysis, unit
        )
    return _interval_for(records, quantity, metric, analysis, unit)


def _term(name: str, interval: dict[str, Any]) -> dict[str, Any]:
    return {
        "quantity": name,
        "estimate": interval["estimate"],
        "lo": interval["lo"],
        "hi": interval["hi"],
        "n_units": interval["n_units"],
        "n_replicates": interval["n_replicates"],
    }


def disclosure_terms(
    records: Sequence[dict],
    quantity: str,
    metric: str,
    analysis: AnalysisConfig,
    unit: str = "scene",
) -> dict[str, Any]:
    """Every term of one interpreted effect's disclosure, with its replicates.

    evaluate_quantity returns each term's estimate and interval inside the
    cell's disclosure. This returns the same terms, computed by the same
    calls, with what the disclosure leaves out: each term's n_replicates and
    n_units, the reported effect's own support, and the support of the
    cross-path common cells the terms rest on.

    - reported: the effect on its own population, with its support counts;
    - per_point, splat_pool, path_difference: the three cross-path terms,
      each named by its quantity, for an effect with a second path;
    - cross_path_support: n_scenes, n_camera_pairs, n_feature_comparisons,
      and whether the shared cells meet the frozen support thresholds.

    A quantity with no second path by construction has per_point equal to the
    reported effect, and None for the rest, as its disclosure does. A
    quantity that is not an interpreted effect has no disclosure terms and
    raises ValueError.
    """
    if quantity not in INTERPRETED_EFFECTS:
        raise ValueError(
            f"{quantity!r} is not an interpreted effect, so PROTOCOL 3.9's disclosure "
            "has no terms for it"
        )
    pair = DISCLOSURE_PAIR.get(quantity)
    if pair is None and quantity not in SINGLE_PATH_BY_CONSTRUCTION:
        raise ValueError(
            f"{quantity} is an interpreted effect with no disclosure pair and is "
            "not declared single-path by construction"
        )
    interval, counts = _interval_for(records, quantity, metric, analysis, unit)
    out: dict[str, Any] = {
        "quantity": quantity,
        "metric": metric,
        "unit": unit,
        "reported": {
            **_term(quantity, interval),
            "population": QUANTITY_POPULATION[quantity],
            "n_scenes": counts.n_scenes,
            "n_camera_pairs": counts.n_camera_pairs,
            "n_feature_comparisons": counts.n_feature_comparisons,
            "supported": counts.is_supported(analysis),
        },
        "per_point": None,
        "splat_pool": None,
        "path_difference": None,
        "cross_path_support": None,
    }
    if pair is None:
        out["per_point"] = _term(quantity, interval)
        return out
    pp_name, sp_name, diff_name = pair
    pp, cross = _interval_for(records, pp_name, metric, analysis, unit)
    sp, _ = _interval_for(records, sp_name, metric, analysis, unit)
    diff, _ = _interval_for(records, diff_name, metric, analysis, unit)
    out["per_point"] = _term(pp_name, pp)
    out["splat_pool"] = _term(sp_name, sp)
    out["path_difference"] = _term(diff_name, diff)
    out["cross_path_support"] = {
        "n_scenes": cross.n_scenes,
        "n_camera_pairs": cross.n_camera_pairs,
        "n_feature_comparisons": cross.n_feature_comparisons,
        "supported": cross.is_supported(analysis),
    }
    return out


def comparison_weighted(records: Sequence[dict], quantity: str, metric: str) -> float:
    """PROTOCOL 3.4's comparison-weighted diagnostic for one quantity.

    Each field's mean is weighted by the record's count of feature
    comparisons on the quantity's population, its count field, and the
    quantity's closed form is applied to those means. A record outside the
    population carries no weight. It is a diagnostic shown beside the
    estimate, never the estimate: the estimand is the unweighted mean over
    camera pairs.

    The read deficit has none. Its two terms rest on different counts, the
    near-grid bin and the whole diagnostic support, so no one weight
    describes it, and asking raises ValueError.
    """
    population = quantity_population(quantity, metric)
    if population == READ_DEFICIT:
        raise ValueError(
            f"{quantity!r} is read on the read_deficit population, whose two terms "
            "rest on different counts, so it has no comparison-weighted value"
        )
    formula = _formula_for(quantity, metric)
    fields = _fields_for(population)
    count_field = _count_field_for(population)
    totals = dict.fromkeys(fields, 0.0)
    weights = dict.fromkeys(fields, 0.0)
    for record in records:
        weight = record.get(count_field)
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight <= 0):
            continue
        for field in fields:
            value = record.get(field)
            if isinstance(value, (int, float)) and math.isfinite(value):
                totals[field] += float(value) * weight
                weights[field] += weight
    means = {
        field: (totals[field] / weights[field] if weights[field] else float("nan"))
        for field in fields
    }
    return float(formula(means))


# The region contrasts of Stream Z, specification steps 31 and 32: boundary
# minus interior, and low minus high texture. Each is a difference between two
# regions of the same pairs, so a pair contributes only when it holds both
# regions, and the difference is recomputed inside each replicate from one
# scene draw. No near-zero wording applies to a contrast. reporting_rules.md
# section 6 registers none, and PROTOCOL 3.9's two paths do not define one.
REGION_CONTRASTS = {
    "boundary_minus_interior": ("boundary", "interior"),
    "low_minus_high_texture": ("low_texture", "high_texture"),
}
# The populations a region contrast is read on. Both carry per-region records.
CONTRAST_POPULATIONS = (PER_POINT, SPLAT_POOL)


def contrast_field(field: str, region: str) -> str:
    """A field of one region, as a pivoted contrast record names it."""
    return f"{field}@{region}"


def _positive_count(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def pivot_regions(
    records: Sequence[dict], contrast: str, population: str
) -> list[dict[str, Any]]:
    """One record per camera pair that holds both regions of a contrast.

    records are per-region records, each with scene, camera_pair, and region.
    Records of other regions are ignored. A pair contributes only when both
    regions' records exist and both have a positive count on the population,
    so the two sides of the contrast average over one set of pairs.

    Each pivoted record carries scene, camera_pair, every field of the
    population under both regions as contrast_field names it, each region's
    count, and the population's count field as the sum of the two: the
    comparisons the contrast rests on. Pairs come in sorted order. A region
    record seen twice for one pair raises ValueError.
    """
    if contrast not in REGION_CONTRASTS:
        raise ValueError(
            f"unknown contrast {contrast!r}; the registered contrasts are "
            f"{sorted(REGION_CONTRASTS)}"
        )
    if population not in CONTRAST_POPULATIONS:
        raise ValueError(
            f"a region contrast is read on the {CONTRAST_POPULATIONS} populations, "
            f"not on population {population!r}"
        )
    left, right = REGION_CONTRASTS[contrast]
    fields = _fields_for(population)
    count_field = _count_field_for(population)
    by_pair: dict[str, dict[str, dict]] = {}
    for record in records:
        region = record.get("region")
        if region not in (left, right):
            continue
        slot = by_pair.setdefault(record["camera_pair"], {})
        if region in slot:
            raise ValueError(
                f"camera pair {record['camera_pair']!r} holds its {region} record twice"
            )
        slot[region] = record
    pivot: list[dict[str, Any]] = []
    for pair in sorted(by_pair):
        slot = by_pair[pair]
        if left not in slot or right not in slot:
            continue
        a, b = slot[left], slot[right]
        if not (_positive_count(a.get(count_field)) and _positive_count(b.get(count_field))):
            continue
        if a["scene"] != b["scene"]:
            raise ValueError(
                f"camera pair {pair!r} names scene {a['scene']!r} in {left} and "
                f"{b['scene']!r} in {right}"
            )
        record: dict[str, Any] = {
            "scene": a["scene"],
            "camera_pair": pair,
            count_field: a[count_field] + b[count_field],
            contrast_field(count_field, left): a[count_field],
            contrast_field(count_field, right): b[count_field],
        }
        for field in fields:
            record[contrast_field(field, left)] = a.get(field)
            record[contrast_field(field, right)] = b.get(field)
        pivot.append(record)
    return pivot


def evaluate_contrast(
    pivot: Sequence[dict],
    contrast: str,
    quantity: str,
    metric: str,
    analysis: AnalysisConfig,
    unit: str = "scene",
) -> dict[str, Any]:
    """One region contrast of a quantity, with its paired interval and support.

    The statistic is the quantity on the left region minus the quantity on
    the right region, both read from one replicate's field means, so the
    contrast is paired through lot.paired_bootstrap.paired_interval. pivot
    comes from pivot_regions on the quantity's own population, which must be
    per-point or splat-pool. Support is counted on the both-region pairs.
    The contrast carries no near-zero wording.
    """
    if contrast not in REGION_CONTRASTS:
        raise ValueError(
            f"unknown contrast {contrast!r}; the registered contrasts are "
            f"{sorted(REGION_CONTRASTS)}"
        )
    population = quantity_population(quantity, metric)
    if population not in CONTRAST_POPULATIONS:
        raise ValueError(
            f"{quantity!r} is on population {population!r}, which has no region contrast"
        )
    formula = _formula_for(quantity, metric)
    left, right = REGION_CONTRASTS[contrast]
    fields = _fields_for(population)
    count_field = _count_field_for(population)
    markers = (contrast_field(count_field, left), contrast_field(count_field, right))
    for record in pivot:
        if not all(marker in record for marker in markers):
            raise ValueError(
                f"the pivot was not built on population {population!r}, which "
                f"{quantity!r} is defined on"
            )
    left_fields = [contrast_field(field, left) for field in fields]
    right_fields = [contrast_field(field, right) for field in fields]

    def statistic(means: dict[str, float]) -> float:
        on_left = {field: means[name] for field, name in zip(fields, left_fields)}
        on_right = {field: means[name] for field, name in zip(fields, right_fields)}
        return formula(on_left) - formula(on_right)

    interval = paired_interval(
        pivot, left_fields + right_fields, statistic,
        resamples=analysis.bootstrap_resamples,
        seed=analysis.bootstrap_seed,
        confidence=analysis.bootstrap_confidence,
        unit=unit,
    )
    counts = support_counts(pivot, count_field)
    return {
        "contrast": contrast,
        "quantity": quantity,
        "metric": metric,
        "unit": unit,
        "population": population,
        "left": left,
        "right": right,
        "estimate": interval["estimate"],
        "lo": interval["lo"],
        "hi": interval["hi"],
        "n_units": interval["n_units"],
        "n_replicates": interval["n_replicates"],
        "n_scenes": counts.n_scenes,
        "n_camera_pairs": counts.n_camera_pairs,
        "n_feature_comparisons": counts.n_feature_comparisons,
        "supported": counts.is_supported(analysis),
    }
