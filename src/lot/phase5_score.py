"""Phase 5 scoring: build one pair's record on the three fixed supports.

Kept free of I/O so the arithmetic that produces the headline number can be
tested on analytic scenes rather than only on the cluster. The orchestration
that reads caches, manifests, and the accepted Phase 4 parquets lives in
lot.phase5 and calls into here.

The support discipline this module implements, from Stream V:

    The primary support is decided by Context-Lift Transport-Only, before any
    predictor output is looked at. CL-Transport, Predict-with-Depth, and
    No-Warp-Copy are then scored on exactly that set of samples. The predictor
    produces a complete target grid, so it could otherwise appear to win by
    answering only where answering is easy; fixing the support first removes
    that possibility structurally.

    A predictor output that is nonfinite on a supported sample is a model
    failure on that sample and is scored as one. It is not dropped, because
    dropping it would let a model improve its own score by failing. The count
    travels in the record so a reader can see whether it ever happened.

    No-Warp-Copy on this path is the context feature read at the *landing*
    coordinates, which is the "assume nothing moved" prediction for the
    location being scored. That is the same floor Phase 3 uses, read in the
    forward direction rather than the backward one.

Mean-Feature is the second floor CLAUDE.md requires beside every metric. It
predicts the frozen mean vector everywhere. Each record scores it on its own
support, against the same targets its other arms are scored against. PROTOCOL
3.7 defines it under raw cosine only, so it has no centered columns.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .context_lift import ContextLiftMap
from .encoders import (
    PATCH_SIZE,
    patch_cell_index,
    pixel_to_patch_coords,
    sample_features_bilinear,
    sample_map_bilinear,
)

# A prediction that is not finite cannot be scored. It is recorded as the worst
# value the metric can take rather than removed, so a model cannot raise its
# own score by declining to answer. In a healthy run this never fires and the
# count column is zero everywhere.
MODEL_FAILURE_COSINE = -1.0
# The largest distance two unit vectors can have, which is the L2 companion's
# worst attainable value and therefore the failure value under that metric.
MODEL_FAILURE_L2 = 2.0


def _unit(features: Tensor, eps: float = 1e-12) -> Tensor:
    """Unit-normalize, matching lot.evaluate.unit_normalize's convention."""
    return features / features.norm(dim=-1, keepdim=True).clamp_min(eps)


def _scored_mean(values: Tensor, failures: Tensor, failure_value: float) -> float:
    """Mean over a support, with model failures held at the metric's worst value."""
    if values.numel() == 0:
        return float("nan")
    held = torch.where(failures, torch.full_like(values, failure_value), values)
    return float(held.mean())


def score_predictions(
    prediction: Tensor, target: Tensor, center: Tensor
) -> tuple[float, float, int]:
    """Raw and centered mean cosine of one method against the target features.

    Kept for callers that want only the cosines. score_all_metrics is the full
    PROTOCOL 3.7 record and is what the reported schema is built from.
    """
    metrics = score_all_metrics(prediction, target, center)
    return metrics["cosine_raw"], metrics["cosine_centered"], metrics["n_failures"]


def score_all_metrics(
    prediction: Tensor, target: Tensor, center: Tensor
) -> dict[str, Any]:
    """The four PROTOCOL 3.7 metric columns for one method, plus the failure count.

    prediction, target: [N, C]. center: [C], the frozen Phase 3 global mean.

    Raw and centered cosine each carry an L2 companion on unit-normalized
    features, which CLAUDE.md and PROTOCOL 3.7 require of every reported metric.
    Reporting cosine alone would leave the Phase 5 schema incomplete against the
    frozen protocol and incomparable with the Phase 3 and Phase 4 tables.

    Centering is applied at the output level, immediately before scoring, per
    PROTOCOL 3.7. Nonfinite predictions are counted and scored as failures under
    every metric rather than excluded from any of them: for cosine the failure
    value is the worst attainable, and for L2 it is the largest distance two
    unit vectors can have, so a model cannot improve any reported number by
    declining to answer.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"prediction {tuple(prediction.shape)} != target {tuple(target.shape)}")
    if prediction.shape[0] == 0:
        nan = float("nan")
        return {
            "cosine_raw": nan, "cosine_centered": nan,
            "l2_raw": nan, "l2_centered": nan, "n_failures": 0,
        }
    failures = ~torch.isfinite(prediction).all(dim=-1)
    safe = torch.where(failures[:, None], torch.zeros_like(prediction), prediction)

    def pair(center_vector: Tensor | None) -> tuple[float, float]:
        a, b = (safe, target) if center_vector is None else (
            safe - center_vector, target - center_vector
        )
        # float32 before normalizing, exactly as lot.evaluate.value_agreement
        # does. Every Phase 3 and Phase 4 score is float32 arithmetic, and the
        # comparability this record claims with those tables depends on the two
        # running the same arithmetic rather than merely the same formula.
        # Without the cast this scored in whatever dtype the caller passed:
        # float64 under the suite, float32 in the shipped path, and fp16 if the
        # cache tensors were ever handed over directly, where the normalizing
        # epsilon underflows and a prediction equal to the mean vector yields a
        # NaN the pre-normalization finiteness check does not catch.
        a_n, b_n = _unit(a.to(torch.float32)), _unit(b.to(torch.float32))
        cosine = (a_n * b_n).sum(dim=-1)
        l2 = (a_n - b_n).norm(dim=-1)
        return (
            _scored_mean(cosine, failures, MODEL_FAILURE_COSINE),
            _scored_mean(l2, failures, MODEL_FAILURE_L2),
        )

    cosine_raw, l2_raw = pair(None)
    cosine_centered, l2_centered = pair(center)
    return {
        "cosine_raw": cosine_raw,
        "cosine_centered": cosine_centered,
        "l2_raw": l2_raw,
        "l2_centered": l2_centered,
        "n_failures": int(failures.sum()),
    }


@dataclasses.dataclass
class PrimaryScores:
    """One pair's primary per-point record, on the CL-Transport support.

    Every method carries all four PROTOCOL 3.7 columns: raw and centered cosine
    with their L2 companions. The reported Phase 5 quantities are built on the
    cosines, exactly as Phase 4's are, and the L2 columns travel in the record
    so the shipped schema is complete against the frozen protocol and
    comparable with the Phase 3 and Phase 4 tables. The Mean-Feature floor
    carries its two raw columns, the only ones PROTOCOL 3.7 defines for it.
    """

    n_primary: int
    cl_raw: float
    cl_centered: float
    cl_l2_raw: float
    cl_l2_centered: float
    predict_raw: float
    predict_centered: float
    predict_l2_raw: float
    predict_l2_centered: float
    nowarp_raw: float
    nowarp_centered: float
    nowarp_l2_raw: float
    nowarp_l2_centered: float
    meanfeat_raw: float
    meanfeat_l2_raw: float
    n_predict_nonfinite: int
    support_mask: np.ndarray

    def as_fields(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        row.pop("support_mask")
        return row


def _prefixed(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    """Rename one method's metric columns onto its record prefix."""
    return {
        f"{prefix}_raw": metrics["cosine_raw"],
        f"{prefix}_centered": metrics["cosine_centered"],
        f"{prefix}_l2_raw": metrics["l2_raw"],
        f"{prefix}_l2_centered": metrics["l2_centered"],
    }


def _raw_prefixed(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    """Rename a raw-only floor's metric columns onto its record prefix.

    Mean-Feature predicts the centering vector, so PROTOCOL 3.7 defines it under
    raw cosine only. Its centered columns are not written at all. No
    epsilon-regularized zero vector can then stand in for a centered score.
    """
    return {
        f"{prefix}_raw": metrics["cosine_raw"],
        f"{prefix}_l2_raw": metrics["l2_raw"],
    }


def _mean_feature(target: Tensor, center: Tensor, prefix: str) -> dict[str, float]:
    """The Mean-Feature floor on one record's support, under its prefix.

    target: [N, C], the targets the record's other arms are scored against.
    center: [C], the frozen Phase 3 global mean. The floor predicts it at every
    one of the N targets, as score_landing_offset does.
    """
    meanfeat = center.expand(target.shape[0], -1)
    return _raw_prefixed(score_all_metrics(meanfeat, target, center), prefix)


_EMPTY_METRICS = {
    "cosine_raw": float("nan"), "cosine_centered": float("nan"),
    "l2_raw": float("nan"), "l2_centered": float("nan"), "n_failures": 0,
}


def primary_support(lift: ContextLiftMap, gt_evaluable: Tensor) -> Tensor:
    """V_P5_pp: landed and ground-truth evaluable. Fixed before any scoring.

    Both clauses come from the explicit comparator and ground truth. Nothing the
    predictor produces can widen or narrow this set, which is the property
    Stream V step 18 requires.
    """
    return lift.landed & gt_evaluable


def score_primary(
    lift: ContextLiftMap,
    support: Tensor,
    features_context: Tensor,
    features_target: Tensor,
    center: Tensor,
    predicted_target_grid: Tensor | None,
    target_grid_hw: tuple[int, int],
    patch_size: int = PATCH_SIZE,
) -> PrimaryScores:
    """Score the three primary methods and the Mean-Feature floor on one support.

    features_context, features_target: [C, Hp, Wp] frozen patch-grid features.
    predicted_target_grid: [N_target, C] the predictor's complete grid, or None
        to score only the explicit method and the floors.

    Every method is read at the same landing locations, so the comparison is
    between predictions of the same physical quantity at the same places.
    """
    chosen = torch.nonzero(support, as_tuple=False).reshape(-1)
    if chosen.numel() == 0:
        return PrimaryScores(
            n_primary=0,
            **_prefixed(_EMPTY_METRICS, "cl"),
            **_prefixed(_EMPTY_METRICS, "predict"),
            **_prefixed(_EMPTY_METRICS, "nowarp"),
            **_raw_prefixed(_EMPTY_METRICS, "meanfeat"),
            n_predict_nonfinite=0,
            support_mask=support.cpu().numpy().copy(),
        )

    uv_target = lift.uv_target[chosen]
    uv_context = lift.uv_context[chosen]

    # The quantity every method is predicting: the frozen target feature at the
    # landing location.
    target = sample_features_bilinear(features_target, uv_target, patch_size)

    # Context-Lift Transport-Only: the context patch's own frozen feature,
    # carried to where the geometry says that patch lands.
    cl = sample_features_bilinear(features_context, uv_context, patch_size)

    # No-Warp-Copy: the context map read at the landing coordinates, which is
    # the prediction that assumes the transformation is the identity.
    nowarp = sample_features_bilinear(features_context, uv_target, patch_size)

    cl_metrics = score_all_metrics(cl, target, center)
    nowarp_metrics = score_all_metrics(nowarp, target, center)

    if predicted_target_grid is None:
        predict_metrics = dict(_EMPTY_METRICS)
    else:
        grid_h, grid_w = target_grid_hw
        channels = predicted_target_grid.shape[-1]
        maps = predicted_target_grid.T.reshape(channels, grid_h, grid_w)
        patch_coords = pixel_to_patch_coords(uv_target, patch_size)
        predicted = sample_map_bilinear(maps, patch_coords)
        # The network predicts centered features, so the raw comparison adds the
        # frozen mean back rather than comparing objects of different kinds.
        predict_metrics = score_all_metrics(predicted + center, target, center)

    return PrimaryScores(
        n_primary=int(chosen.numel()),
        **_prefixed(cl_metrics, "cl"),
        **_prefixed(predict_metrics, "predict"),
        **_prefixed(nowarp_metrics, "nowarp"),
        # Mean-Feature reads no location. It meets the same landing targets.
        **_mean_feature(target, center, "meanfeat"),
        n_predict_nonfinite=predict_metrics["n_failures"],
        support_mask=support.cpu().numpy().copy(),
    )


def formulation_support(
    lift: ContextLiftMap,
    primary: Tensor,
    target_lift_cells: np.ndarray,
    target_hw: tuple[int, int],
    patch_size: int = PATCH_SIZE,
) -> Tensor:
    """V_form: the primary support restricted to cells the target-lift set covers.

    The two estimators index different things. A context-lift sample is a context
    patch center that lands somewhere in the target image; a target-lift sample
    is a target patch center. The one index they share is the target patch cell,
    so the intersection is taken there. This population exists only for the
    formulation diagnostic and never redefines the headline support, which is
    why it is computed separately rather than by narrowing `primary` in place.

    target_lift_cells: row-major indices of the target cells the accepted Phase 4
        per-point set scored for this pair.
    """
    cells = patch_cell_index(lift.uv_target, target_hw, patch_size)
    covered = np.isin(cells, np.asarray(target_lift_cells, dtype=np.int64))
    return primary & torch.from_numpy(covered).to(primary.device)


@dataclasses.dataclass
class FormulationScores:
    """Target-lift against context-lift on their common cells. Diagnostic only.

    The unit is the target patch cell, on both arms, with one weight per cell,
    and both arms are scored against the same target: the cell's own feature at
    its centre. That is the only arrangement under which this diagnostic
    measures a difference of formulation and nothing else.

    TL-Reference is Phase 4's accepted per-point estimator, and Phase 4 defined
    it at Phase 3's target patch centres, one sample per cell. It cannot be
    moved to Context-Lift's landing locations without becoming a different
    estimator. So Context-Lift comes to it: the supported context patches
    landing in a cell are pooled to one vector, the output-level rule PROTOCOL
    3.7 already applies to pooled outputs, and scored against the same cell
    target TL-Reference is scored against.

    An earlier version scored per Context-Lift sample at the landing location
    and handed TL-Reference's cell-centre prediction to each of them. That
    compared TL against a target it never predicted, which penalized it by an
    interpolation residual unrelated to its formulation, and it counted TL once
    per landing sample, so cells with more collisions weighed more. It was never
    exercised, because nothing produced TL predictions until evaluate existed.

    The common cells and the per-cell sample counts travel with the record so
    the support an aggregate rests on can be reconstructed.
    """

    n_formulation: int
    tl_form_raw: float
    tl_form_centered: float
    tl_form_l2_raw: float
    tl_form_l2_centered: float
    cl_form_raw: float
    cl_form_centered: float
    cl_form_l2_raw: float
    cl_form_l2_centered: float
    common_cells: np.ndarray
    samples_per_cell: np.ndarray

    def as_fields(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        row.pop("common_cells")
        row.pop("samples_per_cell")
        return row


def formulation_cells(
    lift: ContextLiftMap,
    primary: Tensor,
    pp_scored: np.ndarray,
    target_hw: tuple[int, int],
    patch_size: int = PATCH_SIZE,
) -> tuple[np.ndarray, Tensor]:
    """V_form as cells: TL-Reference scored them and Context-Lift landed in them.

    pp_scored: [cells] bool, the cells the accepted Phase 4 per-point estimator
        scored at the reference level for this pair.

    Returns (common_cells, cl_mask), where cl_mask marks the supported
    Context-Lift samples that land in a common cell. The headline support is
    not narrowed in place; this population exists only for the diagnostic.
    """
    cells = patch_cell_index(lift.uv_target, target_hw, patch_size)
    supported = primary.cpu().numpy()
    landed_cells = np.unique(cells[supported]) if supported.any() else np.zeros(0, np.int64)
    common = np.intersect1d(landed_cells, np.flatnonzero(pp_scored))
    inside = torch.from_numpy(np.isin(cells, common)).to(primary.device)
    return common, primary & inside


def score_formulation(
    lift: ContextLiftMap,
    primary: Tensor,
    features_context: Tensor,
    center: Tensor,
    pp_scored: np.ndarray,
    per_point_cells: np.ndarray,
    tl_reads: Tensor,
    reads_target: Tensor,
    target_hw: tuple[int, int],
    patch_size: int = PATCH_SIZE,
) -> FormulationScores:
    """Score both explicit formulations on the common cells, one weight each.

    pp_scored, per_point_cells, tl_reads, reads_target: Phase 4's per-point
        quantities at the reference level, from lot.phase5_reference, which
        reconciles them against the accepted Phase 4 rows before they get here.
        tl_reads[k] and reads_target[k] are TL-Reference's prediction and the
        target for the Phase 3 sample at cell per_point_cells[k].
    """
    common, cl_mask = formulation_cells(lift, primary, pp_scored, target_hw, patch_size)
    if common.size == 0:
        return FormulationScores(
            n_formulation=0,
            **_prefixed(_EMPTY_METRICS, "tl_form"),
            **_prefixed(_EMPTY_METRICS, "cl_form"),
            common_cells=common,
            samples_per_cell=np.zeros(0, dtype=np.int64),
        )

    # One Phase 3 sample per cell: the lookup is a bijection on its domain, and a
    # duplicate would mean the universe is not what this diagnostic assumes.
    sample_of_cell: dict[int, int] = {}
    for k, cell in enumerate(np.asarray(per_point_cells, dtype=np.int64)):
        if int(cell) in sample_of_cell:
            raise ValueError(
                f"cell {int(cell)} holds more than one Phase 3 per-point sample; "
                "the formulation diagnostic assumes one sample per cell"
            )
        sample_of_cell[int(cell)] = k
    index = torch.as_tensor([sample_of_cell[int(c)] for c in common], dtype=torch.long)
    target_cells = reads_target[index]
    tl_cells = tl_reads[index]

    chosen = torch.nonzero(cl_mask, as_tuple=False).reshape(-1)
    landed = patch_cell_index(lift.uv_target[chosen], target_hw, patch_size)
    cl_samples = sample_features_bilinear(features_context, lift.uv_context[chosen], patch_size)
    cl_cells, samples_per_cell = pool_per_point_to_cells(cl_samples, landed, common)

    return FormulationScores(
        n_formulation=int(common.size),
        **_prefixed(score_all_metrics(tl_cells, target_cells, center), "tl_form"),
        **_prefixed(score_all_metrics(cl_cells, target_cells, center), "cl_form"),
        common_cells=common,
        samples_per_cell=samples_per_cell,
    )


@dataclasses.dataclass
class SplatScores:
    """The secondary operational comparison on Phase 4's scored-cell support.

    The Mean-Feature floor carries its two raw columns only, per PROTOCOL 3.7.
    """

    n_splat: int
    sp_transport_raw: float
    sp_transport_centered: float
    sp_transport_l2_raw: float
    sp_transport_l2_centered: float
    sp_predict_raw: float
    sp_predict_centered: float
    sp_predict_l2_raw: float
    sp_predict_l2_centered: float
    sp_nowarp_raw: float
    sp_nowarp_centered: float
    sp_nowarp_l2_raw: float
    sp_nowarp_l2_centered: float
    sp_meanfeat_raw: float
    sp_meanfeat_l2_raw: float

    def as_fields(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def score_splat_pool(
    scored_cells: np.ndarray,
    transported_est: Tensor,
    features_context_flat: Tensor,
    features_target_flat: Tensor,
    center: Tensor,
    predicted_target_grid: Tensor | None,
) -> SplatScores:
    """Score the operational path on Phase 4's fixed context-image-scale cells.

    transported_est, features_context_flat, features_target_flat: [C, n_cells]
        laid out row major over the target patch grid, matching Phase 4.
    predicted_target_grid: [n_cells, C] centered predictions, or None.

    The explicit arm here is Phase 4's own splat-and-pool output, which was
    verified to use context-side geometry before the design was frozen, so this
    comparison is information symmetric without a new comparator.
    """
    cells = np.asarray(scored_cells, dtype=np.int64)
    if cells.size == 0:
        return SplatScores(
            n_splat=0,
            **_prefixed(_EMPTY_METRICS, "sp_transport"),
            **_prefixed(_EMPTY_METRICS, "sp_predict"),
            **_prefixed(_EMPTY_METRICS, "sp_nowarp"),
            **_raw_prefixed(_EMPTY_METRICS, "sp_meanfeat"),
        )
    index = torch.from_numpy(cells)
    target = features_target_flat[:, index].T
    transport = transported_est[:, index].T
    nowarp = features_context_flat[:, index].T

    if predicted_target_grid is None:
        predict_metrics = dict(_EMPTY_METRICS)
    else:
        predict_metrics = score_all_metrics(
            predicted_target_grid[index] + center, target, center
        )
    return SplatScores(
        n_splat=int(cells.size),
        **_prefixed(score_all_metrics(transport, target, center), "sp_transport"),
        **_prefixed(predict_metrics, "sp_predict"),
        **_prefixed(score_all_metrics(nowarp, target, center), "sp_nowarp"),
        **_mean_feature(target, center, "sp_meanfeat"),
    )


def region_masks(
    lift: ContextLiftMap,
    support: Tensor,
    boundary_cells: np.ndarray,
    lowtex_cells: np.ndarray,
    target_hw: tuple[int, int],
    patch_size: int = PATCH_SIZE,
) -> dict[str, Tensor]:
    """Stream Z: split the primary support by Phase 4's frozen error regions.

    The masks are Phase 4's, computed from ground truth and rendered RGB alone
    under Amendment A5, read at the target cell each sample lands in. Estimated
    depth never defines a category, per PROTOCOL 4.9.
    """
    cells = patch_cell_index(lift.uv_target, target_hw, patch_size)
    boundary = torch.from_numpy(np.isin(cells, np.asarray(boundary_cells, dtype=np.int64)))
    lowtex = torch.from_numpy(np.isin(cells, np.asarray(lowtex_cells, dtype=np.int64)))
    boundary = boundary.to(support.device)
    lowtex = lowtex.to(support.device)
    return {
        "boundary": support & boundary,
        "interior": support & ~boundary,
        "low_texture": support & lowtex,
        "high_texture": support & ~lowtex,
    }


# ---------------------------------------------------------------------------
# The cross-path common-valid set, for PROTOCOL 3.9's two-path disclosure
# ---------------------------------------------------------------------------

# The arms that carry all four PROTOCOL 3.7 columns. The Mean-Feature floor on
# each path carries its raw columns only and is written beside them.
CROSS_PATH_ARMS = ("cl", "predict", "nowarp", "sp_transport", "sp_predict", "sp_nowarp")


def _empty_arm(prefix: str) -> dict[str, float]:
    return _prefixed(_EMPTY_METRICS, prefix)


@dataclasses.dataclass
class CrossPathScores:
    """Both paths recomputed on the cells they share, for the 3.9 disclosure.

    PROTOCOL 3.9 does not permit the path comparison to be a subtraction of two
    numbers computed on different populations: "Every term of such a quantity is
    recomputed on the cross-path common-valid cell set before differencing, and
    its uncertainty comes from a paired scene bootstrap in which one draw of
    scenes serves both paths and the difference is recomputed inside each
    replicate."

    The atomic unit is the target patch cell, on both arms, with one weight per
    cell. That is the whole point of this record and it was the defect in its
    first version: the per-point arm scored every context sample landing in a
    common cell, so a cell that received two samples counted twice on one path
    and once on the other, and collision multiplicity alone could manufacture a
    path difference between operators that agreed at every cell. Now the
    per-point predictions and targets are pooled to one normalized value per
    cell before scoring, which is the same output-level rule the splat contract
    already applies, and both arms then average over the same cells.

    Every arm carries all four PROTOCOL 3.7 columns, except Mean-Feature. That
    floor carries its two raw columns, the only ones PROTOCOL 3.7 defines for
    it. The two paths score their cells against different targets, so each path
    carries its own Mean-Feature floor, as each carries its own No-Warp-Copy.
    The common cells and the per-point samples that fed them travel with the
    record, so the support an aggregate rests on can be reconstructed and
    audited rather than inferred.
    """

    n_intersect: int
    # Explicit and learned, per-point arm, pooled to cells.
    x_cl_raw: float
    x_cl_centered: float
    x_cl_l2_raw: float
    x_cl_l2_centered: float
    x_predict_raw: float
    x_predict_centered: float
    x_predict_l2_raw: float
    x_predict_l2_centered: float
    x_nowarp_raw: float
    x_nowarp_centered: float
    x_nowarp_l2_raw: float
    x_nowarp_l2_centered: float
    x_meanfeat_raw: float
    x_meanfeat_l2_raw: float
    # Explicit and learned, splat arm, on the same cells.
    x_sp_transport_raw: float
    x_sp_transport_centered: float
    x_sp_transport_l2_raw: float
    x_sp_transport_l2_centered: float
    x_sp_predict_raw: float
    x_sp_predict_centered: float
    x_sp_predict_l2_raw: float
    x_sp_predict_l2_centered: float
    x_sp_nowarp_raw: float
    x_sp_nowarp_centered: float
    x_sp_nowarp_l2_raw: float
    x_sp_nowarp_l2_centered: float
    x_sp_meanfeat_raw: float
    x_sp_meanfeat_l2_raw: float
    # The support itself. Excluded from the aggregated fields, carried so an
    # aggregate can prove what it rests on.
    common_cells: np.ndarray
    per_point_mask: np.ndarray
    samples_per_cell: np.ndarray

    def as_fields(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        for key in ("common_cells", "per_point_mask", "samples_per_cell"):
            row.pop(key)
        return row


def cross_path_cells(
    lift: ContextLiftMap,
    primary: Tensor,
    splat_scored_cells: np.ndarray,
    target_hw: tuple[int, int],
    patch_size: int = PATCH_SIZE,
) -> tuple[np.ndarray, Tensor]:
    """The target cells both paths score, and the per-point samples inside them.

    Returns (common_cells, per_point_mask). A cell qualifies when the splat path
    scored it and at least one supported per-point sample landed in it.
    """
    cells = patch_cell_index(lift.uv_target, target_hw, patch_size)
    supported = primary.cpu().numpy()
    landed_cells = np.unique(cells[supported]) if supported.any() else np.zeros(0, np.int64)
    common = np.intersect1d(
        landed_cells, np.asarray(splat_scored_cells, dtype=np.int64), assume_unique=False
    )
    inside = torch.from_numpy(np.isin(cells, common)).to(primary.device)
    return common, primary & inside


def pool_per_point_to_cells(
    values: Tensor, cells: np.ndarray, common: np.ndarray
) -> tuple[Tensor, np.ndarray]:
    """One value per common cell: the mean of the per-point values landing in it.

    values: [M, C] per-point vectors, cells: [M] the cell each landed in, common:
    the common cells, sorted and unique. Returns ([n_common, C], samples_per_cell).

    Pooling happens at the vector level and the cosine is taken afterwards,
    which is PROTOCOL 3.7's output-level rule for pooled outputs. Averaging the
    per-sample cosines instead would be a different, un-frozen estimator.

    Vectorized with one index_add rather than a loop over samples. Evaluation
    calls this for every arm, region, and seed of every test pair, and the loop
    form cost tens of milliseconds a call. Sums accumulate in a different order
    from a sequential loop, so pooled vectors agree with it to floating-point
    rounding rather than bit for bit.
    """
    common = np.asarray(common, dtype=np.int64)
    cells = np.asarray(cells, dtype=np.int64)
    channels = values.shape[-1]
    n_common = int(common.size)
    if n_common == 0:
        return values.new_zeros(0, channels), np.zeros(0, dtype=np.int64)
    if n_common > 1 and not bool(np.all(common[1:] > common[:-1])):
        raise ValueError("common cells must be sorted and unique")
    slot = np.clip(np.searchsorted(common, cells), 0, n_common - 1)
    inside = np.flatnonzero(common[slot] == cells)
    kept_slots = slot[inside]
    pooled = values.new_zeros(n_common, channels)
    pooled.index_add_(
        0,
        torch.from_numpy(kept_slots).to(values.device),
        values[torch.from_numpy(inside).to(values.device)],
    )
    counts = np.bincount(kept_slots, minlength=n_common).astype(np.int64)
    weights = torch.from_numpy(np.maximum(counts, 1)).to(pooled.dtype).to(pooled.device)
    return pooled / weights[:, None], counts


def score_cross_path(
    lift: ContextLiftMap,
    primary: Tensor,
    splat_scored_cells: np.ndarray,
    features_context: Tensor,
    features_target: Tensor,
    center: Tensor,
    predicted_target_grid: Tensor | None,
    transported_est: Tensor,
    features_context_flat: Tensor,
    features_target_flat: Tensor,
    target_hw: tuple[int, int],
    target_grid_hw: tuple[int, int],
    patch_size: int = PATCH_SIZE,
) -> CrossPathScores:
    """Recompute both paths on their common cells, one weight per cell.

    Per-point arm: for every common cell, the supported samples landing in it
    have their predictions and their targets pooled to one vector each, and the
    cell is scored once. Splat arm: the cell is scored once on the values the
    accepted Phase 4 splat produced. Both arms therefore average over the same
    cells with the same weights, and n_intersect is the count of those cells on
    both sides.

    Nothing here re-decides either path's own support, which stays exactly as
    the headline and operational tables use it.
    """
    common, pp_mask = cross_path_cells(
        lift, primary, splat_scored_cells, target_hw, patch_size
    )
    if common.size == 0 or not bool(pp_mask.any()):
        return CrossPathScores(
            n_intersect=0,
            **{f"x_{k}": v for arm in CROSS_PATH_ARMS for k, v in _empty_arm(arm).items()},
            **_raw_prefixed(_EMPTY_METRICS, "x_meanfeat"),
            **_raw_prefixed(_EMPTY_METRICS, "x_sp_meanfeat"),
            common_cells=common,
            per_point_mask=pp_mask.cpu().numpy().copy(),
            samples_per_cell=np.zeros(0, dtype=np.int64),
        )

    chosen = torch.nonzero(pp_mask, as_tuple=False).reshape(-1)
    uv_target = lift.uv_target[chosen]
    uv_context = lift.uv_context[chosen]
    landed_cells = patch_cell_index(uv_target, target_hw, patch_size)

    # Per-point vectors at the landing locations, then pooled to cells.
    target_pp = sample_features_bilinear(features_target, uv_target, patch_size)
    cl_pp = sample_features_bilinear(features_context, uv_context, patch_size)
    nowarp_pp = sample_features_bilinear(features_context, uv_target, patch_size)
    target_cells, samples_per_cell = pool_per_point_to_cells(target_pp, landed_cells, common)
    cl_cells, _ = pool_per_point_to_cells(cl_pp, landed_cells, common)
    nowarp_cells, _ = pool_per_point_to_cells(nowarp_pp, landed_cells, common)

    if predicted_target_grid is None:
        predict_pp_metrics = dict(_EMPTY_METRICS)
        predict_sp_metrics = dict(_EMPTY_METRICS)
    else:
        grid_h, grid_w = target_grid_hw
        channels = predicted_target_grid.shape[-1]
        maps = predicted_target_grid.T.reshape(channels, grid_h, grid_w)
        predicted_pp = sample_map_bilinear(
            maps, pixel_to_patch_coords(uv_target, patch_size)
        ) + center
        predicted_cells, _ = pool_per_point_to_cells(predicted_pp, landed_cells, common)
        predict_pp_metrics = score_all_metrics(predicted_cells, target_cells, center)
        index = torch.from_numpy(common)
        predict_sp_metrics = score_all_metrics(
            predicted_target_grid[index] + center, features_target_flat[:, index].T, center
        )

    # Splat arm on exactly the same cells.
    index = torch.from_numpy(common)
    target_sp = features_target_flat[:, index].T
    return CrossPathScores(
        n_intersect=int(common.size),
        **_prefixed(score_all_metrics(cl_cells, target_cells, center), "x_cl"),
        **_prefixed(predict_pp_metrics, "x_predict"),
        **_prefixed(score_all_metrics(nowarp_cells, target_cells, center), "x_nowarp"),
        # Every sample predicts the mean, so its pooled prediction is the mean.
        **_mean_feature(target_cells, center, "x_meanfeat"),
        **_prefixed(
            score_all_metrics(transported_est[:, index].T, target_sp, center),
            "x_sp_transport",
        ),
        **_prefixed(predict_sp_metrics, "x_sp_predict"),
        **_prefixed(
            score_all_metrics(features_context_flat[:, index].T, target_sp, center),
            "x_sp_nowarp",
        ),
        **_mean_feature(target_sp, center, "x_sp_meanfeat"),
        common_cells=common,
        per_point_mask=pp_mask.cpu().numpy().copy(),
        samples_per_cell=samples_per_cell,
    )


# ---------------------------------------------------------------------------
# The landing-offset diagnostic, pre-registered 2026-10-09
# ---------------------------------------------------------------------------
#
# validation/evidence/phase5/landing_offset_diagnostic.md defines it. Under the
# frozen per-point read, a predictor's grid is read with the target's own
# bilinear weights, so its ceiling is one at every landing. Context-Lift carries
# one patch vector, which an off-grid read of the target grid cannot reproduce.
# Context-Lift with ground-truth context depth, grouped by how far each landing
# falls from the nearest target patch center, sizes that read on real features:
# exact geometry has no dependence on the offset, and the read does. Model free,
# never an estimand, never subtracted from the headline.


def landing_offsets(uv_target: Tensor, patch_size: int = PATCH_SIZE) -> Tensor:
    """[N] float64 distance, in patch units, from each landing to the nearest patch center.

    uv_target: [N, 2] pixel coordinates in the target image, OpenCV convention.
    The patch grid is the frozen one, through lot.encoders.pixel_to_patch_coords:
    a patch center has integer patch coordinates and offset zero, and the
    largest offset, at a cell corner, is sqrt(2) / 2. Computed in float64 from
    the run's landings, so a bin assignment does not rest on float32 rounding
    of the subtraction.
    """
    patch = pixel_to_patch_coords(uv_target.to(torch.float64), patch_size)
    return (patch - torch.round(patch)).norm(dim=-1)


def offset_bins(offsets: Any, upper_edges: Any) -> np.ndarray:
    """The bin of each offset, closed on the right as PROTOCOL 3.4 fixes for bins.

    Bin 0 holds offsets up to and including upper_edges[0]; bin k holds
    upper_edges[k - 1] < offset <= upper_edges[k]; the last bin holds every
    offset above the last edge.
    """
    edges = np.asarray(upper_edges, dtype=np.float64)
    return np.searchsorted(edges, np.asarray(offsets, dtype=np.float64), side="left")


@dataclasses.dataclass
class LandingOffsetScores:
    """One pair's diagnostic columns, named by lot.phase5_estimands."""

    fields: dict[str, Any]

    def as_fields(self) -> dict[str, Any]:
        from .phase5_estimands import landing_offset_record_fields

        return {name: self.fields[name] for name in landing_offset_record_fields()}


def empty_landing_offset() -> LandingOffsetScores:
    """The diagnostic's record where it is not computed: zero counts, no scores."""
    from .phase5_estimands import (
        OFFSET_BINS, OFFSET_WHOLE, offset_count_field, offset_fields,
    )

    fields: dict[str, Any] = {}
    for label in OFFSET_BINS + (OFFSET_WHOLE,):
        fields[offset_count_field(label)] = 0
        fields.update({name: float("nan") for name in offset_fields(label)})
    return LandingOffsetScores(fields)


def score_landing_offset(
    lift: ContextLiftMap,
    support: Tensor,
    features_context: Tensor,
    features_target: Tensor,
    center: Tensor,
    upper_edges: Any,
    patch_size: int = PATCH_SIZE,
) -> LandingOffsetScores:
    """Context-Lift and No-Warp-Copy, scored per offset bin against the landing read.

    lift: a context-lift map built from ground-truth context depth. support: its
    landed and ground-truth evaluable samples, the same rule as the primary
    support. Each sample is scored exactly as score_primary scores Context-Lift
    and the floor: the context patch's own vector, and the context map read at
    the landing, both against the target grid read at the landing. The samples
    are then grouped by landing offset, and the whole support is scored too.
    """
    from .phase5_estimands import (
        MEAN_FEATURE_COLUMNS, METRIC_COLUMNS, N_OFFSET_BINS, OFFSET_BINS,
        OFFSET_WHOLE, offset_count_field,
    )

    if len(upper_edges) + 1 != N_OFFSET_BINS:
        raise ValueError(
            f"{len(upper_edges)} edges make {len(upper_edges) + 1} bins, but the "
            f"pre-registered diagnostic reads {N_OFFSET_BINS} bins"
        )
    chosen = torch.nonzero(support, as_tuple=False).reshape(-1)
    if chosen.numel() == 0:
        return empty_landing_offset()

    uv_target = lift.uv_target[chosen]
    target = sample_features_bilinear(features_target, uv_target, patch_size)
    carried = sample_features_bilinear(features_context, lift.uv_context[chosen], patch_size)
    nowarp = sample_features_bilinear(features_context, uv_target, patch_size)
    bins = offset_bins(landing_offsets(uv_target, patch_size).cpu().numpy(), upper_edges)

    metric_key = {"raw": "cosine_raw", "centered": "cosine_centered",
                  "l2_raw": "l2_raw", "l2_centered": "l2_centered"}
    groups = [(label, np.flatnonzero(bins == k)) for k, label in enumerate(OFFSET_BINS)]
    groups.append((OFFSET_WHOLE, np.arange(chosen.numel())))
    fields: dict[str, Any] = {}
    for label, members in groups:
        index = torch.from_numpy(members).to(target.device)
        fields[offset_count_field(label)] = int(members.size)
        for arm, values in (("cl_oracle", carried), ("nowarp", nowarp)):
            metrics = score_all_metrics(values[index], target[index], center)
            for column in METRIC_COLUMNS:
                fields[f"offset_{arm}_{column}_{label}"] = metrics[metric_key[column]]
        # The Mean-Feature floor predicts the centering vector at every sample.
        # Only its raw columns are defined, so only they are read.
        meanfeat = center.expand(int(members.size), -1)
        metrics = score_all_metrics(meanfeat, target[index], center)
        for column in MEAN_FEATURE_COLUMNS:
            fields[f"offset_meanfeat_{column}_{label}"] = metrics[metric_key[column]]
    return LandingOffsetScores(fields)
