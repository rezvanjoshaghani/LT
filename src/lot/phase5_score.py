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
        a_n, b_n = _unit(a), _unit(b)
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
    comparable with the Phase 3 and Phase 4 tables.
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
    """Score the three primary methods on one fixed support.

    features_context, features_target: [C, Hp, Wp] frozen patch-grid features.
    predicted_target_grid: [N_target, C] the predictor's complete grid, or None
        to score only the explicit method and the floor.

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
    """Target-lift against context-lift on their common support. Diagnostic only."""

    n_formulation: int
    tl_form_raw: float
    tl_form_centered: float
    tl_form_l2_raw: float
    tl_form_l2_centered: float
    cl_form_raw: float
    cl_form_centered: float
    cl_form_l2_raw: float
    cl_form_l2_centered: float

    def as_fields(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def score_formulation(
    lift: ContextLiftMap,
    support: Tensor,
    features_context: Tensor,
    features_target: Tensor,
    center: Tensor,
    target_lift_prediction: Tensor,
    patch_size: int = PATCH_SIZE,
) -> FormulationScores:
    """Score both explicit formulations on the shared population.

    target_lift_prediction: [M, C] the accepted Phase 4 per-point predictions for
        the supported samples, supplied by the caller from the Phase 4 artifacts
        rather than recomputed here, so the diagnostic compares against what
        Phase 4 actually reported.
    """
    chosen = torch.nonzero(support, as_tuple=False).reshape(-1)
    if chosen.numel() == 0:
        return FormulationScores(
            n_formulation=0,
            **_prefixed(_EMPTY_METRICS, "tl_form"),
            **_prefixed(_EMPTY_METRICS, "cl_form"),
        )
    uv_target = lift.uv_target[chosen]
    target = sample_features_bilinear(features_target, uv_target, patch_size)
    cl = sample_features_bilinear(features_context, lift.uv_context[chosen], patch_size)

    return FormulationScores(
        n_formulation=int(chosen.numel()),
        **_prefixed(score_all_metrics(target_lift_prediction, target, center), "tl_form"),
        **_prefixed(score_all_metrics(cl, target, center), "cl_form"),
    )


@dataclasses.dataclass
class SplatScores:
    """The secondary operational comparison on Phase 4's scored-cell support."""

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

@dataclasses.dataclass
class CrossPathScores:
    """Both paths recomputed on the cells they share, for the 3.9 disclosure.

    PROTOCOL 3.9 does not permit the path comparison to be a subtraction of two
    numbers computed on different populations: "Every term of such a quantity is
    recomputed on the cross-path common-valid cell set before differencing, and
    its uncertainty comes from a paired scene bootstrap in which one draw of
    scenes serves both paths and the difference is recomputed inside each
    replicate."

    Comparing V_P5_pp against V_sp directly would mix an operator difference
    with a selection difference, and the selection difference can carry its own
    sign. So both paths are re-scored here on the target cells they share, and
    every column below travels in one record so a single bootstrap draw serves
    the pair.

    The shared index is the target patch cell, which is the only unit the two
    estimators have in common: a per-point sample is a context patch with a
    continuous landing, a splat unit is a target cell.
    """

    n_intersect: int
    x_cl_raw: float
    x_cl_centered: float
    x_predict_raw: float
    x_predict_centered: float
    x_sp_transport_raw: float
    x_sp_transport_centered: float
    x_sp_predict_raw: float
    x_sp_predict_centered: float

    def as_fields(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


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
    """Recompute both paths on their common cells, in one record.

    The per-point terms are scored at the landing locations of the supported
    samples that fall inside the common cells; the splat terms are scored on
    those same cells. Nothing here re-decides either path's own support, which
    stays exactly as the headline and operational tables use it.
    """
    common, pp_mask = cross_path_cells(
        lift, primary, splat_scored_cells, target_hw, patch_size
    )
    if common.size == 0 or not bool(pp_mask.any()):
        return CrossPathScores(
            n_intersect=0,
            **{f"x_{k}": float("nan") for k in (
                "cl_raw", "cl_centered", "predict_raw", "predict_centered",
                "sp_transport_raw", "sp_transport_centered",
                "sp_predict_raw", "sp_predict_centered",
            )},
        )

    per_point = score_primary(
        lift, pp_mask, features_context, features_target, center,
        predicted_target_grid, target_grid_hw, patch_size,
    )
    splat = score_splat_pool(
        common, transported_est, features_context_flat, features_target_flat,
        center, predicted_target_grid,
    )
    return CrossPathScores(
        n_intersect=int(common.size),
        x_cl_raw=per_point.cl_raw,
        x_cl_centered=per_point.cl_centered,
        x_predict_raw=per_point.predict_raw,
        x_predict_centered=per_point.predict_centered,
        x_sp_transport_raw=splat.sp_transport_raw,
        x_sp_transport_centered=splat.sp_transport_centered,
        x_sp_predict_raw=splat.sp_predict_raw,
        x_sp_predict_centered=splat.sp_predict_centered,
    )
