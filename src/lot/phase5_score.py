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


def _cosine(a: Tensor, b: Tensor) -> Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-12)


def _scored_mean(cosine: Tensor, failures: Tensor) -> float:
    """Mean cosine over a support, with model failures held at the worst value."""
    if cosine.numel() == 0:
        return float("nan")
    values = torch.where(failures, torch.full_like(cosine, MODEL_FAILURE_COSINE), cosine)
    return float(values.mean())


def score_predictions(
    prediction: Tensor, target: Tensor, center: Tensor
) -> tuple[float, float, int]:
    """Raw and centered mean cosine of one method against the target features.

    prediction, target: [N, C]. center: [C], the frozen Phase 3 global mean.

    Centering is applied at the output level, immediately before the cosine, per
    PROTOCOL 3.7. Nonfinite predictions are counted and scored as failures under
    both metrics rather than being excluded from either.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"prediction {tuple(prediction.shape)} != target {tuple(target.shape)}")
    if prediction.shape[0] == 0:
        return float("nan"), float("nan"), 0
    failures = ~torch.isfinite(prediction).all(dim=-1)
    safe = torch.where(failures[:, None], torch.zeros_like(prediction), prediction)
    raw = _scored_mean(_cosine(safe, target), failures)
    centered = _scored_mean(_cosine(safe - center, target - center), failures)
    return raw, centered, int(failures.sum())


@dataclasses.dataclass
class PrimaryScores:
    """One pair's primary per-point record, on the CL-Transport support."""

    n_primary: int
    cl_raw: float
    cl_centered: float
    predict_raw: float
    predict_centered: float
    nowarp_raw: float
    nowarp_centered: float
    n_predict_nonfinite: int
    support_mask: np.ndarray

    def as_fields(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        row.pop("support_mask")
        return row


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
        empty = float("nan")
        return PrimaryScores(0, empty, empty, empty, empty, empty, empty, 0,
                             support.cpu().numpy().copy())

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

    cl_raw, cl_centered, _ = score_predictions(cl, target, center)
    nowarp_raw, nowarp_centered, _ = score_predictions(nowarp, target, center)

    if predicted_target_grid is None:
        predict_raw = predict_centered = float("nan")
        n_failures = 0
    else:
        grid_h, grid_w = target_grid_hw
        channels = predicted_target_grid.shape[-1]
        maps = predicted_target_grid.T.reshape(channels, grid_h, grid_w)
        patch_coords = pixel_to_patch_coords(uv_target, patch_size)
        predicted = sample_map_bilinear(maps, patch_coords)
        # The network predicts centered features, so the raw comparison adds the
        # frozen mean back rather than comparing objects of different kinds.
        predict_raw, predict_centered, n_failures = score_predictions(
            predicted + center, target, center
        )

    return PrimaryScores(
        n_primary=int(chosen.numel()),
        cl_raw=cl_raw,
        cl_centered=cl_centered,
        predict_raw=predict_raw,
        predict_centered=predict_centered,
        nowarp_raw=nowarp_raw,
        nowarp_centered=nowarp_centered,
        n_predict_nonfinite=n_failures,
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
    cl_form_raw: float
    cl_form_centered: float

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
        nan = float("nan")
        return FormulationScores(0, nan, nan, nan, nan)
    uv_target = lift.uv_target[chosen]
    target = sample_features_bilinear(features_target, uv_target, patch_size)
    cl = sample_features_bilinear(features_context, lift.uv_context[chosen], patch_size)

    cl_raw, cl_centered, _ = score_predictions(cl, target, center)
    tl_raw, tl_centered, _ = score_predictions(target_lift_prediction, target, center)
    return FormulationScores(
        n_formulation=int(chosen.numel()),
        tl_form_raw=tl_raw,
        tl_form_centered=tl_centered,
        cl_form_raw=cl_raw,
        cl_form_centered=cl_centered,
    )


@dataclasses.dataclass
class SplatScores:
    """The secondary operational comparison on Phase 4's scored-cell support."""

    n_splat: int
    sp_transport_raw: float
    sp_transport_centered: float
    sp_predict_raw: float
    sp_predict_centered: float
    sp_nowarp_raw: float
    sp_nowarp_centered: float

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
        nan = float("nan")
        return SplatScores(0, nan, nan, nan, nan, nan, nan)
    index = torch.from_numpy(cells)
    target = features_target_flat[:, index].T
    transport = transported_est[:, index].T
    nowarp = features_context_flat[:, index].T

    t_raw, t_centered, _ = score_predictions(transport, target, center)
    n_raw, n_centered, _ = score_predictions(nowarp, target, center)
    if predicted_target_grid is None:
        p_raw = p_centered = float("nan")
    else:
        predicted = predicted_target_grid[index] + center
        p_raw, p_centered, _ = score_predictions(predicted, target, center)
    return SplatScores(
        n_splat=int(cells.size),
        sp_transport_raw=t_raw,
        sp_transport_centered=t_centered,
        sp_predict_raw=p_raw,
        sp_predict_centered=p_centered,
        sp_nowarp_raw=n_raw,
        sp_nowarp_centered=n_centered,
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
