"""Stream V: the fixed supports and the scoring, on analytic geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from lot.context_lift import context_lift_map, context_lift_support
from lot.encoders import PATCH_SIZE, patch_cell_index, sample_features_bilinear
from lot.phase5_score import (
    MODEL_FAILURE_COSINE,
    formulation_support,
    primary_support,
    region_masks,
    score_predictions,
    score_primary,
    score_splat_pool,
)

from scenes import (
    GRID,
    IMAGE_SIZE,
    build_two_plane_scene,
    intrinsics,
    make_pose,
)
from lot.geometry import relative_pose

HW = (IMAGE_SIZE, IMAGE_SIZE)
CHANNELS = 8


def _features(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(CHANNELS, GRID, GRID, generator=g, dtype=torch.float64)


def _center() -> torch.Tensor:
    g = torch.Generator().manual_seed(99)
    return torch.randn(CHANNELS, generator=g, dtype=torch.float64) * 0.1


# ---------------------------------------------------------------------------
# score_predictions
# ---------------------------------------------------------------------------

def test_perfect_prediction_scores_one_under_both_metrics():
    target = torch.randn(20, CHANNELS, dtype=torch.float64)
    center = _center()
    raw, centered, failures = score_predictions(target.clone(), target, center)
    assert raw == pytest.approx(1.0)
    assert centered == pytest.approx(1.0)
    assert failures == 0


def test_centering_changes_the_score():
    """Centering is not cosmetic: it removes the shared direction PROTOCOL 3.7
    says dominates a raw DINOv2 cosine."""
    g = torch.Generator().manual_seed(1)
    center = torch.ones(CHANNELS, dtype=torch.float64) * 5.0
    target = torch.randn(20, CHANNELS, generator=g, dtype=torch.float64) + center
    prediction = torch.randn(20, CHANNELS, generator=g, dtype=torch.float64) + center
    raw, centered, _ = score_predictions(prediction, target, center)
    assert raw > centered, (raw, centered)


def test_a_nonfinite_prediction_is_scored_as_failure_not_dropped():
    """A model may not improve its score by declining to answer."""
    target = torch.randn(4, CHANNELS, dtype=torch.float64)
    center = torch.zeros(CHANNELS, dtype=torch.float64)
    prediction = target.clone()
    prediction[2] = float("nan")
    raw, centered, failures = score_predictions(prediction, target, center)
    assert failures == 1
    # Three perfect samples and one at the failure value.
    assert raw == pytest.approx((3 * 1.0 + MODEL_FAILURE_COSINE) / 4)
    assert centered == pytest.approx((3 * 1.0 + MODEL_FAILURE_COSINE) / 4)


def test_dropping_the_failure_would_have_scored_higher():
    """Names the incentive the failure rule exists to remove."""
    target = torch.randn(4, CHANNELS, dtype=torch.float64)
    center = torch.zeros(CHANNELS, dtype=torch.float64)
    prediction = target.clone()
    prediction[2] = float("inf")
    scored, _, _ = score_predictions(prediction, target, center)
    dropped, _, _ = score_predictions(
        prediction[[0, 1, 3]], target[[0, 1, 3]], center
    )
    assert dropped > scored


def test_empty_support_scores_nan_rather_than_zero():
    center = torch.zeros(CHANNELS, dtype=torch.float64)
    raw, centered, failures = score_predictions(
        torch.zeros(0, CHANNELS, dtype=torch.float64),
        torch.zeros(0, CHANNELS, dtype=torch.float64), center,
    )
    assert math.isnan(raw) and math.isnan(centered) and failures == 0


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="!="):
        score_predictions(torch.zeros(3, 4), torch.zeros(2, 4), torch.zeros(4))


# ---------------------------------------------------------------------------
# The primary support is fixed by the explicit comparator
# ---------------------------------------------------------------------------

def _two_plane_setup():
    scene = build_two_plane_scene()
    lift = context_lift_map(
        scene.depth_context, scene.K, scene.K, scene.T_target_from_context, HW, HW
    )
    evaluable = context_lift_support(
        lift, scene.depth_context, scene.depth_target, scene.K, scene.K,
        scene.T_target_from_context,
    )
    return scene, lift, primary_support(lift, evaluable)


def test_primary_support_is_landed_and_evaluable():
    scene, lift, support = _two_plane_setup()
    assert bool((support <= lift.landed).all())
    assert 0 < int(support.sum()) < support.numel()


def test_the_predictor_cannot_change_the_support():
    """Stream V step 18: support is fixed before the predictor is looked at."""
    scene, lift, support = _two_plane_setup()
    center = _center()
    fc, ft = _features(1), _features(2)
    n_target = GRID * GRID

    good = torch.randn(n_target, CHANNELS, dtype=torch.float64)
    wrecked = good.clone()
    wrecked[:] = float("nan")

    a = score_primary(lift, support, fc, ft, center, good, (GRID, GRID))
    b = score_primary(lift, support, fc, ft, center, wrecked, (GRID, GRID))
    assert a.n_primary == b.n_primary
    assert np.array_equal(a.support_mask, b.support_mask)
    # The explicit method and the floor are untouched by the predictor's output.
    assert a.cl_centered == pytest.approx(b.cl_centered)
    assert a.nowarp_centered == pytest.approx(b.nowarp_centered)
    # And the all-failing predictor is scored at the failure value, not dropped.
    assert b.n_predict_nonfinite == b.n_primary
    assert b.predict_centered == pytest.approx(MODEL_FAILURE_COSINE)


def test_context_lift_scores_perfectly_when_the_target_features_are_transported():
    """Sanity on the geometry: if the target grid really is the transported
    context grid, the explicit comparator recovers it and the floor does not."""
    scene, lift, support = _two_plane_setup()
    center = torch.zeros(CHANNELS, dtype=torch.float64)
    fc = _features(7)

    # Build a target feature map that is, by construction, the context feature
    # carried along the analytic correspondence.
    ft = torch.zeros_like(fc)
    filled = torch.zeros(GRID, GRID, dtype=torch.bool)
    chosen = torch.nonzero(support, as_tuple=False).reshape(-1)
    cells = patch_cell_index(lift.uv_target[chosen], HW, PATCH_SIZE)
    values = sample_features_bilinear(fc, lift.uv_context[chosen], PATCH_SIZE)
    for i, cell in enumerate(cells):
        r, c = divmod(int(cell), GRID)
        ft[:, r, c] = values[i]
        filled[r, c] = True

    scores = score_primary(lift, support, fc, ft, center, None, (GRID, GRID))
    assert scores.cl_raw > 0.9, scores.cl_raw
    assert scores.cl_raw > scores.nowarp_raw


def test_no_warp_floor_reads_the_context_map_at_the_landing():
    """The floor must be the identity-transformation prediction, not the
    transported one, or the margin would be zero by construction."""
    scene, lift, support = _two_plane_setup()
    center = torch.zeros(CHANNELS, dtype=torch.float64)
    fc = _features(3)
    ft = _features(4)
    scores = score_primary(lift, support, fc, ft, center, None, (GRID, GRID))

    chosen = torch.nonzero(support, as_tuple=False).reshape(-1)
    expected = sample_features_bilinear(fc, lift.uv_target[chosen], PATCH_SIZE)
    target = sample_features_bilinear(ft, lift.uv_target[chosen], PATCH_SIZE)
    manual, _, _ = score_predictions(expected, target, center)
    assert scores.nowarp_raw == pytest.approx(manual)


def test_empty_support_yields_an_empty_record():
    scene, lift, _ = _two_plane_setup()
    empty = torch.zeros_like(lift.landed)
    scores = score_primary(lift, empty, _features(1), _features(2), _center(),
                           None, (GRID, GRID))
    assert scores.n_primary == 0
    assert math.isnan(scores.cl_raw)
    fields = scores.as_fields()
    assert "support_mask" not in fields


# ---------------------------------------------------------------------------
# The formulation support is separate and cannot widen the headline one
# ---------------------------------------------------------------------------

def test_formulation_support_narrows_and_never_widens():
    scene, lift, support = _two_plane_setup()
    cells = patch_cell_index(lift.uv_target, HW, PATCH_SIZE)
    covered = np.unique(cells)[: len(np.unique(cells)) // 2]
    form = formulation_support(lift, support, covered, HW)
    assert bool((form <= support).all())
    assert int(form.sum()) < int(support.sum())


def test_formulation_support_with_no_overlap_is_empty():
    scene, lift, support = _two_plane_setup()
    form = formulation_support(lift, support, np.array([], dtype=np.int64), HW)
    assert int(form.sum()) == 0
    # And the headline support is untouched by computing it.
    assert int(support.sum()) > 0


# ---------------------------------------------------------------------------
# The operational path
# ---------------------------------------------------------------------------

def test_splat_scoring_uses_exactly_the_given_cells_for_every_method():
    n_cells = GRID * GRID
    g = torch.Generator().manual_seed(5)
    transported = torch.randn(CHANNELS, n_cells, generator=g, dtype=torch.float64)
    context = torch.randn(CHANNELS, n_cells, generator=g, dtype=torch.float64)
    target = torch.randn(CHANNELS, n_cells, generator=g, dtype=torch.float64)
    predicted = torch.randn(n_cells, CHANNELS, generator=g, dtype=torch.float64)
    center = torch.zeros(CHANNELS, dtype=torch.float64)
    cells = np.array([0, 5, 11, 40], dtype=np.int64)

    scores = score_splat_pool(cells, transported, context, target, center, predicted)
    assert scores.n_splat == 4
    manual, _, _ = score_predictions(
        transported[:, torch.from_numpy(cells)].T,
        target[:, torch.from_numpy(cells)].T, center,
    )
    assert scores.sp_transport_raw == pytest.approx(manual)
    assert math.isfinite(scores.sp_predict_raw)
    assert math.isfinite(scores.sp_nowarp_raw)


def test_splat_scoring_without_a_predictor_reports_nan_for_it():
    n_cells = 16
    z = torch.zeros(CHANNELS, n_cells, dtype=torch.float64)
    scores = score_splat_pool(
        np.array([1, 2], dtype=np.int64), z + 1, z + 1, z + 1,
        torch.zeros(CHANNELS, dtype=torch.float64), None,
    )
    assert math.isnan(scores.sp_predict_raw)
    assert scores.n_splat == 2


def test_splat_scoring_on_no_cells():
    z = torch.zeros(CHANNELS, 4, dtype=torch.float64)
    scores = score_splat_pool(
        np.array([], dtype=np.int64), z, z, z,
        torch.zeros(CHANNELS, dtype=torch.float64), None,
    )
    assert scores.n_splat == 0
    assert math.isnan(scores.sp_transport_raw)


# ---------------------------------------------------------------------------
# Stream Z: the error regions partition the primary support
# ---------------------------------------------------------------------------

def test_region_masks_partition_the_support():
    scene, lift, support = _two_plane_setup()
    cells = np.unique(patch_cell_index(lift.uv_target, HW, PATCH_SIZE))
    boundary = cells[::3]
    lowtex = cells[1::4]
    masks = region_masks(lift, support, boundary, lowtex, HW)

    assert torch.equal(masks["boundary"] | masks["interior"], support)
    assert int((masks["boundary"] & masks["interior"]).sum()) == 0
    assert torch.equal(masks["low_texture"] | masks["high_texture"], support)
    assert int((masks["low_texture"] & masks["high_texture"]).sum()) == 0
    for mask in masks.values():
        assert bool((mask <= support).all())
