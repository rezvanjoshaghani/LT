"""Stream V: the fixed supports and the scoring, on analytic geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from lot.context_lift import ContextLiftMap, context_lift_map, context_lift_support
from lot.encoders import PATCH_SIZE, patch_cell_index, sample_features_bilinear
from lot.phase5_score import (
    MODEL_FAILURE_COSINE,
    formulation_support,
    primary_support,
    region_masks,
    score_cross_path,
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
    # The explicit method and both floors are untouched by the predictor's output.
    assert a.cl_centered == pytest.approx(b.cl_centered)
    assert a.nowarp_centered == pytest.approx(b.nowarp_centered)
    assert a.meanfeat_raw == pytest.approx(b.meanfeat_raw)
    assert a.meanfeat_l2_raw == pytest.approx(b.meanfeat_l2_raw)
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
    assert math.isnan(scores.meanfeat_raw) and math.isnan(scores.meanfeat_l2_raw)
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
    assert math.isnan(scores.sp_meanfeat_raw) and math.isnan(scores.sp_meanfeat_l2_raw)


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


# ---------------------------------------------------------------------------
# PROTOCOL 3.7 requires an L2 companion beside every cosine
# ---------------------------------------------------------------------------

def test_every_method_reports_all_four_protocol_columns():
    from lot.phase5_score import score_all_metrics

    target = torch.randn(20, CHANNELS, dtype=torch.float64)
    metrics = score_all_metrics(target.clone(), target, _center())
    assert set(metrics) == {
        "cosine_raw", "cosine_centered", "l2_raw", "l2_centered", "n_failures"
    }
    assert metrics["cosine_raw"] == pytest.approx(1.0)
    assert metrics["l2_raw"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["l2_centered"] == pytest.approx(0.0, abs=1e-6)


def test_l2_matches_the_frozen_phase3_implementation():
    """The companion must be the same quantity Phase 3 and Phase 4 report."""
    from lot.evaluate import value_agreement
    from lot.phase5_score import score_all_metrics

    g = torch.Generator().manual_seed(21)
    prediction = torch.randn(30, CHANNELS, generator=g, dtype=torch.float64)
    target = torch.randn(30, CHANNELS, generator=g, dtype=torch.float64)
    center = _center()

    mine = score_all_metrics(prediction, target, center)
    cosine_raw, l2_raw = value_agreement(prediction, target)
    cosine_centered, l2_centered = value_agreement(prediction, target, center=center)
    assert mine["cosine_raw"] == pytest.approx(cosine_raw, abs=1e-6)
    assert mine["l2_raw"] == pytest.approx(l2_raw, abs=1e-6)
    assert mine["cosine_centered"] == pytest.approx(cosine_centered, abs=1e-6)
    assert mine["l2_centered"] == pytest.approx(l2_centered, abs=1e-6)


def test_l2_scores_a_failure_at_the_worst_attainable_distance():
    from lot.phase5_score import MODEL_FAILURE_L2, score_all_metrics

    target = torch.randn(4, CHANNELS, dtype=torch.float64)
    prediction = target.clone()
    prediction[1] = float("nan")
    metrics = score_all_metrics(prediction, target, torch.zeros(CHANNELS, dtype=torch.float64))
    assert metrics["n_failures"] == 1
    assert metrics["l2_raw"] == pytest.approx((3 * 0.0 + MODEL_FAILURE_L2) / 4, abs=1e-6)


def test_the_primary_record_carries_l2_for_every_method():
    scene, lift, support = _two_plane_setup()
    n_target = GRID * GRID
    predicted = torch.randn(n_target, CHANNELS, dtype=torch.float64)
    fields = score_primary(
        lift, support, _features(1), _features(2), _center(), predicted, (GRID, GRID)
    ).as_fields()
    for method in ("cl", "predict", "nowarp"):
        for column in ("raw", "centered", "l2_raw", "l2_centered"):
            key = f"{method}_{column}"
            assert key in fields, key
            assert math.isfinite(fields[key]), key


def test_the_splat_and_formulation_records_carry_l2_too():
    from lot.phase5_score import SplatScores, score_splat_pool

    n_cells = 16
    g = torch.Generator().manual_seed(8)
    a = torch.randn(CHANNELS, n_cells, generator=g, dtype=torch.float64)
    b = torch.randn(CHANNELS, n_cells, generator=g, dtype=torch.float64)
    c = torch.randn(CHANNELS, n_cells, generator=g, dtype=torch.float64)
    predicted = torch.randn(n_cells, CHANNELS, generator=g, dtype=torch.float64)
    fields = score_splat_pool(
        np.array([0, 3, 9], dtype=np.int64), a, b, c,
        torch.zeros(CHANNELS, dtype=torch.float64), predicted,
    ).as_fields()
    for method in ("sp_transport", "sp_predict", "sp_nowarp"):
        for column in ("raw", "centered", "l2_raw", "l2_centered"):
            assert f"{method}_{column}" in fields


# ---------------------------------------------------------------------------
# The Mean-Feature floor on each record's own support, per PROTOCOL 3.7
# ---------------------------------------------------------------------------
#
# Mean-Feature predicts the frozen mean vector everywhere. Its raw cosine with a
# target depends only on the angle between the two, which a hand-built grid
# makes exact. The mean lies along the fourth axis. The 2 by 2 target cells
# hold E0 + M, M, E1, and -M, whose raw cosines with the mean are 1/sqrt(2),
# 1, 0, and -1.

E0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
E1 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
MEAN = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
FLOOR_HW = (28, 28)
FLOOR_GRID = (2, 2)
SQRT2 = math.sqrt(2.0)


def _pixel(patch_x: float, patch_y: float) -> list[float]:
    """Pixel coordinates of a patch-grid location: patch p sits at pixel 14 p + 6.5."""
    return [14 * patch_x + 6.5, 14 * patch_y + 6.5]


def _floor_scene():
    """Three samples, A, B, and C, on the hand-built target grid.

    A lands a quarter patch right of cell 0's center, so the frozen target there
    is the read 0.75 (E0 + M) + 0.25 M = 0.75 E0 + M. Its raw cosine with the
    mean is 1 / 1.25 = 0.8, where cell 0's own vector would give 1/sqrt(2).
    B lands on the center of cell 2, which holds E1. C lands on the center of
    cell 1, which holds M. The context grid never enters the floor.
    """
    ft = torch.zeros(4, 2, 2, dtype=torch.float64)
    ft[:, 0, 0] = E0 + MEAN
    ft[:, 0, 1] = MEAN
    ft[:, 1, 0] = E1
    ft[:, 1, 1] = -MEAN
    fc = torch.ones_like(ft)
    lift = ContextLiftMap(
        uv_context=torch.tensor([_pixel(0, 0), _pixel(1, 0), _pixel(0, 1)],
                                dtype=torch.float64),
        uv_target=torch.tensor([_pixel(0.25, 0), _pixel(0, 1), _pixel(1, 0)],
                               dtype=torch.float64),
        z_target=torch.ones(3, dtype=torch.float64),
        depth_context=torch.ones(3, dtype=torch.float64),
        depth_valid=torch.ones(3, dtype=torch.bool),
        landed=torch.ones(3, dtype=torch.bool),
    )
    return lift, fc, ft


def _cross_path(lift, support, scored_cells, fc, ft):
    flat_c, flat_t = fc.reshape(4, -1), ft.reshape(4, -1)
    return score_cross_path(
        lift, support, np.array(scored_cells, dtype=np.int64), fc, ft, MEAN, None,
        flat_c, flat_c, flat_t, FLOOR_HW, FLOOR_GRID,
    )


def test_the_primary_floor_is_the_mean_scored_against_the_landing_read():
    """Cosines 0.8, 0, and 1 at A, B, and C. The 0.8 shows the floor is scored
    against the same landing read the other arms are, not the cell's vector."""
    lift, fc, ft = _floor_scene()
    support = torch.ones(3, dtype=torch.bool)
    scores = score_primary(lift, support, fc, ft, MEAN, None, FLOOR_GRID)
    assert scores.meanfeat_raw == pytest.approx((0.8 + 0.0 + 1.0) / 3)
    assert scores.meanfeat_l2_raw == pytest.approx((math.sqrt(0.4) + SQRT2 + 0.0) / 3)


def test_the_primary_floor_reads_only_the_support():
    lift, fc, ft = _floor_scene()
    support = torch.tensor([True, True, False])
    scores = score_primary(lift, support, fc, ft, MEAN, None, FLOOR_GRID)
    assert scores.n_primary == 2
    assert scores.meanfeat_raw == pytest.approx((0.8 + 0.0) / 2)
    assert scores.meanfeat_l2_raw == pytest.approx((math.sqrt(0.4) + SQRT2) / 2)


def test_the_splat_floor_is_the_mean_scored_against_each_scored_cell():
    """Cells 0, 2, and 3 hold E0 + M, E1, and -M: cosines 1/sqrt(2), 0, and -1."""
    _, fc, ft = _floor_scene()
    flat_c, flat_t = fc.reshape(4, -1), ft.reshape(4, -1)
    scores = score_splat_pool(np.array([0, 2, 3]), flat_c, flat_c, flat_t, MEAN, None)
    assert scores.n_splat == 3
    assert scores.sp_meanfeat_raw == pytest.approx((1.0 / SQRT2 + 0.0 - 1.0) / 3)
    assert scores.sp_meanfeat_l2_raw == pytest.approx(
        (math.sqrt(2.0 - SQRT2) + SQRT2 + 2.0) / 3
    )


def test_each_cross_path_arm_scores_the_floor_against_its_own_target():
    """The splat path scored cells 0, 2, and 3, and samples landed in 0, 2, and
    1, so the common cells are 0 and 2. In cell 0 the per-point arm's pooled
    target is the read at A, 0.75 E0 + M, and the splat arm's target is the
    cell's own E0 + M. Cell 2 holds E1 on both arms."""
    lift, fc, ft = _floor_scene()
    scores = _cross_path(lift, torch.ones(3, dtype=torch.bool), [0, 2, 3], fc, ft)
    assert scores.n_intersect == 2
    assert scores.x_meanfeat_raw == pytest.approx((0.8 + 0.0) / 2)
    assert scores.x_meanfeat_l2_raw == pytest.approx((math.sqrt(0.4) + SQRT2) / 2)
    assert scores.x_sp_meanfeat_raw == pytest.approx((1.0 / SQRT2 + 0.0) / 2)
    assert scores.x_sp_meanfeat_l2_raw == pytest.approx(
        (math.sqrt(2.0 - SQRT2) + SQRT2) / 2
    )


def test_an_empty_intersection_gives_no_floor_score():
    lift, fc, ft = _floor_scene()
    scores = _cross_path(lift, torch.ones(3, dtype=torch.bool), [3], fc, ft)
    assert scores.n_intersect == 0
    for name in ("x_meanfeat_raw", "x_meanfeat_l2_raw",
                 "x_sp_meanfeat_raw", "x_sp_meanfeat_l2_raw"):
        assert math.isnan(getattr(scores, name)), name


def test_each_record_writes_exactly_the_columns_the_estimand_layer_reads():
    """One schema, named in lot.phase5_estimands and written by the scorer.
    Mean-Feature has raw columns only. Its centered columns do not exist, so no
    epsilon-regularized zero vector can stand in for them."""
    from lot.phase5_estimands import INTERSECTION_FIELDS, PRIMARY_FIELDS, SPLAT_FIELDS

    lift, fc, ft = _floor_scene()
    support = torch.ones(3, dtype=torch.bool)
    flat_c, flat_t = fc.reshape(4, -1), ft.reshape(4, -1)
    records = {
        "primary": (
            score_primary(lift, support, fc, ft, MEAN, None, FLOOR_GRID).as_fields(),
            set(PRIMARY_FIELDS) | {"n_primary", "n_predict_nonfinite"},
        ),
        "splat": (
            score_splat_pool(np.array([0, 2, 3]), flat_c, flat_c, flat_t, MEAN, None)
            .as_fields(),
            set(SPLAT_FIELDS) | {"n_splat"},
        ),
        "cross_path": (
            _cross_path(lift, support, [0, 2, 3], fc, ft).as_fields(),
            set(INTERSECTION_FIELDS) | {"n_intersect"},
        ),
    }
    for name, (fields, registered) in records.items():
        assert set(fields) == registered, name
        assert not [k for k in fields if "meanfeat" in k and "centered" in k], name
