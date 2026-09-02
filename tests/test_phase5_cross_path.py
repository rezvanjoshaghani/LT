"""The cross-path common-valid producer: one atomic unit, one weight, both arms.

PROTOCOL 3.9's disclosure is only meaningful if the two paths are recomputed on
the same cells with the same weights. The first version of this producer scored
every per-point sample landing in a common cell while scoring the splat arm
once per cell, so collision multiplicity alone could manufacture a path
difference between operators that agreed everywhere. These tests hold the
producer to the cell as the unit, on both arms, and check the values against
the frozen Phase 3 metric implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from lot.context_lift import ContextLiftMap
from lot.evaluate import value_agreement
from lot.phase5_score import (
    CROSS_PATH_ARMS,
    CrossPathScores,
    cross_path_cells,
    pool_per_point_to_cells,
    score_cross_path,
)

PATCH = 14
C = 4
HW = (28, 28)          # a 2x2 target grid
GRID = (2, 2)
N_CELLS = 4

E0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
E1 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
ORTH = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
ZERO = torch.zeros(C, dtype=torch.float64)


def _lift(uv_context, uv_target) -> ContextLiftMap:
    n = uv_context.shape[0]
    return ContextLiftMap(
        uv_context=uv_context, uv_target=uv_target,
        z_target=torch.ones(n, dtype=torch.float64),
        depth_context=torch.ones(n, dtype=torch.float64),
        depth_valid=torch.ones(n, dtype=torch.bool),
        landed=torch.ones(n, dtype=torch.bool),
    )


def _collision_fixture():
    """Two common cells; two samples land in cell 0, one in cell 1.

    Landings sit exactly on patch centers so a bilinear read is exact and the
    only thing that can separate the arms is how they weight the cells. Per
    cell, the explicit effect is +1 in cell 0 and -1 in cell 1 on both paths;
    the learned effect is 0 on both paths. Both operators agree at every cell.
    """
    uv_context = torch.tensor([[6.5, 6.5], [20.5, 6.5], [6.5, 20.5]], dtype=torch.float64)
    uv_target = torch.tensor([[6.5, 6.5], [6.5, 6.5], [20.5, 6.5]], dtype=torch.float64)
    lift = _lift(uv_context, uv_target)
    primary = torch.ones(3, dtype=torch.bool)

    ft = torch.zeros(C, 2, 2, dtype=torch.float64)
    ft[:, 0, 0] = E0
    ft[:, 0, 1] = E1
    fc = torch.zeros(C, 2, 2, dtype=torch.float64)
    fc[:, 0, 0] = E0      # sample 0's context patch
    fc[:, 0, 1] = E0      # sample 1's context patch
    fc[:, 1, 0] = -E1     # sample 2's context patch
    predicted = ORTH[None].repeat(N_CELLS, 1)
    transported = torch.zeros(C, N_CELLS, dtype=torch.float64)
    transported[:, 0] = E0
    transported[:, 1] = -E1
    return lift, primary, fc, ft, predicted, transported


def _score(lift, primary, fc, ft, predicted, transported, cells=(0, 1)):
    return score_cross_path(
        lift, primary, np.array(cells, dtype=np.int64), fc, ft, ZERO, predicted,
        transported, fc.reshape(C, -1), ft.reshape(C, -1), HW, GRID, PATCH,
    )


# ---------------------------------------------------------------------------
# Round-three finding 1: the collision case
# ---------------------------------------------------------------------------

def test_collision_multiplicity_cannot_manufacture_a_path_difference():
    """The reviewer's reproduction, now required to give exactly zero."""
    out = _score(*_collision_fixture())
    pp = out.x_cl_raw - out.x_predict_raw
    sp = out.x_sp_transport_raw - out.x_sp_predict_raw
    assert out.n_intersect == 2
    assert pp == pytest.approx(0.0, abs=1e-12)
    assert sp == pytest.approx(0.0, abs=1e-12)
    assert pp - sp == pytest.approx(0.0, abs=1e-12)


def test_both_arms_average_over_the_same_cells_with_equal_weight():
    out = _score(*_collision_fixture())
    # Cell 0 explicit effect +1, cell 1 explicit effect -1, equal weights: mean 0.
    assert out.x_cl_raw == pytest.approx(0.0, abs=1e-12)
    assert out.x_sp_transport_raw == pytest.approx(0.0, abs=1e-12)
    # A per-sample weighting would have given (+1 + +1 - 1) / 3 on the per-point arm.
    assert out.x_cl_raw != pytest.approx(1.0 / 3.0, abs=1e-6)


def test_the_support_travels_with_the_record():
    lift, primary, fc, ft, predicted, transported = _collision_fixture()
    out = _score(lift, primary, fc, ft, predicted, transported)
    assert list(out.common_cells) == [0, 1]
    assert out.per_point_mask.tolist() == [True, True, True]
    assert out.samples_per_cell.tolist() == [2, 1]
    # And none of it leaks into the aggregated fields.
    fields = out.as_fields()
    for key in ("common_cells", "per_point_mask", "samples_per_cell"):
        assert key not in fields
    assert fields["n_intersect"] == 2


def test_pooling_conserves_samples():
    lift, primary, fc, ft, predicted, transported = _collision_fixture()
    out = _score(lift, primary, fc, ft, predicted, transported)
    assert int(out.samples_per_cell.sum()) == int(out.per_point_mask.sum())


def test_a_cell_only_one_path_scored_is_not_common():
    lift, primary, fc, ft, predicted, transported = _collision_fixture()
    # The splat path scored cell 0 only, so cell 1 drops out of the intersection.
    out = _score(lift, primary, fc, ft, predicted, transported, cells=(0,))
    assert out.n_intersect == 1
    assert list(out.common_cells) == [0]
    assert out.per_point_mask.tolist() == [True, True, False]


def test_an_empty_intersection_is_reported_not_invented():
    lift, primary, fc, ft, predicted, transported = _collision_fixture()
    out = _score(lift, primary, fc, ft, predicted, transported, cells=(2, 3))
    assert out.n_intersect == 0
    assert math.isnan(out.x_cl_raw)
    assert out.common_cells.size == 0
    assert not out.per_point_mask.any()


# ---------------------------------------------------------------------------
# Round-three finding 5: the L2 companions, checked against Phase 3's arithmetic
# ---------------------------------------------------------------------------

def test_every_arm_carries_all_four_protocol_columns():
    out = _score(*_collision_fixture())
    fields = out.as_fields()
    for arm in CROSS_PATH_ARMS:
        for column in ("raw", "centered", "l2_raw", "l2_centered"):
            key = f"x_{arm}_{column}"
            assert key in fields, key
            assert math.isfinite(fields[key]), key


def test_pooled_values_are_scored_by_the_frozen_metric():
    """Producer-level check: the record equals value_agreement on the pooled
    vectors, for cosine and L2, raw and centered."""
    lift, primary, fc, ft, predicted, transported = _collision_fixture()
    center = torch.tensor([0.1, -0.2, 0.05, 0.0], dtype=torch.float64)
    out = score_cross_path(
        lift, primary, np.array([0, 1]), fc, ft, center, predicted,
        transported, fc.reshape(C, -1), ft.reshape(C, -1), HW, GRID, PATCH,
    )
    # Rebuild the pooled per-point vectors by hand.
    target_cells = torch.stack([E0, E1])
    cl_cells = torch.stack([(E0 + E0) / 2, -E1])
    # value_agreement runs in float32 and this producer keeps float64, so the
    # two agree to float32 rounding rather than bit for bit: the same formula,
    # pinned the way Phase 4 pins its bootstrap against the reference loop.
    TOL = 1e-6
    cosine_raw, l2_raw = value_agreement(cl_cells, target_cells)
    cosine_c, l2_c = value_agreement(cl_cells, target_cells, center=center)
    assert out.x_cl_raw == pytest.approx(cosine_raw, abs=TOL)
    assert out.x_cl_l2_raw == pytest.approx(l2_raw, abs=TOL)
    assert out.x_cl_centered == pytest.approx(cosine_c, abs=TOL)
    assert out.x_cl_l2_centered == pytest.approx(l2_c, abs=TOL)
    # Splat arm on the same cells.
    sp_cells = torch.stack([E0, -E1])
    s_raw, s_l2 = value_agreement(sp_cells, target_cells)
    assert out.x_sp_transport_raw == pytest.approx(s_raw, abs=TOL)
    assert out.x_sp_transport_l2_raw == pytest.approx(s_l2, abs=TOL)


def test_l2_and_cosine_agree_about_which_arm_is_closer():
    out = _score(*_collision_fixture())
    # Explicit and learned have equal mean cosine here (0), but the learned arm
    # is orthogonal everywhere while the explicit arm is exact in one cell and
    # opposite in the other; L2 must see that difference where cosine cannot.
    assert out.x_cl_raw == pytest.approx(out.x_predict_raw, abs=1e-12)
    assert out.x_cl_l2_raw != pytest.approx(out.x_predict_l2_raw, abs=1e-6)


# ---------------------------------------------------------------------------
# The pooling helper and the cell selection
# ---------------------------------------------------------------------------

def test_pool_per_point_to_cells_averages_vectors_before_scoring():
    values = torch.tensor([[2.0, 0.0], [0.0, 2.0], [4.0, 4.0]], dtype=torch.float64)
    cells = np.array([5, 5, 9])
    pooled, counts = pool_per_point_to_cells(values, cells, np.array([5, 9]))
    assert torch.allclose(pooled[0], torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert torch.allclose(pooled[1], torch.tensor([4.0, 4.0], dtype=torch.float64))
    assert counts.tolist() == [2, 1]


def test_pool_ignores_samples_outside_the_common_cells():
    values = torch.ones(3, 2, dtype=torch.float64)
    pooled, counts = pool_per_point_to_cells(values, np.array([1, 2, 3]), np.array([2]))
    assert pooled.shape == (1, 2)
    assert counts.tolist() == [1]


def test_cross_path_cells_requires_both_paths():
    lift, primary, *_ = _collision_fixture()
    common, mask = cross_path_cells(lift, primary, np.array([1, 3]), HW, PATCH)
    assert list(common) == [1]
    assert mask.tolist() == [False, False, True]


def test_the_predictor_absent_leaves_learned_arms_nan_and_explicit_intact():
    lift, primary, fc, ft, _, transported = _collision_fixture()
    out = score_cross_path(
        lift, primary, np.array([0, 1]), fc, ft, ZERO, None,
        transported, fc.reshape(C, -1), ft.reshape(C, -1), HW, GRID, PATCH,
    )
    assert math.isnan(out.x_predict_raw) and math.isnan(out.x_sp_predict_raw)
    assert math.isfinite(out.x_cl_raw) and math.isfinite(out.x_sp_transport_raw)
    assert isinstance(out, CrossPathScores)
