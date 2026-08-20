"""PLAN Phase 0, test 2 (visibility half): the occluded strip matches the analytic answer."""

import torch

from lot.visibility import fraction_per_patch, visibility_masks
from scenes import (
    DISOCCLUDED_TGT_COLS,
    GRID,
    IMAGE_SIZE,
    OUT_OF_VIEW_TGT_COLS,
    PATCH,
    Z_BACK,
    build_two_plane_scene,
)


def _masks(scene, **kwargs):
    return visibility_masks(
        scene.depth_target,
        scene.depth_context,
        scene.K,
        scene.K,
        scene.T_target_from_context,
        **kwargs,
    )


def test_two_plane_covisibility_is_exact():
    scene = build_two_plane_scene()
    vm = _masks(scene)
    assert vm.valid.all()
    assert torch.equal(vm.covisible, scene.covisible_target)
    assert torch.equal(vm.disoccluded, ~scene.covisible_target)


def test_disoccluded_strip_matches_analytic_within_one_patch():
    scene = build_two_plane_scene()
    vm = _masks(scene)
    frac = fraction_per_patch(vm.covisible, PATCH)
    expected = torch.ones((GRID, GRID), dtype=torch.float32)
    empty_patch_cols = [5, 6, 14, 15]
    expected[:, empty_patch_cols] = 0.0
    assert torch.equal(frac, expected)


def test_out_of_frustum_region():
    scene = build_two_plane_scene()
    vm = _masks(scene)
    cols = torch.arange(IMAGE_SIZE)
    expected = (cols < OUT_OF_VIEW_TGT_COLS[0])[None, :].expand(IMAGE_SIZE, IMAGE_SIZE)
    assert torch.equal(vm.in_context_frustum, expected)


def test_z_in_context_for_back_plane():
    scene = build_two_plane_scene()
    vm = _masks(scene)
    back_cols = torch.arange(IMAGE_SIZE) >= DISOCCLUDED_TGT_COLS[1]
    back_cols &= torch.arange(IMAGE_SIZE) < OUT_OF_VIEW_TGT_COLS[0]
    # A pure lateral move preserves z, so back-plane points sit at Z_BACK in the context camera.
    assert torch.allclose(
        vm.z_in_context[:, back_cols],
        torch.full_like(vm.z_in_context[:, back_cols], Z_BACK),
    )


def test_relative_tolerance_semantics():
    scene = build_two_plane_scene()
    baseline = _masks(scene)
    # A 1 percent depth perturbation stays inside the 1.5 percent tolerance.
    close = visibility_masks(
        scene.depth_target,
        scene.depth_context * 1.01,
        scene.K,
        scene.K,
        scene.T_target_from_context,
    )
    assert torch.equal(close.covisible, baseline.covisible)
    # A 3 percent perturbation breaks every match.
    far = visibility_masks(
        scene.depth_target,
        scene.depth_context * 1.03,
        scene.K,
        scene.K,
        scene.T_target_from_context,
    )
    assert not far.covisible.any()


def test_invalid_context_depth_is_conservative():
    scene = build_two_plane_scene()
    baseline = _masks(scene)
    hole_row, hole_col = 100, 150
    depth_context = scene.depth_context.clone()
    depth_context[hole_row, hole_col] = 0.0
    vm = visibility_masks(
        scene.depth_target,
        depth_context,
        scene.K,
        scene.K,
        scene.T_target_from_context,
    )
    # The target pixel landing exactly on the hole flips to not co-visible.
    flipped = baseline.covisible & ~vm.covisible
    assert flipped[hole_row, hole_col - 28]
    # Nothing becomes co-visible that was not before.
    assert not (vm.covisible & ~baseline.covisible).any()


def test_fraction_per_patch_small_example():
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[0, 0] = True
    mask[2:, 2:] = True
    frac = fraction_per_patch(mask, 2)
    expected = torch.tensor([[0.25, 0.0], [0.0, 1.0]], dtype=torch.float32)
    assert torch.equal(frac, expected)
