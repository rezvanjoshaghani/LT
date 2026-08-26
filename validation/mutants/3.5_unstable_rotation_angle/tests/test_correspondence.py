"""PLAN Phase 0 test 5 and PROTOCOL 3.2/3.6: sample identity and the frozen nulls."""

import numpy as np
import pytest
import torch

from lot.correspondence import (
    NEIGHBOR_OFFSETS,
    gather_value_pairs,
    sample_correspondences,
)
from lot.encoders import pixel_to_patch_coords
from lot.sample_identity import (
    NEIGHBOR_PATCH_SALT,
    RANDOM_PATCH_SALT,
    derived_draw,
    sample_ids,
)
from lot.visibility import visibility_masks
from scenes import GRID, IMAGE_SIZE, PATCH, build_two_plane_scene

CTX_HW = (IMAGE_SIZE, IMAGE_SIZE)
BOX_LO = 0.5 * PATCH - 0.5
BOX_HI = IMAGE_SIZE - 0.5 * PATCH - 0.5
IDENTITY = ("room_0", "ctx", "tgt")


def _scene_and_masks():
    scene = build_two_plane_scene()
    vm = visibility_masks(
        scene.depth_target,
        scene.depth_context,
        scene.K,
        scene.K,
        scene.T_target_from_context,
    )
    return scene, vm


def _sample(scene, vm, num_samples, mode="patch_center", identity=IDENTITY):
    return sample_correspondences(
        scene.depth_target,
        scene.K,
        scene.K,
        scene.T_target_from_context,
        vm.covisible,
        num_samples,
        CTX_HW,
        *identity,
        patch_size=PATCH,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# PROTOCOL 3.2: sample identity
# ---------------------------------------------------------------------------

def test_sample_id_depends_only_on_the_four_named_inputs():
    """scene, context frame, target frame, and the target-side coordinates."""
    uv = torch.tensor([[6.5, 6.5], [20.5, 34.5]], dtype=torch.float64)
    base = sample_ids("room_0", "c", "t", uv)
    assert np.array_equal(base, sample_ids("room_0", "c", "t", uv))
    # A different value in any one of the four changes the id.
    assert not np.array_equal(base, sample_ids("room_1", "c", "t", uv))
    assert not np.array_equal(base, sample_ids("room_0", "c2", "t", uv))
    assert not np.array_equal(base, sample_ids("room_0", "c", "t2", uv))
    assert base[0] != base[1]


def test_sample_id_is_order_and_batch_independent():
    """The id travels with the correspondence, not with its position in a batch."""
    uv = torch.tensor([[6.5, 6.5], [20.5, 6.5], [6.5, 20.5]], dtype=torch.float64)
    full = sample_ids("room_0", "c", "t", uv)
    permuted = sample_ids("room_0", "c", "t", uv[[2, 0, 1]])
    assert permuted.tolist() == [full[2], full[0], full[1]]
    single = sample_ids("room_0", "c", "t", uv[1:2])
    assert single[0] == full[1]


def test_sample_id_rejects_coordinates_off_the_half_pixel_grid():
    """Ids are defined on that grid so float noise cannot move a record."""
    with pytest.raises(ValueError, match="half a pixel"):
        sample_ids("room_0", "c", "t", torch.tensor([[6.3, 6.5]], dtype=torch.float64))


def test_every_sample_carries_an_id():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 64)
    assert len(samples.sample_id) == samples.uv_target.shape[0]
    assert len(set(samples.sample_id.tolist())) == len(samples.sample_id)


# ---------------------------------------------------------------------------
# PROTOCOL 3.6: the frozen nulls
# ---------------------------------------------------------------------------

def test_nulls_are_hash_deterministic_not_order_dependent():
    """PROTOCOL 3.6: the same record receives the same null regardless of batching."""
    scene, vm = _scene_and_masks()
    many = _sample(scene, vm, 10_000)
    few = _sample(scene, vm, 32)
    shared = {int(i) for i in many.sample_id} & {int(i) for i in few.sample_id}
    assert shared, "the two draws must overlap for this to test anything"
    many_index = {int(i): k for k, i in enumerate(many.sample_id)}
    few_index = {int(i): k for k, i in enumerate(few.sample_id)}
    for identifier in shared:
        a, b = many_index[identifier], few_index[identifier]
        assert torch.equal(many.uv_context_neighbor[a], few.uv_context_neighbor[b])
        assert torch.equal(many.random_patch_index[a], few.random_patch_index[b])


def test_neighbor_offset_is_one_whole_patch_on_an_axis():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 400)
    offsets = (samples.uv_context_neighbor - samples.uv_context_warp).tolist()
    allowed = {(float(dx * PATCH), float(dy * PATCH)) for dx, dy in NEIGHBOR_OFFSETS}
    assert set(map(tuple, offsets)) <= allowed


def test_neighbor_direction_is_unbiased_across_directions():
    """A single fixed direction would confound localization with image geometry."""
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 10_000)
    offsets = (samples.uv_context_neighbor - samples.uv_context_warp).tolist()
    used = {tuple(o) for o in offsets}
    assert len(used) == len(NEIGHBOR_OFFSETS)


def test_neighbor_direction_comes_from_the_sample_id():
    """Where all four offsets are in bounds the hash selects among them directly.

    At a border only some offsets are in bounds, and the hash then indexes the
    surviving ones, which is what makes the null defined everywhere.
    """
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 400)
    expected = derived_draw(samples.sample_id, NEIGHBOR_PATCH_SALT, 4)
    offsets = (samples.uv_context_neighbor - samples.uv_context_warp) / PATCH
    warp = samples.uv_context_warp
    interior = (
        (warp[:, 0] - PATCH >= BOX_LO)
        & (warp[:, 0] + PATCH <= BOX_HI)
        & (warp[:, 1] - PATCH >= BOX_LO)
        & (warp[:, 1] + PATCH <= BOX_HI)
    )
    assert int(interior.sum()) > 0
    for row in torch.nonzero(interior).flatten().tolist():
        choice = int(expected[row])
        assert tuple(offsets[row].tolist()) == tuple(float(v) for v in NEIGHBOR_OFFSETS[choice])


def test_random_patch_is_an_integer_patch_index_from_the_hash():
    """PROTOCOL 3.6 asks for a patch, not a blend of up to four."""
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 128)
    expected = derived_draw(samples.sample_id, RANDOM_PATCH_SALT, GRID * GRID)
    rows = samples.random_patch_index[:, 0].tolist()
    cols = samples.random_patch_index[:, 1].tolist()
    assert [r * GRID + c for r, c in zip(rows, cols)] == expected.tolist()
    assert all(0 <= r < GRID and 0 <= c < GRID for r, c in zip(rows, cols))


def test_random_patch_reads_the_patch_itself():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 32)
    features = torch.rand((6, GRID, GRID))
    values = gather_value_pairs(features, features, samples)
    for row in range(len(samples.sample_id)):
        r, c = samples.random_patch_index[row].tolist()
        assert torch.equal(values["random"][row], features[:, r, c])


def test_every_null_reads_the_context_map_and_differs_only_in_where():
    """PROTOCOL 3.6: image, encoder, and scene held fixed; only the location moves."""
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 200)
    context = torch.full((4, GRID, GRID), 1.0)
    target = torch.full((4, GRID, GRID), 2.0)
    pairs = gather_value_pairs(context, target, samples)
    assert torch.allclose(pairs["target"], torch.full_like(pairs["target"], 2.0))
    for null in ("warp", "no_warp", "neighbor", "random"):
        assert torch.allclose(pairs[null], torch.full_like(pairs[null], 1.0)), null


def test_mean_feature_is_not_built_by_the_sampler():
    """PROTOCOL 3.6 defines it over the training split, which a sampler cannot see."""
    scene, vm = _scene_and_masks()
    pairs = gather_value_pairs(
        scene.features_context, scene.expected_features, _sample(scene, vm, 16)
    )
    assert "mean" not in pairs


# ---------------------------------------------------------------------------
# Correspondence correctness, unchanged from Phase 0
# ---------------------------------------------------------------------------

def test_warp_locations_are_the_analytic_correspondence():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 800, mode="pixel")
    disparity = scene.disparity_for_target_columns(samples.uv_target[:, 0])
    assert torch.allclose(
        samples.uv_context_warp[:, 0], samples.uv_target[:, 0] + disparity, atol=1e-9, rtol=0
    )
    assert torch.allclose(
        samples.uv_context_warp[:, 1], samples.uv_target[:, 1], atol=1e-9, rtol=0
    )


def test_sampler_returns_only_covisible_locations():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 800, mode="pixel")
    u = samples.uv_target[:, 0].long()
    v = samples.uv_target[:, 1].long()
    assert vm.covisible[v, u].all()
    assert not vm.disoccluded[v, u].any()


def test_no_warp_copies_the_target_location():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 300, mode="pixel")
    assert torch.equal(samples.uv_context_no_warp, samples.uv_target)


def test_patch_center_mode_covers_exactly_the_covisible_patches():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 10_000)
    assert samples.uv_target.shape == (192, 2)
    patch_cols = pixel_to_patch_coords(samples.uv_target[:, 0], PATCH).round().long()
    assert set(patch_cols.tolist()) == {0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13}


def test_patch_center_rejects_centers_that_straddle_a_depth_edge():
    """Interpolating across a depth edge lifts the center to a point in mid air."""
    scene = build_two_plane_scene()
    depth = torch.full((IMAGE_SIZE, IMAGE_SIZE), 4.0, dtype=torch.float64)
    depth[:, 49:] = 2.0
    samples = sample_correspondences(
        depth,
        scene.K,
        scene.K,
        scene.T_target_from_context,
        torch.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool),
        10_000,
        CTX_HW,
        *IDENTITY,
        patch_size=PATCH,
    )
    cols = set(pixel_to_patch_coords(samples.uv_target[:, 0], PATCH).round().long().tolist())
    assert 3 not in cols
    assert {2, 4}.issubset(cols)


def test_records_without_an_in_bounds_neighbor_are_omitted_and_counted():
    """PROTOCOL 3.6: the omission is counted and documented, not silently absorbed."""
    side = 2 * PATCH
    scene = build_two_plane_scene()
    samples = sample_correspondences(
        torch.full((side, side), 3.0, dtype=torch.float64),
        scene.K,
        scene.K,
        torch.eye(4, dtype=torch.float64),
        torch.ones((side, side), dtype=torch.bool),
        16,
        (side, side),
        *IDENTITY,
        patch_size=PATCH,
        mode="pixel",
    )
    # On a two-patch image every integer pixel inside the sampling box is more
    # than one patch from the far edge and less than one from the near one, so
    # no record has an in-bounds offset at all.
    assert samples.uv_target.shape[0] == 0
    assert samples.neighbor_omitted > 0


def test_value_pairs_on_the_analytic_scene():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 10_000)
    pairs = gather_value_pairs(scene.features_context, scene.expected_features, samples)
    assert torch.equal(pairs["warp"], pairs["target"])
    q = pixel_to_patch_coords(samples.uv_target, PATCH).round().long()
    expected_no_warp = scene.features_context[:, q[:, 1], q[:, 0]].T
    assert torch.equal(pairs["no_warp"].to(torch.float32), expected_no_warp)
    assert not torch.equal(pairs["no_warp"], pairs["warp"])
    assert {v.shape for v in pairs.values()} == {(192, 3)}
