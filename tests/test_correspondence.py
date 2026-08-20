"""PLAN Phase 0, test 5: exact ground-truth pairs and correctly constructed nulls."""

import torch

from lot.correspondence import gather_value_pairs, sample_correspondences
from lot.encoders import pixel_to_patch_coords
from lot.visibility import visibility_masks
from scenes import IMAGE_SIZE, PATCH, build_two_plane_scene

CTX_HW = (IMAGE_SIZE, IMAGE_SIZE)
BOX_LO = 0.5 * PATCH - 0.5
BOX_HI = IMAGE_SIZE - 0.5 * PATCH - 0.5


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


def _sample(scene, vm, num_samples, mode, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return sample_correspondences(
        scene.depth_target,
        scene.K,
        scene.K,
        scene.T_target_from_context,
        vm.covisible,
        num_samples,
        CTX_HW,
        patch_size=PATCH,
        mode=mode,
        generator=generator,
    )


def test_warp_locations_are_the_analytic_correspondence():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 800, "pixel")
    assert samples.uv_target.shape == (800, 2)
    disparity = scene.disparity_for_target_columns(samples.uv_target[:, 0])
    # The unproject-then-project round trip does not cancel bitwise in float64,
    # so continuous coordinates are compared with a 1e-9 pixel tolerance.
    assert torch.allclose(
        samples.uv_context_warp[:, 0], samples.uv_target[:, 0] + disparity, atol=1e-9, rtol=0
    )
    assert torch.allclose(
        samples.uv_context_warp[:, 1], samples.uv_target[:, 1], atol=1e-9, rtol=0
    )


def test_sampler_returns_only_covisible_locations():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 800, "pixel")
    u = samples.uv_target[:, 0].long()
    v = samples.uv_target[:, 1].long()
    assert vm.covisible[v, u].all()
    assert not vm.disoccluded[v, u].any()


def test_no_warp_copies_the_target_location():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 300, "pixel")
    assert torch.equal(samples.uv_context_no_warp, samples.uv_target)


def test_neighbor_is_one_patch_from_the_warp_and_in_bounds():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 300, "pixel")
    diff = (samples.uv_context_neighbor - samples.uv_context_warp).abs()
    sorted_diff, _ = torch.sort(diff, dim=1)
    expected = torch.tensor([0.0, float(PATCH)], dtype=diff.dtype).expand_as(sorted_diff)
    assert torch.allclose(sorted_diff, expected, atol=1e-9, rtol=0)
    assert (samples.uv_context_neighbor >= BOX_LO).all()
    assert (samples.uv_context_neighbor <= BOX_HI).all()


def test_random_locations_are_bounded_and_seeded():
    scene, vm = _scene_and_masks()
    first = _sample(scene, vm, 300, "pixel", seed=7)
    again = _sample(scene, vm, 300, "pixel", seed=7)
    other = _sample(scene, vm, 300, "pixel", seed=8)
    assert (first.uv_context_random >= BOX_LO).all()
    assert (first.uv_context_random <= BOX_HI).all()
    for a, b in zip(first, again):
        assert torch.equal(a, b)
    assert not torch.equal(first.uv_context_random, other.uv_context_random)


def test_patch_center_mode_covers_exactly_the_covisible_patches():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 10_000, "patch_center")
    # 12 co-visible patch columns times 16 rows.
    assert samples.uv_target.shape == (192, 2)
    patch_cols = pixel_to_patch_coords(samples.uv_target[:, 0], PATCH).round().long()
    assert set(patch_cols.tolist()) == {0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13}


def test_value_pairs_on_the_analytic_scene():
    scene, vm = _scene_and_masks()
    samples = _sample(scene, vm, 10_000, "patch_center")
    pairs = gather_value_pairs(scene.features_context, scene.expected_features, samples)

    # Ground-truth warp reproduces the target values exactly.
    assert torch.equal(pairs["warp"], pairs["target"])

    # The no-warp copy reads the context code at the same patch location.
    q = pixel_to_patch_coords(samples.uv_target, PATCH).round().long()
    expected_no_warp = scene.features_context[:, q[:, 1], q[:, 0]].T
    assert torch.equal(pairs["no_warp"].to(torch.float32), expected_no_warp)
    assert not torch.equal(pairs["no_warp"], pairs["warp"])

    # The neighbor reads the context code one patch from the warp location.
    qn = pixel_to_patch_coords(samples.uv_context_neighbor, PATCH).round().long()
    expected_neighbor = scene.features_context[:, qn[:, 1], qn[:, 0]].T
    assert torch.equal(pairs["neighbor"].to(torch.float32), expected_neighbor)

    # The mean null is the mean context feature vector on every row.
    expected_mean = scene.features_context.mean(dim=(1, 2))
    assert torch.allclose(pairs["mean"], expected_mean.expand_as(pairs["mean"]).to(pairs["mean"].dtype))

    shapes = {v.shape for v in pairs.values()}
    assert shapes == {(192, 3)}
