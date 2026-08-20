"""PLAN Phase 0, test 1: projection round trips and transform composition."""

import torch

from lot import geometry
from scenes import build_rotation_scene, intrinsics, random_se3


def _random_points(n: int, generator: torch.Generator) -> torch.Tensor:
    p = torch.rand((n, 3), generator=generator, dtype=torch.float64)
    p[:, 0] = p[:, 0] * 4 - 2
    p[:, 1] = p[:, 1] * 4 - 2
    p[:, 2] = p[:, 2] * 9 + 0.5
    return p


def test_project_unproject_identity():
    g = torch.Generator().manual_seed(0)
    K = torch.tensor(
        [[300.0, 0.0, 160.0], [0.0, 280.0, 120.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    points = _random_points(1000, g)
    uv, z = geometry.project(points, K)
    assert torch.allclose(z, points[:, 2])
    back = geometry.unproject(uv, z, K)
    assert torch.allclose(back, points, atol=1e-10)


def test_unproject_project_identity():
    g = torch.Generator().manual_seed(1)
    K = intrinsics()
    uv = torch.rand((500, 2), generator=g, dtype=torch.float64) * 223
    depth = torch.rand(500, generator=g, dtype=torch.float64) * 9 + 0.5
    points = geometry.unproject(uv, depth, K)
    uv_back, z_back = geometry.project(points, K)
    assert torch.allclose(uv_back, uv, atol=1e-10)
    assert torch.allclose(z_back, depth)


def test_invert_se3_matches_matrix_inverse():
    g = torch.Generator().manual_seed(2)
    for _ in range(20):
        T = random_se3(g)
        assert torch.allclose(geometry.invert_se3(T), torch.linalg.inv(T), atol=1e-10)


def test_relative_pose_definition_and_identity():
    g = torch.Generator().manual_seed(3)
    T_world_from_target = random_se3(g)
    T_world_from_context = random_se3(g)
    T_rel = geometry.relative_pose(T_world_from_target, T_world_from_context)
    direct = geometry.invert_se3(T_world_from_target) @ T_world_from_context
    assert torch.allclose(T_rel, direct, atol=1e-12)
    eye = geometry.relative_pose(T_world_from_target, T_world_from_target)
    assert torch.allclose(eye, torch.eye(4, dtype=torch.float64), atol=1e-12)


def test_relative_pose_maps_points_through_world():
    g = torch.Generator().manual_seed(4)
    T_world_from_target = random_se3(g)
    T_world_from_context = random_se3(g)
    points_ctx = _random_points(200, g)
    via_world = geometry.transform_points(
        geometry.invert_se3(T_world_from_target),
        geometry.transform_points(T_world_from_context, points_ctx),
    )
    direct = geometry.transform_points(
        geometry.relative_pose(T_world_from_target, T_world_from_context), points_ctx
    )
    assert torch.allclose(direct, via_world, atol=1e-10)


def test_composing_two_transforms_equals_direct():
    g = torch.Generator().manual_seed(5)
    T_world_from_a = random_se3(g)
    T_world_from_b = random_se3(g)
    T_world_from_c = random_se3(g)
    direct = geometry.relative_pose(T_world_from_a, T_world_from_c)
    composed = geometry.compose(
        geometry.relative_pose(T_world_from_a, T_world_from_b),
        geometry.relative_pose(T_world_from_b, T_world_from_c),
    )
    assert torch.allclose(composed, direct, atol=1e-10)
    points = _random_points(100, g)
    assert torch.allclose(
        geometry.transform_points(composed, points),
        geometry.transform_points(T_world_from_a.inverse() @ T_world_from_c, points),
        atol=1e-9,
    )


def test_parallax_is_baseline_over_median_depth():
    T = torch.eye(4, dtype=torch.float64)
    T[0, 3] = 0.3
    T[1, 3] = 0.4
    depth = torch.full((10, 10), 2.0, dtype=torch.float64)
    assert torch.allclose(geometry.parallax(T, depth), torch.tensor(0.25, dtype=torch.float64))
    depth[0, :] = 0.0  # invalid entries are ignored
    assert torch.allclose(geometry.parallax(T, depth), torch.tensor(0.25, dtype=torch.float64))


def test_rotation_homography_matches_projection_and_ignores_depth():
    scene = build_rotation_scene(yaw_deg=5.0)
    g = torch.Generator().manual_seed(6)
    uv = geometry.pixel_grid(32, 32, dtype=torch.float64) * 7.0  # spread over the image
    depth_a = torch.rand((32, 32), generator=g, dtype=torch.float64) * 4 + 1
    depth_b = torch.rand((32, 32), generator=g, dtype=torch.float64) * 4 + 1

    def project_through_3d(depth):
        points = geometry.unproject(uv, depth, scene.K)
        moved = geometry.transform_points(scene.T_target_from_context, points)
        uv_out, _ = geometry.project(moved, scene.K)
        return uv_out

    uv_a = project_through_3d(depth_a)
    uv_b = project_through_3d(depth_b)
    assert torch.allclose(uv_a, uv_b, atol=1e-9)

    H = geometry.rotation_homography(scene.K, scene.K, scene.R_target_from_context)
    uv_h = geometry.apply_homography(H, uv)
    assert torch.allclose(uv_a, uv_h, atol=1e-9)
