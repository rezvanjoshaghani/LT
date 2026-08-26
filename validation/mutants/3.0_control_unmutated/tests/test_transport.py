"""PLAN Phase 0, tests 2, 3, 4: exact two-plane transport, pure rotation, coverage."""

import torch

from lot.transport import transport
from scenes import (
    GRID,
    IMAGE_SIZE,
    PATCH,
    SLAB_TGT_COLS,
    Z_FRONT,
    build_rotation_scene,
    build_single_plane_scene,
    build_two_plane_scene,
    patch_codes,
)

OUT_HW = (IMAGE_SIZE, IMAGE_SIZE)


def test_two_plane_transport_lands_exactly():
    scene = build_two_plane_scene()
    features, coverage, zbuffer = transport(
        scene.features_context,
        scene.depth_context,
        scene.K,
        scene.K,
        scene.T_target_from_context,
        OUT_HW,
    )
    assert torch.equal(features, scene.expected_features)
    assert torch.equal(coverage, scene.expected_coverage)
    assert torch.equal(zbuffer, scene.expected_zbuffer)
    assert torch.isfinite(features).all()
    assert torch.isfinite(coverage).all()
    assert not torch.isnan(zbuffer).any()


def test_two_plane_zbuffer_prefers_the_front_surface():
    scene = build_two_plane_scene()
    _, _, zbuffer = transport(
        scene.features_context,
        scene.depth_context,
        scene.K,
        scene.K,
        scene.T_target_from_context,
        OUT_HW,
    )
    # Back-plane splats also land on part of the slab region and must lose.
    contested = zbuffer[:, SLAB_TGT_COLS[0]:42]
    assert torch.equal(contested, torch.full_like(contested, Z_FRONT))


def test_coverage_values():
    scene = build_single_plane_scene(disparity_px=7)
    features, coverage, _ = transport(
        scene.features_context,
        scene.depth_context,
        scene.K,
        scene.K,
        scene.T_target_from_context,
        OUT_HW,
    )
    # Fully supported patches are exactly 1.0, the trailing column is exactly half
    # covered, and pooled features are the exact mixture of the two source patches.
    assert torch.equal(coverage, scene.expected_coverage)
    assert torch.equal(features, scene.expected_features)
    assert torch.isfinite(features).all()
    assert torch.isfinite(coverage).all()

    two_plane = build_two_plane_scene()
    _, coverage2, _ = transport(
        two_plane.features_context,
        two_plane.depth_context,
        two_plane.K,
        two_plane.K,
        two_plane.T_target_from_context,
        OUT_HW,
    )
    holes = coverage2[:, [5, 6, 14, 15]]
    assert torch.equal(holes, torch.zeros_like(holes))
    assert torch.equal(
        coverage2[:, [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]],
        torch.ones((GRID, 12), dtype=torch.float32),
    )


def _reference_pixel_splat(features, depth, K, T_tgt_from_ctx, out_hw, patch):
    """The pixel-level form of the same contract, carrying feature vectors.

    transport accumulates scalar weights from source patch to target patch and
    mixes the features once at the end, because moving a vector per pixel costs
    hundreds of megabytes per call. This does it the direct way instead: average
    the winning splats on each target pixel, then average the pixels that got
    support within each patch. The two must agree.
    """
    from lot.geometry import pixel_grid, project, transform_points, unproject

    height, width = depth.shape
    out_height, out_width = out_hw
    channels = features.shape[0]
    uv = pixel_grid(height, width, dtype=depth.dtype)
    points = transform_points(T_tgt_from_ctx, unproject(uv, depth, K))
    uv_tgt, z_tgt = project(points, K)

    feat_px = torch.zeros((channels, out_height, out_width), dtype=torch.float32)
    hit_px = torch.zeros((out_height, out_width), dtype=torch.bool)
    hits: dict[tuple[int, int], list[tuple[float, int, int]]] = {}
    for v in range(height):
        for u in range(width):
            z = float(z_tgt[v, u])
            if not (z > 0 and float(depth[v, u]) > 0):
                continue
            iu = int(torch.floor(uv_tgt[v, u, 0] + 0.5))
            iv = int(torch.floor(uv_tgt[v, u, 1] + 0.5))
            if not (0 <= iu < out_width and 0 <= iv < out_height):
                continue
            hits.setdefault((iv, iu), []).append((z, v // patch, u // patch))
    for (iv, iu), entries in hits.items():
        zmin = min(e[0] for e in entries)
        winners = [e for e in entries if e[0] <= zmin * (1 + 1e-6)]
        stacked = torch.stack([features[:, r, q].to(torch.float32) for _, r, q in winners])
        feat_px[:, iv, iu] = stacked.mean(dim=0)
        hit_px[iv, iu] = True

    grid_h, grid_w = out_height // patch, out_width // patch
    feat_out = torch.zeros((channels, grid_h, grid_w), dtype=torch.float32)
    coverage = torch.zeros((grid_h, grid_w), dtype=torch.float32)
    for pr in range(grid_h):
        for pc in range(grid_w):
            block_hit = hit_px[pr * patch:(pr + 1) * patch, pc * patch:(pc + 1) * patch]
            n = int(block_hit.sum())
            coverage[pr, pc] = n / (patch * patch)
            if n:
                block = feat_px[:, pr * patch:(pr + 1) * patch, pc * patch:(pc + 1) * patch]
                feat_out[:, pr, pc] = block.sum(dim=(1, 2)) / n
    return feat_out, coverage


def test_weighted_form_matches_the_pixel_level_splat():
    """Guards the weight-matrix accumulation against the direct per-pixel form.

    A random depth map makes splats collide and tie in ways the analytic scenes
    never do, which is exactly where an accumulation refactor could drift.
    """
    generator = torch.Generator().manual_seed(3)
    scene = build_two_plane_scene()
    depth = 2.0 + 2.0 * torch.rand(
        (IMAGE_SIZE, IMAGE_SIZE), generator=generator, dtype=torch.float64
    )
    features = patch_codes()
    result = transport(
        features, depth, scene.K, scene.K, scene.T_target_from_context, OUT_HW
    )
    ref_features, ref_coverage = _reference_pixel_splat(
        features, depth, scene.K, scene.T_target_from_context, OUT_HW, PATCH
    )
    assert torch.equal(result.coverage, ref_coverage)
    assert torch.allclose(result.features, ref_features, atol=1e-4)
    # The scene has to exercise the hard cases or the agreement proves nothing:
    # partly supported patches, and patches nothing reached at all.
    assert ((result.coverage > 0) & (result.coverage < 1)).any()
    assert (result.coverage == 0).any()


def _reference_homography_splat(features, depth, K, R_tgt_from_ctx, patch):
    """Independent forward splat that uses the homography instead of 3D geometry.

    For pure rotation, K @ R @ inv(K) maps context pixels to target pixels and the
    third homogeneous coordinate scales source depth into target depth.
    """
    height, width = depth.shape
    Hm = (K @ R_tgt_from_ctx.to(K.dtype) @ torch.linalg.inv(K)).numpy()
    depth_np = depth.numpy()
    feats_np = features.numpy()
    channels = feats_np.shape[0]
    hits: dict[tuple[int, int], list[tuple[float, int, int]]] = {}
    for v in range(height):
        for u in range(width):
            hx = Hm[0][0] * u + Hm[0][1] * v + Hm[0][2]
            hy = Hm[1][0] * u + Hm[1][1] * v + Hm[1][2]
            hz = Hm[2][0] * u + Hm[2][1] * v + Hm[2][2]
            ut, vt = hx / hz, hy / hz
            zt = depth_np[v][u] * hz
            if zt <= 0:
                continue
            iu = int(ut + 0.5) if ut + 0.5 >= 0 else -1
            iv = int(vt + 0.5) if vt + 0.5 >= 0 else -1
            if not (0 <= iu < width and 0 <= iv < height):
                continue
            hits.setdefault((iv, iu), []).append((zt, v // patch, u // patch))
    feat_px = torch.zeros((channels, height, width), dtype=torch.float32)
    hit_px = torch.zeros((height, width), dtype=torch.bool)
    for (iv, iu), entries in hits.items():
        zmin = min(e[0] for e in entries)
        winners = [e for e in entries if e[0] <= zmin * (1 + 1e-6)]
        for c in range(channels):
            feat_px[c, iv, iu] = sum(float(feats_np[c][r][q]) for _, r, q in winners) / len(winners)
        hit_px[iv, iu] = True
    grid_h, grid_w = height // patch, width // patch
    feat_out = torch.zeros((channels, grid_h, grid_w), dtype=torch.float32)
    coverage = torch.zeros((grid_h, grid_w), dtype=torch.float32)
    for pr in range(grid_h):
        for pc in range(grid_w):
            block_hit = hit_px[pr * patch:(pr + 1) * patch, pc * patch:(pc + 1) * patch]
            n = int(block_hit.sum())
            coverage[pr, pc] = n / (patch * patch)
            if n:
                block = feat_px[:, pr * patch:(pr + 1) * patch, pc * patch:(pc + 1) * patch]
                feat_out[:, pr, pc] = block.sum(dim=(1, 2)) / n
    return feat_out, coverage


def test_pure_rotation_equals_homography_warp():
    scene = build_rotation_scene(yaw_deg=5.0)
    features = patch_codes()
    depth = torch.full((IMAGE_SIZE, IMAGE_SIZE), 2.5, dtype=torch.float64)
    result = transport(
        features, depth, scene.K, scene.K, scene.T_target_from_context, OUT_HW
    )
    ref_features, ref_coverage = _reference_homography_splat(
        features, depth, scene.K, scene.R_target_from_context, PATCH
    )
    assert torch.equal(result.coverage, ref_coverage)
    assert torch.allclose(result.features, ref_features, atol=1e-5)


def test_pure_rotation_transport_is_depth_independent():
    scene = build_rotation_scene(yaw_deg=5.0)
    features = patch_codes()
    depth_a = torch.full((IMAGE_SIZE, IMAGE_SIZE), 2.0, dtype=torch.float64)
    depth_b = torch.full((IMAGE_SIZE, IMAGE_SIZE), 3.7, dtype=torch.float64)
    result_a = transport(features, depth_a, scene.K, scene.K, scene.T_target_from_context, OUT_HW)
    result_b = transport(features, depth_b, scene.K, scene.K, scene.T_target_from_context, OUT_HW)
    assert torch.equal(result_a.features, result_b.features)
    assert torch.equal(result_a.coverage, result_b.coverage)
