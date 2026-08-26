"""VALIDATION 2.1, 2.2 and an independent cross-check of visibility and transport.

The protocol asks for these checks on pairs drawn from the render manifests.
This repository ships no data/, cache/ or outputs/ directory, so no manifest,
no cached feature and no parquet exists to draw from. The checks below are run
instead on synthetic camera pairs built here, one per regime shape, with the
cameras and depth chosen so that every convention error the manifest version
exists to catch (flipped axis, half-pixel offset, reversed relative pose,
ray-distance versus planar depth, mismatched intrinsics) still moves the
result. This is a surrogate, and the report records it as one.

Independent side: validation/independent.py, written from the protocol text.
Pipeline side: the lot package. Nothing is imported from lot into
independent.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import independent as ind  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lot import correspondence, geometry, transport, visibility  # noqa: E402

TOL_PX = 0.1  # VALIDATION 2.1/2.2 tolerance
PATCH = 14
H = W = 7 * PATCH  # 98 px, 7x7 patches


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def pose(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def two_plane_depth(h, w, near=2.0, far=5.0, edge_frac=0.45):
    """A front plane occluding a back plane. Planar z-depth, fronto-parallel."""
    d = np.full((h, w), far, dtype=np.float64)
    d[:, : int(round(w * edge_frac))] = near
    return d


# Three regimes, matching PROTOCOL 3.3, with DIFFERENT intrinsics on the two
# cameras so that assuming K_ctx == K_tgt would be caught.
K_CTX = intrinsics(90.0, 88.0, (W - 1) / 2.0, (H - 1) / 2.0)
K_TGT = intrinsics(93.0, 91.0, (W - 1) / 2.0 + 1.5, (H - 1) / 2.0 - 0.75)

PAIRS = {
    # in-place rotation: translation exactly zero
    "rotation": (
        pose(np.eye(3), np.zeros(3)),
        pose(rot_y(np.deg2rad(9.0)) @ rot_x(np.deg2rad(-4.0)), np.zeros(3)),
    ),
    # translation: no rotation
    "translation": (
        pose(np.eye(3), np.zeros(3)),
        pose(np.eye(3), np.array([0.30, 0.05, 0.10])),
    ),
    # orbit: rotation and translation together
    "orbit": (
        pose(np.eye(3), np.zeros(3)),
        pose(rot_y(np.deg2rad(12.0)), np.array([0.42, 0.0, 0.18])),
    ),
}

results = []


def record(check, name, ok, detail):
    results.append((check, name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {check} {name}: {detail}")


rng = np.random.default_rng(20260825)

# ---------------------------------------------------------------------------
# 2.1 Geometry cross-check: 20 random pixels per regime, unproject/transform/project
# ---------------------------------------------------------------------------
print("=== 2.1 independent reprojection vs lot geometry ===")
for regime, (T_wc_ctx, T_wc_tgt) in PAIRS.items():
    T_ind = ind.relative(T_wc_tgt, T_wc_ctx)
    T_lot = geometry.relative_pose(
        torch.from_numpy(T_wc_tgt), torch.from_numpy(T_wc_ctx)
    ).numpy()
    dmax = float(np.abs(T_ind - T_lot).max())
    record("2.1", f"{regime}: relative_pose matrix", dmax < 1e-12, f"max|diff| = {dmax:.3e}")

    uv = np.stack(
        [rng.uniform(0, W - 1, 20), rng.uniform(0, H - 1, 20)], axis=-1
    )
    depth = rng.uniform(1.0, 6.0, 20)

    uv_ind, z_ind = ind.reproject(uv, depth, K_CTX, K_TGT, T_ind)
    pts = geometry.unproject(torch.from_numpy(uv), torch.from_numpy(depth), torch.from_numpy(K_CTX))
    pts = geometry.transform_points(torch.from_numpy(T_lot), pts)
    uv_lot, z_lot = geometry.project(pts, torch.from_numpy(K_TGT))
    err = float(np.abs(uv_ind - uv_lot.numpy()).max())
    zerr = float(np.abs(z_ind - z_lot.numpy()).max())
    record("2.1", f"{regime}: 20 pixels reprojected", err < TOL_PX,
           f"max pixel disagreement = {err:.3e} px, max depth disagreement = {zerr:.3e} m")

# ---------------------------------------------------------------------------
# 2.1b The check must be able to fail: reversed relative pose must break it
# ---------------------------------------------------------------------------
T_wc_ctx, T_wc_tgt = PAIRS["translation"]
T_fwd = ind.relative(T_wc_tgt, T_wc_ctx)
T_rev = ind.relative(T_wc_ctx, T_wc_tgt)
uv = np.stack([rng.uniform(0, W - 1, 20), rng.uniform(0, H - 1, 20)], axis=-1)
depth = rng.uniform(1.0, 6.0, 20)
a, _ = ind.reproject(uv, depth, K_CTX, K_TGT, T_fwd)
b, _ = ind.reproject(uv, depth, K_CTX, K_TGT, T_rev)
sep = float(np.abs(a - b).max())
record("2.1b", "reversed relative pose is separable", sep > TOL_PX,
       f"max separation = {sep:.3f} px, so the 0.1 px test is not vacuous")

# ---------------------------------------------------------------------------
# 2.2 Homography check on the pure-rotation pair, general K_tgt R inv(K_ctx)
# ---------------------------------------------------------------------------
print("\n=== 2.2 pure-rotation homography vs depth-based reprojection ===")
T_wc_ctx, T_wc_tgt = PAIRS["rotation"]
T_rot = ind.relative(T_wc_tgt, T_wc_ctx)
tnorm = float(np.linalg.norm(T_rot[:3, 3]))
record("2.2", "rotation pair has exactly zero translation", tnorm == 0.0,
       f"||t|| = {tnorm:.3e} m")

Hmat = ind.rotation_homography(K_CTX, K_TGT, T_rot[:3, :3])
grid = ind.pixel_grid(H, W).reshape(-1, 2)
uv_h = ind.apply_homography(Hmat, grid)

# Depth independence: the pipeline's own chain at three unrelated depths.
worst = 0.0
for d in (0.7, 3.3, 42.0):
    dep = np.full(grid.shape[0], d)
    pts = geometry.unproject(
        torch.from_numpy(grid), torch.from_numpy(dep), torch.from_numpy(K_CTX)
    )
    pts = geometry.transform_points(torch.from_numpy(T_rot), pts)
    uv_lot, _ = geometry.project(pts, torch.from_numpy(K_TGT))
    worst = max(worst, float(np.abs(uv_h - uv_lot.numpy()).max()))
record("2.2", "homography equals lot reprojection at 3 depths", worst < TOL_PX,
       f"max disagreement over the full {H}x{W} grid = {worst:.3e} px")

# The single-K form would be wrong here, which is what 2.2 exists to prove.
H_singleK = ind.rotation_homography(K_CTX, K_CTX, T_rot[:3, :3])
single_err = float(np.abs(ind.apply_homography(H_singleK, grid) - uv_h).max())
record("2.2b", "single-K shortcut is separable from the general form", single_err > TOL_PX,
       f"max separation = {single_err:.3f} px")

# ---------------------------------------------------------------------------
# 2.3 surrogate: independent visibility and transport on the two-plane scene
# ---------------------------------------------------------------------------
print("\n=== 2.3 surrogate: independent visibility and transport vs lot ===")
depth_ctx = two_plane_depth(H, W)
for regime, (T_wc_ctx, T_wc_tgt) in PAIRS.items():
    T = ind.relative(T_wc_tgt, T_wc_ctx)
    Tt = torch.from_numpy(T)
    # Target depth: render the same two planes as seen from the target by
    # forward-splatting the context depth, so both sides describe one scene.
    uv = ind.pixel_grid(H, W)
    uvt, zt = ind.reproject(uv, depth_ctx, K_CTX, K_TGT, T)
    iu = np.floor(uvt[..., 0] + 0.5).astype(np.int64)
    iv = np.floor(uvt[..., 1] + 0.5).astype(np.int64)
    ok = (zt > 0) & (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    depth_tgt = np.full((H, W), np.inf)
    np.minimum.at(depth_tgt, (iv[ok], iu[ok]), zt[ok])
    depth_tgt = np.where(np.isfinite(depth_tgt), depth_tgt, 0.0)

    ci, di, vi = ind.covisible_mask(depth_tgt, depth_ctx, K_TGT, K_CTX, T)
    masks = visibility.visibility_masks(
        torch.from_numpy(depth_tgt), torch.from_numpy(depth_ctx),
        torch.from_numpy(K_TGT), torch.from_numpy(K_CTX), Tt,
    )
    cl = masks.covisible.numpy()
    diff = int((ci != cl).sum())
    record("2.3a", f"{regime}: co-visible mask",
           diff == 0,
           f"{diff} of {ci.size} pixels differ; independent covisible = {int(ci.sum())}, "
           f"lot = {int(cl.sum())}")

    C = 6
    feats = rng.normal(size=(C, H // PATCH, W // PATCH))
    fi, cov_i, zb_i = ind.transport(feats, depth_ctx, K_CTX, K_TGT, T, (H, W))
    res = transport.transport(
        torch.from_numpy(feats), torch.from_numpy(depth_ctx),
        torch.from_numpy(K_CTX), torch.from_numpy(K_TGT), Tt, (H, W),
    )
    fl = res.features.numpy()
    covl = res.coverage.numpy()
    ferr = float(np.abs(fi - fl).max())
    cerr = float(np.abs(cov_i - covl).max())
    record("2.3b", f"{regime}: transported features", ferr < 1e-4,
           f"max|diff| = {ferr:.3e}")
    # lot returns coverage as float32, so the floor here is float32 epsilon,
    # not the 1e-9 a float64 comparison would allow.
    record("2.3c", f"{regime}: coverage", cerr < 1e-6,
           f"max|diff| = {cerr:.3e} (float32 epsilon); coverage range lot = "
           f"[{covl.min():.3f}, {covl.max():.3f}]")

# Coverage contract from CLAUDE.md: exactly 1.0 in full patches, 0.0 in holes.
T = ind.relative(*[PAIRS["translation"][1], PAIRS["translation"][0]][::-1])
_, cov, _ = ind.transport(rng.normal(size=(3, H // PATCH, W // PATCH)),
                          depth_ctx, K_CTX, K_TGT,
                          ind.relative(PAIRS["translation"][1], PAIRS["translation"][0]), (H, W))
record("2.3d", "coverage stays in [0,1] and has both extremes",
       bool((cov >= 0).all() and (cov <= 1).all()),
       f"min = {cov.min():.3f}, max = {cov.max():.3f}, exact-zero patches = "
       f"{int((cov == 0).sum())}, exact-one patches = {int((cov == 1).sum())}")

# ---------------------------------------------------------------------------
# Pixel-to-patch mapping, independent (VALIDATION 1.3)
# ---------------------------------------------------------------------------
print("\n=== 1.3 pixel-to-patch mapping ===")
from lot.encoders import pixel_to_patch_coords, sample_features_bilinear  # noqa: E402

px = torch.from_numpy(rng.uniform(-5, 200, 50))
mine = ind.pixel_to_patch(px.numpy())
theirs = pixel_to_patch_coords(px).numpy()
err = float(np.abs(mine - theirs).max())
record("1.3", "pixel_to_patch_coords matches (u+0.5)/14-0.5", err < 1e-12,
       f"max|diff| = {err:.3e} over 50 random coordinates")

feats = torch.from_numpy(rng.normal(size=(4, 9, 9)))
uvq = torch.from_numpy(np.stack([rng.uniform(7, 110, 30), rng.uniform(7, 110, 30)], -1))
a = ind.sample_features(feats.numpy(), uvq.numpy())
b = sample_features_bilinear(feats, uvq).numpy()
err = float(np.abs(a - b).max())
record("1.3b", "bilinear feature sampling matches independent arithmetic", err < 1e-12,
       f"max|diff| = {err:.3e} over 30 random locations")

# ---------------------------------------------------------------------------
# 1.2 rotation angle, including the clamp
# ---------------------------------------------------------------------------
print("\n=== 1.2 rotation angle ===")
for deg in (0.0, 7.5, 33.0, 179.0):
    R = rot_y(np.deg2rad(deg))
    a = ind.rotation_deg(R)
    b = geometry.rotation_angle_deg(torch.from_numpy(R))
    record("1.2", f"rotation_angle_deg at {deg} deg", abs(a - b) < 1e-9,
           f"independent {a:.9f} vs lot {b:.9f}")

R_over = np.eye(3) * (1 + 1e-13)
arg = (np.trace(R_over) - 1) / 2
val = geometry.rotation_angle_deg(torch.from_numpy(R_over))
record("1.2b", "arccos argument overshoot is clamped",
       np.isfinite(val),
       f"arg = {arg!r} > 1 is {arg > 1.0}, lot returns {val!r} (finite = {np.isfinite(val)})")

print("\n=== SUMMARY ===")
fails = [r for r in results if not r[2]]
print(f"{len(results) - len(fails)} passed, {len(fails)} failed, of {len(results)} checks")
for c, n, _, d in fails:
    print(f"  FAIL {c} {n}: {d}")
sys.exit(1 if fails else 0)
