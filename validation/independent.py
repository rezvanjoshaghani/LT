"""Independent reimplementation of the quantities PROTOCOL.md defines.

Written from PROTOCOL.md, CLAUDE.md, and VALIDATION.md text alone. This module
imports nothing from the lot package. It exists so that lot answers can be
checked against a second implementation rather than against themselves.

Conventions taken from the protocol text:
- OpenCV pinhole, x right, y down, z forward. K is 3x3, zero skew.
- Poses stored as T_world_from_camera (4x4).
- T_target_from_context = inv(T_world_from_target) @ T_world_from_context.
- Depth is planar z-depth in meters.
- Pixel (u, v) maps to patch ((u + 0.5) / 14 - 0.5, same for v).
  That mapping puts pixel centers at integer pixel coordinates: patch p spans
  pixels 14p .. 14p+13, whose center is 14p + 6.5, and the mapping sends
  14p + 6.5 to p exactly.
"""

from __future__ import annotations

import numpy as np

PATCH = 14


# --- pose algebra ----------------------------------------------------------

def invert_pose(T):
    """Inverse of a 4x4 rigid transform, by the block formula."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def relative(T_world_from_target, T_world_from_context):
    """T_target_from_context, per PROTOCOL/CLAUDE.md, using a general inverse."""
    return np.linalg.inv(np.asarray(T_world_from_target, dtype=np.float64)) @ np.asarray(
        T_world_from_context, dtype=np.float64
    )


def rotation_deg(R):
    """arccos(clip((trace(R) - 1) / 2, -1, 1)) in degrees. PROTOCOL 3.2."""
    c = (np.trace(np.asarray(R, dtype=np.float64)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# --- projective geometry ---------------------------------------------------

def unproject(uv, depth, K):
    """Lift pixels with planar z-depth to camera-frame points. uv is [..., 2]."""
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uv[..., 0] - cx) * depth / fx
    y = (uv[..., 1] - cy) * depth / fy
    return np.stack((x, y, depth), axis=-1)


def transform(T, pts):
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def project(pts, K):
    """Project camera-frame points. Returns (uv [..., 2], z [...])."""
    pts = np.asarray(pts, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = pts[..., 2]
    u = fx * pts[..., 0] / z + cx
    v = fy * pts[..., 1] / z + cy
    return np.stack((u, v), axis=-1), z


def reproject(uv, depth, K_src, K_dst, T_dst_from_src):
    """Full unproject then transform then project chain."""
    return project(transform(T_dst_from_src, unproject(uv, depth, K_src)), K_dst)


def rotation_homography(K_ctx, K_tgt, R_tgt_from_ctx):
    """K_tgt @ R @ inv(K_ctx). General form: never assumes K_ctx equals K_tgt."""
    return (
        np.asarray(K_tgt, dtype=np.float64)
        @ np.asarray(R_tgt_from_ctx, dtype=np.float64)
        @ np.linalg.inv(np.asarray(K_ctx, dtype=np.float64))
    )


def apply_homography(H, uv):
    uv = np.asarray(uv, dtype=np.float64)
    ones = np.ones(uv.shape[:-1] + (1,), dtype=np.float64)
    p = np.concatenate((uv, ones), axis=-1) @ np.asarray(H, dtype=np.float64).T
    return p[..., :2] / p[..., 2:3]


def pixel_grid(h, w):
    """[h, w, 2] of (u, v) with pixel centers at integers."""
    v, u = np.meshgrid(
        np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij"
    )
    return np.stack((u, v), axis=-1)


# --- patch grid sampling ---------------------------------------------------

def pixel_to_patch(uv_px, patch=PATCH):
    return (np.asarray(uv_px, dtype=np.float64) + 0.5) / patch - 0.5


def patch_to_pixel(uv_patch, patch=PATCH):
    return (np.asarray(uv_patch, dtype=np.float64) + 0.5) * patch - 0.5


def sample_bilinear(grid, xy):
    """Bilinear on a [C,H,W] or [H,W] grid, cell centers at integers, border clamped."""
    grid = np.asarray(grid, dtype=np.float64)
    squeeze = grid.ndim == 2
    g = grid[None] if squeeze else grid
    _, H, W = g.shape
    xy = np.asarray(xy, dtype=np.float64)
    x = np.clip(xy[..., 0], 0, W - 1)
    y = np.clip(xy[..., 1], 0, H - 1)
    x0 = np.clip(np.floor(x), 0, max(W - 2, 0)).astype(np.int64)
    y0 = np.clip(np.floor(y), 0, max(H - 2, 0)).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    wx = x - x0
    wy = y - y0
    out = (
        g[:, y0, x0] * (1 - wx) * (1 - wy)
        + g[:, y0, x1] * wx * (1 - wy)
        + g[:, y1, x0] * (1 - wx) * wy
        + g[:, y1, x1] * wx * wy
    )
    out = np.moveaxis(out, 0, -1)
    return out[..., 0] if squeeze else out


def sample_features(features, uv_px, patch=PATCH):
    return sample_bilinear(features, pixel_to_patch(uv_px, patch))


# --- visibility ------------------------------------------------------------

def covisible_mask(depth_tgt, depth_ctx, K_tgt, K_ctx, T_tgt_from_ctx, rel_tol=0.015):
    """Ground-truth co-visibility on the target grid.

    A valid target pixel is co-visible when its surface point, mapped into the
    context camera, lands inside the context image with positive depth and the
    context z-buffer at that cell records the same depth to within rel_tol.
    The context depth map is read with nearest-cell sampling: it is a z-buffer,
    so interpolating across an edge invents a depth that lies on no surface.
    """
    H, W = depth_tgt.shape
    Hc, Wc = depth_ctx.shape
    T_ctx_from_tgt = np.linalg.inv(np.asarray(T_tgt_from_ctx, dtype=np.float64))
    uv = pixel_grid(H, W)
    valid = (depth_tgt > 0) & np.isfinite(depth_tgt)
    uv_c, z_c = reproject(uv, depth_tgt, K_tgt, K_ctx, T_ctx_from_tgt)
    u, v = uv_c[..., 0], uv_c[..., 1]
    in_frustum = (
        valid & (z_c > 0) & np.isfinite(u) & np.isfinite(v)
        & (u >= -0.5) & (u < Wc - 0.5) & (v >= -0.5) & (v < Hc - 0.5)
    )
    iu = np.clip(np.floor(np.where(in_frustum, u, 0.0) + 0.5).astype(np.int64), 0, Wc - 1)
    iv = np.clip(np.floor(np.where(in_frustum, v, 0.0) + 0.5).astype(np.int64), 0, Hc - 1)
    d = np.where((depth_ctx > 0) & np.isfinite(depth_ctx), depth_ctx, 1e30)
    seen = d[iv, iu]
    agrees = np.abs(z_c - seen) <= rel_tol * z_c
    covis = in_frustum & agrees
    return covis, valid & ~covis, valid


# --- transport: pixel splat, z-buffer, pool to patches ---------------------

def transport(features_ctx, depth_ctx, K_ctx, K_tgt, T_tgt_from_ctx, out_hw, patch=PATCH):
    """Forward splat at pixel resolution, z-buffered, pooled to the patch grid.

    Returns (features [C,Hp,Wp], coverage [Hp,Wp], zbuffer [H_out,W_out]).
    Coverage is the fraction of a target patch pixels that received support.
    Empty patches stay zero.
    """
    C, Hp, Wp = features_ctx.shape
    H, W = depth_ctx.shape
    Ho, Wo = out_hw
    Hpo, Wpo = Ho // patch, Wo // patch
    uv = pixel_grid(H, W)
    uv_t, z_t = reproject(uv, depth_ctx, K_ctx, K_tgt, T_tgt_from_ctx)
    keep = (
        (depth_ctx > 0) & np.isfinite(depth_ctx) & (z_t > 0) & np.isfinite(z_t)
        & np.isfinite(uv_t).all(axis=-1)
    )
    iu = np.floor(np.where(keep, uv_t[..., 0], 0.0) + 0.5).astype(np.int64)
    iv = np.floor(np.where(keep, uv_t[..., 1], 0.0) + 0.5).astype(np.int64)
    keep = keep & (iu >= 0) & (iu < Wo) & (iv >= 0) & (iv < Ho)

    lin = (iv * Wo + iu)[keep]
    z = z_t[keep]
    py = (np.arange(H) // patch)[:, None] * Wp
    px = (np.arange(W) // patch)[None, :]
    src = (py + px)[keep]

    zbuf = np.full(Ho * Wo, np.inf)
    np.minimum.at(zbuf, lin, z)
    winners = z <= zbuf[lin] * (1 + 1e-6)
    lin_w, src_w = lin[winners], src[winners]

    count = np.zeros(Ho * Wo)
    np.add.at(count, lin_w, 1.0)
    hit = count > 0
    hits_per_patch = hit.reshape(Hpo, patch, Wpo, patch).sum(axis=(1, 3)).astype(np.float64)

    tgt_patch = (lin_w // Wo // patch) * Wpo + (lin_w % Wo) // patch
    Wmat = np.zeros((Hpo * Wpo, Hp * Wp))
    np.add.at(Wmat, (tgt_patch, src_w), 1.0 / count[lin_w])
    Wmat = Wmat / np.maximum(hits_per_patch.reshape(-1, 1), 1.0)

    flat = features_ctx.reshape(C, -1).astype(np.float64)
    out = (flat @ Wmat.T).reshape(C, Hpo, Wpo)
    return out, hits_per_patch / float(patch * patch), zbuf.reshape(Ho, Wo)


# --- scoring ---------------------------------------------------------------

def unit(x, eps=1e-12):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def cosine(pred, tgt, center=None):
    """Mean cosine after optional centering. PROTOCOL 3.7."""
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(tgt, dtype=np.float64)
    if center is not None:
        a = a - center
        b = b - center
    return float((unit(a) * unit(b)).sum(axis=-1).mean())


def l2(pred, tgt, center=None):
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(tgt, dtype=np.float64)
    if center is not None:
        a = a - center
        b = b - center
    return float(np.linalg.norm(unit(a) - unit(b), axis=-1).mean())
