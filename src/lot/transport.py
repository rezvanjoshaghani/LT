"""Transport-Only feature reprojection.

Pixel-level forward splat with a z-buffer, pooled to the patch grid. No training,
no learned parameters.

Contract from CLAUDE.md:
transport(features_ctx, depth_ctx_px, K_ctx, K_tgt, T_tgt_from_ctx, out_hw_px)
returns (features_tgt [C, Hp, Wp], coverage [Hp, Wp] in [0, 1], zbuffer).

Steps:
1. Lift every valid context pixel with its planar depth.
2. Map it into the target camera and splat it onto the nearest target pixel,
   carrying the feature of the patch that contains the source pixel.
3. Resolve occlusion with a z-buffer at pixel level. Splats whose depth ties the
   pixel minimum within a small relative epsilon are averaged, which makes exact
   ties deterministic.
4. Average hit pixels into target patches. Empty patches stay zero.

Coverage is the fraction of a target patch's pixels that received support.
Disoccluded regions stay empty. The z-buffer is +inf where no splat landed.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from .encoders import PATCH_SIZE
from .geometry import (
    check_intrinsics,
    common_dtype,
    pixel_grid,
    project,
    transform_points,
    unproject,
)

# Relative depth epsilon for z-buffer ties. Splats within this factor of the
# per-pixel minimum depth count as winners and are averaged.
TIE_RELATIVE_EPS = 1e-6


class TransportResult(NamedTuple):
    features: Tensor  # [C, Hp_out, Wp_out] float32, zeros where no support
    coverage: Tensor  # [Hp_out, Wp_out] float32 in [0, 1]
    zbuffer: Tensor   # [H_out, W_out], +inf where no splat landed


def transport(
    features_ctx: Tensor,
    depth_ctx_px: Tensor,
    K_ctx: Tensor,
    K_tgt: Tensor,
    T_tgt_from_ctx: Tensor,
    out_hw_px: tuple[int, int],
    patch_size: int = PATCH_SIZE,
) -> TransportResult:
    """Reproject context patch features into the target camera.

    features_ctx: [C, Hp, Wp] patch-grid features of the context view.
    depth_ctx_px: [H, W] planar z-depth of the context view in meters, at pixel
        resolution, with H = Hp * patch_size and W = Wp * patch_size. Entries that
        are not finite and positive are skipped.
    K_ctx, K_tgt: 3x3 intrinsics of the context and target cameras.
    T_tgt_from_ctx: the canonical relative transform from geometry.relative_pose.
    out_hw_px: (H_out, W_out) target size in pixels, both divisible by patch_size.
    Every input is moved to the device of features_ctx, which is the device the
    result is returned on.
    Returns float32 features and coverage. The z-buffer keeps the geometry dtype.
    """
    check_intrinsics(K_ctx)
    check_intrinsics(K_tgt)
    if features_ctx.dim() != 3:
        raise ValueError(f"features_ctx must be [C, Hp, Wp], got {tuple(features_ctx.shape)}")
    if depth_ctx_px.dim() != 2:
        raise ValueError(f"depth_ctx_px must be [H, W], got {tuple(depth_ctx_px.shape)}")
    channels, patches_h, patches_w = features_ctx.shape
    height, width = depth_ctx_px.shape
    if patches_h * patch_size != height or patches_w * patch_size != width:
        raise ValueError(
            f"depth {(height, width)} does not match features {(patches_h, patches_w)} "
            f"at patch_size {patch_size}"
        )
    out_height, out_width = out_hw_px
    if out_height % patch_size or out_width % patch_size:
        raise ValueError(f"out_hw_px {out_hw_px} not divisible by patch_size {patch_size}")
    out_patches_h = out_height // patch_size
    out_patches_w = out_width // patch_size

    dtype = common_dtype(depth_ctx_px, K_ctx, K_tgt, T_tgt_from_ctx)
    device = features_ctx.device
    K_ctx = K_ctx.to(device)
    K_tgt = K_tgt.to(device)
    T_tgt_from_ctx = T_tgt_from_ctx.to(device)

    uv = pixel_grid(height, width, dtype=dtype, device=device)
    z = depth_ctx_px.to(device=device, dtype=dtype)
    points_ctx = unproject(uv, z, K_ctx)
    points_tgt = transform_points(T_tgt_from_ctx, points_ctx)
    uv_tgt, z_tgt = project(points_tgt, K_tgt)

    keep = (
        (z > 0)
        & torch.isfinite(z)
        & (z_tgt > 0)
        & torch.isfinite(z_tgt)
        & torch.isfinite(uv_tgt).all(dim=-1)
    )
    safe_uv = torch.where(keep[..., None], uv_tgt, torch.zeros_like(uv_tgt))
    iu = torch.floor(safe_uv[..., 0] + 0.5).long()
    iv = torch.floor(safe_uv[..., 1] + 0.5).long()
    keep &= (iu >= 0) & (iu < out_width) & (iv >= 0) & (iv < out_height)

    keep_flat = keep.reshape(-1)
    lin = (iv * out_width + iu).reshape(-1)[keep_flat]
    z_keep = z_tgt.reshape(-1)[keep_flat]

    zbuffer = torch.full((out_height * out_width,), torch.inf, dtype=dtype, device=device)
    zbuffer.scatter_reduce_(0, lin, z_keep, reduce="amin", include_self=True)
    winners = z_keep <= zbuffer[lin] * (1 + TIE_RELATIVE_EPS)
    lin_w = lin[winners]

    patch_row = (torch.arange(height, device=device) // patch_size)[:, None].expand(height, width)
    patch_col = (torch.arange(width, device=device) // patch_size)[None, :].expand(height, width)
    row_w = patch_row.reshape(-1)[keep_flat][winners]
    col_w = patch_col.reshape(-1)[keep_flat][winners]
    feats_w = features_ctx.to(torch.float32)[:, row_w, col_w]

    accum = torch.zeros((channels, out_height * out_width), dtype=torch.float32, device=device)
    accum.index_add_(1, lin_w, feats_w)
    count = torch.zeros((out_height * out_width,), dtype=torch.float32, device=device)
    count.index_add_(0, lin_w, torch.ones_like(lin_w, dtype=torch.float32))
    hit = count > 0
    features_px = accum / count.clamp(min=1)

    feature_sums = features_px.reshape(
        channels, out_patches_h, patch_size, out_patches_w, patch_size
    ).sum(dim=(2, 4))
    hits_per_patch = hit.to(torch.float32).reshape(
        out_patches_h, patch_size, out_patches_w, patch_size
    ).sum(dim=(1, 3))
    features_out = feature_sums / hits_per_patch.clamp(min=1)
    coverage = hits_per_patch / float(patch_size * patch_size)

    return TransportResult(features_out, coverage, zbuffer.reshape(out_height, out_width))
