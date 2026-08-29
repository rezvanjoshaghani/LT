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

Steps 2 and 4 describe the semantics, not the arithmetic. Every pixel of a
source patch carries that patch's one feature vector, so the splat never has to
move a vector at all. It accumulates scalar weights from source patch to target
patch and mixes the features once at the end with a small matmul. Carrying
vectors per pixel would allocate a [C, H_out * W_out] buffer, which is 824 MB at
518 px with 768 channels and 2.2 GB with VGGT's 2048, several times over per
call. The weight matrix is [Hp_out * Wp_out, Hp * Wp], which is 7.5 MB at the
37 by 37 grids this study uses.
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


class TransportPlan(NamedTuple):
    """The part of a transport that depends only on geometry, not on features.

    weights [Hp_out * Wp_out, Hp * Wp] carries, for each target patch, how much
    of each source patch's feature it receives. Rows sum to one where anything
    landed and to zero in a hole. The plan is what makes comparing encoders
    cheap: two encoders over one pair share all of the geometry and differ only
    by a matmul against the same matrix.
    """

    weights: Tensor
    coverage: Tensor
    zbuffer: Tensor
    source_grid: tuple[int, int]
    target_grid: tuple[int, int]


def transport_plan(
    depth_ctx_px: Tensor,
    K_ctx: Tensor,
    K_tgt: Tensor,
    T_tgt_from_ctx: Tensor,
    out_hw_px: tuple[int, int],
    patch_size: int = PATCH_SIZE,
    device: torch.device | str | None = None,
) -> TransportPlan:
    """Work out where every context pixel lands and who wins each target pixel.

    Takes the same geometry arguments as transport and returns the reusable
    plan. The source patch grid is derived from the depth map, since a depth map
    is always a whole number of patches in this project.
    """
    check_intrinsics(K_ctx)
    check_intrinsics(K_tgt)
    if depth_ctx_px.dim() != 2:
        raise ValueError(f"depth_ctx_px must be [H, W], got {tuple(depth_ctx_px.shape)}")
    height, width = depth_ctx_px.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"depth {(height, width)} not divisible by patch_size {patch_size}")
    patches_h, patches_w = height // patch_size, width // patch_size
    out_height, out_width = out_hw_px
    if out_height % patch_size or out_width % patch_size:
        raise ValueError(f"out_hw_px {out_hw_px} not divisible by patch_size {patch_size}")
    out_patches_h = out_height // patch_size
    out_patches_w = out_width // patch_size

    dtype = common_dtype(depth_ctx_px, K_ctx, K_tgt, T_tgt_from_ctx)
    device = depth_ctx_px.device if device is None else torch.device(device)
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
    far = torch.full_like(zbuffer, -torch.inf)
    far.scatter_reduce_(0, lin, z_keep, reduce="amax", include_self=True)
    winners = z_keep >= far[lin] * (1 - TIE_RELATIVE_EPS)
    lin_w = lin[winners]

    source_patch = (
        (torch.arange(height, device=device) // patch_size)[:, None] * patches_w
        + (torch.arange(width, device=device) // patch_size)[None, :]
    )
    source_of_winner = source_patch.reshape(-1)[keep_flat][winners]

    count = torch.zeros((out_height * out_width,), dtype=torch.float32, device=device)
    count.index_add_(0, lin_w, torch.ones_like(lin_w, dtype=torch.float32))
    hit = count > 0
    hits_per_patch = hit.to(torch.float32).reshape(
        out_patches_h, patch_size, out_patches_w, patch_size
    ).sum(dim=(1, 3))

    # A target pixel splits its weight evenly between the splats that tie on it,
    # and a target patch averages over the pixels that received support. Both
    # divisions fold into the weight, so the matmul below reproduces the
    # pixel-level average exactly while never materializing it.
    target_patch = (lin_w // out_width // patch_size) * out_patches_w + (
        lin_w % out_width
    ) // patch_size
    num_source = patches_h * patches_w
    num_target = out_patches_h * out_patches_w
    weights = torch.zeros((num_target * num_source,), dtype=torch.float32, device=device)
    weights.index_add_(0, target_patch * num_source + source_of_winner, 1.0 / count[lin_w])
    weights = weights.reshape(num_target, num_source)
    weights = weights / hits_per_patch.reshape(num_target, 1).clamp(min=1)

    return TransportPlan(
        weights=weights,
        coverage=hits_per_patch / float(patch_size * patch_size),
        zbuffer=zbuffer.reshape(out_height, out_width),
        source_grid=(patches_h, patches_w),
        target_grid=(out_patches_h, out_patches_w),
    )


def apply_transport_plan(plan: TransportPlan, features_ctx: Tensor) -> Tensor:
    """Mix one context feature map through a plan. Returns [C, Hp_out, Wp_out] float32."""
    if features_ctx.dim() != 3:
        raise ValueError(f"features_ctx must be [C, Hp, Wp], got {tuple(features_ctx.shape)}")
    channels = features_ctx.shape[0]
    if tuple(features_ctx.shape[1:]) != plan.source_grid:
        raise ValueError(
            f"features grid {tuple(features_ctx.shape[1:])} does not match the plan's "
            f"source grid {plan.source_grid}"
        )
    flat = features_ctx.to(device=plan.weights.device, dtype=torch.float32).reshape(
        channels, -1
    )
    return (flat @ plan.weights.mT).reshape(channels, *plan.target_grid)


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

    This is the contract form. Callers transporting several feature maps through
    the same geometry should build one transport_plan and apply it to each.
    """
    if features_ctx.dim() != 3:
        raise ValueError(f"features_ctx must be [C, Hp, Wp], got {tuple(features_ctx.shape)}")
    plan = transport_plan(
        depth_ctx_px,
        K_ctx,
        K_tgt,
        T_tgt_from_ctx,
        out_hw_px,
        patch_size,
        device=features_ctx.device,
    )
    if tuple(features_ctx.shape[1:]) != plan.source_grid:
        raise ValueError(
            f"depth {tuple(depth_ctx_px.shape)} does not match features "
            f"{tuple(features_ctx.shape[1:])} at patch_size {patch_size}"
        )
    return TransportResult(
        apply_transport_plan(plan, features_ctx), plan.coverage, plan.zbuffer
    )
