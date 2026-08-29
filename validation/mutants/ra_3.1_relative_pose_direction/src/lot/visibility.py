"""Ground-truth co-visibility by z-buffer reprojection.

Each valid target pixel is lifted to 3D with the target depth map, mapped into
the context camera, and compared against the context depth map at the projected
location. A rendered depth map is that camera's z-buffer, so the comparison asks
whether the context camera saw the same surface point.

Definitions, fixed by CLAUDE.md:
- Co-visible: the context camera sees the target surface point within a relative
  depth tolerance (default 1.5 percent).
- Disoccluded: a valid target pixel that is not co-visible. Pixels whose surface
  point falls outside the context frustum count as disoccluded.

Visibility buckets always come from ground-truth geometry of both views.
Estimated depth is never the referee.

The context depth map is read with nearest-cell sampling, not bilinear. A depth
map is a z-buffer: cell (v, u) holds the depth of the surface along the ray
through that pixel center. Interpolating between two cells across a depth edge
invents a depth that lies on no surface, and the invented value both rejects
background points the context camera really sees and accepts points it never
saw. The band this corrupts is about one pixel wide along every occlusion edge,
which is a large fraction of the disocclusion strip at small parallax. Nearest
sampling answers the question the definition actually asks: did the context
pixel that images this point record this depth. The cost is that on a strongly
slanted surface the half-pixel quantization shifts the compared depth by half a
local gradient step, which stays well inside the tolerance for the depth ranges
in this study.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from .encoders import sample_map_nearest
from .geometry import (
    common_dtype,
    invert_se3,
    pixel_grid,
    project,
    transform_points,
    unproject,
)

# The tolerance itself lives in configs/analysis.yaml, which PROTOCOL's preamble
# makes part of the protocol. Resolved lazily rather than copied here, so a
# change to the normative value cannot leave a stale literal behind.
def default_relative_depth_tol() -> float:
    from .analysis_config import load_analysis_config

    return load_analysis_config().covisible_relative_depth_tol

# Invalid context depths are replaced by this large finite value before sampling.
# A sample that reads an invalid entry fails the depth tolerance test, so pixels
# landing on depth holes are conservatively not co-visible.
_INVALID_DEPTH_SENTINEL = 1e30


class VisibilityMasks(NamedTuple):
    covisible: Tensor           # [H, W] bool, target grid
    disoccluded: Tensor         # [H, W] bool, valid and not co-visible
    valid: Tensor               # [H, W] bool, target depth finite and positive
    in_context_frustum: Tensor  # [H, W] bool, projects inside the context image with z > 0
    uv_in_context: Tensor       # [H, W, 2] continuous pixel coordinates in the context image
    z_in_context: Tensor        # [H, W] depth of the target surface point in the context camera


def visibility_masks(
    depth_target: Tensor,
    depth_context: Tensor,
    K_target: Tensor,
    K_context: Tensor,
    T_target_from_context: Tensor,
    rel_tol: float | None = None,
) -> VisibilityMasks:
    """Compute per-pixel co-visible and disoccluded masks on the target grid.

    depth_target, depth_context: [H, W] ground-truth planar z-depth in meters.
    K_target, K_context: 3x3 intrinsics.
    T_target_from_context: the canonical relative transform from geometry.relative_pose.
    rel_tol: relative depth tolerance for the co-visibility test. None reads the
        normative value from the analysis config.
    """
    if rel_tol is None:
        rel_tol = default_relative_depth_tol()
    if depth_target.dim() != 2 or depth_context.dim() != 2:
        raise ValueError("depth maps must be [H, W]")
    T_context_from_target = invert_se3(T_target_from_context)
    height, width = depth_target.shape
    ctx_height, ctx_width = depth_context.shape
    dtype = common_dtype(depth_target, depth_context, K_target, K_context, T_target_from_context)
    device = depth_target.device

    uv = pixel_grid(height, width, dtype=dtype, device=device)
    valid = (depth_target > 0) & torch.isfinite(depth_target)
    points_target = unproject(uv, depth_target.to(dtype), K_target)
    points_context = transform_points(T_context_from_target, points_target)
    uv_context, z_context = project(points_context, K_context)

    u = uv_context[..., 0]
    v = uv_context[..., 1]
    # Bounds are the physical image extent, not the pixel-center box: pixel W - 1
    # spans up to W - 0.5, and the context camera images everything inside that.
    # This is the same rule transport.py applies when it decides which target
    # pixel a splat lands on, so the two modules agree on the border.
    in_frustum = (
        valid
        & (z_context > 0)
        & torch.isfinite(u)
        & torch.isfinite(v)
        & (u >= -0.5)
        & (u < ctx_width - 0.5)
        & (v >= -0.5)
        & (v < ctx_height - 0.5)
    )

    d = depth_context.to(dtype)
    d = torch.where(
        (d > 0) & torch.isfinite(d),
        d,
        torch.full_like(d, _INVALID_DEPTH_SENTINEL),
    )
    safe_uv = torch.where(in_frustum[..., None], uv_context, torch.zeros_like(uv_context))
    depth_seen = sample_map_nearest(d, safe_uv)
    depth_agrees = (z_context - depth_seen).abs() <= rel_tol * z_context

    covisible = in_frustum & depth_agrees
    disoccluded = valid & ~covisible
    return VisibilityMasks(covisible, disoccluded, valid, in_frustum, uv_context, z_context)


def fraction_per_patch(mask: Tensor, patch_size: int) -> Tensor:
    """Fraction of set pixels within each patch.

    mask: [H, W] bool or float, with H and W divisible by patch_size.
    Returns [H / patch_size, W / patch_size] float32.
    """
    height, width = mask.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"mask shape {(height, width)} not divisible by patch_size {patch_size}")
    m = mask.to(torch.float32).reshape(
        height // patch_size, patch_size, width // patch_size, patch_size
    )
    return m.mean(dim=(1, 3))
