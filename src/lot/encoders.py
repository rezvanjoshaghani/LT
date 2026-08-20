"""Feature-grid coordinate mapping and sampling.

Phase 0 ships only pure sampling utilities. The frozen encoder wrappers (DINOv2,
VGGT) and the feature cache arrive in Phase 2 and must reuse these functions.

The pixel-to-patch coordinate mapping is defined here once for the whole
repository: pixel (u, v) maps to patch coordinates ((u + 0.5) / patch_size - 0.5,
same for v). Integer patch coordinates are patch centers. The center of patch p
sits at pixel coordinate patch_size * p + (patch_size - 1) / 2.
"""

from __future__ import annotations

import torch
from torch import Tensor

PATCH_SIZE = 14


def pixel_to_patch_coords(uv_px: Tensor, patch_size: int = PATCH_SIZE) -> Tensor:
    """Map continuous pixel coordinates to continuous patch-grid coordinates.

    Pixel (u, v) maps to ((u + 0.5) / patch_size - 0.5, (v + 0.5) / patch_size - 0.5).
    """
    return (uv_px + 0.5) / patch_size - 0.5


def patch_to_pixel_coords(uv_patch: Tensor, patch_size: int = PATCH_SIZE) -> Tensor:
    """Inverse of pixel_to_patch_coords."""
    return (uv_patch + 0.5) * patch_size - 0.5


def sample_map_bilinear(grid: Tensor, xy: Tensor) -> Tensor:
    """Bilinear sampling on a regular grid whose cell centers sit at integer coordinates.

    grid: [H, W] or [C, H, W].
    xy: [..., 2] continuous coordinates (x along width, y along height), in grid units.
    Coordinates outside the grid are clamped to the border.
    Returns [...] for a 2D grid and [..., C] for a 3D grid.
    """
    squeeze = grid.dim() == 2
    g = grid[None] if squeeze else grid
    if g.dim() != 3:
        raise ValueError(f"grid must be [H, W] or [C, H, W], got {tuple(grid.shape)}")
    channels, height, width = g.shape
    dtype = torch.promote_types(g.dtype, xy.dtype)
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    g = g.to(dtype)
    x = xy[..., 0].to(dtype).clamp(0, width - 1)
    y = xy[..., 1].to(dtype).clamp(0, height - 1)
    x0 = x.floor().clamp(max=max(width - 2, 0)).long()
    y0 = y.floor().clamp(max=max(height - 2, 0)).long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    wx = x - x0
    wy = y - y0
    v00 = g[:, y0, x0]
    v01 = g[:, y0, x1]
    v10 = g[:, y1, x0]
    v11 = g[:, y1, x1]
    out = (
        v00 * (1 - wx) * (1 - wy)
        + v01 * wx * (1 - wy)
        + v10 * (1 - wx) * wy
        + v11 * wx * wy
    )
    out = torch.movedim(out, 0, -1)
    return out[..., 0] if squeeze else out


def sample_features_bilinear(features: Tensor, uv_px: Tensor, patch_size: int = PATCH_SIZE) -> Tensor:
    """Sample a patch-grid feature map at continuous pixel coordinates.

    features: [C, Hp, Wp] patch-grid feature map.
    uv_px: [..., 2] pixel coordinates in the image the features were computed from.
    Uses the pixel-to-patch mapping defined above, then bilinear interpolation on
    the patch grid with border clamping.
    Returns [..., C].
    """
    return sample_map_bilinear(features, pixel_to_patch_coords(uv_px, patch_size))
