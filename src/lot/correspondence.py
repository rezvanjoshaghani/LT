"""Experiment Zero correspondence sampling.

Samples target locations that are co-visible in the context view, computes the
ground-truth corresponding location in the context image with ground-truth
depth, and constructs the null locations. Values are read from patch-grid
feature maps with bilinear sampling.

Variants. Each is a prediction of the target feature at a sampled target location:
- warp: context features at the ground-truth corresponding location.
- no_warp: context features at the same pixel coordinates as the target location.
- neighbor: context features one patch away from the warp location, along a
  random in-bounds axis direction. Measures how sharply value agreement depends
  on landing exactly right.
- random: context features at a uniform random in-bounds location.
- mean: the mean feature vector of the context map. Location-free floor at the
  sampler level. The dataset-level Mean-Feature floor arrives with evaluate.py
  in a later phase.

Every location-bearing variant is kept inside the box where bilinear sampling on
the patch grid needs no border clamping. Candidates that would violate that box
are dropped before sampling.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from .encoders import (
    PATCH_SIZE,
    patch_to_pixel_coords,
    pixel_to_patch_coords,
    sample_features_bilinear,
    sample_map_bilinear,
)
from .geometry import (
    common_dtype,
    invert_se3,
    pixel_grid,
    project,
    transform_points,
    unproject,
)


class CorrespondenceSamples(NamedTuple):
    uv_target: Tensor            # [N, 2] pixel coordinates in the target image
    uv_context_warp: Tensor      # [N, 2] ground-truth correspondence in the context image
    uv_context_no_warp: Tensor   # [N, 2] same pixel coordinates as uv_target
    uv_context_neighbor: Tensor  # [N, 2] warp location offset by one patch
    uv_context_random: Tensor    # [N, 2] uniform random in-bounds location


def _sampling_box(hw_px: tuple[int, int], patch_size: int) -> tuple[float, float, float, float]:
    """Pixel-coordinate box where patch-grid bilinear sampling needs no clamping.

    Returns (u_min, u_max, v_min, v_max), all inclusive.
    """
    height, width = hw_px
    lo = 0.5 * patch_size - 0.5
    u_max = width - 0.5 * patch_size - 0.5
    v_max = height - 0.5 * patch_size - 0.5
    return lo, u_max, lo, v_max


def _in_box(uv: Tensor, box: tuple[float, float, float, float]) -> Tensor:
    u_min, u_max, v_min, v_max = box
    return (uv[..., 0] >= u_min) & (uv[..., 0] <= u_max) & (uv[..., 1] >= v_min) & (uv[..., 1] <= v_max)


def sample_correspondences(
    depth_target: Tensor,
    K_target: Tensor,
    K_context: Tensor,
    T_target_from_context: Tensor,
    covisible: Tensor,
    num_samples: int,
    context_hw_px: tuple[int, int],
    patch_size: int = PATCH_SIZE,
    mode: str = "pixel",
    generator: torch.Generator | None = None,
) -> CorrespondenceSamples:
    """Sample co-visible target locations with ground-truth warps and null locations.

    depth_target: [H, W] ground-truth planar z-depth of the target view, meters.
    K_target, K_context: 3x3 intrinsics.
    T_target_from_context: the canonical relative transform from geometry.relative_pose.
    covisible: [H, W] bool mask on the target grid, from visibility.visibility_masks.
    num_samples: number of samples requested. Fewer are returned if fewer candidates exist.
    context_hw_px: (H, W) of the context image in pixels.
    mode: "pixel" samples integer target pixel centers. "patch_center" samples target
        patch centers and keeps a candidate only if the four integer pixels around
        the center are all co-visible.
    generator: torch CPU generator for reproducible sampling.
    """
    if depth_target.shape != covisible.shape:
        raise ValueError("depth_target and covisible must have the same shape")
    height, width = depth_target.shape
    ctx_height, ctx_width = context_hw_px
    if min(height, width, ctx_height, ctx_width) < 2 * patch_size:
        raise ValueError("images must span at least two patches on each side")
    dtype = common_dtype(depth_target, K_target, K_context, T_target_from_context)
    device = depth_target.device

    if mode == "pixel":
        uv_cand = pixel_grid(height, width, dtype=dtype, device=device).reshape(-1, 2)
        keep = covisible.reshape(-1).clone()
    elif mode == "patch_center":
        patches_h = height // patch_size
        patches_w = width // patch_size
        centers = patch_to_pixel_coords(
            pixel_grid(patches_h, patches_w, dtype=dtype, device=device), patch_size
        ).reshape(-1, 2)
        x0 = centers[:, 0].floor().long()
        y0 = centers[:, 1].floor().long()
        keep = (
            covisible[y0, x0]
            & covisible[y0, x0 + 1]
            & covisible[y0 + 1, x0]
            & covisible[y0 + 1, x0 + 1]
        )
        uv_cand = centers
    else:
        raise ValueError(f"unknown mode {mode!r}")

    box_target = _sampling_box((height, width), patch_size)
    box_context = _sampling_box(context_hw_px, patch_size)
    keep &= _in_box(uv_cand, box_target)
    keep &= _in_box(uv_cand, box_context)

    uv_cand = uv_cand[keep]
    depth_at = sample_map_bilinear(depth_target.to(dtype), uv_cand)
    good_depth = (depth_at > 0) & torch.isfinite(depth_at)
    uv_cand = uv_cand[good_depth]
    depth_at = depth_at[good_depth]

    T_context_from_target = invert_se3(T_target_from_context.to(dtype))
    points_target = unproject(uv_cand, depth_at, K_target)
    points_context = transform_points(T_context_from_target, points_target)
    uv_warp, z_context = project(points_context, K_context)
    good_warp = (z_context > 0) & _in_box(uv_warp, box_context)
    uv_cand = uv_cand[good_warp]
    uv_warp = uv_warp[good_warp]

    count = uv_cand.shape[0]
    take = min(num_samples, count)
    order = torch.randperm(count, generator=generator)[:take].to(device)
    uv_target = uv_cand[order]
    uv_warp = uv_warp[order]

    offsets = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=dtype, device=device
    ) * patch_size
    neighbor_options = uv_warp[:, None, :] + offsets[None, :, :]
    option_ok = _in_box(neighbor_options, box_context)
    choice = torch.multinomial(option_ok.to(torch.float32), 1, generator=generator)
    uv_neighbor = torch.take_along_dim(neighbor_options, choice[..., None], dim=1)[:, 0, :]

    ctx_patches_h = ctx_height // patch_size
    ctx_patches_w = ctx_width // patch_size
    rand01 = torch.rand((take, 2), generator=generator, dtype=dtype)
    span = torch.tensor([ctx_patches_w - 1, ctx_patches_h - 1], dtype=dtype)
    uv_random = patch_to_pixel_coords(rand01 * span, patch_size).to(device)

    return CorrespondenceSamples(
        uv_target=uv_target,
        uv_context_warp=uv_warp,
        uv_context_no_warp=uv_target.clone(),
        uv_context_neighbor=uv_neighbor,
        uv_context_random=uv_random,
    )


def gather_value_pairs(
    features_context: Tensor,
    features_target: Tensor,
    samples: CorrespondenceSamples,
    patch_size: int = PATCH_SIZE,
) -> dict[str, Tensor]:
    """Read feature values for every variant.

    features_context, features_target: [C, Hp, Wp] patch-grid feature maps.
    Returns a dict of [N, C] tensors with keys "target", "warp", "no_warp",
    "neighbor", "random", and "mean".
    """
    out = {
        "target": sample_features_bilinear(features_target, samples.uv_target, patch_size),
        "warp": sample_features_bilinear(features_context, samples.uv_context_warp, patch_size),
        "no_warp": sample_features_bilinear(features_context, samples.uv_context_no_warp, patch_size),
        "neighbor": sample_features_bilinear(features_context, samples.uv_context_neighbor, patch_size),
        "random": sample_features_bilinear(features_context, samples.uv_context_random, patch_size),
    }
    mean = features_context.to(out["warp"].dtype).mean(dim=(1, 2))
    out["mean"] = mean[None, :].expand(samples.uv_target.shape[0], -1).clone()
    return out
