"""Experiment Zero correspondence sampling and the frozen nulls of PROTOCOL 3.6.

Samples target locations that are co-visible in the context view, computes the
ground-truth corresponding location in the context image with ground-truth
depth, and constructs the null locations.

The three location controls read from the same context feature map and differ
only in where they read, with image, encoder, and scene held fixed:

- Oracle-Transport reads at the correct correspondence.
- No-Warp-Copy reads at the target's own image coordinate, without transport.
  It measures the position prior.
- Neighbor-Patch reads one patch away from the correct correspondence. The
  direction is drawn hash-deterministically from the record's sample_id among
  the in-bounds axis-aligned unit offsets, so it is reproducible per record,
  unbiased across directions, and defined at image borders. Records with no
  in-bounds offset are omitted and the omission is counted.
- Random-Patch reads one whole patch of the same context image, chosen by a
  fixed hash of the record's sample_id. The index is an integer, so the value
  is a patch the encoder actually produced and not a blend of up to four.

Mean-Feature is not built here. It reads no location, and PROTOCOL 3.6 defines
it as one global vector per encoder over the training split, so it belongs to
the evaluation layer that knows the split, not to a per-image sampler.

Nothing in this module consumes a random generator. Which candidates are
sampled, which direction a neighbor takes, and which patch a random draw lands
on are all functions of sample_id, so a permuted execution order, a different
batch size, or a resumed run reproduces the same records exactly.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from .encoders import (
    PATCH_SIZE,
    patch_to_pixel_coords,
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
from .sample_identity import (
    NEIGHBOR_PATCH_SALT,
    RANDOM_PATCH_SALT,
    derived_draw,
    sample_ids,
)
from .visibility import default_relative_depth_tol

# Salt for choosing which co-visible candidates to evaluate. Selection is a hash
# of sample_id rather than a shuffle, so the chosen set does not depend on how
# many candidates survived upstream filtering or in what order they arrived.
SELECTION_SALT = np.uint64(0xD1B54A32D192ED03)

# The four axis-aligned unit offsets, in patches, in a fixed order. The order is
# part of the null's definition because the hash indexes into it.
NEIGHBOR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def choose_in_bounds_offset(ids: np.ndarray, option_ok: np.ndarray) -> np.ndarray:
    """Pick each record's neighbour direction from its sample_id.

    option_ok: [N, 4] bool, which of NEIGHBOR_OFFSETS is usable for that record.
    Returns [N] indices into NEIGHBOR_OFFSETS.

    The hash indexes the in-bounds options rather than all four, which is what
    makes the null defined at a border instead of omitted there. This is the one
    definition of the rule: both the per-point sampler and the splat path call
    it, so a record cannot be given one direction on one path and another on the
    other, which would silently make the variant incomparable across paths.
    """
    counts = option_ok.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("every record passed here must have an in-bounds offset")
    pick = derived_draw(ids, NEIGHBOR_PATCH_SALT, counts)
    rank = np.cumsum(option_ok, axis=1) - 1
    return (option_ok & (rank == pick[:, None])).argmax(axis=1)


class CorrespondenceSamples(NamedTuple):
    sample_id: np.ndarray        # [N] uint64, the PROTOCOL 3.2 identity
    uv_target: Tensor            # [N, 2] pixel coordinates in the target image
    uv_context_warp: Tensor      # [N, 2] ground-truth correspondence in the context image
    uv_context_no_warp: Tensor   # [N, 2] same pixel coordinates as uv_target
    uv_context_neighbor: Tensor  # [N, 2] warp location offset by one whole patch
    random_patch_index: Tensor   # [N, 2] integer (row, col) patch of the context grid
    neighbor_omitted: int        # records dropped for having no in-bounds offset


def _patch_center_px(index: int, patch_size: int) -> float:
    """Pixel coordinate of the center of one patch, along one axis.

    Uses the mapping defined in encoders.py so this box cannot drift from the
    interpolation it is meant to bound.
    """
    return float(patch_to_pixel_coords(torch.tensor(float(index)), patch_size))


def _sampling_box(hw_px: tuple[int, int], patch_size: int) -> tuple[float, float, float, float]:
    """Pixel-coordinate box where patch-grid bilinear sampling needs no clamping.

    hw_px: (H, W) in pixels. The box spans the first and last patch centers.
    Returns (u_min, u_max, v_min, v_max) in pixel coordinates, centers at
    integers, all bounds inclusive.
    """
    height, width = hw_px
    lo = _patch_center_px(0, patch_size)
    u_max = _patch_center_px(width // patch_size - 1, patch_size)
    v_max = _patch_center_px(height // patch_size - 1, patch_size)
    return lo, u_max, lo, v_max


def _in_box(uv: Tensor, box: tuple[float, float, float, float]) -> Tensor:
    """Mask of the [..., 2] pixel coordinates (u, v) that lie inside a sampling box.

    uv and box are both in pixel coordinates, centers at integers.
    """
    u_min, u_max, v_min, v_max = box
    return (
        (uv[..., 0] >= u_min)
        & (uv[..., 0] <= u_max)
        & (uv[..., 1] >= v_min)
        & (uv[..., 1] <= v_max)
    )


def sample_correspondences(
    depth_target: Tensor,
    K_target: Tensor,
    K_context: Tensor,
    T_target_from_context: Tensor,
    covisible: Tensor,
    num_samples: int | None,
    context_hw_px: tuple[int, int],
    scene: str,
    context_frame_id: str,
    target_frame_id: str,
    patch_size: int = PATCH_SIZE,
    mode: str = "patch_center",
    depth_consistency_tol: float | None = None,
) -> CorrespondenceSamples:
    """Sample co-visible target locations with ground-truth warps and null locations.

    depth_target: [H, W] ground-truth planar z-depth of the target view, meters.
    K_target, K_context: 3x3 intrinsics.
    T_target_from_context: the canonical relative transform from geometry.relative_pose.
    covisible: [H, W] bool mask on the target grid, from visibility.visibility_masks.
    num_samples: how many correspondences to score, or None for every eligible
        one. None is the configured behaviour and removes the selection step
        entirely, so no hash decides which correspondences are evaluated.
    context_hw_px: (H, W) of the context image in pixels.
    scene, context_frame_id, target_frame_id: identity, used to derive sample_id.
    mode: "patch_center" samples target patch centers and keeps a candidate only
        if the four integer pixels around the center are all co-visible and all
        lie on one surface. "pixel" samples integer target pixel centers.
    depth_consistency_tol: relative depth spread allowed across those four
        pixels, used by "patch_center" only.
    """
    if depth_consistency_tol is None:
        depth_consistency_tol = default_relative_depth_tol()
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
        # The depth at a patch center is read by interpolating these four pixels.
        # That is a depth on the surface only when the four share one surface. A
        # center straddling a depth edge would otherwise be lifted to a point in
        # mid air, and its warp location would match neither surface while being
        # reported as ground truth. Reject those centers.
        corners = torch.stack(
            (
                depth_target[y0, x0],
                depth_target[y0, x0 + 1],
                depth_target[y0 + 1, x0],
                depth_target[y0 + 1, x0 + 1],
            ),
            dim=-1,
        ).to(dtype)
        lo = corners.amin(dim=-1)
        hi = corners.amax(dim=-1)
        one_surface = (lo > 0) & torch.isfinite(hi) & ((hi - lo) <= depth_consistency_tol * lo)
        keep = (
            covisible[y0, x0]
            & covisible[y0, x0 + 1]
            & covisible[y0 + 1, x0]
            & covisible[y0 + 1, x0 + 1]
            & one_surface
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

    ids = sample_ids(scene, context_frame_id, target_frame_id, uv_cand)

    # Neighbor-Patch: one whole patch away, direction hashed from sample_id among
    # the offsets that stay inside the context grid.
    offsets = torch.tensor(
        [[float(dx), float(dy)] for dx, dy in NEIGHBOR_OFFSETS], dtype=dtype, device=device
    ) * patch_size
    options = uv_warp[:, None, :] + offsets[None, :, :]
    option_ok = _in_box(options, box_context)
    has_neighbor = option_ok.any(dim=1)
    neighbor_omitted = int((~has_neighbor).sum())

    uv_cand = uv_cand[has_neighbor]
    uv_warp = uv_warp[has_neighbor]
    options = options[has_neighbor]
    option_ok = option_ok[has_neighbor].cpu().numpy()
    ids = ids[has_neighbor.cpu().numpy()]

    # Selection is a hash, not a shuffle, so the evaluated set does not depend on
    # arrival order or on how many candidates survived the filters above.
    count = int(ids.shape[0])
    take = count if num_samples is None else min(num_samples, count)
    if take < count:
        ranked = np.argsort(derived_draw(ids, SELECTION_SALT, 1 << 62), kind="stable")
        chosen = np.sort(ranked[:take])
    else:
        chosen = np.arange(count)
    index = torch.from_numpy(chosen).to(device)
    ids = ids[chosen]
    uv_target = uv_cand[index]
    uv_warp = uv_warp[index]
    options = options[index]
    option_ok = option_ok[chosen]

    selector = choose_in_bounds_offset(ids, option_ok)
    uv_neighbor = options[
        torch.arange(len(selector), device=device), torch.from_numpy(selector).to(device)
    ]

    # Random-Patch: an integer patch of the context grid, so the value read is a
    # patch the encoder produced rather than a blend of up to four.
    ctx_patches_h = ctx_height // patch_size
    ctx_patches_w = ctx_width // patch_size
    flat = derived_draw(ids, RANDOM_PATCH_SALT, ctx_patches_h * ctx_patches_w)
    random_patch_index = torch.from_numpy(
        np.stack((flat // ctx_patches_w, flat % ctx_patches_w), axis=-1)
    ).to(device)

    return CorrespondenceSamples(
        sample_id=ids,
        uv_target=uv_target,
        uv_context_warp=uv_warp,
        uv_context_no_warp=uv_target.clone(),
        uv_context_neighbor=uv_neighbor,
        random_patch_index=random_patch_index,
        neighbor_omitted=neighbor_omitted,
    )


def gather_value_pairs(
    features_context: Tensor,
    features_target: Tensor,
    samples: CorrespondenceSamples,
    patch_size: int = PATCH_SIZE,
) -> dict[str, Tensor]:
    """Read feature values for every location-bearing variant.

    features_context, features_target: [C, Hp, Wp] patch-grid feature maps.
    Returns a dict of [N, C] tensors with keys "target", "warp", "no_warp",
    "neighbor", and "random". Mean-Feature is deliberately absent: it reads no
    location, and PROTOCOL 3.6 defines it over the training split, which this
    function cannot see.
    """
    rows = samples.random_patch_index[:, 0]
    cols = samples.random_patch_index[:, 1]
    return {
        "target": sample_features_bilinear(features_target, samples.uv_target, patch_size),
        "warp": sample_features_bilinear(features_context, samples.uv_context_warp, patch_size),
        "no_warp": sample_features_bilinear(
            features_context, samples.uv_context_no_warp, patch_size
        ),
        "neighbor": sample_features_bilinear(
            features_context, samples.uv_context_neighbor, patch_size
        ),
        # Indexed, never interpolated: PROTOCOL 3.6 asks for a patch.
        "random": features_context[:, rows, cols].T,
    }
