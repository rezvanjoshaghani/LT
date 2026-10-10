"""Context-Lift Transport-Only: the Phase 5 information-symmetric explicit comparator.

Phase 4's accepted per-point Transport-Only lifts the *target* frame's estimated
depth (Amendment A4), because that is the construction which preserves the
Phase 3 sample_id universe. That was the right choice for Phase 4's question,
which was how much accuracy estimated geometry costs an explicit transporter.
It is the wrong comparator for Phase 5's question, because Predict-with-Depth
is forbidden every target-content-derived input, target depth included. Scoring
a learned predictor that may not see target depth against an explicit method
that does would confound a learned-vs-explicit transformation cost with an
information asymmetry, and the confound would sit on the headline number.

Context-Lift Transport-Only removes the confound by construction. It receives
exactly the information the predictor receives:

    I = { F_c, D_c, K_c, K_t, T_target_from_context }

and nothing else. There is no target-depth parameter in any function in this
module, so target depth cannot be supplied even by mistake; the audit in the
Phase 5 test suite asserts that structurally rather than by inspection.

The direction of the map is the visible difference from Phase 4. Phase 4 reads
backwards, from a target sample to the context location that saw it. This reads
forwards, from a context patch to where that patch's surface point lands in the
target image, which is the only direction available without target depth:

    1. read the aligned context depth at a context patch center
    2. unproject into the context camera frame
    3. transform into the target camera frame
    4. project into the target image
    5. the context patch's frozen feature is the transported prediction
    6. score it against the frozen target feature at the landing location

Ground-truth depth appears in this module in exactly one role, and only in
context_lift_support: deciding which samples are evaluable. That is the
support role PROTOCOL 4.9 and the Phase 5 leakage rule both permit. It never
reaches a coordinate, a prediction, or a score.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from torch import Tensor

from .correspondence import _in_box, _sampling_box
from .encoders import (
    PATCH_SIZE,
    patch_grid_shape,
    patch_to_pixel_coords,
    sample_map_bilinear,
)
from .geometry import (
    apply_homography,
    invert_se3,
    project,
    relative_pose,
    rotation_homography,
    transform_points,
    unproject,
)
from .sample_identity import _mix64, pair_seed
from .visibility import visibility_masks

# A namespace salt so a Context-Lift sample id can never collide, or be
# accidentally joined, with a Phase 3 target-side sample_id. The two index
# different things: Phase 3 ids name a target-side coordinate, these name a
# context-side one. A join between them would be meaningless, and without a
# separate namespace it would also be silent.
CONTEXT_LIFT_SALT = np.uint64(0xD1B54A32D192ED03)


def context_lift_sample_ids(
    scene: str, context_frame_id: str, target_frame_id: str, uv_context: Tensor
) -> np.ndarray:
    """Deterministic ids for context-side samples, in their own namespace.

    uv_context: [N, 2] context pixel coordinates, which for this module are
        patch centers and therefore whole multiples of half a pixel.
    """
    if uv_context.dim() != 2 or uv_context.shape[-1] != 2:
        raise ValueError(f"uv_context must be [N, 2], got {tuple(uv_context.shape)}")
    if uv_context.shape[0] == 0:
        return np.zeros(0, dtype=np.uint64)
    scaled = uv_context.detach().to(torch.float64) * 2.0
    quantized = torch.round(scaled)
    if not torch.equal(quantized, scaled):
        raise ValueError(
            "context-side coordinates must be whole multiples of half a pixel; "
            "the id is defined on that grid so it cannot drift with float noise"
        )
    coords = quantized.cpu().numpy()
    if coords.min() < 0:
        raise ValueError("context-side coordinates must be non-negative")
    code = coords[:, 0].astype(np.uint64) * np.uint64(1 << 21) + coords[:, 1].astype(np.uint64)
    seed = pair_seed(scene, context_frame_id, target_frame_id)
    with np.errstate(over="ignore"):
        return _mix64(_mix64(np.uint64(seed) ^ CONTEXT_LIFT_SALT) ^ _mix64(code))


def context_patch_centers(
    context_hw: tuple[int, int], patch_size: int = PATCH_SIZE, dtype=torch.float32
) -> Tensor:
    """The context patch centers, row major, as [N, 2] pixel coordinates.

    These are the sample locations of the context-lift path. Patch centers,
    rather than every pixel, because the feature grid is defined at patch
    stride and a patch center is the one location whose bilinear feature read
    is exactly that patch's own frozen feature.
    """
    grid_h, grid_w = patch_grid_shape(context_hw, patch_size)
    rows = torch.arange(grid_h, dtype=dtype)
    cols = torch.arange(grid_w, dtype=dtype)
    jj, ii = torch.meshgrid(rows, cols, indexing="ij")
    patch_uv = torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=-1)
    return patch_to_pixel_coords(patch_uv, patch_size)


@dataclasses.dataclass
class ContextLiftMap:
    """The forward correspondence produced from context-side information alone."""

    uv_context: Tensor          # [N, 2] context patch centers, pixel coordinates
    uv_target: Tensor           # [N, 2] landing locations in the target image
    z_target: Tensor            # [N] depth of the landed point in the target camera
    depth_context: Tensor       # [N] the aligned context depth actually read
    depth_valid: Tensor         # [N] bool, finite and positive context depth
    landed: Tensor              # [N] bool, depth_valid and in front and inside the box

    def __post_init__(self) -> None:
        n = self.uv_context.shape[0]
        for name in ("uv_target", "z_target", "depth_context", "depth_valid", "landed"):
            if getattr(self, name).shape[0] != n:
                raise ValueError(f"{name} disagrees with uv_context on sample count")


def context_lift_map(
    depth_context_aligned: Tensor,
    K_context: Tensor,
    K_target: Tensor,
    T_target_from_context: Tensor,
    context_hw: tuple[int, int],
    target_hw: tuple[int, int],
    patch_size: int = PATCH_SIZE,
) -> ContextLiftMap:
    """Forward-map every context patch center into the target image.

    depth_context_aligned: [H, W] context-frame planar z-depth in meters, after
        whichever Phase 4 alignment level is in force. This is the only depth
        this function accepts; there is deliberately no target-depth parameter.
    K_context, K_target: 3x3 OpenCV intrinsics.
    T_target_from_context: the canonical relative transform, from
        geometry.relative_pose.

    Returns the correspondence, with a landing rule matching the frozen one:
    positive depth in the receiving camera and a location inside the receiving
    image's patch-center sampling box, so a bilinear feature read needs no
    clamping. Phase 4's per-point rule is the same test applied in the other
    direction.
    """
    if depth_context_aligned.dim() != 2:
        raise ValueError("depth_context_aligned must be [H, W]")
    dtype = depth_context_aligned.dtype
    uv_context = context_patch_centers(context_hw, patch_size, dtype=dtype).to(
        depth_context_aligned.device
    )

    # A patch center sits at a half-integer pixel, so this read is a genuine
    # bilinear interpolation of the four surrounding depth pixels rather than a
    # lookup. sample_map_bilinear is the frozen reader used everywhere else.
    depth = sample_map_bilinear(depth_context_aligned, uv_context)
    depth_valid = torch.isfinite(depth) & (depth > 0)
    # A nonpositive or nonfinite depth must not poison the arithmetic of the
    # samples around it, so it is replaced by a harmless one and masked out.
    safe = torch.where(depth_valid, depth, torch.ones_like(depth))

    points_context = unproject(uv_context, safe, K_context)
    points_target = transform_points(T_target_from_context, points_context)
    uv_target, z_target = project(points_target, K_target)

    box_target = _sampling_box(target_hw, patch_size)
    landed = depth_valid & (z_target > 0) & _in_box(uv_target, box_target)
    return ContextLiftMap(
        uv_context=uv_context,
        uv_target=uv_target,
        z_target=z_target,
        depth_context=depth,
        depth_valid=depth_valid,
        landed=landed,
    )


def context_lift_support(
    lift: ContextLiftMap,
    depth_context_gt: Tensor,
    depth_target_gt: Tensor,
    K_context: Tensor,
    K_target: Tensor,
    T_target_from_context: Tensor,
    rel_tol: float | None = None,
    patch_size: int = PATCH_SIZE,
) -> Tensor:
    """Which landed samples are ground-truth evaluable. Support only, never input.

    A sample is evaluable when the surface point the context patch sees is also
    genuinely visible in the target view, decided by the frozen co-visibility
    rule of the analysis config. The rule is the same function Phase 3 uses,
    called with the two views exchanged, so the relative depth tolerance, the
    frustum bounds, and the border convention are identical rather than merely
    similar.

    Ground truth decides membership here and nothing else. It never touches a
    coordinate, a prediction, or a score: those all come from the estimated
    context depth in context_lift_map, which this function does not modify.
    """
    T_context_from_target = invert_se3(T_target_from_context)
    # Roles exchanged: "co-visible on the context grid" is exactly the question
    # of whether the target camera also sees what the context camera sees.
    masks = visibility_masks(
        depth_target=depth_context_gt,
        depth_context=depth_target_gt,
        K_target=K_context,
        K_context=K_target,
        T_target_from_context=T_context_from_target,
        rel_tol=rel_tol,
    )
    covisible = masks.covisible
    # Read the per-pixel mask at the patch centers. Nearest is correct for a
    # boolean: interpolating a mask would invent fractional visibility.
    uv = lift.uv_context
    cols = torch.round(uv[..., 0]).long().clamp(0, covisible.shape[1] - 1)
    rows = torch.round(uv[..., 1]).long().clamp(0, covisible.shape[0] - 1)
    evaluable = covisible[rows, cols]
    return lift.landed & evaluable


def rotation_homography_landing(
    K_context: Tensor,
    K_target: Tensor,
    T_target_from_context: Tensor,
    context_hw: tuple[int, int],
    patch_size: int = PATCH_SIZE,
    dtype=torch.float32,
) -> Tensor:
    """Where context patch centers land under the analytic pure-rotation homography.

    Depth free by construction. Under zero translation the lifted depth cancels
    algebraically, so context_lift_map must reproduce this for any depth values
    whatsoever; the Phase 5 rotation gate asserts exactly that.
    """
    R = T_target_from_context[:3, :3]
    H = rotation_homography(K_context, K_target, R)
    uv = context_patch_centers(context_hw, patch_size, dtype=dtype).to(H.device)
    return apply_homography(H, uv)


def relative_pose_for_pair(T_world_from_target: Tensor, T_world_from_context: Tensor) -> Tensor:
    """The one relative-transform formula, re-exported so callers cannot reinvent it."""
    return relative_pose(T_world_from_target, T_world_from_context)
