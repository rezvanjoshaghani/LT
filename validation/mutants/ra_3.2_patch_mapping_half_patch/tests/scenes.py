"""Analytic scenes. These constructions are the referee for Phase 0.

Two-plane scene layout. All quantities below are exact in float64:
- Image 224x224, patch size 14, so a 16x16 patch grid.
- fx = fy = 224, cx = cy = 111.5. Pixel centers sit at integer coordinates.
- Context camera at the world origin, identity rotation, looking down +z.
- Target camera translated by +0.5 m along world x, identity rotation.
- Back plane: fronto-parallel at z = 4 m, fills the whole view.
- Front slab: fronto-parallel at z = 2 m, covers context pixel columns 70..125.
- Disparity (context column minus target column): front 56 px (4 patches),
  back 28 px (2 patches). Both are exact integers, so every splat lands on an
  exact pixel center and every region boundary is patch aligned.

Derived analytic answers, used by the tests:
- The slab covers target pixel columns 14..69.
- Target patch column sources: col 0 from back patch col 2; cols 1..4 from front
  patch cols 5..8; cols 5..6 empty (the disoccluded strip, pixel cols 70..97);
  cols 7..13 from back patch cols 9..15; cols 14..15 empty (never inside the
  context view, pixel cols 196..223).
- Co-visible target pixel columns: 0..69 and 98..195. All rows behave alike.
- Target z-buffer after transport: 4 m on back columns 0..13 and 98..195,
  2 m on slab columns 14..69, +inf on empty columns 70..97 and 196..223.

Context patch features are integer codes: channel 0 is the patch row, channel 1
is the patch column, channel 2 is a unique id row * 16 + col + 1. Integer codes
make splat-and-pool averages exact in float32.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor

from lot.geometry import relative_pose

IMAGE_SIZE = 224
PATCH = 14
GRID = IMAGE_SIZE // PATCH
FX = 224.0
CX = 111.5
Z_FRONT = 2.0
Z_BACK = 4.0
BASELINE = 0.5
DISPARITY_FRONT = int(FX * BASELINE / Z_FRONT)  # 56
DISPARITY_BACK = int(FX * BASELINE / Z_BACK)    # 28
SLAB_CTX_COLS = (70, 126)   # half open, context pixel columns
SLAB_TGT_COLS = (SLAB_CTX_COLS[0] - DISPARITY_FRONT, SLAB_CTX_COLS[1] - DISPARITY_FRONT)
DISOCCLUDED_TGT_COLS = (70, 98)     # half open, target pixel columns
OUT_OF_VIEW_TGT_COLS = (196, 224)   # half open, target pixel columns


def intrinsics() -> Tensor:
    return torch.tensor(
        [[FX, 0.0, CX], [0.0, FX, CX], [0.0, 0.0, 1.0]], dtype=torch.float64
    )


def make_pose(R: Tensor | None = None, t: Tensor | None = None) -> Tensor:
    """Build T_world_from_camera from a rotation and a camera center in world."""
    T = torch.eye(4, dtype=torch.float64)
    if R is not None:
        T[:3, :3] = R.to(torch.float64)
    if t is not None:
        T[:3, 3] = t.to(torch.float64)
    return T


def rodrigues(axis: Tensor, angle_rad: float) -> Tensor:
    """Rotation matrix from an axis and an angle, float64."""
    a = axis.to(torch.float64)
    a = a / torch.linalg.vector_norm(a)
    K = torch.tensor(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]],
        dtype=torch.float64,
    )
    eye = torch.eye(3, dtype=torch.float64)
    return eye + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)


def random_se3(generator: torch.Generator) -> Tensor:
    axis = torch.rand(3, generator=generator, dtype=torch.float64) - 0.5
    angle = float(torch.rand(1, generator=generator, dtype=torch.float64)) * 2 * math.pi
    t = (torch.rand(3, generator=generator, dtype=torch.float64) - 0.5) * 4
    return make_pose(rodrigues(axis, angle), t)


def patch_codes(grid_h: int = GRID, grid_w: int = GRID) -> Tensor:
    """Integer per-patch feature codes, [3, grid_h, grid_w] float32."""
    rows = torch.arange(grid_h, dtype=torch.float32)
    cols = torch.arange(grid_w, dtype=torch.float32)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack((rr, cc, rr * grid_w + cc + 1.0))


@dataclasses.dataclass(frozen=True)
class TwoPlaneScene:
    K: Tensor
    T_world_from_context: Tensor
    T_world_from_target: Tensor
    T_target_from_context: Tensor
    depth_context: Tensor          # [H, W] float64
    depth_target: Tensor           # [H, W] float64
    features_context: Tensor       # [3, GRID, GRID] float32 integer codes
    expected_features: Tensor      # [3, GRID, GRID] analytic transport output
    expected_coverage: Tensor      # [GRID, GRID] float32
    expected_zbuffer: Tensor       # [H, W] float64
    source_patch_col: Tensor       # [GRID] long, context patch column per target patch column, -1 if empty
    covisible_target: Tensor       # [H, W] bool

    def disparity_for_target_columns(self, u: Tensor) -> Tensor:
        """Analytic disparity per target pixel column: front on the slab, back elsewhere."""
        in_slab = (u >= SLAB_TGT_COLS[0]) & (u < SLAB_TGT_COLS[1])
        return torch.where(
            in_slab,
            torch.full_like(u, float(DISPARITY_FRONT)),
            torch.full_like(u, float(DISPARITY_BACK)),
        )


def build_two_plane_scene() -> TwoPlaneScene:
    K = intrinsics()
    T_world_from_context = make_pose()
    T_world_from_target = make_pose(t=torch.tensor([BASELINE, 0.0, 0.0]))
    T_target_from_context = relative_pose(T_world_from_target, T_world_from_context)

    cols = torch.arange(IMAGE_SIZE)
    slab_ctx = (cols >= SLAB_CTX_COLS[0]) & (cols < SLAB_CTX_COLS[1])
    depth_context = torch.full((IMAGE_SIZE, IMAGE_SIZE), Z_BACK, dtype=torch.float64)
    depth_context[:, slab_ctx] = Z_FRONT
    slab_tgt = (cols >= SLAB_TGT_COLS[0]) & (cols < SLAB_TGT_COLS[1])
    depth_target = torch.full((IMAGE_SIZE, IMAGE_SIZE), Z_BACK, dtype=torch.float64)
    depth_target[:, slab_tgt] = Z_FRONT

    features_context = patch_codes()

    source_patch_col = torch.full((GRID,), -1, dtype=torch.long)
    source_patch_col[0] = 2
    for c in range(1, 5):
        source_patch_col[c] = c + 4
    for c in range(7, 14):
        source_patch_col[c] = c + 2

    expected_features = torch.zeros_like(features_context)
    expected_coverage = torch.zeros((GRID, GRID), dtype=torch.float32)
    rows = torch.arange(GRID, dtype=torch.float32)
    for c in range(GRID):
        sc = int(source_patch_col[c])
        if sc < 0:
            continue
        expected_features[0, :, c] = rows
        expected_features[1, :, c] = float(sc)
        expected_features[2, :, c] = rows * GRID + sc + 1.0
        expected_coverage[:, c] = 1.0

    expected_zbuffer = torch.full((IMAGE_SIZE, IMAGE_SIZE), torch.inf, dtype=torch.float64)
    expected_zbuffer[:, 0:14] = Z_BACK
    expected_zbuffer[:, SLAB_TGT_COLS[0]:SLAB_TGT_COLS[1]] = Z_FRONT
    expected_zbuffer[:, DISOCCLUDED_TGT_COLS[1]:OUT_OF_VIEW_TGT_COLS[0]] = Z_BACK

    covis_cols = (cols < DISOCCLUDED_TGT_COLS[0]) | (
        (cols >= DISOCCLUDED_TGT_COLS[1]) & (cols < OUT_OF_VIEW_TGT_COLS[0])
    )
    covisible_target = covis_cols[None, :].expand(IMAGE_SIZE, IMAGE_SIZE).clone()

    return TwoPlaneScene(
        K=K,
        T_world_from_context=T_world_from_context,
        T_world_from_target=T_world_from_target,
        T_target_from_context=T_target_from_context,
        depth_context=depth_context,
        depth_target=depth_target,
        features_context=features_context,
        expected_features=expected_features,
        expected_coverage=expected_coverage,
        expected_zbuffer=expected_zbuffer,
        source_patch_col=source_patch_col,
        covisible_target=covisible_target,
    )


@dataclasses.dataclass(frozen=True)
class SinglePlaneScene:
    K: Tensor
    T_target_from_context: Tensor
    depth_context: Tensor
    features_context: Tensor
    expected_features: Tensor
    expected_coverage: Tensor
    disparity_px: int


def build_single_plane_scene(disparity_px: int = 7) -> SinglePlaneScene:
    """One fronto-parallel plane at Z_FRONT with a lateral move.

    The disparity is not a multiple of the patch size, so boundary patches pool
    features from two source patches and the trailing patch column is half covered.
    With integer codes the pooled averages stay exact in float32.
    """
    if not 0 < disparity_px < PATCH:
        raise ValueError("this construction expects a sub-patch disparity")
    K = intrinsics()
    baseline = disparity_px * Z_FRONT / FX
    T_world_from_context = make_pose()
    T_world_from_target = make_pose(t=torch.tensor([baseline, 0.0, 0.0]))
    T_target_from_context = relative_pose(T_world_from_target, T_world_from_context)
    depth_context = torch.full((IMAGE_SIZE, IMAGE_SIZE), Z_FRONT, dtype=torch.float64)
    features_context = patch_codes()

    expected_features = torch.zeros_like(features_context)
    expected_coverage = torch.ones((GRID, GRID), dtype=torch.float32)
    rows = torch.arange(GRID, dtype=torch.float32)
    frac = disparity_px / PATCH
    for c in range(GRID):
        expected_features[0, :, c] = rows
        if c < GRID - 1:
            # Pixels come from source patch c and c + 1 in proportion (1 - frac, frac).
            expected_features[1, :, c] = c + frac
            expected_features[2, :, c] = rows * GRID + c + 1.0 + frac
        else:
            # Only the leading pixels of the last patch column receive support,
            # all from source patch GRID - 1.
            expected_features[1, :, c] = float(GRID - 1)
            expected_features[2, :, c] = rows * GRID + (GRID - 1) + 1.0
            expected_coverage[:, c] = (PATCH - disparity_px) / PATCH
    return SinglePlaneScene(
        K=K,
        T_target_from_context=T_target_from_context,
        depth_context=depth_context,
        features_context=features_context,
        expected_features=expected_features,
        expected_coverage=expected_coverage,
        disparity_px=disparity_px,
    )


@dataclasses.dataclass(frozen=True)
class SubPixelTwoPlaneScene:
    """Two planes whose disparities are not whole pixels.

    The two-plane scene above is built so every disparity is an exact integer.
    That makes its analytic answers exact, but it also means every reprojected
    location lands on a pixel center, where interpolation and nearest sampling
    agree. This scene breaks that: the occlusion edges and the disparities both
    fall between pixels, so the visibility referee is tested where a depth map
    read has to choose what a z-buffer entry means.

    Layout, all in the world frame of the context camera:
    - Context camera at the origin, target camera at (BASELINE_SUB, 0, 0), both
      with identity rotation and the shared intrinsics.
    - Back plane at z = Z_BACK filling the view. Front slab at z = Z_FRONT
      spanning world x in [SLAB_X0, SLAB_X1], which is not patch aligned.
    """

    K: Tensor
    T_target_from_context: Tensor
    depth_context: Tensor    # [H, W] float64
    depth_target: Tensor     # [H, W] float64
    covisible_target: Tensor  # [H, W] bool, from continuous geometry


BASELINE_SUB = 0.1
SLAB_X0 = -0.1
SLAB_X1 = 0.4


def _two_plane_depth(camera_x: float) -> Tensor:
    """Depth seen by a camera at world (camera_x, 0, 0) with identity rotation.

    A pixel sees the front slab when its ray crosses z = Z_FRONT inside the slab,
    and the back plane otherwise.
    """
    u = torch.arange(IMAGE_SIZE, dtype=torch.float64)
    x_at_front = camera_x + Z_FRONT * (u - CX) / FX
    on_slab = (x_at_front >= SLAB_X0) & (x_at_front <= SLAB_X1)
    d = torch.where(on_slab, torch.full_like(u, Z_FRONT), torch.full_like(u, Z_BACK))
    return d[None, :].expand(IMAGE_SIZE, IMAGE_SIZE).clone()


def build_subpixel_two_plane_scene() -> SubPixelTwoPlaneScene:
    K = intrinsics()
    T_world_from_context = make_pose()
    T_world_from_target = make_pose(t=torch.tensor([BASELINE_SUB, 0.0, 0.0]))
    T_target_from_context = relative_pose(T_world_from_target, T_world_from_context)
    depth_context = _two_plane_depth(0.0)
    depth_target = _two_plane_depth(BASELINE_SUB)

    # Continuous ground truth, per target pixel column. Lift the target pixel to
    # its surface point, ask where that point lands in the context image, and ask
    # whether the context ray to it crosses the slab first.
    u = torch.arange(IMAGE_SIZE, dtype=torch.float64)
    depth = depth_target[0]
    x_world = BASELINE_SUB + depth * (u - CX) / FX
    u_in_context = CX + FX * x_world / depth
    inside_context = (u_in_context >= -0.5) & (u_in_context < IMAGE_SIZE - 0.5)
    x_at_front = x_world * Z_FRONT / depth
    behind_slab = (depth > Z_FRONT) & (x_at_front >= SLAB_X0) & (x_at_front <= SLAB_X1)
    covisible = (inside_context & ~behind_slab)[None, :].expand(IMAGE_SIZE, IMAGE_SIZE)

    return SubPixelTwoPlaneScene(
        K=K,
        T_target_from_context=T_target_from_context,
        depth_context=depth_context,
        depth_target=depth_target,
        covisible_target=covisible.clone(),
    )


@dataclasses.dataclass(frozen=True)
class RotationScene:
    K: Tensor
    T_target_from_context: Tensor
    R_target_from_context: Tensor


def build_rotation_scene(yaw_deg: float = 5.0) -> RotationScene:
    K = intrinsics()
    R = rodrigues(torch.tensor([0.0, 1.0, 0.0]), math.radians(yaw_deg))
    T_world_from_context = make_pose()
    T_world_from_target = make_pose(R=R)
    T_target_from_context = relative_pose(T_world_from_target, T_world_from_context)
    return RotationScene(
        K=K,
        T_target_from_context=T_target_from_context,
        R_target_from_context=T_target_from_context[:3, :3],
    )


@dataclasses.dataclass(frozen=True)
class SignedTwoPlaneScene:
    """The two-plane occlusion scene with the baseline direction as a parameter.

    Which surface wins a contested target pixel is decided by depth. Which one
    is *written last* is decided by raster order, and raster order follows the
    source column. Under a positive-x baseline the contributing source for a
    target pixel sits at target-x plus the disparity, so the near surface, with
    the larger disparity, always contributes from a later column and therefore
    writes last. Nearest-depth and last-write-wins then agree on every contested
    pixel and no test on this scene can tell them apart. Flipping the baseline
    puts the far surface's contributor later and separates the two policies.
    """

    sign: int
    K: Tensor
    T_target_from_context: Tensor
    depth_context: Tensor
    features_context: Tensor


def build_signed_two_plane_scene(sign: int = 1) -> SignedTwoPlaneScene:
    """Front slab at Z_FRONT over context columns 70..125, back plane behind."""
    if sign not in (1, -1):
        raise ValueError("sign must be 1 or -1")
    K = intrinsics()
    T_world_from_context = make_pose()
    T_world_from_target = make_pose(t=torch.tensor([sign * BASELINE, 0.0, 0.0]))
    T_target_from_context = relative_pose(T_world_from_target, T_world_from_context)
    cols = torch.arange(IMAGE_SIZE)
    slab = (cols >= SLAB_CTX_COLS[0]) & (cols < SLAB_CTX_COLS[1])
    depth_context = torch.full((IMAGE_SIZE, IMAGE_SIZE), Z_BACK, dtype=torch.float64)
    depth_context[:, slab] = Z_FRONT
    return SignedTwoPlaneScene(
        sign=sign,
        K=K,
        T_target_from_context=T_target_from_context,
        depth_context=depth_context,
        features_context=patch_codes(),
    )
