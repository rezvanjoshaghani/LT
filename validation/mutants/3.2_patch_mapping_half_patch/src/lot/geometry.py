"""Pinhole camera geometry. Small pure functions, no hidden state.

Conventions, used everywhere in this repository:
- OpenCV pinhole camera. x right, y down, z forward.
- Intrinsics K are 3x3 with zero skew.
- Pixel coordinates are (u, v). u runs along columns to the right. v runs along rows down.
- Pixel centers sit at integer coordinates. Pixel (0, 0) spans [-0.5, 0.5] on both axes.
- Depth is planar z-depth in meters. It is the z coordinate in the camera frame, not ray length.
- Poses are stored as T_world_from_camera, 4x4.
- The relative transform between two cameras is T_target_from_context. It is computed by
  relative_pose below. That function is the only place the formula is written.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def common_dtype(*tensors: Tensor) -> torch.dtype:
    """Promote the dtypes of all inputs to one floating point dtype."""
    dtype = tensors[0].dtype
    for t in tensors[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    return dtype


def check_intrinsics(K: Tensor) -> None:
    """Validate a 3x3 zero-skew OpenCV intrinsics matrix. Raises ValueError on mismatch.

    Focal lengths must be positive. A negative focal length mirrors an image axis,
    which every downstream function would carry out silently. The zero entries are
    checked with a tolerance, matching check_se3, so a matrix that picked up float
    noise in a constrained entry is not rejected for it.
    """
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {tuple(K.shape)}")
    last_row = torch.tensor([0.0, 0.0, 1.0], dtype=K.dtype, device=K.device)
    zero = torch.zeros((), dtype=K.dtype, device=K.device)
    if not torch.allclose(K[0, 1], zero) or not torch.allclose(K[1, 0], zero):
        raise ValueError("K must have zero skew")
    if not torch.allclose(K[2], last_row):
        raise ValueError("K must have last row [0, 0, 1]")
    if float(K[0, 0]) <= 0 or float(K[1, 1]) <= 0:
        raise ValueError(
            f"K must have positive focal lengths, got {float(K[0, 0])}, {float(K[1, 1])}"
        )


def check_se3(T: Tensor) -> None:
    """Validate the shape and last row of a 4x4 rigid transform."""
    if T.shape != (4, 4):
        raise ValueError(f"T must be 4x4, got {tuple(T.shape)}")
    last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=T.dtype, device=T.device)
    if not torch.allclose(T[3], last_row):
        raise ValueError("T must have last row [0, 0, 0, 1]")


def pixel_grid(
    height: int,
    width: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Grid of pixel-center coordinates.

    Returns [height, width, 2]. Entry (i, j) is (u=j, v=i). Centers sit at integers.
    """
    v = torch.arange(height, dtype=dtype, device=device)
    u = torch.arange(width, dtype=dtype, device=device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack((uu, vv), dim=-1)


def unproject(uv: Tensor, depth: Tensor, K: Tensor) -> Tensor:
    """Lift pixels with planar z-depth to 3D points in the camera frame.

    uv: [..., 2] pixel coordinates (u, v), centers at integers.
    depth: [...] planar z-depth in meters.
    K: 3x3 intrinsics.
    Returns [..., 3] points in the camera frame (x right, y down, z forward).
    """
    check_intrinsics(K)
    dtype = common_dtype(uv, depth, K)
    uv = uv.to(dtype)
    depth = depth.to(dtype)
    K = K.to(dtype)
    x = (uv[..., 0] - K[0, 2]) * depth / K[0, 0]
    y = (uv[..., 1] - K[1, 2]) * depth / K[1, 1]
    return torch.stack((x, y, depth), dim=-1)


def project(points_cam: Tensor, K: Tensor) -> tuple[Tensor, Tensor]:
    """Project camera-frame 3D points to pixel coordinates.

    points_cam: [..., 3] points in the camera frame.
    K: 3x3 intrinsics.
    Returns (uv [..., 2], z [...]). z is planar depth. Points with z <= 0 sit behind
    the camera. Their uv is meaningless and the caller must filter on z.
    """
    check_intrinsics(K)
    dtype = common_dtype(points_cam, K)
    p = points_cam.to(dtype)
    K = K.to(dtype)
    z = p[..., 2]
    u = K[0, 0] * p[..., 0] / z + K[0, 2]
    v = K[1, 1] * p[..., 1] / z + K[1, 2]
    return torch.stack((u, v), dim=-1), z


def transform_points(T: Tensor, points: Tensor) -> Tensor:
    """Apply a 4x4 rigid transform to [..., 3] points."""
    check_se3(T)
    dtype = common_dtype(T, points)
    T = T.to(dtype)
    points = points.to(dtype)
    return points @ T[:3, :3].mT + T[:3, 3]


def invert_se3(T: Tensor) -> Tensor:
    """Invert a 4x4 rigid transform without a general matrix inverse."""
    check_se3(T)
    R = T[:3, :3]
    t = T[:3, 3]
    out = torch.eye(4, dtype=T.dtype, device=T.device)
    out[:3, :3] = R.mT
    out[:3, 3] = -(R.mT @ t)
    return out


def compose(T_a_from_b: Tensor, T_b_from_c: Tensor) -> Tensor:
    """Compose transforms. Returns T_a_from_c."""
    return T_a_from_b @ T_b_from_c


def relative_pose(T_world_from_target: Tensor, T_world_from_context: Tensor) -> Tensor:
    """The relative camera transform used everywhere in this project.

    T_target_from_context = inv(T_world_from_target) @ T_world_from_context.
    This is the only place the formula is written. Import it, do not rewrite it.
    """
    return invert_se3(T_world_from_target) @ T_world_from_context


def rotation_homography(K_ctx: Tensor, K_tgt: Tensor, R_tgt_from_ctx: Tensor) -> Tensor:
    """Homography mapping context pixels to target pixels under pure camera rotation.

    Valid only when the relative transform has zero translation.
    Returns the 3x3 matrix K_tgt @ R_tgt_from_ctx @ inv(K_ctx).
    """
    check_intrinsics(K_ctx)
    check_intrinsics(K_tgt)
    dtype = common_dtype(K_ctx, K_tgt, R_tgt_from_ctx)
    return K_tgt.to(dtype) @ R_tgt_from_ctx.to(dtype) @ torch.linalg.inv(K_ctx.to(dtype))


def apply_homography(H: Tensor, uv: Tensor) -> Tensor:
    """Apply a 3x3 homography to [..., 2] pixel coordinates."""
    dtype = common_dtype(H, uv)
    H = H.to(dtype)
    uv = uv.to(dtype)
    ones = torch.ones_like(uv[..., :1])
    p = torch.cat((uv, ones), dim=-1) @ H.mT
    return p[..., :2] / p[..., 2:3]


def baseline_m(T_tgt_from_ctx: Tensor) -> Tensor:
    """Distance between the two camera centers, in meters.

    T_tgt_from_ctx: the relative transform from relative_pose. Its translation
    norm is the baseline.
    """
    check_se3(T_tgt_from_ctx)
    return torch.linalg.vector_norm(T_tgt_from_ctx[:3, 3])


def parallax_from_median_depth(T_tgt_from_ctx: Tensor, median_depth_m: float) -> Tensor:
    """Parallax magnitude: camera baseline over median scene depth.

    This is the definition the whole project uses. It is written once here, and
    parallax below is the convenience form that takes a depth map instead of a
    number already reduced.
    """
    if not (median_depth_m > 0):
        raise ValueError(f"median depth must be positive, got {median_depth_m}")
    return baseline_m(T_tgt_from_ctx) / median_depth_m


def parallax(T_tgt_from_ctx: Tensor, depth: Tensor) -> Tensor:
    """Parallax magnitude for a depth map: baseline over its median valid depth.

    depth: any tensor of planar z-depths from the scene, in meters. Entries that are
    not finite and positive are ignored.
    Returns a scalar tensor.
    """
    valid = depth[(depth > 0) & torch.isfinite(depth)]
    if valid.numel() == 0:
        raise ValueError("no valid depths for parallax")
    return parallax_from_median_depth(T_tgt_from_ctx, float(valid.median()))


def rotation_angle_deg(R: Tensor) -> float:
    """Angle of a 3x3 rotation matrix in degrees, in [0, 180].

    Convention free: this is the magnitude of the rotation, whatever axis it is
    about. Used to stratify the in-place rotation regime, where every pair has
    zero baseline and the viewpoint change is entirely angular.

    Computed as atan2 of the skew magnitude against the trace term rather than
    as acos of the trace term alone. The two agree mathematically and not
    numerically. Near identity the cosine is 1 minus something of order theta
    squared, so float64 rounding of the trace puts a floor of about 8.5e-7
    degrees on what acos can resolve. That floor sits above zero_rotation_tol_deg,
    which would make the zero-rotation bin a statement about arithmetic noise
    rather than about the camera. The skew term is linear in theta near zero, so
    atan2 resolves small angles to full precision, and it stays stable near 180
    degrees where the cosine is again flat.
    """
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {tuple(R.shape)}")
    # For a rotation by theta about a unit axis, R - R^T is 2 sin(theta) times
    # the axis cross-product matrix, whose Frobenius norm is sqrt(2). So the
    # norm is 2 sqrt(2) sin(theta), and the trace is 1 + 2 cos(theta).
    M = R.to(torch.float64)
    sine = float(torch.linalg.matrix_norm(M - M.mT)) / (2.0 * math.sqrt(2.0))
    cosine = (float(torch.diagonal(M).sum()) - 1.0) / 2.0
    return math.degrees(math.atan2(sine, cosine))
