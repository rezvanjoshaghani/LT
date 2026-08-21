"""Replica rendering: camera programs, per-scene manifests, depth convention, QC.

PLAN Phase 1. Three camera programs generate posed frames per scene:
- rotation: fixed position, yaw and pitch sweeps at fixed angular steps.
- translation: lateral and forward moves sized to hit target parallax values,
  where parallax is baseline over median scene depth at the base viewpoint.
- orbit: rotation around a scene anchor point at two radii, camera looking at
  the anchor.

Everything except the actual rendering is pure and runs without Habitat-Sim:
pose conventions, camera programs, depth-convention classification, manifest
read, write, and validation, and contact sheets. Habitat-Sim is imported
lazily inside the render functions only, so this module imports on any
platform and the pure parts are tested by tests/test_render_replica.py.

Conventions:
- This repository uses OpenCV camera axes: x right, y down, z forward.
  Poses are T_world_from_camera, 4x4. See geometry.py.
- Habitat-Sim uses OpenGL camera axes: x right, y up, z backward, in a
  world where +y is up. The two agree on x and flip y and z. Conversion
  happens in opencv_pose_from_opengl and opengl_pose_from_opencv, and
  nowhere else. The world frame itself is Habitat's y-up world and is
  shared by all poses of a scene; only camera axes are converted.
- Depth on disk is planar z-depth in meters, float32, one .npy per frame,
  aligned to the RGB image. The renderer never assumes what Habitat's depth
  sensor returns. It classifies the convention empirically from probe views
  (classify_depth_convention) and converts euclidean ray distance to planar
  z-depth if needed. The finding is recorded in the manifest metadata.
- Quaternions are (w, x, y, z), scalar first.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from lot.geometry import check_intrinsics, check_se3, pixel_grid

REGIMES = ("rotation", "translation", "orbit")

# Canonical Replica scene split: 18 scenes, 13 train, 5 test. The test set
# takes one scene from each family; hotel has a single scene, so the hotel
# family is test-only by construction.
REPLICA_SCENES_TEST = (
    "apartment_2",
    "frl_apartment_4",
    "hotel_0",
    "office_4",
    "room_2",
)
REPLICA_SCENES_TRAIN = (
    "apartment_0",
    "apartment_1",
    "frl_apartment_0",
    "frl_apartment_1",
    "frl_apartment_2",
    "frl_apartment_3",
    "frl_apartment_5",
    "office_0",
    "office_1",
    "office_2",
    "office_3",
    "room_0",
    "room_1",
)
REPLICA_SCENES = REPLICA_SCENES_TRAIN + REPLICA_SCENES_TEST

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# Pose conventions
# ---------------------------------------------------------------------------

def _axis_flip() -> Tensor:
    """The change of camera axes between OpenGL and OpenCV, as a 4x4.

    Rotation part diag(1, -1, -1): x is shared, y and z flip. Self-inverse.
    """
    return torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))


def opencv_pose_from_opengl(T_world_from_camera_gl: Tensor) -> Tensor:
    """Convert T_world_from_camera from OpenGL camera axes to OpenCV camera axes."""
    check_se3(T_world_from_camera_gl)
    return T_world_from_camera_gl.to(torch.float64) @ _axis_flip()


def opengl_pose_from_opencv(T_world_from_camera_cv: Tensor) -> Tensor:
    """Convert T_world_from_camera from OpenCV camera axes to OpenGL camera axes."""
    check_se3(T_world_from_camera_cv)
    return T_world_from_camera_cv.to(torch.float64) @ _axis_flip()


def quat_to_rotmat(w: float, x: float, y: float, z: float) -> Tensor:
    """Rotation matrix from a (w, x, y, z) quaternion, float64. Normalizes first."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float64,
    )


def rotmat_to_quat(R: Tensor) -> tuple[float, float, float, float]:
    """(w, x, y, z) quaternion from a 3x3 rotation matrix, w >= 0."""
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {tuple(R.shape)}")
    m = R.to(torch.float64)
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = float(m[2, 1] - m[1, 2]) / s
        y = float(m[0, 2] - m[2, 0]) / s
        z = float(m[1, 0] - m[0, 1]) / s
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = math.sqrt(1.0 + float(m[0, 0] - m[1, 1] - m[2, 2])) * 2
        w = float(m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = float(m[0, 1] + m[1, 0]) / s
        z = float(m[0, 2] + m[2, 0]) / s
    elif m[1, 1] >= m[2, 2]:
        s = math.sqrt(1.0 + float(m[1, 1] - m[0, 0] - m[2, 2])) * 2
        w = float(m[0, 2] - m[2, 0]) / s
        x = float(m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = float(m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + float(m[2, 2] - m[0, 0] - m[1, 1])) * 2
        w = float(m[1, 0] - m[0, 1]) / s
        x = float(m[0, 2] + m[2, 0]) / s
        y = float(m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    return w, x, y, z


def intrinsics_from_hfov(height: int, width: int, hfov_deg: float) -> Tensor:
    """OpenCV intrinsics for a pinhole camera with the given horizontal FOV.

    The horizontal FOV spans the full image width. Square pixels. Pixel
    centers sit at integer coordinates, so the principal point is
    ((width - 1) / 2, (height - 1) / 2).
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not 0 < hfov_deg < 180:
        raise ValueError("hfov_deg must be in (0, 180)")
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return torch.tensor(
        [
            [fx, 0.0, (width - 1) / 2.0],
            [0.0, fx, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def look_at_cv(eye: Tensor, target: Tensor, up_world: Tensor) -> Tensor:
    """T_world_from_camera (OpenCV axes) for a camera at eye looking at target.

    The camera z axis points from eye to target. The camera y axis (down)
    points opposite the given world up direction, projected off the optical
    axis. Raises ValueError when the viewing direction is parallel to up;
    pass a different up_world reference for such views.
    """
    eye = eye.to(torch.float64)
    target = target.to(torch.float64)
    up = up_world.to(torch.float64)
    z = target - eye
    zn = torch.linalg.vector_norm(z)
    if zn < 1e-9:
        raise ValueError("eye and target coincide")
    z = z / zn
    down = -up / torch.linalg.vector_norm(up)
    x = torch.linalg.cross(down, z)
    xn = torch.linalg.vector_norm(x)
    if xn < 1e-6:
        raise ValueError("viewing direction is parallel to up_world; pass another up")
    x = x / xn
    y = torch.linalg.cross(z, x)
    T = torch.eye(4, dtype=torch.float64)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = eye
    return T


def _rotation_about_camera_axis(axis: str, angle_deg: float) -> Tensor:
    """3x3 rotation about the camera's own x or y axis, float64.

    Composed as R_new = R_base @ this. About y (down): positive turns the
    optical axis toward camera +x, a turn to the right. About x (right):
    positive turns the optical axis toward camera -y, a look upward.
    """
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "y":
        return torch.tensor(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64
        )
    if axis == "x":
        return torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64
        )
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


# ---------------------------------------------------------------------------
# Depth convention
# ---------------------------------------------------------------------------

def ray_norm_map(height: int, width: int, K: Tensor) -> Tensor:
    """Per-pixel ratio of euclidean ray distance to planar z-depth.

    Entry (v, u) is the norm of K^-1 (u, v, 1), which is >= 1. Euclidean ray
    distance equals planar z-depth times this map.
    """
    check_intrinsics(K)
    uv = pixel_grid(height, width, dtype=torch.float64)
    K64 = K.to(torch.float64)
    x = (uv[..., 0] - K64[0, 2]) / K64[0, 0]
    y = (uv[..., 1] - K64[1, 2]) / K64[1, 1]
    return torch.sqrt(x * x + y * y + 1.0)


def euclidean_to_planar_depth(depth: Tensor, K: Tensor) -> Tensor:
    """Convert euclidean ray distance to planar z-depth. Shape [H, W] in, same out."""
    if depth.dim() != 2:
        raise ValueError(f"depth must be [H, W], got {tuple(depth.shape)}")
    n = ray_norm_map(depth.shape[0], depth.shape[1], K).to(depth.dtype)
    return depth / n


def _plane_residual_spread(z: Tensor, x: Tensor, y: Tensor) -> float:
    """Robust spread of z around a fitted affine plane in (x, y), over median z.

    Two-pass fit: least squares on all samples, then a refit on inliers
    within three median absolute deviations of the first residuals, so
    moderate clutter does not drag the plane. Zero for any plane.
    """
    med_z = z.median()
    if float(med_z) <= 0:
        return float("inf")
    A = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    coef = torch.linalg.lstsq(A, z.unsqueeze(-1)).solution
    r = (z.unsqueeze(-1) - A @ coef).squeeze(-1)
    med = r.median()
    keep = (r - med).abs() <= 3 * (r - med).abs().median() + 1e-9
    if int(keep.sum()) >= 16:
        A, z = A[keep], z[keep]
        coef = torch.linalg.lstsq(A, z.unsqueeze(-1)).solution
        r = (z.unsqueeze(-1) - A @ coef).squeeze(-1)
    r_med = r.median()
    return float((r - r_med).abs().median() / med_z)


def classify_depth_convention(
    depth: Tensor,
    K: Tensor,
    center_crop: float = 0.5,
    flat_tol: float = 0.01,
    margin: float = 3.0,
    max_samples: int = 20000,
) -> dict[str, Any]:
    """Decide whether a depth map of a planar surface is planar z or euclidean.

    A planar surface is an affine function of the normalized image
    coordinates under the correct planar z-depth interpretation, whatever
    its tilt. Under the wrong interpretation the ray-norm factor leaves a
    curved residual no plane can absorb. Each interpretation (the map
    itself, and the map divided by ray_norm_map) is therefore scored by the
    robust residual spread around a fitted plane; the verdict picks the
    flatter one, which must be flatter than flat_tol and at least margin
    times flatter than the alternative, else 'ambiguous'. The plane fit
    means walls seen at an angle and floors of slightly tilted scans still
    classify; the earlier constant-depth test required fronto-parallel
    views and failed on scenes a few degrees off gravity alignment.

    Statistics use the central crop only, a robust two-pass fit, and at
    most max_samples pixels. Returns a dict with keys verdict ('planar_z',
    'euclidean_ray', or 'ambiguous'), spread_planar, spread_euclidean,
    median_m, valid_fraction.
    """
    if depth.dim() != 2:
        raise ValueError(f"depth must be [H, W], got {tuple(depth.shape)}")
    if not 0 < center_crop <= 1:
        raise ValueError("center_crop must be in (0, 1]")
    h, w = depth.shape
    d = depth.to(torch.float64)
    dh = int(round(h * (1 - center_crop) / 2))
    dw = int(round(w * (1 - center_crop) / 2))
    crop = d[dh : h - dh, dw : w - dw]
    K64 = K.to(torch.float64)
    uv = pixel_grid(h, w, dtype=torch.float64)[dh : h - dh, dw : w - dw]
    x = (uv[..., 0] - K64[0, 2]) / K64[0, 0]
    y = (uv[..., 1] - K64[1, 2]) / K64[1, 1]
    n = torch.sqrt(x * x + y * y + 1.0)
    valid = torch.isfinite(crop) & (crop > 0)
    valid_fraction = float(valid.float().mean())
    result: dict[str, Any] = {
        "verdict": "ambiguous",
        "spread_planar": float("inf"),
        "spread_euclidean": float("inf"),
        "median_m": float("nan"),
        "valid_fraction": valid_fraction,
    }
    if valid_fraction < 0.5:
        return result
    dv, xv, yv, nv = crop[valid], x[valid], y[valid], n[valid]
    step = max(1, dv.numel() // max_samples)
    dv, xv, yv, nv = dv[::step], xv[::step], yv[::step], nv[::step]
    spread_planar = _plane_residual_spread(dv, xv, yv)
    spread_euclidean = _plane_residual_spread(dv / nv, xv, yv)
    result["spread_planar"] = spread_planar
    result["spread_euclidean"] = spread_euclidean
    result["median_m"] = float(dv.median())
    if spread_planar < flat_tol and spread_euclidean > margin * spread_planar:
        result["verdict"] = "planar_z"
    elif spread_euclidean < flat_tol and spread_planar > margin * spread_euclidean:
        result["verdict"] = "euclidean_ray"
    return result


# ---------------------------------------------------------------------------
# Camera programs
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PosedFrame:
    """One planned frame: regime tag, program parameters, and the camera pose."""

    regime: str
    params: dict[str, Any]
    T_world_from_camera: Tensor  # 4x4 float64, OpenCV camera axes


def program_rotation(
    T_base: Tensor,
    yaw_offsets_deg: list[float],
    pitch_offsets_deg: list[float],
) -> list[PosedFrame]:
    """In-place rotation: fixed position, yaw and pitch sweeps at fixed steps.

    The yaw sweep includes every offset given. The pitch sweep skips zero so
    the base pose appears exactly once. Positive yaw turns right, positive
    pitch looks up (see _rotation_about_camera_axis).
    """
    check_se3(T_base)
    T_base = T_base.to(torch.float64)
    frames = []
    for yaw in yaw_offsets_deg:
        T = T_base.clone()
        T[:3, :3] = T_base[:3, :3] @ _rotation_about_camera_axis("y", yaw)
        frames.append(
            PosedFrame(
                regime="rotation",
                params={"sweep": "yaw", "yaw_deg": float(yaw), "pitch_deg": 0.0},
                T_world_from_camera=T,
            )
        )
    for pitch in pitch_offsets_deg:
        if pitch == 0:
            continue
        T = T_base.clone()
        T[:3, :3] = T_base[:3, :3] @ _rotation_about_camera_axis("x", pitch)
        frames.append(
            PosedFrame(
                regime="rotation",
                params={"sweep": "pitch", "yaw_deg": 0.0, "pitch_deg": float(pitch)},
                T_world_from_camera=T,
            )
        )
    return frames


def program_translation(
    T_base: Tensor,
    parallax_values: list[float],
    median_depth_m: float,
) -> list[PosedFrame]:
    """Pure translation: lateral and forward moves sized to hit target parallax.

    Parallax is baseline over median scene depth at the base viewpoint, so
    the baseline for target parallax p is p * median_depth_m. Moves go along
    the base camera's x axis (lateral) and z axis (forward), both directions,
    rotation unchanged. Includes the zero-baseline base frame once.
    """
    check_se3(T_base)
    if median_depth_m <= 0:
        raise ValueError("median_depth_m must be positive")
    T_base = T_base.to(torch.float64)
    frames = [
        PosedFrame(
            regime="translation",
            params={
                "axis": "none",
                "sign": 0,
                "baseline_m": 0.0,
                "parallax_target": 0.0,
                "median_depth_m": float(median_depth_m),
            },
            T_world_from_camera=T_base.clone(),
        )
    ]
    axes = {"lateral": T_base[:3, 0], "forward": T_base[:3, 2]}
    for p in parallax_values:
        if p <= 0:
            raise ValueError("parallax values must be positive")
        baseline = p * median_depth_m
        for axis_name, axis_vec in axes.items():
            for sign in (1, -1):
                T = T_base.clone()
                T[:3, 3] = T_base[:3, 3] + sign * baseline * axis_vec
                frames.append(
                    PosedFrame(
                        regime="translation",
                        params={
                            "axis": axis_name,
                            "sign": sign,
                            "baseline_m": float(baseline),
                            "parallax_target": float(p),
                            "median_depth_m": float(median_depth_m),
                        },
                        T_world_from_camera=T,
                    )
                )
    return frames


def program_orbit(
    T_base: Tensor,
    anchor_distance_m: float,
    radius_scales: list[float],
    azimuth_offsets_deg: list[float],
    up_world: Tensor,
) -> list[PosedFrame]:
    """Orbit around a scene anchor point at two (or more) radii.

    The anchor sits anchor_distance_m in front of the base camera along its
    optical axis. For each radius scale s, the camera moves on a circle of
    radius s * anchor_distance_m around the anchor, in the plane spanned by
    the base camera's x and z axes, and always looks at the anchor. Azimuth
    zero at scale 1.0 reproduces the base pose.
    """
    check_se3(T_base)
    if anchor_distance_m <= 0:
        raise ValueError("anchor_distance_m must be positive")
    T_base = T_base.to(torch.float64)
    x_base = T_base[:3, 0]
    z_base = T_base[:3, 2]
    eye_base = T_base[:3, 3]
    anchor = eye_base + anchor_distance_m * z_base
    frames = []
    for scale in radius_scales:
        if scale <= 0:
            raise ValueError("radius scales must be positive")
        radius = scale * anchor_distance_m
        for az in azimuth_offsets_deg:
            a = math.radians(az)
            eye = anchor - radius * math.cos(a) * z_base + radius * math.sin(a) * x_base
            T = look_at_cv(eye, anchor, up_world)
            frames.append(
                PosedFrame(
                    regime="orbit",
                    params={
                        "radius_m": float(radius),
                        "radius_scale": float(scale),
                        "azimuth_deg": float(az),
                        "anchor_distance_m": float(anchor_distance_m),
                        "anchor_world": [float(v) for v in anchor],
                    },
                    T_world_from_camera=T,
                )
            )
    return frames


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class FrameRecord:
    """One rendered frame as stored in the per-scene manifest."""

    frame_id: str
    scene: str
    regime: str
    params: dict[str, Any]
    T_world_from_camera: Tensor  # 4x4 float64, OpenCV camera axes
    K: Tensor  # 3x3 float64
    height: int
    width: int
    rgb_path: str  # relative to the manifest's directory
    depth_path: str  # relative to the manifest's directory


@dataclasses.dataclass(frozen=True)
class Manifest:
    scene: str
    metadata: dict[str, Any]
    frames: list[FrameRecord]


def write_manifest(path: Path, manifest: Manifest) -> None:
    """Serialize a manifest to JSON. Poses and intrinsics as nested lists."""
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "scene": manifest.scene,
        "metadata": manifest.metadata,
        "frames": [
            {
                "frame_id": f.frame_id,
                "scene": f.scene,
                "regime": f.regime,
                "params": f.params,
                "T_world_from_camera": f.T_world_from_camera.to(torch.float64).tolist(),
                "K": f.K.to(torch.float64).tolist(),
                "height": f.height,
                "width": f.width,
                "rgb_path": f.rgb_path,
                "depth_path": f.depth_path,
            }
            for f in manifest.frames
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_manifest(path: Path) -> Manifest:
    """Load a manifest written by write_manifest. Raises ValueError on version mismatch."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise ValueError(f"manifest version {version}, expected {MANIFEST_VERSION}")
    frames = [
        FrameRecord(
            frame_id=f["frame_id"],
            scene=f["scene"],
            regime=f["regime"],
            params=f["params"],
            T_world_from_camera=torch.tensor(f["T_world_from_camera"], dtype=torch.float64),
            K=torch.tensor(f["K"], dtype=torch.float64),
            height=int(f["height"]),
            width=int(f["width"]),
            rgb_path=f["rgb_path"],
            depth_path=f["depth_path"],
        )
        for f in payload["frames"]
    ]
    return Manifest(scene=payload["scene"], metadata=payload["metadata"], frames=frames)


def validate_manifest(manifest: Manifest, root: Path, check_files: bool = True) -> None:
    """Validate a manifest. Raises ValueError with the first problem found.

    Checks: frames exist, frame ids unique, regimes allowed, intrinsics and
    poses well formed with orthonormal rotations, depth convention resolved
    in metadata with planar z-depth on disk, and, when check_files is True,
    every referenced file exists and the depth arrays have the declared
    shape and float32 dtype.
    """
    if not manifest.frames:
        raise ValueError("manifest has no frames")
    seen: set[str] = set()
    dc = manifest.metadata.get("depth_convention")
    if not isinstance(dc, dict):
        raise ValueError("metadata.depth_convention missing")
    if dc.get("raw_verdict") not in ("planar_z", "euclidean_ray"):
        raise ValueError(f"depth convention unresolved: {dc.get('raw_verdict')!r}")
    if dc.get("stored_depth") != "planar_z":
        raise ValueError("stored depth must be planar_z")
    root = Path(root)
    for f in manifest.frames:
        if f.frame_id in seen:
            raise ValueError(f"duplicate frame_id {f.frame_id}")
        seen.add(f.frame_id)
        if f.regime not in REGIMES:
            raise ValueError(f"{f.frame_id}: unknown regime {f.regime!r}")
        if f.scene != manifest.scene:
            raise ValueError(f"{f.frame_id}: scene {f.scene!r} != {manifest.scene!r}")
        check_intrinsics(f.K)
        check_se3(f.T_world_from_camera)
        R = f.T_world_from_camera[:3, :3]
        if not torch.allclose(R @ R.mT, torch.eye(3, dtype=R.dtype), atol=1e-5):
            raise ValueError(f"{f.frame_id}: rotation not orthonormal")
        if abs(float(torch.linalg.det(R)) - 1.0) > 1e-5:
            raise ValueError(f"{f.frame_id}: rotation determinant != 1")
        if f.height <= 0 or f.width <= 0:
            raise ValueError(f"{f.frame_id}: bad image size")
        if check_files:
            rgb = root / f.rgb_path
            depth = root / f.depth_path
            if not rgb.is_file():
                raise ValueError(f"{f.frame_id}: missing rgb file {rgb}")
            if not depth.is_file():
                raise ValueError(f"{f.frame_id}: missing depth file {depth}")
            arr = np.load(depth)
            if arr.shape != (f.height, f.width):
                raise ValueError(f"{f.frame_id}: depth shape {arr.shape}")
            if arr.dtype != np.float32:
                raise ValueError(f"{f.frame_id}: depth dtype {arr.dtype}")


# ---------------------------------------------------------------------------
# QC contact sheets
# ---------------------------------------------------------------------------

def write_contact_sheet(
    path: Path,
    entries: list[tuple[str, np.ndarray, np.ndarray]],
    ncols: int = 6,
) -> None:
    """Write a QC grid of frames: RGB on top, depth (viridis) below, per tile.

    entries: list of (title, rgb [H, W, 3] uint8, depth [H, W] float).
    Invalid depth (non-positive or non-finite) renders black. Each depth tile
    is normalized on its own and annotated with its min and max in meters.
    """
    if not entries:
        raise ValueError("no entries for contact sheet")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = min(ncols, len(entries))
    nrows = math.ceil(len(entries) / ncols)
    fig, axes = plt.subplots(
        2 * nrows,
        ncols,
        figsize=(2.4 * ncols, 5.0 * nrows),
        squeeze=False,
    )
    for ax_row in axes:
        for ax in ax_row:
            ax.set_axis_off()
    # Depth tiles are colormapped by hand into RGBA so the result does not
    # depend on the matplotlib version. Colormap.copy and set_bad only
    # exist from matplotlib 3.4 on, and cluster environments resolve older
    # builds.
    cmap = plt.get_cmap("viridis")
    for i, (title, rgb, depth) in enumerate(entries):
        r, c = divmod(i, ncols)
        ax_rgb = axes[2 * r][c]
        ax_depth = axes[2 * r + 1][c]
        ax_rgb.imshow(rgb)
        ax_rgb.set_title(title, fontsize=6)
        d = np.asarray(depth, dtype=np.float64)
        valid = np.isfinite(d) & (d > 0)
        rgba = np.zeros(d.shape + (4,))
        rgba[..., 3] = 1.0  # invalid pixels render black
        if valid.any():
            vmin = float(d[valid].min())
            vmax = float(d[valid].max())
            span = vmax - vmin if vmax > vmin else 1.0
            norm = np.clip((np.where(valid, d, vmin) - vmin) / span, 0.0, 1.0)
            rgba = np.asarray(cmap(norm))
            rgba[~valid] = (0.0, 0.0, 0.0, 1.0)
            ax_depth.set_title(f"depth {vmin:.2f}..{vmax:.2f} m", fontsize=6)
        ax_depth.imshow(rgba)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RenderConfig:
    """One rendering experiment. Loaded from a yaml file, one file per run."""

    replica_root: Path
    output_root: Path
    scenes: list[str]
    seed: int = 0
    image_height: int = 518
    image_width: int = 518
    hfov_deg: float = 90.0
    eye_height_m: float = 1.5
    scene_relpath: str = "mesh.ply"
    navmesh_relpath: str = "habitat/mesh_semantic.navmesh"
    gpu_device_id: int = 0
    viewpoints_per_scene: int = 6
    max_viewpoint_attempts: int = 400
    min_median_depth_m: float = 1.0
    max_median_depth_m: float = 8.0
    min_clearance_m: float = 0.5
    min_valid_fraction: float = 0.7
    min_viewpoint_separation_m: float = 0.75
    yaw_offsets_deg: list[float] = dataclasses.field(
        default_factory=lambda: [-30.0, -22.5, -15.0, -7.5, 0.0, 7.5, 15.0, 22.5, 30.0]
    )
    pitch_offsets_deg: list[float] = dataclasses.field(
        default_factory=lambda: [-15.0, -7.5, 7.5, 15.0]
    )
    parallax_values: list[float] = dataclasses.field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.4]
    )
    orbit_radius_scales: list[float] = dataclasses.field(
        default_factory=lambda: [0.6, 1.0]
    )
    orbit_azimuth_offsets_deg: list[float] = dataclasses.field(
        default_factory=lambda: [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    )
    anchor_distance_min_m: float = 1.0
    anchor_distance_max_m: float = 6.0
    qc_frames_per_regime: int = 12

    def __post_init__(self) -> None:
        self.replica_root = Path(self.replica_root)
        self.output_root = Path(self.output_root)
        if not self.scenes:
            raise ValueError("config lists no scenes")
        unknown = [s for s in self.scenes if s not in REPLICA_SCENES]
        if unknown:
            raise ValueError(f"unknown Replica scenes: {unknown}")
        if self.image_height % 14 or self.image_width % 14:
            raise ValueError("image size must be a multiple of the patch size 14")


def load_config(path: Path) -> RenderConfig:
    """Load a RenderConfig from yaml. Unknown keys are an error, not a warning."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    allowed = {f.name for f in dataclasses.fields(RenderConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    missing = [k for k in ("replica_root", "output_root", "scenes") if k not in raw]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    return RenderConfig(**raw)


def _config_echo(cfg: RenderConfig) -> dict[str, Any]:
    d = dataclasses.asdict(cfg)
    d["replica_root"] = str(d["replica_root"])
    d["output_root"] = str(d["output_root"])
    return d


# ---------------------------------------------------------------------------
# Habitat rendering. Everything below needs habitat_sim and runs on Linux.
# ---------------------------------------------------------------------------

def _import_habitat():
    try:
        import habitat_sim
    except ImportError as e:
        raise RuntimeError(
            "habitat_sim is not installed. Rendering runs on Linux; install per "
            "scripts/README.md and run there. All non-rendering functions of "
            "this module work without it."
        ) from e
    return habitat_sim


def scene_seed(seed: int, scene: str) -> int:
    """Deterministic per-scene seed derived from the run seed and scene name."""
    return (seed * 100003 + zlib.crc32(scene.encode("utf-8"))) % (2**31)


def make_sim(cfg: RenderConfig, scene: str):
    """Create a Habitat simulator for one Replica scene with color and depth sensors.

    Both sensors are mounted at the agent origin with zero rotation, so the
    sensor pose equals the agent pose. Returns (sim, navmesh_source) where
    navmesh_source records how the navmesh was obtained: 'simulator' when
    Habitat loaded one on its own, 'dataset' when loaded from
    cfg.navmesh_relpath, 'recomputed' when computed from the scene mesh
    because the dataset ships none.
    """
    habitat_sim = _import_habitat()
    scene_path = cfg.replica_root / scene / cfg.scene_relpath
    if not scene_path.is_file():
        raise FileNotFoundError(f"scene mesh not found: {scene_path}")
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene_path)
    backend.gpu_device_id = cfg.gpu_device_id
    backend.enable_physics = False

    def sensor_spec(uuid: str, sensor_type):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [cfg.image_height, cfg.image_width]
        spec.hfov = cfg.hfov_deg
        spec.position = [0.0, 0.0, 0.0]
        spec.orientation = [0.0, 0.0, 0.0]
        return spec

    agent_cfg = habitat_sim.agent.AgentConfiguration(
        sensor_specifications=[
            sensor_spec("color", habitat_sim.SensorType.COLOR),
            sensor_spec("depth", habitat_sim.SensorType.DEPTH),
        ]
    )
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    seed = scene_seed(cfg.seed, scene)
    sim.seed(seed)
    navmesh_source = "simulator"
    navmesh_path = cfg.replica_root / scene / cfg.navmesh_relpath
    if not sim.pathfinder.is_loaded and navmesh_path.is_file():
        sim.pathfinder.load_nav_mesh(str(navmesh_path))
        navmesh_source = "dataset"
    if not sim.pathfinder.is_loaded:
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_height = cfg.eye_height_m
        settings.agent_radius = 0.2
        if not sim.recompute_navmesh(sim.pathfinder, settings):
            raise RuntimeError(
                f"no navmesh for {scene} at {navmesh_path} and recompute "
                "failed. Viewpoint sampling needs a navmesh."
            )
        navmesh_source = "recomputed"
    sim.pathfinder.seed(seed)
    return sim, navmesh_source


def render_at_pose(
    sim, T_world_from_camera_cv: Tensor
) -> tuple[np.ndarray, np.ndarray, Tensor]:
    """Render RGB and raw depth at a pose given in OpenCV camera axes.

    Returns (rgb [H, W, 3] uint8, depth_raw [H, W] float32 as the sensor
    reports it, T_world_from_camera_cv read back from the simulator). The
    read-back pose is the ground truth stored in the manifest.
    """
    habitat_sim = _import_habitat()
    import quaternion  # numpy-quaternion, installed with habitat_sim

    T_gl = opengl_pose_from_opencv(T_world_from_camera_cv)
    w, x, y, z = rotmat_to_quat(T_gl[:3, :3])
    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = T_gl[:3, 3].numpy().astype(np.float32)
    state.rotation = quaternion.quaternion(w, x, y, z)
    agent.set_state(state, reset_sensors=True)
    obs = sim.get_sensor_observations()
    rgb = np.ascontiguousarray(obs["color"][..., :3]).astype(np.uint8)
    depth_raw = np.ascontiguousarray(obs["depth"]).astype(np.float32)
    sensor_state = agent.get_state().sensor_states["color"]
    q = sensor_state.rotation
    T_rb = torch.eye(4, dtype=torch.float64)
    T_rb[:3, :3] = quat_to_rotmat(float(q.w), float(q.x), float(q.y), float(q.z))
    T_rb[:3, 3] = torch.tensor(np.asarray(sensor_state.position, dtype=np.float64))
    return rgb, depth_raw, opencv_pose_from_opengl(T_rb)


_UP_WORLD = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)


def probe_depth_convention(
    sim, K: Tensor, eye_height_m: float, probe_dir: Path, n_points: int = 3
) -> tuple[dict[str, Any], bool]:
    """Resolve the depth convention empirically from probe views.

    From up to n_points navigable points, renders a floor view, a ceiling
    view, and four horizontal views, and classifies each with
    classify_depth_convention. The classifier fits a plane, so obliquely
    seen walls and floors of slightly tilted scans still vote. All
    confident verdicts must agree, and probing stops early once three
    agree. Every probe render, and the classification stats as
    classification.json, are saved under probe_dir for the QC record.
    Returns (metadata dict, convert_needed). Raises RuntimeError when no
    probe is confident or the verdicts disagree; per PLAN, that is an
    anomaly to report, not to tune away.
    """
    from PIL import Image

    probe_dir.mkdir(parents=True, exist_ok=True)
    up_ref = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    down = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)
    probes: list[dict[str, Any]] = []
    verdicts: list[str] = []
    points_used = 0
    try:
        for pi in range(n_points):
            points_used = pi + 1
            point = np.asarray(
                sim.pathfinder.get_random_navigable_point(), dtype=np.float64
            )
            eye = torch.tensor(point) + torch.tensor(
                [0.0, eye_height_m, 0.0], dtype=torch.float64
            )
            poses = {
                f"p{pi}_floor_down": look_at_cv(eye, eye + down, up_ref),
                f"p{pi}_ceiling_up": look_at_cv(eye, eye - down, up_ref),
            }
            for yaw in (0.0, 90.0, 180.0, 270.0):
                a = math.radians(yaw)
                direction = torch.tensor(
                    [math.sin(a), 0.0, math.cos(a)], dtype=torch.float64
                )
                poses[f"p{pi}_wall_yaw{int(yaw):03d}"] = look_at_cv(
                    eye, eye + direction, _UP_WORLD
                )
            for name, pose in poses.items():
                rgb, depth_raw, _ = render_at_pose(sim, pose)
                Image.fromarray(rgb).save(probe_dir / f"{name}.png")
                np.save(probe_dir / f"{name}_depth_raw.npy", depth_raw)
                stats = classify_depth_convention(torch.from_numpy(depth_raw), K)
                stats["name"] = name
                probes.append(stats)
                if stats["verdict"] != "ambiguous":
                    verdicts.append(stats["verdict"])
            if len(verdicts) >= 3 and len(set(verdicts)) == 1:
                break
    finally:
        (probe_dir / "classification.json").write_text(
            json.dumps(probes, indent=1), encoding="utf-8"
        )
    if not verdicts:
        raise RuntimeError(
            f"depth convention: every probe ambiguous; inspect {probe_dir}"
        )
    if len(set(verdicts)) != 1:
        raise RuntimeError(
            f"depth convention: probes disagree {sorted(set(verdicts))}; "
            f"inspect {probe_dir}"
        )
    raw_verdict = verdicts[0]
    metadata = {
        "raw_verdict": raw_verdict,
        "converted_to_planar": raw_verdict == "euclidean_ray",
        "stored_depth": "planar_z",
        "probe_points": points_used,
        "probes": probes,
    }
    return metadata, raw_verdict == "euclidean_ray"


def sample_viewpoints(
    sim,
    cfg: RenderConfig,
    K: Tensor,
    rng: np.random.Generator,
    to_planar: Callable[[np.ndarray], np.ndarray],
) -> list[dict[str, Any]]:
    """Sample base viewpoints on the navmesh with a depth quality filter.

    A candidate stands eye_height_m above a random navigable point with a
    random horizontal heading. It is accepted when the planar depth at its
    base view has enough valid pixels, a median inside the configured range,
    enough clearance in the central crop, and enough distance from already
    accepted viewpoints. Returns viewpoint dicts with the base pose and the
    median depth used to size translation and orbit programs.
    """
    accepted: list[dict[str, Any]] = []
    positions: list[np.ndarray] = []
    attempts = 0
    while len(accepted) < cfg.viewpoints_per_scene and attempts < cfg.max_viewpoint_attempts:
        attempts += 1
        point = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float64)
        if positions and min(
            float(np.linalg.norm(point - p)) for p in positions
        ) < cfg.min_viewpoint_separation_m:
            continue
        yaw = float(rng.uniform(0.0, 360.0))
        a = math.radians(yaw)
        eye = torch.tensor(point, dtype=torch.float64) + torch.tensor(
            [0.0, cfg.eye_height_m, 0.0], dtype=torch.float64
        )
        direction = torch.tensor([math.sin(a), 0.0, math.cos(a)], dtype=torch.float64)
        T_base = look_at_cv(eye, eye + direction, _UP_WORLD)
        _, depth_raw, _ = render_at_pose(sim, T_base)
        depth = to_planar(depth_raw)
        valid = np.isfinite(depth) & (depth > 0.05)
        valid_fraction = float(valid.mean())
        if valid_fraction < cfg.min_valid_fraction:
            continue
        median_depth = float(np.median(depth[valid]))
        if not cfg.min_median_depth_m <= median_depth <= cfg.max_median_depth_m:
            continue
        h, w = depth.shape
        ch, cw = h // 4, w // 4
        center = depth[ch : h - ch, cw : w - cw]
        center_valid = center[np.isfinite(center) & (center > 0.05)]
        if center_valid.size == 0 or float(center_valid.min()) < cfg.min_clearance_m:
            continue
        accepted.append(
            {
                "index": len(accepted),
                "position": [float(v) for v in point],
                "yaw_deg": yaw,
                "median_depth_m": median_depth,
                "valid_fraction": valid_fraction,
                "T_base": T_base,
            }
        )
        positions.append(point)
    if not accepted:
        raise RuntimeError(
            f"no viewpoint passed the quality filter in {attempts} attempts"
        )
    return accepted


def _viewpoint_programs(cfg: RenderConfig, vp: dict[str, Any]) -> list[PosedFrame]:
    """The three camera programs for one accepted viewpoint."""
    T_base = vp["T_base"]
    anchor_distance = min(
        max(vp["median_depth_m"], cfg.anchor_distance_min_m), cfg.anchor_distance_max_m
    )
    frames: list[PosedFrame] = []
    frames += program_rotation(T_base, cfg.yaw_offsets_deg, cfg.pitch_offsets_deg)
    frames += program_translation(T_base, cfg.parallax_values, vp["median_depth_m"])
    frames += program_orbit(
        T_base,
        anchor_distance,
        cfg.orbit_radius_scales,
        cfg.orbit_azimuth_offsets_deg,
        _UP_WORLD,
    )
    return frames


def render_scene(cfg: RenderConfig, scene: str) -> Path:
    """Render one Replica scene end to end and write its manifest and QC sheets.

    Never overwrites: refuses to run when the scene's manifest already
    exists. Returns the manifest path.
    """
    from PIL import Image

    out_dir = cfg.output_root / scene
    manifest_path = out_dir / MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(
            f"{manifest_path} exists; outputs are never overwritten. Move or "
            "delete the scene directory to re-render."
        )
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    K = intrinsics_from_hfov(cfg.image_height, cfg.image_width, cfg.hfov_deg)
    sim, navmesh_source = make_sim(cfg, scene)
    try:
        depth_meta, convert = probe_depth_convention(
            sim, K, cfg.eye_height_m, out_dir / "probes"
        )
        print(f"[{scene}] depth convention: {depth_meta['raw_verdict']}"
              f" (convert={convert})")

        def to_planar(depth_raw: np.ndarray) -> np.ndarray:
            d = torch.from_numpy(depth_raw.astype(np.float64))
            if convert:
                d = euclidean_to_planar_depth(d, K)
            return d.to(torch.float32).numpy()

        rng = np.random.default_rng(scene_seed(cfg.seed, scene))
        viewpoints = sample_viewpoints(sim, cfg, K, rng, to_planar)
        print(f"[{scene}] {len(viewpoints)} viewpoints accepted")

        records: list[FrameRecord] = []
        pose_err_max = 0.0
        for vp in viewpoints:
            counters: dict[str, int] = {}
            for frame in _viewpoint_programs(cfg, vp):
                k = counters.get(frame.regime, 0)
                counters[frame.regime] = k + 1
                frame_id = f"{scene}_vp{vp['index']:02d}_{frame.regime}_{k:03d}"
                rgb, depth_raw, T_readback = render_at_pose(
                    sim, frame.T_world_from_camera
                )
                pose_err = float(
                    (T_readback - frame.T_world_from_camera).abs().max()
                )
                pose_err_max = max(pose_err_max, pose_err)
                depth = to_planar(depth_raw)
                rgb_rel = f"rgb/{frame_id}.png"
                depth_rel = f"depth/{frame_id}.npy"
                Image.fromarray(rgb).save(out_dir / rgb_rel)
                np.save(out_dir / depth_rel, depth)
                params = dict(frame.params)
                params["viewpoint"] = vp["index"]
                records.append(
                    FrameRecord(
                        frame_id=frame_id,
                        scene=scene,
                        regime=frame.regime,
                        params=params,
                        T_world_from_camera=T_readback,
                        K=K,
                        height=cfg.image_height,
                        width=cfg.image_width,
                        rgb_path=rgb_rel,
                        depth_path=depth_rel,
                    )
                )
    finally:
        sim.close()

    habitat_sim = _import_habitat()
    per_regime = {r: sum(1 for f in records if f.regime == r) for r in REGIMES}
    metadata = {
        "scene": scene,
        "seed": cfg.seed,
        "scene_seed": scene_seed(cfg.seed, scene),
        "habitat_sim_version": getattr(habitat_sim, "__version__", "unknown"),
        "navmesh": navmesh_source,
        "depth_convention": depth_meta,
        "viewpoints": [
            {key: v for key, v in vp.items() if key != "T_base"} for vp in viewpoints
        ],
        "frames_per_regime": per_regime,
        "pose_readback_max_abs_err": pose_err_max,
        "config": _config_echo(cfg),
    }
    manifest = Manifest(scene=scene, metadata=metadata, frames=records)
    write_manifest(manifest_path, manifest)
    validate_manifest(load_manifest(manifest_path), out_dir, check_files=True)
    write_scene_qc(out_dir, manifest, cfg.qc_frames_per_regime)
    print(f"[{scene}] {len(records)} frames {per_regime}; manifest {manifest_path}")
    return manifest_path


def write_scene_qc(out_dir: Path, manifest: Manifest, frames_per_regime: int) -> None:
    """Write one contact sheet per regime from the files the manifest references."""
    from PIL import Image

    for regime in REGIMES:
        frames = [f for f in manifest.frames if f.regime == regime]
        if not frames:
            continue
        step = max(1, len(frames) // frames_per_regime)
        picked = frames[::step][:frames_per_regime]
        entries = []
        for f in picked:
            rgb = np.asarray(Image.open(out_dir / f.rgb_path))[..., :3]
            depth = np.load(out_dir / f.depth_path)
            entries.append((f.frame_id, rgb, depth))
        write_contact_sheet(out_dir / "qc" / f"qc_{regime}.png", entries)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render Replica scenes for LT.")
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scene", type=str, help="render a single scene by name")
    group.add_argument(
        "--scene-index",
        type=int,
        help="render a single scene by index into the config scene list "
        "(for SLURM array jobs)",
    )
    parser.add_argument(
        "--list-scenes", action="store_true", help="print the scene list and exit"
    )
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--validate-only",
        action="store_true",
        help="validate existing manifests instead of rendering",
    )
    action_group.add_argument(
        "--qc-only",
        action="store_true",
        help="regenerate QC contact sheets from existing manifests",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.list_scenes:
        for i, s in enumerate(cfg.scenes):
            print(i, s)
        return
    if args.scene is not None:
        if args.scene not in cfg.scenes:
            raise SystemExit(f"scene {args.scene!r} not in config scene list")
        scenes = [args.scene]
    elif args.scene_index is not None:
        scenes = [cfg.scenes[args.scene_index]]
    else:
        scenes = list(cfg.scenes)
    if args.validate_only:
        failures = []
        for scene in scenes:
            out_dir = cfg.output_root / scene
            try:
                validate_manifest(
                    load_manifest(out_dir / MANIFEST_NAME), out_dir, check_files=True
                )
            except FileNotFoundError:
                print(f"[{scene}] MISSING: no manifest at {out_dir / MANIFEST_NAME}")
                failures.append(scene)
            except ValueError as e:
                print(f"[{scene}] INVALID: {e}")
                failures.append(scene)
            else:
                print(f"[{scene}] manifest valid")
        if failures:
            raise SystemExit(
                f"{len(failures)} of {len(scenes)} scenes failed validation: "
                + ", ".join(failures)
            )
        print(f"all {len(scenes)} scenes valid")
        return
    for scene in scenes:
        if args.qc_only:
            out_dir = cfg.output_root / scene
            manifest = load_manifest(out_dir / MANIFEST_NAME)
            validate_manifest(manifest, out_dir, check_files=True)
            write_scene_qc(out_dir, manifest, cfg.qc_frames_per_regime)
            print(f"[{scene}] QC sheets written under {out_dir / 'qc'}")
        else:
            render_scene(cfg, scene)


if __name__ == "__main__":
    main()
