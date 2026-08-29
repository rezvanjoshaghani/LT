"""Phase 4, rung 1: the estimated-geometry tax. PROTOCOL 4.1 through 4.9.

Replace Oracle-Transport's ground-truth depth with VGGT estimated depth under
the alignment ladder of PROTOCOL 4.3, keep everything else frozen from Phase 3,
and measure the additional transport loss. No training, no learned predictor.

What stays frozen from Phase 3, byte for byte: the pair sample and its seed,
the correspondence universe and every sample_id, the ground-truth visibility
buckets, the transport and rasterization defaults, the metrics, the floors,
the mean vector, and the bin config. Estimated depth changes only where a
correspondence lands and which samples remain transportable. PROTOCOL 4.9.

Three validity notions, kept distinct and never sharing a mask name:

- transport pre-validity (5a) and post-alignment transport validity (5c):
  finite, positive estimated depth, plus the frozen confidence rule. The rule
  is frozen as vggt_confidence_threshold null, meaning no confidence gating
  (Amendment A3). Positive scaling cannot change this set, so it is identical
  across the no-alignment, scene-scale, and image-scale levels; that identity
  is asserted per pair and a difference stops the run.
- calibration-population validity (5b): transport pre-validity plus finite,
  positive ground-truth depth. Used only to estimate alignment parameters and
  never narrows the transport population.
- scoring validity (5d): the prediction must also land. On the per-point path
  the estimated warp needs positive depth in the context camera and must fall
  inside the context sampling box; on the splat path the cell needs estimated
  coverage plus the frozen ground-truth co-visibility rule. Landing depends on
  the level when translation is nonzero, so the scored set is level-dependent
  and is reported as coverage, never asserted equal across levels.

Alignment application, Amendment A4: each level yields one transform per
pair. Level 1 is the leave-target-out scene scalar; Level 2 and the affine
sensitivity are estimated from the pair's context image alone. The transform
applies to whichever VGGT map serves as a transport input on a path: the
context frame's map on the splat path, the target frame's map on the
per-point path. The target frame's ground truth never enters any estimator,
and a per-record assertion enforces it (PROTOCOL 4.3).

The pure-rotation gates of PROTOCOL 4.5 run inline for every rotation pair at
every level, against the tolerances Amendment A3 added to the analysis
config. A breach raises Phase4GateError with the evidence the execution plan
demands and stops the run. The forced-collision machinery is an isolated copy
of the transport plan that cannot change ordinary transport; its
forcing-disabled form is asserted equal to lot.transport.transport_plan once
per scene and permanently in the test suite.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .analysis_config import DEFAULT_CONFIG_PATH, AnalysisConfig, load_analysis_config
from .correspondence import _in_box, _sampling_box
from .datasets import (
    assert_translation_parallax_floor,
    load_scene_pairs,
    scene_split,
    subsample_by_stratum,
)
from .encoders import (
    DEPTH_NAME,
    PATCH_SIZE,
    archive_digest,
    cache_dir,
    load_cache_meta,
    sample_features_bilinear,
    sample_map_bilinear,
)
from .evaluate import (
    MEAN_FEATURE,
    NO_WARP_COPY,
    ORACLE_TRANSPORT,
    PER_POINT,
    SPLAT_POOL,
    _SceneCache,
    agreement_metrics,
    git_commit,
    load_or_build_mean_vector,
    pack_mask,
    pair_geometry,
    read_run_metadata,
    write_rows,
)
from .geometry import invert_se3, pixel_grid, project, relative_pose, transform_points, unproject
from .render_replica import MANIFEST_NAME, load_manifest, validate_manifest
from .transport import TIE_RELATIVE_EPS, apply_transport_plan, transport_plan
from .visibility import fraction_per_patch

PHASE4_VERSION = 1

# Alignment levels in ladder order. Names appear verbatim in rows, tables,
# and figures, per the method-name rule in CLAUDE.md.
VGGT_NO_ALIGN = "VGGT-NoAlign"
VGGT_SCENE_SCALE = "VGGT-SceneScale"
VGGT_IMAGE_SCALE = "VGGT-ImageScale"
VGGT_IMAGE_AFFINE = "VGGT-ImageAffine"

LEVELS = (
    ("none", VGGT_NO_ALIGN),
    ("scene", VGGT_SCENE_SCALE),
    ("image", VGGT_IMAGE_SCALE),
    ("affine", VGGT_IMAGE_AFFINE),
)
MULTIPLICATIVE_LEVELS = ("none", "scene", "image")

GT_LEVEL = "gt"
POPULATION_FULL = "full"
POPULATION_MATCHED = "matched"


class Phase4GateError(RuntimeError):
    """A Phase 4 correctness gate failed. The run stops here."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Phase4Config:
    """One Phase 4 run. Numeric constants come from the analysis config."""

    experiment_name: str
    renders_root: Path
    cache_root: Path
    output_root: Path
    scenes: list[str]
    feature_encoder: str = "dinov2_vitb14"
    depth_encoder: str = "vggt_1b"
    # The Phase 3 run directory whose provenance-validated mean vector this
    # run reuses. PROTOCOL 4.1 freezes the floors and the centering statistic.
    mean_vector_dir: Path = Path("outputs/experiment_zero")
    seed: int = 0
    analysis_config: Path = DEFAULT_CONFIG_PATH
    geometry_dtype: str = "float32"

    def __post_init__(self) -> None:
        self.renders_root = Path(self.renders_root)
        self.cache_root = Path(self.cache_root)
        self.output_root = Path(self.output_root)
        self.mean_vector_dir = Path(self.mean_vector_dir)
        self.analysis_config = Path(self.analysis_config)
        if not self.scenes:
            raise ValueError("config lists no scenes")
        if self.geometry_dtype not in ("float32", "float64"):
            raise ValueError("geometry_dtype must be float32 or float64")

    @property
    def eval_dir(self) -> Path:
        return self.output_root / self.experiment_name / "eval"

    @property
    def evidence_dir(self) -> Path:
        return self.output_root / self.experiment_name / "evidence"

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float32 if self.geometry_dtype == "float32" else torch.float64


def load_phase4_config(path: Path) -> Phase4Config:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    allowed = {f.name for f in dataclasses.fields(Phase4Config)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return Phase4Config(**raw)


def phase4_measurement_digest(analysis: AnalysisConfig) -> str:
    """Identity of the values that decide what Phase 4 rows contain.

    The Phase 3 measurement digest is left untouched so the corrected Phase 3
    parquet stays readable. Phase 4 extends that identity with the one new
    value that decides Phase 4 validity, the frozen confidence rule.
    """
    payload = json.dumps(
        {
            "phase3_measurement": analysis.measurement_digest(),
            "vggt_confidence_threshold": analysis.vggt_confidence_threshold,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Estimated depth: cache, resampling, convention, validity
# ---------------------------------------------------------------------------

def load_depth_archive(cache_root: Path, encoder: str, scene: str) -> dict[str, Any]:
    """Open one scene's estimated-depth cache, verified against its digest.

    Returns {"depth": {frame_id: fp32 [H, W]}, "conf": {frame_id: fp32 | None},
    "meta": cache meta}. The digest is checked over the bytes actually read,
    the same discipline the feature cache gets in evaluation.
    """
    meta = load_cache_meta(cache_root, encoder, scene)
    if not meta.get("has_depth"):
        raise ValueError(f"{encoder} / {scene}: cache exports no depth")
    path = cache_dir(cache_root, encoder, scene) / DEPTH_NAME
    recorded = meta.get("depth_digest")
    found = archive_digest(path)
    if recorded != found:
        raise ValueError(
            f"{encoder} / {scene}: depth digest {found} does not match the "
            f"{recorded} recorded when the cache was written"
        )
    depth: dict[str, np.ndarray] = {}
    conf: dict[str, np.ndarray | None] = {}
    with np.load(path) as archive:
        for name in archive.files:
            if name.endswith("__conf"):
                continue
            depth[name] = archive[name].astype(np.float32)
            conf_name = f"{name}__conf"
            conf[name] = (
                archive[conf_name].astype(np.float32) if conf_name in archive.files else None
            )
    return {"depth": depth, "conf": conf, "meta": meta}


def resample_depth_nearest(
    depth: np.ndarray, dst_hw: tuple[int, int]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bring estimated depth to render resolution by nearest neighbor.

    PROTOCOL 4.1 freezes nearest-neighbor resampling and forbids bilinear:
    interpolation across a depth discontinuity manufactures a depth that lies
    on no surface. The current caches store depth at render resolution
    already, so this normally records the asserted identity; the general rule
    is implemented so the assertion is a fact rather than an assumption.
    Aspect-ratio changes and crops are refused, because the intrinsics at
    transport resolution would no longer describe the pixels.
    """
    src_h, src_w = depth.shape
    dst_h, dst_w = dst_hw
    record = {
        "src_hw": [int(src_h), int(src_w)],
        "dst_hw": [int(dst_h), int(dst_w)],
        "crop": None,
        "method": "identity" if (src_h, src_w) == (dst_h, dst_w) else "nearest",
    }
    if (src_h, src_w) == (dst_h, dst_w):
        return depth, record
    if src_h * dst_w != src_w * dst_h:
        raise ValueError(
            f"resampling {depth.shape} to {dst_hw} changes the aspect ratio; "
            "the intrinsics at transport resolution would not describe the pixels"
        )
    rows = np.floor((np.arange(dst_h) + 0.5) * src_h / dst_h).astype(np.int64)
    cols = np.floor((np.arange(dst_w) + 0.5) * src_w / dst_w).astype(np.int64)
    return depth[rows][:, cols], record


def secant_map(K: Tensor, height: int, width: int) -> np.ndarray:
    """Per-pixel secant of the angle from the optical axis. [H, W] float64."""
    K64 = K.to(torch.float64)
    ray_x = (torch.arange(width, dtype=torch.float64) - K64[0, 2]) / K64[0, 0]
    ray_y = (torch.arange(height, dtype=torch.float64) - K64[1, 2]) / K64[1, 1]
    yy, xx = torch.meshgrid(ray_y, ray_x, indexing="ij")
    return torch.sqrt(1.0 + xx * xx + yy * yy).numpy()


def secant_regression(
    vggt_depth: np.ndarray, gt_depth: np.ndarray, K: Tensor, threshold: float
) -> dict[str, Any]:
    """PROTOCOL 4.1's deterministic depth-convention test for one frame.

    Regress the per-pixel ratio of estimated to ground-truth depth against
    the secant of the pixel's angle from the optical axis, over pixels where
    both are finite and positive. A planar map gives a flat ratio; a ray map
    gives ratio proportional to the secant. The classification compares the
    fitted slope against the frozen threshold times the fitted ratio at the
    optical axis. No frame or region is chosen by inspection.
    """
    height, width = gt_depth.shape
    sec = secant_map(K, height, width)
    valid = (
        np.isfinite(vggt_depth) & (vggt_depth > 0) & np.isfinite(gt_depth) & (gt_depth > 0)
    )
    if int(valid.sum()) < 16:
        return {"verdict": "unresolved", "n": int(valid.sum())}
    ratio = (vggt_depth[valid] / gt_depth[valid]).astype(np.float64)
    x = sec[valid]
    design = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(design, ratio, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    center_scale = intercept + slope
    if not center_scale > 0:
        return {
            "verdict": "unresolved",
            "n": int(valid.sum()),
            "intercept": intercept,
            "slope": slope,
        }
    verdict = "ray_distance" if slope > threshold * center_scale else "planar_z"
    return {
        "verdict": verdict,
        "n": int(valid.sum()),
        "intercept": intercept,
        "slope": slope,
        "slope_share_of_center_scale": slope / center_scale,
        "threshold": threshold,
    }


def transport_prevalid(
    depth: np.ndarray, conf: np.ndarray | None, analysis: AnalysisConfig
) -> np.ndarray:
    """Validity 5a and 5c: finite, positive, and the frozen confidence rule."""
    valid = np.isfinite(depth) & (depth > 0)
    threshold = analysis.vggt_confidence_threshold
    if threshold is not None:
        if conf is None:
            raise ValueError(
                "a confidence threshold is frozen but the cache carries no confidence"
            )
        valid &= conf >= threshold
    return valid


def calibration_valid(prevalid: np.ndarray, gt_depth: np.ndarray) -> np.ndarray:
    """Validity 5b: transport pre-validity plus valid ground truth."""
    return prevalid & np.isfinite(gt_depth) & (gt_depth > 0)


# ---------------------------------------------------------------------------
# Alignment ladder, PROTOCOL 4.3
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FrameCalibration:
    """Per-frame calibration facts, computed once per scene."""

    ratios: np.ndarray          # float64, gt over vggt on calibration-valid pixels
    n_calibration: int
    image_scale: float          # median of ratios, nan when empty
    affine_s: float
    affine_b: float
    affine_n: int
    affine_failed: bool


def frame_calibration(
    vggt_depth: np.ndarray, gt_depth: np.ndarray, prevalid: np.ndarray
) -> FrameCalibration:
    """Level 2 and affine estimators for one frame.

    The affine fit is ordinary unweighted least squares of ground truth on
    estimated depth: no confidence weighting, no robust loss, no clipping,
    s unconstrained during fitting. A nonpositive fitted s marks the frame's
    affine row failed; it is reported, never repaired.
    """
    calib = calibration_valid(prevalid, gt_depth)
    n = int(calib.sum())
    nan = float("nan")
    if n == 0:
        return FrameCalibration(np.zeros(0), 0, nan, nan, nan, 0, True)
    est = vggt_depth[calib].astype(np.float64)
    gt = gt_depth[calib].astype(np.float64)
    ratios = gt / est
    scale = float(np.median(ratios))
    if n < 2:
        return FrameCalibration(ratios, n, scale, nan, nan, n, True)
    sum_e, sum_g = float(est.sum()), float(gt.sum())
    sum_ee, sum_eg = float(est @ est), float(est @ gt)
    denominator = n * sum_ee - sum_e * sum_e
    if denominator <= 0:
        return FrameCalibration(ratios, n, scale, nan, nan, n, True)
    s = (n * sum_eg - sum_e * sum_g) / denominator
    b = (sum_g - s * sum_e) / n
    failed = not (math.isfinite(s) and s > 0 and math.isfinite(b))
    return FrameCalibration(ratios, n, scale, float(s), float(b), n, failed)


def scene_scale_leave_target_out(
    calibrations: dict[str, FrameCalibration], target_frame_id: str
) -> tuple[float, dict[str, int]]:
    """Level 1: the leave-target-out scene oracle scale.

    One multiplicative scalar per pair: the median of ground truth over
    estimated depth pooled over the calibration-valid pixels of every scene
    frame except the pair's target. The audit maps each contributing frame to
    its pixel count; the target is absent by construction and the caller
    asserts that per evaluated record.
    """
    parts = [
        c.ratios
        for fid, c in calibrations.items()
        if fid != target_frame_id and c.n_calibration
    ]
    audit = {
        fid: c.n_calibration
        for fid, c in calibrations.items()
        if fid != target_frame_id and c.n_calibration
    }
    if not parts:
        raise Phase4GateError(
            f"Level 1 calibration population is empty with {target_frame_id} excluded"
        )
    scale = float(np.median(np.concatenate(parts)))
    if not (math.isfinite(scale) and scale > 0):
        raise Phase4GateError(
            f"Level 1 scale {scale} is not finite and positive for target {target_frame_id}"
        )
    return scale, audit


def aligned_depth(
    level: str,
    vggt_depth: np.ndarray,
    scene_scale: float,
    frame_calib: FrameCalibration,
) -> np.ndarray | None:
    """Apply one level's transform to one estimated map, or None when failed."""
    if level == "none":
        return vggt_depth
    if level == "scene":
        return vggt_depth * np.float32(scene_scale)
    if level == "image":
        if not (math.isfinite(frame_calib.image_scale) and frame_calib.image_scale > 0):
            raise Phase4GateError(
                f"Level 2 image scale {frame_calib.image_scale} is not finite and positive"
            )
        return vggt_depth * np.float32(frame_calib.image_scale)
    if level == "affine":
        if frame_calib.affine_failed:
            return None
        return vggt_depth * np.float32(frame_calib.affine_s) + np.float32(frame_calib.affine_b)
    raise ValueError(f"unknown alignment level {level!r}")


# ---------------------------------------------------------------------------
# Forced-collision transport: an isolated diagnostic copy, PROTOCOL 4.5
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SplatPlanDetail:
    """A transport plan plus the internals the pure-rotation gate needs.

    winner_keys encodes each winning splat as source_pixel * n_target_pixels
    plus target_pixel, sorted, so a forced run can test membership with one
    searchsorted pass instead of a Python set.
    """

    weights: Tensor             # [n_target_patches, n_source_patches] float32
    coverage: Tensor            # [Hp_out, Wp_out] float32
    keep: Tensor                # [H*W] bool over source pixels
    winner_keys: np.ndarray     # sorted int64


def splat_plan_detail(
    depth_ctx_px: Tensor,
    K_ctx: Tensor,
    K_tgt: Tensor,
    T_tgt_from_ctx: Tensor,
    out_hw_px: tuple[int, int],
    patch_size: int = PATCH_SIZE,
    source_keep: Tensor | None = None,
    forced_winner_keys: np.ndarray | None = None,
) -> SplatPlanDetail:
    """The transport plan with optional source restriction and forced winners.

    A documented copy of lot.transport.transport_plan, existing so the frozen
    default path is never modified. With source_keep and forced_winner_keys
    both None it reproduces the default plan exactly;
    assert_forcing_disabled_matches_default checks that at run time and the
    suite checks it permanently. Forcing replaces the z-buffer winner rule
    with an externally supplied (source pixel, target pixel) key set, which
    is how the gate imposes Oracle's collision ordering on estimated-depth
    transport under pure rotation.
    """
    height, width = depth_ctx_px.shape
    patches_h, patches_w = height // patch_size, width // patch_size
    out_height, out_width = out_hw_px
    out_patches_h, out_patches_w = out_height // patch_size, out_width // patch_size
    dtype = depth_ctx_px.dtype if depth_ctx_px.dtype.is_floating_point else torch.float32

    uv = pixel_grid(height, width, dtype=dtype)
    z = depth_ctx_px.to(dtype)
    points_ctx = unproject(uv, z, K_ctx.to(dtype))
    points_tgt = transform_points(T_tgt_from_ctx.to(dtype), points_ctx)
    uv_tgt, z_tgt = project(points_tgt, K_tgt.to(dtype))

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
    if source_keep is not None:
        keep &= source_keep.reshape(height, width)

    keep_flat = keep.reshape(-1)
    source_index = torch.arange(height * width)[keep_flat]
    lin = (iv * out_width + iu).reshape(-1)[keep_flat]
    z_keep = z_tgt.reshape(-1)[keep_flat]
    n_cells = out_height * out_width

    if forced_winner_keys is None:
        zbuffer = torch.full((n_cells,), torch.inf, dtype=dtype)
        zbuffer.scatter_reduce_(0, lin, z_keep, reduce="amin", include_self=True)
        winners = z_keep <= zbuffer[lin] * (1 + TIE_RELATIVE_EPS)
    else:
        keys = source_index.numpy().astype(np.int64) * np.int64(n_cells) + lin.numpy()
        position = np.searchsorted(forced_winner_keys, keys)
        position = np.clip(position, 0, len(forced_winner_keys) - 1)
        winners = torch.from_numpy(
            (len(forced_winner_keys) > 0) & (forced_winner_keys[position] == keys)
        )
    lin_w = lin[winners]
    src_w = source_index[winners]

    source_patch = (
        (torch.arange(height) // patch_size)[:, None] * patches_w
        + (torch.arange(width) // patch_size)[None, :]
    )
    source_of_winner = source_patch.reshape(-1)[keep_flat][winners]

    count = torch.zeros((n_cells,), dtype=torch.float32)
    count.index_add_(0, lin_w, torch.ones_like(lin_w, dtype=torch.float32))
    hit = count > 0
    hits_per_patch = hit.to(torch.float32).reshape(
        out_patches_h, patch_size, out_patches_w, patch_size
    ).sum(dim=(1, 3))

    target_patch = (lin_w // out_width // patch_size) * out_patches_w + (
        lin_w % out_width
    ) // patch_size
    num_source = patches_h * patches_w
    num_target = out_patches_h * out_patches_w
    weights = torch.zeros((num_target * num_source,), dtype=torch.float32)
    weights.index_add_(0, target_patch * num_source + source_of_winner, 1.0 / count[lin_w])
    weights = weights.reshape(num_target, num_source)
    weights = weights / hits_per_patch.reshape(num_target, 1).clamp(min=1)

    winner_keys = np.sort(
        src_w.numpy().astype(np.int64) * np.int64(n_cells) + lin_w.numpy()
    )
    return SplatPlanDetail(
        weights=weights,
        coverage=hits_per_patch / float(patch_size * patch_size),
        keep=keep_flat,
        winner_keys=winner_keys,
    )


def assert_forcing_disabled_matches_default(
    depth: Tensor, K_ctx: Tensor, K_tgt: Tensor, T: Tensor, out_hw: tuple[int, int]
) -> None:
    """The diagnostic copy with forcing disabled must be the frozen default."""
    default = transport_plan(depth, K_ctx, K_tgt, T, out_hw)
    detail = splat_plan_detail(depth, K_ctx, K_tgt, T, out_hw)
    if not torch.equal(default.weights, detail.weights) or not torch.equal(
        default.coverage, detail.coverage
    ):
        raise Phase4GateError(
            "the forced-collision machinery with forcing disabled does not "
            "reproduce the frozen transport plan; the diagnostic copy has "
            "drifted from lot.transport.transport_plan"
        )


# ---------------------------------------------------------------------------
# Error-localization masks, PROTOCOL 4.8, frozen before outcomes are viewed
# ---------------------------------------------------------------------------

def depth_boundary_mask(gt_depth: Tensor, analysis: AnalysisConfig) -> np.ndarray:
    """Pixels near a ground-truth depth discontinuity. [H, W] bool.

    Central-difference gradient magnitude of ground-truth depth, compared
    against the frozen threshold as a fraction of local depth, then dilated by
    the frozen radius. Amendment A5 records the operationalization. Pixels
    bordering invalid depth count as boundary; a hole edge is a discontinuity.
    """
    depth = gt_depth.to(torch.float64).numpy()
    valid = np.isfinite(depth) & (depth > 0)
    safe = np.where(valid, depth, 0.0)
    gy, gx = np.gradient(safe)
    magnitude = np.sqrt(gx * gx + gy * gy)
    local = np.where(valid, np.abs(depth), np.inf)
    edge = magnitude > analysis.depth_boundary_gradient_threshold * local
    invalid_adjacent = np.zeros_like(valid)
    invalid = ~valid
    invalid_adjacent[1:] |= invalid[:-1]
    invalid_adjacent[:-1] |= invalid[1:]
    invalid_adjacent[:, 1:] |= invalid[:, :-1]
    invalid_adjacent[:, :-1] |= invalid[:, 1:]
    mask = (edge | invalid_adjacent) & valid
    radius = int(analysis.depth_boundary_dilation_px)
    if radius > 0:
        pooled = torch.nn.functional.max_pool2d(
            torch.from_numpy(mask[None, None].astype(np.float32)),
            2 * radius + 1,
            stride=1,
            padding=radius,
        )
        mask = pooled[0, 0].numpy() > 0
    return mask


def boundary_cells_from_mask(boundary_px: np.ndarray) -> np.ndarray:
    """Patch-level boundary flag: any dilated boundary pixel in the patch.

    The dilation radius already controls the band width, so the patch rule is
    presence, not majority. Amendment A5 records this. [Hp * Wp] bool.
    """
    per_patch = fraction_per_patch(torch.from_numpy(boundary_px), PATCH_SIZE)
    return (per_patch.reshape(-1).numpy() > 0)


def low_texture_cells(rgb: np.ndarray, analysis: AnalysisConfig) -> np.ndarray:
    """Patch-level low-texture flag from the RGB gradient. [Hp * Wp] bool.

    Grayscale is the channel mean scaled to [0, 1]; the statistic is the mean
    central-difference gradient magnitude over each patch; low texture means
    the statistic falls below the frozen threshold. Amendment A5.
    """
    gray = rgb[..., :3].astype(np.float64).mean(axis=-1) / 255.0
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    height, width = magnitude.shape
    per_patch = magnitude.reshape(
        height // PATCH_SIZE, PATCH_SIZE, width // PATCH_SIZE, PATCH_SIZE
    ).mean(axis=(1, 3))
    return (per_patch < analysis.texture_gradient_threshold).reshape(-1)


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------

def _per_sample_cosine(a: Tensor, b: Tensor, center: Tensor | None) -> Tensor:
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    if center is not None:
        a = a - center
        b = b - center
    a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return (a * b).sum(dim=-1)


def _metrics_with_intersection(
    prediction: Tensor,
    target: Tensor,
    center: Tensor,
    centered_defined: bool,
    in_shared: Tensor,
) -> dict[str, float]:
    out = agreement_metrics(prediction, target, center, centered_defined)
    shared = agreement_metrics(
        prediction[in_shared], target[in_shared], center, centered_defined
    )
    out["cosine_intersect_mean"] = shared["cosine_mean"]
    out["cosine_centered_intersect_mean"] = shared["cosine_centered_mean"]
    return out


@dataclasses.dataclass
class PairDepthInputs:
    """Everything estimated-depth about one pair, before any level applies."""

    est_context: np.ndarray            # converted, unaligned, render resolution
    est_target: np.ndarray
    scene_scale: float                 # Level 1 scalar for this pair
    scene_audit: dict[str, int]        # contributing frame to calibration pixels
    context_calib: FrameCalibration    # Level 2 and affine, context image only


@dataclasses.dataclass
class _LevelData:
    """One alignment level's per-pair intermediates, before row emission."""

    variant_name: str
    landed: np.ndarray                 # [N] bool over per-point samples (5d)
    n_transport_valid_pp: int          # 5c count over per-point samples
    n_transport_valid_ctx: int         # 5c pixel count on the context map
    uv_warp_est: Tensor
    est_reads: Tensor
    pp_scored: np.ndarray              # universe mask, per-point 5d
    sp_scored: np.ndarray              # universe mask, splat 5d inside Phase 3
    n_est_outside: int                 # splat cells outside the Phase 3 set
    transported_est: Tensor            # [C, cells] pooled estimated features
    context_aligned: np.ndarray
    scale: float


def evaluate_pair_phase4(
    geometry: Any,
    pair: Any,
    depth_context_gt: Tensor,
    depth_inputs: PairDepthInputs,
    features_context: Tensor,
    features_target: Tensor,
    mean_vector: Tensor,
    analysis: AnalysisConfig,
    boundary_cells: np.ndarray,
    lowtex_cells: np.ndarray,
    torch_dtype: torch.dtype,
    K_context: Tensor,
    K_target: Tensor,
    T_target_from_context: Tensor,
    gate_evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every Phase 4 row for one pair. Returns (rows, per-pair audit).

    The Phase 3 population is the referee: the per-point sample set and the
    splat cell set come from pair_geometry unchanged, and every matched
    quantity is computed on the estimated-valid subset of them, per level and
    per path, with the mask persisted on the row as proof.
    """
    rows: list[dict[str, Any]] = []
    center = mean_vector.to(torch.float32)
    channels = features_context.shape[0]
    size = geometry.size
    universe_boundary = pack_mask(boundary_cells)
    universe_lowtex = pack_mask(lowtex_cells)

    samples = geometry.samples
    n_pp = int(samples.uv_target.shape[0])
    per_point_cells = geometry.per_point_cells
    target_hw = depth_inputs.est_target.shape
    box_context = _sampling_box(depth_inputs.est_context.shape, PATCH_SIZE)
    T_context_from_target = invert_se3(T_target_from_context)

    reads_target = sample_features_bilinear(features_target, samples.uv_target)
    reads_warp_gt = sample_features_bilinear(features_context, samples.uv_context_warp)
    reads_no_warp = sample_features_bilinear(features_context, samples.uv_target)

    flat_context = features_context.to(torch.float32).reshape(channels, -1)
    flat_target = features_target.to(torch.float32).reshape(channels, -1)
    splat_cells = np.flatnonzero(geometry.splat_mask)
    transported_gt = apply_transport_plan(geometry.plan, features_context).reshape(channels, -1)

    base = pair.as_row()
    base["covisible_fraction"] = geometry.covisible_fraction
    base["parallax"] = geometry.parallax
    base["encoder"] = "dinov2_vitb14"

    nan = float("nan")
    empty_gate = {
        "gate_coord_max_px": nan,
        "gate_score_max_abs": nan,
        "gate_forced_max_abs": nan,
        "collision_tax_raw": nan,
        "collision_tax_centered": nan,
    }

    def emit(path: str, level: str, population: str, variant: str, mask: np.ndarray,
             prediction: Tensor, target: Tensor, in_shared: Tensor,
             n_shared: int, extra: dict[str, Any]) -> None:
        rows.append(
            {
                **base,
                "path": path,
                "level": level,
                "population": population,
                "variant": variant,
                "n": int(mask.sum()),
                "n_intersect": n_shared,
                "sample_mask": pack_mask(mask),
                "boundary_mask": universe_boundary,
                "lowtex_mask": universe_lowtex,
                **_metrics_with_intersection(
                    prediction, target, center, variant != MEAN_FEATURE, in_shared
                ),
                **extra,
            }
        )

    # PROTOCOL 4.3's inviolable rule, asserted for every evaluated record.
    if pair.target_frame_id in depth_inputs.scene_audit:
        raise Phase4GateError(
            f"{pair.context_frame_id} -> {pair.target_frame_id}: the target "
            "frame contributes calibration pixels to its own Level 1 estimator"
        )
    audit: dict[str, Any] = {
        "scene_scale": depth_inputs.scene_scale,
        "scene_audit_frames": len(depth_inputs.scene_audit),
        "scene_audit_pixels": int(sum(depth_inputs.scene_audit.values())),
        "affine_failed": bool(depth_inputs.context_calib.affine_failed),
        "levels": {},
    }

    def calib_columns(level: str) -> dict[str, Any]:
        if level == "scene":
            return {"calib_pixels": audit["scene_audit_pixels"],
                    "calib_frames": audit["scene_audit_frames"]}
        if level in ("image", "affine"):
            return {"calib_pixels": depth_inputs.context_calib.n_calibration,
                    "calib_frames": 1}
        return {"calib_pixels": 0, "calib_frames": 0}

    # ---- ground-truth rows on the full Phase 3 population ------------------
    shared_full = geometry.cross_path_mask
    n_shared_full = int(shared_full.sum())
    gt_extra = {
        "n_transport_valid": n_pp,
        "n_gt": n_pp,
        "n_est_outside": 0,
        "scale": nan,
        "affine_s": nan,
        "affine_b": nan,
        "calib_pixels": 0,
        "calib_frames": 0,
        **empty_gate,
    }
    if n_pp:
        in_shared_pp = torch.from_numpy(shared_full[per_point_cells])
        mean_feature_pp = center[None, :].expand(n_pp, -1)
        for variant, prediction in (
            (ORACLE_TRANSPORT, reads_warp_gt),
            (NO_WARP_COPY, reads_no_warp),
            (MEAN_FEATURE, mean_feature_pp),
        ):
            emit(PER_POINT, GT_LEVEL, POPULATION_FULL, variant,
                 geometry.per_point_mask, prediction, reads_target, in_shared_pp,
                 n_shared_full, gt_extra)
    if splat_cells.size:
        in_shared_sp = torch.from_numpy(shared_full[splat_cells])
        targets_sp = flat_target[:, splat_cells].T
        sp_extra = {**gt_extra, "n_transport_valid": int(splat_cells.size),
                    "n_gt": int(splat_cells.size)}
        mean_feature_sp = center[None, :].expand(int(splat_cells.size), -1)
        for variant, prediction in (
            (ORACLE_TRANSPORT, transported_gt[:, splat_cells].T),
            (NO_WARP_COPY, flat_context[:, splat_cells].T),
            (MEAN_FEATURE, mean_feature_sp),
        ):
            emit(SPLAT_POOL, GT_LEVEL, POPULATION_FULL, variant,
                 geometry.splat_mask, prediction, targets_sp, in_shared_sp,
                 n_shared_full, sp_extra)

    # ---- alignment levels: intermediates first -----------------------------
    is_rotation = pair.regime == "rotation"
    levels: dict[str, _LevelData] = {}
    for level, variant_name in LEVELS:
        target_aligned = aligned_depth(
            level, depth_inputs.est_target, depth_inputs.scene_scale,
            depth_inputs.context_calib,
        )
        context_aligned = aligned_depth(
            level, depth_inputs.est_context, depth_inputs.scene_scale,
            depth_inputs.context_calib,
        )
        if target_aligned is None or context_aligned is None:
            audit["levels"][level] = {"affine_failed": True}
            continue

        # Per-point path. 5c is a depth-valid read at the frozen samples; 5d
        # adds landing: positive depth in the context camera and a warp inside
        # the context sampling box.
        est_read = sample_map_bilinear(
            torch.from_numpy(target_aligned).to(torch_dtype), samples.uv_target
        )
        read_valid = torch.isfinite(est_read) & (est_read > 0)
        safe_read = torch.where(read_valid, est_read, torch.ones_like(est_read))
        points_target = unproject(samples.uv_target, safe_read, K_target)
        points_context = transform_points(T_context_from_target, points_target)
        uv_warp_est, z_est = project(points_context, K_context)
        landed = (
            read_valid & (z_est > 0) & _in_box(uv_warp_est, box_context)
        ).numpy()
        pp_scored = np.zeros(size, dtype=bool)
        pp_scored[per_point_cells[landed]] = True

        # Splat path through the frozen operator with the aligned context map.
        est_plan = transport_plan(
            torch.from_numpy(context_aligned).to(torch_dtype),
            K_context, K_target, T_target_from_context, target_hw,
        )
        est_cov = (est_plan.coverage.reshape(-1) > 0).numpy()
        sp_scored_own = geometry.splat_covisible_ok & est_cov
        sp_scored = sp_scored_own & geometry.splat_mask
        levels[level] = _LevelData(
            variant_name=variant_name,
            landed=landed,
            n_transport_valid_pp=int(read_valid.sum()),
            n_transport_valid_ctx=int(
                (np.isfinite(context_aligned) & (context_aligned > 0)).sum()
            ),
            uv_warp_est=uv_warp_est,
            est_reads=sample_features_bilinear(features_context, uv_warp_est),
            pp_scored=pp_scored,
            sp_scored=sp_scored,
            n_est_outside=int((sp_scored_own & ~geometry.splat_mask).sum()),
            transported_est=apply_transport_plan(est_plan, features_context).reshape(
                channels, -1
            ),
            context_aligned=context_aligned,
            scale=(
                depth_inputs.scene_scale if level == "scene"
                else depth_inputs.context_calib.image_scale if level == "image"
                else nan
            ),
        )
        audit["levels"][level] = {
            "pp_transport_valid": levels[level].n_transport_valid_pp,
            "pp_scored": int(pp_scored.sum()),
            "sp_scored": int(sp_scored.sum()),
            "n_est_outside": levels[level].n_est_outside,
        }

    # Step 10: the multiplicative levels share one transport-valid set, on
    # both the per-point samples and the context map. The scored (5d) sets may
    # legitimately differ and are reported, never asserted.
    present = [l for l in MULTIPLICATIVE_LEVELS if l in levels]
    for left, right in zip(present, present[1:]):
        left_valid = levels[left].n_transport_valid_pp
        if left_valid != levels[right].n_transport_valid_pp or (
            levels[left].n_transport_valid_ctx != levels[right].n_transport_valid_ctx
        ):
            raise Phase4GateError(
                f"{pair.context_frame_id} -> {pair.target_frame_id}: "
                f"transport-valid sets differ between levels {left} and "
                f"{right}; positive scaling cannot change finite-and-positive, "
                "so this is an implementation error"
            )

    # Ground-truth splat internals for the forced gate, once per pair.
    gt_detail = None
    if is_rotation and levels:
        gt_detail = splat_plan_detail(
            depth_context_gt, K_context, K_target, T_target_from_context, target_hw
        )

    # ---- row emission per level -------------------------------------------
    for level, data in levels.items():
        shared_level = data.pp_scored & data.sp_scored
        n_shared = int(shared_level.sum())
        common_extra = {
            "n_gt": n_pp,
            "n_est_outside": 0,
            "scale": data.scale,
            "affine_s": depth_inputs.context_calib.affine_s if level == "affine" else nan,
            "affine_b": depth_inputs.context_calib.affine_b if level == "affine" else nan,
            **calib_columns(level),
        }

        # Per-point rows on V = this level's scored subset of Phase 3.
        selected = torch.from_numpy(data.landed)
        chosen = np.flatnonzero(data.landed)
        gate_cols = dict(empty_gate)
        if chosen.size:
            targets_v = reads_target[selected]
            if is_rotation:
                residuals = (
                    data.uv_warp_est[selected] - samples.uv_context_warp[selected]
                ).abs()
                coord_max = float(residuals.max())
                gt_raw = _per_sample_cosine(reads_warp_gt[selected], targets_v, None)
                est_raw = _per_sample_cosine(data.est_reads[selected], targets_v, None)
                gt_cen = _per_sample_cosine(reads_warp_gt[selected], targets_v, center)
                est_cen = _per_sample_cosine(data.est_reads[selected], targets_v, center)
                score_max = max(
                    float((gt_raw - est_raw).abs().max()),
                    float((gt_cen - est_cen).abs().max()),
                )
                gate_cols["gate_coord_max_px"] = coord_max
                gate_cols["gate_score_max_abs"] = score_max
                gate_evidence.append(
                    {
                        "pair": f"{pair.context_frame_id} -> {pair.target_frame_id}",
                        "level": level,
                        "path": PER_POINT,
                        "coord_max_px": coord_max,
                        "score_max_abs": score_max,
                        "n": int(chosen.size),
                    }
                )
                if coord_max > analysis.rotation_gate_coord_tol_px:
                    worst = int(residuals.max(dim=-1).values.argmax())
                    raise Phase4GateError(
                        "PROTOCOL 4.5 per-point coordinate gate failed: scene "
                        f"{pair.scene}, context {pair.context_frame_id}, target "
                        f"{pair.target_frame_id}, level {level}, sample_id "
                        f"{int(samples.sample_id[chosen[worst]])}, oracle uv "
                        f"{samples.uv_context_warp[selected][worst].tolist()}, "
                        f"estimated uv {data.uv_warp_est[selected][worst].tolist()}, "
                        f"residual {coord_max:.3e} px over tolerance "
                        f"{analysis.rotation_gate_coord_tol_px:g}"
                    )
                if score_max > analysis.rotation_gate_score_tol:
                    raise Phase4GateError(
                        "PROTOCOL 4.5 per-point score gate failed: scene "
                        f"{pair.scene}, {pair.context_frame_id} -> "
                        f"{pair.target_frame_id}, level {level}, max score "
                        f"residual {score_max:.3e} over tolerance "
                        f"{analysis.rotation_gate_score_tol:g}"
                    )
            in_shared = torch.from_numpy(shared_level[per_point_cells[data.landed]])
            extra = {
                **common_extra,
                "n_transport_valid": data.n_transport_valid_pp,
                **gate_cols,
            }
            mean_feature_v = center[None, :].expand(int(chosen.size), -1)
            for variant, prediction in (
                (data.variant_name, data.est_reads[selected]),
                (ORACLE_TRANSPORT, reads_warp_gt[selected]),
                (NO_WARP_COPY, reads_no_warp[selected]),
                (MEAN_FEATURE, mean_feature_v),
            ):
                emit(PER_POINT, level, POPULATION_MATCHED, variant, data.pp_scored,
                     prediction, targets_v, in_shared, n_shared, extra)

            # PROTOCOL 4.8 splits of the same matched set, so the boundary and
            # texture contrasts are computable from aggregated rows. Masks and
            # thresholds are frozen ground-truth quantities; empty splits are
            # simply absent and the report layer treats absence as absence.
            for population, cell_flag in (
                ("boundary", boundary_cells),
                ("interior", ~boundary_cells),
                ("lowtex", lowtex_cells),
                ("hightex", ~lowtex_cells),
            ):
                keep = data.landed & cell_flag[per_point_cells]
                if not keep.any():
                    continue
                split_mask = np.zeros(size, dtype=bool)
                split_mask[per_point_cells[keep]] = True
                split_selected = torch.from_numpy(keep)
                split_targets = reads_target[split_selected]
                split_shared = torch.from_numpy(
                    (shared_level & split_mask)[per_point_cells[keep]]
                )
                split_n_shared = int((shared_level & split_mask).sum())
                for variant, prediction in (
                    (data.variant_name, data.est_reads[split_selected]),
                    (ORACLE_TRANSPORT, reads_warp_gt[split_selected]),
                    (NO_WARP_COPY, reads_no_warp[split_selected]),
                ):
                    emit(PER_POINT, level, population, variant, split_mask,
                         prediction, split_targets, split_shared, split_n_shared,
                         extra)

        # Splat rows on V = this level's scored subset of Phase 3.
        cells_v = np.flatnonzero(data.sp_scored)
        sp_gate_cols = dict(empty_gate)
        if cells_v.size:
            targets_v = flat_target[:, cells_v].T
            if is_rotation and gt_detail is not None:
                # PROTOCOL 4.5 forced-collision-order gate. The common source
                # set is the intersection of both methods' kept splats;
                # Oracle's winner ordering is built on it and imposed on both.
                est_detail = splat_plan_detail(
                    torch.from_numpy(data.context_aligned).to(torch_dtype),
                    K_context, K_target, T_target_from_context, target_hw,
                )
                common = gt_detail.keep & est_detail.keep
                gt_common = splat_plan_detail(
                    depth_context_gt, K_context, K_target, T_target_from_context,
                    target_hw, source_keep=common,
                )
                forced_est = splat_plan_detail(
                    torch.from_numpy(data.context_aligned).to(torch_dtype),
                    K_context, K_target, T_target_from_context, target_hw,
                    source_keep=common, forced_winner_keys=gt_common.winner_keys,
                )
                pooled_oracle = flat_context @ gt_common.weights.mT
                pooled_forced = flat_context @ forced_est.weights.mT
                both_covered = (
                    (gt_common.coverage.reshape(-1) > 0)
                    & (forced_est.coverage.reshape(-1) > 0)
                ).numpy()
                gate_cells = np.flatnonzero(data.sp_scored & both_covered)
                if gate_cells.size:
                    gate_targets = flat_target[:, gate_cells].T
                    oracle_scores = _per_sample_cosine(
                        pooled_oracle[:, gate_cells].T, gate_targets, None
                    )
                    forced_scores = _per_sample_cosine(
                        pooled_forced[:, gate_cells].T, gate_targets, None
                    )
                    forced_max = float((oracle_scores - forced_scores).abs().max())
                    unforced_raw = _per_sample_cosine(
                        data.transported_est[:, gate_cells].T, gate_targets, None
                    )
                    forced_cen = _per_sample_cosine(
                        pooled_forced[:, gate_cells].T, gate_targets, center
                    )
                    unforced_cen = _per_sample_cosine(
                        data.transported_est[:, gate_cells].T, gate_targets, center
                    )
                    sp_gate_cols["gate_forced_max_abs"] = forced_max
                    sp_gate_cols["collision_tax_raw"] = float(
                        (forced_scores - unforced_raw).mean()
                    )
                    sp_gate_cols["collision_tax_centered"] = float(
                        (forced_cen - unforced_cen).mean()
                    )
                    gate_evidence.append(
                        {
                            "pair": f"{pair.context_frame_id} -> {pair.target_frame_id}",
                            "level": level,
                            "path": SPLAT_POOL,
                            "forced_max_abs": forced_max,
                            "collision_tax_raw": sp_gate_cols["collision_tax_raw"],
                            "n_cells": int(gate_cells.size),
                        }
                    )
                    if forced_max > analysis.rotation_gate_forced_tol:
                        raise Phase4GateError(
                            "PROTOCOL 4.5 forced-collision-order gate failed: "
                            f"scene {pair.scene}, {pair.context_frame_id} -> "
                            f"{pair.target_frame_id}, level {level}, max forced "
                            f"score residual {forced_max:.3e} over tolerance "
                            f"{analysis.rotation_gate_forced_tol:g}"
                        )
            in_shared = torch.from_numpy(shared_level[cells_v])
            extra = {
                **common_extra,
                "n_transport_valid": data.n_transport_valid_ctx,
                "n_gt": int(splat_cells.size),
                "n_est_outside": data.n_est_outside,
                **sp_gate_cols,
            }
            mean_feature_v = center[None, :].expand(int(cells_v.size), -1)
            for variant, prediction in (
                (data.variant_name, data.transported_est[:, cells_v].T),
                (ORACLE_TRANSPORT, transported_gt[:, cells_v].T),
                (NO_WARP_COPY, flat_context[:, cells_v].T),
                (MEAN_FEATURE, mean_feature_v),
            ):
                emit(SPLAT_POOL, level, POPULATION_MATCHED, variant, data.sp_scored,
                     prediction, targets_v, in_shared, n_shared, extra)

            for population, cell_flag in (
                ("boundary", boundary_cells),
                ("interior", ~boundary_cells),
                ("lowtex", lowtex_cells),
                ("hightex", ~lowtex_cells),
            ):
                split_mask = data.sp_scored & cell_flag
                split_cells_v = np.flatnonzero(split_mask)
                if not split_cells_v.size:
                    continue
                split_targets = flat_target[:, split_cells_v].T
                split_shared = torch.from_numpy((shared_level & split_mask)[split_cells_v])
                split_n_shared = int((shared_level & split_mask).sum())
                for variant, prediction in (
                    (data.variant_name, data.transported_est[:, split_cells_v].T),
                    (ORACLE_TRANSPORT, transported_gt[:, split_cells_v].T),
                    (NO_WARP_COPY, flat_context[:, split_cells_v].T),
                ):
                    emit(SPLAT_POOL, level, population, variant, split_mask,
                         prediction, split_targets, split_shared, split_n_shared,
                         extra)

    return rows, audit


# ---------------------------------------------------------------------------
# Scene driver
# ---------------------------------------------------------------------------

def evaluate_scene_phase4(
    cfg: Phase4Config,
    scene: str,
    mean_vector: Tensor,
    analysis: AnalysisConfig,
    convention: dict[str, Any],
    regimes: tuple[str, ...] | None = None,
    collect_rows: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Phase 4 over one scene's frozen Phase 3 pair sample."""
    from PIL import Image

    scene_root = cfg.renders_root / scene
    manifest = load_manifest(scene_root / MANIFEST_NAME)
    validate_manifest(
        manifest,
        scene_root,
        check_files=False,
        rotation_position_bound_m=analysis.rotation_position_bound_m,
        translation_rotation_bound_deg=analysis.translation_rotation_bound_deg,
    )
    frames = {f.frame_id: f for f in manifest.frames}
    pairs = subsample_by_stratum(
        load_scene_pairs(cfg.renders_root, scene, config=analysis),
        analysis.max_pairs_per_stratum,
        seed=cfg.seed,
        config=analysis,
    )
    if regimes is not None:
        pairs = [p for p in pairs if p.regime in regimes]

    verdict = convention["scenes"][scene]["verdict"]
    if verdict not in ("planar_z", "ray_distance"):
        raise Phase4GateError(f"{scene}: depth convention unresolved: {verdict!r}")
    depth_cache = load_depth_archive(cfg.cache_root, cfg.depth_encoder, scene)

    cache = _SceneCache(scene_root, cfg.cache_root, [cfg.feature_encoder], scene, manifest)
    est_maps: dict[str, np.ndarray] = {}
    calibrations: dict[str, FrameCalibration] = {}
    resample_records: dict[str, dict[str, Any]] = {}
    boundary_by_frame: dict[str, np.ndarray] = {}
    lowtex_by_frame: dict[str, np.ndarray] = {}

    for frame in manifest.frames:
        raw = depth_cache["depth"][frame.frame_id]
        resampled, record = resample_depth_nearest(raw, (frame.height, frame.width))
        resample_records[frame.frame_id] = record
        if verdict == "ray_distance":
            resampled = (
                resampled / secant_map(frame.K, frame.height, frame.width)
            ).astype(np.float32)
        est_maps[frame.frame_id] = resampled
        conf_raw = depth_cache["conf"][frame.frame_id]
        conf = (
            resample_depth_nearest(conf_raw, (frame.height, frame.width))[0]
            if conf_raw is not None
            else None
        )
        prevalid = transport_prevalid(resampled, conf, analysis)
        calibrations[frame.frame_id] = frame_calibration(
            resampled, cache.depth(frame.depth_path).numpy(), prevalid
        )

    rows: list[dict[str, Any]] = []
    gate_evidence: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    forcing_checked = False
    affine_failed_pairs = 0
    universe_size = 0

    for pair in pairs:
        context = frames[pair.context_frame_id]
        target = frames[pair.target_frame_id]
        T_target_from_context = relative_pose(
            target.T_world_from_camera, context.T_world_from_camera
        ).to(cfg.torch_dtype)
        depth_context = cache.depth(context.depth_path).to(cfg.torch_dtype)
        depth_target = cache.depth(target.depth_path).to(cfg.torch_dtype)
        geometry = pair_geometry(
            depth_context,
            depth_target,
            context.K.to(cfg.torch_dtype),
            target.K.to(cfg.torch_dtype),
            T_target_from_context,
            scene,
            pair.context_frame_id,
            pair.target_frame_id,
            analysis,
        )
        universe_size = geometry.size
        if not geometry.scorable:
            continue
        assert_translation_parallax_floor(
            pair.regime, geometry.parallax, analysis,
            f"{scene} {pair.context_frame_id} -> {pair.target_frame_id}",
        )
        if not forcing_checked:
            assert_forcing_disabled_matches_default(
                depth_context, context.K.to(cfg.torch_dtype),
                target.K.to(cfg.torch_dtype), T_target_from_context,
                tuple(depth_target.shape),
            )
            forcing_checked = True

        if pair.target_frame_id not in boundary_by_frame:
            boundary_by_frame[pair.target_frame_id] = boundary_cells_from_mask(
                depth_boundary_mask(depth_target, analysis)
            )
            rgb = np.asarray(Image.open(scene_root / target.rgb_path))
            lowtex_by_frame[pair.target_frame_id] = low_texture_cells(rgb, analysis)

        scene_scale, scene_audit = scene_scale_leave_target_out(
            calibrations, pair.target_frame_id
        )
        inputs = PairDepthInputs(
            est_context=est_maps[pair.context_frame_id],
            est_target=est_maps[pair.target_frame_id],
            scene_scale=scene_scale,
            scene_audit=scene_audit,
            context_calib=calibrations[pair.context_frame_id],
        )
        if inputs.context_calib.affine_failed:
            affine_failed_pairs += 1
        pair_rows, audit = evaluate_pair_phase4(
            geometry,
            pair,
            depth_context,
            inputs,
            cache.features(cfg.feature_encoder, pair.context_frame_id),
            cache.features(cfg.feature_encoder, pair.target_frame_id),
            mean_vector,
            analysis,
            boundary_by_frame[pair.target_frame_id],
            lowtex_by_frame[pair.target_frame_id],
            cfg.torch_dtype,
            context.K.to(cfg.torch_dtype),
            target.K.to(cfg.torch_dtype),
            T_target_from_context,
            gate_evidence,
        )
        if collect_rows:
            rows.extend(pair_rows)
        audits[f"{pair.context_frame_id} -> {pair.target_frame_id}"] = audit
    cache.close()

    metadata = {
        "phase4_version": PHASE4_VERSION,
        "scene": scene,
        "pairs_evaluated": len(audits),
        "affine_failed_pairs": affine_failed_pairs,
        "gate_checks": len(gate_evidence),
        "gate_coord_max_px": max(
            (g.get("coord_max_px", 0.0) for g in gate_evidence), default=0.0
        ),
        "gate_score_max_abs": max(
            (g.get("score_max_abs", 0.0) for g in gate_evidence), default=0.0
        ),
        "gate_forced_max_abs": max(
            (g.get("forced_max_abs", 0.0) for g in gate_evidence), default=0.0
        ),
        "depth_convention": convention["scenes"][scene],
        "resample": next(iter(resample_records.values())) if resample_records else None,
        "git_commit": git_commit(),
        "seed": cfg.seed,
        "feature_encoder": cfg.feature_encoder,
        "depth_encoder": cfg.depth_encoder,
        "analysis_config_digest": analysis.digest(),
        "analysis_measurement_digest": analysis.measurement_digest(),
        "phase4_measurement_digest": phase4_measurement_digest(analysis),
        "analysis_reporting_digest": analysis.reporting_digest(),
        "features_digest": load_cache_meta(
            cfg.cache_root, cfg.feature_encoder, scene
        )["features_digest"],
        "depth_digest": depth_cache["meta"]["depth_digest"],
        "depth_weights_fingerprint": depth_cache["meta"]["weights_fingerprint"],
        "depth_weights_revision": depth_cache["meta"].get("weights_revision", "unpinned"),
        "depth_code_revision": depth_cache["meta"].get("code_revision", "unknown"),
        "universe_size": universe_size,
        "run_scenes": sorted(cfg.scenes),
        "target_exclusion_asserted_per_record": True,
    }
    return rows, {"metadata": metadata, "gate_evidence": gate_evidence, "audits": audits}


# ---------------------------------------------------------------------------
# Convention report over all scenes
# ---------------------------------------------------------------------------

def convention_report(cfg: Phase4Config, analysis: AnalysisConfig) -> dict[str, Any]:
    """PROTOCOL 4.1's deterministic convention test for the whole run.

    For the first rotation-program frame of every scene, regress resampled
    estimated depth over ground truth against the secant of the pixel angle.
    The report is written before any alignment level runs; evaluation refuses
    to start without it.
    """
    report: dict[str, Any] = {
        "threshold": analysis.depth_convention_slope_threshold,
        "scenes": {},
    }
    for scene in cfg.scenes:
        scene_root = cfg.renders_root / scene
        manifest = load_manifest(scene_root / MANIFEST_NAME)
        first = next(f for f in manifest.frames if f.regime == "rotation")
        depth_cache = load_depth_archive(cfg.cache_root, cfg.depth_encoder, scene)
        resampled, _ = resample_depth_nearest(
            depth_cache["depth"][first.frame_id], (first.height, first.width)
        )
        gt = np.load(scene_root / first.depth_path)
        report["scenes"][scene] = {
            "frame": first.frame_id,
            **secant_regression(
                resampled, gt, first.K, analysis.depth_convention_slope_threshold
            ),
        }
    verdicts = {s: v["verdict"] for s, v in report["scenes"].items()}
    report["verdicts"] = verdicts
    report["unanimous"] = len(set(verdicts.values())) == 1
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4: the estimated-geometry tax.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", type=str)
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--convention", action="store_true",
        help="write the depth-convention report and exit; runs before "
        "anything else per PROTOCOL 4.1",
    )
    parser.add_argument(
        "--doc-verdict", type=str, default=None,
        choices=["planar_z", "ray_distance"],
        help="the depth convention established from VGGT source or "
        "documentation, recorded as the primary authority; a material "
        "disagreement with the regression stops the run",
    )
    parser.add_argument(
        "--gates-only", action="store_true",
        help="run the pure-rotation correctness gates for every level and "
        "write gate evidence, no parquet; PROTOCOL 4.5 runs before any other "
        "result is interpreted",
    )
    args = parser.parse_args(argv)
    cfg = load_phase4_config(args.config)
    analysis = load_analysis_config(cfg.analysis_config)
    if args.list_scenes:
        for index, scene in enumerate(cfg.scenes):
            print(index, scene)
        return

    cfg.evidence_dir.mkdir(parents=True, exist_ok=True)
    convention_path = cfg.evidence_dir / "convention_report.json"

    if args.convention:
        report = convention_report(cfg, analysis)
        if args.doc_verdict is not None:
            report["documentation_verdict"] = args.doc_verdict
            disagree = [
                scene for scene, v in report["verdicts"].items()
                if v in ("planar_z", "ray_distance") and v != args.doc_verdict
            ]
            report["documentation_disagrees_scenes"] = disagree
            if disagree:
                convention_path.write_text(json.dumps(report, indent=1))
                raise SystemExit(
                    "the documented VGGT convention and the deterministic "
                    f"regression disagree on {len(disagree)} scenes; both are "
                    f"recorded in {convention_path}. STOP."
                )
        convention_path.write_text(json.dumps(report, indent=1))
        print(json.dumps(report["verdicts"], indent=1))
        print(f"unanimous: {report['unanimous']} -> {convention_path}")
        return

    if not convention_path.exists():
        raise SystemExit(
            f"{convention_path} does not exist. Run --convention first; "
            "PROTOCOL 4.1 decides the convention before any alignment level runs."
        )
    convention = json.loads(convention_path.read_text(encoding="utf-8"))

    if args.scene is not None:
        scenes = [args.scene]
    elif args.scene_index is not None:
        scenes = [cfg.scenes[args.scene_index]]
    else:
        scenes = list(cfg.scenes)

    train = [s for s in cfg.scenes if scene_split(s) == "train"]
    mean_vector = load_or_build_mean_vector(
        cfg.cache_root, cfg.feature_encoder, train, cfg.mean_vector_dir
    )

    for scene in scenes:
        started = time.perf_counter()
        if args.gates_only:
            _, evidence = evaluate_scene_phase4(
                cfg, scene, mean_vector, analysis, convention,
                regimes=("rotation",), collect_rows=False,
            )
            out = cfg.evidence_dir / f"gates_{scene}.json"
            out.write_text(json.dumps(
                {
                    "metadata": evidence["metadata"],
                    "gate_evidence": evidence["gate_evidence"],
                },
                indent=1,
            ))
            meta = evidence["metadata"]
            print(
                f"[{scene}] gates PASS: coord {meta['gate_coord_max_px']:.2e} px, "
                f"score {meta['gate_score_max_abs']:.2e}, forced "
                f"{meta['gate_forced_max_abs']:.2e} over {meta['gate_checks']} "
                f"checks -> {out}"
            )
            continue

        path = cfg.eval_dir / f"{scene}.parquet"
        if path.exists():
            if not args.resume:
                raise SystemExit(f"{path} exists; pass --resume to skip finished scenes")
            stored = read_run_metadata(path)
            differing = stored is None or any(
                stored.get(field) != value
                for field, value in (
                    ("phase4_version", PHASE4_VERSION),
                    ("seed", cfg.seed),
                    ("phase4_measurement_digest", phase4_measurement_digest(analysis)),
                    ("run_scenes", sorted(cfg.scenes)),
                )
            )
            if differing:
                raise SystemExit(
                    f"{path} was written by a different run; move the "
                    "directory aside rather than mixing populations"
                )
            print(f"[{scene}] results exist, skipping")
            continue
        rows, evidence = evaluate_scene_phase4(cfg, scene, mean_vector, analysis, convention)
        write_rows(path, rows, evidence["metadata"])
        (cfg.evidence_dir / f"audit_{scene}.json").write_text(
            json.dumps(
                {"gate_evidence": evidence["gate_evidence"], "audits": evidence["audits"]},
                indent=1,
            )
        )
        elapsed = time.perf_counter() - started
        print(f"[{scene}] {len(rows)} rows in {elapsed:.1f} s -> {path}")


if __name__ == "__main__":
    main()
