"""Experiment Zero: how far a frozen feature transports, with no training.

PROTOCOL 3.5. Warp the context features into the target camera using
ground-truth depth, compare the warped values against the target's own features
where the two views see the same surface, and report the result against the
floors that say what a trivial answer would score.

Two paths, because they answer different questions. The per-point path samples
co-visible target patch centres and reads the context feature at the
ground-truth correspondence; nothing is splatted or pooled, so it is the
cleanest reading of value agreement. The splat-and-pool path forward splats
every context pixel, resolves occlusion with a z-buffer, pools back to the patch
grid, and compares patch to patch. PROTOCOL 3.9 reads the gap between them as
the cost of the machinery rather than of the representation.

Every scored record carries the sample identity of PROTOCOL 3.2. Both paths are
indexed on one universe, the target patch grid, so a record on either path names
the same physical correspondence and the two can be intersected. Rows persist
the validity bitmask over that universe, which is what makes a paired difference
a statement about the same samples rather than about two populations.

All five variants are scored on their path's common-valid subset, so a
difference between any two of them is already paired in the sense of PROTOCOL
3.7. The mask on every row is what lets an auditor verify that rather than trust
it.

Rows carry continuous rotation_deg and parallax and no bin labels: binning is the
analysis layer's job and its edges live in the committed analysis config.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .analysis_config import DEFAULT_CONFIG_PATH, AnalysisConfig, load_analysis_config
from .correspondence import (
    NEIGHBOR_OFFSETS,
    choose_in_bounds_offset,
    gather_value_pairs,
    sample_correspondences,
)
from .datasets import load_scene_pairs, subsample_by_stratum
from .encoders import ENCODERS, PATCH_SIZE, cache_dir, patch_grid_shape
from .geometry import relative_pose
from .render_replica import MANIFEST_NAME, REPLICA_SCENES, load_manifest
from .sample_identity import (
    NEIGHBOR_PATCH_SALT,
    RANDOM_PATCH_SALT,
    derived_draw,
    sample_ids,
)
from .transport import apply_transport_plan, transport_plan
from .visibility import fraction_per_patch, visibility_masks

# Method names, used verbatim in the results table, the figures, and the paper.
ORACLE_TRANSPORT = "Oracle-Transport"
NO_WARP_COPY = "No-Warp-Copy"
MEAN_FEATURE = "Mean-Feature"
NEIGHBOR_PATCH = "Neighbor-Patch"
RANDOM_PATCH = "Random-Patch"

PER_POINT = "per_point"
SPLAT_POOL = "splat_pool"

VARIANTS = (ORACLE_TRANSPORT, NO_WARP_COPY, NEIGHBOR_PATCH, RANDOM_PATCH, MEAN_FEATURE)
PATHS = (PER_POINT, SPLAT_POOL)

# The per-point sampler names its reads for what they do; the table names them
# for what they are as methods.
_READ_NAMES = {
    "warp": ORACLE_TRANSPORT,
    "no_warp": NO_WARP_COPY,
    "neighbor": NEIGHBOR_PATCH,
    "random": RANDOM_PATCH,
}

EVAL_VERSION = 2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def unit_normalize(features: Tensor, eps: float = 1e-12) -> Tensor:
    """Scale [..., C] feature vectors to unit length. Zero vectors stay zero."""
    return features / features.norm(dim=-1, keepdim=True).clamp(min=eps)


def value_agreement(
    prediction: Tensor, target: Tensor, center: Tensor | None = None
) -> tuple[float, float]:
    """Mean cosine and mean L2 between predicted and true features.

    prediction, target: [N, C]. center: an optional [C] vector subtracted from
    both before normalizing, which is the centered metric of PROTOCOL 3.7.
    Returns (nan, nan) for an empty selection rather than raising.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if prediction.numel() == 0:
        return float("nan"), float("nan")
    a = prediction.to(torch.float32)
    b = target.to(torch.float32)
    if center is not None:
        a = a - center.to(a.dtype)
        b = b - center.to(b.dtype)
    a = unit_normalize(a)
    b = unit_normalize(b)
    return float((a * b).sum(dim=-1).mean()), float((a - b).norm(dim=-1).mean())


def agreement_metrics(
    prediction: Tensor, target: Tensor, center: Tensor, centered_defined: bool = True
) -> dict[str, float]:
    """The four metric columns of PROTOCOL 3.2 for one prediction.

    centered_defined is False for Mean-Feature alone. Its prediction is the mean
    vector itself, so centering sends it to the zero vector and its centered
    cosine is undefined. PROTOCOL 3.7 requires that be recorded as not
    applicable and forbids manufacturing a score from an epsilon-regularized
    zero vector, so the centered columns of a Mean-Feature row are nonfinite and
    no other row in the table carries a nonfinite value.
    """
    cosine, l2 = value_agreement(prediction, target)
    if centered_defined:
        cosine_centered, l2_centered = value_agreement(prediction, target, center=center)
    else:
        cosine_centered, l2_centered = float("nan"), float("nan")
    return {
        "cosine_mean": cosine,
        "l2_mean": l2,
        "cosine_centered_mean": cosine_centered,
        "l2_centered_mean": l2_centered,
    }


# ---------------------------------------------------------------------------
# The global mean vector: floor and centering statistic in one object
# ---------------------------------------------------------------------------

def dataset_mean_vector(cache_root: Path, encoder: str, scenes: Sequence[str]) -> Tensor:
    """One global D-vector per encoder, over all frames and all positions.

    PROTOCOL 3.6 freezes Mean-Feature as exactly this object, and PROTOCOL 3.7
    makes the same vector the centering statistic, so the two cannot drift
    apart. A position-conditioned mean map is explicitly not used: subtracting a
    per-position mean would remove the stationary positional component that the
    position-indexed VGGT finding measures, and a map used as a prediction can
    beat the correct answer, which is the artifact the frozen definition exists
    to prevent.
    """
    total: Tensor | None = None
    count = 0
    for scene in scenes:
        path = cache_dir(cache_root, encoder, scene) / "features.npz"
        with np.load(path) as archive:
            for name in archive.files:
                array = torch.from_numpy(archive[name]).to(torch.float32)
                per_frame = array.reshape(array.shape[0], -1).mean(dim=1)
                total = per_frame if total is None else total + per_frame
                count += 1
    if total is None or count == 0:
        raise ValueError(f"no cached features for {encoder} in {list(scenes)}")
    return total / count


def load_or_build_mean_vector(
    cache_root: Path, encoder: str, scenes: Sequence[str], out_dir: Path
) -> Tensor:
    """Read the global mean vector from out_dir, computing and storing it once."""
    out_dir = Path(out_dir)
    path = out_dir / f"mean_vector_{encoder}.npy"
    if path.exists():
        return torch.from_numpy(np.load(path)).to(torch.float32)
    mean = dataset_mean_vector(cache_root, encoder, scenes)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, mean.numpy())
    return mean


# ---------------------------------------------------------------------------
# Validity masks over the shared universe
# ---------------------------------------------------------------------------

def assert_unique_sample_ids(ids: np.ndarray, where: str) -> None:
    """A hash collision would silently merge two correspondences' masks.

    sample_id is a 64-bit mix, so a collision inside one pair is vanishingly
    unlikely, but "vanishingly unlikely" is not "impossible" and the failure
    mode is silent: two distinct physical correspondences would share a mask bit
    and every paired difference computed from that mask would be wrong without
    any surface symptom. Cheap to check, so it is checked.
    """
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError(
            f"sample_id collision in {where}: {len(ids)} records but "
            f"{len(np.unique(ids))} distinct ids"
        )


def pack_mask(mask: np.ndarray) -> bytes:
    """Pack a boolean universe mask to bytes for storage."""
    return np.packbits(np.asarray(mask, dtype=bool)).tobytes()


def unpack_mask(blob: bytes, size: int) -> np.ndarray:
    """Inverse of pack_mask, given the universe size."""
    return np.unpackbits(np.frombuffer(blob, dtype=np.uint8))[:size].astype(bool)


def universe_sample_ids(
    scene: str, context_frame_id: str, target_frame_id: str, grid: tuple[int, int]
) -> np.ndarray:
    """sample_id for every cell of the target patch grid, row major.

    Both paths index this one universe, so a per-point record and a
    splat-and-pool record naming the same patch centre carry the same id.
    """
    patches_h, patches_w = grid
    rows = torch.arange(patches_h, dtype=torch.float64)
    cols = torch.arange(patches_w, dtype=torch.float64)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    centers = torch.stack(
        (
            (grid_c.reshape(-1) + 0.5) * PATCH_SIZE - 0.5,
            (grid_r.reshape(-1) + 0.5) * PATCH_SIZE - 0.5,
        ),
        dim=-1,
    )
    return sample_ids(scene, context_frame_id, target_frame_id, centers)


# ---------------------------------------------------------------------------
# One pair
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PairGeometry:
    """Everything about a pair that does not depend on which encoder is used."""

    covisible_fraction: float
    parallax: float
    samples: Any
    plan: Any
    grid: tuple[int, int]
    universe_ids: np.ndarray
    per_point_mask: np.ndarray      # [Hp * Wp] bool, cells scored per-point
    splat_mask: np.ndarray          # [Hp * Wp] bool, cells scored splat-and-pool
    random_patch: np.ndarray        # [Hp * Wp] int, hashed source patch per cell
    coverage_mean: float

    @property
    def size(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def scorable(self) -> bool:
        return bool(self.per_point_mask.any() and self.splat_mask.any())


def pair_parallax(baseline_m: float, depth_target: Tensor, covisible: Tensor) -> float:
    """PROTOCOL 3.2's parallax: median of baseline over ground-truth depth.

    The median runs over the pair's co-visible point set, not over the whole
    frame. Baseline is constant within a pair, so this equals baseline over the
    median co-visible depth; the population is what matters, because a
    whole-frame median is a different and larger one and this quantity is the
    binning variable for the primary parallax analysis.
    """
    values = depth_target[covisible]
    values = values[(values > 0) & torch.isfinite(values)]
    if values.numel() == 0:
        return float("nan")
    return float(baseline_m / values.median())


def pair_geometry(
    depth_context: Tensor,
    depth_target: Tensor,
    K_context: Tensor,
    K_target: Tensor,
    T_target_from_context: Tensor,
    scene: str,
    context_frame_id: str,
    target_frame_id: str,
    config: AnalysisConfig,
    patch_size: int = PATCH_SIZE,
    sample_mode: str = "patch_center",
) -> PairGeometry:
    """Visibility, correspondence sampling, the splat plan, and the masks."""
    masks = visibility_masks(
        depth_target,
        depth_context,
        K_target,
        K_context,
        T_target_from_context,
        rel_tol=config.covisible_relative_depth_tol,
    )
    covisible_fraction = float(masks.covisible.to(torch.float32).mean())
    baseline = float(torch.linalg.vector_norm(T_target_from_context[:3, 3]))
    parallax = pair_parallax(baseline, depth_target, masks.covisible)

    samples = sample_correspondences(
        depth_target,
        K_target,
        K_context,
        T_target_from_context,
        masks.covisible,
        config.points_per_pair,
        tuple(depth_context.shape),
        scene,
        context_frame_id,
        target_frame_id,
        patch_size=patch_size,
        mode=sample_mode,
        depth_consistency_tol=config.covisible_relative_depth_tol,
    )
    plan = transport_plan(
        depth_context,
        K_context,
        K_target,
        T_target_from_context,
        tuple(depth_target.shape),
        patch_size,
    )
    grid = patch_grid_shape(tuple(depth_target.shape), patch_size)
    patches_h, patches_w = grid
    size = patches_h * patches_w

    per_point_mask = np.zeros(size, dtype=bool)
    if samples.uv_target.shape[0]:
        cols = torch.round((samples.uv_target[:, 0] + 0.5) / patch_size - 0.5).long()
        rows = torch.round((samples.uv_target[:, 1] + 0.5) / patch_size - 0.5).long()
        per_point_mask[(rows * patches_w + cols).cpu().numpy()] = True

    covisible_per_patch = fraction_per_patch(masks.covisible, patch_size)
    splat_mask = (
        (covisible_per_patch >= config.min_covisible_fraction) & (plan.coverage > 0)
    ).reshape(-1).cpu().numpy()

    ids = universe_sample_ids(scene, context_frame_id, target_frame_id, grid)
    where = f"{scene} {context_frame_id} -> {target_frame_id}"
    assert_unique_sample_ids(ids, f"{where} universe")
    assert_unique_sample_ids(samples.sample_id, f"{where} per-point samples")
    coverage = plan.coverage.reshape(-1)[torch.from_numpy(splat_mask)]
    return PairGeometry(
        covisible_fraction=covisible_fraction,
        parallax=parallax,
        samples=samples,
        plan=plan,
        grid=grid,
        universe_ids=ids,
        per_point_mask=per_point_mask,
        splat_mask=splat_mask,
        random_patch=derived_draw(ids, RANDOM_PATCH_SALT, size),
        coverage_mean=float(coverage.mean()) if coverage.numel() else float("nan"),
    )


def cross_path_record_difference(geometry: PairGeometry) -> dict[str, int]:
    """How the two paths' record sets differ, and why.

    After the shared direction rule and the removal of the neighbour drop, the
    only legitimate source of a difference is transport coverage: cells the warp
    does not support cannot be scored on the splat path, while the per-point
    path reads its correspondence directly. That is a property of the operator
    and is reported. Anything else appearing here means one path is selecting
    records by a rule the other does not apply, which is the fault the shared
    rule exists to prevent, so the components are counted separately rather than
    summarized.
    """
    per_point = geometry.per_point_mask
    splat = geometry.splat_mask
    uncovered = (geometry.plan.coverage.reshape(-1) <= 0).cpu().numpy()
    only_per_point = per_point & ~splat
    return {
        "both": int((per_point & splat).sum()),
        "per_point_only": int(only_per_point.sum()),
        "per_point_only_uncovered": int((only_per_point & uncovered).sum()),
        "splat_only": int((splat & ~per_point).sum()),
    }


def assert_source_read_sets_agree(geometry: PairGeometry, where: str) -> None:
    """Every cell one path reads must be readable on the other, bar coverage.

    The read-equality test compares records present on both paths and therefore
    cannot see a record that is absent from one. This closes that gap: a cell
    the per-point path scored but the splat path did not must be a cell the warp
    failed to cover, never a cell some null's own rule removed.
    """
    difference = cross_path_record_difference(geometry)
    unexplained = difference["per_point_only"] - difference["per_point_only_uncovered"]
    if unexplained:
        raise RuntimeError(
            f"{where}: {unexplained} cells are scored per-point but absent from "
            "the splat path for a reason other than transport coverage; the two "
            "paths are selecting records by different rules"
        )


def _shifted_source(features: Tensor, offset: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Context map shifted by one patch, with a validity flag per source patch.

    Returns (shifted [C, Hp * Wp], valid [Hp * Wp]). Source patches whose shifted
    read falls outside the grid are zeroed and marked invalid, so a target patch
    drawing any weight from one can be omitted rather than scored against a
    fabricated value.
    """
    channels, patches_h, patches_w = features.shape
    dx, dy = offset
    valid = torch.zeros((patches_h, patches_w), dtype=torch.bool, device=features.device)
    shifted = torch.zeros_like(features)
    src_r0, src_r1 = max(0, dy), min(patches_h, patches_h + dy)
    src_c0, src_c1 = max(0, dx), min(patches_w, patches_w + dx)
    dst_r0, dst_r1 = src_r0 - dy, src_r1 - dy
    dst_c0, dst_c1 = src_c0 - dx, src_c1 - dx
    shifted[:, dst_r0:dst_r1, dst_c0:dst_c1] = features[:, src_r0:src_r1, src_c0:src_c1]
    valid[dst_r0:dst_r1, dst_c0:dst_c1] = True
    return shifted.reshape(channels, -1), valid.reshape(-1)


def splat_neighbor_prediction(
    geometry: PairGeometry, features_context: Tensor
) -> tuple[Tensor, np.ndarray, np.ndarray]:
    """Neighbor-Patch through the splat plan: transport a one-patch-shifted read.

    PROTOCOL 3.6 says the null reads one patch away from the correct
    correspondence and is transported identically. On this path the
    correspondence is the plan, so the same weights are applied to sources
    displaced by one patch in the record's own direction.

    The direction is chosen by the same rule the per-point sampler uses, from
    the same sample_id, among the offsets that leave the cell's whole support in
    bounds. Hashing among all four instead and dropping the ones that leak would
    give a border record one direction here and another there, which would make
    the variant incomparable across paths while looking correct in isolation.

    Every cell keeps its record. PROTOCOL 3.6's omission clause covers only the
    case where no offset is in bounds at all, which cannot arise on a patch grid
    of two or more: a cell's support can span the full width or the full height,
    but not both, so at least one axis always has a usable direction. Dropping
    cells whose chosen direction happened to leak would be selecting the record
    set by a different rule on this path than on the other, which is the same
    fault the shared direction rule exists to prevent. The impossible case is
    asserted rather than silently absorbed.

    Returns the prediction per universe cell, the mask of cells with any usable
    offset, and the chosen offset index per cell.
    """
    weights = geometry.plan.weights
    channels = features_context.shape[0]
    cells_total = weights.shape[0]
    features = features_context.to(device=weights.device, dtype=torch.float32)

    shifted_maps = []
    option_ok = np.zeros((cells_total, len(NEIGHBOR_OFFSETS)), dtype=bool)
    for index, offset in enumerate(NEIGHBOR_OFFSETS):
        shifted, valid = _shifted_source(features, offset)
        shifted_maps.append(shifted)
        leaked = (weights * (~valid).to(weights.dtype)[None, :]).sum(dim=1)
        option_ok[:, index] = (leaked <= 0).cpu().numpy()

    defined = option_ok.any(axis=1)
    stranded = int((~defined).sum())
    if stranded:
        raise RuntimeError(
            f"{stranded} cells have no in-bounds neighbour offset on a "
            f"{geometry.grid[0]} by {geometry.grid[1]} patch grid, which the "
            "geometry of the offsets makes impossible; the plan or the grid is wrong"
        )
    direction = choose_in_bounds_offset(geometry.universe_ids, option_ok)

    prediction = torch.zeros(
        (channels, cells_total), dtype=torch.float32, device=weights.device
    )
    for index in range(len(NEIGHBOR_OFFSETS)):
        cells = np.flatnonzero(defined & (direction == index))
        if cells.size == 0:
            continue
        rows = torch.from_numpy(cells).to(weights.device)
        prediction[:, rows] = shifted_maps[index] @ weights[rows].mT
    return prediction, defined, direction


def evaluate_pair_for_encoder(
    geometry: PairGeometry,
    features_context: Tensor,
    features_target: Tensor,
    mean_vector: Tensor,
    patch_size: int = PATCH_SIZE,
) -> list[dict[str, Any]]:
    """Every variant on both paths for one pair and one encoder.

    Returns partial rows without the pair identity, which the caller attaches.
    All five variants on a path are scored on that path's common-valid subset,
    so a difference between any two is paired by construction and the stored
    mask proves it.
    """
    rows: list[dict[str, Any]] = []
    center = mean_vector.to(torch.float32)

    # ---- per-point path -------------------------------------------------
    reads = gather_value_pairs(features_context, features_target, geometry.samples, patch_size)
    target_values = reads["target"]
    count = int(target_values.shape[0])
    predictions: dict[str, Tensor] = {_READ_NAMES[key]: reads[key] for key in _READ_NAMES}
    predictions[MEAN_FEATURE] = center[None, :].expand(count, -1)
    blob = pack_mask(geometry.per_point_mask)
    for variant in VARIANTS:
        rows.append(
            {
                "path": PER_POINT,
                "variant": variant,
                "n": count,
                "sample_mask": blob,
                **agreement_metrics(
                    predictions[variant], target_values, center, variant != MEAN_FEATURE
                ),
                "coverage_mean": float("nan"),
            }
        )

    # ---- splat-and-pool path --------------------------------------------
    channels = features_context.shape[0]
    transported = apply_transport_plan(geometry.plan, features_context).reshape(channels, -1)
    neighbor, _, _ = splat_neighbor_prediction(geometry, features_context)
    # The splat record set is transport coverage and ground-truth co-visibility,
    # nothing else. A cross-path n difference is then a property of the operator,
    # which is worth reporting, rather than an artifact of one null.
    selected = geometry.splat_mask
    index = torch.from_numpy(np.flatnonzero(selected)).to(transported.device)
    flat_context = features_context.to(torch.float32).reshape(channels, -1)
    flat_target = features_target.to(torch.float32).reshape(channels, -1)
    random_source = torch.from_numpy(geometry.random_patch).to(flat_context.device)
    splat_predictions = {
        ORACLE_TRANSPORT: transported[:, index].T,
        NO_WARP_COPY: flat_context[:, index].T,
        NEIGHBOR_PATCH: neighbor[:, index].T,
        RANDOM_PATCH: flat_context[:, random_source[index]].T,
        MEAN_FEATURE: center[None, :].expand(int(index.numel()), -1),
    }
    targets = flat_target[:, index].T
    blob = pack_mask(selected)
    for variant in VARIANTS:
        rows.append(
            {
                "path": SPLAT_POOL,
                "variant": variant,
                "n": int(index.numel()),
                "sample_mask": blob,
                **agreement_metrics(
                    splat_predictions[variant], targets, center, variant != MEAN_FEATURE
                ),
                "coverage_mean": geometry.coverage_mean,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EvalConfig:
    """One evaluation run. Numeric constants come from the analysis config."""

    experiment_name: str
    renders_root: Path
    cache_root: Path
    output_root: Path
    scenes: list[str]
    encoders: list[str]
    mean_vector_scenes: list[str] = dataclasses.field(default_factory=list)
    seed: int = 0
    analysis_config: Path = DEFAULT_CONFIG_PATH
    # Manifest intrinsics and poses load as float64, which would drag the whole
    # per-pair pixel pipeline with them for one to two orders of magnitude on a
    # GPU. Every tolerance in play is far coarser than float32.
    geometry_dtype: str = "float32"

    def __post_init__(self) -> None:
        self.renders_root = Path(self.renders_root)
        self.cache_root = Path(self.cache_root)
        self.output_root = Path(self.output_root)
        self.analysis_config = Path(self.analysis_config)
        if not self.scenes:
            raise ValueError("config lists no scenes")
        unknown = [s for s in self.scenes if s not in REPLICA_SCENES]
        if unknown:
            raise ValueError(f"unknown Replica scenes: {unknown}")
        if not self.encoders:
            raise ValueError("config lists no encoders")
        unknown_encoders = [e for e in self.encoders if e not in ENCODERS]
        if unknown_encoders:
            raise ValueError(f"unknown encoders: {unknown_encoders}")
        if self.geometry_dtype not in ("float32", "float64"):
            raise ValueError("geometry_dtype must be float32 or float64")
        if not self.mean_vector_scenes:
            # PROTOCOL 3.6 averages the mean vector over the training split, so
            # the floor and the centering statistic never adapt to the frames
            # they are a floor for.
            from .datasets import scene_split

            self.mean_vector_scenes = [s for s in self.scenes if scene_split(s) == "train"]

    @property
    def eval_dir(self) -> Path:
        return self.output_root / self.experiment_name / "eval"

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float32 if self.geometry_dtype == "float32" else torch.float64


def load_eval_config(path: Path) -> EvalConfig:
    """Load an EvalConfig from yaml. Unknown keys are an error, not a warning."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    allowed = {f.name for f in dataclasses.fields(EvalConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    missing = [
        k
        for k in ("experiment_name", "renders_root", "cache_root", "output_root", "scenes", "encoders")
        if k not in raw
    ]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    return EvalConfig(**raw)


# ---------------------------------------------------------------------------
# Scene evaluation
# ---------------------------------------------------------------------------

class _SceneCache:
    """Lazily read depth maps and cached features for one scene, once each."""

    def __init__(self, scene_root: Path, cache_root: Path, encoders: Sequence[str], scene: str):
        self.scene_root = Path(scene_root)
        self._depth: dict[str, Tensor] = {}
        self._archives = {
            encoder: np.load(cache_dir(cache_root, encoder, scene) / "features.npz")
            for encoder in encoders
        }
        self._features: dict[tuple[str, str], Tensor] = {}

    def depth(self, relative_path: str) -> Tensor:
        if relative_path not in self._depth:
            self._depth[relative_path] = torch.from_numpy(
                np.load(self.scene_root / relative_path)
            )
        return self._depth[relative_path]

    def features(self, encoder: str, frame_id: str) -> Tensor:
        key = (encoder, frame_id)
        if key not in self._features:
            self._features[key] = torch.from_numpy(self._archives[encoder][frame_id])
        return self._features[key]

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()


def evaluate_scene(
    cfg: EvalConfig, scene: str, mean_vectors: dict[str, Tensor], analysis: AnalysisConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one scene's sampled pairs. Returns (rows, run metadata)."""
    scene_root = cfg.renders_root / scene
    manifest = load_manifest(scene_root / MANIFEST_NAME)
    frames = {f.frame_id: f for f in manifest.frames}
    pairs = subsample_by_stratum(
        load_scene_pairs(cfg.renders_root, scene, config=analysis),
        analysis.max_pairs_per_stratum,
        seed=cfg.seed,
        config=analysis,
    )
    cache = _SceneCache(scene_root, cfg.cache_root, cfg.encoders, scene)
    rows: list[dict[str, Any]] = []
    dropped_unscorable = 0
    neighbor_omitted = 0
    universe_size = 0
    try:
        for pair in pairs:
            context = frames[pair.context_frame_id]
            target = frames[pair.target_frame_id]
            T_target_from_context = relative_pose(
                target.T_world_from_camera, context.T_world_from_camera
            ).to(cfg.torch_dtype)
            geometry = pair_geometry(
                cache.depth(context.depth_path).to(cfg.torch_dtype),
                cache.depth(target.depth_path).to(cfg.torch_dtype),
                context.K.to(cfg.torch_dtype),
                target.K.to(cfg.torch_dtype),
                T_target_from_context,
                scene,
                pair.context_frame_id,
                pair.target_frame_id,
                analysis,
            )
            neighbor_omitted += geometry.samples.neighbor_omitted
            universe_size = geometry.size
            if not geometry.scorable:
                # PROTOCOL 3.2 permits exactly one nonfinite representation, the
                # centered columns of Mean-Feature. A pair with no scored
                # surface would otherwise write nonfinite metrics everywhere, so
                # it is dropped and counted here instead.
                dropped_unscorable += 1
                continue
            base = pair.as_row()
            base["covisible_fraction"] = geometry.covisible_fraction
            base["parallax"] = geometry.parallax
            for encoder in cfg.encoders:
                for row in evaluate_pair_for_encoder(
                    geometry,
                    cache.features(encoder, pair.context_frame_id),
                    cache.features(encoder, pair.target_frame_id),
                    mean_vectors[encoder],
                ):
                    rows.append({**base, "encoder": encoder, **row})
    finally:
        cache.close()
    metadata = {
        "eval_version": EVAL_VERSION,
        "scene": scene,
        "pairs_considered": len(pairs),
        "pairs_dropped_unscorable": dropped_unscorable,
        "neighbor_patch_omitted_records": neighbor_omitted,
        "universe_size": universe_size,
        "encoders": list(cfg.encoders),
        "seed": cfg.seed,
        "max_pairs_per_stratum": analysis.max_pairs_per_stratum,
        # Why realized bin populations differ from the design targets. Strata are
        # formed on a whole-frame parallax proxy, because PROTOCOL 3.2's
        # statistic is a median over the co-visible set and is not known until
        # visibility has been computed. The proxy is a sampling covariate only:
        # pairs are never selected on outcomes, so per-bin conditional estimates
        # are unbiased, and the rows carry the protocol statistic that the
        # analysis actually bins on. A re-audit comparing design targets against
        # realized counts should expect them to differ for this reason.
        "stratum_parallax_is_a_sampling_proxy": True,
        "stratum_parallax_definition": (
            "baseline over the context frame's whole-frame median depth, from "
            "the frame-stats sidecar; used to form strata only"
        ),
        "row_parallax_definition": (
            "PROTOCOL 3.2: median of baseline over ground-truth depth across "
            "the pair's co-visible point set"
        ),
    }
    return rows, metadata


def write_rows(path: Path, rows: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Write evaluation rows to parquet beside their run metadata. Never overwrites."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} exists; outputs are never overwritten. Delete it to re-evaluate."
        )
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    pq.write_table(pa.Table.from_pylist(list(rows)), tmp)
    tmp.replace(path)
    path.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=1), encoding="utf-8"
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read evaluation rows back from parquet."""
    import pyarrow.parquet as pq

    return pq.read_table(Path(path)).to_pylist()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment Zero for LT.")
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scene", type=str, help="evaluate a single scene by name")
    group.add_argument(
        "--scene-index",
        type=int,
        help="evaluate a single scene by index into the config scene list "
        "(for SLURM array jobs)",
    )
    parser.add_argument(
        "--list-scenes", action="store_true", help="print the scene list and exit"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip scenes that already have results instead of failing",
    )
    args = parser.parse_args(argv)
    cfg = load_eval_config(args.config)
    analysis = load_analysis_config(cfg.analysis_config)
    if args.list_scenes:
        for i, scene in enumerate(cfg.scenes):
            print(i, scene)
        return
    if args.scene is not None:
        if args.scene not in cfg.scenes:
            raise SystemExit(f"scene {args.scene!r} not in config scene list")
        scenes = [args.scene]
    elif args.scene_index is not None:
        if not 0 <= args.scene_index < len(cfg.scenes):
            raise SystemExit(
                f"--scene-index {args.scene_index} outside 0..{len(cfg.scenes) - 1}"
            )
        scenes = [cfg.scenes[args.scene_index]]
    else:
        scenes = list(cfg.scenes)

    run_dir = cfg.output_root / cfg.experiment_name
    mean_vectors = {
        encoder: load_or_build_mean_vector(
            cfg.cache_root, encoder, cfg.mean_vector_scenes, run_dir
        )
        for encoder in cfg.encoders
    }
    for scene in scenes:
        path = cfg.eval_dir / f"{scene}.parquet"
        if path.exists():
            if args.resume:
                print(f"[{scene}] results exist, skipping")
                continue
            raise SystemExit(f"{path} exists; pass --resume to skip finished scenes")
        started = time.perf_counter()
        rows, metadata = evaluate_scene(cfg, scene, mean_vectors, analysis)
        write_rows(path, rows, metadata)
        pairs = len({(r["context_frame_id"], r["target_frame_id"]) for r in rows})
        elapsed = time.perf_counter() - started
        print(
            f"[{scene}] {pairs} pairs, {len(rows)} rows in {elapsed:.1f} s "
            f"({pairs / max(elapsed, 1e-9):.1f} pairs/s), "
            f"{metadata['pairs_dropped_unscorable']} dropped -> {path}"
        )


if __name__ == "__main__":
    main()
