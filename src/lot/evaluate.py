"""Experiment Zero: how far a frozen feature transports, with no training.

PLAN Phase 3. Warp the context features into the target camera using
ground-truth depth, compare the warped values against the target's own
features where the two views see the same surface, and report the result
against the floors that say what a trivial answer would score.

Two paths, because they answer different questions.

The per-point path asks whether a feature value is a property of the surface
it sits on. It samples co-visible target patch centers, computes where each one
lands in the context image from ground-truth geometry, and reads the context
feature there by bilinear interpolation. The target side is the encoder's own
output read without interpolation, and nothing is splatted or pooled, so this
is the cleanest reading of value agreement.

The splat-and-pool path asks whether the whole operation survives being done
for real: forward splat every context pixel, resolve occlusion with a
z-buffer, pool back to the patch grid, and compare patch to patch. It carries
the resampling and occlusion handling the per-point path leaves out, so the
gap between the two paths is the cost of the machinery rather than of the
representation.

Every number is reported beside No-Warp-Copy and Mean-Feature. On its own a
cosine of 0.8 says nothing: frozen features of indoor scenes are similar to
each other everywhere, so the floors are what make a number mean something.
The table stores those floors as ordinary rows rather than pre-computed
margins, so a margin is always a subtraction between two rows of the same pair
and can never disagree with its own parts.

Rows are written one parquet file per scene under
outputs/{experiment_name}/eval/, so an array job writes without collisions and
never overwrites a finished scene.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from .correspondence import gather_value_pairs, sample_correspondences
from .datasets import PairRecord, load_scene_pairs, subsample_by_stratum, summarize_pairs
from .encoders import ENCODERS, PATCH_SIZE, cache_dir, load_cache_meta
from .geometry import relative_pose
from .render_replica import MANIFEST_NAME, REPLICA_SCENES, load_manifest
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

# The per-point sampler names its variants for what it does; the table names
# them for what they are as methods.
_VARIANT_NAMES = {
    "warp": ORACLE_TRANSPORT,
    "no_warp": NO_WARP_COPY,
    "neighbor": NEIGHBOR_PATCH,
    "random": RANDOM_PATCH,
    "mean": MEAN_FEATURE,
}

EVAL_VERSION = 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def unit_normalize(features: Tensor, eps: float = 1e-12) -> Tensor:
    """Scale [..., C] feature vectors to unit length. Zero vectors stay zero."""
    return features / features.norm(dim=-1, keepdim=True).clamp(min=eps)


def value_agreement(prediction: Tensor, target: Tensor) -> tuple[float, float]:
    """Mean cosine and mean L2 between predicted and true features.

    prediction, target: [N, C]. Both are unit normalized first, as CLAUDE.md
    requires, so cosine and L2 measure direction only and are two readings of
    the same quantity. Both are reported because the ladder is read in both.
    Returns (nan, nan) for an empty selection rather than raising, since a pair
    with no co-visible surface is a legitimate outcome to record.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if prediction.numel() == 0:
        return float("nan"), float("nan")
    a = unit_normalize(prediction.to(torch.float32))
    b = unit_normalize(target.to(torch.float32))
    cosine = (a * b).sum(dim=-1)
    l2 = (a - b).norm(dim=-1)
    return float(cosine.mean()), float(l2.mean())


# ---------------------------------------------------------------------------
# Dataset mean feature map
# ---------------------------------------------------------------------------

def dataset_mean_feature_map(
    cache_root: Path, encoder: str, scenes: Sequence[str]
) -> Tensor:
    """Mean cached feature map over the given scenes, as [C, Hp, Wp] float32.

    The Mean-Feature floor is a map, not a single vector, so it keeps whatever
    positional regularity the encoder has: floors below, ceilings above. That
    makes it the honest floor for a predictor that has learned nothing about
    this pair but everything about how rooms are laid out.
    """
    total: Tensor | None = None
    count = 0
    for scene in scenes:
        path = cache_dir(cache_root, encoder, scene) / "features.npz"
        with np.load(path) as archive:
            for name in archive.files:
                array = torch.from_numpy(archive[name]).to(torch.float32)
                total = array if total is None else total + array
                count += 1
    if total is None or count == 0:
        raise ValueError(f"no cached features for {encoder} in {list(scenes)}")
    return total / count


def load_or_build_mean_feature_map(
    cache_root: Path, encoder: str, scenes: Sequence[str], out_dir: Path
) -> Tensor:
    """Read the mean feature map from out_dir, computing and storing it once."""
    out_dir = Path(out_dir)
    path = out_dir / f"mean_feature_{encoder}.npy"
    if path.exists():
        return torch.from_numpy(np.load(path)).to(torch.float32)
    mean = dataset_mean_feature_map(cache_root, encoder, scenes)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, mean.numpy())
    return mean


# ---------------------------------------------------------------------------
# One pair
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PairGeometry:
    """Everything about a pair that does not depend on which encoder is used.

    Computing this once and reusing it across encoders is most of why running
    two encoders costs barely more than one: the visibility test, the
    correspondence sampling, and the splat plan are all encoder blind.
    """

    covisible_fraction: float
    samples: Any
    plan: Any
    patch_selection: Tensor  # [Hp, Wp] bool, patches the splat path is scored on
    coverage_mean: float


def pair_geometry(
    depth_context: Tensor,
    depth_target: Tensor,
    K_context: Tensor,
    K_target: Tensor,
    T_target_from_context: Tensor,
    points_per_pair: int,
    min_covisible_fraction: float,
    patch_size: int = PATCH_SIZE,
    sample_mode: str = "patch_center",
    generator: torch.Generator | None = None,
) -> PairGeometry:
    """Visibility, correspondence sampling, and the splat plan for one pair.

    sample_mode is "patch_center" by default, which samples the target at the
    centers of its patches. There the target value is the encoder's own output
    read without interpolation, which is what "the target's own features" means
    and what a predictor in a later phase would have to produce. Sampling at
    arbitrary pixels instead forces a bilinear read of the target feature map,
    and near a depth edge that blends patches whose correspondences differ, so
    the score picks up interpolation error that has nothing to do with whether
    the representation transports. Measured on the analytic two-plane scene,
    where the exact answer is 1.0, pixel sampling scores 0.970 and patch-center
    sampling scores 1.0. Sampling at patch centers also puts this path on the
    same locations as the splat path, so the difference between the two is the
    cost of the machinery rather than a change of where they look.
    """
    masks = visibility_masks(
        depth_target, depth_context, K_target, K_context, T_target_from_context
    )
    covisible_fraction = float(masks.covisible.to(torch.float32).mean())
    samples = sample_correspondences(
        depth_target,
        K_target,
        K_context,
        T_target_from_context,
        masks.covisible,
        points_per_pair,
        tuple(depth_context.shape),
        patch_size=patch_size,
        mode=sample_mode,
        generator=generator,
    )
    plan = transport_plan(
        depth_context,
        K_context,
        K_target,
        T_target_from_context,
        tuple(depth_target.shape),
        patch_size,
    )
    covisible_per_patch = fraction_per_patch(masks.covisible, patch_size)
    # Score the splat path only where the ground truth says the surface was
    # visible and the splat actually landed something. Scoring a hole would
    # measure the zeros transport leaves behind, not the representation.
    selection = (covisible_per_patch >= min_covisible_fraction) & (plan.coverage > 0)
    coverage_mean = (
        float(plan.coverage[selection].mean()) if bool(selection.any()) else float("nan")
    )
    return PairGeometry(
        covisible_fraction=covisible_fraction,
        samples=samples,
        plan=plan,
        patch_selection=selection,
        coverage_mean=coverage_mean,
    )


def evaluate_pair_for_encoder(
    geometry: PairGeometry,
    features_context: Tensor,
    features_target: Tensor,
    mean_feature_map: Tensor,
    patch_size: int = PATCH_SIZE,
) -> list[dict[str, Any]]:
    """Both paths and every variant for one pair and one encoder.

    Returns partial rows: the metrics and the counts, without the pair identity,
    which the caller attaches.
    """
    rows: list[dict[str, Any]] = []

    values = gather_value_pairs(features_context, features_target, geometry.samples, patch_size)
    target_values = values["target"]
    for key, name in _VARIANT_NAMES.items():
        cosine, l2 = value_agreement(values[key], target_values)
        rows.append(
            {
                "path": PER_POINT,
                "variant": name,
                "n": int(target_values.shape[0]),
                "cosine_mean": cosine,
                "l2_mean": l2,
                "coverage_mean": float("nan"),
            }
        )

    selection = geometry.patch_selection
    transported = apply_transport_plan(geometry.plan, features_context)
    selected = selection.reshape(-1)
    predictions = {
        ORACLE_TRANSPORT: transported,
        NO_WARP_COPY: features_context,
        MEAN_FEATURE: mean_feature_map,
    }
    target_patches = features_target.to(torch.float32).reshape(features_target.shape[0], -1)
    target_patches = target_patches[:, selected].T
    for name, prediction in predictions.items():
        flat = prediction.to(torch.float32).reshape(prediction.shape[0], -1)[:, selected].T
        cosine, l2 = value_agreement(flat, target_patches)
        rows.append(
            {
                "path": SPLAT_POOL,
                "variant": name,
                "n": int(target_patches.shape[0]),
                "cosine_mean": cosine,
                "l2_mean": l2,
                "coverage_mean": geometry.coverage_mean,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EvalConfig:
    """One evaluation experiment. Loaded from a yaml file, one file per run."""

    experiment_name: str
    renders_root: Path
    cache_root: Path
    output_root: Path
    scenes: list[str]
    encoders: list[str]
    max_pairs_per_stratum: int = 40
    points_per_pair: int = 512
    min_covisible_fraction: float = 0.5
    sample_mode: str = "patch_center"
    mean_feature_scenes: list[str] = dataclasses.field(default_factory=list)
    seed: int = 0
    # Manifest intrinsics and poses load as float64, which would drag the whole
    # per-pair pixel pipeline into float64 and cost one to two orders of
    # magnitude on a GPU. Every tolerance in play is far coarser than float32:
    # 1.5 percent for co-visibility, 1e-6 relative for z-buffer ties, half a
    # pixel for rounding.
    geometry_dtype: str = "float32"

    def __post_init__(self) -> None:
        self.renders_root = Path(self.renders_root)
        self.cache_root = Path(self.cache_root)
        self.output_root = Path(self.output_root)
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
        if self.sample_mode not in ("patch_center", "pixel"):
            raise ValueError("sample_mode must be patch_center or pixel")
        if self.geometry_dtype not in ("float32", "float64"):
            raise ValueError("geometry_dtype must be float32 or float64")
        if self.max_pairs_per_stratum <= 0 or self.points_per_pair <= 0:
            raise ValueError("max_pairs_per_stratum and points_per_pair must be positive")
        if not self.mean_feature_scenes:
            # The floor must not be fitted to the scenes it is a floor for, so it
            # defaults to the training split of whatever this run covers.
            from .datasets import scene_split

            self.mean_feature_scenes = [s for s in self.scenes if scene_split(s) == "train"]

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


def evaluate_scene(cfg: EvalConfig, scene: str, mean_maps: dict[str, Tensor]) -> list[dict[str, Any]]:
    """Evaluate one scene's sampled pairs for every configured encoder."""
    scene_root = cfg.renders_root / scene
    manifest = load_manifest(scene_root / MANIFEST_NAME)
    frames = {f.frame_id: f for f in manifest.frames}
    pairs = subsample_by_stratum(
        load_scene_pairs(cfg.renders_root, scene),
        cfg.max_pairs_per_stratum,
        seed=cfg.seed,
    )
    cache = _SceneCache(scene_root, cfg.cache_root, cfg.encoders, scene)
    rows: list[dict[str, Any]] = []
    try:
        for index, pair in enumerate(pairs):
            context = frames[pair.context_frame_id]
            target = frames[pair.target_frame_id]
            K_context = context.K.to(cfg.torch_dtype)
            K_target = target.K.to(cfg.torch_dtype)
            T_target_from_context = relative_pose(
                target.T_world_from_camera, context.T_world_from_camera
            ).to(cfg.torch_dtype)
            geometry = pair_geometry(
                cache.depth(context.depth_path).to(cfg.torch_dtype),
                cache.depth(target.depth_path).to(cfg.torch_dtype),
                K_context,
                K_target,
                T_target_from_context,
                cfg.points_per_pair,
                cfg.min_covisible_fraction,
                sample_mode=cfg.sample_mode,
                generator=torch.Generator().manual_seed(cfg.seed * 1_000_003 + index),
            )
            base = pair.as_row()
            base["covisible_fraction"] = geometry.covisible_fraction
            for encoder in cfg.encoders:
                for row in evaluate_pair_for_encoder(
                    geometry,
                    cache.features(encoder, pair.context_frame_id),
                    cache.features(encoder, pair.target_frame_id),
                    mean_maps[encoder],
                ):
                    rows.append({**base, "encoder": encoder, **row})
    finally:
        cache.close()
    return rows


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write evaluation rows to parquet. Refuses to overwrite."""
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
    mean_maps = {
        encoder: load_or_build_mean_feature_map(
            cfg.cache_root, encoder, cfg.mean_feature_scenes, run_dir
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
        rows = evaluate_scene(cfg, scene, mean_maps)
        write_rows(path, rows)
        pairs = len({(r["context_frame_id"], r["target_frame_id"]) for r in rows})
        elapsed = time.perf_counter() - started
        print(
            f"[{scene}] {pairs} pairs, {len(rows)} rows in {elapsed:.1f} s "
            f"({pairs / max(elapsed, 1e-9):.1f} pairs/s) -> {path}"
        )


if __name__ == "__main__":
    main()
