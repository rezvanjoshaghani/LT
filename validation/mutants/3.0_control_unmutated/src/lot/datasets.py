"""Context-target pairs, regime tags, parallax bins, and scene splits.

A pair is one context frame and one target frame of the same scene, the same
base viewpoint, and the same camera program. Pairs never cross viewpoints:
viewpoints are at least 0.75 m apart and generally look at different parts of
the scene, so a cross-viewpoint pair would mostly measure how little the two
views share rather than how well a representation transports.

Pairs are ordered. Transport is a directional operation, and the disoccluded
region of a pair depends on which frame is the context, so context A to target
B and context B to target A are two different measurements rather than one
measured twice.

Parallax is baseline over median scene depth, taken from the context frame:
the context view is the one being transported, so its scale is the one that
decides how much the geometry has to move. The per-frame medians come from the
frame-stats sidecar written in Phase 1, so building pairs reads no depth files.

Nothing here is stored to disk. Pair construction is pure and deterministic
given a manifest, a sidecar, and a seed, and every evaluation row carries its
pair's identity, so the pairs are recoverable from the results alone.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .geometry import baseline_m, parallax_from_median_depth, relative_pose, rotation_angle_deg
from .render_replica import (
    FRAME_STATS_NAME,
    MANIFEST_NAME,
    REGIMES,
    REPLICA_SCENES_TEST,
    REPLICA_SCENES_TRAIN,
    FrameRecord,
    Manifest,
    load_frame_stats,
    load_manifest,
    usable_frame_ids,
)

# A viewpoint change has two components and they are not interchangeable. The
# baseline decides how much depth-dependent re-mapping is needed; the rotation
# decides how far a surface point travels across the image. Stratifying by only
# one of them pools situations that differ in the other, which is what happened
# in the first Experiment Zero run: every in-place rotation pair has zero
# baseline, so one parallax bin held 7.5 to 60 degrees of yaw, and 60 degrees of
# a 90 degree field of view sweeps a point across two thirds of the frame.
# Every pair is therefore binned on both axes.

# Upper edges of the parallax bins. The open-ended last bin catches pairs that
# combine two opposite translations, which reach twice the largest single
# baseline the camera program targets.
PARALLAX_BIN_EDGES: tuple[float, ...] = (0.025, 0.05, 0.1, 0.2, 0.4, float("inf"))

# Upper edges of the rotation bins, in degrees. The camera programs produce
# pairwise angles from 7.5 up to 60.
ROTATION_BIN_EDGES: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, float("inf"))

# Below these a pair has no baseline, or no rotation, at all. Both are exactly
# zero by construction for the programs that hold one fixed, so these only have
# to absorb floating point noise in the pose arithmetic, not make a judgement
# about what counts as small.
ZERO_PARALLAX_TOL = 1e-9
ZERO_ROTATION_TOL_DEG = 1e-6

ZERO_BIN = "zero"
ZERO_PARALLAX_BIN = ZERO_BIN  # kept for readers of the earlier name


def _bin_label(value: float, edges: Sequence[float], zero_tol: float, name: str) -> str:
    """Label the bin a non-negative value falls in.

    Bins are half open on the left, so a value sits in the first bin whose upper
    edge it does not exceed. Exact zero gets its own label rather than being
    folded into the smallest bin, because no baseline at all, or no rotation at
    all, is a different physical situation from a small one.
    """
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    if value < zero_tol:
        return ZERO_BIN
    lower = 0.0
    for edge in edges:
        if value <= edge:
            return f"{lower:g}-{edge:g}" if math.isfinite(edge) else f"{lower:g}+"
        lower = edge
    raise AssertionError("the last bin edge must be infinite")


def _bin_order(edges: Sequence[float]) -> list[str]:
    """Bin labels in increasing order, for stable figures and tables."""
    labels = [ZERO_BIN]
    lower = 0.0
    for edge in edges:
        labels.append(f"{lower:g}-{edge:g}" if math.isfinite(edge) else f"{lower:g}+")
        lower = edge
    return labels


def parallax_bin(value: float) -> str:
    """Label the parallax bin a value falls in. Pure rotation lands in "zero"."""
    return _bin_label(value, PARALLAX_BIN_EDGES, ZERO_PARALLAX_TOL, "parallax")


def parallax_bin_order() -> list[str]:
    """Parallax bin labels in increasing order."""
    return _bin_order(PARALLAX_BIN_EDGES)


def rotation_bin(degrees: float) -> str:
    """Label the rotation bin an angle falls in. Pure translation lands in "zero"."""
    return _bin_label(degrees, ROTATION_BIN_EDGES, ZERO_ROTATION_TOL_DEG, "rotation")


def rotation_bin_order() -> list[str]:
    """Rotation bin labels in increasing order."""
    return _bin_order(ROTATION_BIN_EDGES)


def scene_split(scene: str) -> str:
    """Which split a scene belongs to. Raises ValueError for an unknown scene."""
    if scene in REPLICA_SCENES_TRAIN:
        return "train"
    if scene in REPLICA_SCENES_TEST:
        return "test"
    raise ValueError(f"unknown scene {scene!r}")


@dataclasses.dataclass(frozen=True)
class PairRecord:
    """One directed context-target pair and the quantities it is stratified by."""

    scene: str
    split: str
    viewpoint: int
    regime: str
    context_frame_id: str
    target_frame_id: str
    baseline_m: float
    context_median_depth_m: float
    parallax: float
    parallax_bin: str
    rotation_deg: float
    rotation_bin: str

    def as_row(self) -> dict[str, Any]:
        """Flat dict for a results table."""
        return dataclasses.asdict(self)


def _viewpoint_of(frame: FrameRecord) -> int:
    viewpoint = frame.params.get("viewpoint")
    if viewpoint is None:
        raise ValueError(f"{frame.frame_id}: manifest params carry no viewpoint")
    return int(viewpoint)


def pair_quantities(
    context: FrameRecord, target: FrameRecord, context_median_depth_m: float
) -> tuple[float, float, float]:
    """Baseline, parallax, and rotation magnitude for one directed pair.

    Returns (baseline_m, parallax, rotation_deg). The relative transform comes
    from geometry.relative_pose, the one place the formula is written.
    """
    T_target_from_context = relative_pose(
        target.T_world_from_camera, context.T_world_from_camera
    )
    return (
        float(baseline_m(T_target_from_context)),
        float(parallax_from_median_depth(T_target_from_context, context_median_depth_m)),
        rotation_angle_deg(T_target_from_context[:3, :3]),
    )


def build_scene_pairs(
    manifest: Manifest,
    stats: dict[str, Any],
    regimes: Sequence[str] = REGIMES,
    **usability_policy: float,
) -> list[PairRecord]:
    """Every directed pair of usable frames sharing a scene, viewpoint, and regime.

    manifest: the scene's render manifest.
    stats: the scene's frame-stats sidecar, which supplies both the usability
        verdict and the per-frame median depth the parallax is taken against.
    usability_policy: forwarded to render_replica.usable_frame_ids, so an
        evaluation can widen or narrow what counts as a usable frame without
        touching this function.
    """
    usable = usable_frame_ids(stats, **usability_policy)
    measured = stats["frames"]
    split = scene_split(manifest.scene)
    groups: dict[tuple[int, str], list[FrameRecord]] = {}
    for frame in manifest.frames:
        if frame.frame_id not in usable or frame.regime not in regimes:
            continue
        groups.setdefault((_viewpoint_of(frame), frame.regime), []).append(frame)

    pairs: list[PairRecord] = []
    for (viewpoint, regime), frames in sorted(groups.items()):
        for context in frames:
            median = measured[context.frame_id]["median_m"]
            if median is None or not math.isfinite(median) or median <= 0:
                # usable_frame_ids already requires enough valid depth for a
                # median to exist, so this only guards a policy widened past it.
                continue
            for target in frames:
                if target.frame_id == context.frame_id:
                    continue
                baseline, value, rotation = pair_quantities(context, target, float(median))
                pairs.append(
                    PairRecord(
                        scene=manifest.scene,
                        split=split,
                        viewpoint=viewpoint,
                        regime=regime,
                        context_frame_id=context.frame_id,
                        target_frame_id=target.frame_id,
                        baseline_m=baseline,
                        context_median_depth_m=float(median),
                        parallax=value,
                        parallax_bin=parallax_bin(value),
                        rotation_deg=rotation,
                        rotation_bin=rotation_bin(rotation),
                    )
                )
    return pairs


def load_scene_pairs(renders_root: Path, scene: str, **kwargs: Any) -> list[PairRecord]:
    """Build one scene's pairs from its manifest and frame-stats sidecar on disk."""
    scene_root = Path(renders_root) / scene
    manifest = load_manifest(scene_root / MANIFEST_NAME)
    stats = load_frame_stats(scene_root / FRAME_STATS_NAME)
    return build_scene_pairs(manifest, stats, **kwargs)


def stratum_of(pair: PairRecord) -> tuple[str, str, str, str]:
    """The stratum a pair is sampled and reported within.

    Scene, regime, and both axes of viewpoint change. Binning on one axis alone
    would pool pairs that differ on the other: rotation pairs all share a
    parallax of zero, and translation pairs all share a rotation of zero, so
    either axis by itself collapses one whole regime into a single cell.
    """
    return (pair.scene, pair.regime, pair.parallax_bin, pair.rotation_bin)


def subsample_by_stratum(
    pairs: Iterable[PairRecord],
    max_per_stratum: int,
    seed: int = 0,
) -> list[PairRecord]:
    """Take at most max_per_stratum pairs from each stratum, deterministically.

    Strata are unbalanced by construction: a camera program produces far more
    small-parallax pairs than large ones, so an unstratified sample would report
    mostly the easy end. Sampling within the stratum keeps every bin populated
    and keeps the per-bin estimates comparable.

    The order of the result follows the input, so a run is reproducible from the
    seed alone and does not depend on dictionary iteration order.
    """
    if max_per_stratum <= 0:
        raise ValueError("max_per_stratum must be positive")
    pairs = list(pairs)
    buckets: dict[tuple[str, ...], list[int]] = {}
    for index, pair in enumerate(pairs):
        buckets.setdefault(stratum_of(pair), []).append(index)
    keep: set[int] = set()
    for stratum in sorted(buckets):
        indices = buckets[stratum]
        if len(indices) <= max_per_stratum:
            keep.update(indices)
            continue
        # One generator per stratum, seeded by the stratum itself, so adding or
        # removing a scene cannot change which pairs another scene contributes.
        generator = torch.Generator().manual_seed(
            (hash_stratum(stratum) + seed) % (2**31)
        )
        order = torch.randperm(len(indices), generator=generator)[:max_per_stratum]
        keep.update(indices[int(i)] for i in order)
    return [pair for index, pair in enumerate(pairs) if index in keep]


def hash_stratum(stratum: tuple[str, ...]) -> int:
    """Stable integer for a stratum. Python's hash is salted per process."""
    import zlib

    return zlib.crc32("|".join(stratum).encode("utf-8"))


def summarize_pairs(pairs: Sequence[PairRecord]) -> dict[str, Any]:
    """Counts by split, regime, and parallax bin, for a run log."""
    by_regime: dict[str, int] = {regime: 0 for regime in REGIMES}
    by_parallax: dict[str, int] = {label: 0 for label in parallax_bin_order()}
    by_rotation: dict[str, int] = {label: 0 for label in rotation_bin_order()}
    by_split: dict[str, int] = {"train": 0, "test": 0}
    for pair in pairs:
        by_regime[pair.regime] += 1
        by_parallax[pair.parallax_bin] += 1
        by_rotation[pair.rotation_bin] += 1
        by_split[pair.split] += 1
    return {
        "total": len(pairs),
        "by_split": by_split,
        "by_regime": by_regime,
        "by_parallax_bin": by_parallax,
        "by_rotation_bin": by_rotation,
    }
