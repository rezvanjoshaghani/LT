"""Context-target pairs, regime tags, scene splits, and sampling strata.

A pair is one context frame and one target frame of the same scene, the same
base viewpoint, and the same camera program. Pairs never cross viewpoints:
viewpoints are at least 0.75 m apart and generally look at different parts of
the scene, so a cross-viewpoint pair would mostly measure how little the two
views share rather than how well a representation transports.

Pairs are ordered. Transport is a directional operation, and the disoccluded
region of a pair depends on which frame is the context, so context A to target B
and context B to target A are two different measurements.

Two things this module deliberately does not do.

It does not compute the reported parallax. PROTOCOL 3.2 defines that as the
median of baseline over ground-truth depth across the pair's co-visible point
set, which is only known once visibility has been computed, so evaluate.py
produces it. What lives here is a cheap whole-frame proxy read from the
frame-stats sidecar, used solely to spread the sample across strata. It never
reaches a row, and no analysis bins on it.

It does not put bin labels in rows. PROTOCOL 3.2: "Bin labels never appear in
rows. Bin edges live only in a committed analysis config applied by the figures
code." Binning here exists only to choose which pairs to evaluate, and it reads
the same committed config the figures do.
"""

from __future__ import annotations

import dataclasses
import math
import zlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .analysis_config import AnalysisConfig, load_analysis_config
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

ZERO_BIN = "zero"

# The columns a scored record carries about its pair. Continuous geometry only:
# no bin label, and no parallax, which evaluate.py computes on the co-visible
# set and attaches there.
ROW_FIELDS = (
    "scene",
    "split",
    "viewpoint",
    "regime",
    "context_frame_id",
    "target_frame_id",
    "baseline_m",
    "context_median_depth_m",
    "rotation_deg",
)


def bin_label(value: float, edges: Sequence[float], zero_tol: float, name: str) -> str:
    """Label the bin a non-negative value falls in.

    Intervals are closed on the right, so a value equal to an edge belongs to
    the lower bin, per the analysis config's bin_right_closed. Exact zero gets
    its own label: no baseline at all, or no rotation at all, is a different
    physical situation from a small one.
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


def bin_order(edges: Sequence[float]) -> list[str]:
    """Bin labels in increasing order, for stable figures and tables."""
    labels = [ZERO_BIN]
    lower = 0.0
    for edge in edges:
        labels.append(f"{lower:g}-{edge:g}" if math.isfinite(edge) else f"{lower:g}+")
        lower = edge
    return labels


def parallax_bin(value: float, config: AnalysisConfig) -> str:
    """Label the parallax bin a value falls in, using the committed edges."""
    return bin_label(value, config.parallax_edges(), config.zero_parallax_tol, "parallax")


def parallax_bin_order(config: AnalysisConfig) -> list[str]:
    return bin_order(config.parallax_edges())


def rotation_bin(degrees: float, config: AnalysisConfig) -> str:
    """Label the rotation bin an angle falls in, using the committed edges."""
    return bin_label(degrees, config.rotation_edges(), config.zero_rotation_tol_deg, "rotation")


def rotation_bin_order(config: AnalysisConfig) -> list[str]:
    return bin_order(config.rotation_edges())


def scene_split(scene: str) -> str:
    """Which split a scene belongs to. Raises ValueError for an unknown scene."""
    if scene in REPLICA_SCENES_TRAIN:
        return "train"
    if scene in REPLICA_SCENES_TEST:
        return "test"
    raise ValueError(f"unknown scene {scene!r}")


@dataclasses.dataclass(frozen=True)
class PairRecord:
    """One directed context-target pair.

    stratum_parallax is the whole-frame proxy used only to spread the sample. It
    is not the protocol's parallax and never reaches a row.
    """

    scene: str
    split: str
    viewpoint: int
    regime: str
    context_frame_id: str
    target_frame_id: str
    baseline_m: float
    context_median_depth_m: float
    rotation_deg: float
    stratum_parallax: float

    def as_row(self) -> dict[str, Any]:
        """The pair columns of a scored record. Continuous geometry, no labels."""
        return {field: getattr(self, field) for field in ROW_FIELDS}


def _viewpoint_of(frame: FrameRecord) -> int:
    viewpoint = frame.params.get("viewpoint")
    if viewpoint is None:
        raise ValueError(f"{frame.frame_id}: manifest params carry no viewpoint")
    return int(viewpoint)


def pair_quantities(
    context: FrameRecord, target: FrameRecord, context_median_depth_m: float
) -> tuple[float, float, float]:
    """Baseline, proxy parallax, and rotation magnitude for one directed pair.

    The relative transform comes from geometry.relative_pose, the one place the
    formula is written.
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
    config: AnalysisConfig,
    regimes: Sequence[str] = REGIMES,
) -> list[PairRecord]:
    """Every directed pair of usable frames sharing a scene, viewpoint, and regime."""
    usable = usable_frame_ids(
        stats,
        min_valid_fraction=config.frame_min_valid_fraction,
        min_clearance_m=config.frame_min_clearance_m,
    )
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
                continue
            for target in frames:
                if target.frame_id == context.frame_id:
                    continue
                baseline, proxy, rotation = pair_quantities(context, target, float(median))
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
                        rotation_deg=rotation,
                        stratum_parallax=proxy,
                    )
                )
    return pairs


def load_scene_pairs(
    renders_root: Path, scene: str, config: AnalysisConfig | None = None, **kwargs: Any
) -> list[PairRecord]:
    """Build one scene's pairs from its manifest and frame-stats sidecar on disk."""
    config = config if config is not None else load_analysis_config()
    scene_root = Path(renders_root) / scene
    manifest = load_manifest(scene_root / MANIFEST_NAME)
    stats = load_frame_stats(scene_root / FRAME_STATS_NAME)
    return build_scene_pairs(manifest, stats, config, **kwargs)


def stratum_of(pair: PairRecord, config: AnalysisConfig) -> tuple[str, str, str, str]:
    """The stratum a pair is sampled within.

    Scene, regime, and both axes of viewpoint change. Binning on one axis alone
    would pool pairs that differ on the other: rotation pairs all share a
    parallax of zero, and translation pairs all share a rotation of zero, so
    either axis by itself collapses a whole regime into one cell.
    """
    return (
        pair.scene,
        pair.regime,
        parallax_bin(pair.stratum_parallax, config),
        rotation_bin(pair.rotation_deg, config),
    )


def subsample_by_stratum(
    pairs: Iterable[PairRecord],
    max_per_stratum: int,
    seed: int = 0,
    config: AnalysisConfig | None = None,
) -> list[PairRecord]:
    """Take at most max_per_stratum pairs from each stratum, deterministically.

    Strata are unbalanced by construction: a camera program produces far more
    small-parallax pairs than large ones, so an unstratified sample would report
    mostly the easy end.

    Each stratum is seeded by its own identity, so adding or removing a scene
    cannot change which pairs another scene contributes.
    """
    if max_per_stratum <= 0:
        raise ValueError("max_per_stratum must be positive")
    config = config if config is not None else load_analysis_config()
    pairs = list(pairs)
    buckets: dict[tuple[str, ...], list[int]] = {}
    for index, pair in enumerate(pairs):
        buckets.setdefault(stratum_of(pair, config), []).append(index)
    keep: set[int] = set()
    for stratum in sorted(buckets):
        indices = buckets[stratum]
        if len(indices) <= max_per_stratum:
            keep.update(indices)
            continue
        generator = torch.Generator().manual_seed((hash_stratum(stratum) + seed) % (2**31))
        order = torch.randperm(len(indices), generator=generator)[:max_per_stratum]
        keep.update(indices[int(i)] for i in order)
    return [pair for index, pair in enumerate(pairs) if index in keep]


def hash_stratum(stratum: tuple[str, ...]) -> int:
    """Stable integer for a stratum. Python's hash is salted per process."""
    return zlib.crc32("|".join(stratum).encode("utf-8"))


def assert_translation_parallax_floor(
    pairs: Sequence[PairRecord], config: AnalysisConfig
) -> None:
    """PROTOCOL 3.4: (0, 0.025) is empty for translation-program pairs.

    The translation program sizes its moves from a design floor, so no
    translation pair should land between exact zero and the first edge. The
    assertion is enforced rather than silently absorbed by a bin, because a bin
    that quietly accepts them would hide a program that stopped honouring its
    own floor. Orbit pairs may legitimately fall there and are not checked.
    """
    if not config.assert_translation_parallax_floor:
        return
    first_edge = config.parallax_edges()[0]
    offenders = [
        pair
        for pair in pairs
        if pair.regime == "translation"
        and config.zero_parallax_tol <= pair.stratum_parallax < first_edge
    ]
    if offenders:
        example = offenders[0]
        raise ValueError(
            f"{len(offenders)} translation pairs fall in the open interval "
            f"(0, {first_edge:g}) that PROTOCOL 3.4 asserts empty; first is "
            f"{example.context_frame_id} to {example.target_frame_id} at "
            f"parallax {example.stratum_parallax:g}"
        )


def summarize_pairs(pairs: Sequence[PairRecord], config: AnalysisConfig) -> dict[str, Any]:
    """Counts by split, regime, and both binning axes, for a run log."""
    by_regime: dict[str, int] = {regime: 0 for regime in REGIMES}
    by_parallax: dict[str, int] = {label: 0 for label in parallax_bin_order(config)}
    by_rotation: dict[str, int] = {label: 0 for label in rotation_bin_order(config)}
    by_split: dict[str, int] = {"train": 0, "test": 0}
    for pair in pairs:
        by_regime[pair.regime] += 1
        by_parallax[parallax_bin(pair.stratum_parallax, config)] += 1
        by_rotation[rotation_bin(pair.rotation_deg, config)] += 1
        by_split[pair.split] += 1
    return {
        "total": len(pairs),
        "by_split": by_split,
        "by_regime": by_regime,
        "by_stratum_parallax_bin": by_parallax,
        "by_rotation_bin": by_rotation,
    }
