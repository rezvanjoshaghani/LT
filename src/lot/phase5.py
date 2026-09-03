"""Phase 5, rung 2: the learned-versus-explicit transformation limitation.

The orchestration layer. It reads the caches and the accepted Phase 4 artifacts,
assembles what the model may see, drives training and evaluation, and writes the
per-pair records the estimand layer turns into reported quantities. The science
lives in the modules it calls:

    lot.context_lift        the information-symmetric explicit comparator
    lot.predictors          the learned comparator
    lot.train               training, the overfit gate, the input-use controls
    lot.phase5_score        the three fixed supports and the scoring
    lot.phase5_estimands    the reported quantities and their paired intervals
    lot.phase5_folds        the frozen scene assignment

Aligned context depth is not recomputed by a second implementation here. This
module calls Phase 4's own `frame_calibration` and `aligned_depth` through the
same per-frame sequence Phase 4 runs, so the depth both headline methods consume
is identical to Phase 4's by construction rather than by comparison. That is
what lets Stream Q's "exactly the same aligned context depth" be a structural
claim instead of a hope.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

from .analysis_config import DEFAULT_CONFIG_PATH, AnalysisConfig, load_analysis_config
from .context_lift import context_lift_map, context_lift_support
from .datasets import load_scene_pairs, scene_split, subsample_by_stratum
from .encoders import PATCH_SIZE, load_cache_meta, patch_grid_shape
from .evaluate import _SceneCache, git_commit, load_or_build_mean_vector
from .geometry import relative_pose
from .phase4 import (
    Phase4GateError,
    FrameCalibration,
    aligned_depth,
    frame_calibration,
    load_depth_archive,
    resample_depth_nearest,
    run_convention,
    secant_map,
    transport_prevalid,
)
from .phase5_folds import Fold, fold_of_test_scene, frozen_folds
from .phase5_score import primary_support, score_primary
from .predictors import PredictorConfig, camera_vector
from .render_replica import MANIFEST_NAME, REPLICA_SCENES, load_manifest
from .train import TrainingExample

PHASE5_VERSION = 1


@dataclasses.dataclass
class Phase5Config:
    """The frozen Phase 5 configuration, read from configs/phase5.yaml."""

    experiment_name: str = "phase5_rung2"
    renders_root: str = "data/replica_renders"
    cache_root: str = "cache/features"
    output_root: str = "outputs"
    feature_encoder: str = "dinov2_vitb14"
    depth_encoder: str = "vggt_1b"
    mean_vector_dir: str = "outputs/experiment_zero"
    phase4_dir: str = "outputs/phase4_rung1"
    analysis_config: str = str(DEFAULT_CONFIG_PATH)
    primary_alignment_level: str = "image"
    sensitivity_alignment_levels: tuple[str, ...] = ("affine",)
    diagnostic_alignment_levels: tuple[str, ...] = ("none",)
    model: dict[str, Any] = dataclasses.field(default_factory=dict)
    training: dict[str, Any] = dataclasses.field(default_factory=dict)
    tiny_overfit: dict[str, Any] = dataclasses.field(default_factory=dict)
    controls: dict[str, Any] = dataclasses.field(default_factory=dict)
    seed: int = 0

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.experiment_name

    @property
    def evidence_dir(self) -> Path:
        return self.run_dir / "evidence"

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float32

    # The only fields that may sit outside the configuration identity. Both
    # decide where results are written and nothing about what is measured, so a
    # run relocated to another directory is the same experiment. Every other
    # field is inside the digest by default: the list is an allowlist of
    # exclusions rather than an allowlist of inclusions, so a field added to
    # this config later is covered without anyone remembering to add it.
    RELOCATION_FIELDS = ("output_root", "experiment_name")

    def digest(self) -> str:
        """Content identity of everything that decides what a Phase 5 run measures.

        This is what a gate receipt is bound to, so its coverage is a
        correctness property rather than a convenience. An earlier version
        covered only the architecture, the training settings, the overfit gate,
        and the primary level, which left the encoders, every input path, the
        pair-subsampling seed, the sensitivity levels, and the controls outside
        it. A same-commit configuration pointing at a different feature cache or
        a different accepted Phase 4 run would then have inherited a PASS
        receipt from a gate that never examined its inputs.
        """
        import hashlib

        payload = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in self.RELOCATION_FIELDS
        }
        payload["version"] = PHASE5_VERSION
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
        ).hexdigest()


def load_phase5_config(path: Path) -> Phase5Config:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    allowed = {f.name for f in dataclasses.fields(Phase5Config)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    for key in ("sensitivity_alignment_levels", "diagnostic_alignment_levels"):
        if key in raw and raw[key] is not None:
            raw[key] = tuple(raw[key])
    return Phase5Config(**raw)


def predictor_config_from(cfg: Phase5Config, image_hw: tuple[int, int]) -> PredictorConfig:
    """Build the frozen architecture from the config and the cache's grid shape."""
    grid = patch_grid_shape(image_hw, PATCH_SIZE)
    model = cfg.model
    return PredictorConfig(
        d_model=int(model.get("d_model", 384)),
        n_blocks=int(model.get("n_blocks", 6)),
        n_heads=int(model.get("n_heads", 6)),
        ffn_dim=int(model.get("ffn_dim", 1536)),
        dropout=float(model.get("dropout", 0.1)),
        feature_dim=int(model.get("feature_dim", 768)),
        patch_size=PATCH_SIZE,
        context_grid=grid,
        target_grid=grid,
    )


def phase5_mean_vector(cfg: Phase5Config) -> Tensor:
    """The frozen Phase 3 centering statistic, reused rather than rebuilt.

    The scene list is Phase 3's own train split, not Phase 5's folds. That is
    deliberate and load bearing: the stored vector carries a provenance record
    naming the scenes it was built from, and the loader refuses a mismatch, so
    passing the Phase 5 folds here would either fail loudly or, in a fresh
    directory, silently build a different centering statistic and move both the
    Mean-Feature floor and every centered score away from Phase 3 and Phase 4.
    """
    train = [s for s in REPLICA_SCENES if scene_split(s) == "train"]
    return load_or_build_mean_vector(
        Path(cfg.cache_root), cfg.feature_encoder, train, Path(cfg.mean_vector_dir)
    )


def phase5_scene_pairs(
    cfg: Phase5Config, analysis: AnalysisConfig, scene: str
) -> list[Any]:
    """One scene's sampled pairs, drawn exactly as Phase 3 and Phase 4 draw them.

    Defined once and used by both the evaluation path and the integration gate,
    so the two cannot drift onto different populations.
    """
    return subsample_by_stratum(
        load_scene_pairs(cfg.renders_root, scene, config=analysis),
        analysis.max_pairs_per_stratum,
        seed=cfg.seed,
        config=analysis,
    )


# ---------------------------------------------------------------------------
# Per-scene inputs, assembled through Phase 4's own alignment code
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SceneInputs:
    """One scene's frames, aligned context depth, and cached features."""

    scene: str
    manifest: Any
    cache: _SceneCache
    frames: dict[str, Any]
    est_maps: dict[str, np.ndarray]
    calibrations: dict[str, FrameCalibration]
    convention: str

    def close(self) -> None:
        self.cache.close()


def load_convention_record(cfg: Phase5Config) -> dict[str, Any]:
    """The Phase 4 convention record. One checkpoint, one convention (A6).

    Read from the accepted Phase 4 run rather than recomputed, so Phase 5 cannot
    disagree with Phase 4 about what the depth cache means.
    """
    path = Path(cfg.phase4_dir) / "evidence" / "convention_record.json"
    if not path.exists():
        raise Phase4GateError(
            f"the accepted Phase 4 convention record is missing at {path}; "
            "Phase 5 reads the convention rather than re-deciding it"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_frame_depth(
    raw_depth: np.ndarray,
    raw_conf: np.ndarray | None,
    frame: Any,
    gt_depth: np.ndarray,
    verdict: str,
    analysis: AnalysisConfig,
) -> tuple[np.ndarray, FrameCalibration]:
    """One frame's aligned estimated depth and its context-image calibration.

    This is the per-frame sequence lot.phase4.evaluate_scene_phase4 runs inline:
    resample to the rendered frame size, convert only if the run's single
    convention says ray distance, apply the frozen 5a validity rule to the map
    itself so invalid pixels become NaN for every downstream consumer, then
    calibrate from the context image alone.

    Phase 4 is accepted and pinned, so it cannot be refactored to call this;
    the two therefore have to be held together by evidence rather than by
    sharing code. tests/test_phase5_depth_equivalence.py replays Phase 4's
    inline sequence from its own source and asserts this function reproduces it
    bit for bit, which is the same discipline lot.paired_bootstrap uses to share
    Phase 4's bootstrap. Without that test the docstring claim that both
    headline methods consume Phase 4's depth "by construction" would rest on
    two copies nobody compares.
    """
    resampled, _ = resample_depth_nearest(raw_depth, (frame.height, frame.width))
    if verdict == "ray_distance":
        resampled = (
            resampled / secant_map(frame.K, frame.height, frame.width)
        ).astype(np.float32)
    conf = (
        resample_depth_nearest(raw_conf, (frame.height, frame.width))[0]
        if raw_conf is not None
        else None
    )
    prevalid = transport_prevalid(resampled, conf, analysis)
    aligned = np.where(prevalid, resampled, np.float32(np.nan)).astype(np.float32)
    return aligned, frame_calibration(aligned, gt_depth, prevalid)


def build_scene_inputs(
    cfg: Phase5Config, analysis: AnalysisConfig, scene: str, convention: dict[str, Any]
) -> SceneInputs:
    """Mirror Phase 4's per-frame depth preparation, step for step.

    The per-frame work lives in prepare_frame_depth, which an equivalence test
    pins against Phase 4's inline sequence. Any divergence would mean the two
    headline methods no longer share their geometry, which is the condition
    Stream AD makes a stop.
    """
    scene_root = Path(cfg.renders_root) / scene
    manifest = load_manifest(scene_root / MANIFEST_NAME)
    feature_meta = load_cache_meta(cfg.cache_root, cfg.feature_encoder, scene)
    if not feature_meta.get("features_digest"):
        raise Phase4GateError(f"{scene}: the feature cache carries no digest")

    depth_cache = load_depth_archive(cfg.cache_root, cfg.depth_encoder, scene)
    verdict = run_convention(convention, depth_cache["meta"])

    cache = _SceneCache(scene_root, cfg.cache_root, [cfg.feature_encoder], scene, manifest)
    est_maps: dict[str, np.ndarray] = {}
    calibrations: dict[str, FrameCalibration] = {}
    for frame in manifest.frames:
        aligned, calibration = prepare_frame_depth(
            depth_cache["depth"][frame.frame_id],
            depth_cache["conf"][frame.frame_id],
            frame,
            cache.depth(frame.depth_path).numpy(),
            verdict,
            analysis,
        )
        est_maps[frame.frame_id] = aligned
        calibrations[frame.frame_id] = calibration

    return SceneInputs(
        scene=scene,
        manifest=manifest,
        cache=cache,
        frames={f.frame_id: f for f in manifest.frames},
        est_maps=est_maps,
        calibrations=calibrations,
        convention=verdict,
    )


def aligned_context_depth(
    inputs: SceneInputs, context_frame_id: str, level: str
) -> np.ndarray | None:
    """The context frame's depth under one alignment level, or None if it failed.

    Level 2 and affine are estimated from the context image alone, so no scene
    scalar is needed and the leave-target-out estimator never enters. Native
    scale passes the map through. A failed affine fit returns None and the pair
    is reported as having no arm at that level, exactly as Phase 4 reports it.
    """
    if level == "scene":
        raise ValueError(
            "the scene-scale level is a Phase 4 diagnostic and is not a Phase 5 "
            "condition; Phase 5 uses context-image scale, affine, or native"
        )
    return aligned_depth(
        level, inputs.est_maps[context_frame_id], float("nan"),
        inputs.calibrations[context_frame_id],
    )


# ---------------------------------------------------------------------------
# Example assembly
# ---------------------------------------------------------------------------

def build_example(
    cfg: Phase5Config,
    analysis: AnalysisConfig,
    inputs: SceneInputs,
    pair: Any,
    level: str,
    center: Tensor,
) -> TrainingExample | None:
    """One pair reduced to permitted model inputs plus its supervision support.

    Returns None when the pair has no usable arm at this level, which happens
    when an affine fit failed or when nothing survives the support rules. A
    dropped pair is counted by the caller and never silently absorbed.
    """
    dtype = cfg.torch_dtype
    context = inputs.frames[pair.context_frame_id]
    target = inputs.frames[pair.target_frame_id]
    context_depth = aligned_context_depth(inputs, pair.context_frame_id, level)
    if context_depth is None:
        return None

    K_context = context.K.to(dtype)
    K_target = target.K.to(dtype)
    T_target_from_context = relative_pose(
        target.T_world_from_camera, context.T_world_from_camera
    ).to(dtype)
    context_hw = (context.height, context.width)
    target_hw = (target.height, target.width)

    lift = context_lift_map(
        torch.from_numpy(context_depth).to(dtype),
        K_context, K_target, T_target_from_context, context_hw, target_hw,
    )
    evaluable = context_lift_support(
        lift,
        inputs.cache.depth(context.depth_path).to(dtype),
        inputs.cache.depth(target.depth_path).to(dtype),
        K_context, K_target, T_target_from_context,
        rel_tol=analysis.covisible_relative_depth_tol,
    )
    support = primary_support(lift, evaluable)
    if not bool(support.any()):
        return None

    features_context = inputs.cache.features(cfg.feature_encoder, pair.context_frame_id)
    features_target = inputs.cache.features(cfg.feature_encoder, pair.target_frame_id)
    channels = features_context.shape[0]

    from .encoders import pixel_to_patch_coords, sample_features_bilinear

    chosen = torch.nonzero(support, as_tuple=False).reshape(-1)
    target_features = sample_features_bilinear(
        features_target.to(dtype), lift.uv_target[chosen], PATCH_SIZE
    )
    return TrainingExample(
        scene=inputs.scene,
        context_frame_id=pair.context_frame_id,
        target_frame_id=pair.target_frame_id,
        regime=pair.regime,
        features_context=features_context.to(dtype).reshape(channels, -1).T,
        depth_context_aligned=torch.from_numpy(context_depth).to(dtype),
        camera=camera_vector(
            T_target_from_context[None], K_context[None], K_target[None],
            context_hw, target_hw,
        )[0],
        context_valid=torch.from_numpy(
            (np.isfinite(context_depth) & (context_depth > 0))
            .reshape(
                patch_grid_shape(context_hw, PATCH_SIZE)[0], PATCH_SIZE,
                patch_grid_shape(context_hw, PATCH_SIZE)[1], PATCH_SIZE,
            )
            .any(axis=(1, 3))
            .reshape(-1)
        ),
        query_patch_coords=pixel_to_patch_coords(lift.uv_target[chosen], PATCH_SIZE),
        target_centered=target_features - center.to(dtype),
        support=torch.ones(chosen.numel(), dtype=torch.bool),
    )


def iter_examples(
    cfg: Phase5Config,
    analysis: AnalysisConfig,
    scenes: Sequence[str],
    level: str,
    center: Tensor,
    convention: dict[str, Any],
    regimes: Sequence[str] | None = None,
    limit_per_scene: int | None = None,
) -> Iterator[TrainingExample]:
    """Stream examples for a set of scenes, one scene's cache open at a time."""
    for scene in scenes:
        inputs = build_scene_inputs(cfg, analysis, scene, convention)
        try:
            pairs = phase5_scene_pairs(cfg, analysis, scene)
            if regimes is not None:
                pairs = [p for p in pairs if p.regime in set(regimes)]
            if limit_per_scene is not None:
                pairs = pairs[:limit_per_scene]
            for pair in pairs:
                example = build_example(cfg, analysis, inputs, pair, level, center)
                if example is not None:
                    yield example
        finally:
            inputs.close()


def fold_and_seed_for_task(task_index: int, seeds: Sequence[int]) -> tuple[Fold, int]:
    """Map a SLURM array index onto (fold, seed) using the frozen seed list.

    The mapping lives here rather than in the sbatch template so it moves with
    the frozen config, and so a run record can name the fold and seed it
    actually trained rather than an index a reader has to decode.
    """
    folds = frozen_folds()
    if not seeds:
        raise ValueError("the frozen seed list is empty")
    total = len(folds) * len(seeds)
    if not 0 <= task_index < total:
        raise ValueError(f"task index {task_index} outside 0..{total - 1}")
    return folds[task_index // len(seeds)], seeds[task_index % len(seeds)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 5, rung 2")
    parser.add_argument("--config", type=Path, default=Path("configs/phase5.yaml"))
    parser.add_argument(
        "--mode",
        choices=("describe", "check", "overfit", "train", "controls", "evaluate"),
        default="describe",
    )
    parser.add_argument("--task-index", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_phase5_config(args.config)
    analysis = load_analysis_config(Path(cfg.analysis_config))
    seeds = tuple(cfg.training.get("seeds", (0, 1, 2)))

    if args.mode == "describe":
        # Everything that can be reported without touching a cache, so the
        # frozen identities can be checked before any cluster time is spent.
        folds = frozen_folds()
        from .phase5_folds import fold_digest

        print(f"experiment          {cfg.experiment_name}")
        print(f"commit              {git_commit()}")
        print(f"config digest       {cfg.digest()}")
        print(f"fold digest         {fold_digest(folds)}")
        print(f"measurement digest  {analysis.measurement_digest()}")
        print(f"primary level       {cfg.primary_alignment_level}")
        print(f"seeds               {list(seeds)}")
        for fold in folds:
            print(
                f"fold {fold.index}: train {len(fold.train)} val {len(fold.val)} "
                f"test {len(fold.test)} -> {list(fold.test)}"
            )
        for index in range(len(folds) * len(seeds)):
            fold, seed = fold_and_seed_for_task(index, seeds)
            print(f"task {index}: fold {fold.index} seed {seed}")
        return

    if args.mode == "check":
        from .phase5_check import format_report
        from .phase5_gate import run_integration_gate

        from .phase5_check import write_once
        from .phase5_receipt import KIND_INTEGRATION, stamp_receipt

        report = run_integration_gate(cfg, analysis)
        destination = cfg.evidence_dir / "integration_gate.json"
        # Stamped through the one function that knows what a receipt must carry,
        # and written without destroying an earlier one.
        stamped = stamp_receipt(
            json.loads(report.to_json()), args.config, KIND_INTEGRATION
        )
        outcome = write_once(
            destination, json.dumps(stamped, indent=2, sort_keys=True, default=str)
        )
        print(format_report(report))
        print(f"\nevidence written to {destination}")
        if outcome["archived_previous"]:
            print(f"previous receipt kept at {outcome['archived_previous']}")
        raise SystemExit(0 if report.passed else 1)

    raise SystemExit(
        f"mode {args.mode!r} needs the caches and the accepted Phase 4 artifacts, "
        "which are cluster resident, and is only permitted after the integration "
        "gate passes. Run it from Borah via scripts/run_phase5.sh."
    )


if __name__ == "__main__":
    main()
