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
    # The landing-offset diagnostic, pre-registered 2026-10-09. The default is
    # the registered specification, so a configuration built in code measures
    # what the shipped one does; the shipped YAML states it explicitly.
    landing_offset: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "depth": "ground_truth",
            "upper_edges_patch": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
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


def landing_offset_edges(cfg: Phase5Config) -> tuple[float, ...]:
    """The pre-registered offset bin edges, in patch units, refused if malformed.

    validation/evidence/phase5/landing_offset_diagnostic.md registers the
    diagnostic. Its depth is ground truth. Its edges are positive, strictly
    increasing, below the cell-corner distance sqrt(2) / 2 so the open last bin
    can hold a landing, and exactly as many as the estimand layer reads.
    """
    import math

    from .phase5_estimands import N_OFFSET_BINS

    spec = dict(cfg.landing_offset)
    unknown = sorted(set(spec) - {"depth", "upper_edges_patch"})
    if unknown:
        raise ValueError(f"landing_offset: unknown keys {unknown}")
    if spec.get("depth") != "ground_truth":
        raise ValueError(
            f"landing_offset.depth is {spec.get('depth')!r}; the diagnostic is "
            "registered on ground_truth context depth, so that depth error cannot "
            "mix into what it measures"
        )
    edges = tuple(float(edge) for edge in spec.get("upper_edges_patch", ()))
    if len(edges) + 1 != N_OFFSET_BINS:
        raise ValueError(
            f"landing_offset: {len(edges)} edges make {len(edges) + 1} bins, but the "
            f"estimand layer reads {N_OFFSET_BINS} bins"
        )
    if any(upper <= lower for lower, upper in zip((0.0,) + edges, edges)):
        raise ValueError(
            f"landing_offset edges must be positive and strictly increasing: {edges}"
        )
    if edges[-1] >= math.sqrt(0.5):
        raise ValueError(
            f"landing_offset: the last edge {edges[-1]} is at or beyond the cell "
            "corner, sqrt(2) / 2, so the open last bin could never hold a landing"
        )
    return edges


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
        # The ratios array is the calibration population itself, float64 over
        # every calibration pixel, about 2 MB per frame. Phase 5 reads only the
        # fitted scale and affine terms, and the scene-scale level that needs
        # the ratios is not a Phase 5 condition. Holding it would cost roughly
        # 600 MB per resident scene for nothing; the fitted terms are kept
        # exactly, so no aligned depth changes.
        calibrations[frame.frame_id] = dataclasses.replace(
            calibration, ratios=np.zeros(0, dtype=np.float64)
        )

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

@dataclasses.dataclass(frozen=True)
class PairCameras:
    """One pair's frames and camera quantities, in the frozen geometry dtype."""

    context: Any
    target: Any
    K_context: Tensor
    K_target: Tensor
    T_target_from_context: Tensor
    context_hw: tuple[int, int]
    target_hw: tuple[int, int]


def pair_cameras(cfg: Phase5Config, inputs: SceneInputs, pair: Any) -> PairCameras:
    dtype = cfg.torch_dtype
    context = inputs.frames[pair.context_frame_id]
    target = inputs.frames[pair.target_frame_id]
    return PairCameras(
        context=context,
        target=target,
        K_context=context.K.to(dtype),
        K_target=target.K.to(dtype),
        T_target_from_context=relative_pose(
            target.T_world_from_camera, context.T_world_from_camera
        ).to(dtype),
        context_hw=(context.height, context.width),
        target_hw=(target.height, target.width),
    )


def context_valid_tokens(context_depth: np.ndarray) -> Tensor:
    """[N_ctx] bool, row major: the context patch holds at least one valid depth.

    Uses lot.visibility.fraction_per_patch, the one patch reduction the project
    defines, so the token order is the patch-grid order the feature cache uses.
    """
    from .visibility import fraction_per_patch

    valid = torch.from_numpy(np.isfinite(context_depth) & (context_depth > 0))
    return (fraction_per_patch(valid, PATCH_SIZE) > 0).reshape(-1)


def model_inputs(
    cfg: Phase5Config,
    inputs: SceneInputs,
    pair: Any,
    cams: PairCameras,
    context_depth: np.ndarray,
) -> dict[str, Tensor]:
    """Exactly what Predict-with-Depth receives, for training and for evaluation.

    One builder serves both, so the predictor at test time is handed precisely
    the kind of input it was trained on. Every field is context side or camera
    side; nothing here reads the target frame.
    """
    dtype = cfg.torch_dtype
    features_context = inputs.cache.features(cfg.feature_encoder, pair.context_frame_id)
    channels = features_context.shape[0]
    return {
        "features_context": features_context.to(dtype).reshape(channels, -1).T,
        "depth_context_aligned": torch.from_numpy(context_depth).to(dtype),
        "camera": camera_vector(
            cams.T_target_from_context[None], cams.K_context[None],
            cams.K_target[None], cams.context_hw, cams.target_hw,
        )[0],
        "context_valid": context_valid_tokens(context_depth),
    }


@dataclasses.dataclass(frozen=True)
class ExamplePlan:
    """The expensive, deterministic part of an example, computed once.

    The primary support depends only on the aligned context depth, ground
    truth, and the cameras, all of which are fixed, so it is computed once per
    pair and reused for every pass. Rebuilding an example from its plan then
    needs no visibility computation over the full image, which is the dominant
    per-example cost. support is [N_ctx] bool over the context patch centres.
    """

    scene: str
    pair: Any
    level: str
    support: np.ndarray

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.scene, self.pair.context_frame_id, self.pair.target_frame_id)


def plan_example(
    cfg: Phase5Config,
    analysis: AnalysisConfig,
    inputs: SceneInputs,
    pair: Any,
    level: str,
) -> ExamplePlan | None:
    """The primary support for one pair, or None when there is nothing to score.

    None means either the level has no arm for this pair (an affine fit that
    failed) or nothing survives the support rules. The caller counts both.
    """
    context_depth = aligned_context_depth(inputs, pair.context_frame_id, level)
    if context_depth is None:
        return None
    dtype = cfg.torch_dtype
    cams = pair_cameras(cfg, inputs, pair)
    lift = context_lift_map(
        torch.from_numpy(context_depth).to(dtype),
        cams.K_context, cams.K_target, cams.T_target_from_context,
        cams.context_hw, cams.target_hw,
    )
    evaluable = context_lift_support(
        lift,
        inputs.cache.depth(cams.context.depth_path).to(dtype),
        inputs.cache.depth(cams.target.depth_path).to(dtype),
        cams.K_context, cams.K_target, cams.T_target_from_context,
        rel_tol=analysis.covisible_relative_depth_tol,
    )
    support = primary_support(lift, evaluable)
    if not bool(support.any()):
        return None
    return ExamplePlan(inputs.scene, pair, level, support.cpu().numpy().copy())


def materialize_example(
    cfg: Phase5Config,
    inputs: SceneInputs,
    plan: ExamplePlan,
    center: Tensor,
) -> TrainingExample:
    """Rebuild a training example from its plan. Cheap and exact.

    The forward map is recomputed, which is a few thousand points, and the
    cached support selects the supervised landings from it. The result is
    identical to what plan_example's support would give if recomputed, because
    the support is a deterministic function of fixed inputs.
    """
    from .encoders import pixel_to_patch_coords, sample_features_bilinear

    dtype = cfg.torch_dtype
    pair = plan.pair
    context_depth = aligned_context_depth(inputs, pair.context_frame_id, plan.level)
    cams = pair_cameras(cfg, inputs, pair)
    lift = context_lift_map(
        torch.from_numpy(context_depth).to(dtype),
        cams.K_context, cams.K_target, cams.T_target_from_context,
        cams.context_hw, cams.target_hw,
    )
    chosen = torch.nonzero(torch.from_numpy(plan.support), as_tuple=False).reshape(-1)
    features_target = inputs.cache.features(cfg.feature_encoder, pair.target_frame_id)
    target_features = sample_features_bilinear(
        features_target.to(dtype), lift.uv_target[chosen], PATCH_SIZE
    )
    return TrainingExample(
        scene=inputs.scene,
        context_frame_id=pair.context_frame_id,
        target_frame_id=pair.target_frame_id,
        regime=pair.regime,
        **model_inputs(cfg, inputs, pair, cams, context_depth),
        query_patch_coords=pixel_to_patch_coords(lift.uv_target[chosen], PATCH_SIZE),
        target_centered=target_features - center.to(dtype),
        support=torch.ones(chosen.numel(), dtype=torch.bool),
    )


def build_example(
    cfg: Phase5Config,
    analysis: AnalysisConfig,
    inputs: SceneInputs,
    pair: Any,
    level: str,
    center: Tensor,
) -> TrainingExample | None:
    """One pair reduced to permitted model inputs plus its supervision support.

    Returns None when the pair has no usable arm at this level. A dropped pair
    is counted by the caller and never silently absorbed.
    """
    plan = plan_example(cfg, analysis, inputs, pair, level)
    return None if plan is None else materialize_example(cfg, inputs, plan, center)


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
    parser.add_argument(
        "--level", default=None,
        help="alignment level; defaults to the frozen primary level",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--scene-index", type=int, default=None,
        help="evaluate one test scene, by its index in the fixed evaluation order",
    )
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

    run_mode(cfg, analysis, args)


def scene_image_hw(cfg: Phase5Config, scene: str) -> tuple[int, int]:
    """The rendered frame size of one scene, which fixes the predictor's grid."""
    manifest = load_manifest(Path(cfg.renders_root) / scene / MANIFEST_NAME)
    sizes = {(f.height, f.width) for f in manifest.frames}
    if len(sizes) != 1:
        raise ValueError(f"{scene}: frames have several sizes {sorted(sizes)}")
    return next(iter(sizes))


def require_receipts(cfg: Phase5Config, config_path: Path, mode: str) -> None:
    """The authoritative gate check. Every launcher and worker passes through here.

    The shell launcher and the SLURM template also check, for an early and
    friendly refusal, but this is the boundary that cannot be bypassed: any way
    of running a mode is a way of calling this entry point. overfit needs the
    integration gate; everything after it needs both gates.
    """
    from .phase5_receipt import KIND_INTEGRATION, KIND_OVERFIT, verify

    gate = cfg.evidence_dir / "integration_gate.json"
    problems = verify(gate, config_path, "the Borah integration gate",
                      kind=KIND_INTEGRATION)
    if mode != "overfit":
        problems += verify(
            cfg.evidence_dir / "tiny_overfit.json", config_path,
            "the tiny-subset overfit gate", kind=KIND_OVERFIT, gate_receipt=gate,
        )
    if problems:
        import sys

        for problem in problems:
            print(problem, file=sys.stderr)
        raise SystemExit(
            f"mode {mode!r} is not permitted: its prerequisite gates do not stand "
            "for this commit, configuration, and set of inputs"
        )


def run_mode(cfg: Phase5Config, analysis: AnalysisConfig, args: Any) -> None:
    """overfit, train, controls, and evaluate, each behind its gates."""
    from .phase5_check import write_once
    from .phase5_modes import (
        SceneStore,
        evaluation_scenes,
        load_checkpoint,
        primary_evaluation_complete,
        resolve_device,
        run_controls,
        run_evaluate_scene,
        run_overfit,
        run_train_task,
    )
    from .phase5_receipt import KIND_OVERFIT, current_identity, stamp_receipt
    from .train import training_config_from

    mode = args.mode
    level = args.level or cfg.primary_alignment_level
    declared = (
        cfg.primary_alignment_level,
        *cfg.sensitivity_alignment_levels,
        *cfg.diagnostic_alignment_levels,
    )
    if level not in declared:
        raise SystemExit(f"level {level!r} is not declared in the frozen config: {declared}")

    require_receipts(cfg, args.config, mode)

    folds = frozen_folds()
    if level != cfg.primary_alignment_level:
        if mode == "overfit":
            raise SystemExit(
                "the overfit gate establishes that the trunk can represent the "
                "mapping and runs once, at the primary level"
            )
        missing = primary_evaluation_complete(
            cfg.run_dir, cfg.primary_alignment_level, evaluation_scenes(folds)
        )
        if missing:
            raise SystemExit(
                f"level {level!r} is a sensitivity or diagnostic condition and runs "
                "after the primary result is complete; primary evaluation is "
                f"missing for {missing}"
            )

    train_cfg = training_config_from(cfg.training)
    convention = load_convention_record(cfg)
    center = phase5_mean_vector(cfg)
    device = resolve_device(args.device)
    model_cfg = predictor_config_from(cfg, scene_image_hw(cfg, folds[0].train[0]))
    gate_receipt = cfg.evidence_dir / "integration_gate.json"

    with SceneStore(cfg, analysis, convention) as store:
        if mode == "overfit":
            tiny = cfg.tiny_overfit
            result = run_overfit(
                cfg, analysis, store, folds[0], model_cfg, train_cfg, center, device,
                level, int(tiny["n_pairs"]), tuple(tiny["regimes"]),
                float(tiny["threshold_centered_cosine"]), int(tiny["max_steps"]),
                int(tiny["seed"]),
            )
            stamped = stamp_receipt(result, args.config, KIND_OVERFIT,
                                    gate_receipt=gate_receipt)
            outcome = write_once(
                cfg.evidence_dir / "tiny_overfit.json",
                json.dumps(stamped, indent=2, sort_keys=True, default=str),
            )
            verdict = "PASS" if result["passed"] else "FAIL"
            print(f"TINY-SUBSET OVERFIT GATE: {verdict}")
            print(f"  reached centered cosine {result['reached_centered_cosine']:.4f} "
                  f"against {result['threshold']} in {result['steps']} steps")
            for item in result["subset"]:
                print(f"  {item['regime']:<12} {item['scene']} "
                      f"{item['context_frame_id']} -> {item['target_frame_id']}")
            print(f"receipt written to {outcome['written']}")
            if not result["passed"]:
                print("Phase 5 stops here: a predictor that cannot fit the frozen "
                      "subset cannot support a scientific reading of its "
                      "underperformance.")
            raise SystemExit(0 if result["passed"] else 1)

        if mode == "train":
            fold, seed = fold_and_seed_for_task(args.task_index, train_cfg.seeds)
            payload = run_train_task(
                cfg, analysis, store, fold, seed, model_cfg, train_cfg, center,
                device, level, cfg.run_dir,
            )
            print(f"fold {fold.index} seed {seed} level {level}: best validation "
                  f"centered cosine {payload['best_validation_centered_cosine']:.4f} "
                  f"at step {payload['best_step']} of {payload['steps_run']}")
            print(f"checkpoint {payload['checkpoint']}")
            return

        if mode == "controls":
            controls = cfg.controls
            for key in ("pose_shuffle", "depth_shuffle"):
                if controls.get(key) is not True:
                    raise SystemExit(
                        f"controls.{key} is {controls.get(key)!r}; both shuffles are "
                        "run together and the config may not disable one"
                    )
            results: dict[str, Any] = {}
            missing: list[str] = []
            for fold in folds:
                for seed in train_cfg.seeds:
                    try:
                        model = load_checkpoint(cfg.run_dir, level, fold, seed,
                                                model_cfg, train_cfg, device)
                    except FileNotFoundError as error:
                        missing.append(str(error))
                        continue
                    results[f"fold{fold.index}_seed{seed}"] = run_controls(
                        cfg, analysis, store, fold, model, model_cfg, center, level,
                        train_cfg.batch_pairs, int(controls["seed"]),
                    )
                # A fold's validation scenes are not read again by later folds.
                store.close()
            payload = {
                "level": level, "results": results, "missing": missing,
                **current_identity(args.config),
            }
            outcome = write_once(
                cfg.evidence_dir / f"input_use_controls_{level}.json",
                json.dumps(payload, indent=2, sort_keys=True, default=str),
            )
            for name, result in results.items():
                pose, depth = result["pose_shuffle"], result["depth_shuffle"]
                print(f"{name}: pose {pose['degradation']:+.4f} "
                      f"({pose['n_unchanged']} of {result['n_pairs']} unchanged)  "
                      f"depth {depth['degradation']:+.4f} "
                      f"({depth['n_unchanged']} of {result['n_pairs']} unchanged)")
            print(f"written to {outcome['written']}")
            if missing:
                print(f"{len(missing)} checkpoint(s) missing; controls are incomplete")
            raise SystemExit(1 if missing else 0)

        if mode == "evaluate":
            scenes = evaluation_scenes(folds)
            if args.scene_index is not None:
                if not 0 <= args.scene_index < len(scenes):
                    raise SystemExit(
                        f"scene index {args.scene_index} outside 0..{len(scenes) - 1}"
                    )
                scenes = [scenes[args.scene_index]]
            for scene in scenes:
                result = run_evaluate_scene(
                    cfg, analysis, store, scene, fold_of_test_scene(scene, folds),
                    train_cfg.seeds, model_cfg, train_cfg, center, level, device,
                    cfg.run_dir,
                )
                # One test scene resident at a time.
                store.close()
                print(f"{scene}: {result['status']} {result['path']}")
            return

    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
