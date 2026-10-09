"""Phase 5 modes: overfit, train, controls, evaluate.

The orchestration behind `python -m lot.phase5 --mode ...`. Every function here
takes its dependencies explicitly: the configuration, the fold, the centering
vector, the model shape, the training settings, the device, and where outputs
go. The command line resolves the real ones from the frozen configuration; the
suite passes synthetic ones and drives every mode end to end, which is what the
four code reviews repeatedly found missing.

Receipts are not checked in this module. The command line checks them before
calling in, because it is the single entry point every launcher and every SLURM
worker passes through. A mode function that also checked would leave the suite
unable to exercise the modes without fabricating receipts.

Memory. A training run holds its training and validation scenes resident for
the whole run, about 1.2 GB each with the calibration ratios dropped, because
every pass reads all of them. Evaluation holds one test scene at a time.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

from .analysis_config import AnalysisConfig
from .context_lift import context_lift_map, context_lift_support
from .encoders import PATCH_SIZE, patch_grid_shape
from .evaluate import git_commit, vector_digest
from .phase5 import (
    ExamplePlan,
    Phase5Config,
    SceneInputs,
    aligned_context_depth,
    build_scene_inputs,
    landing_offset_edges,
    materialize_example,
    model_inputs,
    pair_cameras,
    phase5_scene_pairs,
    plan_example,
)
from .phase5_check import sha256_file, supersede, write_once
from .phase5_folds import Fold
from .phase5_reference import (
    ReferenceMismatch,
    read_phase4_reference,
    recompute_reference_arms,
    reconcile_reference,
    region_cells,
)
from .phase5_score import (
    FormulationScores,
    empty_landing_offset,
    primary_support,
    region_masks,
    score_cross_path,
    score_formulation,
    score_landing_offset,
    score_primary,
    score_splat_pool,
)
from .predictors import PredictWithDepth, PredictorConfig, build_predictor
from .train import (
    SHUFFLED_FIELDS,
    TrainingConfig,
    assert_scenes_in_role,
    derangement,
    evaluate_validation,
    run_tiny_overfit_gate,
    train_fold,
)

# Stream Z's regions. "all" is the whole primary support; the four splits are
# Phase 4's frozen ground-truth boundary and texture masks, read at the target
# cell each sample lands in.
REGIONS = ("all", "boundary", "interior", "low_texture", "high_texture")


# ---------------------------------------------------------------------------
# Scenes and example plans
# ---------------------------------------------------------------------------

class SceneStore:
    """Scene inputs built on first use and held until closed."""

    def __init__(self, cfg: Phase5Config, analysis: AnalysisConfig,
                 convention: dict[str, Any]):
        self.cfg = cfg
        self.analysis = analysis
        self.convention = convention
        self._scenes: dict[str, SceneInputs] = {}

    def get(self, scene: str) -> SceneInputs:
        if scene not in self._scenes:
            self._scenes[scene] = build_scene_inputs(
                self.cfg, self.analysis, scene, self.convention
            )
        return self._scenes[scene]

    def close(self) -> None:
        for inputs in self._scenes.values():
            inputs.close()
        self._scenes.clear()

    def __enter__(self) -> "SceneStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


@dataclasses.dataclass(frozen=True)
class PlanCensus:
    """What happened to every pair of one scene. Nothing is dropped silently."""

    scene: str
    level: str
    n_pairs: int
    n_planned: int
    n_no_arm: int            # the level has no arm for this pair: affine failed
    n_empty_support: int     # nothing survives the primary support rules


def plan_scene(
    cfg: Phase5Config,
    analysis: AnalysisConfig,
    inputs: SceneInputs,
    level: str,
) -> tuple[list[ExamplePlan], PlanCensus]:
    """Every usable pair of one scene, with an account of the unusable ones."""
    pairs = phase5_scene_pairs(cfg, analysis, inputs.scene)
    plans: list[ExamplePlan] = []
    no_arm = empty = 0
    for pair in pairs:
        if aligned_context_depth(inputs, pair.context_frame_id, level) is None:
            no_arm += 1
            continue
        plan = plan_example(cfg, analysis, inputs, pair, level)
        if plan is None:
            empty += 1
            continue
        plans.append(plan)
    return plans, PlanCensus(inputs.scene, level, len(pairs), len(plans), no_arm, empty)


def assert_grid_matches(inputs: SceneInputs, model_cfg: PredictorConfig) -> None:
    """The predictor's token grid must be the scene's patch grid, on both sides."""
    sizes = {(f.height, f.width) for f in inputs.frames.values()}
    if len(sizes) != 1:
        raise ValueError(f"{inputs.scene}: frames have several sizes {sorted(sizes)}")
    grid = patch_grid_shape(next(iter(sizes)), PATCH_SIZE)
    if grid != tuple(model_cfg.context_grid) or grid != tuple(model_cfg.target_grid):
        raise ValueError(
            f"{inputs.scene}: patch grid {grid} does not match the predictor's "
            f"context {model_cfg.context_grid} and target {model_cfg.target_grid}"
        )


# ---------------------------------------------------------------------------
# overfit: Stream U step 14
# ---------------------------------------------------------------------------

def select_tiny_subset(
    cfg: Phase5Config,
    analysis: AnalysisConfig,
    store: SceneStore,
    fold: Fold,
    level: str,
    n_pairs: int,
    regimes: Sequence[str],
    train_scenes: Sequence[str] | None = None,
) -> list[ExamplePlan]:
    """The frozen tiny subset: deterministic, training scenes only, every regime.

    Slot i asks for regime regimes[i % len(regimes)] from training scene
    fold.train[i % len(fold.train)], taking the first usable pair of that regime
    in the frozen pair order and moving to the next training scene in fold order
    when a scene has none. Nothing here reads an outcome. The choice depends on
    scene names, the frozen pair order, and whether a pair has any supported
    sample, which is a fact about geometry rather than about any method.
    """
    if not regimes:
        raise ValueError("the tiny subset needs at least one regime")
    train = list(fold.train if train_scenes is None else train_scenes)
    assert_scenes_in_role(fold, train, "train", "tiny subset selection")
    pairs_of: dict[str, list[Any]] = {}
    chosen: list[ExamplePlan] = []
    used: set[tuple[str, str, str]] = set()
    for slot in range(n_pairs):
        regime = regimes[slot % len(regimes)]
        plan = None
        for offset in range(len(train)):
            scene = train[(slot + offset) % len(train)]
            if scene not in pairs_of:
                pairs_of[scene] = phase5_scene_pairs(cfg, analysis, scene)
            inputs = store.get(scene)
            for pair in pairs_of[scene]:
                if pair.regime != regime:
                    continue
                if (scene, pair.context_frame_id, pair.target_frame_id) in used:
                    continue
                plan = plan_example(cfg, analysis, inputs, pair, level)
                if plan is not None:
                    break
            if plan is not None:
                break
        if plan is None:
            raise ValueError(
                f"no usable {regime} pair in any training scene of fold "
                f"{fold.index}; the frozen tiny subset cannot be assembled"
            )
        used.add(plan.key)
        chosen.append(plan)
    return chosen


def run_overfit(
    cfg: Phase5Config,
    analysis: AnalysisConfig,
    store: SceneStore,
    fold: Fold,
    model_cfg: PredictorConfig,
    train_cfg: TrainingConfig,
    center: Tensor,
    device: str,
    level: str,
    n_pairs: int,
    regimes: Sequence[str],
    threshold: float,
    max_steps: int,
    seed: int,
    train_scenes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit the frozen tiny subset and report the verdict with the subset named.

    Never raises on a failed fit: the caller writes the receipt either way, so a
    failure is recorded as evidence, and then stops. The subset is listed in the
    result so a reader can see exactly which pairs the verdict is about.
    """
    plans = select_tiny_subset(cfg, analysis, store, fold, level, n_pairs, regimes,
                               train_scenes)
    examples = [materialize_example(cfg, store.get(p.scene), p, center) for p in plans]
    for plan in plans:
        assert_grid_matches(store.get(plan.scene), model_cfg)
    result = run_tiny_overfit_gate(
        examples, model_cfg, train_cfg, model_cfg.target_grid, threshold, max_steps,
        fold=fold, expected_pairs=n_pairs, required_regimes=regimes, seed=seed,
        device=device, raise_on_failure=False,
    )
    return {
        "passed": bool(result.passed),
        "reached_centered_cosine": result.reached_centered_cosine,
        "threshold": result.threshold,
        "steps": result.steps,
        "n_pairs": result.n_pairs,
        "regimes": list(result.regimes),
        "fold": fold.index,
        "level": level,
        "seed": seed,
        "subset": [
            {
                "scene": plan.scene,
                "context_frame_id": plan.pair.context_frame_id,
                "target_frame_id": plan.pair.target_frame_id,
                "regime": plan.pair.regime,
                "n_supported": int(plan.support.sum()),
            }
            for plan in plans
        ],
    }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def checkpoint_path(run_dir: Path, level: str, fold_index: int, seed: int) -> Path:
    return Path(run_dir) / "checkpoints" / level / f"fold{fold_index}_seed{seed}.pt"


def training_record_path(run_dir: Path, level: str, fold_index: int, seed: int) -> Path:
    return checkpoint_path(run_dir, level, fold_index, seed).with_suffix(".json")


def load_checkpoint(
    run_dir: Path,
    level: str,
    fold: Fold,
    seed: int,
    model_cfg: PredictorConfig,
    train_cfg: TrainingConfig,
    device: str,
) -> PredictWithDepth:
    """Load the selected model of one (fold, seed), proven to be training's own.

    The checkpoint must be the file the finished training run recorded, by
    content: its sha256 is compared against the training record written after
    selection. A checkpoint without a record, or whose bytes differ from the
    recorded ones, belongs to no completed run and is refused, as is one trained
    for another fold or seed or under another training configuration.
    """
    path = checkpoint_path(run_dir, level, fold.index, seed)
    record_file = training_record_path(run_dir, level, fold.index, seed)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint for fold {fold.index} seed {seed} at {path}")
    if not record_file.exists():
        raise FileNotFoundError(
            f"{path} has no training record at {record_file}; the training run that "
            "produced it did not finish, so it is not a selected checkpoint"
        )
    record = json.loads(record_file.read_text(encoding="utf-8"))
    if record.get("checkpoint_sha256") != sha256_file(path):
        raise ValueError(
            f"{path} is not the checkpoint its training record names; its bytes "
            "changed after selection"
        )
    state = torch.load(path, map_location=device, weights_only=False)
    for key, want in (
        ("fold", fold.index),
        ("seed", seed),
        ("training_config_digest", train_cfg.digest()),
    ):
        if state.get(key) != want:
            raise ValueError(f"{path}: {key} is {state.get(key)!r}, expected {want!r}")
    model = build_predictor(model_cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# train: Stream U steps 15 and 16
# ---------------------------------------------------------------------------

def run_train_task(
    cfg: Phase5Config,
    analysis: AnalysisConfig,
    store: SceneStore,
    fold: Fold,
    seed: int,
    model_cfg: PredictorConfig,
    train_cfg: TrainingConfig,
    center: Tensor,
    device: str,
    level: str,
    run_dir: Path,
    train_scenes: Sequence[str] | None = None,
    val_scenes: Sequence[str] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Train one (fold, seed) model and write its checkpoint and record.

    Plans are computed once per scene and reused for every pass, so the
    full-image visibility computation behind each support runs once per pair
    rather than once per pair per pass. Training order is a fresh deterministic
    permutation per pass, seeded by the training seed and the pass index.

    A previous run's checkpoint and record for the same (level, fold, seed) are
    moved aside to numbered siblings before training starts, so a resubmitted
    task never destroys the run it replaces.
    """
    train_scenes = tuple(fold.train if train_scenes is None else train_scenes)
    val_scenes = tuple(fold.val if val_scenes is None else val_scenes)
    assert_scenes_in_role(fold, train_scenes, "train", "train mode")
    assert_scenes_in_role(fold, val_scenes, "val", "train mode")

    train_plans: list[ExamplePlan] = []
    val_plans: list[ExamplePlan] = []
    census: list[PlanCensus] = []
    for scenes, bucket in ((train_scenes, train_plans), (val_scenes, val_plans)):
        for scene in scenes:
            inputs = store.get(scene)
            assert_grid_matches(inputs, model_cfg)
            plans, scene_census = plan_scene(cfg, analysis, inputs, level)
            bucket.extend(plans)
            census.append(scene_census)
    if not train_plans:
        raise ValueError(
            f"fold {fold.index}: no usable training pair at level {level}; "
            f"census {[dataclasses.asdict(c) for c in census]}"
        )
    if not val_plans:
        raise ValueError(
            f"fold {fold.index}: no usable validation pair at level {level}; "
            "checkpoint selection would have nothing to read"
        )

    passes = itertools.count()

    def train_examples() -> Iterator[Any]:
        index = next(passes)
        order = np.random.default_rng([seed, index]).permutation(len(train_plans))
        for i in order:
            plan = train_plans[int(i)]
            yield materialize_example(cfg, store.get(plan.scene), plan, center)

    def val_examples() -> Iterator[Any]:
        for plan in val_plans:
            yield materialize_example(cfg, store.get(plan.scene), plan, center)

    ckpt = checkpoint_path(run_dir, level, fold.index, seed)
    record_file = ckpt.with_suffix(".json")
    superseded = {"checkpoint": supersede(ckpt), "record": supersede(record_file)}

    _, record = train_fold(
        fold, seed, model_cfg, train_cfg, train_examples, val_examples,
        tuple(model_cfg.target_grid), device=device, checkpoint_path=ckpt,
        max_steps=max_steps,
    )
    if not ckpt.exists():
        raise RuntimeError(
            f"fold {fold.index} seed {seed}: training finished without selecting a "
            "checkpoint, so validation never produced a finite score"
        )
    payload = {
        **dataclasses.asdict(record),
        "level": level,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "n_train_examples": len(train_plans),
        "n_val_examples": len(val_plans),
        "census": [dataclasses.asdict(c) for c in census],
        "superseded": superseded,
        "config_digest": cfg.digest(),
        "commit": git_commit(),
    }
    write_once(record_file, json.dumps(payload, indent=2, sort_keys=True, default=str))
    return payload


# ---------------------------------------------------------------------------
# controls: Stream U step 17
# ---------------------------------------------------------------------------

def run_controls(
    cfg: Phase5Config,
    analysis: AnalysisConfig,
    store: SceneStore,
    fold: Fold,
    model: PredictWithDepth,
    model_cfg: PredictorConfig,
    center: Tensor,
    level: str,
    batch_pairs: int,
    control_seed: int,
    val_scenes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Pose and depth shuffles over one fold's whole validation set.

    Diagnostics, never thresholds. One derangement is drawn over every
    validation pair, exactly as lot.train.run_input_use_controls draws it over
    an in-memory list, and both shuffles use it against one shared baseline.

    A validation set is about 2,800 examples of roughly 10 MB each, too large
    to hold. So each example is materialized as the evaluation consumes it, and
    its moved fields are taken from the donor example built beside it. Memory
    stays near two batches. An earlier version shuffled only inside fixed
    chunks of 64 consecutive pairs. Consecutive pairs often share a context
    frame, so a depth shuffle could hand a pair its own depth map back and
    dilute the control. Drawing over the whole set restores the frozen
    control's population.

    An exchange that leaves an example's moved fields unchanged is counted per
    shuffle, because it dilutes the control and the reader should see by how
    much.
    """
    val_scenes = tuple(fold.val if val_scenes is None else val_scenes)
    assert_scenes_in_role(fold, val_scenes, "val", "controls mode")
    plans: list[ExamplePlan] = []
    for scene in val_scenes:
        inputs = store.get(scene)
        assert_grid_matches(inputs, model_cfg)
        plans.extend(plan_scene(cfg, analysis, inputs, level)[0])
    if len(plans) < 2:
        raise ValueError(
            f"fold {fold.index}: {len(plans)} usable validation pair(s); a shuffle "
            "needs at least two examples to exchange inputs between"
        )
    grid = tuple(model_cfg.target_grid)

    def example_at(index: int) -> Any:
        plan = plans[index]
        return materialize_example(cfg, store.get(plan.scene), plan, center)

    def stream(order: np.ndarray | None = None, fields: Sequence[str] = (),
               unchanged: list[int] | None = None) -> Iterator[Any]:
        for index in range(len(plans)):
            example = example_at(index)
            if order is not None:
                donor = example_at(int(order[index]))
                moved = {field: getattr(donor, field) for field in fields}
                if all(torch.equal(moved[f], getattr(example, f)) for f in fields):
                    unchanged[0] += 1
                example = dataclasses.replace(example, **moved)
            yield example

    with torch.no_grad():
        baseline = evaluate_validation(model, stream(), grid, batch_pairs)
    order = derangement(len(plans), control_seed)
    out: dict[str, Any] = {"n_pairs": len(plans)}
    for name, fields in SHUFFLED_FIELDS.items():
        unchanged = [0]
        with torch.no_grad():
            score = evaluate_validation(
                model, stream(order, fields, unchanged), grid, batch_pairs
            )
        if score.n_samples != baseline.n_samples:
            raise RuntimeError(
                f"{name} changed the supervised sample count from "
                f"{baseline.n_samples} to {score.n_samples}; a shuffle must move "
                "inputs only, never the support"
            )
        out[name] = {
            "baseline_centered_cosine": baseline.centered_cosine,
            "shuffled_centered_cosine": score.centered_cosine,
            "degradation": baseline.centered_cosine - score.centered_cosine,
            "n_samples": baseline.n_samples,
            "n_unchanged": unchanged[0],
        }
    return out


# ---------------------------------------------------------------------------
# evaluate: Streams V and W
# ---------------------------------------------------------------------------

def _empty_formulation() -> FormulationScores:
    nan = float("nan")
    return FormulationScores(
        0, nan, nan, nan, nan, nan, nan, nan, nan,
        np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
    )


def evaluation_scenes(folds: Sequence[Fold]) -> list[str]:
    """Every test scene, in a fixed order a SLURM array index can address."""
    return [scene for fold in folds for scene in fold.test]


def evaluate_scene(
    cfg: Phase5Config,
    analysis: AnalysisConfig,
    inputs: SceneInputs,
    fold: Fold,
    models: dict[int, PredictWithDepth],
    model_cfg: PredictorConfig,
    center: Tensor,
    reference: dict[str, Any],
    level: str,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every Phase 5 record for one test scene, at one level, for every seed.

    One record per (pair, seed, region). The explicit arms do not depend on the
    seed and repeat across a pair's seed records; the predictor columns are the
    seed's own. Seeds are never ensembled before scoring.

    For each pair, in order: the primary support from Context-Lift and ground
    truth; Phase 4's arms at this level, recomputed and reconciled against the
    accepted rows; the ground-truth region masks, recomputed and reconciled;
    the pre-registered landing-offset diagnostic on its own ground-truth
    support; every seed's predicted grid; then every score on its own fixed
    population.
    """
    from PIL import Image

    scene = inputs.scene
    assert_scenes_in_role_test(fold, scene)
    assert_grid_matches(inputs, model_cfg)
    grid = tuple(model_cfg.target_grid)
    center = center.to(torch.float32)

    pairs = phase5_scene_pairs(cfg, analysis, scene)
    phase5_keys = {(p.context_frame_id, p.target_frame_id) for p in pairs}
    drifted = sorted(reference["all_pairs"] - phase5_keys)
    if drifted:
        raise ReferenceMismatch(
            f"{scene}: Phase 4 evaluated {len(drifted)} pair(s) Phase 5 would not, "
            f"for example {drifted[:3]}; the two phases are not reading one "
            "population"
        )

    scene_root = Path(cfg.renders_root) / scene
    dtype = cfg.torch_dtype
    offset_edges = landing_offset_edges(cfg)
    no_offset = empty_landing_offset().as_fields()
    rows: list[dict[str, Any]] = []
    audit = {"pairs": len(pairs), "evaluated": 0, "no_arm": 0,
             "worst_per_point_residual": 0.0, "worst_splat_residual": 0.0}

    for pair in pairs:
        ctx, tgt = pair.context_frame_id, pair.target_frame_id
        where = f"{scene} {ctx} -> {tgt} level {level}"
        context_depth = aligned_context_depth(inputs, ctx, level)
        if context_depth is None:
            audit["no_arm"] += 1
            continue
        cams = pair_cameras(cfg, inputs, pair)
        depth_c_gt = inputs.cache.depth(cams.context.depth_path).to(dtype)
        depth_t_gt = inputs.cache.depth(cams.target.depth_path).to(dtype)
        fc = inputs.cache.features(cfg.feature_encoder, ctx)
        ft = inputs.cache.features(cfg.feature_encoder, tgt)

        lift = context_lift_map(
            torch.from_numpy(context_depth).to(dtype),
            cams.K_context, cams.K_target, cams.T_target_from_context,
            cams.context_hw, cams.target_hw,
        )
        support = primary_support(lift, context_lift_support(
            lift, depth_c_gt, depth_t_gt, cams.K_context, cams.K_target,
            cams.T_target_from_context, rel_tol=analysis.covisible_relative_depth_tol,
        ))

        persisted = reference["pairs"].get((ctx, tgt))
        arms = recompute_reference_arms(
            depth_c_gt, depth_t_gt, inputs.est_maps[ctx], inputs.est_maps[tgt],
            inputs.calibrations[ctx], fc, ft, cams.K_context, cams.K_target,
            cams.T_target_from_context, scene, ctx, tgt, analysis, level, dtype,
        )
        if arms is None:
            # The context depth exists but Phase 4's arm does not: the two
            # phases disagree about whether this level applies to the pair.
            raise ReferenceMismatch(f"{where}: Phase 5 has an arm, Phase 4 has none")
        worst = reconcile_reference(arms, persisted, center, where)
        audit["worst_per_point_residual"] = max(audit["worst_per_point_residual"], worst["pp"])
        audit["worst_splat_residual"] = max(audit["worst_splat_residual"], worst["sp"])

        rgb = np.asarray(Image.open(scene_root / cams.target.rgb_path))
        boundary, lowtex = region_cells(depth_t_gt, rgb, analysis, persisted, where)
        point_regions = {
            "all": support,
            **region_masks(lift, support, np.flatnonzero(boundary),
                           np.flatnonzero(lowtex), cams.target_hw),
        }
        cell_regions = {
            "all": arms.sp_scored,
            "boundary": arms.sp_scored & boundary,
            "interior": arms.sp_scored & ~boundary,
            "low_texture": arms.sp_scored & lowtex,
            "high_texture": arms.sp_scored & ~lowtex,
        }

        visible = model_inputs(cfg, inputs, pair, cams, context_depth)
        base = {
            **pair.as_row(),
            "parallax": arms.geometry.parallax,
            "covisible_fraction": arms.geometry.covisible_fraction,
            "level": level,
            "fold": fold.index,
        }
        # Both formulations are explicit, so the diagnostic is the same for
        # every seed and is computed once per pair.
        formulation = score_formulation(
            lift, support, fc, center, arms.pp_scored,
            arms.geometry.per_point_cells, arms.tl_reads, arms.reads_target,
            cams.target_hw,
        )
        # The pre-registered landing-offset diagnostic. Context-Lift lifted with
        # ground-truth context depth, on its own landed and evaluable samples.
        # A reference condition only: ground truth never reaches a method's
        # input, a support, or a headline score through it. It does not depend
        # on the alignment level, so every level writes the same columns.
        oracle = context_lift_map(
            depth_c_gt, cams.K_context, cams.K_target, cams.T_target_from_context,
            cams.context_hw, cams.target_hw,
        )
        oracle_support = primary_support(oracle, context_lift_support(
            oracle, depth_c_gt, depth_t_gt, cams.K_context, cams.K_target,
            cams.T_target_from_context, rel_tol=analysis.covisible_relative_depth_tol,
        ))
        offset = score_landing_offset(
            oracle, oracle_support, fc, ft, center, offset_edges
        ).as_fields()
        for seed, model in sorted(models.items()):
            with torch.no_grad():
                predicted = model(
                    visible["features_context"][None].to(device),
                    visible["depth_context_aligned"][None].to(device),
                    visible["camera"][None].to(device),
                    context_valid=visible["context_valid"][None].to(device),
                )[0].to("cpu", torch.float32)
            for region in REGIONS:
                points = point_regions[region]
                cells = np.flatnonzero(cell_regions[region])
                record = {
                    **base,
                    "seed": seed,
                    "region": region,
                    **score_primary(lift, points, fc, ft, center, predicted, grid).as_fields(),
                    **(formulation if region == "all" else _empty_formulation()).as_fields(),
                    **score_splat_pool(
                        cells, arms.transported_est, arms.flat_context,
                        arms.flat_target, center, predicted,
                    ).as_fields(),
                    **score_cross_path(
                        lift, points, cells, fc, ft, center, predicted,
                        arms.transported_est, arms.flat_context, arms.flat_target,
                        cams.target_hw, grid,
                    ).as_fields(),
                    **(offset if region == "all" else no_offset),
                }
                rows.append(record)
        audit["evaluated"] += 1
    return rows, audit


def assert_scenes_in_role_test(fold: Fold, scene: str) -> None:
    """Evaluation reads a scene only through the fold that held it out."""
    if scene not in fold.test:
        raise ValueError(
            f"{scene} is not a test scene of fold {fold.index}; it would be "
            "evaluated by a model that may have trained or selected on it"
        )


def evaluation_metadata(
    cfg: Phase5Config,
    analysis: AnalysisConfig,
    scene: str,
    fold: Fold,
    level: str,
    center: Tensor,
    run_dir: Path,
    seeds: Sequence[int],
    reference: dict[str, Any],
    phase4_parquet: Path,
    audit: dict[str, Any],
) -> dict[str, Any]:
    """The run record each evaluation parquet carries inside itself."""
    from .phase5_folds import fold_digest, frozen_folds

    return {
        "phase": 5,
        "scene": scene,
        "fold": fold.index,
        "level": level,
        "seeds": list(seeds),
        "commit": git_commit(),
        "config_digest": cfg.digest(),
        "fold_digest": fold_digest(frozen_folds()),
        "measurement_digest": analysis.measurement_digest(),
        "mean_vector_digest": vector_digest(center.detach().cpu().numpy()),
        "checkpoints": {
            str(seed): sha256_file(checkpoint_path(run_dir, level, fold.index, seed))
            for seed in seeds
        },
        "phase4_parquet_sha256": sha256_file(phase4_parquet),
        "phase4_commit": reference["metadata"].get("git_commit"),
        "audit": audit,
    }


def run_evaluate_scene(
    cfg: Phase5Config,
    analysis: AnalysisConfig,
    store: SceneStore,
    scene: str,
    fold: Fold,
    seeds: Sequence[int],
    model_cfg: PredictorConfig,
    train_cfg: TrainingConfig,
    center: Tensor,
    level: str,
    device: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Evaluate one test scene end to end and write its parquet, or resume.

    An existing output is never overwritten: the scene is reported as already
    done and skipped, which is also what makes a resubmitted array task resume
    rather than repeat finished work.
    """
    from .evaluate import write_rows

    out = Path(run_dir) / "eval" / level / f"{scene}.parquet"
    if out.exists():
        return {"scene": scene, "status": "exists", "path": str(out)}
    models = {
        seed: load_checkpoint(run_dir, level, fold, seed, model_cfg, train_cfg, device)
        for seed in seeds
    }
    phase4_eval = Path(cfg.phase4_dir) / "eval"
    reference = read_phase4_reference(phase4_eval, scene, level)
    inputs = store.get(scene)
    rows, audit = evaluate_scene(
        cfg, analysis, inputs, fold, models, model_cfg, center, reference, level, device
    )
    if not rows:
        raise ValueError(
            f"{scene}: no pair produced a record at level {level}; a test scene with "
            f"nothing evaluable is a stop, not an empty result. Audit {audit}"
        )
    metadata = evaluation_metadata(
        cfg, analysis, scene, fold, level, center, run_dir, seeds, reference,
        phase4_eval / f"{scene}.parquet", audit,
    )
    write_rows(out, rows, metadata)
    return {"scene": scene, "status": "written", "path": str(out), "rows": len(rows),
            "audit": audit}


def primary_evaluation_complete(run_dir: Path, primary_level: str,
                                scenes: Sequence[str]) -> list[str]:
    """Test scenes still missing a primary-level evaluation. Empty means complete.

    The sensitivity conditions run "after the primary result is frozen and
    interpreted". Code cannot see interpretation, but it can refuse to begin a
    sensitivity run while the primary result is incomplete, which is the part
    of that ordering a program can enforce.
    """
    return [
        scene for scene in scenes
        if not (Path(run_dir) / "eval" / primary_level / f"{scene}.parquet").exists()
    ]


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested and no CUDA device is visible")
    return requested
