"""Phase 5 modes: overfit, train, controls, lock, evaluate.

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

Provenance, reporting_rules.md section 7. Every training record, the controls
file, and every evaluation run record names the receipts that licensed it, as
a licence: each receipt's sha256, keyed by its file stem. The command line
hashes the receipts it just verified and passes the licence in. This module
records what it is handed. Each of those artifacts also records when it was
written, in UTC. Every evaluation attempt writes a start to the evaluation
ledger before any work, and a close when it ends inside Python. An attempt
killed outright leaves its start alone, so it is still on record. Each
evaluation run record also embeds its fold's training records, its controls
entries, and the overfit verdict, so the tables need nothing beside the
evaluation parquets. lot.phase5_provenance checks all of it against the files.

The checkpoint lock, reporting_rules.md section 7, runs after the controls and
before evaluation. This module checks what the lock binds and assembles its
payload. The command line stamps the payload as a receipt and writes it. The
lock binds the receipts that licensed training, so every training record and
the controls file must name those receipts as their licence.

Memory. A training run holds its training and validation scenes resident for
the whole run, about 1.2 GB each with the calibration ratios dropped, because
every pass reads all of them. Evaluation holds one test scene at a time.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import itertools
import json
import os
import secrets
import signal
from datetime import datetime, timezone
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
from .phase5_check import (
    environment_identity,
    sha256_file,
    supersede,
    utc_timestamp,
    write_once,
)
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

# The layout of the evaluation run record. Version 1 adds the provenance of
# reporting_rules.md section 7. A record without this field predates it.
# Version 1 also embeds the fold's training records, its controls entries, and
# the overfit verdict, so every table is regenerable from the evaluation
# parquets alone. No Phase 5 evaluation had run when they were added.
PHASE5_EVAL_VERSION = 1

# The overfit receipt's fields every evaluation run record carries: the
# verdict, and the gate's own fold, level, and seed.
OVERFIT_VERDICT_FIELDS = (
    "passed", "reached_centered_cosine", "threshold", "steps", "n_pairs",
    "regimes", "subset", "fold", "level", "seed",
)


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


def planned_scenes(plans: Sequence[ExamplePlan]) -> list[str]:
    """The scenes that contributed at least one plan, once each, in plan order.

    This is what a role actually consumed. A scene whose every pair had no arm
    or an empty support contributes no example, so it is not listed. The plan
    census still accounts for each of its pairs.
    """
    return list(dict.fromkeys(plan.scene for plan in plans))


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


def fold_seed_key(fold_index: int, seed: int) -> str:
    """How one (fold, seed) is keyed in the controls file and in the lock."""
    return f"fold{fold_index}_seed{seed}"


def load_checkpoint(
    run_dir: Path,
    level: str,
    fold: Fold,
    seed: int,
    model_cfg: PredictorConfig,
    train_cfg: TrainingConfig,
    device: str,
    *,
    config_digest: str,
) -> PredictWithDepth:
    """Load the selected model of one (fold, seed), proven to be training's own.

    The checkpoint must be the file the finished training run recorded, by
    content: its sha256 is compared against the training record written after
    selection. A checkpoint without a record, or whose bytes differ from the
    recorded ones, belongs to no completed run and is refused, as is one trained
    for another fold or seed or under another training configuration.

    config_digest is the digest of the configuration of the run that is loading.
    The training record must name the same one. reporting_rules.md section 7
    makes evaluation refuse a checkpoint trained under another configuration.
    The training-config digest alone cannot see that. It covers the training
    settings only, not the inputs, the alignment levels, or the architecture.
    The digest is required, so no caller can skip the binding by omission.
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
    if record.get("config_digest") != config_digest:
        raise ValueError(
            f"{path}: its training record names config digest "
            f"{record.get('config_digest')!r}, but this run's config digest is "
            f"{config_digest!r}. The checkpoint was trained under another "
            "configuration, so it does not belong to this run."
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
    licence: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Train one (fold, seed) model and write its checkpoint and record.

    Plans are computed once per scene and reused for every pass, so the
    full-image visibility computation behind each support runs once per pair
    rather than once per pair per pass. Training order is a fresh deterministic
    permutation per pass, seeded by the training seed and the pass index.

    A previous run's checkpoint and record for the same (level, fold, seed) are
    moved aside to numbered siblings before training starts, so a resubmitted
    task never destroys the run it replaces.

    The record keeps every field it had and adds the provenance of
    reporting_rules.md section 7: the licence it was handed, the scenes each
    role actually planned from, and when it was written, in UTC.
    train_scenes and val_scenes in the record are the fold's roles. The planned
    lists are what training and selection consumed. A run handed no licence
    records an empty one, so it never reads as a licensed run.
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
        "licence": dict(licence or {}),
        "train_scenes_planned": planned_scenes(train_plans),
        "val_scenes_planned": planned_scenes(val_plans),
        "written_utc": utc_timestamp(),
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

    The result names the validation scenes the control actually planned from,
    as val_scenes_planned, for reporting_rules.md section 7. The fold's role
    alone would not show a scene that contributed no usable pair.
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
    out["val_scenes_planned"] = planned_scenes(plans)
    return out


# ---------------------------------------------------------------------------
# Provenance: licences and the controls file
# ---------------------------------------------------------------------------

def receipt_licence(
    required: Sequence[Path], optional: Sequence[Path] = ()
) -> dict[str, str]:
    """The sha256 of each receipt that licensed a run, keyed by its file stem.

    The stem is kept when a rerun moves a receipt aside, as
    {stem}.superseded.N.json, so a reader can find the exact receipt from its
    stem and hash after a rerun. A required receipt that is absent is an error:
    the command line verified it moments earlier, so its absence means the
    evidence moved under the run. An optional one is recorded only when present.
    """
    licence: dict[str, str] = {}
    entries = [(Path(p), True) for p in required] + [(Path(p), False) for p in optional]
    for path, needed in entries:
        if not path.exists():
            if needed:
                raise FileNotFoundError(
                    f"the receipt {path} licensed this run and is now absent"
                )
            continue
        if path.stem in licence:
            raise ValueError(f"two receipts share the stem {path.stem!r}")
        licence[path.stem] = sha256_file(path)
    return licence


def controls_payload(
    level: str,
    results: dict[str, dict[str, Any]],
    missing: Sequence[str],
    identity: dict[str, Any],
    licence: dict[str, str],
    checkpoints: dict[str, str],
    written_utc: str | None = None,
) -> dict[str, Any]:
    """The input-use controls file, assembled in one place.

    The level, each (fold, seed)'s result, the missing checkpoints, and the run
    identity are the file as it was before provenance was added, unchanged.
    reporting_rules.md section 7 adds three entries. licence names the receipts
    that licensed the run. checkpoints names every checkpoint the controls
    loaded, by sha256, under the same fold{f}_seed{s} key as its result.
    written_utc says when the file was written, now unless a time is given.
    Each result also names the validation scenes it planned from.

    A result without its checkpoint's hash would be unbound, and a hash without
    a result would name a checkpoint the controls never ran, so both refuse.
    """
    unbound = sorted(set(results) - set(checkpoints))
    unrun = sorted(set(checkpoints) - set(results))
    if unbound or unrun:
        raise ValueError(
            f"controls results and checkpoint hashes disagree: results without a "
            f"hash {unbound}, hashes without a result {unrun}"
        )
    added = {"licence": dict(licence), "checkpoints": dict(checkpoints),
             "written_utc": written_utc or utc_timestamp()}
    clash = sorted(set(identity) & ({"level", "results", "missing"} | set(added)))
    if clash:
        raise ValueError(f"the run identity would overwrite controls fields {clash}")
    return {"level": level, "results": results, "missing": list(missing),
            **identity, **added}


def controls_path(evidence_dir: Path, level: str) -> Path:
    """The input-use controls file of one alignment level."""
    return Path(evidence_dir) / f"input_use_controls_{level}.json"


# ---------------------------------------------------------------------------
# lock: the checkpoint lock, reporting_rules.md section 7
# ---------------------------------------------------------------------------

class CheckpointLockError(ValueError):
    """The checkpoint lock was refused. The message lists every problem found."""


def checkpoint_lock_path(evidence_dir: Path, level: str) -> Path:
    """The checkpoint lock receipt of one alignment level."""
    return Path(evidence_dir) / f"checkpoint_lock_{level}.json"


def checkpoint_lock_files(
    cfg: Phase5Config, level: str, folds: Sequence[Fold], seeds: Sequence[int]
) -> dict[str, Any]:
    """Every file the checkpoint lock of a level binds, where a run under cfg reads it.

    checkpoints and training_records each map a fold_seed_key to a path, for
    every fold in folds and every seed in seeds. controls is the level's
    controls file. The lock is built from these paths, and the receipt verifier
    hashes the same paths again, so both read one definition of what the lock
    covers.
    """
    run_dir = Path(cfg.run_dir)
    runs = [(fold_seed_key(fold.index, seed), fold.index, seed)
            for fold in folds for seed in seeds]
    return {
        "checkpoints": {key: checkpoint_path(run_dir, level, f, s) for key, f, s in runs},
        "training_records": {
            key: training_record_path(run_dir, level, f, s) for key, f, s in runs
        },
        "controls": controls_path(cfg.evidence_dir, level),
    }


def checkpoint_lock_payload(
    cfg: Phase5Config,
    level: str,
    folds: Sequence[Fold],
    seeds: Sequence[int],
    model_cfg: PredictorConfig,
    train_cfg: TrainingConfig,
    device: str,
    identity: dict[str, Any],
    *,
    licence: dict[str, str],
) -> dict[str, Any]:
    """The checkpoint lock of one level, before it is stamped as a receipt.

    For every fold in folds and every seed in seeds, three checks:

    - the checkpoint loads through load_checkpoint. That proves its bytes are
      the ones its training record names, that the record names cfg's config
      digest, and that it was trained for that fold and seed under train_cfg;
    - the training record's identity. It must name the level, its own fold and
      seed, the commit and config digest of identity, and train_cfg's digest;
    - the training record's licence. It must be licence, the receipts the lock
      will bind. Training then ran under exactly those receipts.

    Then the level's controls file. It must exist, be for this level, carry
    identity, and name licence as its licence. It must list no missing
    checkpoint and have a result for every (fold, seed) and for nothing else.
    It must name each checkpoint by the sha256 the checkpoint has now, so the
    controls ran on the models locked.

    identity is the run identity the lock will be stamped with, as
    lot.phase5_receipt.current_identity returns it. Its config digest must be
    cfg's own. licence is the sha256 of the integration and overfit receipts
    the lock will be stamped with, keyed by file stem, as receipt_licence
    returns it. A rerun of either gate after training changes it, and the lock
    then refuses until training and the controls have rerun under it. A
    config digest mismatch or an empty licence is the caller's error and
    raises ValueError.

    Every problem is collected before refusing, so one run names them all.
    Any problem raises CheckpointLockError. Otherwise the payload names the
    level, the folds, the seeds, and each file it checked, by path and sha256.
    It says passed, because it exists only when every check passed.
    """
    if identity.get("config_digest") != cfg.digest():
        raise ValueError(
            f"the run identity names config digest {identity.get('config_digest')!r}, "
            f"but the configuration being locked has config digest {cfg.digest()!r}"
        )
    if not licence:
        raise ValueError(
            "the lock needs the licence it will bind, the integration and overfit "
            "receipts by sha256, to check that training ran under them"
        )
    licence = dict(licence)
    files = checkpoint_lock_files(cfg, level, folds, seeds)
    problems: list[str] = []
    live: dict[str, str] = {}
    for fold in folds:
        for seed in seeds:
            key = fold_seed_key(fold.index, seed)
            try:
                load_checkpoint(cfg.run_dir, level, fold, seed, model_cfg, train_cfg,
                                device, config_digest=cfg.digest())
            except Exception as error:  # noqa: BLE001
                # Any failure to load refuses the lock. It is reported with the
                # others rather than raised alone.
                problems.append(
                    f"{key}: the checkpoint does not load. {type(error).__name__}: {error}"
                )
                continue
            record = json.loads(files["training_records"][key].read_text(encoding="utf-8"))
            expected = {
                "level": level,
                "fold": fold.index,
                "seed": seed,
                "commit": identity.get("commit"),
                "config_digest": identity.get("config_digest"),
                "training_config_digest": train_cfg.digest(),
            }
            for field, want in expected.items():
                if record.get(field) != want:
                    problems.append(
                        f"{key}: its training record names {field} "
                        f"{record.get(field)!r}, but the lock expects {want!r}"
                    )
            if record.get("licence") != licence:
                problems.append(
                    f"{key}: its training record names licence "
                    f"{record.get('licence')!r}, but the lock binds {licence!r}. "
                    "Training ran under other receipts."
                )
            live[key] = sha256_file(files["checkpoints"][key])
    problems += _controls_lock_problems(
        files["controls"], level, identity, list(files["checkpoints"]), live, licence
    )
    if problems:
        raise CheckpointLockError(
            f"the checkpoint lock for level {level!r} cannot be written:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )

    def entry(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": sha256_file(path)}

    return {
        "passed": True,
        "level": level,
        "folds": [fold.index for fold in folds],
        "seeds": [int(seed) for seed in seeds],
        "checkpoints": {
            key: {"path": str(path), "sha256": live[key]}
            for key, path in files["checkpoints"].items()
        },
        "training_records": {
            key: entry(path) for key, path in files["training_records"].items()
        },
        "controls": entry(files["controls"]),
    }


def _controls_lock_problems(
    path: Path,
    level: str,
    identity: dict[str, Any],
    keys: Sequence[str],
    live: dict[str, str],
    licence: dict[str, str],
) -> list[str]:
    """What keeps a level's controls file out of the lock. Empty means nothing.

    keys are the fold_seed_keys the lock covers. live maps each key whose
    checkpoint loaded to that checkpoint's sha256 now. A key whose checkpoint
    did not load is already refused, so its hash is not compared here.
    licence is the receipts the lock binds, which the controls must have run
    under.
    """
    if not path.exists():
        return [f"no controls file at {path}; run the controls mode for level "
                f"{level!r} first"]
    try:
        controls = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"the controls file {path} cannot be read: {error}"]
    if not isinstance(controls, dict):
        return [f"the controls file {path} does not hold one controls result"]

    problems: list[str] = []
    if controls.get("level") != level:
        problems.append(
            f"the controls file is for level {controls.get('level')!r}, not {level!r}"
        )
    for field, want in identity.items():
        if controls.get(field) != want:
            problems.append(
                f"the controls file names {field} {controls.get(field)!r}, but the "
                f"lock expects {want!r}"
            )
    if controls.get("licence") != licence:
        problems.append(
            f"the controls file names licence {controls.get('licence')!r}, but the "
            f"lock binds {licence!r}. The controls ran under other receipts."
        )
    missing = controls.get("missing")
    if missing is None:
        problems.append("the controls file does not list its missing checkpoints")
    elif missing:
        problems.append(f"the controls file lists missing checkpoints: {missing}")
    results = controls.get("results")
    results = results if isinstance(results, dict) else {}
    absent = sorted(set(keys) - set(results))
    if absent:
        problems.append(f"the controls file has no result for {absent}")
    extra = sorted(set(results) - set(keys))
    if extra:
        problems.append(
            f"the controls file has results for {extra}, which this lock does not cover"
        )
    hashes = controls.get("checkpoints")
    hashes = hashes if isinstance(hashes, dict) else {}
    for key in keys:
        if key in live and key in results and hashes.get(key) != live[key]:
            problems.append(
                f"{key}: the controls ran on checkpoint {hashes.get(key)!r}, but the "
                f"checkpoint's sha256 is now {live[key]!r}"
            )
    return problems


# ---------------------------------------------------------------------------
# evaluate: Streams V and W
# ---------------------------------------------------------------------------

def _empty_formulation() -> FormulationScores:
    """The formulation record on a region row: no cells, every score NaN.

    Every field is named. A column added to FormulationScores then fails here
    loudly instead of shifting another column's value, which the earlier
    positional form would have done in silence.
    """
    nan = float("nan")
    return FormulationScores(
        n_formulation=0,
        tl_form_raw=nan, tl_form_centered=nan,
        tl_form_l2_raw=nan, tl_form_l2_centered=nan,
        cl_form_raw=nan, cl_form_centered=nan,
        cl_form_l2_raw=nan, cl_form_l2_centered=nan,
        nowarp_form_raw=nan, nowarp_form_centered=nan,
        nowarp_form_l2_raw=nan, nowarp_form_l2_centered=nan,
        meanfeat_form_raw=nan, meanfeat_form_l2_raw=nan,
        common_cells=np.zeros(0, dtype=np.int64),
        samples_per_cell=np.zeros(0, dtype=np.int64),
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
    # no_arm_pairs names each pair counted in no_arm, as [context frame, target
    # frame, regime], for reporting_rules.md section 7.
    audit = {"pairs": len(pairs), "evaluated": 0, "no_arm": 0,
             "worst_per_point_residual": 0.0, "worst_splat_residual": 0.0,
             "no_arm_pairs": []}

    for pair in pairs:
        ctx, tgt = pair.context_frame_id, pair.target_frame_id
        where = f"{scene} {ctx} -> {tgt} level {level}"
        context_depth = aligned_context_depth(inputs, ctx, level)
        if context_depth is None:
            audit["no_arm"] += 1
            audit["no_arm_pairs"].append([ctx, tgt, pair.regime])
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
            arms.geometry.samples.uv_target, cams.target_hw,
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
    *,
    licence: dict[str, str],
    aligned_depth_digest: str,
) -> dict[str, Any]:
    """The run record each evaluation parquet carries inside itself.

    The entries from phase to audit are the record as it was first written.
    The rest is the provenance of reporting_rules.md section 7:

    - eval_version, the layout of this record;
    - analysis_reporting_digest, the reporting constants the rows will be read
      under;
    - licence, the sha256 of each receipt that licensed this evaluation;
    - training_records, each seed's training record by sha256, with the commit
      and config digest it names;
    - aligned_depth_digest, the scene's aligned context depth, the value gate
      step 4 records for the same scene;
    - environment, what ran this;
    - written_utc, when this record was assembled, just before the write.

    The pairs with no arm are in the audit, as no_arm_pairs.

    Three more entries make every table regenerable from the evaluation
    parquets alone, as CLAUDE.md requires. See embedded_evidence.
    """
    from .phase5_folds import fold_digest, frozen_folds

    training_records: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        path = training_record_path(run_dir, level, fold.index, seed)
        record = json.loads(path.read_text(encoding="utf-8"))
        training_records[str(seed)] = {
            "sha256": sha256_file(path),
            "commit": record.get("commit"),
            "config_digest": record.get("config_digest"),
        }
    embedded = embedded_evidence(cfg, run_dir, level, fold, seeds, training_records)

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
        "eval_version": PHASE5_EVAL_VERSION,
        "analysis_reporting_digest": analysis.reporting_digest(),
        "licence": dict(licence),
        "training_records": training_records,
        "aligned_depth_digest": aligned_depth_digest,
        "environment": environment_identity(),
        "written_utc": utc_timestamp(),
        **embedded,
    }


def _read_bound_json(path: Path) -> tuple[str, Any]:
    """A JSON file's sha256 and its content, from one read of its bytes."""
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest(), json.loads(data.decode("utf-8"))


def embedded_evidence(
    cfg: Phase5Config,
    run_dir: Path,
    level: str,
    fold: Fold,
    seeds: Sequence[int],
    training_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """What an evaluation run record embeds beyond its provenance.

    CLAUDE.md requires every table and figure to be regenerable from the
    evaluation parquets alone. The training-adequacy table, the validation
    curves, and the controls read files beside the run, so each run record
    carries their content for its fold:

    - training_record_contents, each seed's training record whole, keyed by
      seed. training_records names each by sha256, and the bytes embedded
      must be the bytes it names, or a record changed between the two reads;
    - controls, the controls file's entries for this fold's (fold, seed)
      keys: each result and the sha256 of the checkpoint it ran on, with the
      file's own sha256 and written_utc;
    - overfit_verdict, the overfit receipt's OVERFIT_VERDICT_FIELDS, with the
      receipt's sha256.

    Each file is read where the command line writes it. The command line
    verified the receipts and the lock before evaluating, so a file that is
    absent now, or lacks this fold's entries, moved under the run. A record
    without it could never be reported, so it raises instead.
    """
    contents: dict[str, Any] = {}
    for seed in seeds:
        path = training_record_path(run_dir, level, fold.index, seed)
        sha, record = _read_bound_json(path)
        if sha != training_records[str(seed)]["sha256"]:
            raise ValueError(
                f"{path} changed while this run record was assembled: it hashed to "
                f"{training_records[str(seed)]['sha256']} and then to {sha}"
            )
        contents[str(seed)] = record

    controls_file = controls_path(cfg.evidence_dir, level)
    if not controls_file.exists():
        raise FileNotFoundError(
            f"no controls file at {controls_file}; the evaluation run record embeds "
            "the controls entries of its fold, and the lock bound that file"
        )
    controls_sha, controls = _read_bound_json(controls_file)
    controls = controls if isinstance(controls, dict) else {}
    keys = [fold_seed_key(fold.index, seed) for seed in seeds]
    results = controls.get("results") if isinstance(controls.get("results"), dict) else {}
    hashes = (controls.get("checkpoints")
              if isinstance(controls.get("checkpoints"), dict) else {})
    absent = [key for key in keys if key not in results or key not in hashes]
    if absent:
        raise ValueError(
            f"the controls file {controls_file} has no result or checkpoint hash for "
            f"{absent}, so this fold's controls cannot be embedded"
        )

    # Where the command line writes the overfit receipt and verifies it.
    overfit_file = Path(cfg.evidence_dir) / "tiny_overfit.json"
    if not overfit_file.exists():
        raise FileNotFoundError(
            f"no overfit receipt at {overfit_file}; the evaluation run record "
            "embeds its verdict"
        )
    overfit_sha, receipt = _read_bound_json(overfit_file)
    if not isinstance(receipt, dict):
        raise ValueError(f"{overfit_file} does not hold one receipt")

    return {
        "training_record_contents": contents,
        "controls": {
            "sha256": controls_sha,
            "written_utc": controls.get("written_utc"),
            "results": {key: results[key] for key in keys},
            "checkpoints": {key: hashes[key] for key in keys},
        },
        "overfit_verdict": {
            "sha256": overfit_sha,
            **{field: receipt.get(field) for field in OVERFIT_VERDICT_FIELDS},
        },
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
    licence: dict[str, str] | None = None,
    ledger_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one test scene end to end and write its parquet, or resume.

    An existing output is never overwritten. It is resumed, reported as
    already done and skipped, only when its run record shows it is this run's
    evaluation of the scene: see resume_problems. That is what makes a
    resubmitted array task resume rather than repeat finished work. An output
    that is not this run's is refused and left as it is.

    licence goes into the run record and the ledger as it is handed in. With a
    ledger directory, every call is one attempt with its own id. It writes a
    start record before any work, so a ledger that cannot be written stops the
    call before it computes anything. When the call ends inside Python, it
    writes a close record: written, exists, or error with its message. An error
    is recorded and then raised. An attempt killed outright, by SIGKILL, the
    out-of-memory killer, or a lost node, runs no Python, so it leaves its
    start alone. SIGTERM ends a SLURM task at its time limit and on scancel.
    Python's default for it also runs no cleanup, so the command line turns it
    into SystemExit with stop_on_termination, and the error close is written.
    """
    licence = dict(licence or {})
    out = Path(run_dir) / "eval" / level / f"{scene}.parquet"

    def attempt_once() -> dict[str, Any]:
        return _evaluate_scene_attempt(
            cfg, analysis, store, scene, fold, seeds, model_cfg, train_cfg, center,
            level, device, run_dir, out, licence,
        )

    if ledger_dir is None:
        return attempt_once()
    ledger_dir = Path(ledger_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    attempt = new_attempt_id()
    record_evaluation_attempt(ledger_dir, scene, level, LEDGER_STARTED, licence, out,
                              attempt=attempt)
    closed = False
    try:
        result = attempt_once()
        # The close is written inside the try. A signal that lands after the
        # parquet is written, and before this close, still gets an error
        # close, and that close names the parquet's sha256.
        record_evaluation_attempt(ledger_dir, scene, level, result["status"], licence,
                                  out, attempt=attempt)
        closed = True
    except BaseException as error:
        # BaseException, so KeyboardInterrupt and SystemExit are recorded too.
        if not closed:
            record_evaluation_attempt(
                ledger_dir, scene, level, "error", licence, out,
                message=f"{type(error).__name__}: {error}", attempt=attempt,
            )
        raise
    return result


def _evaluate_scene_attempt(
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
    out: Path,
    licence: dict[str, str],
) -> dict[str, Any]:
    """One attempt of run_evaluate_scene, without the ledger."""
    from .evaluate import write_rows
    from .phase5_check import aligned_depth_digest

    if out.exists():
        problems = resume_problems(cfg, out, scene, fold, seeds, level, run_dir)
        if problems:
            raise ValueError(
                f"{out} exists and is not this run's evaluation of {scene} at level "
                f"{level}, so it is not resumed. Outputs are never overwritten: move "
                "it aside to evaluate this scene.\n"
                + "\n".join(f"  - {problem}" for problem in problems)
            )
        return {"scene": scene, "status": "exists", "path": str(out)}
    models = {
        seed: load_checkpoint(run_dir, level, fold, seed, model_cfg, train_cfg, device,
                              config_digest=cfg.digest())
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
        licence=licence, aligned_depth_digest=aligned_depth_digest(inputs),
    )
    write_rows(out, rows, metadata)
    return {"scene": scene, "status": "written", "path": str(out), "rows": len(rows),
            "audit": audit}


def _live_sha256(path: Path) -> str | None:
    """A file's sha256 now, or None when it is absent."""
    return sha256_file(path) if Path(path).exists() else None


def resume_problems(
    cfg: Phase5Config,
    out: Path,
    scene: str,
    fold: Fold,
    seeds: Sequence[int],
    level: str,
    run_dir: Path,
) -> list[str]:
    """Why an existing output is not this run's evaluation. Empty means it is.

    An output is resumed only when its run record names this phase, scene,
    level, fold, seeds, and configuration digest. It must also name, by
    sha256, the checkpoints and training records this run reads now. Those
    are the files the checkpoint lock binds, and require_receipts verified
    the lock against them when the job started. So an output written under
    other checkpoints, before a retraining and a new lock, is never credited
    to the current lock. The files are compared rather than the lock's own
    hash, so a new lock over unchanged files still resumes. Comparing the
    training records as well catches a retraining that happens to reproduce
    the same checkpoint bytes.

    A file that is not a parquet, or a parquet with no run record, is not an
    evaluation of this run either.
    """
    from .evaluate import read_run_metadata

    try:
        meta = read_run_metadata(out)
    except Exception as error:  # noqa: BLE001
        # Any failure to read the record refuses the resume. It is reported
        # rather than raised alone.
        return [f"its run record cannot be read: {type(error).__name__}: {error}"]
    if not isinstance(meta, dict):
        return ["it carries no run record"]
    expected = {
        "phase": 5,
        "scene": scene,
        "level": level,
        "fold": fold.index,
        "seeds": [int(seed) for seed in seeds],
        "config_digest": cfg.digest(),
    }
    problems = [
        f"its run record names {field} {meta.get(field)!r}, but this attempt is for "
        f"{want!r}"
        for field, want in expected.items() if meta.get(field) != want
    ]
    live_checkpoints = {
        str(seed): _live_sha256(checkpoint_path(run_dir, level, fold.index, seed))
        for seed in seeds
    }
    if meta.get("checkpoints") != live_checkpoints:
        problems.append(
            f"its run record names checkpoints {meta.get('checkpoints')!r}, but the "
            f"checkpoints this run reads now have sha256 {live_checkpoints!r}"
        )
    records = meta.get("training_records")
    recorded = (
        {key: (entry or {}).get("sha256") for key, entry in records.items()}
        if isinstance(records, dict) else records
    )
    live_records = {
        str(seed): _live_sha256(training_record_path(run_dir, level, fold.index, seed))
        for seed in seeds
    }
    if recorded != live_records:
        problems.append(
            f"its run record names training records {recorded!r}, but the training "
            f"records this run reads now have sha256 {live_records!r}"
        )
    return problems


# ---------------------------------------------------------------------------
# The evaluation ledger, reporting_rules.md section 7
# ---------------------------------------------------------------------------
#
# Every evaluation attempt is recorded, so acceptance can require exactly one
# evaluation per scene and level. The rule's text names one appended file,
# evaluation_ledger.jsonl. The eighteen evaluation tasks run concurrently on a
# network file system, where appends from several nodes can interleave or be
# lost. So each record goes into its own file under a name no other can take,
# created exclusively, and nothing is ever appended or overwritten.
# read_evaluation_ledger reads the directory back as the one ledger it is.
# build_evaluation_ledger renders it to the file the rule names, one line per
# attempt, written once by a single writer after every task has ended.
# reporting_rules.md section 9 records this reading of section 7.
#
# An attempt begins when run_evaluate_scene is entered for one scene and level.
# A refusal before that, by a receipt, the lock, the level, or the scene index,
# computes nothing and names no scene, so it is not an attempt and writes
# nothing here. The job's own output file under the evidence directory keeps
# the refusal.
#
# An attempt writes two records under one attempt id: a start before any work,
# and a close when it ends inside Python. A start with no close is an attempt
# that was killed outright, and is listed as unfinished. What acceptance reads
# from the ledger is lot.phase5_acceptance.ledger_problems, under
# reporting_rules.md sections 7 and 9. Its docstring states each rule: every
# start is an attempt, an evaluation is a written close, and each scene and
# level has exactly one attempt on record as having written its live parquet.

LEDGER_EVENT = "evaluate"
LEDGER_STARTED = "started"
LEDGER_CLOSES = ("written", "exists", "error")
LEDGER_STATUSES = (LEDGER_STARTED,) + LEDGER_CLOSES
# An attempt whose start has no close. Never written as a record.
LEDGER_UNFINISHED = "unfinished"
# The file reporting_rules.md section 7 names, rendered from the directory.
LEDGER_FILE = "evaluation_ledger.jsonl"
# Fields a start and its close must agree on.
LEDGER_ATTEMPT_FIELDS = ("event", "scene", "level", "licence", "path", "commit")


def new_attempt_id() -> str:
    """An id for one evaluation attempt: the process id and 16 random hex digits."""
    return f"{os.getpid()}-{secrets.token_hex(8)}"


def record_evaluation_attempt(
    ledger_dir: Path,
    scene: str,
    level: str,
    status: str,
    licence: dict[str, str],
    out: Path,
    message: str | None = None,
    *,
    attempt: str,
) -> Path:
    """Write one ledger record of an attempt and return its path.

    status is "started" for the start, or the close: written, exists, or
    error. Fields: event, attempt, scene, level, status, message (None unless
    the attempt failed), commit, licence, utc, the parquet path, and the
    parquet's sha256 when the file exists at the time of the record. The name
    is the level, the scene, the record's UTC time, the attempt id, and the
    status. Every field is computed before the file is created, so the file is
    written in one call.
    """
    if status not in LEDGER_STATUSES:
        raise ValueError(f"ledger status {status!r} is not one of {LEDGER_STATUSES}")
    moment = datetime.now(timezone.utc)
    out = Path(out)
    record = {
        "event": LEDGER_EVENT,
        "attempt": attempt,
        "scene": scene,
        "level": level,
        "status": status,
        "message": message,
        "commit": git_commit(),
        "licence": dict(licence),
        "utc": utc_timestamp(moment),
        "path": str(out),
        "parquet_sha256": sha256_file(out) if out.exists() else None,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    name = (f"{level}__{scene}__{moment.strftime('%Y%m%dT%H%M%S%fZ')}__"
            f"{attempt}__{status}.json")
    path = Path(ledger_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    # Mode "x" creates the file or fails. A record never replaces another.
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(text)
    return path


def read_evaluation_ledger(ledger_dir: Path) -> list[dict[str, Any]]:
    """Every record in a ledger directory, oldest first, each with its file name.

    An absent directory holds no records. An unreadable record file is an
    error, never skipped: it is evidence of an attempt that did not finish
    writing its record.
    """
    directory = Path(ledger_dir)
    if not directory.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"evaluation ledger file {path} cannot be read: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"evaluation ledger file {path} does not hold one attempt")
        records.append({**record, "file": path.name})
    return sorted(records, key=lambda record: (record.get("utc") or "", record["file"]))


def evaluation_attempts(ledger_dir: Path) -> list[dict[str, Any]]:
    """One entry per attempt, its start paired with its close, in start order.

    status is the close's status, or "unfinished" when the start has no close.
    Each entry carries the attempt's event, scene, level, licence, path, and
    commit; its start and close times; the close's message; the parquet's
    sha256 at the start and at the close; and both file names.

    A ledger that does not describe attempts consistently is refused: a record
    without an attempt id or with an unknown status, a close without a start,
    two starts or two closes for one attempt, or a start and close that
    disagree on any of LEDGER_ATTEMPT_FIELDS.
    """
    starts: dict[str, dict[str, Any]] = {}
    closes: dict[str, dict[str, Any]] = {}
    for record in read_evaluation_ledger(ledger_dir):
        attempt = record.get("attempt")
        if not attempt:
            raise ValueError(
                f"evaluation ledger file {record['file']} names no attempt, so it "
                "cannot be paired with its start or close"
            )
        status = record.get("status")
        if status not in LEDGER_STATUSES:
            raise ValueError(
                f"evaluation ledger file {record['file']} has status {status!r}, not "
                f"one of {LEDGER_STATUSES}"
            )
        bucket, kind = (starts, "starts") if status == LEDGER_STARTED else (closes, "closes")
        if attempt in bucket:
            raise ValueError(
                f"attempt {attempt} has two {kind}: {bucket[attempt]['file']} and "
                f"{record['file']}"
            )
        bucket[attempt] = record
    orphans = sorted(set(closes) - set(starts))
    if orphans:
        raise ValueError(
            f"the evaluation ledger holds closes with no start for attempts {orphans}: "
            + ", ".join(closes[a]["file"] for a in orphans)
        )

    attempts: list[dict[str, Any]] = []
    for attempt, start in starts.items():
        close = closes.get(attempt)
        if close is not None:
            differ = [f for f in LEDGER_ATTEMPT_FIELDS if start.get(f) != close.get(f)]
            if differ:
                raise ValueError(
                    f"attempt {attempt}: its start {start['file']} and close "
                    f"{close['file']} disagree on {differ}"
                )
        attempts.append({
            "attempt": attempt,
            **{field: start.get(field) for field in LEDGER_ATTEMPT_FIELDS},
            "status": close["status"] if close else LEDGER_UNFINISHED,
            "message": close.get("message") if close else None,
            "started_utc": start.get("utc"),
            "finished_utc": close.get("utc") if close else None,
            "parquet_sha256_at_start": start.get("parquet_sha256"),
            "parquet_sha256": close.get("parquet_sha256") if close else None,
            "start_file": start["file"],
            "close_file": close["file"] if close else None,
        })
    return sorted(attempts, key=lambda a: (a["started_utc"] or "", a["attempt"]))


def build_evaluation_ledger(ledger_dir: Path, destination: Path) -> dict[str, Any]:
    """Render the ledger directory to the file reporting_rules.md section 7 names.

    One line per attempt, as evaluation_attempts returns it, in start order,
    each a JSON object with sorted keys. The same directory always gives the
    same text. It is written through write_once, so an earlier rendering is
    archived rather than overwritten. Run it once, after every evaluation task
    has ended: a single writer is what keeps the file whole on a network file
    system. Returns what write_once did.
    """
    text = "".join(
        json.dumps(attempt, sort_keys=True) + "\n"
        for attempt in evaluation_attempts(ledger_dir)
    )
    return write_once(Path(destination), text)


@contextlib.contextmanager
def stop_on_termination() -> Iterator[None]:
    """Within the block, SIGTERM raises SystemExit, so evaluation records its close.

    SLURM ends a task at its time limit, on scancel, and on preemption with
    SIGTERM, and then SIGKILL. Python's default for SIGTERM ends the process
    without unwinding, so no except or finally block would run and the attempt
    would keep only its start. The handler ignores any further SIGTERM, so a
    second one cannot interrupt the close being written, and raises
    SystemExit naming the signal. The previous handler is restored on leaving
    the block. Signal handlers can be installed in the main thread only, so
    the command line installs it around the evaluation loop, and
    run_evaluate_scene stays a plain library call.
    """
    def handler(signum: int, frame: Any) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(f"evaluation stopped by {signal.Signals(signum).name}")

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


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
