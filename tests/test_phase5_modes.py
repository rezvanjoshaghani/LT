"""The four Phase 5 modes, driven end to end on synthetic on-disk scenes.

Every earlier review round recorded the same gap: no mode had ever run, so
producer-level defects survived a green suite. This module closes it. Three
scenes are built on disk with rendered depth and RGB, a manifest, a feature
cache, and a VGGT depth cache. The test scene also gets a genuine Phase 3 run
and a genuine Phase 4 run, written to parquet by the real writers, so the
Phase 5 evaluator reconciles its recomputed references against a real Phase 4
artifact rather than against a fixture that agrees with itself.

The scenes take real names from the frozen fold 0, so every role assertion is
the real one: the training scene is in fold 0's training split, the validation
scene in its validation split, and the test scene in its sealed test split.

The construction is analytic. Estimated depth is ground truth divided by three
everywhere, so image-scale alignment recovers ground truth exactly, aligned
context depth equals ground truth, and TL-Reference at the image level equals
Oracle-Transport. Several answers follow without fitting anything.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from lot.analysis_config import load_analysis_config
from lot.encoders import cache_dir
from lot.evaluate import read_rows, write_rows
from lot.phase5_check import sha256_file
from lot.phase4 import Phase4Config, build_convention_record, convention_report, evaluate_scene_phase4
from lot.phase5 import (
    Phase5Config,
    build_example,
    build_scene_inputs,
    context_valid_tokens,
    materialize_example,
    plan_example,
    phase5_scene_pairs,
)
from lot.phase5_folds import frozen_folds
from lot.phase5_modes import (
    REGIONS,
    SceneStore,
    checkpoint_path,
    evaluate_scene,
    load_checkpoint,
    plan_scene,
    primary_evaluation_complete,
    run_controls,
    run_evaluate_scene,
    run_overfit,
    run_train_task,
    select_tiny_subset,
    training_record_path,
)
from lot.phase5_reference import ReferenceMismatch, read_phase4_reference
from lot.predictors import PredictorConfig
from lot.train import TrainingConfig

from test_phase4 import build_scene, planar_authority, run_phase3

FOLD = frozen_folds()[0]
TRAIN_SCENE = FOLD.train[0]
VAL_SCENE = FOLD.val[0]
TEST_SCENE = "room_0"
LEVEL = "image"
SEEDS = (0, 1)

# A small trunk of the frozen shape. The synthetic frames are 112 px, an 8 by 8
# patch grid, and the features keep the real 768 channels.
MODEL = PredictorConfig(
    d_model=32, n_blocks=1, n_heads=4, ffn_dim=64, dropout=0.0,
    feature_dim=768, context_grid=(8, 8), target_grid=(8, 8),
)
TRAIN = TrainingConfig(
    learning_rate=1e-3, weight_decay=0.0, batch_pairs=4, max_steps=4,
    warmup_steps=1, validation_every_steps=2, early_stopping_patience=99,
    seeds=SEEDS,
)

# The receipts that licensed a run, as the command line records them: each
# receipt's sha256, keyed by its file stem. The values here are synthetic. The
# modes record what they are handed and never check receipts themselves.
LICENCE = {"integration_gate": "1" * 64, "tiny_overfit": "2" * 64}

# ISO 8601 in UTC, always to the microsecond, with the offset written out.
UTC_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")


def _assert_utc(value) -> None:
    """A recorded time is a UTC stamp in the one format, and not in the future."""
    from datetime import datetime, timezone

    assert isinstance(value, str) and UTC_STAMP.match(value), value
    assert datetime.fromisoformat(value) <= datetime.now(timezone.utc)


def _licence_of(receipts: dict[str, Path]) -> dict[str, str]:
    return {stem: sha256_file(path) for stem, path in receipts.items()}


def _ledger(world) -> Path:
    return Path(world["cfg"].evidence_dir) / "evaluation_ledger"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Three scenes on disk, and real Phase 3 and Phase 4 runs on the test one."""
    assert TEST_SCENE in FOLD.test, "the test scene must be sealed in fold 0"
    root = tmp_path_factory.mktemp("phase5")
    for scene in (TRAIN_SCENE, VAL_SCENE, TEST_SCENE):
        build_scene(root, scene=scene)

    analysis = load_analysis_config()
    p4 = Phase4Config(
        experiment_name="phase4_rung1", renders_root=root, cache_root=root / "cache",
        output_root=root / "out", scenes=[TEST_SCENE], mean_vector_dir=root / "out" / "mv",
    )
    phase3_eval_dir, mean_vector = run_phase3(root, p4, analysis, scene=TEST_SCENE)
    p4.phase3_eval_dir = phase3_eval_dir
    convention = build_convention_record(
        convention_report(p4, analysis), planar_authority(),
        json.loads((cache_dir(p4.cache_root, "vggt_1b", TEST_SCENE) / "meta.json").read_text()),
    )
    rows, evidence = evaluate_scene_phase4(p4, TEST_SCENE, mean_vector, analysis, convention)
    phase4_dir = root / "p4"
    write_rows(phase4_dir / "eval" / f"{TEST_SCENE}.parquet", rows, evidence["metadata"])

    cfg = Phase5Config(
        renders_root=str(root), cache_root=str(root / "cache"),
        output_root=str(root / "p5"), phase4_dir=str(phase4_dir),
    )
    return {
        "root": root, "cfg": cfg, "analysis": analysis, "convention": convention,
        "center": mean_vector.to(torch.float32),
    }


@pytest.fixture
def store(world):
    with SceneStore(world["cfg"], world["analysis"], world["convention"]) as s:
        yield s


@pytest.fixture(scope="module")
def trained(world):
    """Both seeds of fold 0, trained on the synthetic training scene."""
    out = {}
    with SceneStore(world["cfg"], world["analysis"], world["convention"]) as store:
        for seed in SEEDS:
            out[seed] = run_train_task(
                world["cfg"], world["analysis"], store, FOLD, seed, MODEL, TRAIN,
                world["center"], "cpu", LEVEL, Path(world["cfg"].run_dir),
                train_scenes=[TRAIN_SCENE], val_scenes=[VAL_SCENE], licence=LICENCE,
            )
    return out


# ---------------------------------------------------------------------------
# Plans and examples
# ---------------------------------------------------------------------------

def test_plan_census_accounts_for_every_pair(world, store):
    plans, census = plan_scene(world["cfg"], world["analysis"], store.get(TRAIN_SCENE), LEVEL)
    assert census.n_pairs == census.n_planned + census.n_no_arm + census.n_empty_support
    assert census.n_planned == len(plans) > 0


def test_materializing_a_plan_reproduces_build_example(world, store):
    """The cached support must give exactly what a fresh build gives."""
    inputs = store.get(TRAIN_SCENE)
    pair = phase5_scene_pairs(world["cfg"], world["analysis"], TRAIN_SCENE)[0]
    fresh = build_example(world["cfg"], world["analysis"], inputs, pair, LEVEL, world["center"])
    plan = plan_example(world["cfg"], world["analysis"], inputs, pair, LEVEL)
    rebuilt = materialize_example(world["cfg"], inputs, plan, world["center"])
    for field in dataclasses.fields(fresh):
        a, b = getattr(fresh, field.name), getattr(rebuilt, field.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), field.name
        else:
            assert a == b, field.name


def test_context_valid_tokens_are_row_major_patches():
    depth = np.full((28, 42), 2.0, dtype=np.float32)       # a 2 by 3 patch grid
    depth[0:14, 14:28] = np.nan                            # patch (row 0, col 1)
    depth[14:28, 28:42] = -1.0                             # patch (row 1, col 2)
    valid = context_valid_tokens(depth)
    assert valid.tolist() == [True, False, True, True, True, False]


def test_a_patch_with_one_valid_pixel_stays_valid():
    depth = np.full((14, 14), np.nan, dtype=np.float32)
    depth[7, 7] = 3.0
    assert context_valid_tokens(depth).tolist() == [True]


def test_scene_inputs_drop_the_calibration_ratios_and_keep_the_fit(world):
    inputs = build_scene_inputs(world["cfg"], world["analysis"], TRAIN_SCENE, world["convention"])
    try:
        calibration = next(iter(inputs.calibrations.values()))
        assert calibration.ratios.size == 0
        # Ground truth over estimate is three everywhere, so the fit is exact.
        assert calibration.image_scale == pytest.approx(3.0, rel=1e-5)
    finally:
        inputs.close()


# ---------------------------------------------------------------------------
# overfit
# ---------------------------------------------------------------------------

def test_tiny_subset_is_deterministic_training_only_and_spans_regimes(world, store):
    args = (world["cfg"], world["analysis"], store, FOLD, LEVEL, 2,
            ("rotation", "translation"), [TRAIN_SCENE])
    first = select_tiny_subset(*args)
    second = select_tiny_subset(*args)
    assert [p.key for p in first] == [p.key for p in second]
    assert {p.pair.regime for p in first} == {"rotation", "translation"}
    assert all(p.scene in FOLD.train for p in first)


def test_tiny_subset_refuses_a_scene_outside_the_training_split(world, store):
    with pytest.raises(ValueError, match="not in its train split"):
        select_tiny_subset(world["cfg"], world["analysis"], store, FOLD, LEVEL, 1,
                           ("rotation",), [VAL_SCENE])


def test_tiny_subset_names_a_missing_regime(world, store):
    with pytest.raises(ValueError, match="no usable orbit pair"):
        select_tiny_subset(world["cfg"], world["analysis"], store, FOLD, LEVEL, 3,
                           ("rotation", "translation", "orbit"), [TRAIN_SCENE])


@pytest.mark.parametrize("threshold,expect", [(-1.0, True), (1.5, False)])
def test_overfit_records_its_verdict_and_never_raises(world, store, threshold, expect):
    result = run_overfit(
        world["cfg"], world["analysis"], store, FOLD, MODEL, TRAIN, world["center"],
        "cpu", LEVEL, 2, ("rotation", "translation"), threshold, 3, 0,
        train_scenes=[TRAIN_SCENE],
    )
    assert result["passed"] is expect
    assert result["fold"] == FOLD.index and result["level"] == LEVEL
    assert len(result["subset"]) == 2
    assert all(item["n_supported"] > 0 for item in result["subset"])
    json.dumps(result)   # a receipt must be serializable


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def test_training_writes_a_checkpoint_and_a_bound_record(world, trained):
    for seed, payload in trained.items():
        ckpt = checkpoint_path(Path(world["cfg"].run_dir), LEVEL, FOLD.index, seed)
        assert ckpt.exists()
        record = json.loads(training_record_path(
            Path(world["cfg"].run_dir), LEVEL, FOLD.index, seed
        ).read_text())
        assert record["seed"] == seed and record["fold"] == FOLD.index
        assert record["level"] == LEVEL
        assert record["n_train_examples"] > 0 and record["n_val_examples"] > 0
        assert math.isfinite(record["best_validation_centered_cosine"])
        assert record["census"], "the plan census must travel with the record"


# Every field a training record carried before provenance was added. Each keeps
# its name and its value. reporting_rules.md section 7 adds the others.
TRAINING_RECORD_BASE_FIELDS = (
    "fold", "seed", "train_scenes", "val_scenes", "test_scenes", "steps_run",
    "best_step", "best_validation_centered_cosine", "parameter_count",
    "training_config_digest", "stopped_early", "history",
    "level", "checkpoint", "checkpoint_sha256", "n_train_examples",
    "n_val_examples", "census", "superseded", "config_digest", "commit",
)
TRAINING_RECORD_PROVENANCE = (
    "licence", "train_scenes_planned", "val_scenes_planned", "written_utc",
)


def test_the_training_record_carries_its_provenance(world, trained):
    """The receipts that licensed the run, when the record was written, and the
    scenes each role actually consumed. train_scenes and val_scenes stay the
    fold's roles, so on this fixture the two differ: the run was handed one
    scene per role, and the fold names nine and three."""
    for seed, payload in trained.items():
        record = json.loads(training_record_path(
            Path(world["cfg"].run_dir), LEVEL, FOLD.index, seed
        ).read_text())
        assert set(record) == set(TRAINING_RECORD_BASE_FIELDS) | set(
            TRAINING_RECORD_PROVENANCE
        )
        assert record == json.loads(json.dumps(payload, default=str))
        assert record["licence"] == LICENCE
        assert record["train_scenes_planned"] == [TRAIN_SCENE]
        assert record["val_scenes_planned"] == [VAL_SCENE]
        assert record["train_scenes"] == list(FOLD.train) != [TRAIN_SCENE]
        assert record["val_scenes"] == list(FOLD.val) != [VAL_SCENE]
        _assert_utc(record["written_utc"])


def test_planned_scenes_names_each_contributing_scene_once_in_plan_order():
    """A scene that yields no plan consumed nothing and is not listed."""
    from lot.phase5_modes import planned_scenes

    plans = [SimpleNamespace(scene=scene) for scene in ("b", "a", "b", "c", "a")]
    assert planned_scenes(plans) == ["b", "a", "c"]
    assert planned_scenes([]) == []


def test_the_selected_checkpoint_loads_and_matches_its_record(world, trained):
    model = load_checkpoint(Path(world["cfg"].run_dir), LEVEL, FOLD, 0, MODEL, TRAIN, "cpu",
                            config_digest=world["cfg"].digest())
    assert not model.training


def test_a_tampered_checkpoint_is_refused(world, trained, tmp_path):
    run_dir = tmp_path / "copy"
    source = checkpoint_path(Path(world["cfg"].run_dir), LEVEL, FOLD.index, 0)
    target = checkpoint_path(run_dir, LEVEL, FOLD.index, 0)
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes() + b"x")
    training_record_path(run_dir, LEVEL, FOLD.index, 0).write_text(
        training_record_path(Path(world["cfg"].run_dir), LEVEL, FOLD.index, 0).read_text()
    )
    with pytest.raises(ValueError, match="not the checkpoint its training record names"):
        load_checkpoint(run_dir, LEVEL, FOLD, 0, MODEL, TRAIN, "cpu",
                        config_digest=world["cfg"].digest())


def test_a_checkpoint_without_a_record_is_refused(world, trained, tmp_path):
    source = checkpoint_path(Path(world["cfg"].run_dir), LEVEL, FOLD.index, 0)
    target = checkpoint_path(tmp_path, LEVEL, FOLD.index, 0)
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    with pytest.raises(FileNotFoundError, match="did not finish"):
        load_checkpoint(tmp_path, LEVEL, FOLD, 0, MODEL, TRAIN, "cpu",
                        config_digest=world["cfg"].digest())


def _copy_training_run(world, run_dir: Path, seeds=SEEDS, **record_changes) -> dict:
    """The world's fold 0 checkpoints and records under another run directory.

    The checkpoint bytes are copied as they are, so each record still names its
    checkpoint by content. record_changes edits every copied record, and a value
    of None removes that field. Returns each checkpoint's sha256 by its
    controls key.
    """
    hashes = {}
    for seed in seeds:
        source = checkpoint_path(Path(world["cfg"].run_dir), LEVEL, FOLD.index, seed)
        target = checkpoint_path(run_dir, LEVEL, FOLD.index, seed)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        record = json.loads(training_record_path(
            Path(world["cfg"].run_dir), LEVEL, FOLD.index, seed
        ).read_text())
        for field, value in record_changes.items():
            if value is None:
                record.pop(field, None)
            else:
                record[field] = value
        training_record_path(run_dir, LEVEL, FOLD.index, seed).write_text(
            json.dumps(record), encoding="utf-8"
        )
        hashes[f"fold{FOLD.index}_seed{seed}"] = sha256_file(target)
    return hashes


def test_a_checkpoint_trained_under_another_configuration_is_refused(world, trained, tmp_path):
    """reporting_rules.md section 7: evaluation refuses a checkpoint whose training
    record names a different config digest. The record is the one load_checkpoint
    already trusts for the checkpoint's bytes, so it is the record that is bound."""
    own = world["cfg"].digest()
    other = dataclasses.replace(world["cfg"], seed=world["cfg"].seed + 1).digest()
    assert other != own

    def load(run_dir, digest):
        return load_checkpoint(run_dir, LEVEL, FOLD, 0, MODEL, TRAIN, "cpu",
                               config_digest=digest)

    _copy_training_run(world, tmp_path / "same", seeds=(0,))
    assert not load(tmp_path / "same", own).training
    with pytest.raises(ValueError, match="config digest"):
        load(tmp_path / "same", other)

    _copy_training_run(world, tmp_path / "other", seeds=(0,), config_digest=other)
    with pytest.raises(ValueError, match="config digest"):
        load(tmp_path / "other", own)

    _copy_training_run(world, tmp_path / "unnamed", seeds=(0,), config_digest=None)
    with pytest.raises(ValueError, match="config digest"):
        load(tmp_path / "unnamed", own)


def test_retraining_keeps_the_previous_run(world, trained, tmp_path):
    cfg = dataclasses.replace(world["cfg"], output_root=str(tmp_path))
    with SceneStore(cfg, world["analysis"], world["convention"]) as store:
        for _ in range(2):
            run_train_task(cfg, world["analysis"], store, FOLD, 0, MODEL, TRAIN,
                           world["center"], "cpu", LEVEL, Path(cfg.run_dir),
                           train_scenes=[TRAIN_SCENE], val_scenes=[VAL_SCENE])
    kept = sorted(p.name for p in (Path(cfg.run_dir) / "checkpoints" / LEVEL).iterdir())
    assert "fold0_seed0.superseded.1.pt" in kept
    assert "fold0_seed0.superseded.1.json" in kept
    assert "fold0_seed0.pt" in kept


def test_training_refuses_a_validation_scene_in_the_training_role(world, store):
    with pytest.raises(ValueError, match="not in its train split"):
        run_train_task(world["cfg"], world["analysis"], store, FOLD, 0, MODEL, TRAIN,
                       world["center"], "cpu", LEVEL, Path("unused"),
                       train_scenes=[VAL_SCENE], val_scenes=[VAL_SCENE])


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

def _load(world, seed=0):
    return load_checkpoint(Path(world["cfg"].run_dir), LEVEL, FOLD, seed, MODEL, TRAIN, "cpu",
                           config_digest=world["cfg"].digest())


def test_controls_pool_both_shuffles_over_the_validation_set(world, trained, store):
    model = _load(world)
    out = run_controls(world["cfg"], world["analysis"], store, FOLD, model, MODEL,
                       world["center"], LEVEL, 4, 20260830, val_scenes=[VAL_SCENE])
    # The scenes the control actually planned from, which the fold's role
    # alone would not show: the fold names three validation scenes.
    assert out["val_scenes_planned"] == [VAL_SCENE]
    assert set(out) == {"n_pairs", "pose_shuffle", "depth_shuffle", "val_scenes_planned"}
    for name in ("pose_shuffle", "depth_shuffle"):
        result = out[name]
        assert result["n_samples"] > 0
        assert math.isfinite(result["baseline_centered_cosine"])
        assert result["degradation"] == pytest.approx(
            result["baseline_centered_cosine"] - result["shuffled_centered_cosine"]
        )
    # Both shuffles rest on one baseline over one set of samples.
    assert out["pose_shuffle"]["baseline_centered_cosine"] == pytest.approx(
        out["depth_shuffle"]["baseline_centered_cosine"]
    )


def test_controls_refuse_a_training_scene(world, trained, store):
    model = _load(world)
    with pytest.raises(ValueError, match="not in its val split"):
        run_controls(world["cfg"], world["analysis"], store, FOLD, model, MODEL,
                     world["center"], LEVEL, 4, 1, val_scenes=[TRAIN_SCENE])


def test_streaming_controls_equal_the_in_memory_controls(world, trained, store):
    """One derangement over the whole set, exactly as the in-memory control draws it.

    lot.train.run_input_use_controls holds every example. The mode streams them
    and builds each shuffled example from a donor. On a set small enough to
    hold, the two must agree to the last bit: same order, same moved fields,
    same batches.
    """
    from lot.train import run_input_use_controls

    model = _load(world)
    inputs = store.get(VAL_SCENE)
    plans = plan_scene(world["cfg"], world["analysis"], inputs, LEVEL)[0]
    examples = [materialize_example(world["cfg"], inputs, p, world["center"]) for p in plans]
    held = {r.name: r for r in run_input_use_controls(
        model, examples, MODEL.target_grid, 4, 20260830, FOLD)}
    streamed = run_controls(world["cfg"], world["analysis"], store, FOLD, model, MODEL,
                            world["center"], LEVEL, 4, 20260830, val_scenes=[VAL_SCENE])
    assert streamed["n_pairs"] == len(examples)
    for name, result in held.items():
        assert streamed[name]["baseline_centered_cosine"] == result.baseline_centered_cosine
        assert streamed[name]["shuffled_centered_cosine"] == result.shuffled_centered_cosine
        assert streamed[name]["n_samples"] == result.n_samples


def test_controls_count_exchanges_that_change_nothing(world, trained, store):
    """The fixture writes one depth map for every frame, so its depth shuffle is a no-op.

    Every aligned context map is the same array, so exchanging them changes no
    input and cannot change the score. The count makes that visible, rather
    than letting a control that measured nothing read as a finding that the
    network ignores depth. Camera vectors differ between pairs, so the pose
    shuffle changes most examples.
    """
    model = _load(world)
    out = run_controls(world["cfg"], world["analysis"], store, FOLD, model, MODEL,
                       world["center"], LEVEL, 4, 20260830, val_scenes=[VAL_SCENE])
    assert out["depth_shuffle"]["n_unchanged"] == out["n_pairs"]
    assert out["depth_shuffle"]["degradation"] == 0.0
    assert out["pose_shuffle"]["n_unchanged"] < out["n_pairs"]


# The controls file as it was before provenance was added: these keep their
# names and values. The identity is lot.phase5_receipt.current_identity's.
CONTROLS_BASE_FIELDS = ("level", "results", "missing",
                        "commit", "config_digest", "fold_digest", "measurement_digest")
CONTROLS_PROVENANCE = ("licence", "checkpoints", "written_utc")


def _controls_inputs():
    identity = {"commit": "c" * 40, "config_digest": "d" * 64,
                "fold_digest": "f" * 64, "measurement_digest": "m" * 32}
    result = {
        "n_pairs": 3,
        "pose_shuffle": {"baseline_centered_cosine": 0.5, "shuffled_centered_cosine": 0.25,
                         "degradation": 0.25, "n_samples": 9, "n_unchanged": 0},
        "depth_shuffle": {"baseline_centered_cosine": 0.5, "shuffled_centered_cosine": 0.5,
                          "degradation": 0.0, "n_samples": 9, "n_unchanged": 3},
        "val_scenes_planned": [VAL_SCENE],
    }
    results = {"fold0_seed0": result, "fold0_seed1": dict(result)}
    checkpoints = {"fold0_seed0": "3" * 64, "fold0_seed1": "4" * 64}
    return identity, results, checkpoints


def test_the_controls_file_carries_its_provenance():
    """reporting_rules.md section 7. The file names the receipts that licensed it,
    every checkpoint it loaded by sha256, and when it was written. Each result
    names the validation scenes it planned. What the file held before is
    unchanged."""
    from lot.phase5_modes import controls_payload

    identity, results, checkpoints = _controls_inputs()
    missing = ["no checkpoint for fold 1 seed 0"]
    stamp = "2026-10-10T08:00:00.000000+00:00"
    payload = controls_payload(LEVEL, results, missing, identity, LICENCE, checkpoints,
                               written_utc=stamp)
    assert set(payload) == set(CONTROLS_BASE_FIELDS) | set(CONTROLS_PROVENANCE)
    assert {field: payload[field] for field in CONTROLS_BASE_FIELDS} == {
        "level": LEVEL, "results": results, "missing": missing, **identity,
    }
    assert payload["licence"] == LICENCE
    assert payload["checkpoints"] == checkpoints
    assert payload["written_utc"] == stamp
    assert all(r["val_scenes_planned"] == [VAL_SCENE] for r in payload["results"].values())
    json.dumps(payload)
    # Written now unless a time is given.
    _assert_utc(controls_payload(LEVEL, results, missing, identity, LICENCE,
                                 checkpoints)["written_utc"])


def test_the_controls_file_binds_every_result_to_its_checkpoint():
    """A result without its checkpoint's hash is unbound. A hash without a result
    names a checkpoint the controls never ran. Either is refused."""
    from lot.phase5_modes import controls_payload

    identity, results, checkpoints = _controls_inputs()
    unbound = {"fold0_seed0": checkpoints["fold0_seed0"]}
    with pytest.raises(ValueError, match="fold0_seed1"):
        controls_payload(LEVEL, results, [], identity, LICENCE, unbound)
    extra = {**checkpoints, "fold2_seed2": "5" * 64}
    with pytest.raises(ValueError, match="fold2_seed2"):
        controls_payload(LEVEL, results, [], identity, LICENCE, extra)
    with pytest.raises(ValueError, match="licence"):
        controls_payload(LEVEL, results, [], {**identity, "licence": {}}, LICENCE,
                         checkpoints)


# ---------------------------------------------------------------------------
# lock: the checkpoint lock, reporting_rules.md section 7
# ---------------------------------------------------------------------------

def _lock_identity(world) -> dict:
    """The run identity a lock of the world's training run is stamped with.

    The commit and config digest are the ones the world's training records
    name. The other two digests are synthetic, because the lock compares them
    only with the controls file, which these tests write.
    """
    record = json.loads(training_record_path(
        Path(world["cfg"].run_dir), LEVEL, FOLD.index, SEEDS[0]
    ).read_text())
    return {"commit": record["commit"], "config_digest": world["cfg"].digest(),
            "fold_digest": "f" * 64, "measurement_digest": "m" * 32}


def _lock_run(world, tmp_path, seeds=SEEDS, **record_changes):
    """The world's fold 0 training run, copied under a relocated run directory.

    output_root is outside the config digest, so the relocated configuration is
    the world's own and every copied record still names it. Returns the
    configuration and each checkpoint's sha256 under its controls key.
    """
    cfg = dataclasses.replace(world["cfg"], output_root=str(tmp_path / "relocated"))
    assert cfg.digest() == world["cfg"].digest()
    return cfg, _copy_training_run(world, Path(cfg.run_dir), seeds=seeds, **record_changes)


def _write_lock_controls(cfg, identity, hashes, damage=None) -> Path:
    """A controls file for exactly these checkpoints, as the controls mode writes it.

    damage, when given, edits the payload before it is written.
    """
    from lot.phase5_modes import controls_payload

    result = {"n_pairs": 2, "val_scenes_planned": [VAL_SCENE],
              "pose_shuffle": {"degradation": 0.1, "n_unchanged": 0},
              "depth_shuffle": {"degradation": 0.0, "n_unchanged": 2}}
    payload = controls_payload(LEVEL, {key: dict(result) for key in hashes}, [],
                               identity, LICENCE, hashes)
    if damage is not None:
        damage(payload)
    path = Path(cfg.evidence_dir) / f"input_use_controls_{LEVEL}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _lock_payload(cfg, identity, licence=LICENCE):
    from lot.phase5_modes import checkpoint_lock_payload

    return checkpoint_lock_payload(cfg, LEVEL, [FOLD], SEEDS, MODEL, TRAIN, "cpu", identity,
                                   licence=licence)


def test_the_lock_binds_each_checkpoint_its_record_and_the_controls_file(
    world, trained, tmp_path
):
    """Nothing is locked before the controls file exists. Then each checkpoint,
    its training record, and the controls file are named by path and sha256."""
    from lot.phase5_modes import CheckpointLockError

    cfg, hashes = _lock_run(world, tmp_path)
    identity = _lock_identity(world)
    controls = Path(cfg.evidence_dir) / f"input_use_controls_{LEVEL}.json"
    with pytest.raises(CheckpointLockError, match="no controls file"):
        _lock_payload(cfg, identity)
    assert _write_lock_controls(cfg, identity, hashes) == controls

    payload = _lock_payload(cfg, identity)
    assert payload["passed"] is True
    assert payload["level"] == LEVEL
    assert payload["folds"] == [FOLD.index]
    assert payload["seeds"] == list(SEEDS)
    assert set(payload["checkpoints"]) == set(payload["training_records"]) == set(hashes)
    for seed in SEEDS:
        key = f"fold{FOLD.index}_seed{seed}"
        ckpt = checkpoint_path(Path(cfg.run_dir), LEVEL, FOLD.index, seed)
        record = training_record_path(Path(cfg.run_dir), LEVEL, FOLD.index, seed)
        assert payload["checkpoints"][key] == {"path": str(ckpt), "sha256": hashes[key]}
        assert payload["training_records"][key] == {
            "path": str(record), "sha256": sha256_file(record),
        }
    assert payload["controls"] == {"path": str(controls), "sha256": sha256_file(controls)}
    json.dumps(payload)


@pytest.mark.parametrize("remove,message", [
    (checkpoint_path, "no checkpoint for fold 0 seed 1"),
    (training_record_path, "has no training record"),
], ids=["checkpoint", "training record"])
def test_the_lock_refuses_a_missing_checkpoint(world, trained, tmp_path, remove, message):
    """The missing run is named, and only it: the controls file covers both seeds."""
    from lot.phase5_modes import CheckpointLockError

    cfg, hashes = _lock_run(world, tmp_path)
    identity = _lock_identity(world)
    _write_lock_controls(cfg, identity, hashes)
    remove(Path(cfg.run_dir), LEVEL, FOLD.index, 1).unlink()
    with pytest.raises(CheckpointLockError, match=message) as refused:
        _lock_payload(cfg, identity)
    assert "fold0_seed1" in str(refused.value)
    assert "fold0_seed0" not in str(refused.value)


@pytest.mark.parametrize("field,value,message", [
    ("config_digest", "0" * 64, "config digest '0{64}'"),
    ("level", "affine", "level 'affine'"),
    ("commit", "0" * 40, "commit '0{40}'"),
    ("fold", 2, "fold 2"),
    ("seed", 7, "seed 7"),
    ("training_config_digest", "0" * 64, "training_config_digest '0{64}'"),
])
def test_the_lock_refuses_a_training_record_that_names_another_run(
    world, trained, tmp_path, field, value, message
):
    """The training records' identity. A record under another configuration is
    already refused by load_checkpoint. The level, the commit, the fold, the
    seed, and the training settings are checked against the run being locked."""
    from lot.phase5_modes import CheckpointLockError

    cfg, hashes = _lock_run(world, tmp_path, **{field: value})
    identity = _lock_identity(world)
    _write_lock_controls(cfg, identity, hashes)
    with pytest.raises(CheckpointLockError, match=message):
        _lock_payload(cfg, identity)


def _drop_seed_one(payload):
    payload["results"].pop("fold0_seed1")
    payload["checkpoints"].pop("fold0_seed1")


@pytest.mark.parametrize("damage,message", [
    (_drop_seed_one, r"no result for \['fold0_seed1'\]"),
    (lambda p: p["missing"].append("no checkpoint for fold 0 seed 1"),
     "lists missing checkpoints"),
    (lambda p: p.pop("missing"), "does not list its missing checkpoints"),
    (lambda p: p["checkpoints"].update(fold0_seed1="0" * 64),
     "fold0_seed1: the controls ran on checkpoint '0{64}'"),
    (lambda p: p["results"].update(fold1_seed0={}), r"results for \['fold1_seed0'\]"),
    (lambda p: p.update(level="affine"), "controls file is for level 'affine'"),
    (lambda p: p.update(config_digest="0" * 64), "controls file names config_digest"),
    (lambda p: p.update(commit="0" * 40), "controls file names commit"),
], ids=["entry absent", "missing listed", "missing unrecorded", "other checkpoint",
        "extra result", "other level", "other configuration", "other commit"])
def test_the_lock_refuses_controls_that_do_not_cover_the_checkpoints(
    world, trained, tmp_path, damage, message
):
    from lot.phase5_modes import CheckpointLockError

    cfg, hashes = _lock_run(world, tmp_path)
    identity = _lock_identity(world)
    _write_lock_controls(cfg, identity, hashes, damage)
    with pytest.raises(CheckpointLockError, match=message):
        _lock_payload(cfg, identity)


def test_the_lock_refuses_an_identity_for_another_configuration(world, trained, tmp_path):
    """The identity the lock is stamped with must describe the configuration it
    checks. A mismatch is the caller's error, not a defect in the run."""
    from lot.phase5_modes import CheckpointLockError

    cfg, hashes = _lock_run(world, tmp_path)
    identity = {**_lock_identity(world), "config_digest": "0" * 64}
    _write_lock_controls(cfg, identity, hashes)
    with pytest.raises(ValueError, match="config digest") as refused:
        _lock_payload(cfg, identity)
    assert not isinstance(refused.value, CheckpointLockError)


# Receipts other than the ones the lock binds: the overfit gate was rerun.
OTHER_LICENCE = {"integration_gate": "1" * 64, "tiny_overfit": "3" * 64}


@pytest.mark.parametrize("licence", [{}, None, OTHER_LICENCE],
                         ids=["empty", "absent", "other receipts"])
def test_the_lock_refuses_training_licensed_by_other_receipts(
    world, trained, tmp_path, licence
):
    """Every training record must name, as its licence, the receipts the lock
    binds. A record trained under earlier receipts, or under none, is refused
    by name, so the lock never binds receipts the training did not run under."""
    from lot.phase5_modes import CheckpointLockError

    cfg, hashes = _lock_run(world, tmp_path, licence=licence)
    identity = _lock_identity(world)
    _write_lock_controls(cfg, identity, hashes)
    with pytest.raises(CheckpointLockError, match="licence") as refused:
        _lock_payload(cfg, identity)
    for seed in SEEDS:
        assert (f"fold{FOLD.index}_seed{seed}: its training record names licence"
                in str(refused.value))


@pytest.mark.parametrize("licence", [{}, OTHER_LICENCE], ids=["empty", "other receipts"])
def test_the_lock_refuses_controls_licensed_by_other_receipts(
    world, trained, tmp_path, licence
):
    from lot.phase5_modes import CheckpointLockError

    cfg, hashes = _lock_run(world, tmp_path)
    identity = _lock_identity(world)
    _write_lock_controls(cfg, identity, hashes, lambda p: p.update(licence=licence))
    with pytest.raises(CheckpointLockError, match="the controls file names licence"):
        _lock_payload(cfg, identity)


def test_the_lock_needs_the_licence_it_binds(world, trained, tmp_path):
    """An empty licence is the caller's error, not a defect in the run."""
    from lot.phase5_modes import CheckpointLockError

    cfg, hashes = _lock_run(world, tmp_path)
    identity = _lock_identity(world)
    _write_lock_controls(cfg, identity, hashes)
    with pytest.raises(ValueError, match="licence") as refused:
        _lock_payload(cfg, identity, licence={})
    assert not isinstance(refused.value, CheckpointLockError)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def evaluated(world, trained):
    with SceneStore(world["cfg"], world["analysis"], world["convention"]) as store:
        result = run_evaluate_scene(
            world["cfg"], world["analysis"], store, TEST_SCENE, FOLD, SEEDS, MODEL,
            TRAIN, world["center"], LEVEL, "cpu", Path(world["cfg"].run_dir),
            licence=LICENCE, ledger_dir=_ledger(world),
        )
    return result


def test_evaluation_writes_one_parquet_and_reconciles_with_phase4(evaluated):
    """Reaching a written file means every pair's recomputed TL and splat arms
    matched Phase 4's persisted masks bit for bit and its scores within bound."""
    assert evaluated["status"] == "written"
    audit = evaluated["audit"]
    assert audit["evaluated"] > 0
    assert audit["worst_per_point_residual"] <= 1e-5
    assert audit["worst_splat_residual"] <= 1e-5


def test_every_pair_has_every_seed_and_region(evaluated):
    rows = read_rows(Path(evaluated["path"]))
    keys = {(r["context_frame_id"], r["target_frame_id"]) for r in rows}
    for key in keys:
        present = {(r["seed"], r["region"]) for r in rows
                   if (r["context_frame_id"], r["target_frame_id"]) == key}
        assert present == {(s, region) for s in SEEDS for region in REGIONS}
    assert all(r["scene"] == TEST_SCENE and r["level"] == LEVEL for r in rows)


# The Mean-Feature floor on each record's support, and the count that support
# is measured by. PROTOCOL 3.7 defines it under raw metrics only.
MEAN_FEATURE_SUPPORT = {
    "meanfeat": "n_primary",
    "sp_meanfeat": "n_splat",
    "x_meanfeat": "n_intersect",
    "x_sp_meanfeat": "n_intersect",
}
MEAN_FEATURE_COLUMNS = tuple(
    f"{prefix}_{column}" for prefix in MEAN_FEATURE_SUPPORT for column in ("raw", "l2_raw")
)
# The two floors on the formulation support, reporting_rules.md section 6. They
# ride on the whole-support rows only, as the formulation diagnostic does.
FORMULATION_FLOOR_COLUMNS = (
    "nowarp_form_raw", "nowarp_form_centered",
    "nowarp_form_l2_raw", "nowarp_form_l2_centered",
    "meanfeat_form_raw", "meanfeat_form_l2_raw",
)
# The formulation columns that existed before the floors.
FORMULATION_BASE_COLUMNS = (
    "n_formulation",
    "tl_form_raw", "tl_form_centered", "tl_form_l2_raw", "tl_form_l2_centered",
    "cl_form_raw", "cl_form_centered", "cl_form_l2_raw", "cl_form_l2_centered",
)


def test_explicit_columns_do_not_depend_on_the_seed(evaluated):
    rows = read_rows(Path(evaluated["path"]))
    by_pair = {}
    for r in rows:
        if r["region"] != "all":
            continue
        by_pair.setdefault((r["context_frame_id"], r["target_frame_id"]), []).append(r)
    for records in by_pair.values():
        for column in ("cl_centered", "nowarp_centered", "sp_transport_centered",
                       "tl_form_centered", "n_primary", "offset_n_all",
                       "offset_cl_oracle_centered_all", "offset_nowarp_centered_b1",
                       *MEAN_FEATURE_COLUMNS, *FORMULATION_BASE_COLUMNS,
                       *FORMULATION_FLOOR_COLUMNS):
            # NaN is never equal to itself, so it is mapped to one marker; a pair
            # with no primary support has NaN explicit scores under every seed.
            values = {
                ("nan" if isinstance(r[column], float) and math.isnan(r[column])
                 else round(r[column], 12) if isinstance(r[column], float)
                 else r[column])
                for r in records
            }
            assert len(values) == 1, column


def test_every_row_carries_the_mean_feature_floor_on_each_support(evaluated):
    """CLAUDE.md: every reported metric travels with both floors. A record whose
    support is empty carries no floor score, and no row carries a centered one."""
    rows = read_rows(Path(evaluated["path"]))
    for r in rows:
        assert not [k for k in r if "meanfeat" in k and "centered" in k]
        for prefix, count in MEAN_FEATURE_SUPPORT.items():
            for column in ("raw", "l2_raw"):
                value = r[f"{prefix}_{column}"]
                assert math.isfinite(value) == (r[count] > 0), (prefix, column, r[count])
    # Both branches occur on this fixture, so neither half of the rule is vacuous.
    for count in set(MEAN_FEATURE_SUPPORT.values()):
        assert {r[count] > 0 for r in rows} == {True, False}, count


# ---------------------------------------------------------------------------
# What the explicit arms measure, against the analytic field
# ---------------------------------------------------------------------------
#
# The fixture's features are an analytic field of the world point, so at every
# supported landing three things can be compared that real data cannot supply:
# the context patch vector Context-Lift carries, the field at the point the
# target camera sees there, and the frozen scoring target, which is the target
# grid read bilinearly at the landing.
#
# The fixture writes one depth map for every frame. That map is consistent
# across views only for sideways translation and small rotation. Everywhere
# else no landing passes the 1.5 percent co-visibility rule, so the support is
# empty, for Phase 4's per-point arm exactly as for Context-Lift. That is why
# 28 of the 92 test pairs carry support.

def _bilinear(grid: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """[C, H, W] or [H, W] read at continuous (u, v), cell centers at integers.

    Written here rather than imported, so the reference shares no code with
    lot.encoders. Callers keep (u, v) inside the grid.
    """
    g = grid if grid.ndim == 3 else grid[None]
    u0 = np.minimum(np.floor(u).astype(int), g.shape[2] - 2)
    v0 = np.minimum(np.floor(v).astype(int), g.shape[1] - 2)
    du, dv = u - u0, v - v0
    out = (g[:, v0, u0] * (1 - du) * (1 - dv) + g[:, v0, u0 + 1] * du * (1 - dv)
           + g[:, v0 + 1, u0] * (1 - du) * dv + g[:, v0 + 1, u0 + 1] * du * dv)
    return out if grid.ndim == 3 else out[0]


def _camera_point(uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """OpenCV pinhole unprojection of pixels with planar z-depth."""
    return np.stack(((uv[:, 0] - K[0, 2]) * depth / K[0, 0],
                     (uv[:, 1] - K[1, 2]) * depth / K[1, 1], depth), axis=-1)


def _centered_cosines(a: np.ndarray, b: np.ndarray, center: np.ndarray) -> np.ndarray:
    a, b = a - center, b - center
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a * b).sum(-1)


def _raw_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _centered_cosines(a, b, np.zeros(a.shape[-1]))


@pytest.fixture(scope="module")
def landings(world):
    """Every supported test pair, with its landings recomputed independently.

    The reference is float64 numpy from ground-truth depth and the two world
    poses. It shares no code with lot.geometry, lot.context_lift, or
    lot.encoders, so an error common to the evaluator and its reference cannot
    hide here.
    """
    from lot.context_lift import context_lift_map, context_lift_support
    from lot.phase5 import aligned_context_depth, pair_cameras
    from lot.phase5_score import primary_support
    from test_phase4 import surface_field

    cfg, analysis = world["cfg"], world["analysis"]
    out = []
    with SceneStore(cfg, analysis, world["convention"]) as store:
        inputs = store.get(TEST_SCENE)
        for pair in phase5_scene_pairs(cfg, analysis, TEST_SCENE):
            cams = pair_cameras(cfg, inputs, pair)
            gt_c = inputs.cache.depth(cams.context.depth_path).to(torch.float32)
            gt_t = inputs.cache.depth(cams.target.depth_path).to(torch.float32)
            context_depth = aligned_context_depth(inputs, pair.context_frame_id, LEVEL)
            lift = context_lift_map(
                torch.from_numpy(context_depth), cams.K_context, cams.K_target,
                cams.T_target_from_context, cams.context_hw, cams.target_hw,
            )
            support = primary_support(lift, context_lift_support(
                lift, gt_c, gt_t, cams.K_context, cams.K_target,
                cams.T_target_from_context, rel_tol=analysis.covisible_relative_depth_tol,
            ))
            if not support.any():
                continue

            chosen = support.numpy()
            uv_c = lift.uv_context.numpy().astype(np.float64)[chosen]
            K_c = cams.context.K.numpy().astype(np.float64)
            K_t = cams.target.K.numpy().astype(np.float64)
            T_c = cams.context.T_world_from_camera.numpy().astype(np.float64)
            T_t = cams.target.T_world_from_camera.numpy().astype(np.float64)
            depth_c = gt_c.numpy().astype(np.float64)
            depth_t = gt_t.numpy().astype(np.float64)

            # The surface point each context patch center sees, carried into
            # the target camera with T_world_from_camera on both sides.
            points = (_camera_point(uv_c, _bilinear(depth_c, uv_c[:, 0], uv_c[:, 1]), K_c)
                      @ T_c[:3, :3].T + T_c[:3, 3])
            in_target = (points - T_t[:3, 3]) @ T_t[:3, :3]
            uv_t = np.stack((K_t[0, 0] * in_target[:, 0] / in_target[:, 2] + K_t[0, 2],
                             K_t[1, 1] * in_target[:, 1] / in_target[:, 2] + K_t[1, 2]),
                            axis=-1)
            # The point the target camera itself sees at that landing.
            seen = (_camera_point(uv_t, _bilinear(depth_t, uv_t[:, 0], uv_t[:, 1]), K_t)
                    @ T_t[:3, :3].T + T_t[:3, 3])

            fc = inputs.cache.features(cfg.feature_encoder, pair.context_frame_id)
            ft = inputs.cache.features(cfg.feature_encoder, pair.target_frame_id)
            fc64, ft64 = fc.numpy().astype(np.float64), ft.numpy().astype(np.float64)
            # Patch coordinates under the frozen mapping (u + 0.5) / 14 - 0.5.
            pu, pv = (uv_t[:, 0] + 0.5) / 14 - 0.5, (uv_t[:, 1] + 0.5) / 14 - 0.5
            cols = np.rint((uv_c[:, 0] - 6.5) / 14).astype(int)
            rows = np.rint((uv_c[:, 1] - 6.5) / 14).astype(int)
            out.append({
                "pair": pair,
                "key": (pair.context_frame_id, pair.target_frame_id),
                "cams": cams, "lift": lift, "support": support,
                "depth_gt": (gt_c, gt_t),
                "features": (fc, ft),
                "uv_target": uv_t,
                # Context-Lift carries the context patch's own vector, unread.
                "carried": fc64[:, rows, cols].T,
                "target_read": _bilinear(ft64, pu, pv).T,
                "nowarp_read": _bilinear(fc64, pu, pv).T,
                "field": surface_field(seen).T,
            })
    assert out, "the fixture must yield supported test pairs"
    return out


def _record(evaluated, key):
    return next(r for r in read_rows(Path(evaluated["path"]))
                if (r["context_frame_id"], r["target_frame_id"]) == key
                and r["seed"] == SEEDS[0] and r["region"] == "all")


def test_context_lift_lands_where_an_independent_reprojection_does(world, evaluated, landings):
    """The record's explicit scores are the scores at independently derived landings.

    Aligned context depth is ground truth up to the fp16 cache, which moves a
    translation landing by about two thousandths of a pixel; rotation landings
    do not depend on depth at all. A pose-direction, half-pixel, or intrinsics
    error would move every landing by a pixel or more.
    """
    center = world["center"].numpy().astype(np.float64)
    for item in landings:
        uv = item["lift"].uv_target.numpy().astype(np.float64)[item["support"].numpy()]
        assert np.abs(uv - item["uv_target"]).max() < 1e-2, item["key"]
        record = _record(evaluated, item["key"])
        cl = _centered_cosines(item["carried"], item["target_read"], center).mean()
        nowarp = _centered_cosines(item["nowarp_read"], item["target_read"], center).mean()
        # Mean-Feature predicts the frozen mean at every landing, against the
        # same read, and is scored under raw cosine only.
        meanfeat = _raw_cosines(
            np.broadcast_to(center, item["target_read"].shape), item["target_read"]
        ).mean()
        assert record["cl_centered"] == pytest.approx(cl, abs=1e-4), item["key"]
        assert record["nowarp_centered"] == pytest.approx(nowarp, abs=1e-4), item["key"]
        assert record["meanfeat_raw"] == pytest.approx(meanfeat, abs=1e-4), item["key"]


def test_context_lift_carries_the_surface_point_the_target_sees(world, landings):
    """Against the analytic field, Context-Lift's geometry is exact.

    Sideways translation is consistent in the fixture's shared depth map, so the
    carried vector is the field at the seen point up to the fp16 cache. Rotation
    pairs pass the co-visibility rule only where the shared map is within 1.5
    percent of the true depth, and that admitted depth error is what separates
    the two there. On that same truth Context-Lift beats No-Warp-Copy on every
    pair, because only No-Warp-Copy pays for the displacement.
    """
    center = world["center"].numpy().astype(np.float64)
    for item in landings:
        geometry = _centered_cosines(item["carried"], item["field"], center).mean()
        floor = _centered_cosines(item["nowarp_read"], item["field"], center).mean()
        bound = 0.9999 if item["pair"].regime == "translation" else 0.99
        assert geometry >= bound, (item["key"], geometry)
        assert geometry > floor, (item["key"], geometry, floor)


def test_landing_scoring_caps_context_lift_below_a_grid_predictor(world, landings):
    """Pins a property of the frozen per-point read, recorded as a Phase 5 finding.

    The scoring target is the target grid read bilinearly at the landing, and a
    predictor's grid is read with the same weights, so a predictor that returns
    the true target grid scores exactly one. Context-Lift carries one patch
    vector, which an off-grid read of the target grid does not reproduce even
    when the geometry is exact, as the previous test establishes. Under the
    frozen rule a grid predictor's ceiling is therefore one and Context-Lift's
    is lower, by the interpolation residual of the target features. On this
    fixture that residual is large because the field turns by about one to two
    radians across a patch. Its size on DINOv2 features is an open question for
    real data.
    See validation/evidence/phase5/landing_read_asymmetry.md.
    """
    from lot.phase5_score import score_primary

    center = world["center"]
    for item in landings:
        fc, ft = item["features"]
        ideal = ft.to(torch.float32).reshape(ft.shape[0], -1).T - center
        scores = score_primary(item["lift"], item["support"], fc, ft, center, ideal, (8, 8))
        assert scores.predict_centered == pytest.approx(1.0, abs=1e-6), item["key"]
        assert scores.cl_centered < scores.predict_centered, item["key"]


def test_region_splits_partition_the_primary_support(evaluated):
    rows = read_rows(Path(evaluated["path"]))
    for r_all in (r for r in rows if r["region"] == "all" and r["seed"] == SEEDS[0]):
        key = (r_all["context_frame_id"], r_all["target_frame_id"])
        split = {r["region"]: r["n_primary"] for r in rows
                 if (r["context_frame_id"], r["target_frame_id"]) == key
                 and r["seed"] == SEEDS[0]}
        assert split["boundary"] + split["interior"] == r_all["n_primary"]
        assert split["low_texture"] + split["high_texture"] == r_all["n_primary"]


def test_the_formulation_runs_only_on_the_whole_support(evaluated):
    rows = read_rows(Path(evaluated["path"]))
    assert all(r["n_formulation"] == 0 for r in rows if r["region"] != "all")
    assert any(r["n_formulation"] > 0 for r in rows if r["region"] == "all")


def test_the_formulation_floors_ride_only_on_the_whole_support_rows(evaluated):
    """Both floors exist on every row. On a whole-support row they are scored
    exactly when the formulation support is not empty. On a region row they are
    NaN, as every formulation column is. No centered Mean-Feature column exists."""
    rows = read_rows(Path(evaluated["path"]))
    for r in rows:
        assert not [k for k in r if "meanfeat_form" in k and "centered" in k]
        for column in FORMULATION_FLOOR_COLUMNS:
            if r["region"] == "all":
                assert math.isfinite(r[column]) == (r["n_formulation"] > 0), column
            else:
                assert math.isnan(r[column]), (r["region"], column)
    # Both branches occur on the whole-support rows, so neither is vacuous.
    assert {r["n_formulation"] > 0 for r in rows if r["region"] == "all"} == {True, False}


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _four_columns(prediction: np.ndarray, target: np.ndarray,
                  center: np.ndarray | None) -> dict[str, float]:
    """PROTOCOL 3.7's columns in float64, written without lot.

    center None gives the raw columns only, as Mean-Feature has. Its centered
    prediction is the zero vector, which has no direction.
    """
    out = {}
    shifts = [("raw", np.zeros(target.shape[-1]))]
    if center is not None:
        shifts.append(("centered", center))
    for name, shift in shifts:
        a, b = _unit_rows(prediction - shift), _unit_rows(target - shift)
        out[name] = float((a * b).sum(-1).mean())
        out[f"l2_{name}"] = float(np.linalg.norm(a - b, axis=-1).mean())
    return out


def test_target_lift_on_aligned_depth_is_target_lift_on_ground_truth(world, evaluated, landings):
    """At image scale the aligned maps are ground truth, so TL-Reference is its oracle.

    The formulation record is rebuilt with ground-truth depth handed to Phase
    4's estimator in place of both aligned maps, at the pass-through level. The
    scored cells must be identical and the score equal up to the fp16 depth
    cache. Nothing here asserts how high the score is: TL-Reference reads the
    context grid off-grid, so on this fixture it pays the same kind of
    interpolation residual Context-Lift pays on the target side.
    """
    from lot.phase5_reference import recompute_reference_arms
    from lot.phase5_score import score_formulation

    cfg, analysis, center = world["cfg"], world["analysis"], world["center"]
    checked = 0
    with SceneStore(cfg, analysis, world["convention"]) as store:
        inputs = store.get(TEST_SCENE)
        for item in landings:
            record = _record(evaluated, item["key"])
            if record["n_formulation"] == 0:
                continue
            ctx, tgt = item["key"]
            cams = item["cams"]
            gt_c, gt_t = item["depth_gt"]
            fc, ft = item["features"]

            def arms(est_c, est_t, level):
                return recompute_reference_arms(
                    gt_c, gt_t, est_c, est_t, inputs.calibrations[ctx], fc, ft,
                    cams.K_context, cams.K_target, cams.T_target_from_context,
                    TEST_SCENE, ctx, tgt, analysis, level, cfg.torch_dtype,
                )

            aligned = arms(inputs.est_maps[ctx], inputs.est_maps[tgt], LEVEL)
            oracle = arms(gt_c.numpy(), gt_t.numpy(), "none")
            assert np.array_equal(aligned.tl_landed, oracle.tl_landed), item["key"]
            assert np.array_equal(aligned.pp_scored, oracle.pp_scored), item["key"]
            landed = torch.from_numpy(aligned.tl_landed)
            residual = (aligned.tl_reads[landed] - oracle.tl_reads[landed]).abs().max()
            assert residual < 1e-3, item["key"]
            formulation = score_formulation(
                item["lift"], item["support"], fc, center, oracle.pp_scored,
                oracle.geometry.per_point_cells, oracle.tl_reads, oracle.reads_target,
                oracle.geometry.samples.uv_target, cams.target_hw,
            )
            assert formulation.n_formulation == record["n_formulation"], item["key"]
            assert record["tl_form_centered"] == pytest.approx(
                formulation.tl_form_centered, abs=1e-4), item["key"]
            assert record["cl_form_centered"] == pytest.approx(
                formulation.cl_form_centered, abs=1e-6), item["key"]
            checked += 1
    assert checked > 0


def test_every_formulation_column_matches_an_independent_reading(world, evaluated, landings):
    """The pre-existing formulation columns and both floors, recomputed in float64.

    TL-Reference's predictions and the Phase 3 sample coordinates come from
    Phase 4's arms, which evaluate reconciled against the persisted Phase 4
    rows. The rest is read here without lot.encoders. The cell target and
    No-Warp-Copy are the target and context grids read at the cell's Phase 3
    sample coordinate. Context-Lift pools the carried vectors of the
    independent landings in the cell. Mean-Feature is the frozen mean. Each
    common cell carries one weight.
    """
    from lot.phase5_reference import recompute_reference_arms
    from lot.phase5_score import formulation_cells

    cfg, analysis = world["cfg"], world["analysis"]
    center = world["center"].numpy().astype(np.float64)
    checked = 0
    with SceneStore(cfg, analysis, world["convention"]) as store:
        inputs = store.get(TEST_SCENE)
        for item in landings:
            record = _record(evaluated, item["key"])
            ctx, tgt = item["key"]
            cams = item["cams"]
            gt_c, gt_t = item["depth_gt"]
            fc, ft = item["features"]
            arms = recompute_reference_arms(
                gt_c, gt_t, inputs.est_maps[ctx], inputs.est_maps[tgt],
                inputs.calibrations[ctx], fc, ft, cams.K_context, cams.K_target,
                cams.T_target_from_context, TEST_SCENE, ctx, tgt, analysis, LEVEL,
                cfg.torch_dtype,
            )
            common, _ = formulation_cells(
                item["lift"], item["support"], arms.pp_scored, cams.target_hw
            )
            assert record["n_formulation"] == common.size, item["key"]
            if common.size == 0:
                for column in FORMULATION_FLOOR_COLUMNS:
                    assert math.isnan(record[column]), (item["key"], column)
                continue

            per_point_cells = np.asarray(arms.geometry.per_point_cells)
            sample = np.array([int(np.flatnonzero(per_point_cells == c).item())
                               for c in common])
            uv = arms.geometry.samples.uv_target.numpy().astype(np.float64)[sample]
            pu, pv = (uv[:, 0] + 0.5) / 14 - 0.5, (uv[:, 1] + 0.5) / 14 - 0.5
            fc64, ft64 = fc.numpy().astype(np.float64), ft.numpy().astype(np.float64)
            target = _bilinear(ft64, pu, pv).T
            nowarp = _bilinear(fc64, pu, pv).T
            tl = arms.tl_reads.numpy().astype(np.float64)[sample]
            grid_w = cams.target_hw[1] // 14
            landed = (np.rint((item["uv_target"][:, 1] - 6.5) / 14) * grid_w
                      + np.rint((item["uv_target"][:, 0] - 6.5) / 14)).astype(np.int64)
            pooled = np.stack([item["carried"][landed == c].mean(axis=0) for c in common])
            meanfeat = np.broadcast_to(center, target.shape)

            expected = {}
            for prefix, prediction in (("tl_form", tl), ("cl_form", pooled),
                                       ("nowarp_form", nowarp)):
                for column, value in _four_columns(prediction, target, center).items():
                    expected[f"{prefix}_{column}"] = value
            floor = _four_columns(meanfeat, target, None)
            expected["meanfeat_form_raw"] = floor["raw"]
            expected["meanfeat_form_l2_raw"] = floor["l2_raw"]
            assert set(expected) == set(FORMULATION_BASE_COLUMNS[1:]) | set(
                FORMULATION_FLOOR_COLUMNS
            )
            for column, value in expected.items():
                assert record[column] == pytest.approx(value, abs=1e-4), (item["key"], column)
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# The landing-offset diagnostic, pre-registered 2026-10-09
# ---------------------------------------------------------------------------

def test_the_offset_diagnostic_rides_only_on_the_whole_support_rows(evaluated):
    from lot.phase5_estimands import OFFSET_BINS

    rows = read_rows(Path(evaluated["path"]))
    for r in rows:
        if r["region"] != "all":
            assert r["offset_n_all"] == 0
            assert math.isnan(r["offset_cl_oracle_centered_all"])
    whole = [r for r in rows if r["region"] == "all"]
    assert any(r["offset_n_all"] > 0 for r in whole)
    for r in whole:
        assert r["offset_n_all"] == sum(r[f"offset_n_{label}"] for label in OFFSET_BINS)


def test_on_this_fixture_the_oracle_support_is_the_primary_support(evaluated):
    """Aligned depth is ground truth up to the fp16 cache here, so lifting with
    either lands the same samples. On real data the two supports differ by
    exactly the samples estimated depth moves across the landing rule."""
    for r in read_rows(Path(evaluated["path"])):
        if r["region"] == "all":
            assert r["offset_n_all"] == r["n_primary"], (r["context_frame_id"],
                                                         r["target_frame_id"])


def test_the_offset_bins_match_an_independent_reprojection(world, evaluated, landings):
    """Counts and scores per bin, from float64 landings computed without lot.

    On this fixture ground truth and aligned depth land the same samples, so
    the independent landings of the primary support are the oracle landings.
    The closest fixture offset sits 4e-4 patch from an edge, far beyond float
    noise, so the bin of every sample is decided the same way on both sides.
    """
    from lot.phase5_estimands import OFFSET_BINS

    edges = np.array(world["cfg"].landing_offset["upper_edges_patch"])
    center = world["center"].numpy().astype(np.float64)
    for item in landings:
        record = _record(evaluated, item["key"])
        patch = (item["uv_target"] + 0.5) / 14 - 0.5
        offset = np.linalg.norm(patch - np.round(patch), axis=1)
        assert np.abs(offset[:, None] - edges[None, :]).min() > 1e-6
        bins = (offset[:, None] > edges[None, :]).sum(axis=1)
        cosines = _centered_cosines(item["carried"], item["target_read"], center)
        for k, label in enumerate(OFFSET_BINS):
            members = bins == k
            assert record[f"offset_n_{label}"] == int(members.sum()), (item["key"], label)
            if members.any():
                assert record[f"offset_cl_oracle_centered_{label}"] == pytest.approx(
                    cosines[members].mean(), abs=1e-4), (item["key"], label)


def test_the_parquet_carries_its_run_record(evaluated):
    from lot.evaluate import read_run_metadata

    meta = read_run_metadata(Path(evaluated["path"]))
    assert meta["scene"] == TEST_SCENE and meta["fold"] == FOLD.index
    assert set(meta["checkpoints"]) == {str(s) for s in SEEDS}
    assert meta["phase4_parquet_sha256"]


# The run record as it was first written. These keep their names and values.
# reporting_rules.md section 7 adds the rest.
EVAL_RECORD_BASE_FIELDS = (
    "phase", "scene", "fold", "level", "seeds", "commit", "config_digest",
    "fold_digest", "measurement_digest", "mean_vector_digest", "checkpoints",
    "phase4_parquet_sha256", "phase4_commit", "audit",
)
EVAL_RECORD_PROVENANCE = (
    "eval_version", "analysis_reporting_digest", "licence", "training_records",
    "aligned_depth_digest", "environment", "written_utc",
)
AUDIT_BASE_FIELDS = (
    "pairs", "evaluated", "no_arm", "worst_per_point_residual", "worst_splat_residual",
)


def test_the_run_record_carries_its_provenance(world, evaluated):
    """Everything section 7 asks of an evaluation parquet, read back from the file."""
    from lot.evaluate import read_run_metadata
    from lot.phase5_check import aligned_depth_digest, environment_identity
    from lot.phase5_modes import PHASE5_EVAL_VERSION

    meta = read_run_metadata(Path(evaluated["path"]))
    assert set(meta) == set(EVAL_RECORD_BASE_FIELDS) | set(EVAL_RECORD_PROVENANCE)
    assert meta["eval_version"] == PHASE5_EVAL_VERSION == 1
    assert meta["analysis_reporting_digest"] == world["analysis"].reporting_digest()
    assert meta["licence"] == LICENCE

    # Each seed's training record, by content, with the identity it names.
    run_dir = Path(world["cfg"].run_dir)
    expected = {}
    for seed in SEEDS:
        path = training_record_path(run_dir, LEVEL, FOLD.index, seed)
        record = json.loads(path.read_text())
        expected[str(seed)] = {"sha256": sha256_file(path), "commit": record["commit"],
                               "config_digest": record["config_digest"]}
    assert meta["training_records"] == expected
    assert {r["config_digest"] for r in expected.values()} == {meta["config_digest"]}

    # The same digest gate step 4 records, of the inputs this scene was read from.
    with SceneStore(world["cfg"], world["analysis"], world["convention"]) as store:
        assert meta["aligned_depth_digest"] == aligned_depth_digest(store.get(TEST_SCENE))
    assert meta["environment"] == environment_identity()
    _assert_utc(meta["written_utc"])

    # No pair of this fixture lacks an arm at the image level.
    audit = meta["audit"]
    assert set(audit) == set(AUDIT_BASE_FIELDS) | {"no_arm_pairs"}
    assert audit["no_arm"] == len(audit["no_arm_pairs"]) == 0
    assert audit == evaluated["audit"]


def test_pairs_with_no_arm_are_named_in_the_audit(world, trained, store, monkeypatch):
    """A pair whose context depth has no arm at this level is listed, not only counted.

    Two pairs keep the run cheap. The first pair's context frame loses its arm,
    so that pair is named, and the second is evaluated as usual.
    """
    import lot.phase5_modes as modes

    pairs = phase5_scene_pairs(world["cfg"], world["analysis"], TEST_SCENE)
    lost = pairs[0]
    kept = next(p for p in pairs if p.context_frame_id != lost.context_frame_id)
    real = modes.aligned_context_depth

    def without_arm(inputs, context_frame_id, level):
        if context_frame_id == lost.context_frame_id:
            return None
        return real(inputs, context_frame_id, level)

    monkeypatch.setattr(modes, "phase5_scene_pairs", lambda *args: [lost, kept])
    monkeypatch.setattr(modes, "aligned_context_depth", without_arm)
    reference = read_phase4_reference(Path(world["cfg"].phase4_dir) / "eval", TEST_SCENE, LEVEL)
    reference["all_pairs"] = {(p.context_frame_id, p.target_frame_id) for p in (lost, kept)}

    rows, audit = evaluate_scene(world["cfg"], world["analysis"], store.get(TEST_SCENE), FOLD,
                                 _models(world), MODEL, world["center"], reference, LEVEL,
                                 "cpu")
    assert (audit["pairs"], audit["evaluated"], audit["no_arm"]) == (2, 1, 1)
    assert audit["no_arm_pairs"] == [[lost.context_frame_id, lost.target_frame_id, lost.regime]]
    assert {(r["context_frame_id"], r["target_frame_id"]) for r in rows} == {
        (kept.context_frame_id, kept.target_frame_id)
    }
    json.dumps(audit)


def test_a_written_evaluation_leaves_one_ledger_attempt(world, evaluated):
    """reporting_rules.md section 7: every evaluation attempt is recorded.

    An attempt writes a start before any work and a close when it ends. Both
    name one attempt id. The parquet is written between the two.
    """
    from lot.evaluate import read_run_metadata
    from lot.phase5_modes import evaluation_attempts, read_evaluation_ledger

    records = read_evaluation_ledger(_ledger(world))
    written = [a for a in records if a["status"] == "written"]
    assert len(written) == 1
    (attempt,) = written
    (start,) = [a for a in records
                if a["status"] == "started" and a["attempt"] == attempt["attempt"]]
    meta = read_run_metadata(Path(evaluated["path"]))
    for record in (start, attempt):
        assert record["event"] == "evaluate"
        assert (record["scene"], record["level"]) == (TEST_SCENE, LEVEL)
        assert record["licence"] == LICENCE
        assert record["commit"] == meta["commit"]
        assert record["message"] is None
        assert record["path"] == evaluated["path"]
        _assert_utc(record["utc"])
        # Named by level, scene, the record's own time, the attempt, and status.
        assert record["file"].startswith(f"{LEVEL}__{TEST_SCENE}__")
        assert record["file"].endswith(f"__{record['attempt']}__{record['status']}.json")
    # The start saw no parquet. The close names the one the attempt wrote.
    assert start["parquet_sha256"] is None
    assert attempt["parquet_sha256"] == sha256_file(Path(evaluated["path"]))
    # One fixed format makes string order time order.
    assert start["utc"] <= meta["written_utc"] <= attempt["utc"]

    (paired,) = [a for a in evaluation_attempts(_ledger(world))
                 if a["attempt"] == attempt["attempt"]]
    assert paired["status"] == "written"
    assert paired["started_utc"] == start["utc"] and paired["finished_utc"] == attempt["utc"]


def test_evaluation_resumes_instead_of_overwriting(world, evaluated):
    from lot.phase5_modes import read_evaluation_ledger

    before = Path(evaluated["path"]).read_bytes()
    attempts_before = {a["file"] for a in read_evaluation_ledger(_ledger(world))}
    with SceneStore(world["cfg"], world["analysis"], world["convention"]) as store:
        again = run_evaluate_scene(
            world["cfg"], world["analysis"], store, TEST_SCENE, FOLD, SEEDS, MODEL,
            TRAIN, world["center"], LEVEL, "cpu", Path(world["cfg"].run_dir),
            licence=LICENCE, ledger_dir=_ledger(world),
        )
    assert again["status"] == "exists"
    assert Path(evaluated["path"]).read_bytes() == before
    # The resumed call is an attempt too. It leaves a start and an exists
    # close beside the earlier files, without touching them.
    after = read_evaluation_ledger(_ledger(world))
    new = [a for a in after if a["file"] not in attempts_before]
    assert len(new) == 2 and len(after) == len(attempts_before) + 2
    assert sorted(a["status"] for a in new) == ["exists", "started"]
    assert len({a["attempt"] for a in new}) == 1
    for record in new:
        assert record["scene"] == TEST_SCENE
        assert record["parquet_sha256"] == sha256_file(Path(evaluated["path"]))


def test_evaluation_refuses_a_checkpoint_trained_under_another_configuration(
    world, trained, tmp_path
):
    """The refusal comes before any scene is read, and the attempt is recorded."""
    from lot.phase5_modes import read_evaluation_ledger

    run_dir = tmp_path / "run"
    _copy_training_run(world, run_dir)
    other = dataclasses.replace(world["cfg"], seed=world["cfg"].seed + 1)
    ledger = tmp_path / "ledger"
    with SceneStore(other, world["analysis"], world["convention"]) as store:
        with pytest.raises(ValueError, match="config digest"):
            run_evaluate_scene(
                other, world["analysis"], store, TEST_SCENE, FOLD, SEEDS, MODEL, TRAIN,
                world["center"], LEVEL, "cpu", run_dir, licence=LICENCE, ledger_dir=ledger,
            )
    assert not (run_dir / "eval").exists()
    start, close = read_evaluation_ledger(ledger)
    assert start["status"] == "started" and close["status"] == "error"
    assert start["attempt"] == close["attempt"]
    assert close["message"].startswith("ValueError:") and "config digest" in close["message"]
    for record in (start, close):
        assert record["parquet_sha256"] is None
        assert record["licence"] == LICENCE


def test_ledger_attempts_never_share_a_file(tmp_path):
    """Concurrent array tasks write side by side, so no two attempts may collide."""
    from lot.phase5_modes import (
        new_attempt_id, read_evaluation_ledger, record_evaluation_attempt,
    )

    out = tmp_path / "eval" / LEVEL / f"{TEST_SCENE}.parquet"
    ids = [new_attempt_id() for _ in range(5)]
    assert len(set(ids)) == 5
    paths = set()
    for attempt in ids:
        for status in ("started", "exists"):
            paths.add(record_evaluation_attempt(tmp_path / "ledger", TEST_SCENE, LEVEL,
                                                status, LICENCE, out, attempt=attempt))
    assert len(paths) == 10
    assert len(read_evaluation_ledger(tmp_path / "ledger")) == 10
    with pytest.raises(ValueError, match="status"):
        record_evaluation_attempt(tmp_path / "ledger", TEST_SCENE, LEVEL, "skipped",
                                  LICENCE, out, attempt=new_attempt_id())
    (tmp_path / "ledger" / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="broken.json"):
        read_evaluation_ledger(tmp_path / "ledger")
    assert read_evaluation_ledger(tmp_path / "absent") == []


def _ledger_entry(ledger, status, attempt, scene=TEST_SCENE, out=None, message=None,
                  licence=LICENCE):
    from lot.phase5_modes import record_evaluation_attempt

    out = out or ledger.parent / "eval" / LEVEL / f"{scene}.parquet"
    return record_evaluation_attempt(ledger, scene, LEVEL, status, licence, out,
                                     message=message, attempt=attempt)


def test_the_ledger_pairs_each_start_with_its_close(tmp_path):
    """One entry per attempt. A start with no close is an unfinished attempt:
    the process was killed outside Python's reach."""
    from lot.phase5_modes import evaluation_attempts

    ledger = tmp_path / "ledger"
    _ledger_entry(ledger, "started", "11-aa")
    _ledger_entry(ledger, "error", "11-aa", message="ValueError: boom")
    _ledger_entry(ledger, "started", "12-bb")
    _ledger_entry(ledger, "written", "12-bb")
    _ledger_entry(ledger, "started", "13-cc")
    attempts = evaluation_attempts(ledger)
    assert [(a["attempt"], a["status"]) for a in attempts] == [
        ("11-aa", "error"), ("12-bb", "written"), ("13-cc", "unfinished"),
    ]
    for attempt in attempts:
        _assert_utc(attempt["started_utc"])
        assert (attempt["scene"], attempt["level"]) == (TEST_SCENE, LEVEL)
        assert attempt["licence"] == LICENCE
    assert attempts[0]["message"] == "ValueError: boom"
    assert attempts[0]["started_utc"] <= attempts[0]["finished_utc"]
    assert attempts[2]["finished_utc"] is None and attempts[2]["close_file"] is None
    assert evaluation_attempts(tmp_path / "absent") == []


@pytest.mark.parametrize("entries, message", [
    ([("error", "21-aa", TEST_SCENE)], "no start"),
    ([("started", "22-aa", TEST_SCENE), ("started", "22-aa", TEST_SCENE)], "two starts"),
    ([("started", "23-aa", TEST_SCENE), ("written", "23-aa", TEST_SCENE),
      ("exists", "23-aa", TEST_SCENE)], "two closes"),
    ([("started", "24-aa", TEST_SCENE), ("written", "24-aa", "room_1")], "scene"),
], ids=["close without start", "two starts", "two closes", "start and close disagree"])
def test_an_inconsistent_ledger_is_refused(tmp_path, entries, message):
    from lot.phase5_modes import evaluation_attempts

    ledger = tmp_path / "ledger"
    for status, attempt, scene in entries:
        _ledger_entry(ledger, status, attempt, scene=scene,
                      out=tmp_path / "eval" / LEVEL / f"{TEST_SCENE}.parquet")
    with pytest.raises(ValueError, match=message):
        evaluation_attempts(ledger)


def test_a_ledger_record_without_an_attempt_id_is_refused(tmp_path):
    from lot.phase5_modes import evaluation_attempts

    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "old.json").write_text(json.dumps({
        "event": "evaluate", "scene": TEST_SCENE, "level": LEVEL, "status": "written",
        "utc": "2026-10-10T08:00:00.000000+00:00",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="attempt"):
        evaluation_attempts(ledger)


def test_the_ledger_renders_to_the_file_the_rule_names(tmp_path):
    """reporting_rules.md section 7 names evaluation_ledger.jsonl. Each attempt
    writes its own files, because appends from many nodes are not safe on a
    network file system. The named file is rendered from them, one line per
    attempt in start order. The same directory always gives the same bytes,
    and a rebuild archives the earlier file rather than overwriting it."""
    from lot.phase5_modes import (
        LEDGER_FILE, build_evaluation_ledger, evaluation_attempts,
    )

    ledger = tmp_path / "evidence" / "evaluation_ledger"
    destination = tmp_path / "evidence" / LEDGER_FILE
    assert LEDGER_FILE == "evaluation_ledger.jsonl"
    for attempt, close in (("31-aa", "error"), ("32-bb", "written"), ("33-cc", "exists"),
                           ("34-dd", None)):
        _ledger_entry(ledger, "started", attempt)
        if close is not None:
            _ledger_entry(ledger, close, attempt,
                          message="ValueError: boom" if close == "error" else None)

    outcome = build_evaluation_ledger(ledger, destination)
    assert outcome == {"written": str(destination), "archived_previous": None}
    first = destination.read_bytes()
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == evaluation_attempts(ledger)
    assert [json.loads(line)["status"] for line in lines] == [
        "error", "written", "exists", "unfinished",
    ]

    again = build_evaluation_ledger(ledger, destination)
    assert again["archived_previous"] == str(
        destination.with_name("evaluation_ledger.superseded.1.jsonl")
    )
    assert destination.read_bytes() == first
    assert Path(again["archived_previous"]).read_bytes() == first


def test_an_attempt_records_its_start_before_any_work(world, tmp_path, monkeypatch):
    """The start exists before the attempt touches a checkpoint or a scene."""
    import lot.phase5_modes as modes

    ledger = tmp_path / "ledger"
    seen = {}

    def attempt(cfg, analysis, store, scene, fold, seeds, model_cfg, train_cfg, center,
                level, device, run_dir, out, licence):
        seen["records"] = modes.read_evaluation_ledger(ledger)
        return {"scene": scene, "status": "exists", "path": str(out)}

    monkeypatch.setattr(modes, "_evaluate_scene_attempt", attempt)
    run_evaluate_scene(world["cfg"], world["analysis"], None, TEST_SCENE, FOLD, SEEDS,
                       MODEL, TRAIN, world["center"], LEVEL, "cpu", tmp_path / "run",
                       licence=LICENCE, ledger_dir=ledger)
    (start,) = seen["records"]
    assert start["status"] == "started"
    assert (start["scene"], start["level"], start["licence"]) == (TEST_SCENE, LEVEL, LICENCE)
    assert [a["status"] for a in modes.evaluation_attempts(ledger)] == ["exists"]


# A child process that enters run_evaluate_scene and is then killed outright,
# as SIGKILL, the out-of-memory killer, or a lost node kill a SLURM task. No
# Python cleanup runs. The attempt body can first write the parquet's path, so
# the kill lands after the write and before the close.
_KILLED_CHILD = r"""
import os
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import lot.phase5_modes as modes

write_first = sys.argv[6] == "after the write"

def killed(cfg, analysis, store, scene, fold, seeds, model_cfg, train_cfg, center,
           level, device, run_dir, out, licence):
    if write_first:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"written before the kill")
    os._exit(137)

modes._evaluate_scene_attempt = killed
modes.run_evaluate_scene(None, None, None, sys.argv[2], None, (0, 1), None, None, None,
                         sys.argv[3], "cpu", Path(sys.argv[4]),
                         licence={"integration_gate": "1" * 64},
                         ledger_dir=Path(sys.argv[5]))
"""


@pytest.mark.parametrize("when", ["before the write", "after the write"])
def test_a_killed_attempt_leaves_its_start_and_reads_as_unfinished(tmp_path, when):
    import subprocess
    import sys

    from lot.phase5_modes import build_evaluation_ledger, evaluation_attempts

    src = Path(__file__).resolve().parents[1] / "src"
    run_dir, ledger = tmp_path / "run", tmp_path / "ledger"
    child = subprocess.run(
        [sys.executable, "-c", _KILLED_CHILD, str(src), TEST_SCENE, LEVEL, str(run_dir),
         str(ledger), when],
        capture_output=True, text=True, timeout=600,
    )
    assert child.returncode == 137, child.stderr
    out = run_dir / "eval" / LEVEL / f"{TEST_SCENE}.parquet"
    assert out.exists() is (when == "after the write")
    (start,) = [json.loads(p.read_text(encoding="utf-8")) for p in ledger.iterdir()]
    assert start["status"] == "started" and start["scene"] == TEST_SCENE
    assert start["parquet_sha256"] is None
    (attempt,) = evaluation_attempts(ledger)
    assert attempt["status"] == "unfinished"
    assert attempt["attempt"] == start["attempt"]
    build_evaluation_ledger(ledger, tmp_path / "evaluation_ledger.jsonl")
    (line,) = (tmp_path / "evaluation_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["status"] == "unfinished"


def _raise_sigterm():
    """Deliver SIGTERM to this process, as SLURM does at a time limit.

    Refuses first unless a Python handler is installed. Without one, SIGTERM
    would end the test process instead of failing the test.
    """
    import signal

    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler), f"no SIGTERM handler is installed: {handler!r}"
    signal.raise_signal(signal.SIGTERM)


@pytest.mark.parametrize("when", ["before the write", "after the write"])
def test_sigterm_during_an_attempt_is_recorded_as_its_error_close(
    world, tmp_path, monkeypatch, when
):
    """A time limit, scancel, or preemption sends SIGTERM. Evaluation turns it
    into SystemExit, so the attempt's error close is written. After the parquet
    is written, the close names it, so acceptance can tie the file to the
    attempt that wrote it. The previous handler is restored afterwards."""
    import signal

    import lot.phase5_modes as modes

    ledger = tmp_path / "ledger"
    out = tmp_path / "run" / "eval" / LEVEL / f"{TEST_SCENE}.parquet"

    def attempt(cfg, analysis, store, scene, fold, seeds, model_cfg, train_cfg, center,
                level, device, run_dir, out, licence):
        if when == "after the write":
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"written before the signal")
        _raise_sigterm()
        raise AssertionError("SIGTERM did not stop the attempt")

    monkeypatch.setattr(modes, "_evaluate_scene_attempt", attempt)
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit, match="SIGTERM"):
        with modes.stop_on_termination():
            run_evaluate_scene(world["cfg"], world["analysis"], None, TEST_SCENE, FOLD,
                               SEEDS, MODEL, TRAIN, world["center"], LEVEL, "cpu",
                               tmp_path / "run", licence=LICENCE, ledger_dir=ledger)
    assert signal.getsignal(signal.SIGTERM) == before
    start, close = modes.read_evaluation_ledger(ledger)
    assert (start["status"], close["status"]) == ("started", "error")
    assert start["attempt"] == close["attempt"]
    assert close["message"] == "SystemExit: evaluation stopped by SIGTERM"
    if when == "after the write":
        assert close["parquet_sha256"] == sha256_file(out)
    else:
        assert close["parquet_sha256"] is None


def test_a_resumed_scene_must_be_this_runs_evaluation(world, evaluated, tmp_path):
    """An existing output is resumed only when its run record names this scene,
    level, fold, seeds, and configuration, and the checkpoints and training
    records the run reads now, by sha256. The training run and the parquet are
    copied byte for byte, so the copy resumes."""
    import shutil

    from lot.phase5_modes import evaluation_attempts

    run_dir = tmp_path / "run"
    for seed in SEEDS:
        for path_of in (checkpoint_path, training_record_path):
            source = path_of(Path(world["cfg"].run_dir), LEVEL, FOLD.index, seed)
            target = path_of(run_dir, LEVEL, FOLD.index, seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    out = run_dir / "eval" / LEVEL / f"{TEST_SCENE}.parquet"
    out.parent.mkdir(parents=True)
    shutil.copyfile(evaluated["path"], out)
    ledger = tmp_path / "ledger"
    with SceneStore(world["cfg"], world["analysis"], world["convention"]) as store:
        result = run_evaluate_scene(
            world["cfg"], world["analysis"], store, TEST_SCENE, FOLD, SEEDS, MODEL, TRAIN,
            world["center"], LEVEL, "cpu", run_dir, licence=LICENCE, ledger_dir=ledger,
        )
    assert result["status"] == "exists"
    assert [a["status"] for a in evaluation_attempts(ledger)] == ["exists"]


def _rewrite_run_record(world, evaluated, out: Path, **changes) -> dict:
    """A real parquet at out whose run record differs from the evaluated one."""
    from lot.evaluate import read_run_metadata

    meta = read_run_metadata(Path(evaluated["path"]))
    meta.update(changes)
    rows = read_rows(Path(evaluated["path"]))[:5]
    write_rows(out, rows, meta)
    return meta


@pytest.mark.parametrize("changes, named", [
    ({"checkpoints": {"0": "0" * 64, "1": "1" * 64}}, "checkpoints"),
    ({"training_records": {"0": {"sha256": "0" * 64}, "1": {"sha256": "1" * 64}}},
     "training records"),
    ({"scene": "room_1"}, "scene"),
    ({"level": "affine"}, "level"),
    ({"fold": 2}, "fold"),
    ({"seeds": [0]}, "seeds"),
    ({"config_digest": "0" * 64}, "config_digest"),
], ids=["other checkpoints", "other training records", "other scene", "other level",
        "other fold", "other seeds", "other configuration"])
def test_a_resumed_output_from_another_run_is_refused_and_kept(
    world, evaluated, tmp_path, changes, named
):
    """The retrain-then-relock path: an output written under other checkpoints
    is never credited to the current lock. The attempt is recorded as an error,
    and the file is left exactly as it was."""
    from lot.phase5_modes import evaluation_attempts

    # The training run the output must match, copied byte for byte.
    run_dir = tmp_path / "run"
    for seed in SEEDS:
        for path_of in (checkpoint_path, training_record_path):
            target = path_of(run_dir, LEVEL, FOLD.index, seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                path_of(Path(world["cfg"].run_dir), LEVEL, FOLD.index, seed).read_bytes()
            )
    out = run_dir / "eval" / LEVEL / f"{TEST_SCENE}.parquet"
    _rewrite_run_record(world, evaluated, out, **changes)
    before = out.read_bytes()
    ledger = tmp_path / "ledger"
    with pytest.raises(ValueError, match=named):
        run_evaluate_scene(world["cfg"], world["analysis"], None, TEST_SCENE, FOLD, SEEDS,
                           MODEL, TRAIN, world["center"], LEVEL, "cpu", run_dir,
                           licence=LICENCE, ledger_dir=ledger)
    assert out.read_bytes() == before
    (attempt,) = evaluation_attempts(ledger)
    assert attempt["status"] == "error" and named in attempt["message"]
    assert attempt["parquet_sha256"] == sha256_file(out)


@pytest.mark.parametrize("content, named", [
    (b"an earlier evaluation", "cannot be read"),
    (None, "no run record"),
], ids=["not a parquet", "no run record"])
def test_an_output_that_is_not_a_phase5_evaluation_is_refused(
    world, evaluated, tmp_path, content, named
):
    import pyarrow as pa

    out = tmp_path / "eval" / LEVEL / f"{TEST_SCENE}.parquet"
    out.parent.mkdir(parents=True)
    if content is None:
        pq.write_table(pa.Table.from_pylist(read_rows(Path(evaluated["path"]))[:3]), out)
    else:
        out.write_bytes(content)
    before = out.read_bytes()
    with pytest.raises(ValueError, match=named):
        run_evaluate_scene(world["cfg"], world["analysis"], None, TEST_SCENE, FOLD, SEEDS,
                           MODEL, TRAIN, world["center"], LEVEL, "cpu", tmp_path,
                           licence=LICENCE)
    assert out.read_bytes() == before


def _models(world):
    return {s: _load(world, s) for s in SEEDS}


def test_a_scene_outside_the_folds_test_set_is_refused(world, trained, store):
    reference = read_phase4_reference(Path(world["cfg"].phase4_dir) / "eval", TEST_SCENE, LEVEL)
    with pytest.raises(ValueError, match="not a test scene of fold"):
        evaluate_scene(world["cfg"], world["analysis"], store.get(TRAIN_SCENE), FOLD,
                       _models(world), MODEL, world["center"], reference, LEVEL, "cpu")


def test_a_tampered_phase4_mask_stops_evaluation(world, trained, store):
    reference = read_phase4_reference(Path(world["cfg"].phase4_dir) / "eval", TEST_SCENE, LEVEL)
    key, rows = next((k, v) for k, v in reference["pairs"].items() if v.pp_mask is not None
                     and v.pp_mask.any())
    flipped = rows.pp_mask.copy()
    flipped[np.flatnonzero(flipped)[0]] = False
    reference["pairs"][key] = dataclasses.replace(rows, pp_mask=flipped)
    with pytest.raises(ReferenceMismatch, match="scored set differs from Phase 4"):
        evaluate_scene(world["cfg"], world["analysis"], store.get(TEST_SCENE), FOLD,
                       _models(world), MODEL, world["center"], reference, LEVEL, "cpu")


def test_population_drift_between_phases_stops_evaluation(world, trained, store):
    reference = read_phase4_reference(Path(world["cfg"].phase4_dir) / "eval", TEST_SCENE, LEVEL)
    reference["all_pairs"] = set(reference["all_pairs"]) | {("ghost_ctx", "ghost_tgt")}
    with pytest.raises(ReferenceMismatch, match="not reading one"):
        evaluate_scene(world["cfg"], world["analysis"], store.get(TEST_SCENE), FOLD,
                       _models(world), MODEL, world["center"], reference, LEVEL, "cpu")


def test_sensitivity_runs_wait_for_the_primary_result(tmp_path):
    scenes = ["a", "b"]
    assert primary_evaluation_complete(tmp_path, "image", scenes) == ["a", "b"]
    (tmp_path / "eval" / "image").mkdir(parents=True)
    (tmp_path / "eval" / "image" / "a.parquet").write_bytes(b"")
    assert primary_evaluation_complete(tmp_path, "image", scenes) == ["b"]


# ---------------------------------------------------------------------------
# The command line is the boundary
# ---------------------------------------------------------------------------

def _config_file(world, tmp_path) -> Path:
    import yaml

    cfg = world["cfg"]
    raw = yaml.safe_load(Path("configs/phase5.yaml").read_text(encoding="utf-8"))
    raw.update(renders_root=cfg.renders_root, cache_root=cfg.cache_root,
               output_root=str(tmp_path / "cli"), phase4_dir=cfg.phase4_dir)
    path = tmp_path / "phase5.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", ["overfit", "train", "controls", "lock", "evaluate"])
def test_every_mode_refuses_without_its_gates(world, tmp_path, mode, capsys):
    from lot.phase5 import main

    with pytest.raises(SystemExit) as stop:
        main(["--config", str(_config_file(world, tmp_path)), "--mode", mode])
    assert "not permitted" in str(stop.value)
    err = capsys.readouterr().err
    assert "could not be read" in err
    # Only evaluate also needs the checkpoint lock, and it names its level's lock.
    assert (f"checkpoint_lock_{LEVEL}.json" in err) == (mode == "evaluate")


def test_an_undeclared_level_is_refused(world, tmp_path):
    from lot.phase5 import main

    with pytest.raises(SystemExit, match="not declared"):
        main(["--config", str(_config_file(world, tmp_path)), "--mode", "train",
              "--level", "scene"])


@pytest.fixture
def cli(world, tmp_path, monkeypatch):
    """The command line on the synthetic world, past its receipt check.

    Receipt verification is lot.phase5_receipt's job and is tested there. Here
    the receipts are placeholder files, so what is checked is what the command
    line records about them. The frozen architecture and training settings are
    replaced by the small ones the world was trained with, and the convention
    and centering vector by the world's own.
    """
    import lot.phase5 as phase5
    import lot.train as train_module

    path = _config_file(world, tmp_path)
    cfg = phase5.load_phase5_config(path)
    evidence = Path(cfg.evidence_dir)
    evidence.mkdir(parents=True)
    receipts = {}
    for stem in ("integration_gate", "tiny_overfit"):
        receipts[stem] = evidence / f"{stem}.json"
        receipts[stem].write_text(json.dumps({"placeholder": stem}), encoding="utf-8")
    monkeypatch.setattr(phase5, "require_receipts", lambda *args: None)
    monkeypatch.setattr(phase5, "load_convention_record", lambda c: world["convention"])
    monkeypatch.setattr(phase5, "phase5_mean_vector", lambda c: world["center"])
    monkeypatch.setattr(phase5, "predictor_config_from", lambda c, hw: MODEL)
    monkeypatch.setattr(train_module, "training_config_from", lambda raw: TRAIN)

    def run(*args):
        phase5.main(["--config", str(path), "--device", "cpu", *args])

    return {"cfg": cfg, "run": run, "receipts": receipts, "path": path}


def test_the_command_line_hands_training_the_receipts_that_licensed_it(cli, monkeypatch):
    import lot.phase5_modes as modes

    handed = {}

    def train(*args, **kwargs):
        handed.update(kwargs)
        return {"best_validation_centered_cosine": 0.5, "best_step": 2, "steps_run": 4,
                "checkpoint": "fold0_seed0.pt"}

    monkeypatch.setattr(modes, "run_train_task", train)
    cli["run"]("--mode", "train", "--task-index", "0")
    assert handed["licence"] == _licence_of(cli["receipts"])


def test_the_command_line_writes_the_controls_file_with_its_provenance(
    world, trained, cli, monkeypatch
):
    """Fold 0's two checkpoints are present and folds 1 and 2 are not, so the
    file lists them as missing and the job exits nonzero. The controls
    themselves are replaced: their numbers are tested above, and here only what
    the command line records around them is checked."""
    import lot.phase5_modes as modes
    from lot.phase5_folds import frozen_folds

    cfg = cli["cfg"]

    def controls(cfg_, analysis, store, fold, model, model_cfg, center, level,
                 batch_pairs, control_seed):
        assert not model.training
        return {"n_pairs": 2, "val_scenes_planned": [VAL_SCENE],
                "pose_shuffle": {"degradation": 0.1, "n_unchanged": 0},
                "depth_shuffle": {"degradation": 0.0, "n_unchanged": 2}}

    monkeypatch.setattr(modes, "run_controls", controls)

    # Trained under the world's configuration, which is not this one: refused.
    _copy_training_run(world, Path(cfg.run_dir))
    with pytest.raises(ValueError, match="config digest"):
        cli["run"]("--mode", "controls")

    hashes = _copy_training_run(world, Path(cfg.run_dir), config_digest=cfg.digest())
    with pytest.raises(SystemExit) as stop:
        cli["run"]("--mode", "controls")
    assert stop.value.code == 1
    payload = json.loads(
        (Path(cfg.evidence_dir) / f"input_use_controls_{LEVEL}.json").read_text()
    )
    assert set(payload) == set(CONTROLS_BASE_FIELDS) | set(CONTROLS_PROVENANCE)
    assert payload["licence"] == _licence_of(cli["receipts"])
    assert payload["checkpoints"] == hashes
    assert set(payload["results"]) == set(hashes)
    assert len(payload["missing"]) == (len(frozen_folds()) - 1) * len(SEEDS)
    assert payload["config_digest"] == cfg.digest()
    _assert_utc(payload["written_utc"])


def test_the_command_line_records_each_evaluation_attempt_with_its_licence(cli):
    """An output that is not this run's evaluation is never credited to the
    current lock. Each attempt on it is refused and recorded, a start and an
    error close, and the file is kept as it was. The checkpoint lock of the
    level licenses evaluation with the two gates, so every record's licence
    names it. Past the receipt check, which this fixture replaces, an absent
    lock still stops evaluation before any attempt begins."""
    from lot.phase5_folds import frozen_folds
    from lot.phase5_modes import (
        evaluation_attempts, evaluation_scenes, read_evaluation_ledger,
    )

    cfg = cli["cfg"]
    ledger = Path(cfg.evidence_dir) / "evaluation_ledger"
    index = str(evaluation_scenes(frozen_folds()).index(TEST_SCENE))
    out = Path(cfg.run_dir) / "eval" / LEVEL / f"{TEST_SCENE}.parquet"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"an earlier evaluation")

    lock = Path(cfg.evidence_dir) / f"checkpoint_lock_{LEVEL}.json"
    with pytest.raises(FileNotFoundError, match=lock.name):
        cli["run"]("--mode", "evaluate", "--scene-index", index)
    assert read_evaluation_ledger(ledger) == []

    lock.write_text(json.dumps({"placeholder": "lock"}), encoding="utf-8")
    for _ in range(2):
        with pytest.raises(ValueError, match="cannot be read"):
            cli["run"]("--mode", "evaluate", "--scene-index", index)

    records = read_evaluation_ledger(ledger)
    assert sorted(r["status"] for r in records) == ["error", "error", "started", "started"]
    licence = {**_licence_of(cli["receipts"]), lock.stem: sha256_file(lock)}
    for record in records:
        assert record["licence"] == licence
        assert (record["scene"], record["level"]) == (TEST_SCENE, LEVEL)
        assert record["parquet_sha256"] == sha256_file(out)
    assert [a["status"] for a in evaluation_attempts(ledger)] == ["error", "error"]
    assert out.read_bytes() == b"an earlier evaluation"


def test_the_command_line_turns_sigterm_into_a_recorded_error(cli, monkeypatch):
    """SLURM ends a task at its time limit with SIGTERM. Evaluate turns it into
    SystemExit, so the attempt's error close is written, and restores the
    previous handler on the way out."""
    import signal

    import lot.phase5_modes as modes
    from lot.phase5_folds import frozen_folds
    from lot.phase5_modes import evaluation_attempts, evaluation_scenes

    cfg = cli["cfg"]
    ledger = Path(cfg.evidence_dir) / "evaluation_ledger"
    index = str(evaluation_scenes(frozen_folds()).index(TEST_SCENE))
    lock = Path(cfg.evidence_dir) / f"checkpoint_lock_{LEVEL}.json"
    lock.write_text(json.dumps({"placeholder": "lock"}), encoding="utf-8")

    def attempt(*args):
        _raise_sigterm()
        raise AssertionError("SIGTERM did not stop the attempt")

    monkeypatch.setattr(modes, "_evaluate_scene_attempt", attempt)
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit, match="evaluation stopped by SIGTERM"):
        cli["run"]("--mode", "evaluate", "--scene-index", index)
    assert signal.getsignal(signal.SIGTERM) == before
    (attempt,) = evaluation_attempts(ledger)
    assert attempt["status"] == "error"
    assert attempt["message"] == "SystemExit: evaluation stopped by SIGTERM"


def _placeholder_controls(cfg, analysis, store, fold, model, model_cfg, center, level,
                          batch_pairs, control_seed):
    """Stands in for run_controls, whose numbers are tested above."""
    assert not model.training
    return {"n_pairs": 2, "val_scenes_planned": [VAL_SCENE],
            "pose_shuffle": {"degradation": 0.1, "n_unchanged": 0},
            "depth_shuffle": {"degradation": 0.0, "n_unchanged": 2}}


def test_the_command_line_locks_what_training_and_the_controls_wrote(
    world, trained, cli, monkeypatch
):
    """The chain on fold 0, each step through the command line: training's
    checkpoints and records, then the controls, then the lock. With the three
    frozen folds expected, folds 1 and 2 were never trained, so the lock is
    refused and names them. On fold 0 alone it holds, and the receipt it writes
    is the one the verifier accepts. A rerun keeps the earlier lock."""
    import lot.phase5 as phase5
    import lot.phase5_folds as phase5_folds
    import lot.phase5_modes as modes
    from lot.phase5_receipt import (
        BOUND_FIELDS,
        GATE_RECEIPT_DIGEST,
        KIND_LOCK,
        OVERFIT_RECEIPT_DIGEST,
        current_identity,
        verify,
    )

    cfg, receipts = cli["cfg"], cli["receipts"]
    lock = Path(cfg.evidence_dir) / f"checkpoint_lock_{LEVEL}.json"
    controls = Path(cfg.evidence_dir) / f"input_use_controls_{LEVEL}.json"
    monkeypatch.setattr(modes, "run_controls", _placeholder_controls)
    # Trained under the receipts this command line verifies, as training at
    # one commit would be.
    hashes = _copy_training_run(world, Path(cfg.run_dir), config_digest=cfg.digest(),
                                licence=_licence_of(receipts))

    with pytest.raises(SystemExit) as stop:
        cli["run"]("--mode", "controls")
    assert stop.value.code == 1
    with pytest.raises(SystemExit, match="no checkpoint for fold 1 seed 0") as refused:
        cli["run"]("--mode", "lock")
    assert "lists missing checkpoints" in str(refused.value)
    assert not lock.exists()

    # The same chain over fold 0 alone. The fold digest follows the folds, so
    # the receipt's identity and the verifier's agree.
    monkeypatch.setattr(phase5, "frozen_folds", lambda: [FOLD])
    monkeypatch.setattr(phase5_folds, "frozen_folds", lambda: [FOLD])
    with pytest.raises(SystemExit) as stop:
        cli["run"]("--mode", "controls")
    assert stop.value.code == 0
    cli["run"]("--mode", "lock")

    report = json.loads(lock.read_text(encoding="utf-8"))
    assert report["kind"] == KIND_LOCK and report["passed"] is True
    assert report["level"] == LEVEL
    assert {key: entry["sha256"] for key, entry in report["checkpoints"].items()} == hashes
    assert report["controls"]["sha256"] == sha256_file(controls)
    assert report[GATE_RECEIPT_DIGEST] == sha256_file(receipts["integration_gate"])
    assert report[OVERFIT_RECEIPT_DIGEST] == sha256_file(receipts["tiny_overfit"])
    assert {field: report[field] for field in BOUND_FIELDS} == current_identity(cli["path"])
    _assert_utc(report["stamped_utc"])

    def problems():
        return verify(lock, cli["path"], "the checkpoint lock", kind=KIND_LOCK,
                      gate_receipt=receipts["integration_gate"],
                      overfit_receipt=receipts["tiny_overfit"], level=LEVEL)

    assert problems() == []
    first = lock.read_bytes()
    cli["run"]("--mode", "lock")
    assert (lock.parent / f"checkpoint_lock_{LEVEL}.superseded.1.json").read_bytes() == first
    assert problems() == []

    record = training_record_path(Path(cfg.run_dir), LEVEL, FOLD.index, SEEDS[1])
    record.write_text(record.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert any(f"training record fold{FOLD.index}_seed{SEEDS[1]} changed" in p
               for p in problems())


def test_the_command_line_refuses_a_lock_after_a_gate_rerun(world, trained, cli, monkeypatch):
    """Training and the controls ran under the first receipts. The overfit gate
    is then rerun, as a rerun of check would force. The lock would bind the new
    receipts, which training never ran under, so it is refused, naming every
    training record and the controls file, and nothing is written. Writing the
    lock again cannot repair this. Training and the controls must rerun under
    the new receipts first."""
    import lot.phase5 as phase5
    import lot.phase5_folds as phase5_folds
    import lot.phase5_modes as modes
    from lot.phase5_check import write_once

    cfg, receipts = cli["cfg"], cli["receipts"]
    lock = Path(cfg.evidence_dir) / f"checkpoint_lock_{LEVEL}.json"
    monkeypatch.setattr(modes, "run_controls", _placeholder_controls)
    monkeypatch.setattr(phase5, "frozen_folds", lambda: [FOLD])
    monkeypatch.setattr(phase5_folds, "frozen_folds", lambda: [FOLD])
    _copy_training_run(world, Path(cfg.run_dir), config_digest=cfg.digest(),
                       licence=_licence_of(receipts))
    with pytest.raises(SystemExit) as stop:
        cli["run"]("--mode", "controls")
    assert stop.value.code == 0

    write_once(receipts["tiny_overfit"], json.dumps({"placeholder": "overfit rerun"}))
    with pytest.raises(SystemExit, match="licence") as refused:
        cli["run"]("--mode", "lock")
    for seed in SEEDS:
        assert (f"fold{FOLD.index}_seed{seed}: its training record names licence"
                in str(refused.value))
    assert "the controls file names licence" in str(refused.value)
    assert not lock.exists()


def _stamp_gates(config: Path) -> dict[str, Path]:
    """Integration and overfit receipts stamped for config, in its evidence directory."""
    from lot.phase5 import load_phase5_config
    from lot.phase5_receipt import KIND_INTEGRATION, KIND_OVERFIT, stamp_receipt

    evidence = Path(load_phase5_config(config).evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    gate = evidence / "integration_gate.json"
    gate.write_text(json.dumps(stamp_receipt({"passed": True}, config, KIND_INTEGRATION)),
                    encoding="utf-8")
    overfit = evidence / "tiny_overfit.json"
    overfit.write_text(json.dumps(stamp_receipt({"passed": True}, config, KIND_OVERFIT,
                                                gate_receipt=gate)), encoding="utf-8")
    return {"gate": gate, "overfit": overfit}


def test_evaluate_refuses_without_a_lock_that_matches_the_live_files(
    world, tmp_path, monkeypatch, capsys
):
    """reporting_rules.md section 7, through the real receipt check with real
    stamped receipts. Only the integration receipt's binding to the resolved
    inputs is replaced: the synthetic world cannot hold the eighteen scenes it
    hashes, and tests/test_phase5_receipt.py covers it. With both gates
    standing, every mode before evaluate is permitted, and evaluate is refused
    for want of its lock. A lock over the live files permits it. A locked
    checkpoint replaced afterwards refuses it again, by name."""
    import lot.phase5_receipt as receipt_module
    from lot.phase5 import load_phase5_config, main, require_receipts
    from lot.phase5_receipt import KIND_LOCK, stamp_receipt
    from test_phase5_receipt import write_locked_files

    monkeypatch.setattr(receipt_module, "_artifact_problems", lambda *args: [])
    config = _config_file(world, tmp_path)
    cfg = load_phase5_config(config)
    gates = _stamp_gates(config)
    for mode in ("train", "controls", "lock"):
        require_receipts(cfg, config, mode, LEVEL)

    lock = Path(cfg.evidence_dir) / f"checkpoint_lock_{LEVEL}.json"
    with pytest.raises(SystemExit, match="not permitted"):
        main(["--config", str(config), "--mode", "evaluate"])
    err = capsys.readouterr().err
    refusals = [line for line in err.splitlines() if "could not be read" in line]
    assert len(refusals) == 1 and lock.name in refusals[0], err
    assert "integration gate" not in err and "overfit gate" not in err, err

    files = write_locked_files(config, LEVEL)
    report = stamp_receipt({"passed": True, "level": LEVEL, **files}, config, KIND_LOCK,
                           gate_receipt=gates["gate"], overfit_receipt=gates["overfit"])
    lock.write_text(json.dumps(report), encoding="utf-8")
    require_receipts(cfg, config, "evaluate", LEVEL)

    checkpoint = Path(files["checkpoints"]["fold2_seed1"]["path"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    with pytest.raises(SystemExit, match="not permitted"):
        require_receipts(cfg, config, "evaluate", LEVEL)
    assert "checkpoint fold2_seed1 changed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Integration gate step 8, driven on the synthetic world
# ---------------------------------------------------------------------------
#
# The first real run of the gate stopped at step 8 on cell 1369 of a 37 by 37
# grid. Step 8 mapped every context patch to a target cell, including patches
# that never landed, and patch_cell_index does not bound-check: a patch just
# past the right edge wraps into the next row's first cell, and only the last
# row's overflow leaves the grid. These tests are written from that failure.

def _lift_and_support(world, store, pair):
    from lot.context_lift import context_lift_map, context_lift_support
    from lot.phase5 import aligned_context_depth, pair_cameras
    from lot.phase5_score import primary_support

    cfg, analysis = world["cfg"], world["analysis"]
    inputs = store.get(TEST_SCENE)
    cams = pair_cameras(cfg, inputs, pair)
    dtype = cfg.torch_dtype
    gt_c = inputs.cache.depth(cams.context.depth_path).to(dtype)
    gt_t = inputs.cache.depth(cams.target.depth_path).to(dtype)
    depth = aligned_context_depth(inputs, pair.context_frame_id, LEVEL)
    lift = context_lift_map(
        torch.from_numpy(depth).to(dtype), cams.K_context, cams.K_target,
        cams.T_target_from_context, cams.context_hw, cams.target_hw,
    )
    support = primary_support(lift, context_lift_support(
        lift, gt_c, gt_t, cams.K_context, cams.K_target, cams.T_target_from_context,
        rel_tol=analysis.covisible_relative_depth_tol,
    ))
    return inputs, cams, gt_c, gt_t, lift, support


def _overflowing_pair(world, store):
    """A supported pair whose unlanded edge patches map outside the target grid."""
    from lot.encoders import patch_cell_index

    for pair in phase5_scene_pairs(world["cfg"], world["analysis"], TEST_SCENE):
        _, cams, _, _, lift, support = _lift_and_support(world, store, pair)
        if not support.any():
            continue
        n_cells = (cams.target_hw[0] // 14) * (cams.target_hw[1] // 14)
        every = patch_cell_index(lift.uv_target, cams.target_hw, 14)
        if ((every < 0) | (every >= n_cells)).any():
            return pair
    raise AssertionError("the fixture must hold a supported pair with unlanded overflow")


def _step8_arguments(world, store, pair):
    """Gate step 8's inputs for one pair, assembled as evaluate assembles them."""
    from lot.phase5_reference import read_phase4_reference, recompute_reference_arms

    cfg, analysis = world["cfg"], world["analysis"]
    inputs, cams, gt_c, gt_t, lift, support = _lift_and_support(world, store, pair)
    ctx, tgt = pair.context_frame_id, pair.target_frame_id
    fc = inputs.cache.features(cfg.feature_encoder, ctx)
    ft = inputs.cache.features(cfg.feature_encoder, tgt)
    arms = recompute_reference_arms(
        gt_c, gt_t, inputs.est_maps[ctx], inputs.est_maps[tgt], inputs.calibrations[ctx],
        fc, ft, cams.K_context, cams.K_target, cams.T_target_from_context,
        TEST_SCENE, ctx, tgt, analysis, LEVEL, cfg.torch_dtype,
    )
    reference = read_phase4_reference(Path(cfg.phase4_dir) / "eval", TEST_SCENE, LEVEL)
    return {
        "lift": lift, "support": support, "arms": arms,
        "persisted": reference["pairs"].get((ctx, tgt)),
        "center": world["center"], "features_context": fc,
        "target_hw": cams.target_hw, "where": f"{TEST_SCENE} {ctx} -> {tgt}",
    }


def test_gate_step_eight_maps_only_the_supported_samples(world, store, evaluated):
    """The pair overflows the grid with unlanded patches, and step 8 now passes.

    Its report is the formulation evaluate recorded for the same pair, because
    step 8 runs the same functions on the same inputs.
    """
    from lot.phase5_gate import formulation_support_evidence

    pair = _overflowing_pair(world, store)
    args = _step8_arguments(world, store, pair)
    evidence = formulation_support_evidence(**args)
    assert evidence["cl_supported_samples"] == int(args["support"].sum())
    assert evidence["common_cells"] > 0
    record = _record(evaluated, (pair.context_frame_id, pair.target_frame_id))
    assert evidence["common_cells"] == record["n_formulation"]
    assert evidence["tl_form_centered"] == pytest.approx(record["tl_form_centered"], abs=1e-12)
    assert evidence["cl_form_centered"] == pytest.approx(record["cl_form_centered"], abs=1e-12)


def test_gate_step_eight_stops_when_tl_reference_disagrees_with_phase4(world, store):
    from lot.phase5_check import IMPLEMENTATION_BUG, GateStop
    from lot.phase5_gate import formulation_support_evidence

    args = _step8_arguments(world, store, _overflowing_pair(world, store))
    persisted = args["persisted"]
    args["persisted"] = dataclasses.replace(persisted, pp_mask=~persisted.pp_mask)
    with pytest.raises(GateStop) as stop:
        formulation_support_evidence(**args)
    assert stop.value.step == "8"
    assert stop.value.classification == IMPLEMENTATION_BUG
    assert "Phase 4" in stop.value.message


def test_gate_step_eight_stops_when_a_target_cell_is_ambiguous(world, store):
    """Two TL-Reference samples in one target cell leave the intersection undefined."""
    from lot.phase5_check import IMPLEMENTATION_BUG, GateStop
    from lot.phase5_gate import formulation_support_evidence

    args = _step8_arguments(world, store, _overflowing_pair(world, store))
    arms = args["arms"]
    cells = np.array(arms.geometry.per_point_cells, copy=True)
    cells[1] = cells[0]
    args["arms"] = dataclasses.replace(
        arms, geometry=dataclasses.replace(arms.geometry, per_point_cells=cells)
    )
    with pytest.raises(GateStop) as stop:
        formulation_support_evidence(**args)
    assert stop.value.step == "8"
    assert stop.value.classification == IMPLEMENTATION_BUG
    assert "uniquely" in stop.value.message


def test_gate_step_eight_runs_the_tested_function():
    """The closure in run_integration_gate must call what these tests drive."""
    import inspect

    from lot import phase5_gate

    source = inspect.getsource(phase5_gate.run_integration_gate)
    step8 = source[source.index("def step8"):source.index("def step9")]
    assert "formulation_support_evidence(" in step8
    assert "patch_cell_index(lift.uv_target," not in step8


# ---------------------------------------------------------------------------
# Gate step 17: the pure-rotation gate across the rotation regime
# ---------------------------------------------------------------------------
#
# The fixture's rotation frames share one camera position, so its rotation
# pairs are pure rotations. Step 17's per-pair function runs here on each of
# them with TL-Reference recomputed by the call evaluate makes.

COMPARISONS = ("context_lift", "depth_substitution", "tl_reference")


def _rotation_pair_arguments(world, store, pair):
    """Step 17's inputs for one pair, assembled as evaluate assembles them."""
    from lot.context_lift import context_lift_map
    from lot.phase5 import aligned_context_depth
    from lot.phase5_gate import ROTATION_SUBSTITUTE_DEPTH_M
    from lot.phase5_reference import recompute_reference_arms

    cfg, analysis = world["cfg"], world["analysis"]
    inputs, cams, gt_c, gt_t, lift, support = _lift_and_support(world, store, pair)
    ctx, tgt = pair.context_frame_id, pair.target_frame_id
    dtype = cfg.torch_dtype
    fc = inputs.cache.features(cfg.feature_encoder, ctx)
    ft = inputs.cache.features(cfg.feature_encoder, tgt)
    arms = recompute_reference_arms(
        gt_c, gt_t, inputs.est_maps[ctx], inputs.est_maps[tgt], inputs.calibrations[ctx],
        fc, ft, cams.K_context, cams.K_target, cams.T_target_from_context,
        TEST_SCENE, ctx, tgt, analysis, LEVEL, dtype,
    )
    depth = torch.from_numpy(aligned_context_depth(inputs, ctx, LEVEL)).to(dtype)
    substituted = context_lift_map(
        torch.full_like(depth, ROTATION_SUBSTITUTE_DEPTH_M),
        cams.K_context, cams.K_target, cams.T_target_from_context,
        cams.context_hw, cams.target_hw,
    )
    args = {
        "lift": lift, "substituted": substituted,
        "tl_read_uv_context": arms.tl_read_uv_context,
        "tl_read_depth_context": arms.tl_read_depth_context,
        "tl_landed": arms.tl_landed,
        "uv_target_samples": arms.geometry.samples.uv_target,
        "K_context": cams.K_context, "K_target": cams.K_target,
        "T_target_from_context": cams.T_target_from_context,
        "context_hw": cams.context_hw,
        "tol_px": analysis.rotation_gate_coord_tol_px,
        "position_bound_m": analysis.rotation_position_bound_m,
        "where": f"{TEST_SCENE} {ctx} -> {tgt} level {LEVEL}",
    }
    return args, arms, support, fc


def _rotation_pairs(world):
    pairs = phase5_scene_pairs(world["cfg"], world["analysis"], TEST_SCENE)
    return [p for p in pairs if p.regime == "rotation"]


def test_reference_arms_expose_where_tl_reference_reads(world, store):
    """The new fields are the locations tl_reads came from, not a second computation."""
    from lot.correspondence import _in_box, _sampling_box
    from lot.encoders import PATCH_SIZE, sample_features_bilinear

    seen = 0
    for pair in phase5_scene_pairs(world["cfg"], world["analysis"], TEST_SCENE):
        args, arms, _, fc = _rotation_pair_arguments(world, store, pair)
        n = arms.reads_target.shape[0]
        assert arms.tl_read_uv_context.shape == (n, 2)
        assert arms.tl_read_depth_context.shape == (n,)
        assert torch.equal(
            sample_features_bilinear(fc, arms.tl_read_uv_context), arms.tl_reads
        )
        landed = torch.from_numpy(arms.tl_landed)
        box = _sampling_box(args["context_hw"], PATCH_SIZE)
        assert bool((arms.tl_read_depth_context[landed] > 0).all())
        assert bool(_in_box(arms.tl_read_uv_context[landed], box).all())
        seen += int(landed.sum())
    assert seen > 0


def test_step_seventeen_passes_every_supported_rotation_pair_of_the_fixture(world, store):
    from lot.phase5_gate import rotation_pair_evidence

    analysis = world["analysis"]
    tol = analysis.rotation_gate_coord_tol_px
    checked = 0
    for pair in _rotation_pairs(world):
        args, arms, support, _ = _rotation_pair_arguments(world, store, pair)
        if not bool(support.any()):
            continue
        translation = float(torch.linalg.vector_norm(args["T_target_from_context"][:3, 3]))
        assert translation <= analysis.rotation_position_bound_m, "not a pure rotation"
        evidence = rotation_pair_evidence(**args)
        assert evidence["checked"] is True
        assert evidence["context_lift"]["n_compared"] == int(args["lift"].landed.sum())
        assert evidence["depth_substitution"]["n_compared"] == int(args["lift"].landed.sum())
        assert evidence["tl_reference"]["n_compared"] == int(arms.tl_landed.sum()) > 0
        for name in COMPARISONS:
            assert evidence[name]["max_residual_px"] <= tol, (pair, name)
        checked += 1
    assert checked > 0, "the fixture must hold a supported rotation pair"


def test_step_seventeen_stops_a_tl_reference_read_in_the_wrong_direction(world, store):
    """TL-Reference reading where the forward map sends its sample must stop.

    The fault is injected into the read locations only, so Context-Lift's two
    comparisons pass first and the stop names TL-Reference.
    """
    from lot.geometry import apply_homography, rotation_homography
    from lot.phase5_check import IMPLEMENTATION_BUG, GateStop
    from lot.phase5_gate import rotation_pair_evidence

    for pair in _rotation_pairs(world):
        args, arms, support, _ = _rotation_pair_arguments(world, store, pair)
        if support.any() and arms.tl_landed.any():
            break
    else:
        raise AssertionError("the fixture must hold a supported rotation pair")
    T = args["T_target_from_context"]
    forward = rotation_homography(args["K_context"], args["K_target"], T[:3, :3])
    args["tl_read_uv_context"] = apply_homography(forward, args["uv_target_samples"])
    with pytest.raises(GateStop) as stop:
        rotation_pair_evidence(**args)
    assert stop.value.step == "17"
    assert stop.value.classification == IMPLEMENTATION_BUG
    assert stop.value.evidence["comparison"] == "tl_reference"
    assert "TL-Reference" in stop.value.message
    assert args["where"] in stop.value.message


def test_step_seventeen_checks_every_rotation_pair_of_the_scene(world, store):
    from lot.phase5_gate import rotation_regime_summary, rotation_scene_evidence

    analysis = world["analysis"]
    tol = analysis.rotation_gate_coord_tol_px
    scene = rotation_scene_evidence(world["cfg"], analysis, store.get(TEST_SCENE))

    # The expected account, from each estimator's own landed flags. The
    # fixture writes one depth map for every frame, which is not consistent
    # under rotation, so Phase 3 draws no sample from its wider rotation
    # pairs. TL-Reference then has nothing to compare there, while
    # Context-Lift still lands and is still checked.
    both = cl_only = 0
    for pair in _rotation_pairs(world):
        args, arms, _, _ = _rotation_pair_arguments(world, store, pair)
        cl, tl = bool(args["lift"].landed.any()), bool(arms.tl_landed.any())
        both += cl and tl
        cl_only += cl and not tl
    n_rotation = len(_rotation_pairs(world))
    assert scene["n_rotation_pairs"] == n_rotation > 0
    assert scene["n_pairs_no_arm"] == 0
    assert scene["n_pairs_checked"] == both > 0
    assert scene["n_pairs_context_lift_only"] == cl_only
    assert scene["n_pairs_tl_reference_only"] == 0
    assert scene["n_pairs_nothing_landed"] == n_rotation - both - cl_only
    assert scene["max_translation_norm_m"] <= analysis.rotation_position_bound_m
    summary = rotation_regime_summary(
        {TEST_SCENE: scene}, tol, analysis.rotation_position_bound_m
    )
    assert summary["n_rotation_pairs"] == n_rotation
    assert summary["n_pairs_checked"] == both
    for name in COMPARISONS:
        assert summary["worst"][name]["max_residual_px"] <= tol
        assert summary["worst"][name]["pair"].startswith(TEST_SCENE)
    json.dumps(summary)
