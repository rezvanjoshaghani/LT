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
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from lot.analysis_config import load_analysis_config
from lot.encoders import cache_dir
from lot.evaluate import read_rows, write_rows
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
                train_scenes=[TRAIN_SCENE], val_scenes=[VAL_SCENE],
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


def test_the_selected_checkpoint_loads_and_matches_its_record(world, trained):
    model = load_checkpoint(Path(world["cfg"].run_dir), LEVEL, FOLD, 0, MODEL, TRAIN, "cpu")
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
        load_checkpoint(run_dir, LEVEL, FOLD, 0, MODEL, TRAIN, "cpu")


def test_a_checkpoint_without_a_record_is_refused(world, trained, tmp_path):
    source = checkpoint_path(Path(world["cfg"].run_dir), LEVEL, FOLD.index, 0)
    target = checkpoint_path(tmp_path, LEVEL, FOLD.index, 0)
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    with pytest.raises(FileNotFoundError, match="did not finish"):
        load_checkpoint(tmp_path, LEVEL, FOLD, 0, MODEL, TRAIN, "cpu")


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

def test_controls_pool_both_shuffles_over_the_validation_set(world, trained, store):
    model = load_checkpoint(Path(world["cfg"].run_dir), LEVEL, FOLD, 0, MODEL, TRAIN, "cpu")
    out = run_controls(world["cfg"], world["analysis"], store, FOLD, model, MODEL,
                       world["center"], LEVEL, 4, 20260830, val_scenes=[VAL_SCENE])
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
    model = load_checkpoint(Path(world["cfg"].run_dir), LEVEL, FOLD, 0, MODEL, TRAIN, "cpu")
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

    model = load_checkpoint(Path(world["cfg"].run_dir), LEVEL, FOLD, 0, MODEL, TRAIN, "cpu")
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
    model = load_checkpoint(Path(world["cfg"].run_dir), LEVEL, FOLD, 0, MODEL, TRAIN, "cpu")
    out = run_controls(world["cfg"], world["analysis"], store, FOLD, model, MODEL,
                       world["center"], LEVEL, 4, 20260830, val_scenes=[VAL_SCENE])
    assert out["depth_shuffle"]["n_unchanged"] == out["n_pairs"]
    assert out["depth_shuffle"]["degradation"] == 0.0
    assert out["pose_shuffle"]["n_unchanged"] < out["n_pairs"]


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def evaluated(world, trained):
    with SceneStore(world["cfg"], world["analysis"], world["convention"]) as store:
        result = run_evaluate_scene(
            world["cfg"], world["analysis"], store, TEST_SCENE, FOLD, SEEDS, MODEL,
            TRAIN, world["center"], LEVEL, "cpu", Path(world["cfg"].run_dir),
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
                       "offset_cl_oracle_centered_all", "offset_nowarp_centered_b1"):
            # NaN is never equal to itself, so it is mapped to one marker; a pair
            # with no primary support has NaN explicit scores under every seed.
            values = {
                ("nan" if isinstance(r[column], float) and math.isnan(r[column])
                 else round(r[column], 12) if isinstance(r[column], float)
                 else r[column])
                for r in records
            }
            assert len(values) == 1, column


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
        assert record["cl_centered"] == pytest.approx(cl, abs=1e-4), item["key"]
        assert record["nowarp_centered"] == pytest.approx(nowarp, abs=1e-4), item["key"]


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
                cams.target_hw,
            )
            assert formulation.n_formulation == record["n_formulation"], item["key"]
            assert record["tl_form_centered"] == pytest.approx(
                formulation.tl_form_centered, abs=1e-4), item["key"]
            assert record["cl_form_centered"] == pytest.approx(
                formulation.cl_form_centered, abs=1e-6), item["key"]
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


def test_evaluation_resumes_instead_of_overwriting(world, evaluated):
    before = Path(evaluated["path"]).read_bytes()
    with SceneStore(world["cfg"], world["analysis"], world["convention"]) as store:
        again = run_evaluate_scene(
            world["cfg"], world["analysis"], store, TEST_SCENE, FOLD, SEEDS, MODEL,
            TRAIN, world["center"], LEVEL, "cpu", Path(world["cfg"].run_dir),
        )
    assert again["status"] == "exists"
    assert Path(evaluated["path"]).read_bytes() == before


def _models(world):
    return {s: load_checkpoint(Path(world["cfg"].run_dir), LEVEL, FOLD, s, MODEL, TRAIN, "cpu")
            for s in SEEDS}


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


@pytest.mark.parametrize("mode", ["overfit", "train", "controls", "evaluate"])
def test_every_mode_refuses_without_its_gates(world, tmp_path, mode, capsys):
    from lot.phase5 import main

    with pytest.raises(SystemExit) as stop:
        main(["--config", str(_config_file(world, tmp_path)), "--mode", mode])
    assert "not permitted" in str(stop.value)
    assert "could not be read" in capsys.readouterr().err


def test_an_undeclared_level_is_refused(world, tmp_path):
    from lot.phase5 import main

    with pytest.raises(SystemExit, match="not declared"):
        main(["--config", str(_config_file(world, tmp_path)), "--mode", "train",
              "--level", "scene"])
