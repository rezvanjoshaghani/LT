"""PLAN Phase 3: value-level transportability, its floors, and the results table."""

import json
import math

import numpy as np
import pytest
import torch

from lot.evaluate import (
    MEAN_FEATURE,
    NO_WARP_COPY,
    ORACLE_TRANSPORT,
    PER_POINT,
    SPLAT_POOL,
    EvalConfig,
    agreement_metrics,
    dataset_mean_feature_map,
    evaluate_pair_for_encoder,
    evaluate_scene,
    load_eval_config,
    pair_geometry,
    read_rows,
    unit_normalize,
    value_agreement,
    write_rows,
)
from lot.render_replica import (
    FrameRecord,
    Manifest,
    intrinsics_from_hfov,
    program_rotation,
    program_translation,
    write_frame_stats,
    write_manifest,
)
from scenes import GRID, build_two_plane_scene, patch_codes
from test_render_replica import base_pose


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_value_agreement_endpoints():
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert value_agreement(a, a) == (pytest.approx(1.0), pytest.approx(0.0))
    orthogonal = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    cosine, l2 = value_agreement(a, orthogonal)
    assert cosine == pytest.approx(0.0)
    assert l2 == pytest.approx(math.sqrt(2.0))
    opposite = -a
    cosine, l2 = value_agreement(a, opposite)
    assert cosine == pytest.approx(-1.0)
    assert l2 == pytest.approx(2.0)


def test_value_agreement_ignores_magnitude():
    """Metrics are on unit-normalized features, so scale must not show up."""
    g = torch.Generator().manual_seed(0)
    a = torch.rand((32, 8), generator=g)
    b = torch.rand((32, 8), generator=g)
    plain = value_agreement(a, b)
    scaled = value_agreement(a * 17.0, b * 0.03)
    assert plain[0] == pytest.approx(scaled[0], abs=1e-6)
    assert plain[1] == pytest.approx(scaled[1], abs=1e-6)


def test_value_agreement_on_nothing_is_nan_not_an_error():
    """A pair with no co-visible surface is a result to record, not a crash."""
    empty = torch.zeros((0, 4))
    cosine, l2 = value_agreement(empty, empty)
    assert math.isnan(cosine) and math.isnan(l2)


def test_unit_normalize_leaves_zero_vectors_alone():
    out = unit_normalize(torch.zeros((2, 3)))
    assert torch.equal(out, torch.zeros((2, 3)))


# ---------------------------------------------------------------------------
# Both paths on the analytic scene, where the right answer is known exactly
# ---------------------------------------------------------------------------

def analytic_pair(seed=0):
    """The two-plane scene with random features and their exact transport.

    scenes.py records which context patch column each target patch column draws
    from, so the target feature map can be built exactly for any context map.
    That makes Oracle-Transport's correct score exactly 1.0 and gives the floors
    something to be worse than.
    """
    scene = build_two_plane_scene()
    generator = torch.Generator().manual_seed(seed)
    context = torch.rand((16, GRID, GRID), generator=generator) - 0.5
    target = torch.zeros_like(context)
    for column in range(GRID):
        source = int(scene.source_patch_col[column])
        if source >= 0:
            target[:, :, column] = context[:, :, source]
    return scene, context, target


def test_oracle_transport_is_exact_on_the_analytic_scene():
    scene, context, target = analytic_pair()
    geometry = pair_geometry(
        scene.depth_context,
        scene.depth_target,
        scene.K,
        scene.K,
        scene.T_target_from_context,
        points_per_pair=256,
        min_covisible_fraction=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    rows = evaluate_pair_for_encoder(geometry, context, target, torch.zeros_like(context))
    scores = {(r["path"], r["variant"]): r for r in rows}

    splat = scores[(SPLAT_POOL, ORACLE_TRANSPORT)]
    assert splat["n"] > 0
    assert splat["cosine_mean"] == pytest.approx(1.0, abs=1e-5)
    assert splat["l2_mean"] == pytest.approx(0.0, abs=1e-3)

    point = scores[(PER_POINT, ORACLE_TRANSPORT)]
    assert point["n"] > 0
    assert point["cosine_mean"] == pytest.approx(1.0, abs=1e-5)


def test_pixel_sampling_scores_worse_than_the_truth_it_is_measuring():
    """Why the per-point path samples patch centers rather than arbitrary pixels.

    At an arbitrary pixel the target value is a bilinear blend of patches, and
    near a depth edge those patches have different correspondences, so the blend
    cannot be reproduced from any single context location. The exact answer here
    is 1.0, so anything below it is the sampler's error, not the encoder's.
    """
    scene, context, target = analytic_pair()
    scores = {}
    for mode in ("patch_center", "pixel"):
        geometry = pair_geometry(
            scene.depth_context,
            scene.depth_target,
            scene.K,
            scene.K,
            scene.T_target_from_context,
            points_per_pair=256,
            min_covisible_fraction=0.5,
            sample_mode=mode,
            generator=torch.Generator().manual_seed(0),
        )
        rows = evaluate_pair_for_encoder(geometry, context, target, torch.zeros_like(context))
        scores[mode] = next(
            r["cosine_mean"]
            for r in rows
            if r["path"] == PER_POINT and r["variant"] == ORACLE_TRANSPORT
        )
    assert scores["patch_center"] == pytest.approx(1.0, abs=1e-5)
    assert scores["pixel"] < 0.99


def test_the_floors_are_beaten_by_transport():
    """Without this the numbers mean nothing: random features share no direction."""
    scene, context, target = analytic_pair()
    geometry = pair_geometry(
        scene.depth_context,
        scene.depth_target,
        scene.K,
        scene.K,
        scene.T_target_from_context,
        points_per_pair=256,
        min_covisible_fraction=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    rows = evaluate_pair_for_encoder(geometry, context, target, torch.zeros_like(context))
    scores = {(r["path"], r["variant"]): r["cosine_mean"] for r in rows}
    for path in (PER_POINT, SPLAT_POOL):
        assert scores[(path, ORACLE_TRANSPORT)] > scores[(path, NO_WARP_COPY)] + 0.5


def test_every_variant_is_reported_on_both_paths():
    scene, context, target = analytic_pair()
    geometry = pair_geometry(
        scene.depth_context,
        scene.depth_target,
        scene.K,
        scene.K,
        scene.T_target_from_context,
        points_per_pair=128,
        min_covisible_fraction=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    rows = evaluate_pair_for_encoder(geometry, context, target, torch.zeros_like(context))
    # CLAUDE.md requires both floors beside every reported metric.
    for path in (PER_POINT, SPLAT_POOL):
        variants = {r["variant"] for r in rows if r["path"] == path}
        assert {ORACLE_TRANSPORT, NO_WARP_COPY, MEAN_FEATURE} <= variants


def test_splat_path_is_scored_only_where_the_splat_landed():
    """A hole must not be scored: its transported feature is zero, not a prediction."""
    scene, context, target = analytic_pair()
    geometry = pair_geometry(
        scene.depth_context,
        scene.depth_target,
        scene.K,
        scene.K,
        scene.T_target_from_context,
        points_per_pair=128,
        min_covisible_fraction=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    # The analytic scene has four empty target patch columns.
    assert not geometry.patch_selection[:, [5, 6, 14, 15]].any()
    assert geometry.patch_selection[:, [0, 1, 2, 3, 4, 7, 8]].all()
    assert geometry.coverage_mean == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Dataset mean feature map
# ---------------------------------------------------------------------------

def test_dataset_mean_feature_map_averages_every_frame(tmp_path):
    from lot.encoders import cache_dir

    directory = cache_dir(tmp_path, "dinov2_vitb14", "room_0")
    directory.mkdir(parents=True)
    arrays = {
        "a": np.zeros((3, 2, 2), dtype=np.float16),
        "b": np.full((3, 2, 2), 2.0, dtype=np.float16),
    }
    np.savez(directory / "features.npz", **arrays)
    mean = dataset_mean_feature_map(tmp_path, "dinov2_vitb14", ["room_0"])
    assert mean.shape == (3, 2, 2)
    assert torch.allclose(mean, torch.ones((3, 2, 2)))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def base_config(tmp_path, **overrides):
    values = dict(
        experiment_name="experiment_zero",
        renders_root=tmp_path,
        cache_root=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        scenes=["room_0"],
        encoders=["dinov2_vitb14"],
    )
    values.update(overrides)
    return EvalConfig(**values)


def test_config_rejects_unknown_scenes_and_encoders(tmp_path):
    with pytest.raises(ValueError, match="unknown Replica scenes"):
        base_config(tmp_path, scenes=["not_a_scene"])
    with pytest.raises(ValueError, match="unknown encoders"):
        base_config(tmp_path, encoders=["clip"])


def test_config_defaults_the_floor_to_the_training_split(tmp_path):
    """The Mean-Feature floor must not be fitted to the scenes it floors."""
    cfg = base_config(tmp_path, scenes=["room_0", "hotel_0"])
    assert cfg.mean_feature_scenes == ["room_0"]  # hotel_0 is a test scene


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "experiment_name: x\nrenders_root: a\ncache_root: b\noutput_root: c\n"
        "scenes: [room_0]\nencoders: [dinov2_vitb14]\nnonsense: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown config keys"):
        load_eval_config(path)


def test_geometry_defaults_to_float32(tmp_path):
    """float64 would cost one to two orders of magnitude on a GPU for no accuracy."""
    assert base_config(tmp_path).torch_dtype == torch.float32
    with pytest.raises(ValueError, match="geometry_dtype"):
        base_config(tmp_path, geometry_dtype="float16")


def test_shipped_config_loads():
    from pathlib import Path

    cfg = load_eval_config(Path(__file__).resolve().parents[1] / "configs" / "experiment_zero.yaml")
    assert cfg.encoders and cfg.scenes


# ---------------------------------------------------------------------------
# End to end on a synthetic scene
# ---------------------------------------------------------------------------

SIDE = 56


def build_eval_scene(root, scene="room_0", encoders=("dinov2_vitb14",), channels=8, seed=0):
    """A renders directory and feature cache complete enough to evaluate."""
    from PIL import Image

    from lot.encoders import cache_dir

    scene_root = root / scene
    (scene_root / "rgb").mkdir(parents=True)
    (scene_root / "depth").mkdir(parents=True)
    K = intrinsics_from_hfov(SIDE, SIDE, 90.0)
    posed = program_rotation(base_pose(), [-5.0, 0.0, 5.0], [])
    posed += program_translation(base_pose(), [0.1], 3.0)
    generator = torch.Generator().manual_seed(seed)
    frames, features = [], {}
    counters: dict[str, int] = {}
    for frame in posed:
        index = counters.get(frame.regime, 0)
        counters[frame.regime] = index + 1
        frame_id = f"{scene}_vp00_{frame.regime}_{index:03d}"
        Image.fromarray(np.zeros((SIDE, SIDE, 3), dtype=np.uint8)).save(
            scene_root / f"rgb/{frame_id}.png"
        )
        np.save(scene_root / f"depth/{frame_id}.npy", np.full((SIDE, SIDE), 3.0, dtype=np.float32))
        frames.append(
            FrameRecord(
                frame_id=frame_id,
                scene=scene,
                regime=frame.regime,
                params=dict(frame.params, viewpoint=0),
                T_world_from_camera=frame.T_world_from_camera,
                K=K,
                height=SIDE,
                width=SIDE,
                rgb_path=f"rgb/{frame_id}.png",
                depth_path=f"depth/{frame_id}.npy",
            )
        )
        features[frame_id] = (
            torch.rand((channels, SIDE // 14, SIDE // 14), generator=generator)
            .to(torch.float16)
            .numpy()
        )
    manifest = Manifest(
        scene=scene,
        metadata={
            "depth_convention": {"raw_verdict": "planar_z", "stored_depth": "planar_z"}
        },
        frames=frames,
    )
    write_manifest(scene_root / "manifest.json", manifest)
    write_frame_stats(scene_root, manifest)
    for encoder in encoders:
        directory = cache_dir(root / "cache", encoder, scene)
        directory.mkdir(parents=True)
        np.savez(directory / "features.npz", **features)
    return manifest


def test_evaluate_scene_end_to_end(tmp_path):
    build_eval_scene(tmp_path)
    cfg = base_config(tmp_path, max_pairs_per_stratum=4, points_per_pair=64)
    mean = {"dinov2_vitb14": torch.zeros((8, SIDE // 14, SIDE // 14))}
    rows = evaluate_scene(cfg, "room_0", mean)
    assert rows
    pairs = {(r["context_frame_id"], r["target_frame_id"]) for r in rows}
    # Five variants on the per-point path, three on the splat path, per encoder.
    assert len(rows) == len(pairs) * 8
    for row in rows:
        assert row["encoder"] == "dinov2_vitb14"
        assert row["scene"] == "room_0"
        assert row["split"] == "train"
        assert row["path"] in (PER_POINT, SPLAT_POOL)
        assert 0.0 <= row["covisible_fraction"] <= 1.0
        assert row["parallax_bin"]


def test_results_round_trip_through_parquet(tmp_path):
    build_eval_scene(tmp_path)
    cfg = base_config(tmp_path, max_pairs_per_stratum=2, points_per_pair=32)
    rows = evaluate_scene(
        cfg, "room_0", {"dinov2_vitb14": torch.zeros((8, SIDE // 14, SIDE // 14))}
    )
    path = tmp_path / "eval" / "room_0.parquet"
    write_rows(path, rows)
    back = read_rows(path)
    assert len(back) == len(rows)
    assert set(back[0]) == set(rows[0])
    assert back[0]["variant"] == rows[0]["variant"]
    with pytest.raises(FileExistsError):
        write_rows(path, rows)


def test_margins_are_a_subtraction_between_rows_of_one_pair(tmp_path):
    """The table stores floors as rows, so a margin can never disagree with its parts."""
    build_eval_scene(tmp_path)
    cfg = base_config(tmp_path, max_pairs_per_stratum=2, points_per_pair=32)
    rows = evaluate_scene(
        cfg, "room_0", {"dinov2_vitb14": torch.zeros((8, SIDE // 14, SIDE // 14))}
    )
    key = ("context_frame_id", "target_frame_id", "encoder", "path")
    grouped: dict[tuple, dict[str, float]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in key), {})[row["variant"]] = row["cosine_mean"]
    assert grouped
    for scores in grouped.values():
        assert ORACLE_TRANSPORT in scores and NO_WARP_COPY in scores


def test_centering_restores_range_when_one_direction_dominates():
    """The reason the results table carries both readings.

    Measured on the caches, VGGT puts 0.91 of a feature's norm in a single
    shared direction against DINOv2's 0.42. A shared direction that large
    forces every cosine high regardless of content, so the raw metric stops
    resolving anything. Subtracting the dataset mean removes a constant that
    says nothing about which surface a patch sits on.
    """
    generator = torch.Generator().manual_seed(0)
    content = torch.randn((256, 32), generator=generator)
    other = torch.randn((256, 32), generator=generator)
    offset = torch.zeros(32)
    offset[0] = 30.0
    center = ((content + offset).mean(dim=0) + (other + offset).mean(dim=0)) / 2

    raw = agreement_metrics(content + offset, other + offset, center)
    # The shared offset pins the raw cosine high between unrelated features.
    assert raw["cosine_mean"] > 0.95
    # Centering exposes that they are unrelated.
    assert abs(raw["cosine_centered_mean"]) < 0.15
    # Identical features stay perfect either way.
    same = agreement_metrics(content + offset, content + offset, center)
    assert same["cosine_mean"] == pytest.approx(1.0, abs=1e-5)
    assert same["cosine_centered_mean"] == pytest.approx(1.0, abs=1e-5)


def test_every_row_carries_both_readings(tmp_path):
    build_eval_scene(tmp_path)
    cfg = base_config(tmp_path, max_pairs_per_stratum=2, points_per_pair=32)
    rows = evaluate_scene(
        cfg, "room_0", {"dinov2_vitb14": torch.zeros((8, SIDE // 14, SIDE // 14))}
    )
    for row in rows:
        assert "cosine_mean" in row and "cosine_centered_mean" in row
        assert "l2_mean" in row and "l2_centered_mean" in row
