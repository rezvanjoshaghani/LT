"""Phase 4 tests: alignment ladder, validity notions, and the 4.5 gates.

The analytic fixture is a synthetic scene complete enough to drive the whole
Phase 4 pipeline: rendered depth and RGB on disk, a manifest, a frame-stats
sidecar, a fabricated DINOv2 feature cache with surface-attached features,
and a fabricated VGGT depth cache whose estimated depth is ground truth
divided by three. That construction makes the right answers analytic: the
scene and image scales must recover exactly 3, the aligned levels must erase
the tax, and the unaligned level must pay one.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
import torch

from lot.analysis_config import load_analysis_config
from lot.encoders import CACHE_VERSION, cache_dir, features_digest
from lot.evaluate import MEAN_FEATURE, NO_WARP_COPY, ORACLE_TRANSPORT, PER_POINT, SPLAT_POOL
from lot.phase4 import (
    LEVELS,
    Phase4Config,
    Phase4GateError,
    VGGT_IMAGE_SCALE,
    VGGT_NO_ALIGN,
    VGGT_SCENE_SCALE,
    aligned_depth,
    assert_forcing_disabled_matches_default,
    boundary_cells_from_mask,
    convention_report,
    depth_boundary_mask,
    evaluate_scene_phase4,
    frame_calibration,
    load_depth_archive,
    low_texture_cells,
    phase4_measurement_digest,
    resample_depth_nearest,
    scene_scale_leave_target_out,
    secant_map,
    secant_regression,
    splat_plan_detail,
    transport_prevalid,
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
from lot.transport import transport_plan

SIDE = 112
CHANNELS = 768
SCENE = "room_0"
EST_SCALE = 3.0  # ground truth over estimated depth, everywhere


def base_pose() -> torch.Tensor:
    T = torch.eye(4, dtype=torch.float64)
    T[:3, 3] = torch.tensor([0.0, 1.5, 0.0], dtype=torch.float64)
    return T


def surface_features(frame: FrameRecord, depth: np.ndarray) -> np.ndarray:
    """Features that are a fixed smooth function of the world point."""
    K = frame.K.numpy().astype(np.float64)
    T = frame.T_world_from_camera.numpy().astype(np.float64)
    hp, wp = SIDE // 14, SIDE // 14
    centers = np.arange(hp) * 14 + 6.5
    vv, uu = np.meshgrid(centers, centers, indexing="ij")
    d = depth.astype(np.float64)[np.rint(vv).astype(int), np.rint(uu).astype(int)]
    x = (uu - K[0, 2]) * d / K[0, 0]
    y = (vv - K[1, 2]) * d / K[1, 1]
    world = np.stack((x, y, d), axis=-1) @ T[:3, :3].T + T[:3, 3]
    rng = np.random.default_rng(4242)
    out = np.zeros((CHANNELS, hp, wp))
    for c in range(CHANNELS):
        k = rng.normal(size=3) * 1.4
        out[c] = np.sin(world @ k + rng.uniform(0, 6.28))
    return out.astype(np.float16)


def build_scene(root, est_transform=lambda gt: gt / EST_SCALE):
    """A full synthetic scene with feature and estimated-depth caches."""
    from PIL import Image

    scene_root = root / SCENE
    (scene_root / "rgb").mkdir(parents=True)
    (scene_root / "depth").mkdir(parents=True)
    K = intrinsics_from_hfov(SIDE, SIDE, 90.0)

    yy, xx = np.mgrid[0:SIDE, 0:SIDE]
    depth = np.where(xx < SIDE // 2, 2.0, 4.0).astype(np.float32) + 0.002 * yy
    rgb = np.zeros((SIDE, SIDE, 3), np.uint8)
    rng = np.random.default_rng(7)
    rgb[:, : SIDE // 2] = rng.integers(0, 255, (SIDE, SIDE // 2, 3), dtype=np.uint8)

    posed = program_rotation(base_pose(), [-10.0, -5.0, 0.0, 5.0, 10.0], []) + (
        program_translation(base_pose(), [0.05, 0.1], 3.0)
    )
    frames, features, est_depth = [], {}, {}
    counters: dict[str, int] = {}
    for frame in posed:
        index = counters.get(frame.regime, 0)
        counters[frame.regime] = index + 1
        fid = f"{SCENE}_vp00_{frame.regime}_{index:03d}"
        Image.fromarray(rgb).save(scene_root / f"rgb/{fid}.png")
        np.save(scene_root / f"depth/{fid}.npy", depth)
        record = FrameRecord(
            frame_id=fid, scene=SCENE, regime=frame.regime,
            params=dict(frame.params, viewpoint=0),
            T_world_from_camera=frame.T_world_from_camera, K=K,
            height=SIDE, width=SIDE,
            rgb_path=f"rgb/{fid}.png", depth_path=f"depth/{fid}.npy",
        )
        frames.append(record)
        features[fid] = surface_features(record, depth)
        est_depth[fid] = est_transform(depth).astype(np.float16)
        est_depth[f"{fid}__conf"] = np.ones_like(depth, dtype=np.float16)

    manifest = Manifest(
        scene=SCENE,
        metadata={"depth_convention": {"raw_verdict": "planar_z", "stored_depth": "planar_z"}},
        frames=frames,
    )
    write_manifest(scene_root / "manifest.json", manifest)
    write_frame_stats(scene_root, manifest)

    feature_dir = cache_dir(root / "cache", "dinov2_vitb14", SCENE)
    feature_dir.mkdir(parents=True)
    np.savez(feature_dir / "features.npz", **features)
    (feature_dir / "meta.json").write_text(json.dumps({
        "cache_version": CACHE_VERSION, "encoder": "dinov2_vitb14", "scene": SCENE,
        "channels": CHANNELS, "patch_size": 14, "patch_grid": [SIDE // 14, SIDE // 14],
        "image_hw": [SIDE, SIDE], "dtype": "float16", "frame_count": len(features),
        "frame_ids": [f.frame_id for f in frames], "has_depth": False,
        "weights_fingerprint": "b" * 32, "weights_revision": "2" * 40,
        "code_revision": "3" * 40, "features_digest": features_digest(features),
        "depth_digest": None,
    }, indent=1))

    depth_dir = cache_dir(root / "cache", "vggt_1b", SCENE)
    depth_dir.mkdir(parents=True)
    np.savez(depth_dir / "depth.npz", **est_depth)
    (depth_dir / "meta.json").write_text(json.dumps({
        "cache_version": CACHE_VERSION, "encoder": "vggt_1b", "scene": SCENE,
        "channels": 2048, "patch_size": 14, "patch_grid": [SIDE // 14, SIDE // 14],
        "image_hw": [SIDE, SIDE], "dtype": "float16", "frame_count": len(frames),
        "frame_ids": [f.frame_id for f in frames], "has_depth": True,
        "weights_fingerprint": "c" * 32, "weights_revision": "4" * 40,
        "code_revision": "5" * 40,
        "features_digest": "d" * 32,
        "depth_digest": features_digest(est_depth),
    }, indent=1))
    return manifest


@pytest.fixture(scope="module")
def phase4_run(tmp_path_factory):
    """One end-to-end Phase 4 evaluation of the analytic scene."""
    root = tmp_path_factory.mktemp("phase4")
    manifest = build_scene(root)
    cfg = Phase4Config(
        experiment_name="phase4_test", renders_root=root, cache_root=root / "cache",
        output_root=root / "out", scenes=[SCENE], mean_vector_dir=root / "out" / "mv",
    )
    analysis = load_analysis_config()
    convention = convention_report(cfg, analysis)
    from lot.evaluate import load_or_build_mean_vector

    mean_vector = load_or_build_mean_vector(
        cfg.cache_root, "dinov2_vitb14", [SCENE], cfg.mean_vector_dir
    )
    rows, evidence = evaluate_scene_phase4(cfg, SCENE, mean_vector, analysis, convention)
    return {
        "root": root, "cfg": cfg, "analysis": analysis, "convention": convention,
        "rows": rows, "evidence": evidence, "mean_vector": mean_vector,
        "n_frames": len(manifest.frames),
    }


# ---------------------------------------------------------------------------
# Unit tests: convention, resampling, calibration, validity
# ---------------------------------------------------------------------------

def test_secant_regression_classifies_both_conventions():
    K = intrinsics_from_hfov(SIDE, SIDE, 90.0)
    gt = np.full((SIDE, SIDE), 3.0, dtype=np.float32)
    sec = secant_map(K, SIDE, SIDE).astype(np.float32)
    planar = secant_regression(0.83 * gt, gt, K, 0.05)
    assert planar["verdict"] == "planar_z"
    assert abs(planar["slope"]) < 1e-6
    ray = secant_regression(0.83 * gt * sec, gt, K, 0.05)
    assert ray["verdict"] == "ray_distance"
    assert abs(ray["slope"] - 0.83) < 1e-6
    # The frozen conversion recovers the planar map exactly.
    converted = (0.83 * gt * sec) / sec
    assert np.allclose(converted, 0.83 * gt, rtol=1e-6)


def test_resample_identity_nearest_and_refusals():
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    same, record = resample_depth_nearest(depth, (4, 4))
    assert record["method"] == "identity" and same is depth
    half, record = resample_depth_nearest(depth, (2, 2))
    assert record["method"] == "nearest"
    # Nearest picks a value the source grid actually holds, never a blend.
    assert all(v in depth for v in half.reshape(-1))
    with pytest.raises(ValueError):
        resample_depth_nearest(depth, (2, 4))


def test_frame_calibration_recovers_scale_and_affine():
    rng = np.random.default_rng(0)
    gt = rng.uniform(2.0, 4.0, (50, 50)).astype(np.float32)
    prevalid = np.ones_like(gt, dtype=bool)
    third = frame_calibration((gt / 3.0), gt, prevalid)
    assert abs(third.image_scale - 3.0) < 1e-5
    assert abs(third.affine_s - 3.0) < 1e-3 and abs(third.affine_b) < 1e-3
    assert not third.affine_failed
    shifted = frame_calibration(((gt - 0.5) / 2.0), gt, prevalid)
    assert abs(shifted.affine_s - 2.0) < 1e-3 and abs(shifted.affine_b - 0.5) < 1e-3
    # A negatively correlated estimate fits s below zero and must fail loudly.
    inverted = frame_calibration((10.0 - gt), gt, prevalid)
    assert inverted.affine_failed


def test_scene_scale_excludes_the_target_frame():
    good = frame_calibration(
        np.full((8, 8), 1.0, np.float32), np.full((8, 8), 3.0, np.float32),
        np.ones((8, 8), bool),
    )
    poisoned = frame_calibration(
        np.full((8, 8), 1.0, np.float32), np.full((8, 8), 300.0, np.float32),
        np.ones((8, 8), bool),
    )
    calibrations = {"a": good, "b": good, "target": poisoned}
    scale, audit = scene_scale_leave_target_out(calibrations, "target")
    assert abs(scale - 3.0) < 1e-9
    assert "target" not in audit and set(audit) == {"a", "b"}


def test_transport_prevalid_is_the_frozen_rule():
    analysis = load_analysis_config()
    assert analysis.vggt_confidence_threshold is None
    depth = np.array([[1.0, -1.0], [np.nan, 2.0]], dtype=np.float32)
    valid = transport_prevalid(depth, None, analysis)
    assert valid.tolist() == [[True, False], [False, True]]
    # Positive scaling cannot change the set: the step 10 invariant's algebra.
    calib = frame_calibration(depth, np.full_like(depth, 3.0), valid)
    scaled = aligned_depth("image", depth, 1.0, calib)
    assert np.array_equal(transport_prevalid(scaled, None, analysis), valid)


def test_phase4_measurement_digest_moves_with_the_confidence_rule():
    analysis = load_analysis_config()
    gated = dataclasses.replace(analysis, vggt_confidence_threshold=0.5)
    assert phase4_measurement_digest(analysis) != phase4_measurement_digest(gated)
    assert analysis.measurement_digest() == gated.measurement_digest()


# ---------------------------------------------------------------------------
# Forced-collision machinery
# ---------------------------------------------------------------------------

def test_forcing_disabled_reproduces_the_frozen_plan():
    rng = np.random.default_rng(3)
    depth = torch.from_numpy(rng.uniform(1.0, 5.0, (SIDE, SIDE)).astype(np.float32))
    K = intrinsics_from_hfov(SIDE, SIDE, 90.0)
    T = torch.eye(4, dtype=torch.float64)
    T[0, 3] = 0.2
    default = transport_plan(depth, K, K, T.to(torch.float32), (SIDE, SIDE))
    detail = splat_plan_detail(depth, K, K, T.to(torch.float32), (SIDE, SIDE))
    assert torch.equal(default.weights, detail.weights)
    assert torch.equal(default.coverage, detail.coverage)
    assert_forcing_disabled_matches_default(depth, K, K, T.to(torch.float32), (SIDE, SIDE))


def test_forced_winners_reproduce_the_source_ordering():
    rng = np.random.default_rng(4)
    depth = torch.from_numpy(rng.uniform(1.0, 5.0, (SIDE, SIDE)).astype(np.float32))
    K = intrinsics_from_hfov(SIDE, SIDE, 90.0)
    T = torch.eye(4, dtype=torch.float32)
    reference = splat_plan_detail(depth, K, K, T, (SIDE, SIDE))
    forced = splat_plan_detail(
        depth, K, K, T, (SIDE, SIDE), forced_winner_keys=reference.winner_keys
    )
    assert torch.equal(reference.weights, forced.weights)


# ---------------------------------------------------------------------------
# Localization masks
# ---------------------------------------------------------------------------

def test_boundary_mask_marks_the_edge_and_dilation():
    analysis = load_analysis_config()
    depth = torch.from_numpy(
        np.where(np.arange(SIDE)[None, :] < SIDE // 2, 2.0, 4.0)
        .astype(np.float32) * np.ones((SIDE, 1), np.float32)
    )
    mask = depth_boundary_mask(depth, analysis)
    edge = SIDE // 2
    radius = analysis.depth_boundary_dilation_px
    assert mask[:, edge - 1 : edge + 1].all()
    assert not mask[:, : edge - 1 - radius - 1].any()
    assert not mask[:, edge + radius + 2 :].any()
    cells = boundary_cells_from_mask(mask)
    grid = cells.reshape(SIDE // 14, SIDE // 14)
    assert grid[:, edge // 14 - 1 : edge // 14 + 1].all()
    assert not grid[:, 0].any() and not grid[:, -1].any()


def test_low_texture_cells_split_flat_from_noise():
    analysis = load_analysis_config()
    rgb = np.zeros((SIDE, SIDE, 3), np.uint8)
    rng = np.random.default_rng(7)
    rgb[:, : SIDE // 2] = rng.integers(0, 255, (SIDE, SIDE // 2, 3), dtype=np.uint8)
    cells = low_texture_cells(rgb, analysis).reshape(SIDE // 14, SIDE // 14)
    assert not cells[:, : SIDE // 28].any()
    # The seam patch column can catch one pixel of central-difference bleed
    # from the noisy half; everything past it must read flat.
    assert cells[:, SIDE // 28 + 1 :].all()


# ---------------------------------------------------------------------------
# End-to-end: gates, ladder, schema
# ---------------------------------------------------------------------------

def test_convention_report_classifies_the_fixture_planar(phase4_run):
    verdicts = phase4_run["convention"]["verdicts"]
    assert verdicts == {SCENE: "planar_z"}


def test_depth_archive_digest_is_verified(phase4_run):
    cfg = phase4_run["cfg"]
    archive = load_depth_archive(cfg.cache_root, "vggt_1b", SCENE)
    first = next(iter(archive["depth"].values()))
    assert first.dtype == np.float32 and first.shape == (SIDE, SIDE)
    meta_path = cache_dir(cfg.cache_root, "vggt_1b", SCENE) / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["depth_digest"] = "0" * 32
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_depth_archive(cfg.cache_root, "vggt_1b", SCENE)
    meta["depth_digest"] = features_digest(
        dict(np.load(cache_dir(cfg.cache_root, "vggt_1b", SCENE) / "depth.npz"))
    )
    meta_path.write_text(json.dumps(meta))


def test_rotation_gates_pass_and_are_evidenced(phase4_run):
    evidence = phase4_run["evidence"]
    analysis = phase4_run["analysis"]
    meta = evidence["metadata"]
    assert meta["gate_checks"] > 0
    assert meta["gate_coord_max_px"] <= analysis.rotation_gate_coord_tol_px
    assert meta["gate_score_max_abs"] <= analysis.rotation_gate_score_tol
    assert meta["gate_forced_max_abs"] <= analysis.rotation_gate_forced_tol
    per_point_checks = [g for g in evidence["gate_evidence"] if g["path"] == PER_POINT]
    splat_checks = [g for g in evidence["gate_evidence"] if g["path"] == SPLAT_POOL]
    assert per_point_checks and splat_checks
    for check in splat_checks:
        assert math.isfinite(check["collision_tax_raw"])


def test_multiplicative_levels_share_transport_validity(phase4_run):
    for audit in phase4_run["evidence"]["audits"].values():
        counts = {
            level: audit["levels"][level]["pp_transport_valid"]
            for level in ("none", "scene", "image")
            if level in audit["levels"]
        }
        assert len(set(counts.values())) == 1, counts


def test_target_frame_never_calibrates_level1(phase4_run):
    audits = phase4_run["evidence"]["audits"]
    assert audits
    n_frames = phase4_run["n_frames"]
    for audit in audits.values():
        # Every fixture frame carries calibration pixels, so the leave-target-
        # out population is exactly all frames but one for every pair.
        assert audit["scene_audit_frames"] == n_frames - 1
    assert phase4_run["evidence"]["metadata"]["target_exclusion_asserted_per_record"]


def test_alignment_erases_the_tax_and_no_alignment_pays_one(phase4_run):
    rows = phase4_run["rows"]

    def cell(level, variant, regime):
        values = [
            r["cosine_mean"]
            for r in rows
            if r["level"] == level and r["variant"] == variant
            and r["population"] == "matched" and r["path"] == PER_POINT
            and r["regime"] == regime and math.isfinite(r["cosine_mean"])
        ]
        return float(np.mean(values)) if values else float("nan")

    for regime in ("rotation", "translation"):
        for level, name in (("scene", VGGT_SCENE_SCALE), ("image", VGGT_IMAGE_SCALE)):
            oracle = cell(level, ORACLE_TRANSPORT, regime)
            est = cell(level, name, regime)
            assert abs(oracle - est) < 2e-3, (regime, level, oracle, est)
    # Under pure rotation even the unaligned level is exact: depth cancels.
    assert abs(cell("none", ORACLE_TRANSPORT, "rotation") - cell("none", VGGT_NO_ALIGN, "rotation")) < 1e-5
    # Under translation the wrong scale must cost something real.
    tax_none = cell("none", ORACLE_TRANSPORT, "translation") - cell("none", VGGT_NO_ALIGN, "translation")
    tax_image = cell("image", ORACLE_TRANSPORT, "translation") - cell("image", VGGT_IMAGE_SCALE, "translation")
    assert tax_none > 0.02, tax_none
    assert tax_none > tax_image + 0.02


def test_row_schema_masks_and_permitted_nonfinite(phase4_run):
    from lot.evaluate import unpack_mask

    rows = phase4_run["rows"]
    universe = phase4_run["evidence"]["metadata"]["universe_size"]
    assert universe == (SIDE // 14) ** 2
    seen_levels = {r["level"] for r in rows}
    assert {"gt", "none", "scene", "image", "affine"} <= seen_levels
    for row in rows:
        mask = unpack_mask(row["sample_mask"], universe)
        assert int(mask.sum()) == row["n"]
        assert math.isfinite(row["cosine_mean"])
        if row["variant"] == MEAN_FEATURE:
            assert math.isnan(row["cosine_centered_mean"])
        else:
            assert math.isfinite(row["cosine_centered_mean"])
    populations = {r["population"] for r in rows}
    assert {"full", "matched", "boundary", "interior", "lowtex", "hightex"} <= populations


def test_full_gt_rows_match_phase3_scoring(phase4_run):
    """The Phase 4 gt rows are Phase 3's Oracle and floor, byte for byte."""
    from lot.analysis_config import load_analysis_config
    from lot.evaluate import evaluate_pair_for_encoder, pair_geometry
    from lot.render_replica import MANIFEST_NAME, load_manifest
    from lot.datasets import load_scene_pairs, subsample_by_stratum
    from lot.geometry import relative_pose
    from lot.evaluate import _SceneCache

    run = phase4_run
    cfg = run["cfg"]
    analysis = run["analysis"]
    scene_root = cfg.renders_root / SCENE
    manifest = load_manifest(scene_root / MANIFEST_NAME)
    frames = {f.frame_id: f for f in manifest.frames}
    pairs = subsample_by_stratum(
        load_scene_pairs(cfg.renders_root, SCENE, config=analysis),
        analysis.max_pairs_per_stratum, seed=cfg.seed, config=analysis,
    )
    pair = pairs[0]
    cache = _SceneCache(scene_root, cfg.cache_root, ["dinov2_vitb14"], SCENE, manifest)
    T = relative_pose(
        frames[pair.target_frame_id].T_world_from_camera,
        frames[pair.context_frame_id].T_world_from_camera,
    ).to(cfg.torch_dtype)
    geometry = pair_geometry(
        cache.depth(frames[pair.context_frame_id].depth_path).to(cfg.torch_dtype),
        cache.depth(frames[pair.target_frame_id].depth_path).to(cfg.torch_dtype),
        frames[pair.context_frame_id].K.to(cfg.torch_dtype),
        frames[pair.target_frame_id].K.to(cfg.torch_dtype),
        T, SCENE, pair.context_frame_id, pair.target_frame_id, analysis,
    )
    phase3_rows = evaluate_pair_for_encoder(
        geometry,
        cache.features("dinov2_vitb14", pair.context_frame_id),
        cache.features("dinov2_vitb14", pair.target_frame_id),
        run["mean_vector"],
    )
    cache.close()
    phase3 = {
        (r["path"], r["variant"]): r["cosine_mean"]
        for r in phase3_rows
        if r["variant"] in (ORACLE_TRANSPORT, NO_WARP_COPY)
    }
    phase4 = {
        (r["path"], r["variant"]): r["cosine_mean"]
        for r in run["rows"]
        if r["level"] == "gt"
        and r["context_frame_id"] == pair.context_frame_id
        and r["target_frame_id"] == pair.target_frame_id
        and r["variant"] in (ORACLE_TRANSPORT, NO_WARP_COPY)
    }
    assert phase3 == phase4


def test_a_mislabelled_rotation_pair_trips_the_coordinate_gate(phase4_run):
    """A translation transform wearing the rotation tag must fail the gate.

    This is the mutation-style check that the gate is live: under genuine
    rotation the estimated depth cannot move a correspondence, so the only
    way to see the gate fire is to hand it a pair where depth does matter.
    """
    import dataclasses as dc

    from lot.datasets import load_scene_pairs, subsample_by_stratum
    from lot.evaluate import _SceneCache
    from lot.geometry import relative_pose
    from lot.phase4 import PairDepthInputs, evaluate_pair_phase4, frame_calibration
    from lot.render_replica import MANIFEST_NAME, load_manifest
    from lot.evaluate import pair_geometry

    run = phase4_run
    cfg = run["cfg"]
    analysis = run["analysis"]
    scene_root = cfg.renders_root / SCENE
    manifest = load_manifest(scene_root / MANIFEST_NAME)
    frames = {f.frame_id: f for f in manifest.frames}
    pairs = subsample_by_stratum(
        load_scene_pairs(cfg.renders_root, SCENE, config=analysis),
        analysis.max_pairs_per_stratum, seed=cfg.seed, config=analysis,
    )
    pair = next(p for p in pairs if p.regime == "translation" and p.baseline_m > 0.2)
    lying = dc.replace(pair, regime="rotation")
    cache = _SceneCache(scene_root, cfg.cache_root, ["dinov2_vitb14"], SCENE, manifest)
    depth_context = cache.depth(frames[pair.context_frame_id].depth_path).to(cfg.torch_dtype)
    depth_target = cache.depth(frames[pair.target_frame_id].depth_path).to(cfg.torch_dtype)
    T = relative_pose(
        frames[pair.target_frame_id].T_world_from_camera,
        frames[pair.context_frame_id].T_world_from_camera,
    ).to(cfg.torch_dtype)
    geometry = pair_geometry(
        depth_context, depth_target,
        frames[pair.context_frame_id].K.to(cfg.torch_dtype),
        frames[pair.target_frame_id].K.to(cfg.torch_dtype),
        T, SCENE, pair.context_frame_id, pair.target_frame_id, analysis,
    )
    gt_ctx = depth_context.numpy()
    gt_tgt = depth_target.numpy()
    calib = frame_calibration(gt_ctx / 3.0, gt_ctx, np.ones_like(gt_ctx, bool))
    inputs = PairDepthInputs(
        est_context=(gt_ctx / 3.0).astype(np.float32),
        est_target=(gt_tgt / 3.0).astype(np.float32),
        scene_scale=3.0, scene_audit={"other": 1}, context_calib=calib,
    )
    size = geometry.size
    with pytest.raises(Phase4GateError):
        evaluate_pair_phase4(
            geometry, lying, depth_context, inputs,
            cache.features("dinov2_vitb14", pair.context_frame_id),
            cache.features("dinov2_vitb14", pair.target_frame_id),
            run["mean_vector"], analysis,
            np.zeros(size, dtype=bool), np.zeros(size, dtype=bool),
            cfg.torch_dtype,
            frames[pair.context_frame_id].K.to(cfg.torch_dtype),
            frames[pair.target_frame_id].K.to(cfg.torch_dtype),
            T, [],
        )
    cache.close()


def test_report_ladder_from_the_fixture(phase4_run):
    from lot.phase4_report import build_records, ladder_table, quantity_formulas

    analysis = phase4_run["analysis"]
    records = build_records(phase4_run["rows"], analysis)
    assert records
    ladder = ladder_table(records, analysis)
    by_key = {
        (r["analysis"], r["metric"], r["path"], r["level"]): r for r in ladder
    }
    aligned = by_key[("translation", "cosine_mean", PER_POINT, "image")]
    unaligned = by_key[("translation", "cosine_mean", PER_POINT, "none")]
    assert abs(aligned["depth_tax"]) < 2e-3
    assert unaligned["depth_tax"] > 0.02
    # Retained fraction sits near one when alignment recovers ground truth,
    # and is suppressed rather than reported when the margin is tiny.
    if math.isfinite(aligned["retained_fraction"]):
        assert abs(aligned["retained_fraction"] - 1.0) < 0.1
    formulas = quantity_formulas(analysis)
    tiny = {"oracle_m": 0.5001, "nowarp_m": 0.5, "est": 0.5, "oracle_full": 0.5,
            "transported_fraction": 1.0}
    assert math.isnan(formulas["retained_fraction"](tiny))
    healthy = {"oracle_m": 0.7, "nowarp_m": 0.5, "est": 0.65, "oracle_full": 0.72,
               "transported_fraction": 0.9}
    assert abs(formulas["retained_fraction"](healthy) - 0.75) < 1e-12
    assert abs(formulas["depth_tax"](healthy) - 0.05) < 1e-12
