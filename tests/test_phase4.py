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
    DECISIVE_FUNCTION,
    LEVELS,
    Phase4Config,
    Phase4GateError,
    build_convention_record,
    extract_function,
    run_convention,
    source_authority,
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
    landing_flip_diagnostics,
    scene_scale_leave_target_out,
    secant_map,
    secant_regression,
    splat_plan_detail,
    splat_structure,
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


def planar_authority() -> dict:
    """A source-authority record standing in for the installed VGGT.

    The real one is read from vggt/utils/geometry.py by source_authority();
    the suite must run without VGGT installed, so the tests supply the same
    shape. test_source_authority_signature_detection pins the detector itself
    against both conventions.
    """
    return {
        "verdict": "planar_z", "unambiguous": True,
        "function": "depth_to_cam_coords_points", "module": "vggt/utils/geometry.py",
        "first_line": 87, "lines": [], "reason": None,
        "checks": {"assigns_z_from_depth_directly": True,
                   "scales_x_and_y_by_depth_over_focal": True,
                   "no_ray_rescaling_in_function": True},
        # Matches the code_revision the fixture's depth cache records, which
        # build_convention_record requires: the source cited as authority has
        # to be the source that produced the cached depth.
        "revision": {"distribution": "vggt", "version": "1.0",
                     "commit": "5" * 40, "url": None, "pinned": True},
        "roots": [],
    }


def run_phase3(root, cfg, analysis):
    """The Phase 3 evaluation whose pair population Phase 4 inherits.

    Produced by the real evaluator so the inheritance check in
    evaluate_scene_phase4 is exercised against a genuine parquet rather than
    a fixture that asserts against itself.
    """
    from lot.evaluate import (
        EvalConfig, evaluate_scene, load_or_build_mean_vector, write_rows,
    )

    phase3 = EvalConfig(
        experiment_name="experiment_zero", renders_root=root,
        cache_root=root / "cache", output_root=root / "p3",
        scenes=[SCENE], encoders=["dinov2_vitb14"], seed=cfg.seed,
        mean_vector_scenes=[SCENE],
    )
    mean_vector = load_or_build_mean_vector(
        phase3.cache_root, "dinov2_vitb14", [SCENE],
        phase3.output_root / phase3.experiment_name,
    )
    rows, meta = evaluate_scene(phase3, SCENE, {"dinov2_vitb14": mean_vector}, analysis)
    write_rows(phase3.eval_dir / f"{SCENE}.parquet", rows, meta)
    return phase3.eval_dir, mean_vector


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
    phase3_eval_dir, _ = run_phase3(root, cfg, analysis)
    cfg.phase3_eval_dir = phase3_eval_dir
    diagnostic = convention_report(cfg, analysis)
    convention = build_convention_record(
        diagnostic, planar_authority(),
        json.loads((cache_dir(cfg.cache_root, "vggt_1b", SCENE) / "meta.json").read_text()),
    )
    from lot.evaluate import load_or_build_mean_vector

    mean_vector = load_or_build_mean_vector(
        cfg.cache_root, "dinov2_vitb14", [SCENE], cfg.mean_vector_dir
    )
    rows, evidence = evaluate_scene_phase4(cfg, SCENE, mean_vector, analysis, convention)
    return {
        "root": root, "cfg": cfg, "analysis": analysis, "convention": convention,
        "rows": rows, "evidence": evidence, "mean_vector": mean_vector,
        "n_frames": len(manifest.frames), "convention": convention,
        "phase3_eval_dir": phase3_eval_dir,
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
    structural = splat_plan_detail(
        depth, K, K, T, (SIDE, SIDE), forced_structure=splat_structure(reference)
    )
    assert torch.equal(reference.weights, structural.weights)
    with pytest.raises(ValueError):
        splat_plan_detail(
            depth, K, K, T, (SIDE, SIDE),
            forced_winner_keys=reference.winner_keys,
            forced_structure=splat_structure(reference),
        )


def test_a7_frozen_structure_survives_a_boundary_flip():
    """A deliberately boundary-adjacent landing flips cells between the arms.

    Every source pixel is engineered to land a hair past a floor(u + 0.5)
    boundary under ground-truth depth and a hair before it under estimated
    depth. The A7 frozen structure must still pool identically to the donor,
    the pre-A7 membership rule must lose every winner to the flip, and the
    landing diagnostics must count the flips at vanishing boundary margins.
    """
    side = 28
    z0 = 2.0
    K = intrinsics_from_hfov(side, side, 90.0)
    fx = float(K[0, 0])
    depth_gt = torch.full((side, side), z0, dtype=torch.float32)
    # Ground truth lands every pixel at u + 0.5001; the estimated depth,
    # scaled by five parts in ten thousand, lands it at u + 0.49985.
    T = torch.eye(4, dtype=torch.float32)
    T[0, 3] = 0.5001 * z0 / fx
    depth_est = (depth_gt.to(torch.float64) * (1 + 5e-4)).to(torch.float32)

    gt_detail = splat_plan_detail(depth_gt, K, K, T, (side, side))
    est_detail = splat_plan_detail(depth_est, K, K, T, (side, side))
    common = gt_detail.keep & est_detail.keep
    # The ground-truth arm shifts one pixel right, so its last column leaves
    # the image; the estimated arm keeps it. The intersection drops it.
    assert int(common.sum()) == side * (side - 1)

    gt_common = splat_plan_detail(depth_gt, K, K, T, (side, side), source_keep=common)
    unforced_est = splat_plan_detail(depth_est, K, K, T, (side, side), source_keep=common)
    # Both arms keep exactly the common set; the flip changes no validity.
    assert torch.equal(gt_common.keep, unforced_est.keep)

    # A7: the frozen structure pools identically to its donor even though
    # every landing flipped.
    structural = splat_plan_detail(
        depth_est, K, K, T, (side, side),
        source_keep=common, forced_structure=splat_structure(gt_common),
    )
    assert torch.equal(structural.weights, gt_common.weights)
    assert np.array_equal(structural.winner_keys, gt_common.winner_keys)

    # The pre-A7 membership rule evaluates Oracle keys at this arm's own
    # landings, so the flip makes every key miss and every winner vanish.
    membership = splat_plan_detail(
        depth_est, K, K, T, (side, side),
        source_keep=common, forced_winner_keys=gt_common.winner_keys,
    )
    assert float(membership.weights.abs().sum()) == 0.0

    flips = landing_flip_diagnostics(gt_common, unforced_est, (side, side))
    assert flips["landing_flip_count"] == int(common.sum())
    assert flips["landing_flip_fraction"] == 1.0
    assert flips["landing_flip_cells"] >= 1
    # The engineered margins: 1e-4 past the boundary on one side, 1.5e-4
    # before it on the other, both boundary-adjacent.
    assert flips["landing_flip_margin_min_px"] <= 2e-4
    assert flips["landing_coord_residual_max_px"] <= 1e-3

    # A structure naming a source outside this arm's kept set is refused.
    rogue_source = side * side - 1  # the dropped last-column corner
    n_cells = side * side
    tampered = dataclasses.replace(
        splat_structure(gt_common),
        winner_keys=np.sort(np.append(
            gt_common.winner_keys, np.int64(rogue_source) * n_cells
        )),
    )
    with pytest.raises(Phase4GateError, match="does not keep"):
        splat_plan_detail(
            depth_est, K, K, T, (side, side),
            source_keep=common, forced_structure=tampered,
        )
    # And a structure from another target grid is refused before use.
    with pytest.raises(Phase4GateError, match="different target grid"):
        splat_plan_detail(
            depth_est[:14, :14], K, K, T, (14, 14),
            forced_structure=splat_structure(gt_common),
        )


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
    verdicts = phase4_run["convention"]["secant_diagnostic"]["verdicts"]
    assert verdicts == {SCENE: "planar_z"}


# ---------------------------------------------------------------------------
# Amendment A6: one checkpoint, one convention
# ---------------------------------------------------------------------------

PLANAR_SOURCE = '''
def depth_to_cam_coords_points(depth_map, intrinsic):
    """Convert a depth map to camera coordinates."""
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]
    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map
    return np.stack((x_cam, y_cam, z_cam), axis=-1)
'''

RAY_SOURCE = '''
def depth_to_cam_coords_points(depth_map, intrinsic):
    """A ray-distance head has to rescale along the ray before assigning z."""
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]
    dx = (u - cu) / fu
    dy = (v - cv) / fv
    z_cam = depth_map / np.sqrt(1 + dx * dx + dy * dy)
    x_cam = dx * z_cam
    y_cam = dy * z_cam
    return np.stack((x_cam, y_cam, z_cam), axis=-1)
'''


CACHE_META = {"code_revision": "5" * 40, "weights_fingerprint": "c" * 32,
              "weights_revision": "4" * 40}


def test_source_authority_signature_detection(tmp_path, monkeypatch):
    """The planar signature is recognized, and a ray head is not mistaken for it."""
    import lot.phase4 as phase4

    for source, expect_planar in ((PLANAR_SOURCE, True), (RAY_SOURCE, False)):
        root = tmp_path / ("planar" if expect_planar else "ray")
        (root / "utils").mkdir(parents=True)
        (root / "utils" / "geometry.py").write_text(source, encoding="utf-8")
        monkeypatch.setattr(phase4, "vggt_package_roots", lambda root=root: [root])
        monkeypatch.setattr(
            phase4, "vggt_source_revision",
            lambda: {"distribution": "vggt", "version": "1.0",
                     "commit": "b" * 40, "url": None, "pinned": True},
        )
        evidence = phase4.source_authority()
        assert evidence["unambiguous"] is expect_planar
        assert (evidence["verdict"] == "planar_z") is expect_planar
        assert any("z_cam" in line for line in evidence["lines"])


def test_extract_function_takes_the_whole_body():
    first, body = extract_function(PLANAR_SOURCE, DECISIVE_FUNCTION)
    assert body[0].startswith(f"def {DECISIVE_FUNCTION}")
    assert any("z_cam = depth_map" in line for line in body)
    assert extract_function(PLANAR_SOURCE, "not_a_function") is None


def test_one_checkpoint_one_convention_even_when_diagnostics_disagree():
    """The deliberate regression case: scene verdicts disagree, conversion does not.

    This is the bug class the closure task names. A per-scene reading gave the
    same checkpoint two depth semantics inside one table; the record now
    carries exactly one decision and every scene reads it.
    """
    diagnostic = {
        "threshold": 0.05,
        "scenes": {
            "apartment_1": {"verdict": "ray_distance", "slope": 0.036},
            "office_1": {"verdict": "ray_distance", "slope": 0.457},
            "room_0": {"verdict": "planar_z", "slope": -0.004},
            "hotel_0": {"verdict": "planar_z", "slope": -0.610},
        },
        "verdicts": {"apartment_1": "ray_distance", "office_1": "ray_distance",
                     "room_0": "planar_z", "hotel_0": "planar_z"},
        "unanimous": False,
    }
    record = build_convention_record(diagnostic, planar_authority(), CACHE_META)
    assert record["depth_convention"] == "planar_z"
    assert record["depth_convention_authority"] == "source"
    assert record["depth_convention_conversion_applied"] is False
    assert record["secant_regression_role"] == "diagnostic_only"
    assert record["depth_convention_source_commit"] == "5" * 40
    # The disagreement is recorded, not resolved away.
    assert record["secant_diagnostic_disagrees_with_authority"] == [
        "apartment_1", "office_1"
    ]
    # Every scene, including the two the diagnostic flagged, reads one value.
    for scene in diagnostic["scenes"]:
        assert run_convention(record, CACHE_META) == "planar_z", scene


def test_convention_cannot_be_established_without_source_authority():
    ambiguous = {**planar_authority(), "unambiguous": False, "verdict": "ambiguous",
                 "reason": "signature not found"}
    with pytest.raises(Phase4GateError):
        build_convention_record({"scenes": {}}, ambiguous, {})
    # A record lacking the global decision, or claiming a non-source basis,
    # is refused rather than defaulted.
    with pytest.raises(Phase4GateError):
        run_convention({"scenes": {"a": {"verdict": "planar_z"}}}, CACHE_META)
    with pytest.raises(Phase4GateError):
        run_convention({"depth_convention": "planar_z",
                        "depth_convention_authority": "regression"}, CACHE_META)
    # A record written about another checkpoint does not govern this one.
    stale = build_convention_record({"scenes": {}}, planar_authority(), CACHE_META)
    with pytest.raises(Phase4GateError, match="does not govern|is not the"):
        run_convention(stale, {**CACHE_META, "weights_fingerprint": "f" * 32})


def test_authority_must_be_the_source_that_produced_the_cache():
    """Citing semantics read from other code is not evidence about these depths."""
    with pytest.raises(Phase4GateError, match="not the source that"):
        build_convention_record(
            {"scenes": {}}, planar_authority(), {"code_revision": "9" * 40}
        )
    # Matching revisions are accepted, and an unpinned cache skips the check
    # rather than inventing a comparison it cannot make.
    assert build_convention_record(
        {"scenes": {}}, planar_authority(), {"code_revision": "5" * 40}
    )["depth_convention"] == "planar_z"
    assert build_convention_record(
        {"scenes": {}}, planar_authority(), {"code_revision": "unknown"}
    )["depth_convention"] == "planar_z"


def test_planar_z_applies_no_cosine_conversion(phase4_run):
    """planar_z means the transported depth is the resampled cache, untouched."""
    from lot.phase4 import load_depth_archive, resample_depth_nearest, secant_map

    run = phase4_run
    cfg = run["cfg"]
    assert run["convention"]["depth_convention"] == "planar_z"
    assert run["convention"]["depth_convention_conversion_applied"] is False
    archive = load_depth_archive(cfg.cache_root, "vggt_1b", SCENE)
    frame_id, raw = next(iter(archive["depth"].items()))
    resampled, _ = resample_depth_nearest(raw, (SIDE, SIDE))
    K = intrinsics_from_hfov(SIDE, SIDE, 90.0)
    converted = resampled / secant_map(K, SIDE, SIDE)
    # The conversion would have changed the map materially, so "no conversion"
    # is a claim with content rather than a no-op.
    assert np.abs(converted - resampled).max() > 0.05 * float(np.abs(resampled).max())
    for row in run["rows"]:
        assert row["depth_convention"] == "planar_z" if "depth_convention" in row else True
    assert run["evidence"]["metadata"]["depth_convention"] == "planar_z"
    assert run["evidence"]["metadata"]["depth_convention_conversion_applied"] is False
    assert run["evidence"]["metadata"]["secant_regression_role"] == "diagnostic_only"


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
    # A7: both arms run under Oracle's frozen rasterization structure, so
    # the forced comparison is an identity, not merely inside tolerance.
    assert meta["gate_forced_max_abs"] == 0.0
    per_point_checks = [g for g in evidence["gate_evidence"] if g["path"] == PER_POINT]
    splat_checks = [g for g in evidence["gate_evidence"] if g["path"] == SPLAT_POOL]
    assert per_point_checks and splat_checks
    for check in splat_checks:
        assert check["forced_max_abs"] == 0.0
        assert check["forced_max_abs_centered"] == 0.0
        for metric in ("raw", "centered"):
            umbrella = check[f"unforced_rasterization_tax_{metric}"]
            assignment = check[f"landing_assignment_tax_{metric}"]
            ordering = check[f"collision_ordering_tax_{metric}"]
            assert math.isfinite(umbrella)
            # The decomposition telescopes back to the umbrella.
            assert abs(umbrella - (assignment + ordering)) <= 1e-12
        assert check["landing_flip_count"] >= 0
        assert math.isfinite(check["landing_flip_fraction"])
        assert math.isfinite(check["landing_coord_residual_max_px"])


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
    records, exclusions = build_records(phase4_run["rows"], analysis)
    assert exclusions["mask_mismatched_arms"] == 0
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


# ---------------------------------------------------------------------------
# Provenance and matching, the bug classes a code review demonstrated
# ---------------------------------------------------------------------------

def test_report_refuses_rows_from_another_measurement_config(phase4_run, tmp_path):
    """A directory agrees with itself perfectly under any config; bind to ours."""
    import dataclasses as dc

    from lot.evaluate import write_rows
    from lot.phase4_report import read_phase4_dir

    run = phase4_run
    analysis = run["analysis"]
    # The fixture runs from whatever tree the suite is invoked in, and the
    # report refuses a -dirty provenance. That refusal has its own coverage;
    # here the subject is the measurement digest, so the record is given a
    # clean commit to isolate it.
    meta = {**run["evidence"]["metadata"], "git_commit": "a" * 40}
    eval_dir = tmp_path / "eval"
    write_rows(eval_dir / f"{SCENE}.parquet", run["rows"], meta)
    # The honest config reads it.
    assert read_phase4_dir(eval_dir, analysis)
    # A different frozen validity rule is a different experiment, even though
    # every file in the directory is internally consistent.
    gated = dc.replace(analysis, vggt_confidence_threshold=0.5)
    with pytest.raises(ValueError, match="measurement config"):
        read_phase4_dir(eval_dir, gated)


def test_report_refuses_duplicates_and_unmatched_arms(phase4_run):
    """Subset matching is verified from the masks, not assumed."""
    from lot.phase4_report import build_records

    run = phase4_run
    analysis = run["analysis"]
    rows = run["rows"]
    records, exclusions = build_records(rows, analysis)
    assert records and exclusions["mask_mismatched_arms"] == 0

    # A duplicated row would otherwise overwrite its twin and silently move
    # the estimate; the report must refuse the directory instead.
    duplicated = list(rows) + [dict(rows[0])]
    with pytest.raises(ValueError, match="duplicate"):
        build_records(duplicated, analysis)

    # An arm whose mask differs from its ceiling and floor is excluded and
    # counted: differencing it would compare two populations.
    tampered = [dict(r) for r in rows]
    target = next(
        r for r in tampered
        if r["population"] == "matched" and r["variant"].startswith("VGGT-")
    )
    target["sample_mask"] = bytes(len(bytes(target["sample_mask"])))
    _, exclusions = build_records(tampered, analysis)
    assert exclusions["mask_mismatched_arms"] >= 1


def test_phase4_refuses_a_population_phase3_did_not_score(phase4_run, tmp_path):
    """PROTOCOL 4.1 inherits the pairs, so a divergence is refused not averaged."""
    import pyarrow.parquet as pq

    from lot.evaluate import read_run_metadata, write_rows
    from lot.phase4 import phase3_scene_reference

    run = phase4_run
    real = phase3_scene_reference(run["phase3_eval_dir"], SCENE, "dinov2_vitb14")
    assert real["pairs"], "the fixture's Phase 3 run scored something"

    # A Phase 3 run naming a pair this scene cannot produce must stop Phase 4.
    # The forged run keeps the genuine provenance record, so the population
    # check fires rather than the earlier cache-identity check.
    source = run["phase3_eval_dir"] / f"{SCENE}.parquet"
    table = pq.read_table(source).to_pylist()
    invented = dict(table[0])
    invented["context_frame_id"] = "room_0_vp99_rotation_000"
    forged = tmp_path / "forged"
    forged.mkdir()
    write_rows(forged / f"{SCENE}.parquet", table + [invented], read_run_metadata(source))
    cfg = dataclasses.replace(run["cfg"], phase3_eval_dir=forged)
    with pytest.raises(Phase4GateError, match="absent from the regenerated sample"):
        evaluate_scene_phase4(
            cfg, SCENE, run["mean_vector"], run["analysis"], run["convention"],
        )

    with pytest.raises(Phase4GateError, match="does not exist"):
        phase3_scene_reference(tmp_path / "nowhere", SCENE, "dinov2_vitb14")


def test_phase4_refuses_tampered_phase3_masks_and_scores(phase4_run, tmp_path):
    """Name-level agreement is not inheritance: masks and ceilings must match.

    The review's core case: a Phase 3 run whose pair names are intact but
    whose recorded masks or ceilings differ from what the current inputs
    reproduce means a pose, depth map, or filter moved underneath the names.
    """
    import pyarrow.parquet as pq

    from lot.evaluate import read_run_metadata, write_rows

    run = phase4_run
    source = run["phase3_eval_dir"] / f"{SCENE}.parquet"
    meta = read_run_metadata(source)

    # Tampered mask: flip one byte of one row's persisted validity mask.
    table = pq.read_table(source).to_pylist()
    victim = next(r for r in table if r["variant"] == ORACLE_TRANSPORT)
    mask = bytearray(bytes(victim["sample_mask"]))
    mask[0] ^= 0xFF
    for row in table:
        if (row["context_frame_id"], row["target_frame_id"], row["path"]) == (
            victim["context_frame_id"], victim["target_frame_id"], victim["path"]
        ):
            row["sample_mask"] = bytes(mask)
    forged = tmp_path / "mask"
    forged.mkdir()
    write_rows(forged / f"{SCENE}.parquet", table, meta)
    cfg = dataclasses.replace(run["cfg"], phase3_eval_dir=forged)
    with pytest.raises(Phase4GateError, match="not the one Phase 3 persisted"):
        evaluate_scene_phase4(
            cfg, SCENE, run["mean_vector"], run["analysis"], run["convention"],
        )

    # Tampered ceiling: move one recorded Oracle score past the recon bound.
    table = pq.read_table(source).to_pylist()
    for row in table:
        if row["variant"] == ORACLE_TRANSPORT and row["path"] == PER_POINT:
            row["cosine_mean"] = row["cosine_mean"] + 1e-3
            break
    forged = tmp_path / "score"
    forged.mkdir()
    write_rows(forged / f"{SCENE}.parquet", table, meta)
    cfg = dataclasses.replace(run["cfg"], phase3_eval_dir=forged)
    with pytest.raises(Phase4GateError, match="not the ones Phase 3 measured"):
        evaluate_scene_phase4(
            cfg, SCENE, run["mean_vector"], run["analysis"], run["convention"],
        )

    # A Phase 3 record claiming a different feature cache is refused before
    # any pair is compared.
    stale = dict(meta)
    stale["cache_provenance"] = {
        "dinov2_vitb14": {**meta["cache_provenance"]["dinov2_vitb14"],
                          "features_digest": "0" * 32}
    }
    forged = tmp_path / "provenance"
    forged.mkdir()
    write_rows(forged / f"{SCENE}.parquet", pq.read_table(source).to_pylist(), stale)
    cfg = dataclasses.replace(run["cfg"], phase3_eval_dir=forged)
    with pytest.raises(Phase4GateError, match="feature cache"):
        evaluate_scene_phase4(
            cfg, SCENE, run["mean_vector"], run["analysis"], run["convention"],
        )


def test_report_refuses_a_partial_arm_and_counts_empty_levels(phase4_run):
    """A lost arm is corruption; a whole absent level is counted population."""
    from lot.phase4_report import build_records

    run = phase4_run
    analysis = run["analysis"]
    rows = run["rows"]

    # Remove one no-alignment estimate row: its ceiling and floor survive, so
    # the slot is partial and the directory is corrupt, not smaller.
    partial = [
        r for r in rows
        if not (r["level"] == "none" and r["population"] == "matched"
                and r["variant"] == VGGT_NO_ALIGN
                and r["path"] == PER_POINT
                and r["context_frame_id"] == rows[0]["context_frame_id"]
                and r["target_frame_id"] == rows[0]["target_frame_id"])
    ]
    assert len(partial) < len(rows)
    with pytest.raises(ValueError, match="matched arms are missing"):
        build_records(partial, analysis)

    # Remove a whole level for one pair-path: legitimately absent, counted.
    key = (rows[0]["context_frame_id"], rows[0]["target_frame_id"])
    absent = [
        r for r in rows
        if not (r["level"] == "none"
                and (r["context_frame_id"], r["target_frame_id"]) == key
                and r["path"] == PER_POINT)
    ]
    _, exclusions = build_records(absent, analysis)
    assert exclusions["empty_scored_sets_by_level"].get("none", 0) >= 1


def test_confidence_masked_depth_shrinks_validity_identically(tmp_path_factory):
    """5a-invalid pixels are NaN in the maps every consumer reads (finding 7),
    and the surviving sets stay identical across multiplicative levels as sets
    (finding 8), not merely as counts."""
    root = tmp_path_factory.mktemp("phase4_masked")

    def holed(gt):
        est = (gt / EST_SCALE).astype(np.float32)
        est[:, :14] = -1.0
        return est

    manifest = build_scene(root, est_transform=holed)
    cfg = Phase4Config(
        experiment_name="phase4_masked", renders_root=root, cache_root=root / "cache",
        output_root=root / "out", scenes=[SCENE], mean_vector_dir=root / "out" / "mv",
    )
    analysis = load_analysis_config()
    phase3_eval_dir, _ = run_phase3(root, cfg, analysis)
    cfg.phase3_eval_dir = phase3_eval_dir
    diagnostic = convention_report(cfg, analysis)
    convention = build_convention_record(
        diagnostic, planar_authority(),
        json.loads((cache_dir(cfg.cache_root, "vggt_1b", SCENE) / "meta.json").read_text()),
    )
    from lot.evaluate import load_or_build_mean_vector

    mean_vector = load_or_build_mean_vector(
        cfg.cache_root, "dinov2_vitb14", [SCENE], cfg.mean_vector_dir
    )
    rows, evidence = evaluate_scene_phase4(cfg, SCENE, mean_vector, analysis, convention)
    assert rows
    shrunk = 0
    for audit in evidence["audits"].values():
        valid_counts = {
            level: audit["levels"][level]["pp_transport_valid"]
            for level in ("none", "scene", "image")
            if level in audit["levels"]
        }
        # Identity across levels held internally as sets; visible here as one
        # value, and strictly below the full sample count wherever a read
        # touched the invalidated stripe.
        assert len(set(valid_counts.values())) == 1
        matched = [
            r for r in rows
            if r["level"] == "none" and r["population"] == "matched"
            and r["path"] == PER_POINT
        ]
        if matched and min(valid_counts.values()) < max(r["n_gt"] for r in matched):
            shrunk += 1
    assert shrunk > 0, "the invalid stripe never reduced transport validity"
