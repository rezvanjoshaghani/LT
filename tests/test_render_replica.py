"""PLAN Phase 1: camera programs, pose conventions, depth convention, manifest, QC.

Everything here runs without Habitat-Sim. The rendering path itself is
covered by test_habitat_render_smoke, which runs only where habitat_sim and
the Replica dataset are available (the cluster).
"""

import dataclasses
import importlib.util
import json
import math

import numpy as np
import pytest
import torch

from lot.geometry import (
    parallax,
    project,
    relative_pose,
    transform_points,
    invert_se3,
)
from lot.render_replica import (
    FRAME_STATS_NAME,
    REGIMES,
    REPLICA_SCENES,
    REPLICA_SCENES_TEST,
    REPLICA_SCENES_TRAIN,
    FrameRecord,
    Manifest,
    RenderConfig,
    classify_depth_convention,
    euclidean_to_planar_depth,
    frame_depth_stats,
    frame_is_usable,
    frame_stats_summary,
    load_frame_stats,
    rotation_position_residuals,
    usable_frame_ids,
    write_frame_stats,
    intrinsics_from_hfov,
    load_config,
    load_manifest,
    look_at_cv,
    opencv_pose_from_opengl,
    opengl_pose_from_opencv,
    program_orbit,
    program_rotation,
    program_translation,
    quat_to_rotmat,
    ray_norm_map,
    rotmat_to_quat,
    scene_seed,
    validate_manifest,
    write_contact_sheet,
    write_manifest,
)
from scenes import random_se3, rodrigues

HAVE_HABITAT = importlib.util.find_spec("habitat_sim") is not None

UP = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)


def rotation_angle_deg(R: torch.Tensor) -> float:
    c = (float(torch.trace(R)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def base_pose(yaw_deg: float = 33.0) -> torch.Tensor:
    """A pitchless pose at a generic position with a generic horizontal heading."""
    eye = torch.tensor([1.2, 1.5, -0.7], dtype=torch.float64)
    a = math.radians(yaw_deg)
    direction = torch.tensor([math.sin(a), 0.0, math.cos(a)], dtype=torch.float64)
    return look_at_cv(eye, eye + direction, UP)


# ---------------------------------------------------------------------------
# Pose conventions
# ---------------------------------------------------------------------------

def test_axis_flip_is_an_involution():
    g = torch.Generator().manual_seed(7)
    for _ in range(10):
        T = random_se3(g)
        back = opengl_pose_from_opencv(opencv_pose_from_opengl(T))
        assert torch.allclose(back, T, atol=1e-12)


def test_gl_identity_maps_to_cv_axis_flip():
    T_cv = opencv_pose_from_opengl(torch.eye(4, dtype=torch.float64))
    # A GL camera at identity looks down world -z with +y up. In OpenCV axes
    # that camera has x = +x_world, y (down) = -y_world, z (forward) = -z_world.
    expected = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))
    assert torch.allclose(T_cv, expected)


def test_quaternion_roundtrip_and_known_rotation():
    g = torch.Generator().manual_seed(11)
    for _ in range(20):
        R = random_se3(g)[:3, :3]
        w, x, y, z = rotmat_to_quat(R)
        assert w >= 0
        assert torch.allclose(quat_to_rotmat(w, x, y, z), R, atol=1e-9)
    # 90 degrees about +y: q = (cos 45, 0, sin 45, 0).
    R90 = quat_to_rotmat(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
    expected = rodrigues(torch.tensor([0.0, 1.0, 0.0]), math.pi / 2)
    assert torch.allclose(R90, expected, atol=1e-12)


def test_intrinsics_from_hfov():
    K = intrinsics_from_hfov(224, 224, 2 * math.degrees(math.atan(0.5)))
    # hfov chosen so fx = width: tan(hfov / 2) = (W / 2) / W = 0.5.
    assert torch.allclose(K[0, 0], torch.tensor(224.0, dtype=torch.float64))
    assert K[0, 0] == K[1, 1]
    assert float(K[0, 2]) == 111.5 and float(K[1, 2]) == 111.5
    K90 = intrinsics_from_hfov(518, 518, 90.0)
    assert abs(float(K90[0, 0]) - 259.0) < 1e-9
    with pytest.raises(ValueError):
        intrinsics_from_hfov(224, 224, 0.0)


def test_look_at_geometry():
    eye = torch.tensor([0.5, 1.5, -2.0], dtype=torch.float64)
    target = torch.tensor([3.0, 1.0, 4.0], dtype=torch.float64)
    T = look_at_cv(eye, target, UP)
    R = T[:3, :3]
    assert torch.allclose(R @ R.mT, torch.eye(3, dtype=torch.float64), atol=1e-12)
    assert abs(float(torch.linalg.det(R)) - 1.0) < 1e-12
    # The target sits on the optical axis: it projects to the principal point.
    K = intrinsics_from_hfov(224, 224, 90.0)
    target_cam = transform_points(invert_se3(T), target)
    uv, z = project(target_cam, K)
    assert float(z) > 0
    assert torch.allclose(uv, torch.tensor([111.5, 111.5], dtype=torch.float64), atol=1e-9)
    # Camera y points downward: against world up.
    assert float(T[:3, 1] @ UP) < 0
    with pytest.raises(ValueError):
        look_at_cv(eye, eye + torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64), UP)


# ---------------------------------------------------------------------------
# Camera programs
# ---------------------------------------------------------------------------

YAWS = [-30.0, -22.5, -15.0, -7.5, 0.0, 7.5, 15.0, 22.5, 30.0]
PITCHES = [-15.0, -7.5, 7.5, 15.0]


def test_program_rotation_fixed_position_and_steps():
    T_base = base_pose()
    frames = program_rotation(T_base, YAWS, PITCHES)
    assert len(frames) == len(YAWS) + len(PITCHES)
    assert all(f.regime == "rotation" for f in frames)
    for f in frames:
        assert torch.allclose(f.T_world_from_camera[:3, 3], T_base[:3, 3])
    zero = [f for f in frames if f.params["yaw_deg"] == 0 and f.params["pitch_deg"] == 0]
    assert len(zero) == 1
    assert torch.allclose(zero[0].T_world_from_camera, T_base, atol=1e-12)
    yaw_frames = [f for f in frames if f.params["sweep"] == "yaw"]
    for a, b in zip(yaw_frames, yaw_frames[1:]):
        T_rel = relative_pose(b.T_world_from_camera, a.T_world_from_camera)
        assert abs(rotation_angle_deg(T_rel[:3, :3]) - 7.5) < 1e-9
        assert float(torch.linalg.vector_norm(T_rel[:3, 3])) < 1e-12
    # Each frame's rotation relative to base matches its recorded offset.
    for f in frames:
        T_rel = relative_pose(f.T_world_from_camera, T_base)
        offset = abs(f.params["yaw_deg"]) + abs(f.params["pitch_deg"])
        assert abs(rotation_angle_deg(T_rel[:3, :3].mT) - offset) < 1e-9


def test_program_translation_hits_parallax_targets():
    T_base = base_pose()
    median_depth = 3.7
    values = [0.05, 0.1, 0.2, 0.4]
    frames = program_translation(T_base, values, median_depth)
    assert len(frames) == 1 + 2 * 2 * len(values)
    assert all(f.regime == "translation" for f in frames)
    zero = [f for f in frames if f.params["parallax_target"] == 0]
    assert len(zero) == 1 and torch.allclose(zero[0].T_world_from_camera, T_base)
    depth_map = torch.full((8, 8), median_depth, dtype=torch.float64)
    for f in frames:
        assert torch.allclose(f.T_world_from_camera[:3, :3], T_base[:3, :3])
        T_rel = relative_pose(f.T_world_from_camera, T_base)
        p = float(parallax(T_rel, depth_map))
        assert abs(p - f.params["parallax_target"]) < 1e-12
        move = f.T_world_from_camera[:3, 3] - T_base[:3, 3]
        if f.params["axis"] == "lateral":
            expected = f.params["sign"] * f.params["baseline_m"] * T_base[:3, 0]
            assert torch.allclose(move, expected, atol=1e-12)
        elif f.params["axis"] == "forward":
            expected = f.params["sign"] * f.params["baseline_m"] * T_base[:3, 2]
            assert torch.allclose(move, expected, atol=1e-12)
    with pytest.raises(ValueError):
        program_translation(T_base, values, 0.0)


def test_program_orbit_radii_anchor_and_steps():
    T_base = base_pose()
    d = 3.0
    scales = [0.6, 1.0]
    azimuths = [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    frames = program_orbit(T_base, d, scales, azimuths, UP)
    assert len(frames) == len(scales) * len(azimuths)
    assert all(f.regime == "orbit" for f in frames)
    anchor = T_base[:3, 3] + d * T_base[:3, 2]
    K = intrinsics_from_hfov(224, 224, 90.0)
    for f in frames:
        assert torch.allclose(
            torch.tensor(f.params["anchor_world"], dtype=torch.float64), anchor
        )
        pos = f.T_world_from_camera[:3, 3]
        r = float(torch.linalg.vector_norm(pos - anchor))
        assert abs(r - f.params["radius_m"]) < 1e-9
        assert abs(f.params["radius_m"] - f.params["radius_scale"] * d) < 1e-12
        # The anchor projects to the principal point in every orbit frame.
        anchor_cam = transform_points(invert_se3(f.T_world_from_camera), anchor)
        uv, z = project(anchor_cam, K)
        assert float(z) > 0
        assert torch.allclose(
            uv, torch.tensor([111.5, 111.5], dtype=torch.float64), atol=1e-6
        )
    # Azimuth zero at scale 1.0 reproduces the base pose exactly.
    home = [
        f for f in frames
        if f.params["radius_scale"] == 1.0 and f.params["azimuth_deg"] == 0.0
    ]
    assert len(home) == 1
    assert torch.allclose(home[0].T_world_from_camera, T_base, atol=1e-9)
    # Consecutive azimuths subtend the configured step at the anchor.
    ring = [f for f in frames if f.params["radius_scale"] == 0.6]
    for a, b in zip(ring, ring[1:]):
        va = a.T_world_from_camera[:3, 3] - anchor
        vb = b.T_world_from_camera[:3, 3] - anchor
        cos = float(va @ vb / (torch.linalg.vector_norm(va) * torch.linalg.vector_norm(vb)))
        assert abs(math.degrees(math.acos(max(-1.0, min(1.0, cos)))) - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# Depth convention
# ---------------------------------------------------------------------------

def test_depth_convention_classification_and_conversion():
    K = intrinsics_from_hfov(224, 224, 90.0)
    wall_z = 2.5
    planar = torch.full((224, 224), wall_z, dtype=torch.float64)
    euclid = planar * ray_norm_map(224, 224, K)

    planar_stats = classify_depth_convention(planar, K)
    assert planar_stats["verdict"] == "planar_z"
    assert planar_stats["spread_planar"] < planar_stats["spread_euclidean"]

    euclid_stats = classify_depth_convention(euclid, K)
    assert euclid_stats["verdict"] == "euclidean_ray"

    recovered = euclidean_to_planar_depth(euclid, K)
    assert torch.allclose(recovered, planar, atol=1e-12)

    # Moderate clutter inside the crop does not flip the verdict.
    cluttered = planar.clone()
    cluttered[90:130, 90:130] = 0.9
    assert classify_depth_convention(cluttered, K)["verdict"] == "planar_z"

    # A structureless map is ambiguous, not silently classified.
    g = torch.Generator().manual_seed(3)
    noise = torch.rand((224, 224), generator=g, dtype=torch.float64) * 5 + 0.5
    assert classify_depth_convention(noise, K)["verdict"] == "ambiguous"

    # Mostly invalid depth is ambiguous.
    holes = planar.clone()
    holes[:, :] = 0.0
    assert classify_depth_convention(holes, K)["verdict"] == "ambiguous"


def test_depth_convention_tolerates_tilted_planes():
    """The cluster failure mode: scans a few degrees off gravity alignment.

    A camera looking straight down at a floor tilted by angle t sees planar
    z-depth h / (1 - tan(t) x'), an affine-up-to-negligible-curvature
    function of the normalized image coordinate. The constant-depth test
    rejected these; the plane-fit test must classify them.
    """
    K = intrinsics_from_hfov(224, 224, 90.0)
    h = 1.5
    uv = torch.stack(
        torch.meshgrid(
            torch.arange(224, dtype=torch.float64),
            torch.arange(224, dtype=torch.float64),
            indexing="ij",
        ),
        dim=-1,
    )
    xprime = (uv[..., 1] - K[0, 2]) / K[0, 0]
    for tilt_deg in (2.0, 4.0, 6.0):
        tilted = h / (1.0 - math.tan(math.radians(tilt_deg)) * xprime)
        stats = classify_depth_convention(tilted, K)
        assert stats["verdict"] == "planar_z", (tilt_deg, stats)
        tilted_euclid = tilted * ray_norm_map(224, 224, K)
        stats_e = classify_depth_convention(tilted_euclid, K)
        assert stats_e["verdict"] == "euclidean_ray", (tilt_deg, stats_e)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def fake_scene_dir(tmp_path, n_frames: int = 3, scene: str = "room_0"):
    from PIL import Image

    root = tmp_path / scene
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    K = intrinsics_from_hfov(28, 28, 90.0)
    T_base = base_pose()
    frames = []
    posed = program_rotation(T_base, [-7.5, 0.0, 7.5], [])[:n_frames]
    rng = np.random.default_rng(0)
    for i, pf in enumerate(posed):
        frame_id = f"{scene}_vp00_rotation_{i:03d}"
        rgb_rel = f"rgb/{frame_id}.png"
        depth_rel = f"depth/{frame_id}.npy"
        rgb = rng.integers(0, 255, size=(28, 28, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(root / rgb_rel)
        np.save(root / depth_rel, np.full((28, 28), 2.0, dtype=np.float32))
        frames.append(
            FrameRecord(
                frame_id=frame_id,
                scene=scene,
                regime=pf.regime,
                params=dict(pf.params, viewpoint=0),
                T_world_from_camera=pf.T_world_from_camera,
                K=K,
                height=28,
                width=28,
                rgb_path=rgb_rel,
                depth_path=depth_rel,
            )
        )
    metadata = {
        "scene": scene,
        "depth_convention": {
            "raw_verdict": "planar_z",
            "converted_to_planar": False,
            "stored_depth": "planar_z",
            "probes": [],
        },
    }
    return root, Manifest(scene=scene, metadata=metadata, frames=frames)


def test_manifest_roundtrip_and_validation(tmp_path):
    root, manifest = fake_scene_dir(tmp_path)
    path = root / "manifest.json"
    write_manifest(path, manifest)
    loaded = load_manifest(path)
    assert loaded.scene == manifest.scene
    assert len(loaded.frames) == len(manifest.frames)
    for a, b in zip(loaded.frames, manifest.frames):
        assert a.frame_id == b.frame_id
        assert a.regime == b.regime
        assert a.params == b.params
        assert torch.allclose(a.T_world_from_camera, b.T_world_from_camera, atol=1e-12)
        assert torch.allclose(a.K, b.K, atol=1e-12)
        assert (a.height, a.width) == (b.height, b.width)
    validate_manifest(loaded, root, check_files=True)


def test_manifest_validation_catches_problems(tmp_path):
    root, manifest = fake_scene_dir(tmp_path)
    path = root / "manifest.json"
    write_manifest(path, manifest)

    # Missing depth file.
    (root / manifest.frames[1].depth_path).unlink()
    with pytest.raises(ValueError, match="missing depth"):
        validate_manifest(load_manifest(path), root, check_files=True)
    # Still passes structurally without file checks.
    validate_manifest(load_manifest(path), root, check_files=False)

    # Non-orthonormal rotation.
    payload = json.loads(path.read_text())
    payload["frames"][0]["T_world_from_camera"][0][0] = 2.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="orthonormal"):
        validate_manifest(load_manifest(path), root, check_files=False)

    # Duplicate frame id.
    root2, manifest2 = fake_scene_dir(tmp_path / "dup")
    bad = Manifest(
        scene=manifest2.scene,
        metadata=manifest2.metadata,
        frames=[manifest2.frames[0], manifest2.frames[0]],
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(bad, root2, check_files=False)

    # Unknown regime.
    bad_frame = FrameRecord(
        **{**dataclasses_asdict_frame(manifest2.frames[0]), "regime": "spline"}
    )
    with pytest.raises(ValueError, match="regime"):
        validate_manifest(
            Manifest(manifest2.scene, manifest2.metadata, [bad_frame]),
            root2,
            check_files=False,
        )

    # Unresolved depth convention.
    meta = dict(manifest2.metadata)
    meta["depth_convention"] = dict(meta["depth_convention"], raw_verdict="ambiguous")
    with pytest.raises(ValueError, match="unresolved"):
        validate_manifest(
            Manifest(manifest2.scene, meta, manifest2.frames), root2, check_files=False
        )

    # Missing depth convention entirely.
    with pytest.raises(ValueError, match="depth_convention"):
        validate_manifest(
            Manifest(manifest2.scene, {}, manifest2.frames), root2, check_files=False
        )


def dataclasses_asdict_frame(f: FrameRecord) -> dict:
    return {
        "frame_id": f.frame_id,
        "scene": f.scene,
        "regime": f.regime,
        "params": f.params,
        "T_world_from_camera": f.T_world_from_camera,
        "K": f.K,
        "height": f.height,
        "width": f.width,
        "rgb_path": f.rgb_path,
        "depth_path": f.depth_path,
    }


# ---------------------------------------------------------------------------
# QC and config
# ---------------------------------------------------------------------------

def test_contact_sheet_smoke(tmp_path):
    rng = np.random.default_rng(1)
    entries = []
    for i in range(5):
        rgb = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        depth = rng.uniform(0.5, 5.0, size=(32, 32)).astype(np.float32)
        depth[0, 0] = 0.0  # invalid pixel must not break rendering
        entries.append((f"frame_{i}", rgb, depth))
    # A frame with no valid depth at all must render (black tile), not crash.
    entries.append(
        (
            "all_invalid",
            rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32), dtype=np.float32),
        )
    )
    out = tmp_path / "qc" / "qc_rotation.png"
    write_contact_sheet(out, entries, ncols=3)
    assert out.is_file() and out.stat().st_size > 1000
    with pytest.raises(ValueError):
        write_contact_sheet(tmp_path / "empty.png", [])


def test_write_scene_qc_from_manifest(tmp_path):
    from lot.render_replica import write_scene_qc

    root, manifest = fake_scene_dir(tmp_path)
    write_scene_qc(root, manifest, frames_per_regime=2)
    out = root / "qc" / "qc_rotation.png"
    assert out.is_file() and out.stat().st_size > 1000


def test_validate_only_cli_reports_per_scene(tmp_path, capsys):
    from lot.render_replica import main

    root, manifest = fake_scene_dir(tmp_path)  # creates tmp_path/room_0
    write_manifest(root / "manifest.json", manifest)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "replica_root: /r\n"
        f"output_root: {tmp_path.as_posix()}\n"
        "scenes: [room_0, room_1]\n"
    )
    # One valid scene, one never rendered: every scene gets a status line
    # and the exit names the failures instead of dying on the first one.
    with pytest.raises(SystemExit, match="room_1"):
        main(["--config", str(cfg), "--validate-only"])
    out = capsys.readouterr().out
    assert "[room_0] manifest valid" in out
    assert "[room_1] MISSING" in out
    # Restricted to the valid scene, validation passes cleanly.
    main(["--config", str(cfg), "--scene", "room_0", "--validate-only"])
    assert "all 1 scenes valid" in capsys.readouterr().out


def test_scene_split_is_canonical():
    assert len(REPLICA_SCENES) == 18
    assert len(REPLICA_SCENES_TRAIN) == 13
    assert len(REPLICA_SCENES_TEST) == 5
    assert len(set(REPLICA_SCENES)) == 18
    assert set(REPLICA_SCENES_TRAIN) & set(REPLICA_SCENES_TEST) == set()


def test_load_config(tmp_path):
    cfg_path = tmp_path / "render.yaml"
    cfg_path.write_text(
        "replica_root: /data/replica\n"
        "output_root: data/replica_renders\n"
        "scenes: [room_0]\n"
        "seed: 3\n"
    )
    cfg = load_config(cfg_path)
    assert cfg.scenes == ["room_0"]
    assert cfg.seed == 3
    assert cfg.image_height == 518 and cfg.image_width == 518
    assert cfg.image_height % 14 == 0
    assert len(cfg.yaw_offsets_deg) == 9
    assert scene_seed(cfg.seed, "room_0") != scene_seed(cfg.seed, "room_1")
    assert scene_seed(3, "room_0") == scene_seed(3, "room_0")

    cfg_path.write_text(
        "replica_root: /data/replica\noutput_root: out\nscenes: [room_0]\nbogus: 1\n"
    )
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(cfg_path)

    cfg_path.write_text("replica_root: /data/replica\nscenes: [room_0]\n")
    with pytest.raises(ValueError, match="missing required"):
        load_config(cfg_path)

    cfg_path.write_text(
        "replica_root: /r\noutput_root: o\nscenes: [not_a_scene]\n"
    )
    with pytest.raises(ValueError, match="unknown Replica scenes"):
        load_config(cfg_path)

    cfg_path.write_text(
        "replica_root: /r\noutput_root: o\nscenes: [room_0]\nimage_height: 100\n"
    )
    with pytest.raises(ValueError, match="multiple of the patch size"):
        load_config(cfg_path)


def test_repo_configs_load():
    from pathlib import Path

    configs = Path(__file__).resolve().parents[1] / "configs"
    pilot = load_config(configs / "render_replica_pilot.yaml")
    assert pilot.scenes == ["room_0"]
    full = load_config(configs / "render_replica_all.yaml")
    assert sorted(full.scenes) == sorted(REPLICA_SCENES)
    assert pilot.seed == full.seed


@pytest.mark.skipif(HAVE_HABITAT, reason="habitat_sim is installed")
def test_render_requires_habitat(tmp_path):
    from lot.render_replica import render_scene

    cfg = RenderConfig(
        replica_root=tmp_path, output_root=tmp_path / "out", scenes=["room_0"]
    )
    with pytest.raises(RuntimeError, match="habitat_sim"):
        render_scene(cfg, "room_0")


@pytest.mark.skipif(not HAVE_HABITAT, reason="needs habitat_sim and Replica data")
def test_habitat_render_smoke(tmp_path):
    """Cluster-side smoke test: tiny end-to-end render of the pilot scene.

    Requires REPLICA_ROOT to point at the Replica dataset root.
    """
    import os

    replica_root = os.environ.get("REPLICA_ROOT")
    if not replica_root:
        pytest.skip("REPLICA_ROOT not set")
    from lot.render_replica import MANIFEST_NAME, render_scene

    cfg = RenderConfig(
        replica_root=replica_root,
        output_root=tmp_path / "renders",
        scenes=["room_0"],
        viewpoints_per_scene=1,
        yaw_offsets_deg=[-7.5, 0.0, 7.5],
        pitch_offsets_deg=[7.5],
        parallax_values=[0.1],
        orbit_azimuth_offsets_deg=[-5.0, 0.0, 5.0],
        qc_frames_per_regime=4,
    )
    manifest_path = render_scene(cfg, "room_0")
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest, manifest_path.parent, check_files=True)
    assert set(f.regime for f in manifest.frames) == set(REGIMES)
    dc = manifest.metadata["depth_convention"]
    assert dc["raw_verdict"] in ("planar_z", "euclidean_ray")
    for regime in REGIMES:
        assert (manifest_path.parent / "qc" / f"qc_{regime}.png").is_file()
    with pytest.raises(FileExistsError):
        render_scene(cfg, "room_0")


# ---------------------------------------------------------------------------
# Per-frame depth quality
# ---------------------------------------------------------------------------

def _reject_constant(name):
    raise AssertionError(f"non-standard JSON token {name}")


def test_frame_depth_stats_on_a_clean_frame():
    stats = frame_depth_stats(np.full((28, 28), 3.0, dtype=np.float32))
    assert stats["valid_fraction"] == 1.0
    assert stats["median_m"] == 3.0
    assert stats["center_p01_m"] == 3.0
    assert frame_is_usable(stats)


def test_camera_inside_geometry_is_not_usable():
    """The hazard the gate exists for: the lens buried in a surface."""
    stats = frame_depth_stats(np.full((28, 28), 0.04, dtype=np.float32))
    assert not frame_is_usable(stats)


def test_pointing_into_unscanned_space_is_not_usable():
    depth = np.zeros((28, 28), dtype=np.float32)
    depth[:8] = 3.0
    assert not frame_is_usable(frame_depth_stats(depth))


def test_close_and_distant_views_stay_usable():
    """Median depth stratifies the study; it must never gate it.

    A pitch sweep looking at a floor a metre away and a view down a long
    corridor are both ordinary views, and they sit at opposite ends of the
    parallax axis the whole study is measured against.
    """
    close = frame_depth_stats(np.full((28, 28), 1.0, dtype=np.float32))
    distant = frame_depth_stats(np.full((28, 28), 12.0, dtype=np.float32))
    assert frame_is_usable(close)
    assert frame_is_usable(distant)


def test_clearance_uses_a_percentile_not_the_minimum():
    """One stray near pixel must not veto a frame, a buried lens must."""
    depth = np.full((28, 28), 3.0, dtype=np.float32)
    depth[14, 14] = 0.02
    assert frame_is_usable(frame_depth_stats(depth))
    depth[7:21, 7:21] = 0.02
    assert not frame_is_usable(frame_depth_stats(depth))


def test_all_invalid_depth_is_not_usable_and_does_not_raise():
    stats = frame_depth_stats(np.zeros((28, 28), dtype=np.float32))
    assert stats["valid_fraction"] == 0.0
    assert math.isnan(stats["median_m"])
    assert not frame_is_usable(stats)


def test_frame_stats_sidecar_stores_measurements_not_verdicts(tmp_path):
    """A stored verdict would outlive the reasoning behind it."""
    root, manifest = fake_scene_dir(tmp_path)
    payload = write_frame_stats(root, manifest)
    assert payload["total"] == len(manifest.frames)
    for stats in payload["frames"].values():
        assert "passes" not in stats
        assert set(stats) == {"valid_fraction", "median_m", "center_p01_m", "min_m", "max_m"}
    assert usable_frame_ids(payload) == {f.frame_id for f in manifest.frames}
    assert load_frame_stats(root / FRAME_STATS_NAME)["scene"] == manifest.scene
    with pytest.raises(FileExistsError):
        write_frame_stats(root, manifest)


def test_policy_can_change_without_rereading_depth(tmp_path):
    root, manifest = fake_scene_dir(tmp_path)
    payload = write_frame_stats(root, manifest)
    # fake_scene_dir writes a constant 2.0 m depth everywhere.
    assert len(usable_frame_ids(payload)) == len(manifest.frames)
    assert usable_frame_ids(payload, min_clearance_m=5.0) == set()


def test_frame_stats_summary_counts_by_regime(tmp_path):
    root, manifest = fake_scene_dir(tmp_path)
    summary = frame_stats_summary(write_frame_stats(root, manifest))
    assert summary["usable"] == len(manifest.frames)
    assert summary["by_regime"]["rotation"] == len(manifest.frames)
    assert summary["by_regime"]["orbit"] == 0


def test_frame_stats_sidecar_is_standard_json(tmp_path):
    """A frame with no valid depth writes null, not a bare NaN token."""
    root, manifest = fake_scene_dir(tmp_path)
    np.save(root / manifest.frames[0].depth_path, np.zeros((28, 28), dtype=np.float32))
    payload = write_frame_stats(root, manifest)
    assert len(usable_frame_ids(payload)) == len(manifest.frames) - 1
    json.loads((root / FRAME_STATS_NAME).read_text(encoding="utf-8"),
               parse_constant=_reject_constant)


def test_usability_survives_the_json_round_trip(tmp_path):
    """Non-finite statistics write as null and must read back as nan.

    A predicate that works on freshly measured stats and raises on reloaded
    ones would fail only on a resumed run, which is the run nobody watches.
    """
    root, manifest = fake_scene_dir(tmp_path)
    np.save(root / manifest.frames[0].depth_path, np.zeros((28, 28), dtype=np.float32))
    write_frame_stats(root, manifest)
    reloaded = load_frame_stats(root / FRAME_STATS_NAME)
    assert reloaded["frames"][manifest.frames[0].frame_id]["median_m"] is None
    assert len(usable_frame_ids(reloaded)) == len(manifest.frames) - 1
    assert frame_stats_summary(reloaded)["usable"] == len(manifest.frames) - 1


# ---------------------------------------------------------------------------
# PROTOCOL 3.3: in-place rotation's zero translation, asserted from the manifest
# ---------------------------------------------------------------------------

def test_rotation_frames_sharing_a_position_pass(tmp_path):
    root, manifest = fake_scene_dir(tmp_path)
    residuals = rotation_position_residuals(manifest)
    assert residuals and max(residuals.values()) < 1e-9
    diagnostics = validate_manifest(
        manifest, root, check_files=True, rotation_position_bound_m=1e-6
    )
    assert diagnostics["rotation_position_residuals_m"] == residuals


def test_a_shifted_rotation_frame_is_rejected(tmp_path):
    """The regime's defining property is checked, not trusted.

    Manifest poses are read back from the simulator rather than the planned
    poses, so a drift can appear without anything else noticing. It matters
    twice: a rotation pair whose spread exceeds the zero-parallax tolerance
    silently leaves the zero bin, and PROTOCOL 4.5 makes exactly-zero
    translation a hard invariant for the Phase 4 gate.
    """
    root, manifest = fake_scene_dir(tmp_path)
    shifted = list(manifest.frames)
    moved = shifted[1].T_world_from_camera.clone()
    moved[0, 3] += 1e-3
    shifted[1] = dataclasses.replace(shifted[1], T_world_from_camera=moved)
    broken = dataclasses.replace(manifest, frames=shifted)

    residuals = rotation_position_residuals(broken)
    assert max(residuals.values()) > 1e-4
    with pytest.raises(ValueError, match="do not share a camera position"):
        validate_manifest(broken, root, check_files=False, rotation_position_bound_m=1e-6)
    # Without the bound the assertion is skipped, for callers with no config.
    validate_manifest(broken, root, check_files=False, rotation_position_bound_m=None)


def test_the_bound_comes_from_the_analysis_config():
    from lot.analysis_config import load_analysis_config

    assert load_analysis_config().rotation_position_bound_m > 0


def test_translation_frames_must_share_one_orientation(tmp_path):
    """The mirror of the rotation-program position check.

    PROTOCOL 3.3 makes translation the sole source of the primary parallax
    curve precisely because it holds rotation at exactly zero. A translation
    frame whose orientation drifted puts an unlabelled rotation into the
    marginal that exists to exclude it, and nothing downstream can see it: the
    regime tag is what routes a pair onto the curve.
    """
    from lot.render_replica import translation_rotation_residuals, validate_manifest

    frames = program_translation(base_pose(), [0.1, 0.2, 0.3], 3.0)
    records = []
    for index, frame in enumerate(frames):
        records.append(
            FrameRecord(
                frame_id=f"room_0_vp00_translation_{index:03d}",
                scene="room_0",
                regime="translation",
                params=dict(frame.params, viewpoint=0),
                T_world_from_camera=frame.T_world_from_camera,
                K=intrinsics_from_hfov(28, 28, 90.0),
                height=28,
                width=28,
                rgb_path=f"rgb/{index}.png",
                depth_path=f"depth/{index}.npy",
            )
        )
    manifest = Manifest(
        scene="room_0",
        metadata={"depth_convention": {"raw_verdict": "planar_z", "stored_depth": "planar_z"}},
        frames=records,
    )
    assert max(translation_rotation_residuals(manifest).values()) < 1e-9
    validate_manifest(
        manifest, tmp_path, check_files=False, translation_rotation_bound_deg=1e-4
    )

    # Two degrees of yaw about the camera's own vertical axis, which is what a
    # malformed pose or a lossy read-back would look like.
    a = math.radians(2.0)
    yaw = torch.eye(4, dtype=torch.float64)
    yaw[:3, :3] = torch.tensor(
        [[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]],
        dtype=torch.float64,
    )
    records[-1] = dataclasses.replace(
        records[-1], T_world_from_camera=records[-1].T_world_from_camera @ yaw
    )
    drifted = dataclasses.replace(manifest, frames=records)
    assert max(translation_rotation_residuals(drifted).values()) == pytest.approx(2.0, abs=1e-6)
    with pytest.raises(ValueError, match="do not share one orientation"):
        validate_manifest(
            drifted, tmp_path, check_files=False, translation_rotation_bound_deg=1e-4
        )
