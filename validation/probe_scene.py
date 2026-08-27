"""A synthetic scene complete enough to drive lot.evaluate end to end.

Deliberately imports nothing from lot at module import time. The mutation
drivers put a mutant's src on sys.path first and then import this module, so a
top-level lot import here would bind the wrong package.
"""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np

SIDE = 112         # 8 x 8 patches at stride 14
CHANNELS = 768    # dinov2_vitb14's real width, so the cache the probe
                  # fabricates is one the validator will accept as that
                  # encoder's. A narrower fiction was caught by cache
                  # validation, which is the validator working.
SCENE = "room_0"


def base_pose():
    import torch

    T = torch.eye(4, dtype=torch.float64)
    T[:3, 3] = torch.tensor([0.0, 1.5, 0.0], dtype=torch.float64)
    return T


def smooth_features(hp, wp, channels, seed):
    """Spatially correlated features.

    White noise would make a within-image shuffle exactly equal to a uniform
    random read, which is the degenerate case VALIDATION 3.7 warns against.
    Smooth features keep the shuffle control meaningful.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:hp, 0:wp].astype(np.float64)
    out = np.zeros((channels, hp, wp))
    for c in range(channels):
        fx, fy = rng.uniform(0.2, 1.1, 2)
        ph = rng.uniform(0, 6.28, 2)
        out[c] = np.sin(fx * x + ph[0]) + np.cos(fy * y + ph[1]) + 0.15 * rng.normal(size=(hp, wp))
    return out.astype(np.float16)


def surface_attached_features(frame, depth, channels, patch=14):
    """Features that really are a property of the surface they sit on.

    Each patch centre pixel is lifted with the frame's own depth and pose to a
    world point, and the feature is a fixed smooth function of that world point.
    Two frames therefore assign nearly the same value to the same physical
    surface, which is exactly the property Experiment Zero measures.

    Random features cannot serve here: with them Oracle-Transport has no real
    advantage over No-Warp-Copy, the paired margin sits at zero, and VALIDATION
    3.7's "destroys at least half the margin" criterion has nothing to bite on.
    """
    import torch

    H, W = depth.shape
    hp, wp = H // patch, W // patch
    K = frame.K.numpy().astype(np.float64)
    T = frame.T_world_from_camera.numpy().astype(np.float64)

    p = np.arange(hp) * patch + (patch - 1) / 2.0
    q = np.arange(wp) * patch + (patch - 1) / 2.0
    vv, uu = np.meshgrid(p, q, indexing="ij")
    iv = np.rint(vv).astype(int)
    iu = np.rint(uu).astype(int)
    d = depth.astype(np.float64)[iv, iu]

    x = (uu - K[0, 2]) * d / K[0, 0]
    y = (vv - K[1, 2]) * d / K[1, 1]
    cam = np.stack((x, y, d), axis=-1)
    world = cam @ T[:3, :3].T + T[:3, 3]

    rng = np.random.default_rng(4242)          # same basis for every frame
    out = np.zeros((channels, hp, wp))
    for c in range(channels):
        k = rng.normal(size=3) * 1.4
        ph = rng.uniform(0, 6.28)
        out[c] = np.sin(world @ k + ph)
    return out.astype(np.float16)


def build_scene(root: Path, feature_mode: str = "smooth"):
    from PIL import Image

    from lot.encoders import CACHE_VERSION, cache_dir, features_digest
    from lot.render_replica import (
        FrameRecord,
        Manifest,
        intrinsics_from_hfov,
        program_rotation,
        program_translation,
        write_frame_stats,
        write_manifest,
    )

    root = Path(root)
    scene_root = root / SCENE
    (scene_root / "rgb").mkdir(parents=True)
    (scene_root / "depth").mkdir(parents=True)

    K = intrinsics_from_hfov(SIDE, SIDE, 90.0)
    posed = program_rotation(base_pose(), [-10.0, -5.0, 0.0, 5.0, 10.0], [])
    posed += program_translation(base_pose(), [0.05, 0.1, 0.2], 3.0)

    yy, xx = np.mgrid[0:SIDE, 0:SIDE]
    depth = np.where(xx < SIDE // 2, 2.0, 4.0).astype(np.float32) + 0.002 * yy

    frames, features = [], {}
    counters: dict[str, int] = {}
    for i, frame in enumerate(posed):
        index = counters.get(frame.regime, 0)
        counters[frame.regime] = index + 1
        fid = f"{SCENE}_vp00_{frame.regime}_{index:03d}"
        Image.fromarray(np.zeros((SIDE, SIDE, 3), np.uint8)).save(scene_root / f"rgb/{fid}.png")
        np.save(scene_root / f"depth/{fid}.npy", depth.astype(np.float32))
        frames.append(FrameRecord(
            frame_id=fid, scene=SCENE, regime=frame.regime,
            params=dict(frame.params, viewpoint=0),
            T_world_from_camera=frame.T_world_from_camera, K=K,
            height=SIDE, width=SIDE,
            rgb_path=f"rgb/{fid}.png", depth_path=f"depth/{fid}.npy",
        ))
        if feature_mode == "surface":
            features[fid] = surface_attached_features(frames[-1], depth, CHANNELS)
        else:
            features[fid] = smooth_features(SIDE // 14, SIDE // 14, CHANNELS, seed=100 + i)

    manifest = Manifest(
        scene=SCENE,
        metadata={"depth_convention": {"raw_verdict": "planar_z", "stored_depth": "planar_z"}},
        frames=frames,
    )
    write_manifest(scene_root / "manifest.json", manifest)
    write_frame_stats(scene_root, manifest)
    d = cache_dir(root / "cache", "dinov2_vitb14", SCENE)
    d.mkdir(parents=True)
    np.savez(d / "features.npz", **features)
    # Evaluation validates a cache before opening it, so the probe cache has to
    # carry the provenance a real one does.
    (d / "meta.json").write_text(json.dumps({
        "cache_version": CACHE_VERSION,
        "encoder": "dinov2_vitb14",
        "scene": SCENE,
        "channels": CHANNELS,
        "patch_size": 14,
        "patch_grid": [SIDE // 14, SIDE // 14],
        "image_hw": [SIDE, SIDE],
        "dtype": "float16",
        "frame_count": len(features),
        "frame_ids": [f.frame_id for f in frames],
        "has_depth": False,
        "weights_fingerprint": "validation-probe",
        "weights_revision": "2" * 40,
        "code_revision": "3" * 40,
        "features_digest": features_digest(features),
        "depth_digest": None,
    }, indent=1), encoding="utf-8")
    return manifest


def evaluate_probe(root: Path):
    """Run lot.evaluate over the probe scene. Returns the rows."""
    from lot.evaluate import EvalConfig, dataset_mean_vector, evaluate_scene

    cfg = EvalConfig(
        experiment_name="validation_probe", renders_root=root, cache_root=Path(root) / "cache",
        output_root=Path(root) / "out", scenes=[SCENE], encoders=["dinov2_vitb14"],
        seed=0, mean_vector_scenes=[SCENE],
    )
    mean_vector = dataset_mean_vector(cfg.cache_root, "dinov2_vitb14", [SCENE])
    from lot.analysis_config import load_analysis_config

    rows, _ = evaluate_scene(
        cfg, SCENE, {"dinov2_vitb14": mean_vector}, load_analysis_config()
    )
    return rows
