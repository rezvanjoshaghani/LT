"""A genuine one-path pair through the whole pipeline, not a forced mask.

The unit test that pins the five-versus-ten row arithmetic forces a one-path
geometry by blanking a mask. That proves the arithmetic and nothing else: until
this file, no one-path pair produced by the sampler's own rules had ever
traversed evaluation, storage, loading, pairing, counting, and summary, so
Stream D's real data would have been the first live exercise of the path.

The construction: a depth map whose stripe boundaries sit at columns 7 + 14k.
A patch center at u = 6.5 + 14k reads pixels 6 + 14k and 7 + 14k, one on each
side of a boundary, so the four-corner consistency test rejects every
per-point candidate, while the splat path keeps full support because the
per-patch co-visible fraction stays high. Every pair of the striped viewpoint
is then splat-only by construction of the scene, not by construction of the
test.

Translation, not rotation, for the camera program: a planar z-depth map is only
a consistent surface between two cameras when they share orientation, so under
rotation the same flat map in both frames describes two different planes and
co-visibility collapses for reasons that have nothing to do with this test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from lot.analysis_config import load_analysis_config
from lot.encoders import CACHE_VERSION, cache_dir, features_digest
from lot.evaluate import (
    EvalConfig,
    PER_POINT,
    SPLAT_POOL,
    dataset_mean_vector,
    evaluate_scene,
    write_rows,
)
from lot.figures import (
    assign_bins,
    counts_table,
    paired_records,
    path_agreement,
    read_eval_dir,
    summary_table,
)
from lot.render_replica import (
    FrameRecord,
    Manifest,
    intrinsics_from_hfov,
    program_translation,
    write_frame_stats,
    write_manifest,
)
from test_render_replica import base_pose

ANALYSIS = load_analysis_config()
SIDE = 56          # 4x4 patches: the smallest grid the sampler accepts
CHANNELS = 768     # dinov2_vitb14's registered width; the cache check enforces it
SCENE = "room_0"


def striped_depth() -> np.ndarray:
    depth = np.empty((SIDE, SIDE), dtype=np.float32)
    for col in range(SIDE):
        depth[:, col] = 3.0 if ((col + 7) // 14) % 2 == 0 else 5.0
    return depth


def build_scene(root):
    scene_root = root / SCENE
    (scene_root / "rgb").mkdir(parents=True)
    (scene_root / "depth").mkdir(parents=True)
    K = intrinsics_from_hfov(SIDE, SIDE, 90.0)
    flat = np.full((SIDE, SIDE), 3.0, dtype=np.float32)
    generator = torch.Generator().manual_seed(0)

    frames, features = [], {}
    for viewpoint, depth in ((0, flat), (1, striped_depth())):
        # Baselines sized so the reported parallax clears the design floor on
        # either depth: 0.15 / 5.0 = 0.03 >= 0.025.
        posed = program_translation(base_pose(), [0.05, 0.1], 3.0)
        for index, frame in enumerate(posed):
            fid = f"{SCENE}_vp{viewpoint:02d}_translation_{index:03d}"
            Image.fromarray(np.zeros((SIDE, SIDE, 3), np.uint8)).save(
                scene_root / f"rgb/{fid}.png"
            )
            np.save(scene_root / f"depth/{fid}.npy", depth)
            frames.append(
                FrameRecord(
                    frame_id=fid, scene=SCENE, regime="translation",
                    params=dict(frame.params, viewpoint=viewpoint),
                    T_world_from_camera=frame.T_world_from_camera, K=K,
                    height=SIDE, width=SIDE,
                    rgb_path=f"rgb/{fid}.png", depth_path=f"depth/{fid}.npy",
                )
            )
            grid = SIDE // 14
            coarse = torch.rand((CHANNELS, 3, 3), generator=generator)
            features[fid] = (
                torch.nn.functional.interpolate(
                    coarse[None], size=(grid, grid), mode="bilinear", align_corners=True
                )[0].to(torch.float16).numpy()
            )

    manifest = Manifest(
        scene=SCENE,
        metadata={"depth_convention": {"raw_verdict": "planar_z", "stored_depth": "planar_z"}},
        frames=frames,
    )
    write_manifest(scene_root / "manifest.json", manifest)
    write_frame_stats(scene_root, manifest)
    directory = cache_dir(root / "cache", "dinov2_vitb14", SCENE)
    directory.mkdir(parents=True)
    np.savez(directory / "features.npz", **features)
    import json

    (directory / "meta.json").write_text(
        json.dumps({
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
            "weights_fingerprint": "one-path-test",
            "weights_revision": "0" * 40,
            "code_revision": "1" * 40,
            "features_digest": features_digest(features),
            "depth_digest": None,
        }, indent=1),
        encoding="utf-8",
    )


def test_a_one_path_pair_traverses_the_full_pipeline(tmp_path):
    build_scene(tmp_path)
    cfg = EvalConfig(
        experiment_name="one_path", renders_root=tmp_path, cache_root=tmp_path / "cache",
        output_root=tmp_path / "out", scenes=[SCENE], encoders=["dinov2_vitb14"],
        seed=0, mean_vector_scenes=[SCENE],
    )
    mean = dataset_mean_vector(cfg.cache_root, "dinov2_vitb14", [SCENE])
    rows, metadata = evaluate_scene(cfg, SCENE, {"dinov2_vitb14": mean}, ANALYSIS)

    # The metadata split and the row arithmetic, live rather than forced.
    both = metadata["pairs_scored_both_paths"]
    one = metadata["pairs_scored_one_path"]
    assert both >= 1 and one >= 1, "the scene must produce both population terms"
    assert len(rows) == both * 10 + one * 5

    # Every one-path pair belongs to the striped viewpoint, on the splat path:
    # the sampler rejected the candidates, nothing forced a mask.
    per_pair_paths: dict[tuple, set] = {}
    for row in rows:
        key = (row["context_frame_id"], row["target_frame_id"])
        per_pair_paths.setdefault(key, set()).add(row["path"])
    for (context, _target), paths in per_pair_paths.items():
        if "_vp01_" in context:
            assert paths == {SPLAT_POOL}
        else:
            assert paths == {PER_POINT, SPLAT_POOL}

    # Storage, loading, and the population checks.
    eval_dir = tmp_path / "out" / "one_path" / "eval"
    metadata = {**metadata, "run_scenes": [SCENE]}
    write_rows(eval_dir / f"{SCENE}.parquet", rows, metadata)
    reread = read_eval_dir(eval_dir, ANALYSIS)
    assert len(reread) == len(rows)

    # Pairing: a one-path comparison carries its five variants and no phantom
    # counterpart, so the completeness gate passes without special-casing.
    records, mismatches = paired_records(assign_bins(reread, ANALYSIS))
    assert mismatches == 0

    # The counts view shows the divergence this scene creates: more pairs on
    # the splat path than the per-point path in the striped bins.
    table = counts_table(records, ANALYSIS)
    by_path = {}
    for entry in table:
        by_path.setdefault(entry["path"], 0)
        by_path[entry["path"]] += entry["n_camera_pairs"]
    assert by_path[SPLAT_POOL] > by_path[PER_POINT]

    # Path agreement counts the one-path pairs instead of losing them.
    agreement = path_agreement(assign_bins(reread, ANALYSIS), ANALYSIS)
    assert agreement["single_path_pairs"] == one
    assert agreement["comparisons"] == both

    # And the summary builds.
    assert summary_table(records, ANALYSIS)
