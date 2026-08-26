"""PLAN Phase 3: context-target pairs, regime tags, parallax bins, scene splits."""

import math

import pytest
import torch

from lot.datasets import (
    PARALLAX_BIN_EDGES,
    ZERO_PARALLAX_BIN,
    build_scene_pairs,
    pair_quantities,
    parallax_bin,
    parallax_bin_order,
    scene_split,
    stratum_of,
    subsample_by_stratum,
    summarize_pairs,
)
from lot.render_replica import (
    FrameRecord,
    Manifest,
    intrinsics_from_hfov,
    program_rotation,
    program_translation,
)
from test_render_replica import base_pose


def make_scene(scene="room_0", median_m=2.0, viewpoints=1):
    """A manifest and matching frame-stats sidecar, without touching the disk."""
    K = intrinsics_from_hfov(28, 28, 90.0)
    frames = []
    stats = {}
    for viewpoint in range(viewpoints):
        posed = program_rotation(base_pose(), [-7.5, 0.0, 7.5], [])
        posed += program_translation(base_pose(), [0.1, 0.2], median_m)
        counters: dict[str, int] = {}
        for frame in posed:
            index = counters.get(frame.regime, 0)
            counters[frame.regime] = index + 1
            frame_id = f"{scene}_vp{viewpoint:02d}_{frame.regime}_{index:03d}"
            frames.append(
                FrameRecord(
                    frame_id=frame_id,
                    scene=scene,
                    regime=frame.regime,
                    params=dict(frame.params, viewpoint=viewpoint),
                    T_world_from_camera=frame.T_world_from_camera,
                    K=K,
                    height=28,
                    width=28,
                    rgb_path=f"rgb/{frame_id}.png",
                    depth_path=f"depth/{frame_id}.npy",
                )
            )
            stats[frame_id] = {
                "valid_fraction": 1.0,
                "median_m": median_m,
                "center_p01_m": median_m,
                "min_m": median_m,
                "max_m": median_m,
            }
    manifest = Manifest(scene=scene, metadata={}, frames=frames)
    payload = {
        "frame_stats_version": 2,
        "scene": scene,
        "total": len(frames),
        "regimes": {f.frame_id: f.regime for f in frames},
        "median_depth_quantiles_m": {"p05": median_m, "p50": median_m, "p95": median_m},
        "frames": stats,
    }
    return manifest, payload


# ---------------------------------------------------------------------------
# Bins and splits
# ---------------------------------------------------------------------------

def test_pure_rotation_gets_its_own_bin():
    """Zero baseline is a different situation from a small one, not the smallest bin."""
    assert parallax_bin(0.0) == ZERO_PARALLAX_BIN
    assert parallax_bin(1e-12) == ZERO_PARALLAX_BIN
    assert parallax_bin(0.01) != ZERO_PARALLAX_BIN


def test_bins_are_half_open_on_the_left_and_cover_everything():
    order = parallax_bin_order()
    assert order[0] == ZERO_PARALLAX_BIN
    assert len(order) == len(PARALLAX_BIN_EDGES) + 1
    # Every edge belongs to the bin it closes, and just past it moves on.
    lower = 0.0
    for position, edge in enumerate(PARALLAX_BIN_EDGES[:-1], start=1):
        assert parallax_bin(edge) == order[position]
        assert parallax_bin(edge + 1e-9) == order[position + 1]
        lower = edge
    assert parallax_bin(lower * 10 + 1.0) == order[-1]


def test_bin_labels_are_unique_and_ordered():
    order = parallax_bin_order()
    assert len(set(order)) == len(order)


def test_parallax_bin_rejects_nonsense():
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            parallax_bin(bad)


def test_scene_split_covers_the_canonical_scenes():
    assert scene_split("room_0") == "train"
    assert scene_split("hotel_0") == "test"
    with pytest.raises(ValueError):
        scene_split("not_a_scene")


# ---------------------------------------------------------------------------
# Pair quantities
# ---------------------------------------------------------------------------

def test_in_place_rotation_pairs_have_no_baseline():
    manifest, stats = make_scene()
    pairs = build_scene_pairs(manifest, stats)
    rotation = [p for p in pairs if p.regime == "rotation"]
    assert rotation
    for pair in rotation:
        assert pair.baseline_m < 1e-12
        assert pair.parallax_bin == ZERO_PARALLAX_BIN
        assert pair.rotation_deg > 0


def test_translation_pairs_have_no_rotation_and_the_expected_parallax():
    """A pair of translations is baseline over the context frame's median depth."""
    median = 2.0
    manifest, stats = make_scene(median_m=median)
    pairs = build_scene_pairs(manifest, stats)
    translation = [p for p in pairs if p.regime == "translation"]
    assert translation
    for pair in translation:
        assert pair.rotation_deg < 1e-9
        assert abs(pair.parallax - pair.baseline_m / median) < 1e-12
    # The program moves plus and minus 0.1 and 0.2 of the median depth along two
    # axes, so the largest pair separation is the two opposite 0.2 moves.
    assert abs(max(p.parallax for p in translation) - 0.4) < 1e-9


def test_pair_quantities_match_the_relative_pose_directly():
    manifest, _ = make_scene()
    frames = {f.frame_id: f for f in manifest.frames}
    context, target = manifest.frames[0], manifest.frames[3]
    baseline, value, rotation = pair_quantities(context, target, 2.0)
    from lot.geometry import relative_pose

    T = relative_pose(target.T_world_from_camera, context.T_world_from_camera)
    assert abs(baseline - float(torch.linalg.vector_norm(T[:3, 3]))) < 1e-12
    assert abs(value - baseline / 2.0) < 1e-12
    assert 0.0 <= rotation <= 180.0
    assert frames  # the manifest is keyed as expected


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def test_pairs_are_directed_and_never_self_paired():
    manifest, stats = make_scene()
    pairs = build_scene_pairs(manifest, stats)
    keys = {(p.context_frame_id, p.target_frame_id) for p in pairs}
    assert len(keys) == len(pairs)
    assert not any(a == b for a, b in keys)
    # Direction matters for transport, so both orders are present.
    assert all((b, a) in keys for a, b in keys)


def test_pairs_never_cross_viewpoints_or_regimes():
    manifest, stats = make_scene(viewpoints=3)
    pairs = build_scene_pairs(manifest, stats)
    frames = {f.frame_id: f for f in manifest.frames}
    assert pairs
    for pair in pairs:
        context = frames[pair.context_frame_id]
        target = frames[pair.target_frame_id]
        assert context.params["viewpoint"] == target.params["viewpoint"] == pair.viewpoint
        assert context.regime == target.regime == pair.regime


def test_pair_count_is_every_ordered_pair_within_a_group():
    manifest, stats = make_scene(viewpoints=2)
    pairs = build_scene_pairs(manifest, stats)
    # 3 rotation frames and 9 translation frames per viewpoint, two viewpoints.
    expected = 2 * (3 * 2 + 9 * 8)
    assert len(pairs) == expected


def test_unusable_frames_are_excluded():
    manifest, stats = make_scene()
    dropped = manifest.frames[0].frame_id
    stats["frames"][dropped]["center_p01_m"] = 0.01
    pairs = build_scene_pairs(manifest, stats)
    assert all(dropped not in (p.context_frame_id, p.target_frame_id) for p in pairs)


def test_usability_policy_flows_through():
    manifest, stats = make_scene(median_m=2.0)
    assert build_scene_pairs(manifest, stats)
    assert build_scene_pairs(manifest, stats, min_clearance_m=10.0) == []


def test_regime_selection_flows_through():
    manifest, stats = make_scene()
    pairs = build_scene_pairs(manifest, stats, regimes=("rotation",))
    assert pairs and {p.regime for p in pairs} == {"rotation"}


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------

def test_subsample_caps_each_stratum_and_is_reproducible():
    manifest, stats = make_scene(viewpoints=2)
    pairs = build_scene_pairs(manifest, stats)
    taken = subsample_by_stratum(pairs, max_per_stratum=3, seed=0)
    counts: dict[tuple[str, str, str], int] = {}
    for pair in taken:
        counts[stratum_of(pair)] = counts.get(stratum_of(pair), 0) + 1
    assert counts and max(counts.values()) <= 3
    again = subsample_by_stratum(pairs, max_per_stratum=3, seed=0)
    assert [p.context_frame_id for p in taken] == [p.context_frame_id for p in again]
    other = subsample_by_stratum(pairs, max_per_stratum=3, seed=1)
    assert [p.context_frame_id for p in taken] != [p.context_frame_id for p in other]


def test_subsample_keeps_input_order_and_small_strata_whole():
    manifest, stats = make_scene()
    pairs = build_scene_pairs(manifest, stats)
    taken = subsample_by_stratum(pairs, max_per_stratum=10_000, seed=0)
    assert taken == pairs


def test_one_scene_cannot_change_another_scenes_sample():
    """Strata are seeded by their own identity, so scenes are independent."""
    room, room_stats = make_scene(scene="room_0")
    office, office_stats = make_scene(scene="office_0")
    room_pairs = build_scene_pairs(room, room_stats)
    office_pairs = build_scene_pairs(office, office_stats)
    alone = subsample_by_stratum(room_pairs, max_per_stratum=2, seed=0)
    together = [
        p
        for p in subsample_by_stratum(room_pairs + office_pairs, max_per_stratum=2, seed=0)
        if p.scene == "room_0"
    ]
    assert alone == together


def test_summary_counts_add_up():
    manifest, stats = make_scene(viewpoints=2)
    pairs = build_scene_pairs(manifest, stats)
    summary = summarize_pairs(pairs)
    assert summary["total"] == len(pairs)
    assert sum(summary["by_regime"].values()) == len(pairs)
    assert sum(summary["by_parallax_bin"].values()) == len(pairs)
    assert summary["by_split"]["train"] == len(pairs)


# ---------------------------------------------------------------------------
# Both axes of viewpoint change
# ---------------------------------------------------------------------------

def test_rotation_bins_mirror_the_parallax_bins():
    from lot.datasets import ROTATION_BIN_EDGES, rotation_bin, rotation_bin_order

    order = rotation_bin_order()
    assert order[0] == ZERO_PARALLAX_BIN
    assert len(order) == len(ROTATION_BIN_EDGES) + 1
    assert rotation_bin(0.0) == ZERO_PARALLAX_BIN
    for position, edge in enumerate(ROTATION_BIN_EDGES[:-1], start=1):
        assert rotation_bin(edge) == order[position]
        assert rotation_bin(edge + 1e-3) == order[position + 1]
    with pytest.raises(ValueError):
        rotation_bin(-1.0)


def test_each_regime_varies_on_the_axis_it_actually_moves():
    """Neither axis alone can stratify: each collapses one regime to a single cell."""
    manifest, stats = make_scene()
    pairs = build_scene_pairs(manifest, stats)
    rotation = [p for p in pairs if p.regime == "rotation"]
    translation = [p for p in pairs if p.regime == "translation"]

    # In-place rotation has no baseline, so parallax cannot separate its pairs.
    assert {p.parallax_bin for p in rotation} == {ZERO_PARALLAX_BIN}
    assert len({p.rotation_bin for p in rotation}) > 1
    # Pure translation has no rotation, so the angle cannot separate its pairs.
    assert {p.rotation_bin for p in translation} == {ZERO_PARALLAX_BIN}
    assert len({p.parallax_bin for p in translation}) > 1


def test_strata_split_rotation_by_angle():
    """The first Experiment Zero run pooled 7.5 to 60 degrees into one cell."""
    manifest, stats = make_scene()
    rotation = [p for p in build_scene_pairs(manifest, stats) if p.regime == "rotation"]
    strata = {stratum_of(p) for p in rotation}
    assert len(strata) > 1
    for stratum in strata:
        assert len(stratum) == 4  # scene, regime, parallax bin, rotation bin


def test_subsampling_now_keeps_every_rotation_angle():
    """A cap per stratum only balances angles once the angles are separate strata."""
    manifest, stats = make_scene()
    rotation = [p for p in build_scene_pairs(manifest, stats) if p.regime == "rotation"]
    taken = subsample_by_stratum(rotation, max_per_stratum=1, seed=0)
    assert {p.rotation_bin for p in taken} == {p.rotation_bin for p in rotation}


def test_summary_reports_both_axes():
    manifest, stats = make_scene()
    pairs = build_scene_pairs(manifest, stats)
    summary = summarize_pairs(pairs)
    assert sum(summary["by_parallax_bin"].values()) == len(pairs)
    assert sum(summary["by_rotation_bin"].values()) == len(pairs)
