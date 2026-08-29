"""PROTOCOL 3.3 and 3.4: pairs, regimes, splits, and the sampling strata."""

import math

import pytest
import torch

from lot.analysis_config import load_analysis_config
from lot.datasets import (
    ROW_FIELDS,
    PairRecord,
    ZERO_BIN,
    assert_translation_parallax_floor,
    bin_label,
    bin_order,
    build_scene_pairs,
    pair_quantities,
    parallax_bin,
    parallax_bin_order,
    rotation_bin,
    rotation_bin_order,
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

ANALYSIS = load_analysis_config()


def make_scene(scene="room_0", median_m=2.0, viewpoints=1, parallaxes=(0.1, 0.2)):
    """A manifest and matching frame-stats sidecar, without touching the disk."""
    K = intrinsics_from_hfov(28, 28, 90.0)
    frames = []
    stats = {}
    for viewpoint in range(viewpoints):
        posed = program_rotation(base_pose(), [-7.5, 0.0, 7.5], [])
        posed += program_translation(base_pose(), list(parallaxes), median_m)
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
# PROTOCOL 3.4: binning comes from the committed config
# ---------------------------------------------------------------------------

def test_rotation_edges_are_the_frozen_ten_degree_ladder():
    """PROTOCOL 3.4: equal-width 10-degree bins from 0 to 50 with a 50-plus overflow."""
    assert ANALYSIS.rotation_bin_edges_deg == (10.0, 20.0, 30.0, 40.0, 50.0)
    assert math.isinf(ANALYSIS.rotation_edges()[-1])
    widths = [
        b - a
        for a, b in zip((0.0,) + ANALYSIS.rotation_bin_edges_deg, ANALYSIS.rotation_bin_edges_deg)
    ]
    assert set(widths) == {10.0}


def test_parallax_edges_are_the_adopted_ones():
    assert ANALYSIS.parallax_bin_edges == (0.025, 0.05, 0.1, 0.2, 0.4)


def test_bins_are_closed_on_the_right():
    """A value equal to an edge belongs to the lower bin. MINOR-5, now frozen."""
    assert ANALYSIS.bin_right_closed
    order = bin_order(ANALYSIS.parallax_edges())
    for position, edge in enumerate(ANALYSIS.parallax_bin_edges, start=1):
        assert parallax_bin(edge, ANALYSIS) == order[position]
        assert parallax_bin(edge + 1e-9, ANALYSIS) == order[position + 1]


def test_exact_zero_gets_its_own_bin_on_both_axes():
    assert parallax_bin(0.0, ANALYSIS) == ZERO_BIN
    assert rotation_bin(0.0, ANALYSIS) == ZERO_BIN
    assert parallax_bin(0.01, ANALYSIS) != ZERO_BIN
    assert rotation_bin(1.0, ANALYSIS) != ZERO_BIN


def test_changing_the_config_changes_the_bins_with_no_source_edit():
    """The point of BLOCKER-1: edges live in the config, not in source."""
    import dataclasses

    widened = dataclasses.replace(ANALYSIS, rotation_bin_edges_deg=(25.0, 50.0))
    assert rotation_bin(20.0, ANALYSIS) == "10-20"
    assert rotation_bin(20.0, widened) == "0-25"
    assert rotation_bin_order(widened) == ["zero", "0-25", "25-50", "50+"]


def test_bin_label_rejects_nonsense():
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            parallax_bin(bad, ANALYSIS)


def test_scene_split_covers_the_canonical_scenes():
    assert scene_split("room_0") == "train"
    assert scene_split("hotel_0") == "test"
    with pytest.raises(ValueError):
        scene_split("not_a_scene")


# ---------------------------------------------------------------------------
# PROTOCOL 3.2: rows carry continuous geometry and no labels
# ---------------------------------------------------------------------------

def test_rows_carry_no_bin_label_and_no_proxy():
    """MAJOR-3, and the proxy must never be mistaken for the reported statistic."""
    manifest, stats = make_scene()
    row = build_scene_pairs(manifest, stats, ANALYSIS)[0].as_row()
    assert set(row) == set(ROW_FIELDS)
    assert not any("bin" in key for key in row)
    assert "parallax" not in row
    assert "stratum_parallax" not in row
    assert isinstance(row["rotation_deg"], float)


def test_rotation_deg_is_continuous_and_in_range():
    manifest, stats = make_scene()
    for pair in build_scene_pairs(manifest, stats, ANALYSIS):
        assert 0.0 <= pair.rotation_deg <= 180.0


# ---------------------------------------------------------------------------
# PROTOCOL 3.3: regimes are separate controls
# ---------------------------------------------------------------------------

def test_in_place_rotation_pairs_have_no_baseline():
    manifest, stats = make_scene()
    rotation = [p for p in build_scene_pairs(manifest, stats, ANALYSIS) if p.regime == "rotation"]
    assert rotation
    for pair in rotation:
        assert pair.baseline_m < 1e-12
        assert pair.rotation_deg > 0


def test_translation_pairs_have_no_rotation():
    manifest, stats = make_scene()
    translation = [
        p for p in build_scene_pairs(manifest, stats, ANALYSIS) if p.regime == "translation"
    ]
    assert translation
    for pair in translation:
        assert pair.rotation_deg < 1e-9


def test_each_regime_varies_on_the_axis_it_moves():
    """Neither axis alone can stratify: each collapses one regime to a single cell."""
    manifest, stats = make_scene()
    pairs = build_scene_pairs(manifest, stats, ANALYSIS)
    rotation = [p for p in pairs if p.regime == "rotation"]
    translation = [p for p in pairs if p.regime == "translation"]
    assert {parallax_bin(p.stratum_parallax, ANALYSIS) for p in rotation} == {ZERO_BIN}
    assert len({rotation_bin(p.rotation_deg, ANALYSIS) for p in rotation}) >= 1
    assert {rotation_bin(p.rotation_deg, ANALYSIS) for p in translation} == {ZERO_BIN}
    assert len({parallax_bin(p.stratum_parallax, ANALYSIS) for p in translation}) > 1


def test_pairs_never_cross_viewpoints_or_regimes():
    manifest, stats = make_scene(viewpoints=3)
    pairs = build_scene_pairs(manifest, stats, ANALYSIS)
    frames = {f.frame_id: f for f in manifest.frames}
    assert pairs
    for pair in pairs:
        context = frames[pair.context_frame_id]
        target = frames[pair.target_frame_id]
        assert context.params["viewpoint"] == target.params["viewpoint"] == pair.viewpoint
        assert context.regime == target.regime == pair.regime


def test_pairs_are_directed_and_never_self_paired():
    manifest, stats = make_scene()
    keys = {
        (p.context_frame_id, p.target_frame_id)
        for p in build_scene_pairs(manifest, stats, ANALYSIS)
    }
    assert not any(a == b for a, b in keys)
    assert all((b, a) in keys for a, b in keys)


def test_pair_quantities_match_the_relative_pose_directly():
    manifest, _ = make_scene()
    context, target = manifest.frames[0], manifest.frames[3]
    baseline, proxy, rotation = pair_quantities(context, target, 2.0)
    from lot.geometry import relative_pose

    T = relative_pose(target.T_world_from_camera, context.T_world_from_camera)
    assert abs(baseline - float(torch.linalg.vector_norm(T[:3, 3]))) < 1e-12
    assert abs(proxy - baseline / 2.0) < 1e-12
    assert 0.0 <= rotation <= 180.0


# ---------------------------------------------------------------------------
# PROTOCOL 3.4: the translation floor assertion
# ---------------------------------------------------------------------------

def test_translation_floor_is_enforced_on_the_reported_statistic():
    """MINOR-3, and the quantity matters as much as the check.

    The assertion is about the parallax that is reported and binned, the median
    over the co-visible set. Checking the whole-frame sampling proxy instead
    would let a pair pass here and still be reported inside the interval this
    forbids, since the two are different numbers over different populations.
    """
    floor = ANALYSIS.translation_parallax_design_floor
    # A translation pair inside the forbidden interval is refused.
    with pytest.raises(ValueError, match="asserts empty"):
        assert_translation_parallax_floor("translation", floor / 2, ANALYSIS, "a pair")
    # Exact zero, and anything at or above the floor, are fine.
    assert_translation_parallax_floor("translation", 0.0, ANALYSIS, "a pair")
    assert_translation_parallax_floor("translation", floor, ANALYSIS, "a pair")
    # A pair with no co-visible surface has no reported parallax to check.
    assert_translation_parallax_floor("translation", float("nan"), ANALYSIS, "a pair")


def test_orbit_pairs_are_not_subject_to_the_floor():
    """PROTOCOL 3.4: orbit pairs may legitimately fall in that interval."""
    first_edge = ANALYSIS.parallax_edges()[0]
    assert_translation_parallax_floor("orbit", first_edge / 2, ANALYSIS, "an orbit pair")
    assert_translation_parallax_floor("rotation", first_edge / 2, ANALYSIS, "a rotation pair")


def dataclasses_replace(record, **changes):
    import dataclasses

    return dataclasses.replace(record, **changes)


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------

def test_strata_use_both_axes_and_the_config_edges():
    manifest, stats = make_scene()
    pairs = build_scene_pairs(manifest, stats, ANALYSIS)
    for pair in pairs:
        stratum = stratum_of(pair, ANALYSIS)
        assert len(stratum) == 4
        assert stratum[0] == "room_0"


def test_subsample_caps_each_stratum_and_is_reproducible():
    manifest, stats = make_scene(viewpoints=2)
    pairs = build_scene_pairs(manifest, stats, ANALYSIS)
    taken = subsample_by_stratum(pairs, 3, seed=0, config=ANALYSIS)
    counts: dict[tuple, int] = {}
    for pair in taken:
        key = stratum_of(pair, ANALYSIS)
        counts[key] = counts.get(key, 0) + 1
    assert counts and max(counts.values()) <= 3
    again = subsample_by_stratum(pairs, 3, seed=0, config=ANALYSIS)
    assert [p.context_frame_id for p in taken] == [p.context_frame_id for p in again]


def test_one_scene_cannot_change_another_scenes_sample():
    room, room_stats = make_scene(scene="room_0")
    office, office_stats = make_scene(scene="office_0")
    room_pairs = build_scene_pairs(room, room_stats, ANALYSIS)
    office_pairs = build_scene_pairs(office, office_stats, ANALYSIS)
    alone = subsample_by_stratum(room_pairs, 2, seed=0, config=ANALYSIS)
    together = [
        p
        for p in subsample_by_stratum(room_pairs + office_pairs, 2, seed=0, config=ANALYSIS)
        if p.scene == "room_0"
    ]
    assert alone == together


def test_summary_counts_add_up():
    manifest, stats = make_scene(viewpoints=2)
    pairs = build_scene_pairs(manifest, stats, ANALYSIS)
    summary = summarize_pairs(pairs, ANALYSIS)
    assert summary["total"] == len(pairs)
    assert sum(summary["by_regime"].values()) == len(pairs)
    assert sum(summary["by_stratum_parallax_bin"].values()) == len(pairs)
    assert sum(summary["by_rotation_bin"].values()) == len(pairs)


def test_every_config_value_is_either_read_or_declared_reserved():
    """A normative constant nothing reads describes the code rather than governing it.

    Two keys had gone dead unnoticed: epsilon_margin was read nowhere, and
    bin_right_closed was named in a docstring while the comparison beside it was
    a literal. Both changed the config digest, so editing either invalidated an
    existing run and altered no behaviour. This walks the source rather than
    trusting review to notice the next one.
    """
    import dataclasses
    from pathlib import Path

    from lot.analysis_config import AnalysisConfig

    source_root = Path(__file__).resolve().parents[1] / "src" / "lot"
    body = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(source_root.glob("*.py"))
    )
    # analysis_config.py counts: a field read through an accessor it defines,
    # such as parallax_edges(), is read. as_dict and the digests use getattr, so
    # they cannot make an unread field look read.
    body += (source_root / "analysis_config.py").read_text(encoding="utf-8").split(
        "RESERVED_FOR_LATER_PHASES"
    )[0]
    _ = (
    )
    reserved = set(AnalysisConfig.RESERVED_FOR_LATER_PHASES)
    unread = [
        field.name
        for field in dataclasses.fields(AnalysisConfig)
        if field.name not in reserved and f".{field.name}" not in body
    ]
    assert not unread, (
        f"config values nothing outside analysis_config.py reads: {unread}. "
        "Either wire them up or list them in RESERVED_FOR_LATER_PHASES with a reason."
    )
    # And the reserved list may not name something that is in fact read, which
    # would let a live constant hide behind the exemption.
    live = [name for name in reserved if f".{name}" in body]
    assert not live, f"reserved but actually read: {live}"


def test_the_sampling_design_is_not_moved_by_a_reporting_edit():
    """PROTOCOL 3.4 lets the reporting edges be widened once, from counts.

    That happens after the pairs are drawn, so it cannot be allowed to redefine
    which pairs a later scene would contribute. The strata read their own frozen
    edges, and only those are part of the measurement identity.
    """
    import dataclasses

    pairs = [
        PairRecord(
            scene="room_0", split="train", viewpoint=0, regime="orbit",
            context_frame_id=f"c{i}", target_frame_id=f"t{i}",
            baseline_m=0.2, context_median_depth_m=2.0,
            rotation_deg=angle, stratum_parallax=0.08,
        )
        for i, angle in enumerate((5.0, 15.0, 25.0, 35.0))
    ]
    widened = dataclasses.replace(ANALYSIS, rotation_bin_edges_deg=(50.0,))
    assert [stratum_of(p, ANALYSIS) for p in pairs] == [stratum_of(p, widened) for p in pairs]
    assert widened.measurement_digest() == ANALYSIS.measurement_digest()

    # Editing the sampling design does move the strata, and is gated.
    resampled = dataclasses.replace(ANALYSIS, stratum_rotation_edges_deg=(50.0,))
    assert [stratum_of(p, ANALYSIS) for p in pairs] != [stratum_of(p, resampled) for p in pairs]
    assert resampled.measurement_digest() != ANALYSIS.measurement_digest()


def test_bin_right_closed_decides_which_side_an_edge_belongs_to():
    """The flag was named in a docstring while the comparison was a literal."""
    import dataclasses

    right_open = dataclasses.replace(ANALYSIS, bin_right_closed=False)
    assert rotation_bin(10.0, ANALYSIS) == "0-10"
    assert rotation_bin(10.0, right_open) == "10-20"
    assert parallax_bin(0.05, ANALYSIS) == "0.025-0.05"
    assert parallax_bin(0.05, right_open) == "0.05-0.1"
    # A value strictly inside a bin is unaffected either way.
    assert rotation_bin(15.0, ANALYSIS) == rotation_bin(15.0, right_open) == "10-20"
    # And the overflow bin still catches everything above the last edge.
    assert rotation_bin(500.0, ANALYSIS) == rotation_bin(500.0, right_open) == "50+"


def test_the_floor_assertion_does_not_move_with_a_reporting_edit():
    """The design floor is a measurement value, not the first reporting edge.

    The reporting edges may be widened once from counts after the run. An
    evaluation-time gate that read the first edge would then reject at 0.08
    under one identity what it accepted under an equal one: widening the first
    edge to 0.1 left the measurement digest unchanged while moving the gate.
    """
    import dataclasses

    widened = dataclasses.replace(ANALYSIS, parallax_bin_edges=(0.1, 0.2, 0.4))
    assert widened.measurement_digest() == ANALYSIS.measurement_digest()
    # 0.08 is above the design floor, so it passes under both configs even
    # though it is below the widened first reporting edge.
    assert_translation_parallax_floor("translation", 0.08, ANALYSIS, "a pair")
    assert_translation_parallax_floor("translation", 0.08, widened, "a pair")

    # Moving the floor itself is a measurement change and is gated as one.
    raised = dataclasses.replace(ANALYSIS, translation_parallax_design_floor=0.1)
    assert raised.measurement_digest() != ANALYSIS.measurement_digest()
    with pytest.raises(ValueError, match="asserts empty"):
        assert_translation_parallax_floor("translation", 0.08, raised, "a pair")


def test_bin_right_closed_is_a_measurement_value():
    """Flipping it moves which pairs a capped stratum draws.

    A 10-degree pair sits in the 0-10 stratum right-closed and the 10-20
    stratum right-open, and sampling is capped per stratum, so the flag decides
    the selected population. It was classified as reporting, so two samples
    drawn under opposite conventions compared as one measurement. PROTOCOL 3.4
    also froze the convention outright, so no post-run edit to it is permitted
    in the first place.
    """
    import dataclasses

    from lot.datasets import stratum_of

    flipped = dataclasses.replace(ANALYSIS, bin_right_closed=False)
    assert flipped.measurement_digest() != ANALYSIS.measurement_digest()
    pair = PairRecord(
        scene="room_0", split="train", viewpoint=0, regime="rotation",
        context_frame_id="c", target_frame_id="t",
        baseline_m=0.0, context_median_depth_m=2.0,
        rotation_deg=10.0, stratum_parallax=0.0,
    )
    assert stratum_of(pair, ANALYSIS) != stratum_of(pair, flipped)
