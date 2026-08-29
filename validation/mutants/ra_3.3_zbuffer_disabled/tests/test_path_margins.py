"""The margin-level paired-path table: pairing, ordering, and the wording rules."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lot.analysis_config import load_analysis_config
from lot.evaluate import (
    NEIGHBOR_PATCH,
    NO_WARP_COPY,
    ORACLE_TRANSPORT,
    PER_POINT,
    SPLAT_POOL,
)
from lot.path_margins import (
    build,
    classify,
    paired_bootstrap,
    paired_pair_records,
)
from test_figures import assign_bins, make_row, population, run_meta, write_scene

ANALYSIS = load_analysis_config()
BAND = ANALYSIS.path_agreement_tolerance


def pair_rows(scene, index, values, metric_offset=-0.05, **overrides):
    """One pair's ten rows, with per-path intersection scores set explicitly.

    values maps (path, variant) to the raw intersection cosine.
    """
    rows = []
    for (path, variant), score in values.items():
        rows.append(
            make_row(
                scene=scene,
                path=path,
                variant=variant,
                context_frame_id=f"c{index}",
                target_frame_id=f"t{index}",
                cosine_intersect_mean=score,
                cosine_centered_intersect_mean=score + metric_offset,
                **overrides,
            )
        )
    return rows


def full_pair(scene, index, oracle_pp, nowarp_pp, neighbor_pp,
              oracle_sp, nowarp_sp, neighbor_sp, **overrides):
    """All ten rows of a both-path pair.

    Mean-Feature and Random-Patch carry placeholder scores because
    read_eval_dir reconciles every comparison against the five variants
    PROTOCOL 3.7 requires; the quantities under test read only the other three.
    """
    from lot.evaluate import MEAN_FEATURE, RANDOM_PATCH

    values = {}
    for path, (o, n, nb) in (
        (PER_POINT, (oracle_pp, nowarp_pp, neighbor_pp)),
        (SPLAT_POOL, (oracle_sp, nowarp_sp, neighbor_sp)),
    ):
        values[(path, ORACLE_TRANSPORT)] = o
        values[(path, NO_WARP_COPY)] = n
        values[(path, NEIGHBOR_PATCH)] = nb
        values[(path, MEAN_FEATURE)] = n - 0.30
        values[(path, RANDOM_PATCH)] = n - 0.40
    return pair_rows(scene, index, values, **overrides)


def test_terms_come_from_the_intersection_before_differencing():
    """A quantity is a difference of two operators over one population.

    The evaluation layer scores every variant on the cells both paths scored
    and stores that as its own column. Reading those columns means the three
    terms already share a population when they are differenced, rather than
    being reconciled afterwards.
    """
    rows = assign_bins(
        full_pair("room_0", 0, 0.90, 0.50, 0.70, 0.88, 0.49, 0.71), ANALYSIS
    )
    records = paired_pair_records(rows)
    raw = [r for r in records if r["metric"] == "cosine_mean"][0]
    assert raw["oracle_margin_pp"] == pytest.approx(0.40)
    assert raw["oracle_margin_sp"] == pytest.approx(0.39)
    assert raw["localization_gap_pp"] == pytest.approx(0.20)
    assert raw["localization_gap_sp"] == pytest.approx(0.17)


def test_a_pair_scored_on_one_path_only_is_not_paired():
    """A cross-path difference is undefined without both paths."""
    values = {
        (PER_POINT, ORACLE_TRANSPORT): 0.9,
        (PER_POINT, NO_WARP_COPY): 0.5,
        (PER_POINT, NEIGHBOR_PATCH): 0.7,
    }
    rows = assign_bins(pair_rows("room_0", 0, values), ANALYSIS)
    assert paired_pair_records(rows) == []


def test_the_difference_is_bootstrapped_paired_not_as_two_intervals():
    """Both paths come from one draw, and dM is recomputed inside the replicate.

    Here the two paths differ by exactly 0.01 in every pair, so the paired
    difference has no variance at all and its interval must collapse, while
    each path's own interval is wide. Differencing two independently
    bootstrapped intervals would report a spread the quantity does not have.
    """
    by_scene = {
        f"scene_{s}": [
            {"oracle_margin_pp": 0.10 + 0.05 * s + 0.01 * p,
             "oracle_margin_sp": 0.09 + 0.05 * s + 0.01 * p}
            for p in range(6)
        ]
        for s in range(6)
    }
    stats = paired_bootstrap(by_scene, "oracle_margin", ANALYSIS)
    assert stats["dM"] == pytest.approx(0.01, abs=1e-12)
    assert stats["dM_ci_high"] - stats["dM_ci_low"] == pytest.approx(0.0, abs=1e-12)
    # Each path on its own does vary across the resampled scenes.
    assert stats["M_pp_ci_high"] - stats["M_pp_ci_low"] > 0.02


def test_support_is_established_before_and_fixed_through_the_bootstrap():
    """A replicate that happens to draw few scenes does not re-decide support."""
    rows = []
    for scene in range(4):
        for index in range(9):
            rows += full_pair(f"scene_{scene}", index, 0.9, 0.5, 0.7, 0.9, 0.5, 0.7)
    records = paired_pair_records(assign_bins(rows, ANALYSIS))
    scenes = {r["scene"] for r in records}
    pairs = {r["camera_pair"] for r in records}
    assert len(scenes) == 4 and len(pairs) == 9
    # is_supported reads these counts once; the bootstrap never recomputes them.
    assert ANALYSIS.support_min_scenes <= 4


def test_the_three_way_wording_for_an_oracle_margin():
    def stats(pp, sp, lo_pp=None, hi_pp=None, lo_sp=None, hi_sp=None):
        return {
            "M_pp": pp, "M_sp": sp, "dM": pp - sp,
            "M_pp_ci_low": lo_pp if lo_pp is not None else pp * 0.5,
            "M_pp_ci_high": hi_pp if hi_pp is not None else pp * 1.5,
            "M_sp_ci_low": lo_sp if lo_sp is not None else sp * 0.5,
            "M_sp_ci_high": hi_sp if hi_sp is not None else sp * 1.5,
        }

    both = classify("oracle_margin", stats(0.0020, 0.0018), BAND)
    assert both["case"] == "both_in_band"
    assert "small, sign-consistent positive" in both["sentence"]
    assert "close to the No-Warp floor" in both["sentence"]

    one = classify("oracle_margin", stats(0.0200, 0.0018), BAND)
    assert one["case"] == "one_in_band"
    assert "path-sensitive" in one["sentence"]
    assert "small" not in one["sentence"] and "floor" not in one["sentence"]

    flipped = classify("oracle_margin", stats(0.0020, -0.0018), BAND)
    assert flipped["case"] == "not_robust"
    assert "no robust transport advantage" in flipped["sentence"]

    spans = classify(
        "oracle_margin", stats(0.0020, 0.0018, lo_pp=-0.001, hi_pp=0.004), BAND
    )
    assert spans["case"] == "not_robust"

    # A negative sign is reported as measured, not softened.
    negative = classify("oracle_margin", stats(-0.0020, -0.0018), BAND)
    assert negative["case"] == "both_in_band"
    assert "small, sign-consistent negative" in negative["sentence"]


def test_the_three_way_wording_for_a_localization_gap():
    def stats(pp, sp):
        return {
            "M_pp": pp, "M_sp": sp, "dM": pp - sp,
            "M_pp_ci_low": pp * 0.5, "M_pp_ci_high": pp * 1.5,
            "M_sp_ci_low": sp * 0.5, "M_sp_ci_high": sp * 1.5,
        }

    both = classify("localization_gap", stats(0.0022, 0.0019), BAND)
    assert both["case"] == "both_in_band"
    assert "only a small, sign-consistent improvement over Neighbor-Patch" in both["sentence"]
    one = classify("localization_gap", stats(0.0500, 0.0019), BAND)
    assert "magnitude is path-sensitive" in one["sentence"]


def test_no_sentence_ever_calls_an_effect_significant_but_negligible():
    """The phrase is forbidden, and so is the shape of it."""
    for quantity in ("oracle_margin", "localization_gap"):
        for pp, sp in ((0.002, 0.0018), (0.05, 0.0018), (0.002, -0.0018), (0.5, 0.49)):
            verdict = classify(
                quantity,
                {"M_pp": pp, "M_sp": sp, "dM": pp - sp,
                 "M_pp_ci_low": min(pp * .5, pp * 1.5), "M_pp_ci_high": max(pp * .5, pp * 1.5),
                 "M_sp_ci_low": min(sp * .5, sp * 1.5), "M_sp_ci_high": max(sp * .5, sp * 1.5)},
                BAND,
            )
            text = verdict["sentence"].lower()
            assert "negligible" not in text
            assert "significant" not in text


def test_the_ratio_is_withheld_below_the_band():
    """|dM|/|M| needs a denominator larger than the operator difference."""
    small = classify(
        "oracle_margin",
        {"M_pp": 0.002, "M_sp": 0.0018, "dM": 0.0002,
         "M_pp_ci_low": 0.001, "M_pp_ci_high": 0.003,
         "M_sp_ci_low": 0.001, "M_sp_ci_high": 0.003},
        BAND,
    )
    assert np.isnan(small["abs_dM_over_abs_M_pp"])
    large = classify(
        "oracle_margin",
        {"M_pp": 0.100, "M_sp": 0.098, "dM": 0.002,
         "M_pp_ci_low": 0.09, "M_pp_ci_high": 0.11,
         "M_sp_ci_low": 0.09, "M_sp_ci_high": 0.11},
        BAND,
    )
    assert large["abs_dM_over_abs_M_pp"] == pytest.approx(0.02)


def test_build_runs_end_to_end_and_flags_a_near_zero_cell(tmp_path):
    rows = []
    for scene in range(4):
        # Enough pairs to clear support_min_camera_pairs, so the cells are
        # reported cells and the near-zero flag is exercised on one.
        for index in range(32):
            # A large DINOv2-scale margin and a near-zero one, same directory.
            rows += full_pair(
                f"scene_{scene}", index, 0.90, 0.50, 0.70, 0.90, 0.50, 0.70,
                encoder="dinov2_vitb14",
            )
            rows += full_pair(
                f"scene_{scene}", index, 0.9020, 0.9000, 0.9008,
                0.9019, 0.9000, 0.9007, encoder="vggt_1b",
            )
    eval_dir = tmp_path / "eval"
    scenes = sorted({r["scene"] for r in rows})
    for scene in scenes:
        meta = run_meta(scene, scenes)
        meta["encoders"] = ["dinov2_vitb14", "vggt_1b"]
        meta["cache_provenance"] = {
            e: dict(meta["cache_provenance"]["dinov2_vitb14"])
            for e in ("dinov2_vitb14", "vggt_1b")
        }
        write_scene(eval_dir, [r for r in rows if r["scene"] == scene], meta)

    table = build(eval_dir, ANALYSIS)
    assert table
    flagged = [r for r in table if r["near_zero"] and r["supported"]]
    assert {r["encoder"] for r in flagged} == {"vggt_1b"}, "only the tiny effects flag"
    for row in flagged:
        assert row["sentence"], "a flagged cell must carry restricted wording"
        assert row["replicates"] == ANALYSIS.bootstrap_resamples
    big = [r for r in table if r["encoder"] == "dinov2_vitb14"]
    assert all(not r["near_zero"] for r in big)
