"""PROTOCOL 3.3, 3.4, 3.9, 3.10: the analysis layer, from the parquet and the config."""

import math

import numpy as np
import pytest

from lot.analysis_config import load_analysis_config
from lot.evaluate import (
    MEAN_FEATURE,
    NEIGHBOR_PATCH,
    NO_WARP_COPY,
    ORACLE_TRANSPORT,
    PER_POINT,
    RANDOM_PATCH,
    SPLAT_POOL,
    pack_mask,
    write_rows,
)
from lot.figures import (
    CENTERED,
    RAW,
    assert_single_regime,
    assign_bins,
    bootstrap_interval,
    cell_summary,
    figure_a_null_ladder,
    figure_ceiling_and_floor,
    figure_d_orbit_joint,
    is_supported,
    mean_margin,
    mean_value,
    paired_records,
    path_agreement,
    read_eval_dir,
    restrict_to_regime,
    summary_table,
    support_counts,
    write_table,
)

ANALYSIS = load_analysis_config()
FULL_MASK = pack_mask(np.ones(16, dtype=bool))
HALF_MASK = pack_mask(np.array([True] * 8 + [False] * 8))


def make_row(**overrides):
    row = {
        "scene": "room_0",
        "split": "train",
        "viewpoint": 0,
        "regime": "translation",
        "context_frame_id": "c0",
        "target_frame_id": "t0",
        "baseline_m": 0.2,
        "context_median_depth_m": 2.0,
        "rotation_deg": 0.0,
        "parallax": 0.08,
        "covisible_fraction": 0.8,
        "encoder": "dinov2_vitb14",
        "path": PER_POINT,
        "variant": ORACLE_TRANSPORT,
        "n": 100,
        "n_intersect": 100,
        "coverage_difference": 0,
        "sample_mask": FULL_MASK,
        "cosine_mean": 0.9,
        "l2_mean": 0.4,
        "cosine_centered_mean": 0.8,
        "l2_centered_mean": 0.6,
        "cosine_intersect_mean": 0.9,
        "cosine_centered_intersect_mean": 0.8,
        "coverage_mean": float("nan"),
    }
    row.update(overrides)
    return row


def comparison(scores, **overrides):
    return [make_row(variant=v, cosine_mean=c, **overrides) for v, c in scores.items()]


def population(scenes=4, pairs=40, regime="translation", parallax=0.08, rotation=0.0):
    """Enough independent scenes and camera pairs to clear the support rule.

    Frame ids embed the group, because a comparison is keyed on scene and frame
    ids: reusing them across groups would make later rows overwrite earlier ones
    and quietly collapse the fixture to a single cell.
    """
    tag = f"{regime}_{parallax:g}_{rotation:g}"
    rows = []
    for scene_index in range(scenes):
        for pair_index in range(pairs):
            rows += comparison(
                {ORACLE_TRANSPORT: 0.9, NO_WARP_COPY: 0.5, MEAN_FEATURE: 0.2},
                scene=f"scene_{scene_index}",
                context_frame_id=f"{tag}_c{pair_index}",
                target_frame_id=f"{tag}_t{pair_index}",
                regime=regime,
                parallax=parallax,
                rotation_deg=rotation,
            )
    return rows


# ---------------------------------------------------------------------------
# B1: binning comes from the config, applied here
# ---------------------------------------------------------------------------

def test_bins_are_assigned_from_the_config_at_analysis_time():
    rows = assign_bins([make_row(parallax=0.08, rotation_deg=25.0)], ANALYSIS)
    assert rows[0]["parallax_bin"] == "0.05-0.1"
    assert rows[0]["rotation_bin"] == "20-30"


def test_rows_carrying_bin_labels_are_refused():
    """PROTOCOL 3.2 keeps labels out of rows so the config is the only source."""
    with pytest.raises(ValueError, match="already carry bin labels"):
        assign_bins([make_row(parallax_bin="0-0.025")], ANALYSIS)


def test_changing_the_config_changes_the_binning_with_no_source_edit():
    import dataclasses

    widened = dataclasses.replace(ANALYSIS, rotation_bin_edges_deg=(45.0,))
    assert assign_bins([make_row(rotation_deg=25.0)], ANALYSIS)[0]["rotation_bin"] == "20-30"
    assert assign_bins([make_row(rotation_deg=25.0)], widened)[0]["rotation_bin"] == "0-45"


# ---------------------------------------------------------------------------
# B2: regime discipline
# ---------------------------------------------------------------------------

def test_primary_curves_take_only_their_own_regime():
    """PROTOCOL 3.3: orbit pairs never appear on either marginal."""
    records, _ = paired_records(
        assign_bins(
            population(regime="translation") + population(regime="orbit", rotation=25.0), ANALYSIS
        )
    )
    parallax = restrict_to_regime(records, "parallax_bin")
    rotation = restrict_to_regime(records, "rotation_bin")
    assert {r["regime"] for r in parallax} == {"translation"}
    assert rotation == [] or {r["regime"] for r in rotation} == {"rotation"}


def test_a_foreign_regime_on_a_primary_curve_is_an_error():
    records, _ = paired_records(assign_bins(population(regime="orbit"), ANALYSIS))
    with pytest.raises(ValueError, match="keeps orbit out"):
        assert_single_regime(records, "translation")


# ---------------------------------------------------------------------------
# B5: paired differences on sample identity
# ---------------------------------------------------------------------------

def test_margins_are_paired_within_one_comparison():
    rows = assign_bins(
        comparison({ORACLE_TRANSPORT: 0.9, NO_WARP_COPY: 0.5}, context_frame_id="a")
        + comparison({ORACLE_TRANSPORT: 0.6, NO_WARP_COPY: 0.4}, context_frame_id="b"),
        ANALYSIS,
    )
    records, mismatches = paired_records(rows)
    margins = {r["value"]: r["margin"] for r in records if r["variant"] == ORACLE_TRANSPORT}
    assert margins == {0.9: pytest.approx(0.4), 0.6: pytest.approx(0.2)}
    assert mismatches == 0


def test_a_validity_asymmetry_changes_the_paired_result_and_not_the_naive_one():
    """B5's demonstration, and the reason masks are persisted at all.

    Two variants scored on different record sets have a difference of means that
    is part method and part population. The naive difference cannot tell, since
    it only ever sees the two numbers. The paired form reads the masks, sees the
    populations differ, and refuses rather than reporting a selection effect as a
    method effect.
    """
    rows = assign_bins(
        comparison({ORACLE_TRANSPORT: 0.9, NO_WARP_COPY: 0.5}, context_frame_id="a"), ANALYSIS
    )
    honest, mismatches = paired_records(rows)
    assert mismatches == 0
    naive_before = rows[0]["cosine_mean"] - rows[1]["cosine_mean"]

    for row in rows:
        if row["variant"] == ORACLE_TRANSPORT:
            row["sample_mask"] = HALF_MASK  # scored on half the records
    asymmetric, mismatches = paired_records(rows)
    naive_after = rows[0]["cosine_mean"] - rows[1]["cosine_mean"]

    assert mismatches == 1
    assert not [r for r in asymmetric if r["variant"] == ORACLE_TRANSPORT]
    assert [r for r in honest if r["variant"] == ORACLE_TRANSPORT]
    # The naive difference is unmoved by the asymmetry it cannot see.
    assert naive_after == naive_before


# ---------------------------------------------------------------------------
# B4: support and uncertainty
# ---------------------------------------------------------------------------

def test_support_counts_are_the_three_the_protocol_names():
    records, _ = paired_records(assign_bins(population(scenes=3, pairs=5), ANALYSIS))
    counts = support_counts([r for r in records if r["variant"] == ORACLE_TRANSPORT])
    assert counts["n_scenes"] == 3
    assert counts["n_camera_pairs"] == 5
    assert counts["n_feature_comparisons"] == 3 * 5 * 100


def test_support_rests_on_scenes_and_camera_pairs_not_comparison_count():
    """A single scene with a huge comparison count is still one scene."""
    plenty = {"n_scenes": 1, "n_camera_pairs": 4, "n_feature_comparisons": 10_000_000}
    assert not is_supported(plenty, ANALYSIS)
    enough = {
        "n_scenes": ANALYSIS.support_min_scenes,
        "n_camera_pairs": ANALYSIS.support_min_camera_pairs,
        "n_feature_comparisons": 10,
    }
    assert is_supported(enough, ANALYSIS)


def test_bootstrap_resamples_scenes_not_records():
    """Records within a scene are not independent draws.

    With every scene identical the interval must collapse, whatever the record
    count; resampling records instead would manufacture a narrow interval out of
    repeated measurements of the same scenes.
    """
    records, _ = paired_records(assign_bins(population(scenes=5, pairs=8), ANALYSIS))
    oracle = [r for r in records if r["variant"] == ORACLE_TRANSPORT]
    low, high = bootstrap_interval(oracle, mean_margin, ANALYSIS, unit="scene")
    assert low == pytest.approx(0.4, abs=1e-9)
    assert high == pytest.approx(0.4, abs=1e-9)


def test_bootstrap_widens_when_scenes_disagree():
    rows = []
    for index in range(6):
        rows += comparison(
            {ORACLE_TRANSPORT: 0.9 if index % 2 else 0.5, NO_WARP_COPY: 0.5},
            scene=f"scene_{index}",
        )
    records, _ = paired_records(assign_bins(rows, ANALYSIS))
    oracle = [r for r in records if r["variant"] == ORACLE_TRANSPORT]
    low, high = bootstrap_interval(oracle, mean_margin, ANALYSIS, unit="scene")
    assert high - low > 0.05


def test_bootstrap_replicates_call_the_point_estimate_function():
    """One mechanism: the replicate and the estimate share code.

    A ratio statistic resampled from precomputed per-scene values is wrong, and
    Phase 4 brings ratio statistics, so the recompute form is the one that has
    to be right here too.
    """
    calls = []

    def counting_statistic(records):
        calls.append(len(records))
        return mean_margin(records)

    records, _ = paired_records(assign_bins(population(scenes=3, pairs=2), ANALYSIS))
    bootstrap_interval(records, counting_statistic, ANALYSIS, unit="scene")
    assert len(calls) == ANALYSIS.bootstrap_resamples


def test_cell_summary_carries_both_intervals_and_the_support_verdict():
    records, _ = paired_records(assign_bins(population(scenes=4, pairs=40), ANALYSIS))
    summary = cell_summary(
        [r for r in records if r["variant"] == ORACLE_TRANSPORT], ANALYSIS
    )
    for key in (
        "estimate", "ci_low", "ci_high", "pair_ci_low", "pair_ci_high",
        "n_scenes", "n_camera_pairs", "n_feature_comparisons", "supported",
    ):
        assert key in summary
    assert summary["supported"]


# ---------------------------------------------------------------------------
# PROTOCOL 3.9: agreement on the cross-path intersection
# ---------------------------------------------------------------------------

def test_path_agreement_uses_the_intersection_columns():
    rows = []
    for path, intersect in ((PER_POINT, 0.900), (SPLAT_POOL, 0.9005)):
        rows += comparison(
            {ORACLE_TRANSPORT: 0.5, NO_WARP_COPY: 0.4},  # full-population scores differ wildly
            path=path,
            cosine_intersect_mean=intersect,
            coverage_difference=3,
        )
    result = path_agreement(assign_bins(rows, ANALYSIS), ANALYSIS)
    assert result["comparisons"] == 1
    assert result["aggregate_abs_difference"] == pytest.approx(0.0005, abs=1e-9)
    assert result["max_abs_difference"] == pytest.approx(0.0005, abs=1e-9)
    assert result["within_tolerance"]
    assert result["max_coverage_difference_cells"] == 6


def test_path_agreement_reports_coverage_beside_it_not_inside_it():
    rows = []
    for path in (PER_POINT, SPLAT_POOL):
        rows += comparison(
            {ORACLE_TRANSPORT: 0.9, NO_WARP_COPY: 0.5},
            path=path,
            cosine_intersect_mean=0.9,
            coverage_difference=17,
        )
    result = path_agreement(assign_bins(rows, ANALYSIS), ANALYSIS)
    assert result["aggregate_abs_difference"] == pytest.approx(0.0)
    assert result["mean_coverage_difference_cells"] == pytest.approx(34.0)


# ---------------------------------------------------------------------------
# B3: the four figures
# ---------------------------------------------------------------------------

def full_records():
    rows = []
    for parallax in (0.03, 0.08, 0.3):
        rows += population(scenes=4, pairs=40, regime="translation", parallax=parallax)
    for rotation in (5.0, 25.0, 45.0):
        rows += population(
            scenes=4, pairs=12, regime="rotation", parallax=0.0, rotation=rotation
        )
    for parallax, rotation in ((0.08, 15.0), (0.3, 35.0)):
        rows += population(
            scenes=4, pairs=12, regime="orbit", parallax=parallax, rotation=rotation
        )
    rows += [make_row(variant=RANDOM_PATCH, cosine_mean=0.1, scene=f"scene_{i}") for i in range(4)]
    rows += [make_row(variant=NEIGHBOR_PATCH, cosine_mean=0.7, scene=f"scene_{i}") for i in range(4)]
    records = []
    for metric in (RAW, CENTERED):
        part, _ = paired_records(assign_bins(rows, ANALYSIS), metric=metric)
        records.extend(part)
    return records


def test_all_four_figures_are_produced(tmp_path):
    """PROTOCOL 3.10 requires four; the previous run produced two."""
    records = full_records()
    figure_a_null_ladder(records, tmp_path / "a.png", ANALYSIS, omissions={NEIGHBOR_PATCH: 32})
    figure_ceiling_and_floor(records, tmp_path / "b.png", ANALYSIS, "parallax_bin", "B")
    figure_ceiling_and_floor(records, tmp_path / "c.png", ANALYSIS, "rotation_bin", "C")
    figure_d_orbit_joint(records, tmp_path / "d.png", ANALYSIS)
    for name in "abcd":
        target = tmp_path / f"{name}.png"
        assert target.is_file() and target.stat().st_size > 0


def test_figure_d_needs_orbit_records():
    with pytest.raises(ValueError, match="no orbit"):
        figure_d_orbit_joint(
            [r for r in full_records() if r["regime"] != "orbit"], "unused.png", ANALYSIS
        )


def test_summary_table_carries_support_and_intervals_on_every_row():
    table = summary_table(full_records(), ANALYSIS)
    assert table
    for row in table:
        for key in (
            "n_scenes", "n_camera_pairs", "n_feature_comparisons",
            "margin_ci_low", "margin_ci_high", "margin_pair_ci_low", "supported",
        ):
            assert key in row
        assert row["analysis"] in ("translation", "rotation", "orbit_minus_translation")


def test_table_and_figures_regenerate_from_the_parquet_alone(tmp_path):
    """CLAUDE.md: the figure must come from outputs/eval/*.parquet and the config."""
    rows = []
    for parallax in (0.03, 0.08, 0.3):
        rows += population(scenes=4, pairs=40, parallax=parallax)
    eval_dir = tmp_path / "eval"
    write_rows(eval_dir / "room_0.parquet", rows, {"scene": "room_0"})

    reread = assign_bins(read_eval_dir(eval_dir), ANALYSIS)
    records, mismatches = paired_records(reread)
    assert mismatches == 0
    table = tmp_path / "tables" / "experiment_zero.parquet"
    write_table(table, summary_table(records, ANALYSIS))
    assert table.is_file()
    with pytest.raises(FileExistsError):
        write_table(table, summary_table(records, ANALYSIS))


def test_read_eval_dir_needs_something_to_read(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_eval_dir(tmp_path)


# ---------------------------------------------------------------------------
# Regressions from the external review
# ---------------------------------------------------------------------------

def test_matched_difference_support_needs_both_arms():
    """Two arms that each fail the threshold must not add up to a supported cell."""
    from lot.figures import matched_summaries

    rows = population(scenes=4, pairs=20, regime="orbit", parallax=0.08, rotation=15.0)
    rows += population(scenes=4, pairs=20, regime="translation", parallax=0.08)
    records, _ = paired_records(assign_bins(rows, ANALYSIS))
    joint = [r for r in records if r["variant"] == ORACLE_TRANSPORT]
    summaries = matched_summaries(joint, ("encoder", "parallax_bin"), ANALYSIS)
    assert summaries
    for summary in summaries.values():
        # 20 pairs per arm, threshold 30: pooled would reach 40 and pass.
        assert summary["arm_support"] == {"orbit": 20, "translation": 20}
        assert not summary["supported"]


def test_path_agreement_gates_the_aggregate_and_reports_the_spread():
    """The 0.003 tolerance was established on pooled per-path scores.

    Gating on the worst single pair would apply that number to a statistic it
    was never measured against, so the aggregate is gated and the per-pair
    spread is reported beside it.
    """
    rows = []
    for index, (a, b) in enumerate([(0.90, 0.90), (0.90, 0.90), (0.80, 0.95)]):
        for path, value in ((PER_POINT, a), (SPLAT_POOL, b)):
            rows += comparison(
                {ORACLE_TRANSPORT: 0.5, NO_WARP_COPY: 0.4},
                path=path,
                cosine_intersect_mean=value,
                context_frame_id=f"c{index}",
                target_frame_id=f"t{index}",
            )
    result = path_agreement(assign_bins(rows, ANALYSIS), ANALYSIS)
    assert result["max_abs_difference"] == pytest.approx(0.15, abs=1e-9)
    assert result["pairs_over_tolerance"] == 1
    # The aggregate difference is small even though one pair is far out.
    assert result["aggregate_abs_difference"] == pytest.approx(0.05, abs=1e-9)
    assert not result["within_tolerance"]


def test_the_estimand_is_the_unweighted_pair_mean_with_the_weighted_one_beside_it():
    """PROTOCOL 3.4 makes the camera pair the unit, so the pair mean is the estimand.

    The weighted number is emitted beside it because the weighting is not
    neutral: a pair's comparison count rises with the easier geometry, so a
    weighted mean leans on the pairs that were least difficult to transport.
    """
    from lot.figures import comparison_weighted, mean_margin

    cell = [{"margin": 0.10, "n": 1}, {"margin": 0.50, "n": 99}]
    assert mean_margin(cell) == pytest.approx(0.30)
    assert comparison_weighted(cell, "margin") == pytest.approx(0.4960)
    assert math.isnan(comparison_weighted([], "margin"))
    assert math.isnan(comparison_weighted([{"margin": 0.5, "n": 0}], "margin"))


def test_the_summary_table_carries_both_estimands():
    """Every per-regime row, but not the matched difference.

    orbit_minus_translation is a difference of two arms, so there is no single
    record set to weight and no weighted counterpart to report.
    """
    table = summary_table(full_records(), ANALYSIS)
    per_regime = [r for r in table if r["analysis"] != "orbit_minus_translation"]
    matched = [r for r in table if r["analysis"] == "orbit_minus_translation"]
    assert per_regime and matched
    for row in per_regime:
        assert math.isfinite(row["margin_comparison_weighted"])
        assert math.isfinite(row["value_comparison_weighted"])
    for row in matched:
        assert "margin_comparison_weighted" not in row
