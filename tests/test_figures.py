"""PROTOCOL 3.3, 3.4, 3.9, 3.10: the analysis layer, from the parquet and the config."""

import hashlib
import math
from pathlib import Path

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
from lot.evaluate import EVAL_VERSION
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


# Default scores for the variants a test does not name. PROTOCOL 3.7 gives every
# variant of a path the same records, so a comparison carries all five; a fixture
# that built three was modelling something the protocol forbids.
DEFAULT_SCORES = {
    RANDOM_PATCH: 0.05,
    MEAN_FEATURE: 0.20,
    NO_WARP_COPY: 0.50,
    NEIGHBOR_PATCH: 0.70,
    ORACLE_TRANSPORT: 0.90,
}


def comparison(scores, complete=True, **overrides):
    """One comparison's rows. Pass complete=False to build a partial one on purpose."""
    full = {**DEFAULT_SCORES, **scores} if complete else dict(scores)
    return [make_row(variant=v, cosine_mean=c, **overrides) for v, c in full.items()]


def run_meta(scene, run_scenes=None):
    """The provenance sidecar a complete run writes beside each parquet."""
    return {
        "scene": scene,
        "eval_version": EVAL_VERSION,
        "encoders": ["dinov2_vitb14"],
        "seed": 0,
        "max_pairs_per_stratum": ANALYSIS.max_pairs_per_stratum,
        "git_commit": "0" * 40,
        "run_scenes": sorted(run_scenes or [scene]),
        "analysis_config_digest": ANALYSIS.digest(),
        "analysis_measurement_digest": ANALYSIS.measurement_digest(),
        "analysis_reporting_digest": ANALYSIS.reporting_digest(),
        "cache_provenance": {
            "dinov2_vitb14": {
                "features_digest": hashlib.blake2b(
                    scene.encode("utf-8"), digest_size=16
                ).hexdigest(),
                "weights_fingerprint": "f" * 32,
                "weights_revision": "unpinnable: torch.hub checkpoint URL is unversioned",
                "code_revision": "c" * 40,
            }
        },
    }


def scene_counters(rows):
    """The per-path pair counters the evaluator would have recorded."""
    paths_by_pair = {}
    for row in rows:
        pair = (row["context_frame_id"], row["target_frame_id"])
        paths_by_pair.setdefault(pair, set()).add(row["path"])
    return {
        "pairs_scored_both_paths": sum(1 for p in paths_by_pair.values() if len(p) == 2),
        "pairs_scored_per_point_only": sum(
            1 for p in paths_by_pair.values() if p == {PER_POINT}
        ),
        "pairs_scored_splat_only": sum(
            1 for p in paths_by_pair.values() if p == {SPLAT_POOL}
        ),
    }


def write_scene(eval_dir, rows, meta):
    """Write one scene's parquet with counters matching its rows.

    Counters already present in meta are kept, which is how a test builds a
    deliberate mismatch.
    """
    full = {**scene_counters(rows), **meta}
    write_rows(eval_dir / f"{meta['scene']}.parquet", rows, full)
    return full


def population(scenes=4, pairs=40, regime="translation", parallax=0.08, rotation=0.0):
    """Enough independent scenes and camera pairs to clear the support rule.

    Frame ids embed the group, because a comparison is keyed on scene and frame
    ids: reusing them across groups would make later rows overwrite earlier ones
    and quietly collapse the fixture to a single cell.
    """
    tag = f"{regime}_{parallax:g}_{rotation:g}"
    rows = []
    # A single-scene population carries the scene name the sidecar fixtures
    # use, because read_eval_dir now reconciles each file's rows against its
    # record: a file whose rows name another scene is a mixed directory.
    names = ["room_0"] if scenes == 1 else [f"scene_{i}" for i in range(scenes)]
    for scene_index in range(scenes):
        for pair_index in range(pairs):
            rows += comparison(
                {ORACLE_TRANSPORT: 0.9, NO_WARP_COPY: 0.5, MEAN_FEATURE: 0.2},
                scene=names[scene_index],
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
    assert result["cosine_mean_mean_abs_difference"] == pytest.approx(0.0005, abs=1e-9)
    assert result["cosine_mean_max_abs_difference"] == pytest.approx(0.0005, abs=1e-9)
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
    assert result["cosine_mean_mean_abs_difference"] == pytest.approx(0.0)
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
    # The two extra null rows that used to sit here are gone: population() now
    # emits all five variants, so appending bare Random-Patch and Neighbor-Patch
    # rows on the default frame ids built an incomplete comparison, which is what
    # PROTOCOL 3.7 forbids and what paired_records now refuses.
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
    for scene in sorted({r["scene"] for r in rows}):
        write_scene(
            eval_dir,
            [r for r in rows if r["scene"] == scene],
            run_meta(scene, sorted({r["scene"] for r in rows})),
        )

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


def test_read_eval_dir_refuses_an_incomplete_directory(tmp_path):
    """A failed array task is an absent scene, not a smaller population."""
    eval_dir = tmp_path / "eval"
    rows = population(scenes=1, pairs=4)
    write_rows(eval_dir / "room_0.parquet", rows, run_meta("room_0", ["room_0", "room_1"]))
    with pytest.raises(ValueError, match="1 of the 2 scenes"):
        read_eval_dir(eval_dir)


def test_read_eval_dir_refuses_a_mixed_directory(tmp_path):
    """Two runs that measured different things are not concatenated."""
    eval_dir = tmp_path / "eval"
    scenes = ["room_0", "room_1"]
    write_rows(eval_dir / "room_0.parquet", population(scenes=1, pairs=4), run_meta("room_0", scenes))
    other = run_meta("room_1", scenes)
    other["seed"] = 99
    write_rows(eval_dir / "room_1.parquet", population(scenes=1, pairs=4), other)
    with pytest.raises(ValueError, match="mixes runs: seed"):
        read_eval_dir(eval_dir)


def test_the_parquet_alone_is_enough(tmp_path):
    """CLAUDE.md: every figure regenerable from outputs/eval/*.parquet alone.

    The provenance the analysis refuses to run without is carried inside the
    parquet, so deleting the sidecars leaves intact results analysable. A run
    record that lived only in a companion file would have made those files a
    second required artifact set that nothing declared.
    """
    eval_dir = tmp_path / "eval"
    write_scene(eval_dir, population(scenes=1, pairs=4), run_meta("room_0"))
    (eval_dir / "room_0.meta.json").unlink()
    rows = read_eval_dir(eval_dir)
    assert rows
    records, _ = paired_records(assign_bins(rows, ANALYSIS))
    assert summary_table(records, ANALYSIS)


def test_read_eval_dir_refuses_an_older_eval_version(tmp_path):
    """Rows written before the repair draw their nulls from other hashes."""
    eval_dir = tmp_path / "eval"
    stale = run_meta("room_0")
    stale["eval_version"] = EVAL_VERSION - 1
    write_rows(eval_dir / "room_0.parquet", population(scenes=1, pairs=4), stale)
    with pytest.raises(ValueError, match="eval_version"):
        read_eval_dir(eval_dir)


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


def test_path_agreement_cannot_pass_through_cancellation():
    """Opposite-signed pair errors must not sum away.

    The gated quantity is the mean of the per-pair absolute differences. Two
    earlier choices were both wrong. Gating the largest single pair applies a
    tolerance established on pooled scores to a statistic it was never measured
    against. Gating |mean(a) - mean(b)| fixes that and destroys the measurement:
    here one pair reads 0.2 high and one reads 0.2 low, so that quantity is
    exactly zero while both pairs disagree by far more than the tolerance.
    """
    rows = []
    for index, (a, b) in enumerate([(0.90, 0.70), (0.70, 0.90)]):
        for path, value in ((PER_POINT, a), (SPLAT_POOL, b)):
            rows += comparison(
                {ORACLE_TRANSPORT: 0.5, NO_WARP_COPY: 0.4},
                path=path,
                cosine_intersect_mean=value,
                context_frame_id=f"c{index}",
                target_frame_id=f"t{index}",
            )
    result = path_agreement(assign_bins(rows, ANALYSIS), ANALYSIS)
    assert abs(np.mean([0.90, 0.70]) - np.mean([0.70, 0.90])) == pytest.approx(0.0)
    assert result["cosine_mean_mean_abs_difference"] == pytest.approx(0.20, abs=1e-9)
    assert result["cosine_mean_pairs_over_tolerance"] == 2
    assert not result["within_tolerance"]


def test_path_agreement_gates_the_centered_metric_too():
    """Centering does not act equally on the two paths.

    The splat path's pooled output is a weighted mean over a cell and the
    per-point path's is a single sample, so removing the shared component can
    expose a disagreement the raw metric hides. A gate that read only the raw
    column would certify a centered table it never looked at.
    """
    rows = []
    for path, centered in ((PER_POINT, 0.80), (SPLAT_POOL, 0.50)):
        rows += comparison(
            {ORACLE_TRANSPORT: 0.5, NO_WARP_COPY: 0.4},
            path=path,
            cosine_intersect_mean=0.9,           # raw agrees exactly
            cosine_centered_intersect_mean=centered,
        )
    result = path_agreement(assign_bins(rows, ANALYSIS), ANALYSIS)
    assert result["cosine_mean_mean_abs_difference"] == pytest.approx(0.0)
    assert result["cosine_centered_mean_mean_abs_difference"] == pytest.approx(0.30, abs=1e-9)
    assert not result["within_tolerance"]


def test_path_agreement_refuses_duplicate_rows():
    """A comparison scored twice means the directory mixes runs."""
    rows = comparison({ORACLE_TRANSPORT: 0.5, NO_WARP_COPY: 0.4}, path=PER_POINT)
    rows += comparison({ORACLE_TRANSPORT: 0.5, NO_WARP_COPY: 0.4}, path=SPLAT_POOL)
    rows += comparison({ORACLE_TRANSPORT: 0.9, NO_WARP_COPY: 0.4}, path=PER_POINT)
    result = path_agreement(assign_bins(rows, ANALYSIS), ANALYSIS)
    assert result["duplicate_rows"] == 1
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


def test_paired_records_refuses_a_repeated_comparison_and_variant():
    """A duplicate is a corrupt population, not a per-comparison defect.

    A mask mismatch is a property of one comparison and the analysis proceeds
    without it. A repeated (comparison, variant) means the directory holds two
    runs, so every aggregate over it is drawn from a population nobody chose.
    """
    rows = comparison({ORACLE_TRANSPORT: 0.6, NO_WARP_COPY: 0.4})
    rows += comparison({ORACLE_TRANSPORT: 0.9})
    with pytest.raises(ValueError, match="duplicate"):
        paired_records(assign_bins(rows, ANALYSIS))


def test_omission_count_deduplicates_the_pair_level_column(tmp_path):
    """The column repeats a pair-level number into every row of that pair.

    Summing it over-counts by the number of encoders, paths, and variants, and
    the divisor it was once corrected by, the number of metrics, has no relation
    to that duplication. Deduplicating by pair recovers the quantity, from the
    parquet rather than a sidecar.
    """
    from lot.figures import neighbor_omitted_total

    rows = []
    for index, omitted in enumerate((7, 5)):
        rows += comparison(
            {},
            context_frame_id=f"c{index}",
            target_frame_id=f"t{index}",
            neighbor_omitted=omitted,
        )
    # Five variants per pair, so the naive sum would be 60.
    assert sum(r["neighbor_omitted"] for r in rows) == 60
    assert neighbor_omitted_total(rows) == 12


def test_absolute_scores_carry_the_camera_pair_interval_too():
    """PROTOCOL 3.4 makes the pair interval secondary for every reported number.

    Margins had it and absolute scores did not, so half the table carried only
    the scene-level interval.
    """
    table = summary_table(full_records(), ANALYSIS)
    per_regime = [r for r in table if r["analysis"] != "orbit_minus_translation"]
    assert per_regime
    for row in per_regime:
        for column in ("value_pair_ci_low", "value_pair_ci_high",
                       "margin_pair_ci_low", "margin_pair_ci_high"):
            assert column in row


def test_a_comparison_missing_a_variant_is_refused_not_skipped():
    """A failed method must not shrink the population invisibly.

    Skipping an incomplete comparison takes its pair out of every aggregate and
    leaves a table that reads as if the pair had never been sampled. If a method
    failed on the hardest pairs, that is precisely the population the skip
    removes.
    """
    rows = comparison({ORACLE_TRANSPORT: 0.9}, complete=False)
    rows += comparison({NO_WARP_COPY: 0.5}, complete=False)
    with pytest.raises(ValueError, match="do not carry all 5 variants"):
        paired_records(assign_bins(rows, ANALYSIS))

    # And an Oracle row with no floor at all produced nothing and said nothing.
    orphan = comparison({ORACLE_TRANSPORT: 0.9, MEAN_FEATURE: 0.2}, complete=False)
    with pytest.raises(ValueError, match="missing"):
        paired_records(assign_bins(orphan, ANALYSIS))


def test_nonfinite_is_permitted_only_for_centered_mean_feature():
    """PROTOCOL 3.7 permits it there and nowhere else.

    Mean-Feature has no centered value by construction: the vector subtracted is
    the prediction itself, so the centered prediction is the zero vector and its
    cosine is undefined. A nonfinite anywhere else is a failed method, and
    dropping it silently biases the cell it was in.
    """
    ok = [
        make_row(
            variant=variant,
            cosine_mean=score,
            cosine_centered_mean=float("nan") if variant == MEAN_FEATURE else score - 0.1,
        )
        for variant, score in DEFAULT_SCORES.items()
    ]
    records, _ = paired_records(assign_bins(ok, ANALYSIS), metric=CENTERED)
    assert {r["variant"] for r in records} == set(DEFAULT_SCORES) - {MEAN_FEATURE}

    bad = comparison({ORACLE_TRANSPORT: float("nan")})
    with pytest.raises(ValueError, match="nonfinite"):
        paired_records(assign_bins(bad, ANALYSIS))


def test_a_record_missing_a_required_field_is_refused_not_treated_as_agreeing():
    """Comparing only what is present makes an absent field agree with itself.

    Every value is null, the set of distinct values has one element, and a
    directory written before the field existed passes the check that was added
    to catch it.
    """
    import copy
    import json as _json
    from lot.figures import read_eval_dir

    import pytest as _pytest

    def build(tmp, drop):
        eval_dir = tmp / "eval"
        scenes = ["room_0", "room_1"]
        for scene in scenes:
            meta = copy.deepcopy(run_meta(scene, scenes))
            meta.pop(drop, None)
            write_rows(eval_dir / f"{scene}.parquet", population(scenes=1, pairs=4), meta)
        return eval_dir

    import tempfile

    for field in ("analysis_config_digest", "cache_provenance"):
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = build(Path(tmp), field)
            with _pytest.raises(ValueError, match=f"carry no {field}"):
                read_eval_dir(eval_dir)


def test_a_report_is_bound_to_the_config_that_measured_it_but_not_to_how_it_reports(tmp_path):
    """PROTOCOL 3.4 sets support thresholds after the run, from counts alone.

    So the whole config cannot be required to match, or the documented workflow
    is forbidden. The values that decided what the rows contain must match, or
    the same parquet yields a different report with no complaint, and none of
    those values is recoverable from the rows.
    """
    import dataclasses

    eval_dir = tmp_path / "eval"
    write_scene(eval_dir, population(scenes=1, pairs=4), run_meta("room_0"))

    # A reporting edit is permitted: this is the counts-then-threshold workflow.
    reporting = dataclasses.replace(ANALYSIS, support_min_camera_pairs=99)
    assert reporting.measurement_digest() == ANALYSIS.measurement_digest()
    assert read_eval_dir(eval_dir, reporting)

    # A measurement edit is refused.
    measurement = dataclasses.replace(ANALYSIS, covisible_relative_depth_tol=0.02)
    assert measurement.reporting_digest() == ANALYSIS.reporting_digest()
    with pytest.raises(ValueError, match="measurement config"):
        read_eval_dir(eval_dir, measurement)


def test_the_interval_says_how_many_replicates_produced_it():
    """A quantile over three draws and one over a thousand print identically.

    A cell whose statistic is undefined in a replicate contributes nothing to
    that draw, and there is no honest way to make it; the quantiles are then
    over the draws in which the cell existed. That is tolerable and has to be
    visible.
    """
    import dataclasses

    from lot.figures import bootstrap_cells

    config = dataclasses.replace(ANALYSIS, bootstrap_resamples=200)
    records = [{"scene": "lonely", "camera_pair": ("a", "b"), "bin": "X",
                "margin": 0.42, "n": 10}]
    for i in range(17):
        records.append({"scene": f"s{i}", "camera_pair": (f"c{i}", f"t{i}"),
                        "bin": "Y", "margin": 0.1 + 0.01 * i, "n": 10})

    intervals = bootstrap_cells(records, ("bin",), mean_margin, config, unit="scene")
    low_x, high_x, draws_x = intervals[("X",)]
    _, _, draws_y = intervals[("Y",)]
    # The lonely cell's interval has zero width because every replicate that
    # contains its one scene gives the same estimate. The replicate count is
    # what tells the reader the interval is not what it looks like.
    assert low_x == high_x == pytest.approx(0.42)
    assert 0 < draws_x < config.bootstrap_resamples
    assert draws_y == config.bootstrap_resamples


def test_the_table_carries_the_replicate_counts():
    """Every interval on every row kind carries its replicate count.

    The matched orbit-minus-translation rows matter most: that statistic is NaN
    in any replicate whose resample drops an arm, so its interval is the one
    most likely to rest on a subset of the draws.
    """
    table = summary_table(full_records(), ANALYSIS)
    per_regime = [r for r in table if r["analysis"] != "orbit_minus_translation"]
    matched = [r for r in table if r["analysis"] == "orbit_minus_translation"]
    assert per_regime and matched
    for row in per_regime:
        assert row["bootstrap_resamples"] == ANALYSIS.bootstrap_resamples
        assert 0 <= row["margin_ci_replicates"] <= row["bootstrap_resamples"]
        assert 0 <= row["margin_pair_ci_replicates"] <= row["bootstrap_resamples"]
        assert 0 <= row["value_ci_replicates"] <= row["bootstrap_resamples"]
        assert 0 <= row["value_pair_ci_replicates"] <= row["bootstrap_resamples"]
    for row in matched:
        assert row["bootstrap_resamples"] == ANALYSIS.bootstrap_resamples
        assert 0 <= row["margin_ci_replicates"] <= row["bootstrap_resamples"]
        assert 0 <= row["margin_pair_ci_replicates"] <= row["bootstrap_resamples"]


def test_counts_cover_both_paths():
    """Thresholds set from this view govern splat cells too.

    A pair can be scorable on one path and not the other, so per-bin counts
    differ by path. Showing only the per-point counts let a threshold be chosen
    against numbers that did not describe half the cells it would gate.
    """
    from lot.figures import counts_table, format_counts

    rows = []
    for path, pairs in ((PER_POINT, 40), (SPLAT_POOL, 12)):
        for scene_index in range(4):
            for pair_index in range(pairs):
                rows += comparison(
                    {},
                    scene=f"scene_{scene_index}",
                    context_frame_id=f"{path}_c{pair_index}",
                    target_frame_id=f"{path}_t{pair_index}",
                    path=path,
                )
    records = []
    for metric in (RAW, CENTERED):
        part, _ = paired_records(assign_bins(rows, ANALYSIS), metric=metric)
        records.extend(part)

    table = counts_table(records, ANALYSIS)
    assert table
    assert {row["path"] for row in table} == {PER_POINT, SPLAT_POOL}
    by_path = {row["path"]: row["n_camera_pairs"] for row in table}
    # The two paths carry different support, which is the whole reason both
    # have to be shown: a threshold read off the per-point column alone would
    # have been chosen against numbers that did not describe the splat cells.
    assert by_path[PER_POINT] == 40 and by_path[SPLAT_POOL] == 12
    printed = format_counts(table, ANALYSIS)
    assert "path" in printed and SPLAT_POOL in printed


def test_scenes_evaluated_through_different_inference_code_are_refused(tmp_path):
    """The encoder's identity is weights, revision, and inference code together.

    The same state dict run through two VGGT inference commits produces
    different features, so comparing fingerprints alone accepted a mixture
    wearing one fingerprint.
    """
    import copy

    eval_dir = tmp_path / "eval"
    scenes = ["room_0", "room_1"]
    for scene, commit in zip(scenes, ("a" * 40, "b" * 40)):
        meta = copy.deepcopy(run_meta(scene, scenes))
        meta["cache_provenance"]["dinov2_vitb14"]["code_revision"] = commit
        write_rows(eval_dir / f"{scene}.parquet", population(scenes=1, pairs=4), meta)
    with pytest.raises(ValueError, match="different encoder identities"):
        read_eval_dir(eval_dir)


def test_unpinned_provenance_refuses_the_report(tmp_path):
    """PROTOCOL locks the encoders, so a warning was not a gate.

    'unpinnable: ...' is a declaration that there is nothing to pin, made by a
    loader that pinned everything it could, and it is accepted. 'unpinned' and
    'unknown' mean nobody pinned what could have been.
    """
    import copy

    eval_dir = tmp_path / "eval"
    meta = copy.deepcopy(run_meta("room_0"))
    meta["cache_provenance"]["dinov2_vitb14"]["code_revision"] = "unpinned"
    write_rows(eval_dir / "room_0.parquet", population(scenes=1, pairs=4), meta)
    with pytest.raises(ValueError, match="without a full pin"):
        read_eval_dir(eval_dir)

    # The declared-unpinnable weights marker alone is not a refusal: the
    # default run_meta fixture carries it and reads cleanly.
    clean_dir = tmp_path / "eval_clean"
    write_scene(clean_dir, population(scenes=1, pairs=4), run_meta("room_0"))
    assert read_eval_dir(clean_dir)


def test_a_branch_name_is_not_a_pin(tmp_path):
    """'main' passed the blocklist, and a branch name is a moving ref.

    The rule is positive now: a pin is a full commit hash, or for weights alone
    the explicit unpinnable declaration. A blocklist accepts everything it did
    not think of, which is the opposite of what a gate is for.
    """
    import copy

    eval_dir = tmp_path / "eval"
    meta = copy.deepcopy(run_meta("room_0"))
    meta["cache_provenance"]["dinov2_vitb14"]["code_revision"] = "main"
    write_rows(eval_dir / "room_0.parquet", population(scenes=1, pairs=4), meta)
    with pytest.raises(ValueError, match="without a full pin"):
        read_eval_dir(eval_dir)


def test_provenance_must_cover_the_run(tmp_path):
    """An empty mapping exists, so the presence check passed it.

    Every downstream loop then iterated nothing and refused nothing: a sidecar
    with cache_provenance {} sailed past the whole pin apparatus.
    """
    import copy

    eval_dir = tmp_path / "eval"
    meta = copy.deepcopy(run_meta("room_0"))
    meta["cache_provenance"] = {}
    write_rows(eval_dir / "room_0.parquet", population(scenes=1, pairs=4), meta)
    with pytest.raises(ValueError, match="gates nothing"):
        read_eval_dir(eval_dir)

    # And provenance for a different encoder set than the run declares.
    partial_dir = tmp_path / "eval_partial"
    meta = copy.deepcopy(run_meta("room_0"))
    meta["encoders"] = ["dinov2_vitb14", "vggt_1b"]
    write_rows(partial_dir / "room_0.parquet", population(scenes=1, pairs=4), meta)
    with pytest.raises(ValueError, match="gates nothing"):
        read_eval_dir(partial_dir)


def test_a_missing_fingerprint_is_refused_however_consistently_missing(tmp_path):
    """None compared equal to None across scenes, so an absent field matched.

    For DINOv2 the fingerprint is the only evidence the unversioned checkpoint
    bytes did not change, so its absence is precisely the case that must not
    slide through an identity comparison built on entry.get().
    """
    import copy

    for field in ("weights_fingerprint", "features_digest"):
        eval_dir = tmp_path / f"eval_{field}"
        meta = copy.deepcopy(run_meta("room_0"))
        del meta["cache_provenance"]["dinov2_vitb14"][field]
        write_scene(eval_dir, population(scenes=1, pairs=4), meta)
        with pytest.raises(ValueError, match=f"well-formed {field}"):
            read_eval_dir(eval_dir)

    # And a present but malformed value is the same defect wearing a value.
    eval_dir = tmp_path / "eval_malformed"
    meta = copy.deepcopy(run_meta("room_0"))
    meta["cache_provenance"]["dinov2_vitb14"]["weights_fingerprint"] = "TODO"
    write_scene(eval_dir, population(scenes=1, pairs=4), meta)
    with pytest.raises(ValueError, match="well-formed weights_fingerprint"):
        read_eval_dir(eval_dir)


def test_an_encoder_with_no_rows_is_refused(tmp_path):
    """A declared encoder must appear in the rows.

    A whole encoder disappearing leaves no partial comparison for the
    completeness checks to notice: every comparison that remains is complete.
    """
    import copy

    eval_dir = tmp_path / "eval"
    meta = copy.deepcopy(run_meta("room_0"))
    meta["encoders"] = ["dinov2_vitb14", "vggt_1b"]
    meta["cache_provenance"]["vggt_1b"] = dict(meta["cache_provenance"]["dinov2_vitb14"])
    write_scene(eval_dir, population(scenes=1, pairs=4), meta)
    with pytest.raises(ValueError, match="disappeared without leaving"):
        read_eval_dir(eval_dir)


def test_a_vanished_pair_is_refused_by_the_counter_reconciliation(tmp_path):
    """Deleting a whole pair leaves no incomplete comparison behind.

    The evaluator's per-path counters are the record of the population it
    scored, so rows that hold fewer pairs than the counters declare are a
    modified file, not a smaller run.
    """
    rows = population(scenes=1, pairs=4)
    counters = scene_counters(rows)
    assert counters["pairs_scored_per_point_only"] == 4
    kept = [r for r in rows if r["context_frame_id"] != rows[0]["context_frame_id"]]

    eval_dir = tmp_path / "eval"
    write_scene(eval_dir, kept, {**run_meta("room_0"), **counters})
    with pytest.raises(ValueError, match="added or removed"):
        read_eval_dir(eval_dir)


def test_a_file_holding_another_scenes_rows_is_refused(tmp_path):
    """The filename, the record, and the rows must name one scene."""
    rows = population(scenes=1, pairs=4)  # rows for room_0
    eval_dir = tmp_path / "eval"
    write_scene(eval_dir, rows, run_meta("room_1"))
    with pytest.raises(ValueError, match="another scene's rows"):
        read_eval_dir(eval_dir)
