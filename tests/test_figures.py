"""PLAN Phase 3: the figure and table, built from the evaluation parquet alone."""

import math

import pytest

from lot.evaluate import (
    MEAN_FEATURE,
    NO_WARP_COPY,
    ORACLE_TRANSPORT,
    PER_POINT,
    SPLAT_POOL,
    write_rows,
)
from lot.figures import (
    aggregate,
    format_console_summary,
    margin_versus_parallax_figure,
    paired_records,
    read_eval_dir,
    summary_table,
    write_table,
)


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
        "parallax": 0.1,
        "parallax_bin": "0.05-0.1",
        "rotation_deg": 0.0,
        "covisible_fraction": 0.8,
        "encoder": "dinov2_vitb14",
        "path": PER_POINT,
        "variant": ORACLE_TRANSPORT,
        "n": 100,
        "cosine_mean": 0.9,
        "l2_mean": 0.4,
        "cosine_centered_mean": 0.8,
        "l2_centered_mean": 0.6,
        "coverage_mean": float("nan"),
    }
    row.update(overrides)
    return row


def one_comparison(cosines: dict[str, float], **overrides):
    return [make_row(variant=v, cosine_mean=c, **overrides) for v, c in cosines.items()]


def test_margins_are_paired_within_one_comparison():
    """Two pairs with different floors must not be averaged before subtracting."""
    rows = one_comparison(
        {ORACLE_TRANSPORT: 0.9, NO_WARP_COPY: 0.5}, context_frame_id="a"
    ) + one_comparison(
        {ORACLE_TRANSPORT: 0.6, NO_WARP_COPY: 0.4}, context_frame_id="b"
    )
    records = paired_records(rows)
    margins = {r["value"]: r["margin"] for r in records if r["variant"] == ORACLE_TRANSPORT}
    assert margins == {0.9: pytest.approx(0.4), 0.6: pytest.approx(0.2)}


def test_a_comparison_without_a_floor_is_dropped():
    """A margin needs both halves measured on the same pair."""
    rows = one_comparison({ORACLE_TRANSPORT: 0.9})
    assert paired_records(rows) == []


def test_unscorable_comparisons_are_skipped_not_counted_as_zero():
    """A pair with no co-visible region records nan and must not drag a mean down."""
    rows = one_comparison({ORACLE_TRANSPORT: 0.9, NO_WARP_COPY: 0.5}, context_frame_id="a")
    rows += one_comparison(
        {ORACLE_TRANSPORT: float("nan"), NO_WARP_COPY: float("nan")}, context_frame_id="b"
    )
    records = paired_records(rows)
    assert len(records) == 2
    assert all(math.isfinite(r["value"]) for r in records)


def test_a_floorless_variant_still_reports_when_the_floor_exists():
    rows = one_comparison(
        {ORACLE_TRANSPORT: 0.9, NO_WARP_COPY: 0.5, MEAN_FEATURE: float("nan")}
    )
    variants = {r["variant"] for r in paired_records(rows)}
    assert variants == {ORACLE_TRANSPORT, NO_WARP_COPY}


def test_aggregate_means_and_counts():
    records = [
        {"k": "a", "value": 1.0, "margin": 0.5},
        {"k": "a", "value": 3.0, "margin": 1.5},
        {"k": "b", "value": 2.0, "margin": 0.0},
    ]
    stats = aggregate(records, ("k",), ("value", "margin"))
    assert stats[("a",)] == {"value": 2.0, "margin": 1.0, "n_pairs": 2}
    assert stats[("b",)]["n_pairs"] == 1


def test_summary_table_carries_every_variant_and_its_margin():
    rows = one_comparison(
        {ORACLE_TRANSPORT: 0.9, NO_WARP_COPY: 0.5, MEAN_FEATURE: 0.2}
    )
    table = summary_table(paired_records(rows))
    assert len(table) == 1
    entry = table[0]
    assert entry["parallax_bin"] == "0.05-0.1"
    assert entry[f"value[{ORACLE_TRANSPORT}]"] == pytest.approx(0.9)
    assert entry[f"value[{NO_WARP_COPY}]"] == pytest.approx(0.5)
    assert entry[f"margin[{ORACLE_TRANSPORT}]"] == pytest.approx(0.4)
    assert entry[f"margin[{MEAN_FEATURE}]"] == pytest.approx(-0.3)
    # The floor has no margin over itself.
    assert f"margin[{NO_WARP_COPY}]" not in entry
    assert entry["metric"] == "cosine_mean"


def test_console_summary_mentions_both_paths_and_the_floor():
    rows = one_comparison({ORACLE_TRANSPORT: 0.9, NO_WARP_COPY: 0.5})
    rows += one_comparison(
        {ORACLE_TRANSPORT: 0.8, NO_WARP_COPY: 0.4}, path=SPLAT_POOL
    )
    text = format_console_summary(paired_records(rows))
    assert PER_POINT in text and SPLAT_POOL in text
    assert NO_WARP_COPY in text and ORACLE_TRANSPORT in text


def test_figure_and_table_regenerate_from_parquet_alone(tmp_path):
    """CLAUDE.md: the figure must come from outputs/eval/*.parquet and nothing else."""
    rows = []
    for index, (bin_label, cosine) in enumerate(
        (("zero", 0.98), ("0.05-0.1", 0.9), ("0.2-0.4", 0.7))
    ):
        for path in (PER_POINT, SPLAT_POOL):
            rows += one_comparison(
                {ORACLE_TRANSPORT: cosine, NO_WARP_COPY: 0.5, MEAN_FEATURE: 0.2},
                parallax_bin=bin_label,
                path=path,
                context_frame_id=f"c{index}",
            )
    eval_dir = tmp_path / "eval"
    write_rows(eval_dir / "room_0.parquet", rows)

    records = paired_records(read_eval_dir(eval_dir))
    figure = tmp_path / "figures" / "margin_versus_parallax.png"
    margin_versus_parallax_figure(records, figure)
    assert figure.is_file() and figure.stat().st_size > 0

    table = tmp_path / "tables" / "experiment_zero.parquet"
    write_table(table, summary_table(records))
    assert table.is_file()
    with pytest.raises(FileExistsError):
        write_table(table, summary_table(records))


def test_read_eval_dir_needs_something_to_read(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_eval_dir(tmp_path)
