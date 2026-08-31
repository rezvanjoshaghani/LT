"""The shared bootstrap is the same algorithm Phase 4 accepted, proven not assumed."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lot import paired_bootstrap as pb
from lot import phase4_report as p4


def _records(n_scenes: int = 6, per_scene: int = 7, seed: int = 11) -> list[dict]:
    rng = np.random.default_rng(seed)
    records = []
    for s in range(n_scenes):
        for _ in range(per_scene):
            record = {"scene": f"scene_{s}", "n": int(rng.integers(1, 50))}
            for field in p4.FIELDS:
                record[field] = float(rng.normal())
            records.append(record)
    # A nonfinite value must be skipped by both implementations identically.
    records[3][p4.FIELDS[0]] = float("nan")
    records[9][p4.FIELDS[2]] = float("inf")
    return records


def test_unit_aggregates_match_phase4_exactly():
    records = _records()
    mine = pb.unit_aggregates(records, p4.FIELDS, "scene")
    theirs = p4.scene_aggregates(records)
    assert set(mine) == set(theirs)
    for scene in theirs:
        for field in p4.FIELDS:
            assert mine[scene][field] == theirs[scene][field], (scene, field)


def test_pooled_means_match_phase4_exactly():
    records = _records()
    mine_agg = pb.unit_aggregates(records, p4.FIELDS, "scene")
    theirs_agg = p4.scene_aggregates(records)
    units = sorted(theirs_agg)
    mine = pb.pooled_means(mine_agg, units, p4.FIELDS)
    theirs = p4.pooled_means(theirs_agg, units)
    for field in p4.FIELDS:
        a, b = mine[field], theirs[field]
        assert (math.isnan(a) and math.isnan(b)) or a == b, field


def test_bootstrap_means_reproduce_phase4_bit_for_bit():
    """One algorithm, two entry points. Exact equality, not a tolerance."""
    records = _records()
    theirs_agg = p4.scene_aggregates(records)
    units = sorted(theirs_agg)
    mine = pb.bootstrap_means(
        pb.unit_aggregates(records, p4.FIELDS, "scene"), units, p4.FIELDS,
        resamples=200, seed=20260825,
    )
    theirs = p4.bootstrap_means(theirs_agg, units, resamples=200, seed=20260825)
    assert mine.shape == theirs.shape
    np.testing.assert_array_equal(np.isnan(mine), np.isnan(theirs))
    finite = ~np.isnan(mine)
    np.testing.assert_array_equal(mine[finite], theirs[finite])


def test_bootstrap_respects_the_chunk_boundary():
    records = _records()
    agg = pb.unit_aggregates(records, p4.FIELDS, "scene")
    units = sorted(agg)
    a = pb.bootstrap_means(agg, units, p4.FIELDS, resamples=300, seed=5, chunk=256)
    b = pb.bootstrap_means(agg, units, p4.FIELDS, resamples=300, seed=5, chunk=64)
    np.testing.assert_array_equal(a, b)


def test_paired_interval_recomputes_the_difference_inside_each_replicate():
    """The property the Phase 5 headline rests on.

    A paired interval on a difference is not the difference of two independent
    intervals. Constructed so the two methods are strongly correlated across
    scenes and the difference is nearly constant: the paired interval is then
    tight even though each method's own interval is wide.
    """
    rng = np.random.default_rng(2)
    records = []
    for s in range(12):
        level = rng.normal(0.0, 1.0)   # a big scene effect, shared by both methods
        for _ in range(5):
            records.append({
                "scene": f"scene_{s}",
                "a": level + 0.10 + rng.normal(0.0, 1e-3),
                "b": level + rng.normal(0.0, 1e-3),
            })
    fields = ("a", "b")
    paired = pb.paired_interval(
        records, fields, lambda m: m["a"] - m["b"],
        resamples=500, seed=3, confidence=0.95,
    )
    assert paired["estimate"] == pytest.approx(0.10, abs=2e-3)
    assert paired["hi"] - paired["lo"] < 0.01, "the pairing did not cancel the scene effect"
    assert paired["n_units"] == 12
    assert paired["n_replicates"] == 500

    # Each method alone is wide, which is what makes the pairing load bearing.
    alone = pb.paired_interval(
        records, fields, lambda m: m["a"], resamples=500, seed=3, confidence=0.95
    )
    assert alone["hi"] - alone["lo"] > 0.2


def test_paired_interval_reports_an_empty_cell_rather_than_inventing_one():
    out = pb.paired_interval([], ("a",), lambda m: m["a"], 10, 1, 0.95)
    assert out["n_units"] == 0
    assert math.isnan(out["estimate"])
    assert math.isnan(out["lo"]) and math.isnan(out["hi"])


def test_paired_interval_counts_only_the_replicates_the_statistic_existed_in():
    """PROTOCOL 3.4: a quantile over three draws must not print like one over a
    thousand, so the count travels with the interval."""
    records = [{"scene": "a", "x": 1.0}, {"scene": "b", "x": 2.0}]

    def sometimes_undefined(means: dict[str, float]) -> float:
        return float("nan") if means["x"] > 1.6 else means["x"]

    out = pb.paired_interval(records, ("x",), sometimes_undefined, 200, 7, 0.95)
    assert 0 < out["n_replicates"] < 200


def test_missing_field_is_treated_as_absent_not_as_zero():
    records = [{"scene": "a", "x": 2.0}, {"scene": "a"}, {"scene": "b", "x": 4.0}]
    agg = pb.unit_aggregates(records, ("x",), "scene")
    assert agg["a"]["x"] == (2.0, 1)
    means = pb.pooled_means(agg, ["a", "b"], ("x",))
    assert means["x"] == pytest.approx(3.0)
