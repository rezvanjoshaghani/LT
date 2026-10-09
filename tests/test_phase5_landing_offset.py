"""The landing-offset diagnostic, pre-registered 2026-10-09.

validation/evidence/phase5/landing_offset_diagnostic.md defines it. Context-Lift
with ground-truth context depth is scored against the frozen per-point target,
grouped by how far each landing falls from the nearest target patch center.
Geometry and the change of representation between views have no reason to
depend on that distance. The landing read does. So the decline across the
groups sizes the read on real data, without any model and without touching any
estimand.

The geometry here is analytic: offsets on a hand-placed grid, and scores on two
orthogonal patches whose expected cosines follow by hand.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from lot.analysis_config import load_analysis_config
from lot.context_lift import ContextLiftMap
from lot.phase5_estimands import (
    N_OFFSET_BINS,
    OFFSET_BINS,
    landing_offset_record_fields,
)
from lot.phase5_score import (
    empty_landing_offset,
    landing_offsets,
    offset_bins,
    score_landing_offset,
)

EDGES = (0.1, 0.2, 0.3, 0.4, 0.5)
ANALYSIS = load_analysis_config()


def _pixel(patch_x: float, patch_y: float) -> list[float]:
    """Pixel coordinates of a patch-grid location, under the frozen mapping.

    Pixel u sits at patch coordinate (u + 0.5) / 14 - 0.5, so patch coordinate
    p sits at pixel 14 p + 6.5.
    """
    return [14 * patch_x + 6.5, 14 * patch_y + 6.5]


# ---------------------------------------------------------------------------
# The offset and its bins
# ---------------------------------------------------------------------------

def test_a_landing_offset_is_the_distance_to_the_nearest_patch_center():
    uv = torch.tensor([
        _pixel(0, 0),          # a patch center
        _pixel(0.5, 0),        # midway between two centers
        _pixel(0.5, 0.5),      # a cell corner, the farthest point
        _pixel(0.3, 1),        # 0.3 of a patch from center (0, 1)
        _pixel(0.8, 0),        # 0.2 short of center (1, 0)
        _pixel(2.9, 3.1),      # 0.1 from (3, 3) along each axis
    ], dtype=torch.float64)
    want = [0.0, 0.5, math.sqrt(0.5), 0.3, 0.2, math.hypot(0.1, 0.1)]
    assert landing_offsets(uv).tolist() == pytest.approx(want, abs=1e-12)


def test_a_float32_landing_gets_its_offset_in_float64():
    """Offsets are computed in float64 from the run's float32 landings, so a
    bin assignment does not depend on float32 rounding of the subtraction."""
    uv = torch.tensor([_pixel(0.25, 0)], dtype=torch.float32)
    offsets = landing_offsets(uv)
    assert offsets.dtype == torch.float64
    assert offsets.item() == pytest.approx(0.25, abs=1e-6)


def test_offset_bins_are_closed_on_the_right():
    """PROTOCOL 3.4: a value equal to an edge belongs to the lower bin."""
    offsets = np.array([0.0, 0.1, 0.1 + 1e-9, 0.3, 0.5, 0.5 + 1e-9, math.sqrt(0.5)])
    assert offset_bins(offsets, EDGES).tolist() == [0, 0, 1, 2, 4, 5, 5]


def test_the_registry_has_one_bin_per_edge_plus_the_open_one():
    assert N_OFFSET_BINS == len(EDGES) + 1
    assert OFFSET_BINS == tuple(f"b{k}" for k in range(N_OFFSET_BINS))


# ---------------------------------------------------------------------------
# Scoring by offset
# ---------------------------------------------------------------------------

A = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
B = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)


def _two_patch_case():
    """Context patch (0, 0) carries A. Target patch (0, 0) is A, (1, 0) is B.

    Sample one lands on target patch (0, 0) exactly, so the read is A.
    Sample two lands midway between the two target patches, so the read is
    (A + B) / 2. The context grid holds A at (0, 0) and B at (1, 0) as well, so
    No-Warp-Copy reads (A + B) / 2 at the second landing and matches its target.
    """
    grid = torch.zeros(4, 2, 2, dtype=torch.float64)
    grid[:, :, 0] = A[:, None]
    grid[:, :, 1] = B[:, None]
    features_context = grid.clone()
    features_target = grid.clone()
    lift = ContextLiftMap(
        uv_context=torch.tensor([_pixel(0, 0), _pixel(0, 0)], dtype=torch.float64),
        uv_target=torch.tensor([_pixel(0, 0), _pixel(0.5, 0)], dtype=torch.float64),
        z_target=torch.ones(2, dtype=torch.float64),
        depth_context=torch.ones(2, dtype=torch.float64),
        depth_valid=torch.ones(2, dtype=torch.bool),
        landed=torch.ones(2, dtype=torch.bool),
    )
    support = torch.ones(2, dtype=torch.bool)
    center = torch.zeros(4, dtype=torch.float64)
    return lift, support, features_context, features_target, center


def test_each_bin_is_scored_against_the_landing_read():
    lift, support, fc, ft, center = _two_patch_case()
    fields = score_landing_offset(lift, support, fc, ft, center, EDGES).as_fields()
    half = 1.0 / math.sqrt(2.0)

    # On the patch center the carried vector is the read: one.
    assert fields["offset_n_b0"] == 1
    assert fields["offset_cl_oracle_centered_b0"] == pytest.approx(1.0)
    assert fields["offset_cl_oracle_l2_centered_b0"] == pytest.approx(0.0, abs=1e-12)
    # Midway, the same exact vector meets a blend of two patches. Offset 0.5
    # falls in the bin closed at 0.5.
    assert fields["offset_n_b4"] == 1
    assert fields["offset_cl_oracle_centered_b4"] == pytest.approx(half)
    assert fields["offset_cl_oracle_l2_centered_b4"] == pytest.approx(
        math.sqrt(2.0 - 2.0 * half)
    )
    # No-Warp-Copy reads the context grid at the landing, a blend that matches.
    assert fields["offset_nowarp_centered_b4"] == pytest.approx(1.0)
    # The whole diagnostic support pools both samples.
    assert fields["offset_n_all"] == 2
    assert fields["offset_cl_oracle_centered_all"] == pytest.approx((1.0 + half) / 2.0)
    # Empty bins report a zero count and no score.
    for label in ("b1", "b2", "b3", "b5"):
        assert fields[f"offset_n_{label}"] == 0
        assert math.isnan(fields[f"offset_cl_oracle_centered_{label}"])


def test_centering_is_applied_before_each_bins_cosine():
    """PROTOCOL 3.7: the mean is subtracted from both compared vectors."""
    lift, support, fc, ft, _ = _two_patch_case()
    center = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    fields = score_landing_offset(lift, support, fc + center[:, None, None],
                                  ft + center[:, None, None], center, EDGES).as_fields()
    assert fields["offset_cl_oracle_centered_b4"] == pytest.approx(1.0 / math.sqrt(2.0))
    assert fields["offset_cl_oracle_raw_b4"] != pytest.approx(1.0 / math.sqrt(2.0))


def test_only_the_support_is_scored():
    lift, support, fc, ft, center = _two_patch_case()
    support = torch.tensor([True, False])
    fields = score_landing_offset(lift, support, fc, ft, center, EDGES).as_fields()
    assert fields["offset_n_all"] == 1
    assert fields["offset_n_b4"] == 0


def test_an_empty_support_gives_the_complete_empty_record():
    lift, _, fc, ft, center = _two_patch_case()
    empty = score_landing_offset(lift, torch.zeros(2, dtype=torch.bool), fc, ft,
                                 center, EDGES).as_fields()
    assert empty.keys() == empty_landing_offset().as_fields().keys()
    assert all(empty[f"offset_n_{label}"] == 0 for label in OFFSET_BINS + ("all",))
    assert all(math.isnan(v) for k, v in empty.items() if not k.startswith("offset_n_"))


def test_the_record_carries_exactly_the_registered_fields():
    lift, support, fc, ft, center = _two_patch_case()
    fields = score_landing_offset(lift, support, fc, ft, center, EDGES).as_fields()
    assert tuple(fields) == landing_offset_record_fields()


def test_edges_that_disagree_with_the_registry_are_refused():
    """The estimand layer reads a fixed set of bins, so a record with more or
    fewer would be silently half-read. Scoring refuses instead."""
    lift, support, fc, ft, center = _two_patch_case()
    with pytest.raises(ValueError, match="bins"):
        score_landing_offset(lift, support, fc, ft, center, EDGES[:-1])


# ---------------------------------------------------------------------------
# The reported cells
# ---------------------------------------------------------------------------

COUNTS = {"b0": 3, "b1": 9, "b2": 15, "b3": 21, "b4": 27, "b5": 25}
DROP = {"b0": 0.0, "b1": 0.02, "b2": 0.04, "b3": 0.06, "b4": 0.08, "b5": 0.10}


def _offset_records(n_scenes=8, pairs_per_scene=6, near=0.80, scene_spread=0.10,
                    noise=1e-4, seed=3):
    """Pairs whose oracle score falls with offset, under a large scene effect.

    Within a pair the score drops by DROP from the near-grid bin outward. The
    scene effect moves every bin of a pair together, as a real scene would.
    """
    from lot.phase5_estimands import METRIC_COLUMNS

    rng = np.random.default_rng(seed)
    records = []
    for s in range(n_scenes):
        level = rng.normal(0.0, scene_spread)
        for p in range(pairs_per_scene):
            record = {"scene": f"scene_{s}", "pair": f"{s}_{p}", "regime": "rotation"}
            total = sum(COUNTS.values())
            pooled = {"cl_oracle": 0.0, "nowarp": 0.0}
            for label, n in COUNTS.items():
                values = {
                    "cl_oracle": near - DROP[label] + level + rng.normal(0, noise),
                    "nowarp": 0.5 + level + rng.normal(0, noise),
                }
                record[f"offset_n_{label}"] = n
                for arm, value in values.items():
                    pooled[arm] += value * n / total
                    for column in METRIC_COLUMNS:
                        record[f"offset_{arm}_{column}_{label}"] = value
                for column in ("raw", "l2_raw"):
                    record[f"offset_meanfeat_{column}_{label}"] = 0.3 + level
            record["offset_n_all"] = total
            for arm, value in pooled.items():
                for column in METRIC_COLUMNS:
                    record[f"offset_{arm}_{column}_all"] = value
            for column in ("raw", "l2_raw"):
                record[f"offset_meanfeat_{column}_all"] = 0.3 + level
            records.append(record)
    return records


def test_the_read_deficit_is_paired_within_each_pair():
    """Near-grid minus whole support, recomputed whole in each replicate.

    The scene effect is ten times the deficit and cancels inside each pair. A
    paired interval sees only the deficit; subtracting two cell intervals would
    be swamped by the scene effect.
    """
    from lot.phase5_estimands import evaluate_quantity

    records = _offset_records()
    total = sum(COUNTS.values())
    expected = sum(DROP[label] * n for label, n in COUNTS.items()) / total
    deficit = evaluate_quantity(records, "read_deficit", "centered", ANALYSIS)
    assert deficit.estimate == pytest.approx(expected, abs=2e-4)
    assert deficit.hi - deficit.lo < 2e-3
    level = evaluate_quantity(records, "cl_oracle_offset_b0", "centered", ANALYSIS)
    assert level.hi - level.lo > 0.05


def test_a_bin_cell_uses_only_pairs_with_samples_in_that_bin():
    from lot.phase5_estimands import evaluate_quantity

    records = _offset_records(n_scenes=6, pairs_per_scene=5)
    for record in records[:12]:
        record["offset_n_b0"] = 0
        for key in list(record):
            if key.startswith("offset_") and key.endswith("_b0") and key != "offset_n_b0":
                record[key] = float("nan")
    near = evaluate_quantity(records, "cl_oracle_offset_b0", "centered", ANALYSIS)
    far = evaluate_quantity(records, "cl_oracle_offset_b5", "centered", ANALYSIS)
    deficit = evaluate_quantity(records, "read_deficit", "centered", ANALYSIS)
    assert near.n_camera_pairs == 18
    assert far.n_camera_pairs == 30
    # The deficit is defined on the pairs that have a near-grid sample, and its
    # support counts the near-grid samples it rests on.
    assert deficit.n_camera_pairs == 18
    assert deficit.n_feature_comparisons == 18 * COUNTS["b0"]
    assert math.isfinite(deficit.estimate)


def test_the_margin_in_each_bin_is_over_no_warp_copy_in_that_bin():
    from lot.phase5_estimands import quantity_formulas

    forms = quantity_formulas("centered")
    means = {"offset_cl_oracle_centered_b3": 0.71, "offset_nowarp_centered_b3": 0.52}
    assert forms["cl_oracle_offset_margin_b3"](means) == pytest.approx(0.19)


def test_offset_cells_are_diagnostics_and_never_get_near_zero_wording():
    """No quantity here claims one method over another, so PROTOCOL 3.9's
    near-zero wording cannot apply, however small the value."""
    from lot.phase5_estimands import (
        INTERPRETED_EFFECTS, QUANTITY_POPULATION, evaluate_quantity, quantity_formulas,
    )

    offset_quantities = [q for q in quantity_formulas("centered")
                         if q == "read_deficit" or "_offset_" in q]
    assert "read_deficit" in offset_quantities
    assert len(offset_quantities) == 1 + 3 * (N_OFFSET_BINS + 1)
    for quantity in offset_quantities:
        assert quantity not in INTERPRETED_EFFECTS, quantity
        assert QUANTITY_POPULATION[quantity].startswith("landing_offset"), quantity

    flat = _offset_records()
    for record in flat:
        for column in ("raw", "centered", "l2_raw", "l2_centered"):
            record[f"offset_cl_oracle_{column}_b0"] = record[f"offset_cl_oracle_{column}_all"]
    deficit = evaluate_quantity(flat, "read_deficit", "centered", ANALYSIS)
    assert abs(deficit.estimate) < 1e-12
    assert not deficit.disclosure.get("near_zero", False)
    assert deficit.as_row()["near_zero_wording"] == ""


# ---------------------------------------------------------------------------
# The frozen configuration
# ---------------------------------------------------------------------------

def test_the_shipped_config_declares_the_registered_bins():
    from pathlib import Path

    from lot.phase5 import landing_offset_edges, load_phase5_config

    cfg = load_phase5_config(Path("configs/phase5.yaml"))
    assert landing_offset_edges(cfg) == EDGES
    assert cfg.landing_offset["depth"] == "ground_truth"


@pytest.mark.parametrize("spec, match", [
    ({"depth": "aligned", "upper_edges_patch": list(EDGES)}, "ground_truth"),
    ({"depth": "ground_truth", "upper_edges_patch": [0.1, 0.3, 0.2, 0.4, 0.5]}, "increasing"),
    ({"depth": "ground_truth", "upper_edges_patch": [0.1, 0.2, 0.3, 0.4]}, "bins"),
    ({"depth": "ground_truth", "upper_edges_patch": [0.1, 0.2, 0.3, 0.4, 0.75]}, "corner"),
    ({"depth": "ground_truth", "upper_edges_patch": [0.0, 0.2, 0.3, 0.4, 0.5]}, "increasing"),
    ({"depth": "ground_truth", "upper_edges_patch": list(EDGES), "extra": 1}, "unknown"),
])
def test_a_malformed_offset_spec_is_refused(spec, match):
    import dataclasses

    from lot.phase5 import Phase5Config, landing_offset_edges

    cfg = dataclasses.replace(Phase5Config(), landing_offset=spec)
    with pytest.raises(ValueError, match=match):
        landing_offset_edges(cfg)


def test_the_mean_feature_floor_is_scored_under_raw_cosine_only():
    """PROTOCOL 3.7: Mean-Feature predicts the centering vector, so it has raw
    columns and no centered ones. With the mean along a third axis, its raw
    cosine against A + mean is 1/sqrt(2), and against (A + B)/2 + mean it is
    1/sqrt(1.5)."""
    lift, support, fc, ft, _ = _two_patch_case()
    center = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    fields = score_landing_offset(lift, support, fc + center[:, None, None],
                                  ft + center[:, None, None], center, EDGES).as_fields()
    assert fields["offset_meanfeat_raw_b0"] == pytest.approx(1.0 / math.sqrt(2.0))
    assert fields["offset_meanfeat_raw_b4"] == pytest.approx(1.0 / math.sqrt(1.5))
    assert "offset_meanfeat_centered_b0" not in fields
    assert "offset_meanfeat_l2_centered_b0" not in fields
    assert math.isnan(fields["offset_meanfeat_raw_b2"])


def test_the_mean_feature_cell_exists_under_raw_only():
    from lot.phase5_estimands import quantity_formulas

    assert "meanfeat_offset_b0" in quantity_formulas("raw")
    assert "meanfeat_offset_b0" not in quantity_formulas("centered")
