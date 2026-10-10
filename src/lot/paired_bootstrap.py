"""Field-parameterized paired scene bootstrap, shared by Phase 5.

Phase 4's reporting layer carries this algorithm already, but with its field
list bound as a module constant, and Phase 4 is accepted and pinned so its code
is not edited to make room for a later phase. Copying the algorithm would leave
two implementations free to drift; parameterizing it here and proving the two
agree leaves one algorithm with two entry points, and the proof is a permanent
test that compares this module against lot.phase4_report on Phase 4's own field
list and asserts exact array equality.

What makes the bootstrap paired, which is the property every Phase 5 difference
depends on: one draw of scenes serves every method at once, because all fields
are resampled together from the same multiplicity vector, and the difference
between methods is recomputed inside each replicate from that replicate's own
means. Subtracting two independently drawn intervals would answer a different
question and would usually answer it wrongly, since the two methods' errors on a
scene are strongly correlated.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np


def unit_aggregates(
    records: Sequence[dict], fields: Sequence[str], unit: str = "scene"
) -> dict[Any, dict[str, tuple[float, int]]]:
    """Per resampling unit, the sum and count of finite values of every field.

    A resampled mean is sum-of-sums over sum-of-counts, so a replicate costs one
    pass over the units rather than one pass over every record. The value is
    identical to pooling records: the statistic is still recomputed whole inside
    each replicate from that replicate's own means.
    """
    out: dict[Any, dict[str, list]] = {}
    for record in records:
        slot = out.setdefault(record[unit], {field: [0.0, 0] for field in fields})
        for field in fields:
            value = record.get(field)
            if isinstance(value, (int, float)) and math.isfinite(value):
                slot[field][0] += float(value)
                slot[field][1] += 1
    return {
        unit_value: {field: (total, count) for field, (total, count) in per_field.items()}
        for unit_value, per_field in out.items()
    }


def pooled_means(
    aggregates: dict[Any, dict[str, tuple[float, int]]],
    units: Sequence[Any],
    fields: Sequence[str],
) -> dict[str, float]:
    """Field means over a unit list, which may repeat units as a resample does."""
    means: dict[str, float] = {}
    for field in fields:
        total, count = 0.0, 0
        for unit in units:
            unit_total, unit_count = aggregates[unit][field]
            total += unit_total
            count += unit_count
        means[field] = total / count if count else float("nan")
    return means


def aggregate_arrays(
    aggregates: dict[Any, dict[str, tuple[float, int]]],
    units: Sequence[Any],
    fields: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """One cell's per-unit sums and counts as [n_units, n_fields] arrays."""
    sums = np.zeros((len(units), len(fields)), dtype=np.float64)
    counts = np.zeros((len(units), len(fields)), dtype=np.float64)
    for row, unit in enumerate(units):
        per_field = aggregates[unit]
        for column, field in enumerate(fields):
            total, count = per_field[field]
            sums[row, column] = total
            counts[row, column] = count
    return sums, counts


def bootstrap_means(
    aggregates: dict[Any, dict[str, tuple[float, int]]],
    units: Sequence[Any],
    fields: Sequence[str],
    resamples: int,
    seed: int,
    chunk: int = 256,
) -> np.ndarray:
    """Every replicate's field means for one cell. [resamples, n_fields].

    The draw is successive integers(0, n, size=n) from a generator seeded once,
    and a batched call reproduces that stream bit for bit. A replicate's mean is
    sum-of-sums over sum-of-counts and a resample is a multiplicity vector over
    the units, so a block of replicates is two matrix products rather than a
    Python loop.
    """
    sums, counts = aggregate_arrays(aggregates, units, fields)
    n_units = len(units)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_units, size=(resamples, n_units))
    out = np.empty((resamples, len(fields)), dtype=np.float64)
    for start in range(0, resamples, chunk):
        block = picks[start : start + chunk]
        rows = block.shape[0]
        flat = block + (np.arange(rows) * n_units)[:, None]
        multiplicity = (
            np.bincount(flat.reshape(-1), minlength=rows * n_units)
            .reshape(rows, n_units)
            .astype(np.float64)
        )
        total = multiplicity @ sums
        weight = multiplicity @ counts
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start : start + rows] = np.where(weight > 0, total / weight, np.nan)
    return out


def paired_interval(
    records: Sequence[dict],
    fields: Sequence[str],
    statistic: Callable[[dict[str, float]], float],
    resamples: int,
    seed: int,
    confidence: float,
    unit: str = "scene",
) -> dict[str, Any]:
    """A statistic and its paired interval, recomputed whole in each replicate.

    statistic receives one replicate's field means and returns one number, so a
    difference of methods, a margin over a floor, and a ratio are all expressed
    the same way and all stay paired. The point estimate comes from the pooled
    means, never from the mean of the replicates.

    The returned n_replicates is the number of draws in which the statistic was
    defined. A cell whose statistic is undefined in some replicates yields a
    quantile over the draws in which it existed, and PROTOCOL 3.4 requires that
    count to travel with the interval, because a quantile over three draws and
    one over a thousand otherwise print identically.
    """
    aggregates = unit_aggregates(records, fields, unit)
    units = sorted(aggregates)
    if not units:
        return {
            "estimate": float("nan"), "lo": float("nan"), "hi": float("nan"),
            "n_units": 0, "n_replicates": 0,
        }
    estimate = statistic(pooled_means(aggregates, units, fields))
    replicate_means = bootstrap_means(aggregates, units, fields, resamples, seed)
    values = np.array(
        [statistic(dict(zip(fields, row))) for row in replicate_means], dtype=np.float64
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        lo = hi = float("nan")
    else:
        tail = (1.0 - confidence) / 2.0
        lo = float(np.quantile(finite, tail))
        hi = float(np.quantile(finite, 1.0 - tail))
    return {
        "estimate": float(estimate),
        "lo": lo,
        "hi": hi,
        "n_units": len(units),
        "n_replicates": int(finite.size),
    }
