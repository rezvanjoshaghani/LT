"""Figures and tables, built from outputs/eval/*.parquet and nothing else.

CLAUDE.md requires that every figure be regenerable from the evaluation
parquet alone, so this module never reads a render, a cache, or a config. It
takes a directory of per-scene parquet files and produces the paper figure,
the paper table, and a console summary compact enough to read at a glance.

Margins are computed pair by pair, not as a difference of two averages. Every
variant is scored on the same pair, so the paired difference is the quantity
with meaning, and it stays correct when one variant has no scorable region on
a pair and another does.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .datasets import parallax_bin_order
from .evaluate import (
    MEAN_FEATURE,
    NEIGHBOR_PATCH,
    NO_WARP_COPY,
    ORACLE_TRANSPORT,
    PER_POINT,
    RANDOM_PATCH,
    SPLAT_POOL,
)

PATH_ORDER = (PER_POINT, SPLAT_POOL)
VARIANT_ORDER = (
    ORACLE_TRANSPORT,
    NO_WARP_COPY,
    NEIGHBOR_PATCH,
    RANDOM_PATCH,
    MEAN_FEATURE,
)

# What identifies one measured comparison, up to which method was used.
PAIR_KEYS = ("scene", "context_frame_id", "target_frame_id", "encoder", "path")


def read_eval_dir(eval_dir: Path) -> list[dict[str, Any]]:
    """Read every per-scene parquet in a directory into one list of rows."""
    import pyarrow.parquet as pq

    eval_dir = Path(eval_dir)
    files = sorted(eval_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {eval_dir}")
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def group_by_pair(rows: Iterable[dict[str, Any]]) -> dict[tuple, dict[str, dict[str, Any]]]:
    """Index rows by the comparison they belong to, then by variant."""
    grouped: dict[tuple, dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in PAIR_KEYS)
        grouped.setdefault(key, {})[row["variant"]] = row
    return grouped


def paired_records(
    rows: Iterable[dict[str, Any]], metric: str = "cosine_mean"
) -> list[dict[str, Any]]:
    """One record per comparison and variant, carrying its margin over the floor.

    A comparison contributes only when both the variant and No-Warp-Copy scored
    something on it, so a margin is always a difference measured on one pair.
    """
    records: list[dict[str, Any]] = []
    for key, variants in group_by_pair(rows).items():
        floor = variants.get(NO_WARP_COPY)
        if floor is None or not _finite(floor[metric]):
            continue
        for variant, row in variants.items():
            if not _finite(row[metric]):
                continue
            records.append(
                {
                    "scene": row["scene"],
                    "split": row["split"],
                    "regime": row["regime"],
                    "parallax_bin": row["parallax_bin"],
                    "parallax": row["parallax"],
                    "encoder": row["encoder"],
                    "path": row["path"],
                    "variant": variant,
                    "value": row[metric],
                    "margin": row[metric] - floor[metric],
                    "n": row["n"],
                }
            )
    return records


def aggregate(
    records: Sequence[dict[str, Any]], keys: Sequence[str], fields: Sequence[str]
) -> dict[tuple, dict[str, Any]]:
    """Mean of each field over records sharing the given keys, plus a count."""
    sums: dict[tuple, dict[str, float]] = {}
    counts: dict[tuple, int] = {}
    for record in records:
        key = tuple(record[k] for k in keys)
        bucket = sums.setdefault(key, {field: 0.0 for field in fields})
        for field in fields:
            bucket[field] += float(record[field])
        counts[key] = counts.get(key, 0) + 1
    return {
        key: {**{f: bucket[f] / counts[key] for f in fields}, "n_pairs": counts[key]}
        for key, bucket in sums.items()
    }


def summary_table(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The paper table: every variant's score and margin, by encoder, path, and bin."""
    stats = aggregate(
        records, ("encoder", "path", "parallax_bin", "variant"), ("value", "margin")
    )
    encoders = sorted({r["encoder"] for r in records})
    table: list[dict[str, Any]] = []
    for encoder in encoders:
        for path in PATH_ORDER:
            for bin_label in parallax_bin_order():
                present = [
                    v for v in VARIANT_ORDER if (encoder, path, bin_label, v) in stats
                ]
                if not present:
                    continue
                row: dict[str, Any] = {
                    "encoder": encoder,
                    "path": path,
                    "parallax_bin": bin_label,
                    "n_pairs": stats[(encoder, path, bin_label, present[0])]["n_pairs"],
                }
                for variant in present:
                    entry = stats[(encoder, path, bin_label, variant)]
                    row[f"cosine[{variant}]"] = entry["value"]
                    if variant != NO_WARP_COPY:
                        row[f"margin[{variant}]"] = entry["margin"]
                table.append(row)
    return table


def write_table(path: Path, table: Sequence[dict[str, Any]]) -> None:
    """Write the summary table as parquet. Refuses to overwrite."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; delete it to regenerate.")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(table)), path)


def format_console_summary(records: Sequence[dict[str, Any]]) -> str:
    """A compact reading of the result, for the run log and for the writeup."""
    lines: list[str] = []
    order = [b for b in parallax_bin_order()]
    encoders = sorted({r["encoder"] for r in records})

    oracle = [r for r in records if r["variant"] == ORACLE_TRANSPORT]
    lines.append("Oracle-Transport, mean cosine and margin over No-Warp-Copy")
    lines.append(f"{'encoder':18s} {'path':11s} {'bin':>10s} {'pairs':>6s} "
                 f"{'cosine':>7s} {'floor':>7s} {'margin':>7s}")
    floors = aggregate(
        [r for r in records if r["variant"] == NO_WARP_COPY],
        ("encoder", "path", "parallax_bin"),
        ("value",),
    )
    stats = aggregate(oracle, ("encoder", "path", "parallax_bin"), ("value", "margin"))
    for encoder in encoders:
        for path in PATH_ORDER:
            for bin_label in order:
                key = (encoder, path, bin_label)
                if key not in stats:
                    continue
                entry = stats[key]
                lines.append(
                    f"{encoder:18s} {path:11s} {bin_label:>10s} {entry['n_pairs']:6d} "
                    f"{entry['value']:7.4f} {floors[key]['value']:7.4f} "
                    f"{entry['margin']:+7.4f}"
                )
    lines.append("")
    lines.append("By regime, pooled over parallax")
    lines.append(f"{'encoder':18s} {'path':11s} {'regime':12s} {'pairs':>6s} "
                 f"{'cosine':>7s} {'margin':>7s}")
    by_regime = aggregate(oracle, ("encoder", "path", "regime"), ("value", "margin"))
    for key in sorted(by_regime):
        entry = by_regime[key]
        lines.append(
            f"{key[0]:18s} {key[1]:11s} {key[2]:12s} {entry['n_pairs']:6d} "
            f"{entry['value']:7.4f} {entry['margin']:+7.4f}"
        )
    lines.append("")
    lines.append("Floors and nulls, pooled over everything")
    lines.append(f"{'encoder':18s} {'path':11s} {'variant':18s} {'cosine':>7s} {'margin':>7s}")
    everything = aggregate(records, ("encoder", "path", "variant"), ("value", "margin"))
    for encoder in encoders:
        for path in PATH_ORDER:
            for variant in VARIANT_ORDER:
                key = (encoder, path, variant)
                if key not in everything:
                    continue
                entry = everything[key]
                lines.append(
                    f"{encoder:18s} {path:11s} {variant:18s} "
                    f"{entry['value']:7.4f} {entry['margin']:+7.4f}"
                )
    return "\n".join(lines)


def margin_versus_parallax_figure(records: Sequence[dict[str, Any]], path: Path) -> None:
    """The acceptance figure: margin over No-Warp-Copy against parallax.

    One panel per path, one line per encoder. The floor is the zero line by
    construction, and Mean-Feature is drawn as a second reference so the
    distance between a real result and a location-free guess stays visible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    encoders = sorted({r["encoder"] for r in records})
    order = parallax_bin_order()
    figure, axes = plt.subplots(1, len(PATH_ORDER), figsize=(5.2 * len(PATH_ORDER), 4.2), squeeze=False)
    for column, evaluation_path in enumerate(PATH_ORDER):
        axis = axes[0][column]
        for variant, style in ((ORACLE_TRANSPORT, "-o"), (MEAN_FEATURE, "--s")):
            stats = aggregate(
                [r for r in records if r["variant"] == variant and r["path"] == evaluation_path],
                ("encoder", "parallax_bin"),
                ("margin",),
            )
            for encoder in encoders:
                present = [b for b in order if (encoder, b) in stats]
                if not present:
                    continue
                axis.plot(
                    range(len(present)),
                    [stats[(encoder, b)]["margin"] for b in present],
                    style,
                    label=f"{encoder} {variant}",
                    markersize=4,
                )
                axis.set_xticks(range(len(present)))
                axis.set_xticklabels(present, rotation=45, ha="right", fontsize=8)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(f"{evaluation_path}", fontsize=10)
        axis.set_xlabel("parallax bin (baseline / median depth)", fontsize=9)
        if column == 0:
            axis.set_ylabel("cosine margin over No-Warp-Copy", fontsize=9)
        axis.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=7, loc="best")
    figure.suptitle("Experiment Zero: how far a frozen feature transports", fontsize=11)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the Experiment Zero figure and table from eval parquet."
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where the figure and table go (default: the eval directory's parent)",
    )
    parser.add_argument("--metric", type=str, default="cosine_mean")
    args = parser.parse_args(argv)
    out_dir = args.out_dir or Path(args.eval_dir).parent
    rows = read_eval_dir(args.eval_dir)
    records = paired_records(rows, metric=args.metric)
    scored = len({tuple(r[k] for k in PAIR_KEYS) for r in rows})
    usable = len({(r["scene"], r["encoder"], r["path"], r["variant"]) for r in records})
    print(f"read {len(rows)} rows, {scored} comparisons, {len(records)} scored records")
    print()
    print(format_console_summary(records))
    figure_path = Path(out_dir) / "figures" / "margin_versus_parallax.png"
    table_path = Path(out_dir) / "tables" / "experiment_zero.parquet"
    print()
    # The table is the artifact the numbers live in, so it is written before the
    # figure. A missing plotting library must not cost the run its results.
    write_table(table_path, summary_table(records))
    print(f"table  -> {table_path}")
    try:
        margin_versus_parallax_figure(records, figure_path)
    except ImportError as error:
        print(f"figure skipped: {error}. Install matplotlib and rerun; the table is written.")
    else:
        print(f"figure -> {figure_path}")
    assert usable  # every scene contributed something


if __name__ == "__main__":
    main()
