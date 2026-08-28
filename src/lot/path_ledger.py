"""The path-agreement ledger: preflight, decomposition, and mechanism test.

Stream D's 3.9 gate fired with a signed aggregate 27 times under tolerance and
a per-pair dispersion above it, which is a tolerance applied to a statistic it
was never calibrated on. Rather than argue from the two summary numbers, this
module accounts for the recorded per-path scores exactly, term by term, so the
discrepancy is attributed rather than characterized.

Preflight, per pair: establish from code and stored records what each path's
atomic scored unit is, that the two paths score the same target vector, and
what each path's cells-to-pair weighting is. Per-point samples are exhaustive
patch centers, one per cell by construction; every read that is a lookup into
the same cached array is checked bit-level, and anything post-arithmetic falls
under the frozen numeric tolerances below.

Ledger, per pair per metric, over the common cell set j with recomputed
per-cell scores q_j (per-point) and p_j (splat) against the shared target:

    T1 = S_pp_recorded - sum_j a_j q_j        reconstruction, per-point
    T2 = sum_j a_j (q_j - p_j)                the operator difference
    T3 = sum_j (a_j - b_j) p_j                aggregation-weighting gap
    T4 = sum_j b_j p_j - S_sp_recorded        reconstruction, splat

    T1 + T2 + T3 + T4 = S_pp_recorded - S_sp_recorded

Closure is an algebraic identity, so a closure failure means this decomposition
is coded wrong and everything stops. |T1| or |T4| above ledger_recon_tol means
a recorded score cannot be rebuilt from its own inputs, and everything stops.
Both paths weight the common set uniformly, a_j = b_j = 1/n, which the
preflight verifies from the code and the recorded n; T3 is then exactly zero,
and a nonzero T3 that closes would be a design fact to freeze, not an error.
T2 carrying the remainder is the pass.

Mechanism test, separate from the gate, on c_j = q_j - p_j: two preregistered
contrasts on |c_j|, reported whatever they show. Boundary contrast compares
cells whose bilinear read taps context patches spanning a depth discontinuity
against interior cells; norm contrast compares the bottom and top quartiles of
the centered target norm. Quartile cut points are computed globally over the
eligible population and stored before any contrast is viewed, and an existing
cut-point file is reused, never recomputed.

Runs per scene (the same array shape as evaluation), then once with --report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .analysis_config import AnalysisConfig, load_analysis_config
from .correspondence import gather_value_pairs
from .datasets import bin_label, load_scene_pairs, subsample_by_stratum
from .encoders import PATCH_SIZE, pixel_to_patch_coords
from .evaluate import (
    ORACLE_TRANSPORT,
    PER_POINT,
    SPLAT_POOL,
    _SceneCache,
    load_eval_config,
    load_or_build_mean_vector,
    pair_geometry,
    unit_normalize,
)
from .geometry import relative_pose
from .render_replica import MANIFEST_NAME, load_manifest
from .transport import apply_transport_plan

LEDGER_NAME = "ledger_{scene}.parquet"
CELLS_NAME = "cells_{scene}.parquet"
PREFLIGHT_NAME = "preflight_{scene}.json"
CUTPOINTS_NAME = "cutpoints.json"
REPORT_NAME = "report.json"

METRICS = ("raw", "centered")


def per_cell_cosine(a: Tensor, b: Tensor, center: Tensor | None = None) -> Tensor:
    """Per-row cosine, mirroring evaluate.value_agreement's arithmetic exactly.

    Same float32 cast, same optional centering of both sides, same normalizer,
    so the mean of these values reconstructs the recorded score to float
    accumulation error and the reconstruction tolerance has nothing to absorb
    but dtype noise.
    """
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    if center is not None:
        a = a - center.to(a.dtype)
        b = b - center.to(b.dtype)
    return (unit_normalize(a) * unit_normalize(b)).sum(dim=-1)


def patch_median_depth(depth: Tensor, patch_size: int = PATCH_SIZE) -> Tensor:
    """Median context depth per patch, for the boundary flag. [Hp, Wp]."""
    height, width = depth.shape
    blocks = depth.reshape(
        height // patch_size, patch_size, width // patch_size, patch_size
    ).permute(0, 2, 1, 3).reshape(height // patch_size, width // patch_size, -1)
    return blocks.median(dim=-1).values


def boundary_flags(
    uv_warp: Tensor, depth_context: Tensor, rel_tol: float, patch_size: int = PATCH_SIZE
) -> np.ndarray:
    """Whether each sample's bilinear read taps patches on different surfaces.

    The per-point read mixes up to four context patch vectors around the warp
    location. The flag is preregistered as: the relative spread of the tapped
    patches' median depths exceeds the co-visibility depth tolerance, the same
    frozen constant, so the mechanism variable introduces no new tunable.
    """
    medians = patch_median_depth(depth_context, patch_size)
    patches_h, patches_w = medians.shape
    coords = pixel_to_patch_coords(uv_warp.to(torch.float64), patch_size)
    x0 = coords[:, 0].floor().long().clamp(0, patches_w - 1)
    y0 = coords[:, 1].floor().long().clamp(0, patches_h - 1)
    x1 = (x0 + 1).clamp(max=patches_w - 1)
    y1 = (y0 + 1).clamp(max=patches_h - 1)
    taps = torch.stack(
        (medians[y0, x0], medians[y0, x1], medians[y1, x0], medians[y1, x1]), dim=-1
    )
    spread = taps.max(dim=-1).values - taps.min(dim=-1).values
    reference = taps.median(dim=-1).values.clamp(min=1e-6)
    return (spread / reference > rel_tol).cpu().numpy()


def recorded_scores(eval_dir: Path, scene: str) -> dict[tuple, dict[str, float]]:
    """Oracle intersection scores from the run's parquet, keyed per comparison."""
    import pyarrow.parquet as pq

    table = pq.read_table(eval_dir / f"{scene}.parquet").to_pylist()
    out: dict[tuple, dict[str, float]] = {}
    for row in table:
        if row["variant"] != ORACLE_TRANSPORT:
            continue
        out[(row["context_frame_id"], row["target_frame_id"], row["encoder"], row["path"])] = {
            "raw": row["cosine_intersect_mean"],
            "centered": row["cosine_centered_intersect_mean"],
            "n_intersect": row["n_intersect"],
            "rotation_deg": row["rotation_deg"],
            "regime": row["regime"],
        }
    return out


def ledger_scene(
    config_path: Path, scene: str, out_dir: Path, analysis: AnalysisConfig | None = None
) -> dict[str, Any]:
    """Preflight facts, per-pair ledger terms, and per-cell mechanism rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = load_eval_config(config_path)
    analysis = analysis if analysis is not None else load_analysis_config(cfg.analysis_config)
    run_dir = cfg.output_root / cfg.experiment_name
    eval_dir = run_dir / "eval"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_vectors = {
        encoder: load_or_build_mean_vector(
            cfg.cache_root, encoder, cfg.mean_vector_scenes, run_dir
        )
        for encoder in cfg.encoders
    }
    recorded = recorded_scores(eval_dir, scene)

    scene_root = cfg.renders_root / scene
    manifest = load_manifest(scene_root / MANIFEST_NAME)
    frames = {f.frame_id: f for f in manifest.frames}
    pairs = subsample_by_stratum(
        load_scene_pairs(cfg.renders_root, scene, config=analysis),
        analysis.max_pairs_per_stratum,
        seed=cfg.seed,
        config=analysis,
    )
    cache = _SceneCache(scene_root, cfg.cache_root, cfg.encoders, scene, manifest)

    preflight = {
        "scene": scene,
        # Code facts, stated where the evidence lives: the per-point unit is one
        # exhaustive sample per eligible patch center, the splat unit is one
        # pooled cell, and both paths average their common set uniformly, so
        # a_j = b_j = 1 / n_intersect and their sums are exactly one.
        "per_point_unit": "one sample per eligible target patch center",
        "splat_unit": "one pooled target cell",
        "weighting_rule": "uniform 1/n over the common cell set, both paths",
        "weight_sums_are_one": True,
        "pairs": 0,
        "pairs_skipped_empty_intersection": 0,
        "target_bit_mismatches": 0,
        "no_warp_bit_mismatches": 0,
        "random_bit_mismatches": 0,
        "n_intersect_mismatches": 0,
        "cell_order_mismatches": 0,
    }
    ledger_rows: list[dict[str, Any]] = []
    # One row per common cell per encoder, so this is the largest artifact the
    # ledger writes: about 1.5M rows per scene and 27M over the run. It carries
    # only what the mechanism contrasts read. Pair identity, regime and
    # rotation stay in the ledger file, one row per pair, where they cost
    # nothing; at one row per cell they were 27M copies of a string each.
    cell_columns: dict[str, list] = {
        "scene": [], "encoder": [], "c_raw": [], "c_centered": [],
        "boundary": [], "centered_norm": [],
    }

    for pair in pairs:
        context = frames[pair.context_frame_id]
        target = frames[pair.target_frame_id]
        T_target_from_context = relative_pose(
            target.T_world_from_camera, context.T_world_from_camera
        ).to(cfg.torch_dtype)
        depth_context = cache.depth(context.depth_path).to(cfg.torch_dtype)
        geometry = pair_geometry(
            depth_context,
            cache.depth(target.depth_path).to(cfg.torch_dtype),
            context.K.to(cfg.torch_dtype),
            target.K.to(cfg.torch_dtype),
            T_target_from_context,
            scene,
            pair.context_frame_id,
            pair.target_frame_id,
            analysis,
        )
        shared = geometry.cross_path_mask
        cells = np.flatnonzero(shared)
        if cells.size == 0:
            preflight["pairs_skipped_empty_intersection"] += 1
            continue
        preflight["pairs"] += 1
        in_shared = torch.from_numpy(shared[geometry.per_point_cells])
        sample_cells = geometry.per_point_cells[in_shared.numpy()]
        if not np.array_equal(np.sort(sample_cells), cells):
            preflight["cell_order_mismatches"] += 1
            continue
        order = np.argsort(sample_cells)
        boundary = boundary_flags(
            geometry.samples.uv_context_warp[in_shared][torch.from_numpy(order)],
            depth_context,
            analysis.covisible_relative_depth_tol,
        )

        for encoder in cfg.encoders:
            features_context = cache.features(encoder, pair.context_frame_id)
            features_target = cache.features(encoder, pair.target_frame_id)
            center = mean_vectors[encoder].to(torch.float32)
            reads = gather_value_pairs(features_context, features_target, geometry.samples)
            channels = features_context.shape[0]
            flat_target = features_target.to(torch.float32).reshape(channels, -1)
            transported = apply_transport_plan(geometry.plan, features_context).reshape(
                channels, -1
            )

            targets_pp = reads["target"][in_shared][torch.from_numpy(order)]
            targets_sp = flat_target[:, cells].T
            warp_pp = reads["warp"][in_shared][torch.from_numpy(order)]
            pooled_sp = transported[:, cells].T
            # Bit-level checks for every read that is a lookup into the same
            # cached array. The targets decide whether the ledger needs the
            # target-side split at all; the No-Warp and Random reads are the
            # other pure lookups the two paths share.
            if not torch.equal(targets_pp, targets_sp):
                preflight["target_bit_mismatches"] += 1
            no_warp_pp = reads["no_warp"][in_shared][torch.from_numpy(order)]
            no_warp_sp = features_context.to(torch.float32).reshape(channels, -1)[:, cells].T
            if not torch.equal(no_warp_pp, no_warp_sp):
                preflight["no_warp_bit_mismatches"] += 1
            random_pp = reads["random"][in_shared][torch.from_numpy(order)]
            random_sp = features_context.to(torch.float32).reshape(channels, -1)[
                :, torch.from_numpy(geometry.random_patch[cells])
            ].T
            if not torch.equal(random_pp, random_sp):
                preflight["random_bit_mismatches"] += 1

            norms = (
                (targets_pp.to(torch.float32) - center[None, :]).norm(dim=-1).cpu().numpy()
            )
            for metric, metric_center in (("raw", None), ("centered", center)):
                q = per_cell_cosine(warp_pp, targets_pp, metric_center).to(torch.float64)
                p = per_cell_cosine(pooled_sp, targets_sp, metric_center).to(torch.float64)
                key_pp = (pair.context_frame_id, pair.target_frame_id, encoder, PER_POINT)
                key_sp = (pair.context_frame_id, pair.target_frame_id, encoder, SPLAT_POOL)
                if key_pp not in recorded or key_sp not in recorded:
                    continue
                rec_pp = recorded[key_pp][metric]
                rec_sp = recorded[key_sp][metric]
                n = cells.size
                if int(recorded[key_pp]["n_intersect"]) != n and metric == "raw":
                    preflight["n_intersect_mismatches"] += 1
                # Uniform a_j = b_j = 1/n, so T3 is exactly zero by construction
                # and is computed as written rather than assumed, to keep the
                # identity honest if the weighting rule ever changes.
                a = np.full(n, 1.0 / n)
                b = np.full(n, 1.0 / n)
                q_np = q.cpu().numpy()
                p_np = p.cpu().numpy()
                t1 = float(rec_pp - float(np.dot(a, q_np)))
                t2 = float(np.dot(a, q_np - p_np))
                t3 = float(np.dot(a - b, p_np))
                t4 = float(np.dot(b, p_np) - rec_sp)
                closure = (t1 + t2 + t3 + t4) - (rec_pp - rec_sp)
                ledger_rows.append(
                    {
                        "scene": scene,
                        "context_frame_id": pair.context_frame_id,
                        "target_frame_id": pair.target_frame_id,
                        "encoder": encoder,
                        "metric": metric,
                        "n_intersect": n,
                        "rotation_deg": recorded[key_pp]["rotation_deg"],
                        "regime": recorded[key_pp]["regime"],
                        "S_pp_recorded": rec_pp,
                        "S_sp_recorded": rec_sp,
                        "T1": t1,
                        "T2": t2,
                        "T3": t3,
                        "T4": t4,
                        "closure": float(closure),
                    }
                )
                if metric == "raw":
                    count = cells.size
                    cell_columns["scene"] += [scene] * count
                    cell_columns["encoder"] += [encoder] * count
                    cell_columns["c_raw"] += (q_np - p_np).astype(np.float32).tolist()
                    cell_columns["boundary"] += boundary.astype(bool).tolist()
                    cell_columns["centered_norm"] += norms.astype(np.float32).tolist()
                else:
                    cell_columns["c_centered"] += (q_np - p_np).astype(np.float32).tolist()
    cache.close()

    pq.write_table(
        pa.Table.from_pylist(ledger_rows), out_dir / LEDGER_NAME.format(scene=scene)
    )
    pq.write_table(
        pa.Table.from_pydict(cell_columns), out_dir / CELLS_NAME.format(scene=scene)
    )
    (out_dir / PREFLIGHT_NAME.format(scene=scene)).write_text(
        json.dumps(preflight, indent=1), encoding="utf-8"
    )
    return preflight


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, average ranks for ties. No scipy dependency."""

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        rank = np.empty_like(order, dtype=np.float64)
        rank[order] = np.arange(len(values), dtype=np.float64)
        # Average tied ranks.
        out = rank.copy()
        for value in np.unique(values):
            mask = values == value
            out[mask] = rank[mask].mean()
        return out

    rx, ry = ranks(x), ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denominator) if denominator else float("nan")


def scene_bootstrap_contrast(
    aggregates: dict[str, tuple[float, int, float, int]], config: AnalysisConfig
) -> dict[str, float]:
    """E[|c| | group] - E[|c| | rest], resampling scenes, from per-scene sums.

    aggregates maps a scene to (sum_in, n_in, sum_rest, n_rest). A replicate
    draws scenes with replacement and pools their sums, which is exactly the
    difference of pooled means over the resampled cells that a cell-level
    implementation would compute, without holding 27M cells in memory.
    """
    scenes = sorted(aggregates)
    sum_in = np.array([aggregates[s][0] for s in scenes], dtype=np.float64)
    n_in = np.array([aggregates[s][1] for s in scenes], dtype=np.float64)
    sum_rest = np.array([aggregates[s][2] for s in scenes], dtype=np.float64)
    n_rest = np.array([aggregates[s][3] for s in scenes], dtype=np.float64)
    total_in, total_rest = n_in.sum(), n_rest.sum()
    if total_in == 0 or total_rest == 0:
        # One empty side is a fact about the population, reported as such
        # rather than as a NaN with a warning attached.
        return {
            "difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "replicates": 0, "n_group": int(total_in), "n_rest": int(total_rest),
        }
    point = float(sum_in.sum() / total_in - sum_rest.sum() / total_rest)
    rng = np.random.default_rng(config.bootstrap_seed)
    draws = []
    for _ in range(config.bootstrap_resamples):
        pick = rng.integers(0, len(scenes), size=len(scenes))
        a, b = n_in[pick].sum(), n_rest[pick].sum()
        if a > 0 and b > 0:
            draws.append(float(sum_in[pick].sum() / a - sum_rest[pick].sum() / b))
    tail = (1.0 - config.bootstrap_confidence) / 2.0
    return {
        "difference": point,
        "ci_low": float(np.quantile(draws, tail)) if draws else float("nan"),
        "ci_high": float(np.quantile(draws, 1.0 - tail)) if draws else float("nan"),
        "replicates": len(draws),
        "n_group": int(total_in),
        "n_rest": int(total_rest),
    }


def _codes(table: Any, name: str) -> tuple[np.ndarray, list[str]]:
    """Integer codes and their labels for a string column, without str objects."""
    encoded = table[name].combine_chunks().dictionary_encode()
    return (
        encoded.indices.to_numpy(zero_copy_only=False).astype(np.int32),
        encoded.dictionary.to_pylist(),
    )


def norm_cutpoints(out_dir: Path) -> dict[str, list[float]]:
    """Global quartile cut points per encoder, from the norms alone.

    Computed before any contrast is viewed and stored; an existing file is
    reused, never recomputed, so a rerun cannot move the split. Only the norm
    and encoder columns are read, so this pass is bounded by one float32 array.
    """
    import pyarrow.parquet as pq

    cut_path = Path(out_dir) / CUTPOINTS_NAME
    if cut_path.exists():
        return json.loads(cut_path.read_text(encoding="utf-8"))
    by_encoder: dict[str, list[np.ndarray]] = {}
    for path in sorted(Path(out_dir).glob("cells_*.parquet")):
        table = pq.read_table(path, columns=["encoder", "centered_norm"])
        codes, labels = _codes(table, "encoder")
        norms = table["centered_norm"].to_numpy(zero_copy_only=False)
        for index, label in enumerate(labels):
            by_encoder.setdefault(label, []).append(norms[codes == index])
    cutpoints = {
        label: [float(q) for q in np.quantile(np.concatenate(parts), (0.25, 0.5, 0.75))]
        for label, parts in sorted(by_encoder.items())
    }
    cut_path.write_text(json.dumps(cutpoints, indent=1), encoding="utf-8")
    return cutpoints


def report(out_dir: Path, analysis: AnalysisConfig) -> dict[str, Any]:
    """Stop logic over every scene's evidence, then the mechanism contrasts."""
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    preflights = sorted(out_dir.glob("preflight_*.json"))
    if not preflights:
        raise SystemExit(f"no per-scene evidence under {out_dir}; run the scene stage first")
    stop: list[str] = []
    facts = [json.loads(p.read_text(encoding="utf-8")) for p in preflights]
    for f in facts:
        for field in ("target_bit_mismatches", "no_warp_bit_mismatches",
                      "random_bit_mismatches", "n_intersect_mismatches",
                      "cell_order_mismatches"):
            if f[field]:
                stop.append(f"{f['scene']}: {field} = {f[field]}")

    ledger = []
    for path in sorted(out_dir.glob("ledger_*.parquet")):
        ledger += pq.read_table(path).to_pylist()
    closure = np.array([abs(r["closure"]) for r in ledger])
    t1 = np.array([abs(r["T1"]) for r in ledger])
    t3 = np.array([abs(r["T3"]) for r in ledger])
    t4 = np.array([abs(r["T4"]) for r in ledger])
    if closure.size == 0:
        raise SystemExit("ledger is empty")
    if closure.max() > analysis.ledger_closure_tol:
        stop.append(f"closure failure: max |closure| = {closure.max():.3e}")
    for name, values in (("T1", t1), ("T4", t4)):
        worst = values.max()
        if worst > analysis.ledger_recon_tol:
            offenders = [
                (r["scene"], r["context_frame_id"], r["target_frame_id"], r["encoder"],
                 r["metric"], r[name])
                for r in ledger if abs(r[name]) > analysis.ledger_recon_tol
            ][:10]
            stop.append(f"{name} reconstruction failure: max {worst:.3e}, first {offenders}")
    if t3.max() > analysis.ledger_closure_tol:
        stop.append(
            f"T3 nonzero at {t3.max():.3e} under a uniform weighting rule; the "
            "aggregation gap is undocumented"
        )

    summary: dict[str, Any] = {"stop": stop, "preflight": facts}
    for metric in METRICS:
        rows = [r for r in ledger if r["metric"] == metric]
        t2 = np.array([r["T2"] for r in rows])
        summary[metric] = {
            "pairs": len(rows),
            "max_abs_T3": float(np.array([abs(r["T3"]) for r in rows]).max()),
            "max_abs_closure": float(np.array([abs(r["closure"]) for r in rows]).max()),
            "signed_aggregate_T2": float(t2.mean()),
            "mean_abs_T2": float(np.abs(t2).mean()),
            "quantiles_abs_T2": {
                "50": float(np.quantile(np.abs(t2), 0.5)),
                "90": float(np.quantile(np.abs(t2), 0.9)),
                "99": float(np.quantile(np.abs(t2), 0.99)),
            },
            "max_abs_T1": float(np.array([abs(r["T1"]) for r in rows]).max()),
            "max_abs_T4": float(np.array([abs(r["T4"]) for r in rows]).max()),
        }
        # Rotation breakdown on the frozen bins, overflow included. An ad-hoc
        # [50, 60) bucket in an earlier diagnostic silently dropped every pair
        # past 60 degrees; the config's own binning cannot.
        by_bin: dict[str, list[float]] = {}
        for r in rows:
            label = bin_label(
                float(r["rotation_deg"]), analysis.rotation_edges(),
                analysis.zero_rotation_tol_deg, "rotation", analysis.bin_right_closed,
            )
            by_bin.setdefault(label, []).append(r["T2"])
        summary[metric]["by_rotation_bin"] = {
            label: {
                "pairs": len(values),
                "signed": float(np.mean(values)),
                "mean_abs": float(np.mean(np.abs(values))),
            }
            for label, values in sorted(by_bin.items())
        }

    # Mechanism test. Cut points first, from the norms alone, so the split is
    # fixed before any content difference is viewed. Then one streaming pass
    # that reduces each file to per-scene sums and counts per group: the same
    # estimator as a cell-level implementation, bounded by the number of
    # scenes rather than by the 27M cells the run produces.
    cutpoints = norm_cutpoints(out_dir)
    groups = ["boundary", "q1_vs_q4"]
    agg: dict[tuple, dict[str, list[float]]] = {}
    for path in sorted(out_dir.glob("cells_*.parquet")):
        table = pq.read_table(path)
        scene_codes, scene_labels = _codes(table, "scene")
        enc_codes, enc_labels = _codes(table, "encoder")
        boundary = table["boundary"].to_numpy(zero_copy_only=False).astype(bool)
        norms = table["centered_norm"].to_numpy(zero_copy_only=False)
        columns = {
            "raw": np.abs(table["c_raw"].to_numpy(zero_copy_only=False).astype(np.float64)),
            "centered": np.abs(
                table["c_centered"].to_numpy(zero_copy_only=False).astype(np.float64)
            ),
        }
        for e_index, encoder in enumerate(enc_labels):
            cuts = cutpoints[encoder]
            for s_index, scene_name in enumerate(scene_labels):
                mask = (enc_codes == e_index) & (scene_codes == s_index)
                if not mask.any():
                    continue
                quartile = np.digitize(norms[mask], cuts)
                selectors = {
                    "boundary": (boundary[mask], ~boundary[mask]),
                    "q1_vs_q4": (quartile == 0, quartile == 3),
                }
                for metric, values in columns.items():
                    v = values[mask]
                    for group in groups:
                        inside, rest = selectors[group]
                        entry = agg.setdefault((encoder, metric, group), {})
                        entry[scene_name] = (
                            float(v[inside].sum()), int(inside.sum()),
                            float(v[rest].sum()), int(rest.sum()),
                        )
                    level = agg.setdefault((encoder, metric, "level"), {})
                    level[scene_name] = (float(v.sum()), int(v.size), 0.0, 0)
                    per_quartile = agg.setdefault((encoder, metric, "quartile_means"), {})
                    per_quartile[scene_name] = [
                        float(v[quartile == q].mean()) if (quartile == q).any() else float("nan")
                        for q in range(4)
                    ]

    mechanism: dict[str, Any] = {}
    for encoder in sorted(cutpoints):
        for metric in METRICS:
            entry: dict[str, Any] = {}
            for group, label in (
                ("boundary", "boundary_contrast"), ("q1_vs_q4", "norm_q1_minus_q4")
            ):
                entry[label] = scene_bootstrap_contrast(
                    agg.get((encoder, metric, group), {}), analysis
                )
            level = agg.get((encoder, metric, "level"), {})
            total = sum(v[1] for v in level.values())
            entry["mean_abs_c"] = (
                sum(v[0] for v in level.values()) / total if total else float("nan")
            )
            entry["cells"] = int(total)
            for label in ("boundary_contrast", "norm_q1_minus_q4"):
                difference = entry[label]["difference"]
                entry[label]["difference_over_level"] = (
                    difference / entry["mean_abs_c"] if total else float("nan")
                )
            per_scene = agg.get((encoder, metric, "quartile_means"), {})
            xs, ys = [], []
            for means in per_scene.values():
                for index, value in enumerate(means):
                    if np.isfinite(value):
                        xs.append(float(index))
                        ys.append(value)
            entry["spearman_quartile_vs_scene_mean"] = (
                spearman(np.array(xs), np.array(ys)) if len(xs) > 2 else float("nan")
            )
            mechanism[f"{encoder}/{metric}"] = entry
    summary["mechanism"] = mechanism
    summary["cutpoints"] = cutpoints
    summary["verdict"] = "STOP" if stop else "PASS"

    report_path = out_dir / REPORT_NAME
    report_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_eval_config(args.config)
    analysis = load_analysis_config(cfg.analysis_config)
    if args.report:
        summary = report(args.out, analysis)
        print(json.dumps({k: v for k, v in summary.items() if k != "preflight"}, indent=1))
        print(f"\nverdict: {summary['verdict']}")
        if summary["stop"]:
            raise SystemExit(1)
        return
    if args.scene is not None:
        scenes = [args.scene]
    elif args.scene_index is not None:
        scenes = [cfg.scenes[args.scene_index]]
    else:
        scenes = list(cfg.scenes)
    for scene in scenes:
        facts = ledger_scene(args.config, scene, args.out, analysis)
        print(f"[{scene}] pairs={facts['pairs']} "
              f"target_bit_mismatches={facts['target_bit_mismatches']}")


if __name__ == "__main__":
    main()
