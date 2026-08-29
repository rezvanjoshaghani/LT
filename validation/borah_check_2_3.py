"""VALIDATION 2.3, the centerpiece: independent pixels-to-rows reproduction.

Recomputes every Experiment Zero row for one scene from the frozen inputs
(the cached feature arrays, the manifest with cameras and ground-truth depth)
and compares row by row against the shipped corrected parquet on
(pair, path, variant): persisted mask bit-for-bit, n, n_intersect, and every
metric column within 1e-4.

Independence boundary, per VALIDATION.md ground rule 4 and 2.3:

- Data access is allowed: manifest JSON, npz caches, and parquet are read
  with json, numpy, and pyarrow directly. The pair list under audit comes
  from the shipped parquet itself, which is the persisted identity 2.3 names.
- lot.sample_identity is consumed as the identity oracle only. The corrected
  schema persists validity masks rather than raw id lists, and the hash
  constants are constants of the protocol (PROTOCOL 3.2 and 3.6 define the
  draws as fixed hashes of sample_id); deriving them through the one module
  that holds those constants is using persisted-equivalent identity inputs,
  not re-using pipeline math. Everything that transforms coordinates, decides
  visibility or eligibility, pools, masks, or scores is reimplemented here or
  in validation/independent.py, from the protocol text.
- Geometry runs in float32 to match the run's declared geometry dtype, since
  a landing decision at a cell boundary is part of what one row is; score
  accumulation runs in float64.

Usage (Borah, from the repo root, clean tree):
    python validation/borah_check_2_3.py \
        --renders data/replica_renders --cache cache/features \
        --eval-dir outputs/experiment_zero/eval --scene apartment_0 \
        --out validation/evidence/reaudit/borah_check_2_3.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "src"))

import independent as ind  # noqa: E402
from lot.sample_identity import (  # noqa: E402  (identity oracle only)
    NEIGHBOR_PATCH_SALT,
    RANDOM_PATCH_SALT,
    derived_draw,
    sample_ids,
)

PATCH = 14
VARIANTS = ("Oracle-Transport", "No-Warp-Copy", "Neighbor-Patch", "Random-Patch", "Mean-Feature")
METRIC_COLUMNS = (
    "cosine_mean", "l2_mean", "cosine_centered_mean", "l2_centered_mean",
    "cosine_intersect_mean", "cosine_centered_intersect_mean",
)
NEIGHBOR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))
TOL = 1e-4


# ---------------------------------------------------------------------------
# Inputs, read directly
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = {}
    for frame in payload["frames"]:
        frames[frame["frame_id"]] = {
            "K": np.array(frame["K"], dtype=np.float64),
            "T": np.array(frame["T_world_from_camera"], dtype=np.float64),
            "depth_path": frame["depth_path"],
            "height": frame["height"],
            "width": frame["width"],
        }
    return frames


def load_features(cache: Path, encoder: str, scene: str) -> dict[str, np.ndarray]:
    with np.load(cache / encoder / scene / "features.npz") as archive:
        return {name: archive[name].astype(np.float32) for name in archive.files}


def load_rows(eval_dir: Path, scene: str) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(eval_dir / f"{scene}.parquet").to_pylist()


def recompute_mean_vector(cache: Path, encoder: str, scenes: list[str]) -> np.ndarray:
    """PROTOCOL 3.6's one global vector: mean over all frames and positions."""
    total, count = None, 0
    for scene in sorted(scenes):
        with np.load(cache / encoder / scene / "features.npz") as archive:
            for name in sorted(archive.files):
                values = archive[name].astype(np.float32)
                per_frame = values.reshape(values.shape[0], -1).mean(axis=1)
                total = per_frame if total is None else total + per_frame
                count += 1
    return (total / count).astype(np.float32)


# ---------------------------------------------------------------------------
# Independent per-pair reconstruction
# ---------------------------------------------------------------------------

def sampling_box(hw: tuple[int, int]) -> tuple[float, float, float, float]:
    lo = (0 + 0.5) * PATCH - 0.5
    u_max = (hw[1] // PATCH - 1 + 0.5) * PATCH - 0.5
    v_max = (hw[0] // PATCH - 1 + 0.5) * PATCH - 0.5
    return lo, u_max, lo, v_max


def in_box(uv: np.ndarray, box) -> np.ndarray:
    u_min, u_max, v_min, v_max = box
    return (
        (uv[..., 0] >= u_min) & (uv[..., 0] <= u_max)
        & (uv[..., 1] >= v_min) & (uv[..., 1] <= v_max)
    )


def splat_internals(depth_ctx: np.ndarray, K_ctx, K_tgt, T, out_hw):
    """Winner splat internals in float32: per-cell source-patch weights.

    Reimplements PROTOCOL's pixel-level z-buffered splat and patch pooling
    from the text: every valid context pixel carries its patch's feature to
    the nearest target pixel, the per-pixel depth minimum wins with ties
    averaged at relative 1e-6, hit pixels average into patches. Returns
    (weights [cells, source_patches] float64, coverage [cells]).
    """
    h, w = depth_ctx.shape
    hp, wp = h // PATCH, w // PATCH
    oh, ow = out_hw
    ohp, owp = oh // PATCH, ow // PATCH
    # The whole landing chain runs in float32, matching the run's declared
    # geometry dtype, so near-tie z-buffer winners resolve the same way. The
    # formulas are this script's own; only the precision mirrors the pipeline.
    depth32 = depth_ctx.astype(np.float32)
    Ks = K_ctx.astype(np.float32)
    Kd = K_tgt.astype(np.float32)
    Tf = T.astype(np.float32)
    uv = ind.pixel_grid(h, w).astype(np.float32).reshape(-1, 2)
    z = depth32.reshape(-1)
    x = (uv[:, 0] - Ks[0, 2]) * z / Ks[0, 0]
    y = (uv[:, 1] - Ks[1, 2]) * z / Ks[1, 1]
    pts = np.stack((x, y, z), axis=-1) @ Tf[:3, :3].T + Tf[:3, 3]
    z_t = pts[:, 2]
    uv_t = np.stack(
        (Kd[0, 0] * pts[:, 0] / z_t + Kd[0, 2], Kd[1, 1] * pts[:, 1] / z_t + Kd[1, 2]),
        axis=-1,
    )
    keep = (z > 0) & np.isfinite(z) & (z_t > 0) & np.isfinite(z_t)
    keep &= np.isfinite(uv_t).all(axis=-1)
    iu = np.floor(uv_t[:, 0] + 0.5).astype(np.int64)
    iv = np.floor(uv_t[:, 1] + 0.5).astype(np.int64)
    keep &= (iu >= 0) & (iu < ow) & (iv >= 0) & (iv < oh)

    lin = (iv * ow + iu)[keep]
    zk = z_t[keep].astype(np.float32)
    source_patch = (
        (np.arange(h) // PATCH)[:, None] * wp + (np.arange(w) // PATCH)[None, :]
    ).reshape(-1)[keep]

    zbuffer = np.full(oh * ow, np.inf, dtype=np.float32)
    np.minimum.at(zbuffer, lin, zk)
    winners = zk <= zbuffer[lin] * np.float32(1 + 1e-6)
    lin_w, src_w = lin[winners], source_patch[winners]

    count = np.zeros(oh * ow, dtype=np.float64)
    np.add.at(count, lin_w, 1.0)
    hit_cells = (lin_w // ow // PATCH) * owp + (lin_w % ow) // PATCH
    hit_pixel_cell = np.zeros(oh * ow, dtype=bool)
    hit_pixel_cell[lin_w] = True
    hits = hit_pixel_cell.reshape(ohp, PATCH, owp, PATCH).sum(axis=(1, 3)).reshape(-1)

    weights = np.zeros((ohp * owp, hp * wp), dtype=np.float64)
    np.add.at(weights, (hit_cells, src_w), 1.0 / count[lin_w])
    weights /= np.maximum(hits, 1.0)[:, None]
    coverage = hits / float(PATCH * PATCH)
    return weights, coverage


def covisible_fraction_per_cell(covisible: np.ndarray) -> np.ndarray:
    h, w = covisible.shape
    return covisible.reshape(h // PATCH, PATCH, w // PATCH, PATCH).mean(axis=(1, 3)).reshape(-1)


def pack_mask(mask: np.ndarray) -> bytes:
    return np.packbits(mask.astype(np.uint8)).tobytes()


def reconstruct_pair(scene, ctx_id, tgt_id, frames, depths, features, center, rel_tol,
                     min_covisible_fraction):
    """Every row of one (pair, encoder) from first principles. Returns rows dict."""
    ctx, tgt = frames[ctx_id], frames[tgt_id]
    K_ctx, K_tgt = ctx["K"].astype(np.float32), tgt["K"].astype(np.float32)
    T = ind.relative(tgt["T"], ctx["T"]).astype(np.float32)
    T_inv = ind.invert_pose(T.astype(np.float64)).astype(np.float32)
    depth_ctx = depths[ctx_id].astype(np.float32)
    depth_tgt = depths[tgt_id].astype(np.float32)
    h, w = depth_tgt.shape
    hp, wp = h // PATCH, w // PATCH
    size = hp * wp

    covisible, _, _ = ind.covisible_mask(
        depth_tgt, depth_ctx, K_tgt, K_ctx, T, rel_tol=rel_tol
    )

    # Per-point candidates: target patch centers whose four surrounding pixels
    # are co-visible and share one surface within the frozen tolerance.
    vv, uu = np.meshgrid(
        np.arange(hp) * PATCH + (PATCH - 1) / 2.0,
        np.arange(wp) * PATCH + (PATCH - 1) / 2.0,
        indexing="ij",
    )
    centers = np.stack((uu.reshape(-1), vv.reshape(-1)), axis=-1).astype(np.float32)
    x0 = np.floor(centers[:, 0]).astype(int)
    y0 = np.floor(centers[:, 1]).astype(int)
    corners = np.stack(
        (depth_tgt[y0, x0], depth_tgt[y0, x0 + 1], depth_tgt[y0 + 1, x0],
         depth_tgt[y0 + 1, x0 + 1]), axis=-1,
    )
    lo, hi = corners.min(axis=-1), corners.max(axis=-1)
    one_surface = (lo > 0) & np.isfinite(hi) & ((hi - lo) <= rel_tol * lo)
    keep = (
        covisible[y0, x0] & covisible[y0, x0 + 1]
        & covisible[y0 + 1, x0] & covisible[y0 + 1, x0 + 1] & one_surface
    )
    box_t = sampling_box((h, w))
    box_c = sampling_box(depth_ctx.shape)
    keep &= in_box(centers, box_t) & in_box(centers, box_c)

    depth_at = ind.sample_bilinear(depth_tgt, centers).astype(np.float32)
    keep &= (depth_at > 0) & np.isfinite(depth_at)

    # The warp chain runs in float32 to match the run's declared geometry
    # dtype. The formulas are this script's own; only the precision mirrors
    # the pipeline, because a score defined on float32 coordinates carries
    # float32 coordinate noise that a float64 rederivation would misattribute
    # to disagreement.
    def warp32(uv, depth, K_src, K_dst, T_dst_from_src):
        uv = uv.astype(np.float32)
        depth = depth.astype(np.float32)
        Ks = K_src.astype(np.float32)
        Kd = K_dst.astype(np.float32)
        Tf = T_dst_from_src.astype(np.float32)
        x = (uv[:, 0] - Ks[0, 2]) * depth / Ks[0, 0]
        y = (uv[:, 1] - Ks[1, 2]) * depth / Ks[1, 1]
        pts = np.stack((x, y, depth), axis=-1)
        pts = pts @ Tf[:3, :3].T + Tf[:3, 3]
        z = pts[:, 2]
        u = Kd[0, 0] * pts[:, 0] / z + Kd[0, 2]
        v = Kd[1, 1] * pts[:, 1] / z + Kd[1, 2]
        return np.stack((u, v), axis=-1), z

    uv_warp, z_c = warp32(centers, depth_at, K_tgt, K_ctx, T_inv)
    keep &= (z_c > 0) & in_box(uv_warp, box_c)

    import torch

    ids_universe = sample_ids(
        scene, ctx_id, tgt_id, torch.from_numpy(centers.astype(np.float64))
    )

    # Splat internals, weights, and the splat-side neighbour admissibility.
    weights, coverage = splat_internals(depth_ctx, K_ctx, K_tgt, T, (h, w))
    covis_cells = covisible_fraction_per_cell(covisible)
    splat_mask = (covis_cells >= min_covisible_fraction) & (coverage > 0)

    chp, cwp = depth_ctx.shape[0] // PATCH, depth_ctx.shape[1] // PATCH
    valid_shift = {}
    for dx, dy in NEIGHBOR_OFFSETS:
        ok = np.zeros((chp, cwp), dtype=bool)
        rows_src = slice(max(0, dy), min(chp, chp + dy))
        cols_src = slice(max(0, dx), min(cwp, cwp + dx))
        ok[rows_src.start - dy: rows_src.stop - dy,
           cols_src.start - dx: cols_src.stop - dx] = True
        valid_shift[(dx, dy)] = ok.reshape(-1)
    splat_option_ok = np.zeros((size, 4), dtype=bool)
    for index, offset in enumerate(NEIGHBOR_OFFSETS):
        leaked = weights @ (~valid_shift[offset]).astype(np.float64)
        splat_option_ok[:, index] = leaked <= 0

    # Per-point neighbour admissibility intersected with the splat rule at the
    # sampled cells; the direction is one hash over the intersection.
    options = uv_warp[:, None, :] + np.array(
        [[dx * PATCH, dy * PATCH] for dx, dy in NEIGHBOR_OFFSETS], dtype=np.float32
    )[None, :, :]
    option_ok = in_box(options, box_c)
    cell_of_sample = (
        np.round((centers[:, 1] + 0.5) / PATCH - 0.5).astype(int) * wp
        + np.round((centers[:, 0] + 0.5) / PATCH - 0.5).astype(int)
    )
    option_ok &= splat_option_ok[cell_of_sample]
    has_neighbor = option_ok.any(axis=1)
    keep &= has_neighbor

    chosen = np.flatnonzero(keep)
    ids = ids_universe[chosen]
    uv_t_s = centers[chosen]
    uv_w_s = uv_warp[chosen]
    opts_s = options[chosen]
    ok_s = option_ok[chosen]

    counts = ok_s.sum(axis=1)
    pick = derived_draw(ids, NEIGHBOR_PATCH_SALT, counts)
    rank = np.cumsum(ok_s, axis=1) - 1
    direction = (ok_s & (rank == pick[:, None])).argmax(axis=1)
    uv_n_s = opts_s[np.arange(len(direction)), direction]

    random_flat = derived_draw(ids, RANDOM_PATCH_SALT, chp * cwp)
    random_universe = derived_draw(ids_universe, RANDOM_PATCH_SALT, chp * cwp)

    features_ctx = features[ctx_id]
    features_tgt = features[tgt_id]
    channels = features_ctx.shape[0]
    flat_ctx = features_ctx.reshape(channels, -1)
    flat_tgt = features_tgt.reshape(channels, -1)

    per_point_mask = np.zeros(size, dtype=bool)
    per_point_mask[cell_of_sample[chosen]] = True
    shared = per_point_mask & splat_mask

    def cos32(a, b, c=None):
        # Per-sample arithmetic in float32, mirroring the frozen pipeline's
        # dtype, accumulated in float64. The formula is the protocol's.
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if c is not None:
            a = a - c
            b = b - c
        a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12).astype(np.float32)
        b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12).astype(np.float32)
        return float(np.mean(np.sum(a * b, axis=-1, dtype=np.float64)))

    def l232(a, b, c=None):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if c is not None:
            a = a - c
            b = b - c
        a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12).astype(np.float32)
        b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12).astype(np.float32)
        return float(np.mean(np.linalg.norm((a - b).astype(np.float64), axis=-1)))

    def score(pred, target, in_shared):
        raw_c = cos32(pred, target)
        raw_l = l232(pred, target)
        cen_c = cos32(pred, target, center)
        cen_l = l232(pred, target, center)
        if in_shared.any():
            int_c = cos32(pred[in_shared], target[in_shared])
            int_cc = cos32(pred[in_shared], target[in_shared], center)
        else:
            int_c = int_cc = float("nan")
        return {
            "cosine_mean": raw_c, "l2_mean": raw_l,
            "cosine_centered_mean": cen_c, "l2_centered_mean": cen_l,
            "cosine_intersect_mean": int_c, "cosine_centered_intersect_mean": int_cc,
        }

    def score_mean_feature(target, in_shared):
        pred = np.broadcast_to(center, target.shape)
        out = score(pred, target, in_shared)
        out["cosine_centered_mean"] = float("nan")
        out["l2_centered_mean"] = float("nan")
        out["cosine_centered_intersect_mean"] = float("nan")
        return out

    rows = {}
    target_reads = ind.sample_features(features_tgt, uv_t_s)
    in_shared_pp = shared[cell_of_sample[chosen]]
    reads = {
        "Oracle-Transport": ind.sample_features(features_ctx, uv_w_s),
        "No-Warp-Copy": ind.sample_features(features_ctx, uv_t_s),
        "Neighbor-Patch": ind.sample_features(features_ctx, uv_n_s),
        "Random-Patch": flat_ctx[:, random_flat].T,
    }
    for variant, prediction in reads.items():
        rows[("per_point", variant)] = {
            "mask": pack_mask(per_point_mask), "n": int(keep.sum()),
            "n_intersect": int(shared.sum()),
            **score(prediction, target_reads, in_shared_pp),
        }
    rows[("per_point", "Mean-Feature")] = {
        "mask": pack_mask(per_point_mask), "n": int(keep.sum()),
        "n_intersect": int(shared.sum()),
        **score_mean_feature(target_reads, in_shared_pp),
    }

    cells_sp = np.flatnonzero(splat_mask)
    in_shared_sp = shared[cells_sp]
    pooled = (flat_ctx @ weights.T)
    shifted_pool = np.zeros((channels, size), dtype=np.float64)
    # The universe hash ranks the same intersected admissible set the sampler
    # used at sampled cells; unsampled cells keep the splat rule alone.
    universe_ok = splat_option_ok.copy()
    universe_ok[cell_of_sample[chosen]] = ok_s
    counts_u = universe_ok.sum(axis=1)
    if (counts_u == 0).any():
        raise AssertionError("a cell has no admissible neighbour offset")
    pick_u = derived_draw(ids_universe, NEIGHBOR_PATCH_SALT, counts_u)
    rank_u = np.cumsum(universe_ok, axis=1) - 1
    dir_u = (universe_ok & (rank_u == pick_u[:, None])).argmax(axis=1)
    for index, (dx, dy) in enumerate(NEIGHBOR_OFFSETS):
        cells_dir = np.flatnonzero(dir_u == index)
        if not cells_dir.size:
            continue
        shifted = np.zeros_like(features_ctx)
        rows_src = slice(max(0, dy), min(chp, chp + dy))
        cols_src = slice(max(0, dx), min(cwp, cwp + dx))
        shifted[:, rows_src.start - dy: rows_src.stop - dy,
                cols_src.start - dx: cols_src.stop - dx] = (
            features_ctx[:, rows_src, cols_src]
        )
        shifted_pool[:, cells_dir] = shifted.reshape(channels, -1) @ weights[cells_dir].T

    targets_sp = flat_tgt[:, cells_sp].T
    splat_reads = {
        "Oracle-Transport": pooled[:, cells_sp].T,
        "No-Warp-Copy": flat_ctx[:, cells_sp].T,
        "Neighbor-Patch": shifted_pool[:, cells_sp].T,
        "Random-Patch": flat_ctx[:, random_universe[cells_sp]].T,
    }
    for variant, prediction in splat_reads.items():
        rows[("splat_pool", variant)] = {
            "mask": pack_mask(splat_mask), "n": int(cells_sp.size),
            "n_intersect": int(shared.sum()),
            **score(prediction, targets_sp, in_shared_sp),
        }
    rows[("splat_pool", "Mean-Feature")] = {
        "mask": pack_mask(splat_mask), "n": int(cells_sp.size),
        "n_intersect": int(shared.sum()),
        **score_mean_feature(targets_sp, in_shared_sp),
    }

    baseline = float(np.linalg.norm(T[:3, 3]))
    covis_depths = depth_tgt[covisible & np.isfinite(depth_tgt) & (depth_tgt > 0)]
    parallax = baseline / float(np.median(covis_depths)) if covis_depths.size else float("nan")
    return rows, {
        "rotation_deg": ind.rotation_deg(ind.relative(tgt["T"], ctx["T"])[:3, :3]),
        "parallax": parallax,
        "covisible_fraction": float(covisible.mean()),
    }


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--scene", type=str, default="apartment_0")
    parser.add_argument("--encoder", type=str, default="dinov2_vitb14")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="audit only the first N pairs, for a smoke pass")
    args = parser.parse_args()

    import yaml

    analysis = yaml.safe_load(Path("configs/analysis.yaml").read_text())
    rel_tol = analysis["covisible_relative_depth_tol"]
    min_cf = analysis["min_covisible_fraction"]

    frames = load_manifest(args.renders / args.scene / "manifest.json")
    depths = {
        fid: np.load(args.renders / args.scene / frame["depth_path"])
        for fid, frame in frames.items()
    }
    features = load_features(args.cache, args.encoder, args.scene)
    rows = [r for r in load_rows(args.eval_dir, args.scene) if r["encoder"] == args.encoder]

    record_path = args.eval_dir.parent / f"mean_vector_{args.encoder}.json"
    stored_record = json.loads(record_path.read_text(encoding="utf-8"))
    center = recompute_mean_vector(args.cache, args.encoder, stored_record["scenes"])
    stored_vector = np.load(args.eval_dir.parent / f"mean_vector_{args.encoder}.npy")
    mean_vector_diff = float(np.abs(center - stored_vector.astype(np.float32)).max())

    by_pair: dict[tuple, dict] = {}
    for row in rows:
        by_pair.setdefault(
            (row["context_frame_id"], row["target_frame_id"]), {}
        )[(row["path"], row["variant"])] = row

    pairs = sorted(by_pair)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    summary = {
        "scene": args.scene, "encoder": args.encoder, "pairs": len(pairs),
        "rows_compared": 0, "mask_mismatches": 0, "count_mismatches": 0,
        "metric_max_abs_diff": {c: 0.0 for c in METRIC_COLUMNS},
        "pair_field_max_abs_diff": {"rotation_deg": 0.0, "parallax": 0.0,
                                     "covisible_fraction": 0.0},
        "mean_vector_max_abs_diff": mean_vector_diff,
        "tolerance": TOL,
        "failures": [],
    }

    for pair_index, (ctx_id, tgt_id) in enumerate(pairs):
        shipped = by_pair[(ctx_id, tgt_id)]
        mine, pair_fields = reconstruct_pair(
            args.scene, ctx_id, tgt_id, frames, depths, features, center,
            rel_tol, min_cf,
        )
        any_row = next(iter(shipped.values()))
        for field in ("rotation_deg", "parallax", "covisible_fraction"):
            diff = abs(pair_fields[field] - any_row[field])
            summary["pair_field_max_abs_diff"][field] = max(
                summary["pair_field_max_abs_diff"][field], diff
            )
            if diff > TOL:
                summary["failures"].append(
                    f"{ctx_id}->{tgt_id}: {field} {pair_fields[field]} vs {any_row[field]}"
                )
        for key, theirs in shipped.items():
            ours = mine.get(key)
            if ours is None:
                summary["failures"].append(f"{ctx_id}->{tgt_id}: no reconstruction for {key}")
                continue
            summary["rows_compared"] += 1
            if bytes(theirs["sample_mask"]) != ours["mask"]:
                summary["mask_mismatches"] += 1
                summary["failures"].append(f"{ctx_id}->{tgt_id} {key}: mask differs")
                continue
            if theirs["n"] != ours["n"] or theirs["n_intersect"] != ours["n_intersect"]:
                summary["count_mismatches"] += 1
                summary["failures"].append(
                    f"{ctx_id}->{tgt_id} {key}: n {theirs['n']}/{ours['n']} "
                    f"n_intersect {theirs['n_intersect']}/{ours['n_intersect']}"
                )
                continue
            for column in METRIC_COLUMNS:
                a, b = theirs[column], ours[column]
                if isinstance(a, float) and math.isnan(a) and math.isnan(b):
                    continue
                diff = abs(a - b)
                summary["metric_max_abs_diff"][column] = max(
                    summary["metric_max_abs_diff"][column], diff
                )
                if diff > TOL:
                    summary["failures"].append(
                        f"{ctx_id}->{tgt_id} {key} {column}: {a} vs {b} (diff {diff:.2e})"
                    )
        if (pair_index + 1) % 50 == 0:
            print(f"  {pair_index + 1}/{len(pairs)} pairs", flush=True)

    summary["verdict"] = "PASS" if not summary["failures"] else "FAIL"
    summary["failures"] = summary["failures"][:50]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=1))
    print(f"verdict: {summary['verdict']} -> {args.out}")
    sys.exit(0 if summary["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
