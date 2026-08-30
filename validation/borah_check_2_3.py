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
- The mask-deciding computations run under the run's arithmetic contract:
  float32, the run's declared geometry_dtype, with the same operation order,
  and through the same kernel provider (torch on CPU) wherever the rounding
  of an operation is at a library's discretion (matmul, matrix-vector). The
  formulas below are this script's own, written from the protocol text; what
  is mirrored is only how they are rounded. The reason is the comparison
  contract itself: persisted sample masks are compared bit-for-bit, and a
  float32 eligibility decision at a sampling-box edge or a z-buffer tie is
  reproducible bit-for-bit only by executing the same rounding sequence.
  A float64 rederivation lands one ulp away from the run at such an edge and
  misreports its own coordinate noise as pipeline disagreement, which the
  apartment_0 diagnosis showed concretely: a warp coordinate one float32 ulp
  outside the box (margin -3e-05 px) flipped one sample in twelve pairs.
- Score accumulation stays independent and runs in float64 over per-sample
  float32 values, judged under the frozen 1e-4 tolerance. Independence of
  the audit lives in this file's code and in the comparison, not in freedom
  to round differently from the run it audits.

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
import torch

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

# Invalid context depths are replaced by this sentinel before the co-visibility
# read, so a sample landing on a depth hole fails the tolerance test.
INVALID_DEPTH_SENTINEL = 1e30
# Relative epsilon under which z-buffer ties are averaged, PROTOCOL 3.5.
TIE_RELATIVE_EPS = 1e-6


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
# The arithmetic-contract mirror: float32, the run's operation order, torch
# kernels. Every function here decides masks; none of them scores anything.
# ---------------------------------------------------------------------------

def pixel_grid32(height: int, width: int) -> torch.Tensor:
    """[H, W, 2] of (u, v) float32, pixel centers at integers."""
    v = torch.arange(height, dtype=torch.float32)
    u = torch.arange(width, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack((uu, vv), dim=-1)


def invert_se3_t(T: torch.Tensor) -> torch.Tensor:
    """Block-formula inverse of a rigid transform, in T's own dtype."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = torch.eye(4, dtype=T.dtype)
    out[:3, :3] = R.mT
    out[:3, 3] = -(R.mT @ t)
    return out


def relative_pose32(T_world_from_target: np.ndarray, T_world_from_context: np.ndarray) -> torch.Tensor:
    """T_target_from_context under the run's contract: float64 block-formula
    inverse and compose, then one rounding to float32."""
    A = torch.from_numpy(np.asarray(T_world_from_target, dtype=np.float64))
    B = torch.from_numpy(np.asarray(T_world_from_context, dtype=np.float64))
    return (invert_se3_t(A) @ B).to(torch.float32)


def unproject32(uv: torch.Tensor, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    x = (uv[..., 0] - K[0, 2]) * depth / K[0, 0]
    y = (uv[..., 1] - K[1, 2]) * depth / K[1, 1]
    return torch.stack((x, y, depth), dim=-1)


def transform32(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return points @ T[:3, :3].mT + T[:3, 3]


def project32(points: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = points[..., 2]
    u = K[0, 0] * points[..., 0] / z + K[0, 2]
    v = K[1, 1] * points[..., 1] / z + K[1, 2]
    return torch.stack((u, v), dim=-1), z


def bilinear32(grid: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Bilinear on an [H, W] float32 grid, cell centers at integers, border clamped."""
    height, width = grid.shape
    x = xy[..., 0].clamp(0, width - 1)
    y = xy[..., 1].clamp(0, height - 1)
    x0 = x.floor().clamp(max=max(width - 2, 0)).long()
    y0 = y.floor().clamp(max=max(height - 2, 0)).long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    wx = x - x0
    wy = y - y0
    return (
        grid[y0, x0] * (1 - wx) * (1 - wy)
        + grid[y0, x1] * wx * (1 - wy)
        + grid[y1, x0] * (1 - wx) * wy
        + grid[y1, x1] * wx * wy
    )


def nearest32(grid: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Nearest-cell read on an [H, W] grid, floor(x + 0.5), border clamped."""
    height, width = grid.shape
    x = torch.floor(xy[..., 0] + 0.5).long().clamp(0, width - 1)
    y = torch.floor(xy[..., 1] + 0.5).long().clamp(0, height - 1)
    return grid[y, x]


def covisible32(
    depth_tgt: torch.Tensor,
    depth_ctx: torch.Tensor,
    K_tgt: torch.Tensor,
    K_ctx: torch.Tensor,
    T_tgt_from_ctx: torch.Tensor,
    rel_tol: float,
) -> torch.Tensor:
    """Ground-truth co-visibility on the target grid, [H, W] bool.

    A valid target pixel is co-visible when its surface point, mapped into the
    context camera, lands inside the physical image extent with positive depth
    and the context z-buffer at the nearest cell records the same depth within
    rel_tol. The read is nearest-cell: a depth map is a z-buffer, and a value
    interpolated across an edge lies on no surface.
    """
    T_ctx_from_tgt = invert_se3_t(T_tgt_from_ctx)
    height, width = depth_tgt.shape
    ctx_height, ctx_width = depth_ctx.shape
    uv = pixel_grid32(height, width)
    valid = (depth_tgt > 0) & torch.isfinite(depth_tgt)
    points_tgt = unproject32(uv, depth_tgt, K_tgt)
    points_ctx = transform32(T_ctx_from_tgt, points_tgt)
    uv_ctx, z_ctx = project32(points_ctx, K_ctx)
    u = uv_ctx[..., 0]
    v = uv_ctx[..., 1]
    in_frustum = (
        valid
        & (z_ctx > 0)
        & torch.isfinite(u)
        & torch.isfinite(v)
        & (u >= -0.5)
        & (u < ctx_width - 0.5)
        & (v >= -0.5)
        & (v < ctx_height - 0.5)
    )
    d = torch.where(
        (depth_ctx > 0) & torch.isfinite(depth_ctx),
        depth_ctx,
        torch.full_like(depth_ctx, INVALID_DEPTH_SENTINEL),
    )
    safe_uv = torch.where(in_frustum[..., None], uv_ctx, torch.zeros_like(uv_ctx))
    seen = nearest32(d, safe_uv)
    agrees = (z_ctx - seen).abs() <= rel_tol * z_ctx
    return in_frustum & agrees


def patch_fraction32(mask: torch.Tensor) -> np.ndarray:
    """Fraction of set pixels per patch, float32 mean, [cells] row major."""
    height, width = mask.shape
    return (
        mask.to(torch.float32)
        .reshape(height // PATCH, PATCH, width // PATCH, PATCH)
        .mean(dim=(1, 3))
        .reshape(-1)
        .numpy()
    )


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


def splat_internals(depth_ctx: torch.Tensor, K_ctx: torch.Tensor, K_tgt: torch.Tensor,
                    T: torch.Tensor, out_hw: tuple[int, int]):
    """Winner splat internals: per-cell source-patch weights and coverage.

    Reimplements PROTOCOL's pixel-level z-buffered splat and patch pooling
    from the text: every valid context pixel carries its patch's feature to
    the nearest target pixel, the per-pixel depth minimum wins with ties
    averaged at relative 1e-6, hit pixels average into patches. The landing
    chain runs under the arithmetic contract above, because which pixel wins
    a near-tie is part of what one row is. The weights themselves feed only
    tolerance-checked scores and accumulate in float64. Returns
    (weights [cells, source_patches] float64, coverage [cells]).
    """
    h, w = depth_ctx.shape
    hp, wp = h // PATCH, w // PATCH
    oh, ow = out_hw
    ohp, owp = oh // PATCH, ow // PATCH

    uv = pixel_grid32(h, w)
    z = depth_ctx
    points_ctx = unproject32(uv, z, K_ctx)
    points_tgt = transform32(T, points_ctx)
    uv_t, z_t = project32(points_tgt, K_tgt)
    keep = (
        (z > 0) & torch.isfinite(z) & (z_t > 0) & torch.isfinite(z_t)
        & torch.isfinite(uv_t).all(dim=-1)
    )
    safe_uv = torch.where(keep[..., None], uv_t, torch.zeros_like(uv_t))
    iu = torch.floor(safe_uv[..., 0] + 0.5).long()
    iv = torch.floor(safe_uv[..., 1] + 0.5).long()
    keep &= (iu >= 0) & (iu < ow) & (iv >= 0) & (iv < oh)

    keep_flat = keep.reshape(-1)
    lin_t = (iv * ow + iu).reshape(-1)[keep_flat]
    zk = z_t.reshape(-1)[keep_flat]
    zbuffer = torch.full((oh * ow,), torch.inf, dtype=torch.float32)
    zbuffer.scatter_reduce_(0, lin_t, zk, reduce="amin", include_self=True)
    winners = zk <= zbuffer[lin_t] * (1 + TIE_RELATIVE_EPS)

    source_patch = (
        (np.arange(h) // PATCH)[:, None] * wp + (np.arange(w) // PATCH)[None, :]
    ).reshape(-1)[keep_flat.numpy()]
    lin = lin_t.numpy()
    winners_np = winners.numpy()
    lin_w, src_w = lin[winners_np], source_patch[winners_np]

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


def pack_mask(mask: np.ndarray) -> bytes:
    return np.packbits(mask.astype(np.uint8)).tobytes()


def reconstruct_pair(scene, ctx_id, tgt_id, frames, depths, features, center, rel_tol,
                     min_covisible_fraction):
    """Every row of one (pair, encoder) from first principles. Returns rows dict."""
    ctx, tgt = frames[ctx_id], frames[tgt_id]
    K_ctx = torch.from_numpy(ctx["K"]).to(torch.float32)
    K_tgt = torch.from_numpy(tgt["K"]).to(torch.float32)
    T = relative_pose32(tgt["T"], ctx["T"])
    T_inv = invert_se3_t(T)
    depth_ctx = depths[ctx_id].astype(np.float32)
    depth_tgt = depths[tgt_id].astype(np.float32)
    depth_ctx_t = torch.from_numpy(depth_ctx)
    depth_tgt_t = torch.from_numpy(depth_tgt)
    h, w = depth_tgt.shape
    hp, wp = h // PATCH, w // PATCH
    size = hp * wp

    covisible_t = covisible32(depth_tgt_t, depth_ctx_t, K_tgt, K_ctx, T, rel_tol)
    covisible = covisible_t.numpy()

    # Per-point candidates: target patch centers whose four surrounding pixels
    # are co-visible and share one surface within the frozen tolerance.
    vv, uu = np.meshgrid(
        np.arange(hp) * PATCH + (PATCH - 1) / 2.0,
        np.arange(wp) * PATCH + (PATCH - 1) / 2.0,
        indexing="ij",
    )
    centers = np.stack((uu.reshape(-1), vv.reshape(-1)), axis=-1).astype(np.float32)
    centers_t = torch.from_numpy(centers)
    x0 = centers_t[:, 0].floor().long()
    y0 = centers_t[:, 1].floor().long()
    corners = torch.stack(
        (depth_tgt_t[y0, x0], depth_tgt_t[y0, x0 + 1], depth_tgt_t[y0 + 1, x0],
         depth_tgt_t[y0 + 1, x0 + 1]), dim=-1,
    )
    lo = corners.amin(dim=-1)
    hi = corners.amax(dim=-1)
    one_surface = (lo > 0) & torch.isfinite(hi) & ((hi - lo) <= rel_tol * lo)
    keep = (
        covisible_t[y0, x0] & covisible_t[y0, x0 + 1]
        & covisible_t[y0 + 1, x0] & covisible_t[y0 + 1, x0 + 1] & one_surface
    ).numpy()
    box_t = sampling_box((h, w))
    box_c = sampling_box(depth_ctx.shape)
    keep &= in_box(centers, box_t) & in_box(centers, box_c)

    depth_at = bilinear32(depth_tgt_t, centers_t).numpy()
    keep &= (depth_at > 0) & np.isfinite(depth_at)

    # The ground-truth warp of the surviving candidates, at the shapes the run
    # used: the eligible set is filtered first and the [M, 3] chain runs on it.
    sel = np.flatnonzero(keep)
    sel_t = torch.from_numpy(sel)
    points_sel = unproject32(centers_t[sel_t], torch.from_numpy(depth_at[sel]), K_tgt)
    uv_warp_sel_t, z_sel_t = project32(transform32(T_inv, points_sel), K_ctx)
    uv_warp_sel = uv_warp_sel_t.numpy()
    z_sel = z_sel_t.numpy()

    uv_warp = np.full((size, 2), np.nan, dtype=np.float32)
    uv_warp[sel] = uv_warp_sel
    keep[sel] &= (z_sel > 0) & in_box(uv_warp_sel, box_c)

    # A full-grid warp of every candidate, for diagnostics only: the margins of
    # a cell the run excluded are still worth printing. Values at kept cells
    # are overwritten with the exact-chain ones above.
    points_all = unproject32(centers_t, torch.from_numpy(depth_at), K_tgt)
    uv_warp_all = project32(transform32(T_inv, points_all), K_ctx)[0].numpy()
    uv_warp_all[sel] = uv_warp_sel

    ids_universe = sample_ids(
        scene, ctx_id, tgt_id, torch.from_numpy(centers.astype(np.float64))
    )

    # Splat internals, weights, and the splat-side neighbour admissibility.
    weights, coverage = splat_internals(depth_ctx_t, K_ctx, K_tgt, T, (h, w))
    covis_cells = patch_fraction32(covisible_t)
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
    # sampled cells; the direction is one hash over the intersection. Excluded
    # cells carry NaN warps, which fail in_box, and are never consulted.
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

    baseline = float(torch.linalg.vector_norm(T[:3, 3]))
    covis_depths = depth_tgt[covisible & np.isfinite(depth_tgt) & (depth_tgt > 0)]
    parallax = baseline / float(np.median(covis_depths)) if covis_depths.size else float("nan")
    internals = {
        "per_point_mask": per_point_mask,
        "splat_mask": splat_mask,
        "covisible": covisible,
        "cell_of_sample": cell_of_sample,
        "chosen": chosen,
        "uv_warp_all": uv_warp_all,
        "uv_neighbor": uv_n_s,
        "centers": centers,
        "box_context": box_c,
        "option_ok": ok_s,
        "direction": direction,
        "size": size,
    }
    return rows, {
        "rotation_deg": ind.rotation_deg(ind.relative(tgt["T"], ctx["T"])[:3, :3]),
        "parallax": parallax,
        "covisible_fraction": float(covisible.mean()),
    }, internals


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def audit_scene(
    renders: Path,
    cache: Path,
    eval_dir: Path,
    scene: str,
    encoder: str,
    rel_tol: float,
    min_covisible_fraction: float,
    max_pairs: int | None = None,
) -> dict:
    """Reconstruct every audited pair and compare against the shipped parquet.

    Returns the summary dict; the caller decides what to do with the verdict.
    """
    frames = load_manifest(renders / scene / "manifest.json")
    depths = {
        fid: np.load(renders / scene / frame["depth_path"])
        for fid, frame in frames.items()
    }
    features = load_features(cache, encoder, scene)
    rows = [r for r in load_rows(eval_dir, scene) if r["encoder"] == encoder]

    record_path = eval_dir.parent / f"mean_vector_{encoder}.json"
    stored_record = json.loads(record_path.read_text(encoding="utf-8"))
    center = recompute_mean_vector(cache, encoder, stored_record["scenes"])
    stored_vector = np.load(eval_dir.parent / f"mean_vector_{encoder}.npy")
    mean_vector_diff = float(np.abs(center - stored_vector.astype(np.float32)).max())

    by_pair: dict[tuple, dict] = {}
    for row in rows:
        by_pair.setdefault(
            (row["context_frame_id"], row["target_frame_id"]), {}
        )[(row["path"], row["variant"])] = row

    pairs = sorted(by_pair)
    if max_pairs:
        pairs = pairs[:max_pairs]

    summary = {
        "scene": scene, "encoder": encoder, "pairs": len(pairs),
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
        mine, pair_fields, _ = reconstruct_pair(
            scene, ctx_id, tgt_id, frames, depths, features, center,
            rel_tol, min_covisible_fraction,
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
    return summary


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
    parser.add_argument(
        "--diagnose", nargs=2, metavar=("CONTEXT", "TARGET"), default=None,
        help="localize one pair's disagreement instead of auditing the scene: "
        "name the samples the two implementations select differently and show "
        "how close each sits to a decision boundary",
    )
    args = parser.parse_args()

    import yaml

    analysis = yaml.safe_load(Path("configs/analysis.yaml").read_text())
    rel_tol = analysis["covisible_relative_depth_tol"]
    min_cf = analysis["min_covisible_fraction"]

    if args.diagnose:
        frames = load_manifest(args.renders / args.scene / "manifest.json")
        depths = {
            fid: np.load(args.renders / args.scene / frame["depth_path"])
            for fid, frame in frames.items()
        }
        features = load_features(args.cache, args.encoder, args.scene)
        rows = [r for r in load_rows(args.eval_dir, args.scene)
                if r["encoder"] == args.encoder]
        record_path = args.eval_dir.parent / f"mean_vector_{args.encoder}.json"
        stored_record = json.loads(record_path.read_text(encoding="utf-8"))
        center = recompute_mean_vector(args.cache, args.encoder, stored_record["scenes"])
        by_pair: dict[tuple, dict] = {}
        for row in rows:
            by_pair.setdefault(
                (row["context_frame_id"], row["target_frame_id"]), {}
            )[(row["path"], row["variant"])] = row
        ctx_id, tgt_id = args.diagnose
        shipped = by_pair[(ctx_id, tgt_id)]
        _, _, internals = reconstruct_pair(
            args.scene, ctx_id, tgt_id, frames, depths, features, center,
            rel_tol, min_cf,
        )
        size = internals["size"]
        theirs_pp = np.unpackbits(
            np.frombuffer(bytes(shipped[("per_point", "Oracle-Transport")]["sample_mask"]),
                          dtype=np.uint8)
        )[:size].astype(bool)
        mine_pp = internals["per_point_mask"]
        only_mine = np.flatnonzero(mine_pp & ~theirs_pp)
        only_theirs = np.flatnonzero(theirs_pp & ~mine_pp)
        print(f"pair {ctx_id} -> {tgt_id}")
        print(f"  per-point selected: mine {int(mine_pp.sum())}, "
              f"pipeline {int(theirs_pp.sum())}")
        print(f"  only mine: {only_mine.tolist()}   only pipeline: {only_theirs.tolist()}")
        u_min, u_max, v_min, v_max = internals["box_context"]
        centers = internals["centers"]
        uv_warp = internals["uv_warp_all"]
        patches_w = frames[tgt_id]["width"] // PATCH
        for cell in list(only_mine) + list(only_theirs):
            row, col = divmod(int(cell), patches_w)
            centre = np.array([col * PATCH + (PATCH - 1) / 2.0,
                               row * PATCH + (PATCH - 1) / 2.0], dtype=np.float32)
            index = int(np.argmin(np.abs(centers - centre).sum(axis=1)))
            warp = uv_warp[index]
            margins = {
                "u-lo": float(warp[0] - u_min), "hi-u": float(u_max - warp[0]),
                "v-lo": float(warp[1] - v_min), "hi-v": float(v_max - warp[1]),
            }
            closest = min(margins, key=margins.get)
            side = "mine only" if cell in set(only_mine.tolist()) else "pipeline only"
            print(f"  cell {int(cell):>5} ({side}) centre={centre.tolist()} "
                  f"warp=({warp[0]:.6f}, {warp[1]:.6f})")
            print(f"        distance to the nearest sampling-box edge: "
                  f"{closest} = {margins[closest]:.3e} px")
        return

    summary = audit_scene(
        args.renders, args.cache, args.eval_dir, args.scene, args.encoder,
        rel_tol, min_cf, max_pairs=args.max_pairs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=1))
    print(f"verdict: {summary['verdict']} -> {args.out}")
    sys.exit(0 if summary["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
