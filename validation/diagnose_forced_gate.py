"""Diagnose one pair's PROTOCOL 4.5 forced-collision-order gate breach.

Rebuilds the exact gate inputs for one (pair, level) through the same
library functions evaluate_scene_phase4 uses, then decomposes the forced
residual mechanically:

- the pair's true baseline, from the manifest poses in float64;
- landing flips: common source pixels whose splat target pixel differs
  between the ground-truth arm and the aligned estimated arm, with each
  flipped pixel's distance to its floor(u + 0.5) cell boundary;
- lost winners: Oracle winner keys the forced arm failed to match, and
  whether every one of them is a flipped pixel. If a lost winner is NOT
  flipped, the key-membership machinery itself is defective, which is a
  different finding entirely;
- the per-cell score residual distribution over the gate cells, each
  breaching cell's winner count, and whether it was touched by a flip.

Diagnostic only. Changes nothing, gates nothing, tunes nothing.

Usage (Borah, repo root, any worktree state):
    PYTHONPATH=src python validation/diagnose_forced_gate.py \
        --config configs/phase4.yaml --scene apartment_0 \
        --context apartment_0_vp00_rotation_001 \
        --target apartment_0_vp00_rotation_008
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lot.analysis_config import load_analysis_config  # noqa: E402
from lot.evaluate import _SceneCache, pair_geometry  # noqa: E402
from lot.geometry import (  # noqa: E402
    pixel_grid, project, relative_pose, transform_points, unproject,
)
from lot.phase4 import (  # noqa: E402
    MULTIPLICATIVE_LEVELS,
    _per_sample_cosine,
    aligned_depth,
    frame_calibration,
    load_depth_archive,
    load_phase4_config,
    resample_depth_nearest,
    scene_scale_leave_target_out,
    splat_plan_detail,
    transport_prevalid,
)
from lot.render_replica import MANIFEST_NAME, load_manifest  # noqa: E402
from lot.transport import transport_plan  # noqa: E402


def landing_pixels(depth_map: torch.Tensor, K_ctx, K_tgt, T, out_hw):
    """Each source pixel's splat target pixel and its boundary margins."""
    height, width = depth_map.shape
    uv = pixel_grid(height, width, dtype=torch.float32)
    pts = unproject(uv, depth_map, K_ctx)
    uv_t, z_t = project(transform_points(T, pts), K_tgt)
    iu = torch.floor(uv_t[..., 0] + 0.5).long()
    iv = torch.floor(uv_t[..., 1] + 0.5).long()
    lin = (iv * out_hw[1] + iu).reshape(-1)
    # Distance of the continuous landing to the nearest rasterization
    # boundary, in px: frac in [0, 1) of (u + 0.5); min(frac, 1 - frac).
    fu = (uv_t[..., 0] + 0.5) - torch.floor(uv_t[..., 0] + 0.5)
    fv = (uv_t[..., 1] + 0.5) - torch.floor(uv_t[..., 1] + 0.5)
    margin = torch.minimum(
        torch.minimum(fu, 1 - fu), torch.minimum(fv, 1 - fv)
    ).reshape(-1)
    return lin, margin, uv_t.reshape(-1, 2), z_t.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--context", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    cfg = load_phase4_config(args.config)
    analysis = load_analysis_config()
    scene_root = cfg.renders_root / args.scene
    manifest = load_manifest(scene_root / MANIFEST_NAME)
    frames = {f.frame_id: f for f in manifest.frames}
    context, target = frames[args.context], frames[args.target]

    # The pair's true geometry, in float64 straight from the manifest.
    pos_c = context.T_world_from_camera[:3, 3].to(torch.float64)
    pos_t = target.T_world_from_camera[:3, 3].to(torch.float64)
    T64 = relative_pose(
        target.T_world_from_camera.to(torch.float64),
        context.T_world_from_camera.to(torch.float64),
    )
    T = T64.to(cfg.torch_dtype)
    print(f"pair {args.context} -> {args.target}")
    print(f"  camera position delta (float64): {float((pos_t - pos_c).norm()):.3e} m")
    print(f"  relative translation norm float64: {float(T64[:3, 3].norm()):.3e} m")
    print(f"  relative translation norm float32: {float(T[:3, 3].norm()):.3e} m")

    # Estimated maps and calibrations for the whole scene, the same way the
    # evaluator builds them. The convention is the recorded planar_z, under
    # which no conversion is applied; this diagnostic does not re-litigate it.
    depth_cache = load_depth_archive(cfg.cache_root, cfg.depth_encoder, args.scene)
    cache = _SceneCache(
        scene_root, cfg.cache_root, [cfg.feature_encoder], args.scene, manifest
    )
    est_maps: dict[str, np.ndarray] = {}
    calibrations = {}
    for frame in manifest.frames:
        raw = depth_cache["depth"][frame.frame_id]
        resampled, _ = resample_depth_nearest(raw, (frame.height, frame.width))
        conf_raw = depth_cache["conf"][frame.frame_id]
        conf = (
            resample_depth_nearest(conf_raw, (frame.height, frame.width))[0]
            if conf_raw is not None
            else None
        )
        prevalid = transport_prevalid(resampled, conf, analysis)
        est_maps[frame.frame_id] = np.where(
            prevalid, resampled, np.float32(np.nan)
        ).astype(np.float32)
        calibrations[frame.frame_id] = frame_calibration(
            est_maps[frame.frame_id], cache.depth(frame.depth_path).numpy(), prevalid
        )

    K_context = context.K.to(cfg.torch_dtype)
    K_target = target.K.to(cfg.torch_dtype)
    depth_context = cache.depth(context.depth_path).to(cfg.torch_dtype)
    depth_target = cache.depth(target.depth_path).to(cfg.torch_dtype)
    target_hw = tuple(depth_target.shape)

    geometry = pair_geometry(
        depth_context, depth_target, K_context, K_target, T,
        args.scene, args.context, args.target, analysis,
    )
    scene_scale, _ = scene_scale_leave_target_out(calibrations, args.target)
    context_calib = calibrations[args.context]
    print(f"  scene scale (leave-target-out): {scene_scale:.6f}")
    print(f"  context image scale: {context_calib.image_scale:.6f}")

    features_context = cache.features(cfg.feature_encoder, args.context)
    features_target = cache.features(cfg.feature_encoder, args.target)
    channels = features_context.shape[0]
    flat_context = features_context.to(torch.float32).reshape(channels, -1)
    flat_target = features_target.to(torch.float32).reshape(channels, -1)
    center = torch.from_numpy(
        np.load(cfg.mean_vector_dir / f"mean_vector_{cfg.feature_encoder}.npy")
    ).to(torch.float32)

    gt_detail = splat_plan_detail(depth_context, K_context, K_target, T, target_hw)
    lin_gt, margin_gt, _, _ = landing_pixels(
        depth_context, K_context, K_target, T, target_hw
    )
    n_cells = target_hw[0] * target_hw[1]
    out_patches_w = target_hw[1] // 14

    for level in list(MULTIPLICATIVE_LEVELS) + ["affine"]:
        context_aligned = aligned_depth(
            level, est_maps[args.context], scene_scale, context_calib
        )
        if context_aligned is None:
            print(f"[{level}] affine failed for the context image; skipped")
            continue
        est_t = torch.from_numpy(context_aligned).to(cfg.torch_dtype)
        est_detail = splat_plan_detail(est_t, K_context, K_target, T, target_hw)
        common = gt_detail.keep & est_detail.keep
        gt_common = splat_plan_detail(
            depth_context, K_context, K_target, T, target_hw, source_keep=common
        )
        forced_est = splat_plan_detail(
            est_t, K_context, K_target, T, target_hw,
            source_keep=common, forced_winner_keys=gt_common.winner_keys,
        )

        lin_est, margin_est, _, _ = landing_pixels(
            est_t, K_context, K_target, T, target_hw
        )
        common_np = common.numpy()
        flipped = (lin_gt != lin_est).numpy() & common_np
        n_flips = int(flipped.sum())

        lost_keys = np.setdiff1d(
            gt_common.winner_keys, forced_est.winner_keys, assume_unique=False
        )
        lost_sources = (lost_keys // n_cells).astype(np.int64)
        lost_flipped = (
            int(flipped[lost_sources].sum()) if lost_sources.size else 0
        )

        # The pre-A7 membership comparison, kept for localization: after
        # Amendment A7 the production gate freezes Oracle's full structure
        # and this arm survives only as the tax-decomposition midpoint.
        est_plan = transport_plan(est_t, K_context, K_target, T, target_hw)
        est_cov = (est_plan.coverage.reshape(-1) > 0).numpy()
        sp_scored = geometry.splat_covisible_ok & est_cov & geometry.splat_mask
        both_covered = (
            (gt_common.coverage.reshape(-1) > 0)
            & (forced_est.coverage.reshape(-1) > 0)
        ).numpy()
        gate_cells = np.flatnonzero(sp_scored & both_covered)
        pooled_oracle = flat_context @ gt_common.weights.mT
        pooled_forced = flat_context @ forced_est.weights.mT
        targets = flat_target[:, gate_cells].T
        a = pooled_oracle[:, gate_cells].T
        b = pooled_forced[:, gate_cells].T
        res_raw = (_per_sample_cosine(a, targets, None)
                   - _per_sample_cosine(b, targets, None)).abs().numpy()
        res_cen = (_per_sample_cosine(a, targets, center)
                   - _per_sample_cosine(b, targets, center)).abs().numpy()

        # Which cells a flip touches: the source's GT-arm and est-arm target
        # patches both change composition when the pixel flips or drops.
        flip_lin_gt = lin_gt.numpy()[flipped]
        flip_lin_est = lin_est.numpy()[flipped]

        def to_cell(lin_arr):
            return ((lin_arr // target_hw[1]) // 14) * out_patches_w + (
                (lin_arr % target_hw[1]) // 14
            )

        touched = set(to_cell(flip_lin_gt)) | set(to_cell(flip_lin_est))
        worst = np.argsort(res_cen)[::-1][: args.top]
        tol = analysis.rotation_gate_forced_tol
        print(
            f"[{level}] flips {n_flips} (max boundary margin of a flipped px: "
            + (f"{float(margin_gt.numpy()[flipped].max()):.3e} px" if n_flips else "n/a")
            + f"), lost winners {len(lost_keys)} "
            f"(flipped {lost_flipped}, NOT flipped {len(lost_keys) - lost_flipped})"
        )
        print(
            f"        gate cells {gate_cells.size}, max residual raw "
            f"{res_raw.max() if res_raw.size else 0:.3e} centered "
            f"{res_cen.max() if res_cen.size else 0:.3e}, cells over {tol:g}: "
            f"raw {int((res_raw > tol).sum())} centered {int((res_cen > tol).sum())}"
        )
        for rank in worst:
            if not res_cen.size or res_cen[rank] <= 0:
                break
            cell = int(gate_cells[rank])
            support = int((gt_common.weights[cell] > 0).sum())
            support_forced = int((forced_est.weights[cell] > 0).sum())
            print(
                f"        cell {cell:>5}: centered residual {res_cen[rank]:.3e}, "
                f"raw {res_raw[rank]:.3e}, source patches oracle {support} / "
                f"forced {support_forced}, touched by a flip: {cell in touched}"
            )


if __name__ == "__main__":
    main()
