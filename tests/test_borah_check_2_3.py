"""Validator 2.3 must reproduce the pipeline's masks bit-for-bit.

The 2.3 comparison contract checks persisted sample masks bit-for-bit, so the
validator's mask-deciding arithmetic has to land on the run's exact float32
decisions; a one-ulp drift at a sampling-box edge or a z-buffer tie selects a
different sample and the audit misreports its own rounding as pipeline
disagreement. These tests hold that property on the analytic scene twice over:
an evaluator parquet produced by the real pipeline is audited end to end, and
the mirrored geometry chain is compared against the pipeline's own bitwise,
which catches ulp drift even when no sample happens to sit on a boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))

import borah_check_2_3 as bc  # noqa: E402
from test_phase4 import SCENE, build_scene, run_phase3  # noqa: E402

from lot.analysis_config import load_analysis_config  # noqa: E402
from lot.evaluate import pair_geometry  # noqa: E402
from lot.geometry import relative_pose  # noqa: E402
from lot.render_replica import MANIFEST_NAME, load_manifest  # noqa: E402
from lot.visibility import visibility_masks  # noqa: E402


@pytest.fixture(scope="module")
def audited(tmp_path_factory):
    """The analytic scene with a parquet written by the real evaluator."""
    root = tmp_path_factory.mktemp("check23")
    build_scene(root)
    analysis = load_analysis_config()

    class Cfg:
        seed = 0

    eval_dir, _ = run_phase3(root, Cfg, analysis)
    return {"root": root, "eval_dir": eval_dir, "analysis": analysis}


def test_audit_reproduces_the_analytic_scene(audited):
    analysis = audited["analysis"]
    summary = bc.audit_scene(
        audited["root"], audited["root"] / "cache", audited["eval_dir"], SCENE,
        "dinov2_vitb14", analysis.covisible_relative_depth_tol,
        analysis.min_covisible_fraction,
    )
    assert summary["verdict"] == "PASS", summary["failures"]
    assert summary["pairs"] > 0
    assert summary["rows_compared"] == 10 * summary["pairs"]
    assert summary["mask_mismatches"] == 0
    assert summary["count_mismatches"] == 0
    assert max(summary["metric_max_abs_diff"].values()) <= bc.TOL


def test_mirror_matches_the_pipeline_bitwise(audited):
    """Selection, warps, and nulls agree to the last bit, pair by pair.

    Mask equality alone would pass while coordinates drift, as long as no
    sample sat near an edge; bitwise equality of the warp coordinates is the
    property that makes edge decisions reproducible on any input.
    """
    analysis = audited["analysis"]
    root = audited["root"]
    manifest = load_manifest(root / SCENE / MANIFEST_NAME)
    frames_lot = {f.frame_id: f for f in manifest.frames}
    frames_val = bc.load_manifest(root / SCENE / "manifest.json")
    depths = {
        fid: np.load(root / SCENE / rec["depth_path"])
        for fid, rec in frames_val.items()
    }
    features = bc.load_features(root / "cache", "dinov2_vitb14", SCENE)
    center = bc.recompute_mean_vector(root / "cache", "dinov2_vitb14", [SCENE])

    # Pairs the evaluator actually scored, from the shipped parquet, and the
    # run's declared cast: evaluate_scene moves every depth map to
    # geometry_dtype float32 on read, whatever dtype sits on disk.
    scored = sorted({
        (r["context_frame_id"], r["target_frame_id"])
        for r in bc.load_rows(audited["eval_dir"], SCENE)
    })
    assert len(scored) >= 3
    pairs = [scored[0], scored[len(scored) // 2], scored[-1]]

    def depth32(fid):
        return torch.from_numpy(depths[fid]).to(torch.float32)

    compared_samples = 0
    for ctx_id, tgt_id in pairs:
        ctx, tgt = frames_lot[ctx_id], frames_lot[tgt_id]
        T = relative_pose(
            tgt.T_world_from_camera, ctx.T_world_from_camera
        ).to(torch.float32)
        masks = visibility_masks(
            depth32(tgt_id), depth32(ctx_id),
            tgt.K.to(torch.float32), ctx.K.to(torch.float32), T,
            rel_tol=analysis.covisible_relative_depth_tol,
        )
        geometry = pair_geometry(
            depth32(ctx_id), depth32(tgt_id),
            ctx.K.to(torch.float32), tgt.K.to(torch.float32), T,
            SCENE, ctx_id, tgt_id, analysis,
        )
        _, _, internals = bc.reconstruct_pair(
            SCENE, ctx_id, tgt_id, frames_val, depths, features, center,
            analysis.covisible_relative_depth_tol, analysis.min_covisible_fraction,
        )
        assert np.array_equal(internals["covisible"], masks.covisible.numpy()), (
            f"{ctx_id}->{tgt_id}: covisible mask diverges"
        )
        assert np.array_equal(internals["per_point_mask"], geometry.per_point_mask), (
            f"{ctx_id}->{tgt_id}: per-point selection diverges"
        )
        assert np.array_equal(internals["splat_mask"], geometry.splat_mask), (
            f"{ctx_id}->{tgt_id}: splat selection diverges"
        )
        warp_pipeline = geometry.samples.uv_context_warp.numpy()
        warp_mine = internals["uv_warp_all"][internals["chosen"]]
        assert warp_pipeline.dtype == np.float32 and warp_mine.dtype == np.float32
        assert warp_pipeline.shape == warp_mine.shape
        assert np.array_equal(warp_pipeline, warp_mine), (
            f"{ctx_id}->{tgt_id}: warp coordinates differ; max ulp-level drift "
            f"{np.abs(warp_pipeline - warp_mine).max():.3e}"
        )
        assert np.array_equal(
            internals["option_ok"], geometry.samples.neighbor_option_ok
        ), f"{ctx_id}->{tgt_id}: admissible neighbour sets differ"
        neighbor_pipeline = geometry.samples.uv_context_neighbor.numpy()
        assert np.array_equal(neighbor_pipeline, internals["uv_neighbor"]), (
            f"{ctx_id}->{tgt_id}: Neighbor-Patch locations differ"
        )
        compared_samples += warp_mine.shape[0]
    assert compared_samples > 0, "every compared pair was empty; the test saw nothing"
