"""Phase 4's accepted image-level arms, recomputed and reconciled for Phase 5.

Phase 5 needs three things from the accepted Phase 4 run for every test pair:

    the TL-Reference predictions, which the formulation diagnostic compares
        against Context-Lift Transport-Only;
    the operational splat-pool scored cells and transported features, which
        the secondary comparison and the cross-path disclosure use;
    the ground-truth boundary and texture cell masks of Stream Z.

Phase 4 persisted the masks and the scores but not the per-sample predictions,
so the predictions have to be recomputed. A recomputation is only TL-Reference
if it is provably Phase 4's, and the proof used here is the one Phase 4 itself
used to inherit Phase 3: every recomputed scored set is compared bit for bit
against the mask Phase 4 persisted for that pair, and every recomputed score
against the score Phase 4 persisted, within the existing reconciliation bound.
A pair that fails either comparison stops the run. That is stronger than a
synthetic equivalence test, because it runs on every real pair.

The recomputation is a transcription of the image-level branch of
lot.phase4.evaluate_pair_phase4, through the same primitives with the same
arguments. Phase 4 is pinned and cannot be refactored to share it.
tests/test_phase5_reference.py fails if Phase 4 stops calling the primitives
this transcription assumes, so the two cannot drift in silence.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .analysis_config import AnalysisConfig
from .correspondence import _in_box, _sampling_box
from .encoders import PATCH_SIZE, sample_features_bilinear, sample_map_bilinear
from .evaluate import (
    PER_POINT,
    SPLAT_POOL,
    agreement_metrics,
    pair_geometry,
    read_run_metadata,
    unpack_mask,
)
from .geometry import invert_se3, project, transform_points, unproject
from .phase4 import (
    LEVELS,
    PHASE3_SCORE_RECON_TOL,
    POPULATION_MATCHED,
    aligned_depth,
)
from .transport import apply_transport_plan, transport_plan

# Phase 4 persisted every alignment level, so each Phase 5 condition reconciles
# against the Phase 4 arm at the same level: the primary image-scale run against
# VGGT-ImageScale, the affine sensitivity against VGGT-ImageAffine, the native
# diagnostic against VGGT-NoAlign. The scene-scale level is a Phase 4 diagnostic
# with no Phase 5 condition and is refused.
VARIANT_OF_LEVEL = dict(LEVELS)
PHASE5_LEVELS = ("image", "affine", "none")


def _variant_for(level: str) -> str:
    if level not in PHASE5_LEVELS:
        raise ValueError(
            f"level {level!r} is not a Phase 5 condition; use one of {PHASE5_LEVELS}"
        )
    return VARIANT_OF_LEVEL[level]


class ReferenceMismatch(RuntimeError):
    """A recomputed Phase 4 arm disagrees with what Phase 4 persisted."""


@dataclasses.dataclass
class Phase4PairRows:
    """What Phase 4 persisted for one pair at the reference level."""

    pp_mask: np.ndarray | None        # [cells] bool, TL per-point scored cells
    pp_cosine: float | None
    pp_cosine_centered: float | None
    sp_mask: np.ndarray | None        # [cells] bool, splat est scored cells
    sp_cosine: float | None
    sp_cosine_centered: float | None
    boundary_cells: np.ndarray | None  # [cells] bool, GT, PROTOCOL 4.8
    lowtex_cells: np.ndarray | None    # [cells] bool, GT and RGB, PROTOCOL 4.8


def read_phase4_reference(eval_dir: Path, scene: str, level: str) -> dict[str, Any]:
    """The accepted Phase 4 rows for one scene at one level, keyed by pair.

    Only the rows Phase 5 reconciles against are read: matched population, the
    given level and its variant, both paths. The run record travels with them
    so the caller can bind the identity it is reconciling against. Every pair
    Phase 4 scored at any level is listed too, so the caller can check that
    Phase 5 is evaluating the population Phase 4 evaluated.
    """
    import pyarrow.parquet as pq

    variant = _variant_for(level)
    path = Path(eval_dir) / f"{scene}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; the Phase 5 references are reconciled against "
            "the accepted Phase 4 run and cannot be produced without it"
        )
    meta = read_run_metadata(path)
    if meta is None:
        raise ReferenceMismatch(f"{path} carries no run record")
    size = int(meta["universe_size"])
    table = pq.read_table(
        path,
        columns=["context_frame_id", "target_frame_id", "path", "level",
                 "population", "variant", "sample_mask", "boundary_mask",
                 "lowtex_mask", "cosine_mean", "cosine_centered_mean"],
    )
    pairs: dict[tuple[str, str], Phase4PairRows] = {}
    every_pair: set[tuple[str, str]] = set()
    for row in table.to_pylist():
        key = (row["context_frame_id"], row["target_frame_id"])
        every_pair.add(key)
        slot = pairs.setdefault(
            key, Phase4PairRows(None, None, None, None, None, None, None, None)
        )
        # The ground-truth masks are identical on every row of a pair; read them
        # from whichever row arrives first.
        if slot.boundary_cells is None and row["boundary_mask"] is not None:
            slot.boundary_cells = unpack_mask(bytes(row["boundary_mask"]), size)
            slot.lowtex_cells = unpack_mask(bytes(row["lowtex_mask"]), size)
        if (
            row["level"] != level
            or row["population"] != POPULATION_MATCHED
            or row["variant"] != variant
        ):
            continue
        mask = unpack_mask(bytes(row["sample_mask"]), size)
        if row["path"] == PER_POINT:
            slot.pp_mask = mask
            slot.pp_cosine = row["cosine_mean"]
            slot.pp_cosine_centered = row["cosine_centered_mean"]
        elif row["path"] == SPLAT_POOL:
            slot.sp_mask = mask
            slot.sp_cosine = row["cosine_mean"]
            slot.sp_cosine_centered = row["cosine_centered_mean"]
    return {"pairs": pairs, "all_pairs": every_pair, "metadata": meta,
            "universe_size": size}


@dataclasses.dataclass
class ReferenceArms:
    """Phase 4's image-level arms for one pair, recomputed through its code."""

    geometry: Any                     # lot.evaluate.PairGeometry
    reads_target: Tensor              # [N_pp, C] target at each Phase 3 sample
    tl_reads: Tensor                  # [N_pp, C] TL-Reference prediction per sample
    tl_landed: np.ndarray             # [N_pp] bool, TL warp landed (Phase 4's 5d)
    pp_scored: np.ndarray             # [cells] bool
    sp_scored: np.ndarray             # [cells] bool
    transported_est: Tensor           # [C, cells] splat est arm
    flat_context: Tensor              # [C, cells] float32
    flat_target: Tensor               # [C, cells] float32
    # Where TL-Reference reads, for gate step 17. These are the tensors the
    # transcription already computes, exposed as they are; no value changes.
    tl_read_uv_context: Tensor        # [N_pp, 2] context pixels where tl_reads samples
    tl_read_depth_context: Tensor     # [N_pp] warped point's planar depth, context camera


def recompute_reference_arms(
    depth_context_gt: Tensor,
    depth_target_gt: Tensor,
    est_context: np.ndarray,
    est_target: np.ndarray,
    context_calibration: Any,
    features_context: Tensor,
    features_target: Tensor,
    K_context: Tensor,
    K_target: Tensor,
    T_target_from_context: Tensor,
    scene: str,
    context_frame_id: str,
    target_frame_id: str,
    analysis: AnalysisConfig,
    level: str,
    torch_dtype: torch.dtype = torch.float32,
) -> ReferenceArms | None:
    """Transcription of evaluate_pair_phase4's per-level branch.

    Every call below is the call Phase 4 makes, with the arguments Phase 4
    passes. Under Amendment A4 each level's transform is estimated from the
    context image and applied to whichever map is the transport input: the
    target map on the per-point path, which is what makes this the target-lift
    reference, and the context map on the splat path.

    Returns None when the level has no arm for this pair, which happens exactly
    when Phase 4 skipped it: an affine fit that failed.
    """
    _variant_for(level)
    geometry = pair_geometry(
        depth_context_gt, depth_target_gt, K_context, K_target,
        T_target_from_context, scene, context_frame_id, target_frame_id, analysis,
    )
    samples = geometry.samples
    per_point_cells = geometry.per_point_cells
    size = geometry.size
    channels = features_context.shape[0]
    target_hw = est_target.shape
    box_context = _sampling_box(est_context.shape, PATCH_SIZE)
    T_context_from_target = invert_se3(T_target_from_context)

    reads_target = sample_features_bilinear(features_target, samples.uv_target)

    target_aligned = aligned_depth(level, est_target, float("nan"), context_calibration)
    context_aligned = aligned_depth(level, est_context, float("nan"), context_calibration)
    if target_aligned is None or context_aligned is None:
        return None

    # Per-point path: the target-lift estimator, exactly as Phase 4 runs it.
    est_read = sample_map_bilinear(
        torch.from_numpy(target_aligned).to(torch_dtype), samples.uv_target
    )
    read_valid = torch.isfinite(est_read) & (est_read > 0)
    safe_read = torch.where(read_valid, est_read, torch.ones_like(est_read))
    points_target = unproject(samples.uv_target, safe_read, K_target)
    points_context = transform_points(T_context_from_target, points_target)
    uv_warp_est, z_est = project(points_context, K_context)
    landed = (read_valid & (z_est > 0) & _in_box(uv_warp_est, box_context)).numpy()
    pp_scored = np.zeros(size, dtype=bool)
    pp_scored[per_point_cells[landed]] = True

    # Splat path: the frozen operator on the aligned context map.
    est_plan = transport_plan(
        torch.from_numpy(context_aligned).to(torch_dtype),
        K_context, K_target, T_target_from_context, target_hw,
    )
    est_cov = (est_plan.coverage.reshape(-1) > 0).numpy()
    sp_scored = geometry.splat_covisible_ok & est_cov & geometry.splat_mask

    return ReferenceArms(
        geometry=geometry,
        reads_target=reads_target,
        tl_reads=sample_features_bilinear(features_context, uv_warp_est),
        tl_landed=landed,
        pp_scored=pp_scored,
        sp_scored=sp_scored,
        transported_est=apply_transport_plan(est_plan, features_context).reshape(
            channels, -1
        ),
        flat_context=features_context.to(torch.float32).reshape(channels, -1),
        flat_target=features_target.to(torch.float32).reshape(channels, -1),
        tl_read_uv_context=uv_warp_est,
        tl_read_depth_context=z_est,
    )


def reconcile_reference(
    arms: ReferenceArms,
    persisted: Phase4PairRows | None,
    center: Tensor,
    where: str,
    tol: float = PHASE3_SCORE_RECON_TOL,
) -> dict[str, float]:
    """Prove the recomputed arms are Phase 4's, or stop.

    Masks are compared bit for bit and scores within the reconciliation bound
    Phase 4 itself used to inherit Phase 3. A pair Phase 4 emitted no matched
    row for must recompute to an empty scored set: Phase 4 skips emission
    exactly when nothing was scored, so a non-empty recomputation there is a
    disagreement, not a gap.

    Returns the worst score residuals, which the run record carries.
    """
    worst = {"pp": 0.0, "sp": 0.0}
    for path, recomputed, mask, cosine, cosine_c, prediction, target in (
        ("per_point", arms.pp_scored,
         None if persisted is None else persisted.pp_mask,
         None if persisted is None else persisted.pp_cosine,
         None if persisted is None else persisted.pp_cosine_centered,
         arms.tl_reads[torch.from_numpy(arms.tl_landed)],
         arms.reads_target[torch.from_numpy(arms.tl_landed)]),
        ("splat_pool", arms.sp_scored,
         None if persisted is None else persisted.sp_mask,
         None if persisted is None else persisted.sp_cosine,
         None if persisted is None else persisted.sp_cosine_centered,
         arms.transported_est[:, torch.from_numpy(np.flatnonzero(arms.sp_scored))].T,
         arms.flat_target[:, torch.from_numpy(np.flatnonzero(arms.sp_scored))].T),
    ):
        if mask is None:
            if recomputed.any():
                raise ReferenceMismatch(
                    f"{where}: Phase 4 persisted no {path} row at the reference "
                    f"level, but the recomputation scores {int(recomputed.sum())} "
                    "cells; the recomputation is not Phase 4's"
                )
            continue
        if not np.array_equal(recomputed, mask):
            raise ReferenceMismatch(
                f"{where}: recomputed {path} scored set differs from Phase 4's "
                f"in {int((recomputed != mask).sum())} cells; the recomputation "
                "is not Phase 4's and cannot stand in for TL-Reference"
            )
        metrics = agreement_metrics(prediction, target, center, True)
        residual = max(
            abs(metrics["cosine_mean"] - cosine),
            abs(metrics["cosine_centered_mean"] - cosine_c),
        )
        if not residual <= tol:
            raise ReferenceMismatch(
                f"{where}: recomputed {path} score differs from Phase 4's by "
                f"{residual:.3e}, over the reconciliation bound {tol:g}"
            )
        worst["pp" if path == "per_point" else "sp"] = residual
    return worst


def region_cells(
    depth_target_gt: Tensor,
    rgb_target: np.ndarray,
    analysis: AnalysisConfig,
    persisted: Phase4PairRows | None,
    where: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream Z's ground-truth boundary and low-texture cells for one target.

    Computed with Phase 4's own functions, from ground-truth depth and rendered
    RGB only, so estimated depth never defines a category (PROTOCOL 4.9). When
    Phase 4 persisted the masks for this pair they must agree bit for bit.
    They are recomputed rather than only read because a pair Phase 4 could not
    score has no rows to read, while Context-Lift Transport-Only's own support,
    which is defined from the context side, can still need them.
    """
    from .phase4 import boundary_cells_from_mask, depth_boundary_mask, low_texture_cells

    boundary = boundary_cells_from_mask(depth_boundary_mask(depth_target_gt, analysis))
    lowtex = low_texture_cells(rgb_target, analysis)
    if persisted is not None and persisted.boundary_cells is not None:
        for name, mine, theirs in (
            ("boundary", boundary, persisted.boundary_cells),
            ("low-texture", lowtex, persisted.lowtex_cells),
        ):
            if not np.array_equal(mine, theirs):
                raise ReferenceMismatch(
                    f"{where}: recomputed {name} cells differ from Phase 4's in "
                    f"{int((mine != theirs).sum())} cells"
                )
    return boundary, lowtex
