"""The seventeen-step Phase 5 Borah integration gate, assembled and driven.

Separated from lot.phase5_check, which holds the reusable checks, so the order
of the gate reads as one list in one place. Nothing here trains. Step 12 runs a
single forward pass, the frozen loss, and a single backward pass on a real batch
to prove the pipeline is finite end to end, then discards the gradients; step 16
asserts no checkpoint appeared.

Step ids are stable, so the order of the list is not numeric. Step 17, the
pure-rotation gate across the whole regime, was added after the gate first
passed. It runs after step 14 and before step 15. The pin is written only
after every substantive check has passed, so a gate that stops at step 17
leaves the earlier pin in place. The verdict step stays last.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .phase5_check import (
    FROZEN_DESIGN_MISMATCH,
    IMPLEMENTATION_BUG,
    MISSING_ARTIFACT,
    GateReport,
    GateStop,
    assert_no_checkpoint_written,
    write_once,
    assert_no_forbidden_fields,
    check_folds_against_inventory,
    check_schema,
    check_test_seal,
    describe_tensor,
    environment_identity,
    aligned_depth_digest,
    verify_mean_vector_identity,
    verify_scene_identities,
    model_visible_fields,
    resource_probe,
    run_steps,
    splat_symmetry_evidence,
    step2_resolve_artifacts,
)

# Three scenes from three different families, so a family-specific problem
# cannot hide behind one that happens to work.
PROBE_SCENES = ("apartment_0", "office_0", "room_0")
REGIMES = ("rotation", "translation", "orbit")


def rotation_gate_evidence(
    lift: Any,
    analytic: torch.Tensor,
    substituted: Any,
    tol_px: float,
) -> dict[str, Any]:
    """Step 10's pure-rotation check on one pair, over the landed samples.

    Under zero translation, context-lift must reproduce the analytic homography
    for any depth, and replacing the depth map must not move a landing. Both
    are read on landed samples only. An unlanded ray can sit nearly parallel to
    the target image plane, where its projection runs to tens of thousands of
    pixels and float32 rounding alone exceeds a tolerance set at the 518 px
    frame size. On a 518 px frame with a 90 degree field of view, the residual
    over every context patch passes 1e-3 px from about 45 degrees of rotation,
    while landed samples stay near 6e-5 px. No score reads an unlanded landing,
    so comparing one tests float32 range rather than the mapping.

    lift: context-lift from the aligned depth. analytic: the homography's
    landing for every context patch. substituted: context-lift with the depth
    map replaced. A NaN residual fails, because NaN never satisfies the bound.
    """
    landed = lift.landed
    if not bool(landed.any()):
        raise GateStop(
            "10", "no sample of the rotation pair landed, so the rotation check "
            "has nothing to compare", FROZEN_DESIGN_MISMATCH,
            {"n_context_patches": int(landed.numel())},
        )
    residual = float((lift.uv_target[landed] - analytic[landed]).abs().max())
    depth_free = float(
        (lift.uv_target[landed] - substituted.uv_target[landed]).abs().max()
    )
    evidence = {
        "n_landed_samples": int(landed.sum()),
        "n_context_patches": int(landed.numel()),
        "max_homography_residual_px": residual,
        "tolerance_px": tol_px,
        "max_depth_substitution_shift_px": depth_free,
    }
    if not residual <= tol_px:
        raise GateStop(
            "10", "context-lift disagrees with the analytic rotational homography "
            "on real data", IMPLEMENTATION_BUG, {"rotation": evidence},
        )
    if not depth_free <= tol_px:
        raise GateStop(
            "10", "a pure-rotation landing moved when the depth map was replaced, "
            "so the mapping is not depth free", IMPLEMENTATION_BUG,
            {"rotation": evidence},
        )
    return evidence


def formulation_support_evidence(
    lift: Any,
    support: torch.Tensor,
    arms: Any,
    persisted: Any,
    center: torch.Tensor,
    features_context: torch.Tensor,
    target_hw: tuple[int, int],
    where: str,
) -> dict[str, Any]:
    """Step 8 on one pair: V_form on the target patch cell, built as evaluate builds it.

    Only supported samples are mapped to cells. A sample that did not land has
    a projected coordinate outside the target grid, and patch_cell_index does
    not bound-check it: a patch just past the right edge wraps into the next
    row's first cell and looks valid, and only the last row's overflow leaves
    the grid. The first version of this step mapped every context patch, so the
    first real run stopped on cell 1369 of a 37 by 37 grid. No score was
    affected, because every scoring path masks to the support before it reads a
    cell.

    arms is TL-Reference recomputed through Phase 4's code by the caller, and
    persisted is the accepted Phase 4 row for the pair. They are reconciled
    here, then the two cell sets are intersected by formulation_cells and
    scored by score_formulation, the functions evaluate runs.
    score_formulation refuses a target cell holding more than one TL-Reference
    sample, which is the ambiguity this step's specification rules out.
    """
    from .encoders import PATCH_SIZE, patch_cell_index
    from .phase5_reference import ReferenceMismatch, reconcile_reference
    from .phase5_score import formulation_cells, score_formulation

    n_cells = (target_hw[0] // PATCH_SIZE) * (target_hw[1] // PATCH_SIZE)
    cells = patch_cell_index(lift.uv_target[support], target_hw, PATCH_SIZE)
    outside = sorted({int(c) for c in cells if not 0 <= int(c) < n_cells})
    if outside:
        raise GateStop(
            "8", f"supported samples map outside the target grid: {outside[:10]}",
            IMPLEMENTATION_BUG, {"pair": where, "n_cells": n_cells},
        )
    if arms is None:
        raise GateStop(
            "8", "TL-Reference has no arm at the primary level on a pair "
            "Context-Lift supports", IMPLEMENTATION_BUG, {"pair": where},
        )
    center = center.to(torch.float32)
    try:
        residual = reconcile_reference(arms, persisted, center, where)
    except ReferenceMismatch as error:
        raise GateStop(
            "8", "TL-Reference recomputed on a real pair disagrees with the accepted "
            f"Phase 4 row: {error}", IMPLEMENTATION_BUG, {"pair": where},
        ) from error
    common, cl_mask = formulation_cells(lift, support, arms.pp_scored, target_hw)
    try:
        scores = score_formulation(
            lift, support, features_context, center, arms.pp_scored,
            arms.geometry.per_point_cells, arms.tl_reads, arms.reads_target,
            arms.geometry.samples.uv_target, target_hw,
        )
    except ValueError as error:
        raise GateStop(
            "8", f"the formulation intersection is not uniquely defined: {error}",
            IMPLEMENTATION_BUG, {"pair": where},
        ) from error
    tl_cells = int(np.asarray(arms.pp_scored).sum())
    unique, counts = np.unique(cells, return_counts=True)
    if common.size == 0:
        raise GateStop(
            "8", "the formulation support is empty on a real translation pair",
            FROZEN_DESIGN_MISMATCH,
            {"pair": where, "cl_cells": int(unique.size), "tl_cells": tl_cells},
        )
    return {
        "pair": where,
        "cl_supported_samples": int(support.sum()),
        "cl_distinct_target_cells": int(unique.size),
        "max_cl_samples_per_cell": int(counts.max()),
        "tl_scored_cells": tl_cells,
        "common_cells": int(common.size),
        "cl_samples_in_common_cells": int(cl_mask.sum()),
        "tl_samples_per_common_cell": 1,
        "reconciliation_worst_residual": residual,
        "tl_form_centered": scores.tl_form_centered,
        "cl_form_centered": scores.cl_form_centered,
        "target_grid_cells": n_cells,
        "note": (
            "many context patches may land in one target cell; that is expected "
            "and is not ambiguity. Each TL-Reference sample owns one cell, and "
            "every supported sample maps to a cell inside the grid."
        ),
    }


# ---------------------------------------------------------------------------
# Step 17: the pure-rotation gate across the whole rotation regime
# ---------------------------------------------------------------------------
#
# reporting_rules.md section 7, first bullet, which carries specification
# step 6 from step 10's single probe pair to every rotation pair of every
# scene, with TL-Reference added. "Common-valid" is read as the rules read it:
# each estimator is compared with the homography on its own landed samples.
# The two estimators index different samples, so they agree with each other
# exactly when both agree with the homography.

# The constant depth that replaces the aligned map. Any positive value serves.
# Step 10 substitutes the same value.
ROTATION_SUBSTITUTE_DEPTH_M = 7.0
ROTATION_COMPARISONS = ("context_lift", "depth_substitution", "tl_reference")
ROTATION_READING = (
    "each estimator is compared with the analytic rotational homography on its "
    "own landed samples; per-sample limit is the frozen coordinate tolerance "
    "plus focal length times the pair's translation over the sample's depth in "
    "the receiving camera"
)


def rotation_translation_norm(
    T_target_from_context: torch.Tensor, position_bound_m: float, where: str
) -> float:
    """The pair's recorded camera translation in meters, or a stop above the bound.

    The frozen rotation-position bound allows a rotation pair this much
    translation. A pair above it is not a pure rotation under the frozen
    design, so the stop is a design mismatch, not a bug. A NaN norm stops too.
    """
    from .geometry import baseline_m

    norm = float(baseline_m(T_target_from_context.to(torch.float64)))
    if not norm <= position_bound_m:
        raise GateStop(
            "17", f"{where}: the recorded camera translation {norm:.3e} m exceeds "
            f"the frozen rotation-position bound {position_bound_m:g} m, so the "
            "pair is not a pure rotation under the frozen design",
            FROZEN_DESIGN_MISMATCH,
            {"pair": where, "translation_norm_m": norm,
             "rotation_position_bound_m": position_bound_m},
        )
    return norm


def _focal_px(K: torch.Tensor) -> float:
    """The receiving camera's focal length in pixels.

    The larger of the two, so the allowance never shrinks on a camera whose
    focal lengths differ. Replica renders have square pixels.
    """
    return float(torch.maximum(K[0, 0], K[1, 1]))


def _translation_allowance_px(
    K_receiving: torch.Tensor, translation_m: float, depth_receiving: torch.Tensor
) -> torch.Tensor:
    """Per sample, focal length times translation over depth in the receiving camera.

    This is the first-order displacement a translation adds to a landing on
    the optical axis. Off the axis the exact shift can exceed it, by up to the
    secant of the ray's angle, so by up to 1.41 at the edge of a 90 degree
    frame. The frozen tolerance absorbs that excess while the allowance stays
    below about 2.4e-3 px. At the frozen position bound on a 518 px frame, that
    holds for depths above about 0.11 m. A nearer landed sample on a pair at
    the bound could exceed its limit. That stop is reported, not widened.
    """
    depth = depth_receiving.detach().to(torch.float64).cpu()
    return _focal_px(K_receiving) * translation_m / depth


def _rotation_comparison(
    name: str,
    label: str,
    estimate: torch.Tensor,
    reference: torch.Tensor,
    allowance: torch.Tensor,
    compared: torch.Tensor,
    tol_px: float,
    translation_norm_m: float,
    where: str,
) -> dict[str, Any]:
    """One estimator against its homography reference, on its compared samples.

    estimate, reference: [N, 2] pixel coordinates in the receiving image.
    allowance: [N] translation allowance in pixels, beyond tol_px.
    compared: [N] bool, the estimator's own landed samples.
    The residual of a sample is its larger coordinate difference. Any residual
    above its limit stops with the worst sample, ranked by how far it is over.
    A NaN residual or limit never satisfies the bound and ranks worst.
    """
    index = torch.nonzero(compared.detach().cpu(), as_tuple=False).reshape(-1)
    if index.numel() == 0:
        return {"n_compared": 0, "max_residual_px": None,
                "max_translation_allowance_px": None, "min_headroom_px": None}
    est = estimate.detach().cpu().to(torch.float64)[index]
    ref = reference.detach().cpu().to(torch.float64)[index]
    residual = (est - ref).abs().amax(dim=-1)
    extra = allowance[index]
    limit = tol_px + extra
    headroom = limit - residual
    ranked = torch.nan_to_num(headroom, nan=-math.inf)
    worst = int(torch.argmin(ranked))
    within = residual <= limit
    if not bool(within.all()):
        sample = int(index[worst])
        raise GateStop(
            "17", f"{where}: {label}, beyond its allowance; worst sample {sample}: "
            f"residual {float(residual[worst]):.3e} px, limit "
            f"{float(limit[worst]):.3e} px",
            IMPLEMENTATION_BUG,
            {
                "pair": where,
                "comparison": name,
                "translation_norm_m": translation_norm_m,
                "tolerance_px": tol_px,
                "n_compared": int(index.numel()),
                "n_over_limit": int((~within).sum()),
                "worst_sample": {
                    "index": sample,
                    "residual_px": float(residual[worst]),
                    "limit_px": float(limit[worst]),
                    "translation_allowance_px": float(extra[worst]),
                    "estimate_uv": [float(v) for v in est[worst]],
                    "reference_uv": [float(v) for v in ref[worst]],
                },
            },
        )
    return {
        "n_compared": int(index.numel()),
        "max_residual_px": float(residual.max()),
        "max_translation_allowance_px": float(extra.max()),
        "min_headroom_px": float(headroom.min()),
    }


def rotation_pair_evidence(
    lift: Any,
    substituted: Any,
    tl_read_uv_context: torch.Tensor,
    tl_read_depth_context: torch.Tensor,
    tl_landed: Any,
    uv_target_samples: torch.Tensor,
    K_context: torch.Tensor,
    K_target: torch.Tensor,
    T_target_from_context: torch.Tensor,
    context_hw: tuple[int, int],
    tol_px: float,
    position_bound_m: float,
    where: str,
) -> dict[str, Any]:
    """Step 17 on one rotation pair: three comparisons, each on its own landed samples.

    lift: Context-Lift from the aligned context depth. substituted: Context-Lift
    with that depth map replaced by a constant. Both carry target pixel
    landings for every context patch center.
    tl_read_uv_context, tl_read_depth_context, tl_landed: where TL-Reference
    reads in the context image, the warped point's depth in the context camera,
    and its landed flag, per Phase 3 target sample.
    uv_target_samples: the Phase 3 target sample coordinates, target pixels.
    K_context, K_target, T_target_from_context: the pair's cameras, OpenCV.

    The comparisons, in order:
    Context-Lift landings against rotation_homography_landing, on its landed
    samples, receiving camera target;
    Context-Lift landings against the substituted ones, on the same samples,
    with the allowance for both depths summed;
    TL-Reference read locations against the inverse homography applied to the
    Phase 3 target samples, on its landed samples, receiving camera context.

    The references are computed in float64 from the same cameras, so a residual
    measures the estimator alone. The translation bound is checked first.
    Returns counts and worst residuals. A pair where an estimator landed
    nothing records zero compared samples for it, and is checked only when
    both estimators compared at least one sample.
    """
    from .context_lift import rotation_homography_landing
    from .geometry import apply_homography, rotation_homography

    translation = rotation_translation_norm(T_target_from_context, position_bound_m, where)

    cl_landed = torch.as_tensor(lift.landed).detach().cpu().to(torch.bool)
    tl_mask = torch.as_tensor(np.asarray(tl_landed, dtype=bool))
    n_tl = int(tl_mask.numel())
    shapes_agree = (
        tuple(substituted.uv_target.shape) == tuple(lift.uv_target.shape)
        and tuple(tl_read_uv_context.shape) == (n_tl, 2)
        and tuple(tl_read_depth_context.shape) == (n_tl,)
        and tuple(uv_target_samples.shape) == (n_tl, 2)
    )
    if not shapes_agree:
        raise GateStop(
            "17", f"{where}: the rotation check's inputs disagree on sample count",
            IMPLEMENTATION_BUG,
            {"pair": where, "lift": list(lift.uv_target.shape),
             "substituted": list(substituted.uv_target.shape),
             "tl_read_uv_context": list(tl_read_uv_context.shape),
             "tl_read_depth_context": list(tl_read_depth_context.shape),
             "tl_landed": n_tl, "uv_target_samples": list(uv_target_samples.shape)},
        )

    T64 = T_target_from_context.detach().cpu().to(torch.float64)
    K_c64 = K_context.detach().cpu().to(torch.float64)
    K_t64 = K_target.detach().cpu().to(torch.float64)
    analytic = rotation_homography_landing(
        K_c64, K_t64, T64, context_hw, dtype=torch.float64
    )
    if tuple(analytic.shape) != tuple(lift.uv_target.shape):
        raise GateStop(
            "17", f"{where}: the homography's context grid does not match "
            "Context-Lift's", IMPLEMENTATION_BUG,
            {"pair": where, "homography": list(analytic.shape),
             "lift": list(lift.uv_target.shape)},
        )
    # Target pixels to context pixels: the homography of the inverse rotation.
    backward = rotation_homography(K_t64, K_c64, T64[:3, :3].T)
    tl_analytic = apply_homography(
        backward, uv_target_samples.detach().cpu().to(torch.float64)
    )

    cl_allowance = _translation_allowance_px(K_target, translation, lift.z_target)
    substitution_allowance = cl_allowance + _translation_allowance_px(
        K_target, translation, substituted.z_target
    )
    tl_allowance = _translation_allowance_px(
        K_context, translation, tl_read_depth_context
    )

    evidence: dict[str, Any] = {
        "pair": where,
        "translation_norm_m": translation,
        "n_context_patches": int(cl_landed.numel()),
        "n_tl_samples": n_tl,
    }
    evidence["context_lift"] = _rotation_comparison(
        "context_lift",
        "Context-Lift disagrees with the analytic rotational homography",
        lift.uv_target, analytic, cl_allowance, cl_landed,
        tol_px, translation, where,
    )
    evidence["depth_substitution"] = _rotation_comparison(
        "depth_substitution",
        "a Context-Lift landing moved when the depth map was replaced, so the "
        "mapping is not depth free",
        lift.uv_target, substituted.uv_target, substitution_allowance, cl_landed,
        tol_px, translation, where,
    )
    evidence["tl_reference"] = _rotation_comparison(
        "tl_reference",
        "a TL-Reference read location disagrees with the inverse rotational "
        "homography at its Phase 3 target sample",
        tl_read_uv_context, tl_analytic, tl_allowance, tl_mask,
        tol_px, translation, where,
    )
    evidence["checked"] = (
        evidence["context_lift"]["n_compared"] > 0
        and evidence["tl_reference"]["n_compared"] > 0
    )
    return evidence


def _worst_entry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge worst-residual entries: the largest residual, the smallest headroom."""
    measured = [e for e in entries if e["max_residual_px"] is not None]
    if not measured:
        return {"pair": None, "max_residual_px": None,
                "min_headroom_px": None, "min_headroom_pair": None}
    loudest = max(measured, key=lambda e: e["max_residual_px"])
    tightest = min(measured, key=lambda e: e["min_headroom_px"])
    return {
        "pair": loudest["pair"],
        "max_residual_px": loudest["max_residual_px"],
        "min_headroom_px": tightest["min_headroom_px"],
        "min_headroom_pair": tightest["min_headroom_pair"],
    }


def _largest_translation(entries: list[tuple[float | None, str | None]]) -> tuple[Any, Any]:
    measured = [e for e in entries if e[0] is not None]
    if not measured:
        return None, None
    return max(measured, key=lambda e: e[0])


def rotation_scene_summary(
    scene: str, pairs: list[dict[str, Any]], no_arm: list[str]
) -> dict[str, Any]:
    """Step 17's account of one scene. Every rotation pair is counted once.

    pairs: rotation_pair_evidence for each rotation pair with an arm at the
    primary level. no_arm: the named pairs without one.
    A scene with rotation pairs of which none could be checked stops, because
    the regime-wide check would then say nothing about that scene.
    """
    def lands(evidence: dict[str, Any], name: str) -> bool:
        return evidence[name]["n_compared"] > 0

    n_rotation = len(pairs) + len(no_arm)
    counts = {
        "n_rotation_pairs": n_rotation,
        "n_pairs_checked": sum(1 for e in pairs if e["checked"]),
        "n_pairs_context_lift_only": sum(
            1 for e in pairs if lands(e, "context_lift") and not lands(e, "tl_reference")
        ),
        "n_pairs_tl_reference_only": sum(
            1 for e in pairs if lands(e, "tl_reference") and not lands(e, "context_lift")
        ),
        "n_pairs_nothing_landed": sum(
            1 for e in pairs
            if not lands(e, "context_lift") and not lands(e, "tl_reference")
        ),
        "n_pairs_no_arm": len(no_arm),
    }
    if n_rotation and counts["n_pairs_checked"] == 0:
        raise GateStop(
            "17", f"{scene} has {n_rotation} rotation pairs and none could be "
            "checked: no pair had landed samples for both Context-Lift and "
            "TL-Reference at the primary level",
            FROZEN_DESIGN_MISMATCH,
            {"scene": scene, **counts, "no_arm_pairs": list(no_arm)},
        )
    worst = {
        name: _worst_entry([
            {"pair": e["pair"],
             "max_residual_px": e[name]["max_residual_px"],
             "min_headroom_px": e[name]["min_headroom_px"],
             "min_headroom_pair": e["pair"]}
            for e in pairs
        ])
        for name in ROTATION_COMPARISONS
    }
    norm, at = _largest_translation([(e["translation_norm_m"], e["pair"]) for e in pairs])
    return {
        **counts,
        "worst": worst,
        "max_translation_norm_m": norm,
        "max_translation_pair": at,
        "no_arm_pairs": list(no_arm),
    }


def rotation_regime_summary(
    per_scene: dict[str, dict[str, Any]], tol_px: float, position_bound_m: float
) -> dict[str, Any]:
    """Step 17's verdict over every scene: the overall worst and the largest translation.

    A regime with no rotation pair at all stops, because the check the
    specification asks for would then have run on nothing.
    """
    total = sum(s["n_rotation_pairs"] for s in per_scene.values())
    if total == 0:
        raise GateStop(
            "17", "no scene has a rotation-regime pair at the primary level, so "
            "the pure-rotation gate across the regime has nothing to check",
            FROZEN_DESIGN_MISMATCH, {"scenes": sorted(per_scene)},
        )
    count_keys = (
        "n_pairs_checked", "n_pairs_context_lift_only", "n_pairs_tl_reference_only",
        "n_pairs_nothing_landed", "n_pairs_no_arm",
    )
    norm, at = _largest_translation([
        (s["max_translation_norm_m"], s["max_translation_pair"])
        for s in per_scene.values()
    ])
    return {
        "n_scenes": len(per_scene),
        "n_rotation_pairs": total,
        **{key: sum(s[key] for s in per_scene.values()) for key in count_keys},
        "worst": {
            name: _worst_entry([s["worst"][name] for s in per_scene.values()])
            for name in ROTATION_COMPARISONS
        },
        "max_translation_norm_m": norm,
        "max_translation_pair": at,
        "tolerance_px": tol_px,
        "rotation_position_bound_m": position_bound_m,
        "reading": ROTATION_READING,
        "scenes": per_scene,
    }


def rotation_scene_evidence(cfg: Any, analysis: Any, inputs: Any) -> dict[str, Any]:
    """Step 17 on one scene: every rotation pair, at the primary level.

    Context-Lift is built as evaluate builds it. TL-Reference is recomputed by
    the call evaluate makes, so the gate checks the estimators that are scored.
    Nothing is reconciled with Phase 4 here: evaluate does that on every pair,
    and step 8 does it on one before anything trains.
    """
    from .context_lift import context_lift_map
    from .phase5 import aligned_context_depth, pair_cameras, phase5_scene_pairs
    from .phase5_reference import recompute_reference_arms

    scene = inputs.scene
    level = cfg.primary_alignment_level
    dtype = cfg.torch_dtype
    pairs = [p for p in phase5_scene_pairs(cfg, analysis, scene) if p.regime == "rotation"]
    evidence: list[dict[str, Any]] = []
    no_arm: list[str] = []
    for pair in pairs:
        ctx, tgt = pair.context_frame_id, pair.target_frame_id
        where = f"{scene} {ctx} -> {tgt} level {level}"
        context_depth = aligned_context_depth(inputs, ctx, level)
        if context_depth is None:
            no_arm.append(where)
            continue
        cams = pair_cameras(cfg, inputs, pair)
        depth = torch.from_numpy(context_depth).to(dtype)
        lift = context_lift_map(
            depth, cams.K_context, cams.K_target, cams.T_target_from_context,
            cams.context_hw, cams.target_hw,
        )
        substituted = context_lift_map(
            torch.full_like(depth, ROTATION_SUBSTITUTE_DEPTH_M),
            cams.K_context, cams.K_target, cams.T_target_from_context,
            cams.context_hw, cams.target_hw,
        )
        arms = recompute_reference_arms(
            inputs.cache.depth(cams.context.depth_path).to(dtype),
            inputs.cache.depth(cams.target.depth_path).to(dtype),
            inputs.est_maps[ctx], inputs.est_maps[tgt], inputs.calibrations[ctx],
            inputs.cache.features(cfg.feature_encoder, ctx),
            inputs.cache.features(cfg.feature_encoder, tgt),
            cams.K_context, cams.K_target, cams.T_target_from_context,
            scene, ctx, tgt, analysis, level, dtype,
        )
        if arms is None:
            raise GateStop(
                "17", f"{where}: Context-Lift has an arm at the primary level and "
                "TL-Reference has none", IMPLEMENTATION_BUG, {"pair": where},
            )
        evidence.append(rotation_pair_evidence(
            lift=lift,
            substituted=substituted,
            tl_read_uv_context=arms.tl_read_uv_context,
            tl_read_depth_context=arms.tl_read_depth_context,
            tl_landed=arms.tl_landed,
            uv_target_samples=arms.geometry.samples.uv_target,
            K_context=cams.K_context,
            K_target=cams.K_target,
            T_target_from_context=cams.T_target_from_context,
            context_hw=cams.context_hw,
            tol_px=analysis.rotation_gate_coord_tol_px,
            position_bound_m=analysis.rotation_position_bound_m,
            where=where,
        ))
    return rotation_scene_summary(scene, evidence, no_arm)


def run_integration_gate(cfg: Any, analysis: Any) -> GateReport:
    from .analysis_config import AnalysisConfig  # noqa: F401  (typing clarity)
    from .context_lift import context_lift_map, context_lift_support, rotation_homography_landing
    from .encoders import PATCH_SIZE, patch_cell_index, pixel_to_patch_coords
    from .evaluate import git_commit
    from .geometry import project, relative_pose, transform_points, unproject
    from .phase5 import (
        build_example,
        build_scene_inputs,
        load_convention_record,
        phase5_mean_vector,
        phase5_scene_pairs,
        predictor_config_from,
    )
    from .phase5_folds import REPLICA_SCENES, fold_digest, frozen_folds
    from .phase5_score import primary_support, score_primary
    from .predictors import build_predictor
    from .train import batch_loss

    state: dict[str, Any] = {}
    run_dir = Path(cfg.run_dir)
    checkpoints_before = set(run_dir.glob("**/*.pt")) if run_dir.exists() else set()

    def lift_for(example) -> tuple[Any, Any, Any, Any]:
        """Rebuild the forward map for an example, with its frames and pose."""
        inputs = state["probe_inputs"]
        dtype = cfg.torch_dtype
        context = inputs.frames[example.context_frame_id]
        target = inputs.frames[example.target_frame_id]
        T = relative_pose(
            target.T_world_from_camera, context.T_world_from_camera
        ).to(dtype)
        lift = context_lift_map(
            example.depth_context_aligned,
            context.K.to(dtype), target.K.to(dtype), T,
            (context.height, context.width), (target.height, target.width),
        )
        return lift, context, target, T

    def step1() -> dict[str, Any]:
        return {
            "commit": git_commit(),
            "config_digest": cfg.digest(),
            "fold_digest": fold_digest(frozen_folds()),
            "measurement_digest": analysis.measurement_digest(),
            "primary_alignment_level": cfg.primary_alignment_level,
            "seeds": list(cfg.training.get("seeds", (0, 1, 2))),
            "environment": environment_identity(),
        }

    def step2() -> dict[str, Any]:
        result = step2_resolve_artifacts(cfg)
        state["artifacts"] = result.evidence["artifacts"]
        # Every scene the folds name, compared against the accepted identity,
        # not a probe subset merely recorded.
        state["scene_identities"] = verify_scene_identities(cfg, REPLICA_SCENES, "2")
        return {**result.evidence, "scene_identities": state["scene_identities"]}

    def step3() -> dict[str, Any]:
        convention = load_convention_record(cfg)
        state["convention"] = convention
        inputs = build_scene_inputs(cfg, analysis, PROBE_SCENES[0], convention)
        state["probe_inputs"] = inputs
        frame = next(iter(inputs.frames.values()))
        feature_dim = int(cfg.model.get("feature_dim", 768))
        grid = (frame.height // PATCH_SIZE, frame.width // PATCH_SIZE)

        schema = {
            "dino_features": check_schema(
                "dino_features",
                inputs.cache.features(cfg.feature_encoder, frame.frame_id),
                expected_shape=(feature_dim, grid[0], grid[1]),
            ),
            "gt_depth": check_schema(
                "gt_depth", inputs.cache.depth(frame.depth_path),
                expected_shape=(frame.height, frame.width),
            ),
            "aligned_context_depth": describe_tensor(inputs.est_maps[frame.frame_id]),
            "intrinsics": check_schema("intrinsics", frame.K, expected_shape=(3, 3)),
            "pose": check_schema(
                "T_world_from_camera", frame.T_world_from_camera, expected_shape=(4, 4)
            ),
            "frame_id_type": type(frame.frame_id).__name__,
            "frame_id_example": str(frame.frame_id),
            "n_frames": len(inputs.frames),
            "convention": inputs.convention,
        }
        state["schema"] = schema
        return schema

    def step4() -> dict[str, Any]:
        # Scene inputs for every scene, not only the three probe families. The
        # aligned context depth is a derived artifact with no accepted stored
        # value, so it is recomputed through Phase 4's own code here and its
        # digest recorded per scene; that is the "recomputed maps" line of the
        # pin, and it is what the receipt binds for that input.
        summaries: dict[str, Any] = {}
        for scene in REPLICA_SCENES:
            first = scene == PROBE_SCENES[0]
            inputs = (
                state["probe_inputs"] if first
                else build_scene_inputs(cfg, analysis, scene, state["convention"])
            )
            summaries[scene] = {
                "n_frames": len(inputs.frames),
                "n_aligned_maps": len(inputs.est_maps),
                "n_calibrations": len(inputs.calibrations),
                "n_affine_ok": sum(
                    1 for c in inputs.calibrations.values() if not c.affine_failed
                ),
                "convention": inputs.convention,
                "aligned_depth_digest": aligned_depth_digest(inputs),
            }
            state["scene_identities"][scene]["aligned_depth_digest"] = (
                summaries[scene]["aligned_depth_digest"]
            )
            if not first:
                inputs.close()
        conventions = {s["convention"] for s in summaries.values()}
        if len(conventions) != 1:
            raise GateStop(
                "4", f"scenes were treated under {sorted(conventions)}; one "
                "checkpoint carries one depth convention (Amendment A6)",
                FROZEN_DESIGN_MISMATCH, {"scenes": summaries},
            )
        return {"scenes": summaries, "single_convention": sorted(conventions)[0]}

    def step5() -> dict[str, Any]:
        center = phase5_mean_vector(cfg)
        state["center"] = center
        state["mean_vector"] = verify_mean_vector_identity(
            center, state["scene_identities"], "5"
        )
        inputs = state["probe_inputs"]
        pairs = phase5_scene_pairs(cfg, analysis, inputs.scene)
        per_regime: dict[str, Any] = {}
        for regime in REGIMES:
            candidates = [p for p in pairs if p.regime == regime]
            if not candidates:
                raise GateStop(
                    "5", f"scene {inputs.scene} has no {regime} pairs, so the gate "
                    "cannot build one example per regime",
                    MISSING_ARTIFACT,
                    {"regimes_present": sorted({p.regime for p in pairs})},
                )
            example = None
            for pair in candidates:
                example = build_example(
                    cfg, analysis, inputs, pair, cfg.primary_alignment_level, center
                )
                if example is not None:
                    break
            if example is None:
                raise GateStop(
                    "5", f"no {regime} pair in {inputs.scene} produced a usable "
                    "example at the primary alignment level",
                    FROZEN_DESIGN_MISMATCH, {"n_candidates": len(candidates)},
                )
            leak = assert_no_forbidden_fields(example, f"{regime} example")
            per_regime[regime] = {
                "pair": f"{example.context_frame_id} -> {example.target_frame_id}",
                "n_supported": int(example.support.sum()),
                "fields": leak["fields"],
                "features_context": describe_tensor(example.features_context),
                "depth_context_aligned": describe_tensor(example.depth_context_aligned),
                "camera": describe_tensor(example.camera),
                "query_patch_coords": describe_tensor(example.query_patch_coords),
            }
            state.setdefault("examples", {})[regime] = example
        return {
            "model_visible_fields": list(model_visible_fields()),
            "per_regime": per_regime,
        }

    def step6() -> dict[str, Any]:
        """The headline rests on both methods being scored at one location.

        The check compares the coordinates the *example* actually carries, which
        is the object training and evaluation consume, against an expectation
        the gate derives independently from the lift and the support. An earlier
        version built both sides from the same uv_t tensor, so the comparison
        was between a value and itself and could not fail: build_example could
        have indexed lift.landed instead of the support, applied the patch
        mapping twice, or used a different patch size, and the step still
        reported PASS.
        """
        inputs = state["probe_inputs"]
        example = state["examples"]["translation"]
        lift, context, target, T = lift_for(example)
        dtype = cfg.torch_dtype
        target_hw = (target.height, target.width)

        # Rebuild the support the same way build_example does, from ground truth
        # and the explicit comparator, then derive what the supervision
        # coordinates must be if the two methods read one location.
        evaluable = context_lift_support(
            lift,
            inputs.cache.depth(context.depth_path).to(dtype),
            inputs.cache.depth(target.depth_path).to(dtype),
            context.K.to(dtype), target.K.to(dtype), T,
            rel_tol=analysis.covisible_relative_depth_tol,
        )
        support = primary_support(lift, evaluable)
        chosen = torch.nonzero(support, as_tuple=False).reshape(-1)
        expected = pixel_to_patch_coords(lift.uv_target[chosen], PATCH_SIZE)
        actual = example.query_patch_coords

        if actual.shape != expected.shape:
            raise GateStop(
                "6", "the example's supervision coordinates do not have the shape "
                "the context-lift support implies, so the predictor is not read "
                "at the landing locations CL-Transport is scored at",
                IMPLEMENTATION_BUG,
                {"example_shape": list(actual.shape),
                 "expected_shape": list(expected.shape),
                 "n_supported": int(support.sum())},
            )
        residual = (
            float((actual.to(expected.dtype) - expected).abs().max())
            if actual.numel() else 0.0
        )
        if residual > 0.0:
            raise GateStop(
                "6", "CL-Transport and the predictor are not scored against the "
                f"target feature at the same location; coordinates differ by "
                f"{residual}",
                IMPLEMENTATION_BUG, {"max_abs_coordinate_difference": residual},
            )

        rows: list[dict[str, Any]] = []
        for row_index, index in enumerate(chosen.tolist()[:5]):
            uv_t = lift.uv_target[index]
            rows.append({
                "context_sample_index": index,
                "context_uv": [float(v) for v in lift.uv_context[index]],
                "context_depth_m": float(lift.depth_context[index]),
                "cl_landing_uv": [float(v) for v in uv_t],
                "target_cell_scored": int(
                    patch_cell_index(uv_t[None], target_hw, PATCH_SIZE)[0]
                ),
                "predictor_read_patch_coord": [
                    float(v) for v in actual[row_index]
                ],
                "supervision_read_patch_coord": [
                    float(v) for v in expected[row_index]
                ],
            })
        return {
            "samples": rows,
            "n_landed": int(lift.landed.sum()),
            "n_supported": int(support.sum()),
            "max_abs_coordinate_difference": residual,
        }

    def step7() -> dict[str, Any]:
        inputs = state["probe_inputs"]
        example = state["examples"]["translation"]
        lift, context, target, T = lift_for(example)
        dtype = cfg.torch_dtype
        evaluable = context_lift_support(
            lift,
            inputs.cache.depth(context.depth_path).to(dtype),
            inputs.cache.depth(target.depth_path).to(dtype),
            context.K.to(dtype), target.K.to(dtype), T,
            rel_tol=analysis.covisible_relative_depth_tol,
        )
        support = primary_support(lift, evaluable)
        counts = {
            "candidate_samples": int(lift.landed.numel()),
            "depth_valid": int(lift.depth_valid.sum()),
            "landed": int(lift.landed.sum()),
            "gt_evaluable": int(evaluable.sum()),
            "final_support": int(support.sum()),
        }
        if counts["final_support"] == 0:
            raise GateStop(
                "7", "the primary support is empty on a real translation pair",
                FROZEN_DESIGN_MISMATCH, counts,
            )

        fc = inputs.cache.features(cfg.feature_encoder, example.context_frame_id).to(dtype)
        ft = inputs.cache.features(cfg.feature_encoder, example.target_frame_id).to(dtype)
        center = state["center"].to(dtype)
        grid = (ft.shape[1], ft.shape[2])
        n_cells = grid[0] * grid[1]

        without = score_primary(lift, support, fc, ft, center, None, grid)
        broken = torch.full((n_cells, ft.shape[0]), float("nan"), dtype=dtype)
        with_broken = score_primary(lift, support, fc, ft, center, broken, grid)

        if with_broken.n_primary != without.n_primary:
            raise GateStop(
                "7", "an all-nonfinite predictor changed the support size",
                IMPLEMENTATION_BUG,
                {"without": without.n_primary, "with_broken": with_broken.n_primary},
            )
        for name, a, b in (
            ("explicit score", without.cl_centered, with_broken.cl_centered),
            ("No-Warp floor", without.nowarp_centered, with_broken.nowarp_centered),
        ):
            if abs(a - b) > 1e-12:
                raise GateStop(
                    "7", f"an all-nonfinite predictor changed the {name}",
                    IMPLEMENTATION_BUG, {"without": a, "with_broken": b},
                )
        if with_broken.n_predict_nonfinite != with_broken.n_primary:
            raise GateStop(
                "7", "nonfinite predictions were dropped rather than scored as "
                "failures", IMPLEMENTATION_BUG,
                {"n_primary": with_broken.n_primary,
                 "n_nonfinite": with_broken.n_predict_nonfinite},
            )
        state["primary_support"] = support

        # The cross-path producer, run on the real pair. This is the arithmetic
        # PROTOCOL 3.9's disclosure rests on, and it has to be seen working on
        # real geometry rather than only on synthetic columns.
        from .phase5_score import score_cross_path
        from .transport import apply_transport_plan, transport_plan

        est_plan = transport_plan(
            example.depth_context_aligned, context.K.to(dtype), target.K.to(dtype), T,
            (target.height, target.width),
        )
        transported = apply_transport_plan(est_plan, fc).reshape(fc.shape[0], -1)
        # Stand-in operational support: every cell the plan gave any weight to.
        # The accepted Phase 4 scored-cell mask replaces this in evaluate; the
        # gate needs a real, non-empty cell set to prove the producer runs.
        coverage = est_plan.coverage.reshape(-1) if hasattr(est_plan, "coverage") else None
        scored_cells = (
            np.nonzero(coverage.cpu().numpy() > 0)[0]
            if coverage is not None else np.arange(n_cells)
        )
        cross = score_cross_path(
            lift, support, scored_cells, fc, ft, center, None, transported,
            fc.reshape(fc.shape[0], -1), ft.reshape(ft.shape[0], -1),
            (target.height, target.width), grid,
        )
        if cross.n_intersect == 0:
            raise GateStop(
                "7", "the cross-path common-valid set is empty on a real pair",
                FROZEN_DESIGN_MISMATCH, {"scored_cells": int(scored_cells.size)},
            )
        if int(cross.samples_per_cell.sum()) != int(cross.per_point_mask.sum()):
            raise GateStop(
                "7", "cross-path pooling lost or duplicated per-point samples",
                IMPLEMENTATION_BUG,
                {"pooled": int(cross.samples_per_cell.sum()),
                 "masked": int(cross.per_point_mask.sum())},
            )
        return {
            "counts": counts,
            "cl_centered": without.cl_centered,
            "nowarp_centered": without.nowarp_centered,
            "all_nonfinite_predictor_score": with_broken.predict_centered,
            "failures_counted": with_broken.n_predict_nonfinite,
            "cross_path": {
                "n_intersect": cross.n_intersect,
                "max_samples_per_cell": int(cross.samples_per_cell.max()),
                "x_cl_centered": cross.x_cl_centered,
                "x_sp_transport_centered": cross.x_sp_transport_centered,
            },
        }

    def step8() -> dict[str, Any]:
        # TL-Reference for the translation pair, recomputed exactly as evaluate
        # recomputes it, so its reconciliation with Phase 4 and the cell
        # intersection are proven on real data before anything trains.
        from .phase5_reference import read_phase4_reference, recompute_reference_arms

        inputs = state["probe_inputs"]
        example = state["examples"]["translation"]
        lift, context, target, T = lift_for(example)
        dtype = cfg.torch_dtype
        level = cfg.primary_alignment_level
        ctx, tgt = example.context_frame_id, example.target_frame_id
        fc = inputs.cache.features(cfg.feature_encoder, ctx)
        ft = inputs.cache.features(cfg.feature_encoder, tgt)
        arms = recompute_reference_arms(
            inputs.cache.depth(context.depth_path).to(dtype),
            inputs.cache.depth(target.depth_path).to(dtype),
            inputs.est_maps[ctx], inputs.est_maps[tgt], inputs.calibrations[ctx],
            fc, ft, context.K.to(dtype), target.K.to(dtype), T,
            inputs.scene, ctx, tgt, analysis, level, dtype,
        )
        reference = read_phase4_reference(Path(cfg.phase4_dir) / "eval", inputs.scene, level)
        return formulation_support_evidence(
            lift, state["primary_support"], arms, reference["pairs"].get((ctx, tgt)),
            state["center"], fc, (target.height, target.width),
            f"{inputs.scene} {ctx} -> {tgt} level {level}",
        )

    def step9() -> dict[str, Any]:
        return splat_symmetry_evidence()

    def step10() -> dict[str, Any]:
        evidence: dict[str, Any] = {}
        dtype = cfg.torch_dtype

        rot = state["examples"]["rotation"]
        lift, context, target, T = lift_for(rot)
        hw_c = (context.height, context.width)
        analytic = rotation_homography_landing(
            context.K.to(dtype), target.K.to(dtype), T, hw_c, dtype=dtype
        )
        substituted = context_lift_map(
            torch.full_like(rot.depth_context_aligned, 7.0),
            context.K.to(dtype), target.K.to(dtype), T,
            hw_c, (target.height, target.width),
        )
        evidence["rotation"] = rotation_gate_evidence(
            lift, analytic, substituted, analysis.rotation_gate_coord_tol_px
        )

        tr = state["examples"]["translation"]
        lift, context, target, T = lift_for(tr)
        chosen = torch.nonzero(lift.landed, as_tuple=False).reshape(-1)[:64]
        points = unproject(
            lift.uv_context[chosen], lift.depth_context[chosen], context.K.to(dtype)
        )
        independent, _ = project(transform_points(T, points), target.K.to(dtype))
        translation_residual = float((independent - lift.uv_target[chosen]).abs().max())
        evidence["translation"] = {
            "n_samples": int(chosen.numel()),
            "max_independent_reprojection_residual_px": translation_residual,
        }
        if translation_residual > analysis.rotation_gate_coord_tol_px:
            raise GateStop(
                "10", "context-lift disagrees with an independent reprojection of "
                "the same samples", IMPLEMENTATION_BUG, evidence,
            )
        return evidence

    def step11() -> dict[str, Any]:
        return check_folds_against_inventory(Path(cfg.renders_root))

    def step12() -> dict[str, Any]:
        inputs = state["probe_inputs"]
        frame = next(iter(inputs.frames.values()))
        model_cfg = predictor_config_from(cfg, (frame.height, frame.width))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_predictor(model_cfg).to(device)

        batch_pairs = int(cfg.training.get("batch_pairs", 8))
        examples = list(state["examples"].values())
        batch = [examples[i % len(examples)] for i in range(batch_pairs)]
        # The batch goes through the real training path untouched: no
        # truncation, no manual device transfer. An earlier version of this step
        # trimmed every pair to the shortest supervision length and moved the
        # tensors itself, which made the probe pass while the actual training
        # path could not have batched pairs of differing support at all, nor
        # placed its own tensors on the device. A gate that works around the
        # code it is meant to exercise proves nothing about that code.
        grid = model_cfg.target_grid
        lengths = [int(e.support.numel()) for e in batch]
        shapes = {
            "batch_pairs": len(batch),
            "supervision_lengths": lengths,
            "ragged": len(set(lengths)) > 1,
            "features_context": describe_tensor(batch[0].features_context),
            "depth_context_aligned": describe_tensor(batch[0].depth_context_aligned),
            "camera": describe_tensor(batch[0].camera),
            "context_valid": describe_tensor(batch[0].context_valid),
            "query_patch_coords": describe_tensor(batch[0].query_patch_coords),
            "target_centered": describe_tensor(batch[0].target_centered),
            "support": describe_tensor(batch[0].support),
        }
        assert_no_forbidden_fields(batch[0], "dry-run batch")
        moved = batch

        holder: dict[str, Any] = {}

        def forward():
            loss, cosine, n = batch_loss(model, moved, grid)
            holder.update({"cosine": cosine, "n": n, "loss": float(loss)})
            return loss

        def backward(loss) -> None:
            loss.backward()

        probe = resource_probe(forward, backward, device)

        if not np.isfinite(holder["loss"]):
            raise GateStop(
                "12", "the frozen loss was not finite on the real dry-run batch",
                IMPLEMENTATION_BUG, {"shapes": shapes},
            )
        n_grads = 0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            n_grads += 1
            if not torch.isfinite(parameter.grad).all():
                raise GateStop(
                    "12", "a gradient was not finite on the real dry-run batch",
                    IMPLEMENTATION_BUG, {"shapes": shapes},
                )
        if n_grads == 0:
            raise GateStop(
                "12", "the backward pass produced no gradients at all",
                IMPLEMENTATION_BUG, {"shapes": shapes},
            )
        # No optimizer was constructed and no step was taken.
        model.zero_grad(set_to_none=True)
        state["resource_probe"] = probe
        return {
            "shapes": shapes,
            "loss": holder["loss"],
            "mean_centered_cosine": holder["cosine"],
            "n_supervised_samples": holder["n"],
            "n_parameters_with_gradients": n_grads,
            "parameter_count": model.parameter_count(),
            "device": device,
            "optimizer_constructed": False,
        }

    def step13() -> dict[str, Any]:
        probe = dict(state.get("resource_probe", {}))
        probe["frozen_batch_pairs"] = int(cfg.training.get("batch_pairs", 8))
        probe["frozen_effective_batch_pairs"] = int(
            cfg.training.get("effective_batch_pairs", 8)
        )
        if probe.get("peak_reserved_bytes") and probe.get("total_memory_bytes"):
            headroom = probe["total_memory_bytes"] - probe["peak_reserved_bytes"]
            probe["headroom_bytes"] = headroom
            if headroom < 0:
                raise GateStop(
                    "13", "the frozen batch configuration does not fit in device "
                    "memory. This is reported as a resource mismatch rather than "
                    "resolved by shrinking the effective batch, which is part of "
                    "the frozen training configuration.",
                    FROZEN_DESIGN_MISMATCH, probe,
                )
        return probe

    def step14() -> dict[str, Any]:
        evidence = check_test_seal()
        evidence["test_evaluation_requires_explicit_mode"] = True
        return evidence

    def step15() -> dict[str, Any]:
        pin = {
            "resolved_artifacts": state.get("artifacts", {}),
            "scene_identities": state.get("scene_identities", {}),
            "mean_vector": state.get("mean_vector", {}),
            "schema_summary": state.get("schema", {}),
            "implementation_commit": git_commit(),
            "split_hash": fold_digest(frozen_folds()),
            "config_digest": cfg.digest(),
            "environment": environment_identity(),
            "note": "no Phase 5 test outcome was inspected by this gate",
        }
        destination = Path(cfg.evidence_dir) / "pin_cluster.json"
        outcome = write_once(
            destination, json.dumps(pin, indent=2, sort_keys=True, default=str)
        )
        state["pin_path"] = str(destination)
        return {**outcome, "keys": sorted(pin)}

    def step17() -> dict[str, Any]:
        # Specification step 6 across the whole pure-rotation regime, as
        # reporting_rules.md section 7 records it: every rotation pair of every
        # scene, at the primary level, before Context-Lift is used
        # scientifically. Scene inputs are built as step 4 builds them.
        per_scene: dict[str, Any] = {}
        for scene in REPLICA_SCENES:
            first = scene == PROBE_SCENES[0]
            inputs = (
                state["probe_inputs"] if first
                else build_scene_inputs(cfg, analysis, scene, state["convention"])
            )
            try:
                per_scene[scene] = rotation_scene_evidence(cfg, analysis, inputs)
            finally:
                if not first:
                    inputs.close()
        return rotation_regime_summary(
            per_scene, analysis.rotation_gate_coord_tol_px,
            analysis.rotation_position_bound_m,
        )

    def step16() -> dict[str, Any]:
        evidence = assert_no_checkpoint_written(run_dir, checkpoints_before)
        probe_inputs = state.get("probe_inputs")
        if probe_inputs is not None:
            probe_inputs.close()
        return {**evidence, "pin": state.get("pin_path")}

    return run_steps([
        ("1", "script identity and environment", step1),
        ("2", "resolve real Borah artifacts", step2),
        ("3", "verify real schemas", step3),
        ("4", "build real scene inputs, three families", step4),
        ("5", "one real example per regime, no forbidden fields", step5),
        ("6", "headline landing-location semantics", step6),
        ("7", "primary support V_P5_pp and predictor invariance", step7),
        ("8", "formulation support V_form on target cells", step8),
        ("9", "splat-pool information symmetry on real code", step9),
        ("10", "real-data geometry checks", step10),
        ("11", "frozen folds against the real scene inventory", step11),
        ("12", "dry-run one real batch: forward, loss, backward", step12),
        ("13", "resource probe against the frozen batch", step13),
        ("14", "test seal", step14),
        ("17", "pure-rotation gate across the regime", step17),
        ("15", "complete the deferred pin", step15),
        ("16", "verdict and no-side-effect assertion", step16),
    ])
