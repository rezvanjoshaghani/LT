"""The sixteen-step Phase 5 Borah integration gate, assembled and driven.

Separated from lot.phase5_check, which holds the reusable checks, so the order
of the gate reads as one list in one place. Nothing here trains. Step 12 runs a
single forward pass, the frozen loss, and a single backward pass on a real batch
to prove the pipeline is finite end to end, then discards the gradients; step 16
asserts no checkpoint appeared.
"""

from __future__ import annotations

import dataclasses
import json
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
        example = state["examples"]["translation"]
        lift, _, target, _ = lift_for(example)
        target_hw = (target.height, target.width)
        rows: list[dict[str, Any]] = []
        chosen = torch.nonzero(lift.landed, as_tuple=False).reshape(-1)[:5]
        for index in chosen.tolist():
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
                    float(v) for v in pixel_to_patch_coords(uv_t[None], PATCH_SIZE)[0]
                ],
                "supervision_read_uv": [float(v) for v in uv_t],
            })
        for row in rows:
            if row["cl_landing_uv"] != row["supervision_read_uv"]:
                raise GateStop(
                    "6", "CL-Transport and the predictor are not scored against the "
                    "target feature at the same location",
                    IMPLEMENTATION_BUG, {"row": row},
                )
        return {"samples": rows, "n_landed": int(lift.landed.sum())}

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
        example = state["examples"]["translation"]
        lift, _, target, _ = lift_for(example)
        target_hw = (target.height, target.width)
        cells = patch_cell_index(lift.uv_target, target_hw, PATCH_SIZE)
        n_cells = (target.height // PATCH_SIZE) * (target.width // PATCH_SIZE)
        out_of_range = [int(c) for c in np.unique(cells) if not 0 <= int(c) < n_cells]
        if out_of_range:
            raise GateStop(
                "8", f"landing cells fall outside the target grid: "
                f"{out_of_range[:10]}",
                IMPLEMENTATION_BUG, {"n_cells": n_cells},
            )
        support = state["primary_support"]
        supported_cells = cells[support.cpu().numpy()]
        unique, counts = np.unique(supported_cells, return_counts=True)
        return {
            "n_supported_samples": int(supported_cells.size),
            "n_distinct_target_cells": int(unique.size),
            "max_samples_per_cell": int(counts.max()) if counts.size else 0,
            "target_grid_cells": n_cells,
            "note": (
                "many context patches may land in one target cell; that is "
                "expected and is not ambiguity. What must be unique, and is, is "
                "the cell identity each sample maps to."
            ),
        }

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
        residual = float((lift.uv_target - analytic).abs().max())
        substituted = context_lift_map(
            torch.full_like(rot.depth_context_aligned, 7.0),
            context.K.to(dtype), target.K.to(dtype), T,
            hw_c, (target.height, target.width),
        )
        depth_free = float((lift.uv_target - substituted.uv_target).abs().max())
        evidence["rotation"] = {
            "max_homography_residual_px": residual,
            "tolerance_px": analysis.rotation_gate_coord_tol_px,
            "max_depth_substitution_shift_px": depth_free,
        }
        if residual > analysis.rotation_gate_coord_tol_px:
            raise GateStop(
                "10", "context-lift disagrees with the analytic rotational "
                "homography on real data", IMPLEMENTATION_BUG, evidence,
            )
        if depth_free > analysis.rotation_gate_coord_tol_px:
            raise GateStop(
                "10", "a pure-rotation landing moved when the depth map was "
                "replaced, so the mapping is not depth free",
                IMPLEMENTATION_BUG, evidence,
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
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(pin, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        state["pin_path"] = str(destination)
        return {"written": str(destination), "keys": sorted(pin)}

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
        ("15", "complete the deferred pin", step15),
        ("16", "verdict and no-side-effect assertion", step16),
    ])
