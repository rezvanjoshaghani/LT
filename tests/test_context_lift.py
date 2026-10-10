"""Stream R: validate Context-Lift Transport-Only before it is used scientifically.

Three gates, in the order the Phase 5 specification runs them: a static
information-access audit (step 5), the pure-rotation depth-independence gate
(step 6), and forward-transport tests on analytic geometry (step 7). A failure
in any of them is a stop, because a comparator that transports incorrectly
would make the headline learned-vs-explicit gap meaningless.
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap

import numpy as np
import pytest
import torch

from lot.analysis_config import load_analysis_config
from lot.context_lift import (
    CONTEXT_LIFT_SALT,
    ContextLiftMap,
    context_lift_map,
    context_lift_sample_ids,
    context_lift_support,
    context_patch_centers,
    rotation_homography_landing,
)
from lot.geometry import invert_se3, project, relative_pose, transform_points, unproject
from lot.sample_identity import sample_ids

from scenes import (
    BASELINE,
    DISPARITY_BACK,
    DISPARITY_FRONT,
    FX,
    GRID,
    IMAGE_SIZE,
    PATCH,
    SLAB_CTX_COLS,
    Z_BACK,
    Z_FRONT,
    build_rotation_scene,
    build_two_plane_scene,
    intrinsics,
    make_pose,
)

HW = (IMAGE_SIZE, IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Step 5: static information-access audit
# ---------------------------------------------------------------------------

# Every name that would represent target-side content reaching the comparator's
# geometry. The audit is on the transport path only: context_lift_support is
# allowed ground truth because deciding membership is the one permitted role.
FORBIDDEN_IN_TRANSPORT = (
    "depth_target",
    "target_depth",
    "est_target",
    "target_aligned",
    "pointmap",
    "flow",
    "correspondence",
    "features_target",
    "target_features",
)


def test_transport_path_has_no_target_depth_parameter():
    """The structural form of the audit: it cannot be passed, not merely is not.

    A grep can be defeated by a rename. A signature cannot: if there is no
    parameter for target depth, no caller can supply it.
    """
    params = list(inspect.signature(context_lift_map).parameters)
    assert params == [
        "depth_context_aligned",
        "K_context",
        "K_target",
        "T_target_from_context",
        "context_hw",
        "target_hw",
        "patch_size",
    ]
    for name in params:
        for bad in FORBIDDEN_IN_TRANSPORT:
            assert bad not in name, f"{name} looks like target content"


def _referenced_identifiers(func) -> set[str]:
    """Every identifier the function's code actually references.

    Parsed rather than grepped, so documentation prose cannot fail the audit and,
    more importantly, cannot pass it either: a comment claiming the function
    avoids target depth is not evidence, and a name bound in code is.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return names


def test_transport_path_references_no_target_content():
    """The audit proper: no symbol naming target content is referenced at all."""
    names = _referenced_identifiers(context_lift_map)
    for bad in FORBIDDEN_IN_TRANSPORT:
        offenders = {n for n in names if bad in n}
        assert not offenders, f"context_lift_map references {offenders}"
    # It must also not reach for the frozen ground-truth visibility machinery,
    # which belongs to support and not to transport.
    assert "visibility_masks" not in names
    assert "covisible" not in names
    # The only depth it may name is the aligned context map.
    depth_names = {n for n in names if "depth" in n}
    assert depth_names <= {
        "depth_context_aligned",  # the parameter, the only depth supplied
        "depth",                  # the values read from it
        "depth_valid",            # their finite-and-positive mask
        "depth_context",          # the field the read is returned in
    }, depth_names


def test_support_is_the_only_ground_truth_consumer():
    """Ground truth enters exactly one function, and only to decide membership."""
    support_params = list(inspect.signature(context_lift_support).parameters)
    assert "depth_context_gt" in support_params
    assert "depth_target_gt" in support_params
    # The support function returns a mask and nothing else, so it cannot leak a
    # coordinate or a score back into the transport path.
    scene = build_two_plane_scene()
    lift = context_lift_map(
        scene.depth_context, scene.K, scene.K, scene.T_target_from_context, HW, HW
    )
    support = context_lift_support(
        lift, scene.depth_context, scene.depth_target, scene.K, scene.K,
        scene.T_target_from_context,
    )
    assert support.dtype == torch.bool
    assert support.shape == lift.landed.shape


def test_support_cannot_move_a_coordinate():
    """Changing ground truth changes which samples are evaluable, never where
    the estimated geometry says they land."""
    scene = build_two_plane_scene()
    lift = context_lift_map(
        scene.depth_context, scene.K, scene.K, scene.T_target_from_context, HW, HW
    )
    before = lift.uv_target.clone()
    wrecked_gt = scene.depth_target.clone()
    wrecked_gt[:, :] = Z_BACK * 3.0
    support_a = context_lift_support(
        lift, scene.depth_context, scene.depth_target, scene.K, scene.K,
        scene.T_target_from_context,
    )
    support_b = context_lift_support(
        lift, scene.depth_context, wrecked_gt, scene.K, scene.K,
        scene.T_target_from_context,
    )
    assert torch.equal(lift.uv_target, before)
    assert not torch.equal(support_a, support_b), "the fixture should separate these"


# ---------------------------------------------------------------------------
# Step 6: the pure-rotation gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("yaw_deg", [1.0, 5.0, 12.0, 25.0, 40.0])
def test_pure_rotation_landing_is_depth_independent(yaw_deg):
    """Zero translation means the lifted depth cancels algebraically.

    This is the property that makes pure rotation the cleanest learned-versus-
    explicit test: the explicit comparator has an exact depth-free solution, so
    any gap the predictor shows there is a transformation error and not a depth
    error. Asserted against wildly different depth maps, including one that
    varies per pixel.
    """
    scene = build_rotation_scene(yaw_deg)
    flat_near = torch.full(HW, 1.0, dtype=torch.float64)
    flat_far = torch.full(HW, 97.0, dtype=torch.float64)
    ramp = torch.linspace(0.5, 50.0, IMAGE_SIZE, dtype=torch.float64)
    varying = ramp[None, :].expand(IMAGE_SIZE, IMAGE_SIZE).contiguous()

    landings = [
        context_lift_map(d, scene.K, scene.K, scene.T_target_from_context, HW, HW).uv_target
        for d in (flat_near, flat_far, varying)
    ]
    for other in landings[1:]:
        assert torch.allclose(landings[0], other, atol=1e-9), "depth changed a rotation landing"


@pytest.mark.parametrize("yaw_deg", [1.0, 5.0, 12.0, 25.0, 40.0])
def test_pure_rotation_matches_the_analytic_homography(yaw_deg):
    """The frozen coordinate tolerance is the gate the protocol already uses."""
    analysis = load_analysis_config()
    scene = build_rotation_scene(yaw_deg)
    depth = torch.full(HW, 3.0, dtype=torch.float64)
    lift = context_lift_map(depth, scene.K, scene.K, scene.T_target_from_context, HW, HW)
    analytic = rotation_homography_landing(
        scene.K, scene.K, scene.T_target_from_context, HW, dtype=torch.float64
    )
    residual = (lift.uv_target - analytic).abs().max().item()
    assert residual <= analysis.rotation_gate_coord_tol_px, (
        f"context-lift departs from the rotational homography by {residual} px"
    )


@pytest.mark.parametrize("yaw_deg", [5.0, 25.0])
def test_context_lift_and_target_lift_are_mutual_inverses_under_rotation(yaw_deg):
    """The accepted Phase-4 backward map and this forward map must agree.

    Phase 4 lifts a target sample into the context; this lifts a context patch
    into the target. Under pure rotation both are the same homography read in
    opposite directions, so composing them must return the starting point. This
    is the cross-check that the new comparator has not silently inverted a pose.
    """
    analysis = load_analysis_config()
    scene = build_rotation_scene(yaw_deg)
    depth = torch.full(HW, 4.0, dtype=torch.float64)
    lift = context_lift_map(depth, scene.K, scene.K, scene.T_target_from_context, HW, HW)

    # Phase 4's construction, verbatim in form: read depth at the landing in the
    # target frame, unproject, transform back, project into the context.
    T_ctx_from_tgt = invert_se3(scene.T_target_from_context)
    z_at_landing = lift.z_target
    points_target = unproject(lift.uv_target, z_at_landing, scene.K)
    points_context = transform_points(T_ctx_from_tgt, points_target)
    uv_back, _ = project(points_context, scene.K)

    inside = lift.landed
    residual = (uv_back[inside] - lift.uv_context[inside]).abs().max().item()
    assert residual <= analysis.rotation_gate_coord_tol_px, (
        f"forward then backward moved a point by {residual} px"
    )


# ---------------------------------------------------------------------------
# Step 7: forward-transport tests on analytic geometry
# ---------------------------------------------------------------------------

def test_identity_camera_is_a_fixed_point():
    """Same camera, no motion: every patch lands exactly on itself, any depth."""
    K = intrinsics()
    T = relative_pose(make_pose(), make_pose())
    for value in (0.7, 5.0, 41.0):
        depth = torch.full(HW, value, dtype=torch.float64)
        lift = context_lift_map(depth, K, K, T, HW, HW)
        assert torch.allclose(lift.uv_target, lift.uv_context, atol=1e-9)
        assert bool(lift.landed.all())
        assert torch.allclose(lift.z_target, torch.full_like(lift.z_target, value))


@pytest.mark.parametrize("depth_m,expected_disparity", [(Z_FRONT, DISPARITY_FRONT),
                                                       (Z_BACK, DISPARITY_BACK)])
def test_pure_translation_on_a_plane_shifts_by_the_analytic_disparity(depth_m, expected_disparity):
    """A fronto-parallel plane at depth Z under baseline B shifts by fx*B/Z."""
    K = intrinsics()
    T = relative_pose(make_pose(t=torch.tensor([BASELINE, 0.0, 0.0])), make_pose())
    depth = torch.full(HW, float(depth_m), dtype=torch.float64)
    lift = context_lift_map(depth, K, K, T, HW, HW)

    shift_u = (lift.uv_target[:, 0] - lift.uv_context[:, 0])
    assert torch.allclose(shift_u, torch.full_like(shift_u, -float(expected_disparity)), atol=1e-9)
    # A lateral baseline moves nothing vertically.
    assert torch.allclose(lift.uv_target[:, 1], lift.uv_context[:, 1], atol=1e-9)
    # The plane is fronto-parallel, so the depth in the target camera is unchanged.
    assert torch.allclose(lift.z_target, torch.full_like(lift.z_target, float(depth_m)), atol=1e-9)


def test_two_depth_plane_scene_moves_each_plane_by_its_own_disparity():
    """The depth-dependent case: the near slab and the far background separate."""
    scene = build_two_plane_scene()
    lift = context_lift_map(
        scene.depth_context, scene.K, scene.K, scene.T_target_from_context, HW, HW
    )
    u_ctx = lift.uv_context[:, 0]
    on_slab = (u_ctx >= SLAB_CTX_COLS[0]) & (u_ctx < SLAB_CTX_COLS[1])
    shift = lift.uv_target[:, 0] - u_ctx
    assert torch.allclose(
        shift[on_slab], torch.full_like(shift[on_slab], -float(DISPARITY_FRONT)), atol=1e-9
    )
    assert torch.allclose(
        shift[~on_slab], torch.full_like(shift[~on_slab], -float(DISPARITY_BACK)), atol=1e-9
    )
    assert int(on_slab.sum()) > 0 and int((~on_slab).sum()) > 0


def test_landing_rule_excludes_behind_outside_and_invalid():
    """Validity 5d in the forward direction, one clause at a time."""
    K = intrinsics()
    T = relative_pose(make_pose(), make_pose())

    nonfinite = torch.full(HW, 3.0, dtype=torch.float64)
    nonfinite[:, : PATCH] = float("nan")
    lift = context_lift_map(nonfinite, K, K, T, HW, HW)
    first_column = lift.uv_context[:, 0] < PATCH
    assert not bool(lift.depth_valid[first_column].any())
    assert not bool(lift.landed[first_column].any())

    nonpositive = torch.full(HW, 3.0, dtype=torch.float64)
    nonpositive[:, : PATCH] = -1.0
    lift = context_lift_map(nonpositive, K, K, T, HW, HW)
    assert not bool(lift.landed[first_column].any())

    # A large lateral baseline pushes the left of the image out of the target
    # sampling box, which the landing rule must drop rather than clamp.
    T_far = relative_pose(make_pose(t=torch.tensor([4.0, 0.0, 0.0])), make_pose())
    depth = torch.full(HW, 2.0, dtype=torch.float64)
    lift = context_lift_map(depth, K, K, T_far, HW, HW)
    assert bool((~lift.landed).any()), "the fixture should push some patches out"
    assert bool(lift.depth_valid.all()), "depth is valid everywhere here"
    assert bool((lift.uv_target[lift.landed][:, 0] >= 6.5 - 1e-9).all())


def test_a_point_behind_the_target_camera_does_not_land():
    K = intrinsics()
    # Move the target camera far along +z so the plane at z=2 falls behind it.
    T = relative_pose(make_pose(t=torch.tensor([0.0, 0.0, 10.0])), make_pose())
    depth = torch.full(HW, 2.0, dtype=torch.float64)
    lift = context_lift_map(depth, K, K, T, HW, HW)
    assert bool((lift.z_target < 0).all())
    assert not bool(lift.landed.any())


def test_occlusion_boundary_support_drops_the_disoccluded_side():
    """Where the near slab hides background, the background patch is not evaluable.

    Transport still produces a coordinate for it, which is correct: the explicit
    method attempts the map and the support rule is what decides the sample is
    not scoreable. Coordinates and membership stay separate concerns.
    """
    scene = build_two_plane_scene()
    lift = context_lift_map(
        scene.depth_context, scene.K, scene.K, scene.T_target_from_context, HW, HW
    )
    support = context_lift_support(
        lift, scene.depth_context, scene.depth_target, scene.K, scene.K,
        scene.T_target_from_context,
    )
    assert int(support.sum()) > 0
    assert int((~support).sum()) > 0
    # Everything supported must also have landed; support can only narrow.
    assert bool((support <= lift.landed).all())


def test_patch_centers_are_half_integers_and_row_major():
    uv = context_patch_centers(HW)
    assert uv.shape == (GRID * GRID, 2)
    assert torch.allclose(uv * 2.0, torch.round(uv * 2.0))
    assert uv[0].tolist() == [PATCH / 2 - 0.5, PATCH / 2 - 0.5]
    # Row major: the first GRID entries share a row and advance in u.
    assert torch.allclose(uv[:GRID, 1], torch.full((GRID,), uv[0, 1]))
    assert (uv[1:GRID, 0] > uv[: GRID - 1, 0]).all()


def test_context_lift_ids_are_deterministic_and_namespace_disjoint():
    uv = context_patch_centers(HW, dtype=torch.float64)
    a = context_lift_sample_ids("room_0", "ctx", "tgt", uv)
    b = context_lift_sample_ids("room_0", "ctx", "tgt", uv.clone())
    assert np.array_equal(a, b)
    assert len(set(a.tolist())) == len(a), "ids collide inside one pair"

    # Different pair, different ids.
    c = context_lift_sample_ids("room_0", "ctx", "other", uv)
    assert not np.array_equal(a, c)

    # The Phase 3 target-side namespace must not be reachable from this one, so
    # a join between the two can never silently succeed.
    phase3 = sample_ids("room_0", "ctx", "tgt", uv)
    assert not set(a.tolist()) & set(phase3.tolist())
    assert CONTEXT_LIFT_SALT != 0


def test_context_lift_map_rejects_a_bad_depth_shape():
    K = intrinsics()
    T = relative_pose(make_pose(), make_pose())
    with pytest.raises(ValueError, match=r"\[H, W\]"):
        context_lift_map(torch.zeros(3, 4, 5, dtype=torch.float64), K, K, T, HW, HW)


def test_map_dataclass_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="disagrees"):
        ContextLiftMap(
            uv_context=torch.zeros(4, 2),
            uv_target=torch.zeros(3, 2),
            z_target=torch.zeros(4),
            depth_context=torch.zeros(4),
            depth_valid=torch.zeros(4, dtype=torch.bool),
            landed=torch.zeros(4, dtype=torch.bool),
        )
