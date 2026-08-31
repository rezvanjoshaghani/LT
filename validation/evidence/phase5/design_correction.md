# Phase 5 design correction, recorded before any Phase 5 outcome exists

Stream P, step 1. Written 2026-08-30, before any Phase 5 model is trained, any
Phase 5 checkpoint is selected, and any Phase 5 test metric is computed. No
Phase 5 evaluation artifact of any kind existed when this file was committed.

## The correction

The accepted Phase 4 per-point Transport-Only estimator lifts the **target**
frame's estimated depth. Amendment A4 fixed that deliberately, because lifting
target-side samples is what preserves the Phase 3 `sample_id` universe that
PROTOCOL 3.2 requires every intersection to operate on. For Phase 4's question,
how much accuracy an explicit transporter loses when ground-truth depth is
replaced by an estimate, that construction is sound and Phase 4 is not
reopened.

It is not a valid comparator for Phase 5's question. Phase 5 asks what it costs
to learn the viewpoint transformation instead of computing it, holding the
information constant. Predict-with-Depth is forbidden every target-content
derived input, target depth included. VGGT's estimate of the target frame's
depth is computed from the target image, so an explicit method that consumes it
is reading target content the learned method may not read. Scored against each
other, the difference would confound a learned-versus-explicit transformation
cost with an information asymmetry, and the confound would sit on the headline
number rather than in a footnote.

## The evidence that the asymmetry is real

Traced in the accepted Phase 4 implementation at commit `ff52d13`:

    src/lot/phase4.py:1416   target_aligned = aligned_depth(level, depth_inputs.est_target, ...)
    src/lot/phase4.py:1431   est_read = sample_map_bilinear(target_aligned, samples.uv_target)
    src/lot/phase4.py:1434   points_target = unproject(samples.uv_target, safe_read, K_target)
    src/lot/phase4.py:1435   points_context = transform_points(T_context_from_target, points_target)
    src/lot/phase4.py:2004   est_target=est_maps[pair.target_frame_id]

The complete set of uses of the target-frame estimated map in Phase 4 is lines
1214 (field), 1275 (`.shape` only, the output grid size), 1416, 1417, 1424,
1432, and 2004. Its values reach the per-point path and nothing else.

## What Phase 5 does instead

Phase 5 introduces **Context-Lift Transport-Only** (`CL-Transport`), which maps
forwards from a context patch into the target image using context-frame
estimated depth only. Both headline methods then receive exactly

    I = { F_c, D_c, K_c, K_t, T_target_from_context }

and the only intended difference between them is whether the known projective
transformation is computed or learned.

The accepted Phase 4 per-point Transport-Only result is retained under the name
**`TL-Reference`**. It is a reference and a formulation diagnostic. It is never
substituted into the headline learned-versus-explicit gap.

## The splat-pool path needed no correction, and this was verified, not assumed

Stream W's precondition, checked before the design was frozen. On the
splat-and-pool path the accepted Phase 4 estimated arm builds its plan from the
context map alone:

    src/lot/phase4.py:1446   est_plan = transport_plan(context_aligned, K_context, K_target, ...)
    src/lot/phase4.py:1469   transported_est = apply_transport_plan(est_plan, features_context)

No `transport_plan` or `apply_transport_plan` call anywhere in Phase 4 receives
the target-frame estimated map. The operational comparator is therefore already
information-symmetric and is used unchanged, so the stop condition attached to
that verification did not fire.

## One judgment call, recorded because it was underdetermined

The specification defines the primary support as the samples CL-Transport can
validly transport, and separately defines a formulation support as the
intersection of that set with the target-lift set. The two estimators are
indexed differently: a context-lift sample is a context patch center, and a
target-lift sample is a target patch center, so a literal set intersection
between them is not defined.

Frozen reading, chosen before any outcome was inspected. The headline
comparison follows the specification's step 6 literally and scores at the
landing location, so its samples are context-indexed. The formulation
diagnostic intersects the two populations on the **target patch cell** each
sample occupies, which is the one index both estimators share. The headline
estimand is unaffected by this choice; only the formulation diagnostic's
population depends on it.

## Status

Recorded before Phase 5 test outcomes were inspected. Nothing in this file was
written, revised, or reordered after any Phase 5 result existed, because no
Phase 5 result existed at any point before this commit.
