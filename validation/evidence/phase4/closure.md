# Phase 4 rung 1 preflight closure report

Date: 2026-08-29. Branch: repair/validation-streams-abc.
Fix state: 93b3051. The validator 2.3 rerun executed at c1bed4b from a
clean worktree; the convention record binds its own checkpoint and source
identities and is re-verified at every consumption.

This report closes the depth-convention resolution task and the last open
Stream F item, and lifts the STOP on Phase 4. No Phase 4 scientific output
existed at any point before this closure: no eval parquet, no table, no
figure. Every result will postdate every fix described here.

## 1. Depth convention: planar_z, from source authority

The deterministic secant regression disagreed with the documented
convention on 5 of 18 scenes. The per-frame diagnostic then showed the
regression is unstable within scenes: office_1 flips from ray-distance to
planar across eight frames of one viewpoint's rotation program, so the
per-scene verdict is confounded by scene geometry, not evidence about the
checkpoint. FINDINGS.md records this conservatively as a diagnostic
limitation. It does not claim VGGT has an intrinsic radial depth error.

- `depth_convention_slope_threshold` is unchanged. No threshold was
  retuned after seeing results.
- The convention is established from the pinned source instead, which
  PROTOCOL 4.1 gives primary authority. `vggt/utils/geometry.py`,
  function `depth_to_cam_coords_points`, assigns `z_cam = depth_map`
  directly and scales x and y by depth over focal length. That is
  planar_z, unambiguous. The authority commit a288dd0f... equals the
  `code_revision` recorded in the depth cache metadata, so the source
  cited is the source that produced the cached depth.
- One checkpoint, one convention. The convention record binds the
  checkpoint identity triple and the source commit, and every consumer
  re-verifies the binding against the cache it is about to read. The
  earlier per-scene conversion application was a genuine bug, removed;
  a global-convention invariant test now holds it. Amendment A6 records
  the rule. The frozen originals are untouched.
- The failed diagnostic evidence is preserved in the repository, not
  cleaned up: the flagged convention report and the per-frame diagnostic.
- Cluster verdict: `convention planar_z` PASS, 18 of 18 scenes, at the
  pinned authority.

## 2. Validator 2.3: PASS, with the FAIL explained and preserved

The first full apartment_0 audit (checkout 0443fc2) returned FAIL: 918 of
930 pairs bit-identical, 12 pairs each off by exactly one sample, and all
24 metric failures Neighbor-Patch. The diagnosis localized the mechanism:
the disputed cell's warp coordinate sat one float32 ulp outside the
sampling-box edge 510.5, margin -3.052e-05 px, which is the float32
spacing at that magnitude. The Neighbor-Patch failures were the same
mechanism on the option coordinates, where a one-ulp flip changes the
admissible set the direction hash ranks. Evidence preserved at
`validation/evidence/reaudit/borah_check_2_3_fail_at_0443fc2.json`.

Classification: a validator-side arithmetic defect, not a pipeline
defect. Three mask-deciding computations ran outside the run's declared
contract: a float64 pose inverse rounded to float32, a float64 bilinear
depth read rounded to float32, and numpy matmul where the run used torch
kernels. Each lands one ulp off the run at some coordinates; only
coordinates straddling a boundary become visible.

The fix (93b3051) is confined to `validation/` and `tests/`. Every
computation that decides a mask now runs under the run's arithmetic
contract: float32, the run's operation order, torch kernels where
rounding is at the library's discretion. The formulas remain the
validator's own, written from the protocol text, per VALIDATION.md ground
rule 4. Score accumulation stays independent float64 under the frozen
1e-4. The mask comparison stays bit-for-bit. No tolerance was widened and
no mask slack was added.

Verification before the rerun: `tests/test_borah_check_2_3.py` audits a
parquet written by the real evaluator on the analytic scene (PASS, zero
mismatches) and compares the mirrored chain against the pipeline bitwise
per pair. Re-injecting the float64 bilinear fails the bitwise test at
3.8e-06 drift, the same ulp mechanism, so the test has teeth against
exactly this bug class.

Rerun at c1bed4b, full scene, clean tree:

    pairs 930, rows 9300
    mask mismatches 0, count mismatches 0
    max metric abs diff 5.1e-07 (tolerance 1e-4)
    pair fields: rotation 3.8e-13, parallax 2.8e-05, covisible 3.0e-08
    mean vector max abs diff 4.8e-07
    verdict PASS

No residual remains to classify under VALIDATION.md's severity rules.
The parallax residual of 2.8e-05 is the expected definitional difference
between torch's lower-middle median and numpy's averaged median on even
counts, far inside tolerance, documented and not actionable.

## 3. Verdict

The STOP on Phase 4 is lifted. Stream F is closed; validator 2.3 was its
last item. Phase 4 resumes at Stream H in the runbook order: the
pure-rotation gates array, then the evaluation array, then tables and
figures, then the GPU smoke tests. Nothing in that sequence tunes
anything; a tax that is flat, reversed, or large everywhere is a valid
rung 1 result if the gates pass.
