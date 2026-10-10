# The landing read caps Context-Lift below a grid predictor

Recorded 2026-10-09, while writing the four Phase 5 modes. No Phase 5 result on
real data exists. No model has been trained on real data. Nothing in the frozen
design was changed in response to this note.

## What was observed

The end-to-end test of `evaluate` first asserted that Context-Lift
Transport-Only beats No-Warp-Copy on the synthetic test scene. It failed. On the
eight supported 5 degree rotation pairs, Context-Lift scored a centered cosine
of 0.842 and No-Warp-Copy 0.916. Aligned context depth equals ground truth on
that fixture, so a geometry error was the first suspect. It was not the cause.

## The decomposition

The fixture's features are an analytic field of the world point, sampled at
patch centers. At every supported landing three vectors can therefore be
compared. The first is the context patch vector Context-Lift carries. The
second is the field at the point the target camera sees at the landing. The
third is the frozen scoring target, the target grid read bilinearly at the
landing. Group means of per-pair centered cosines, at the image level:

| pairs | n | frozen CL | frozen No-Warp | CL vs field | field vs read | No-Warp vs field |
|---|---|---|---|---|---|---|
| rotation 5 deg | 8 | 0.8419 | 0.9163 | 0.9984 | 0.8427 | 0.8921 |
| sideways 0.15 m | 8 | 0.9828 | 0.9791 | 1.0000 | 0.9828 | 0.9784 |
| sideways 0.30 m | 6 | 0.9673 | 0.9299 | 1.0000 | 0.9673 | 0.9235 |
| sideways 0.45 m | 4 | 0.9453 | 0.8482 | 1.0000 | 0.9453 | 0.8195 |
| sideways 0.60 m | 2 | 0.9646 | 0.7558 | 1.0000 | 0.9646 | 0.7039 |

Context-Lift's geometry is exact. The vector it carries is the field at the
point the target sees, to four decimals on translation. On rotation the residual
comes from the fixture's shared depth map, which the 1.5 percent co-visibility
rule admits with up to that much depth error. Against the field, Context-Lift
beats No-Warp-Copy on every pair.

All of Context-Lift's shortfall under the frozen score is the read. The frozen
CL column equals the field-versus-read column, to four decimals on translation
and within 0.001 on rotation. Across one patch the fixture's field turns by
0.88 radians on average on the 2 m plane and 1.76 on the 4 m plane. So a
bilinear blend of four target patches is far from the field between patch
centers. No-Warp-Copy is itself a bilinear read, so at small displacement
it shares the target's smoothing. That is why it wins at 5 degrees and loses
once the displacement is large.

## Why this is structural

The frozen per-point rule scores every method against the target grid read at
the landing. A predictor emits a grid, and its grid is read with the same
bilinear weights as the target. A predictor that returns the true target grid
therefore scores exactly one, for any landing. On the supported fixture pairs
it scores 1.0000 with a worst deviation of 2.4e-7.

Context-Lift carries one patch vector. An off-grid read of the target grid does
not reproduce one patch vector even when the geometry is exact. On the fixture
the ideal predictor beats exact-geometry Context-Lift by 0.158 on rotation pairs
(range 0.113 to 0.201) and by 0.031 on translation pairs (range 0.012 to 0.079).

So under the frozen rule the two methods have different ceilings. The grid
predictor's is one. Context-Lift's is the interpolation residual of the target
features at the landing offsets. The headline gap S_CL minus S_Predict on
V_P5_pp contains this difference of ceilings as well as the learned-versus-
computed effect it is meant to measure. The term's direction is known. It
favors the predictor. Its size on DINOv2 features is not known, because it
depends on how fast DINOv2 features vary between adjacent patches.

The splat-pool comparison has a related asymmetry in another form. There the
predictor's cell is compared to the target cell directly, while Context-Lift's
cell is a pooled blend of the context patches whose pixels landed in it.

## What this is not

It is not an implementation defect. The integration gate's step 6 asserts that
both methods are scored against the same target feature location, and they
are. The reviewer's confirmation of landing-location scoring, recorded in
`design_correction.md`, concerns that shared target location. This note
concerns the form of each method's prediction, which that decision did not
address.

It is not a reason to change anything already frozen. The user's instruction
for Phase 5 is that comparator definitions, supports, estimands, and reporting
rules change only for a real implementation or protocol mismatch. The
implementation matches the protocol. Whether and how to respond is a design
decision, open for the user.

## Options, for the user's decision before any real-data result

Three responses were offered. On 2026-10-09 the user chose option 2, with the
first of its two candidates. It is pre-registered and implemented in
`landing_offset_diagnostic.md`. Options 1 and 3 were not taken, and the second
candidate under option 2 is not built.

1. Proceed as frozen and state the asymmetry as a limitation of the per-point
   estimand when the headline is reported.
2. Pre-register a model-free diagnostic that sizes the term on real data
   without touching any estimand. Two candidates exist.
   - Stratify Context-Lift with ground-truth context depth by the landing's
     distance to the nearest target patch center. Exact geometry has no offset
     dependence, and the change of representation between views has no evident
     reason to have one. The read does. So the slope of that curve sizes the
     read term. The strata are not randomized, so the curve would be read with
     that caveat.
   - Read the context-only splat grid that `evaluate` already builds for the
     splat-pool arm through the same landing read as the predictor. That is an
     explicit method in grid form, with exactly Context-Lift's information,
     facing the predictor's read. Its difference from Context-Lift on the
     primary support sizes the read term plus the splat operator's own cost.
     Phase 3 found that cost within 0.003 under ground-truth depth. Phase 4
     found the two operators diverge under estimated depth. So this candidate
     is cleanest at ground-truth context depth. Cells the splat leaves empty
     would need a frozen rule before any number is read.
3. Amend the headline estimand so both methods face the same read. That is a
   change to the frozen design and would need an amendment with a written
   rationale.

## Where it is pinned

`tests/test_phase5_modes.py`:

- `test_context_lift_carries_the_surface_point_the_target_sees` asserts the
  geometry is exact against the field, and that Context-Lift beats No-Warp-Copy
  on that truth.
- `test_landing_scoring_caps_context_lift_below_a_grid_predictor` asserts the
  ideal grid predictor scores one and exact-geometry Context-Lift scores below
  it, on every supported pair.

If the frozen read ever changes, the second test fails. That failure points
back to this note.
