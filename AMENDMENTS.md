# AMENDMENTS.md

Companion to PROTOCOL.md. PROTOCOL.md is never edited after the freeze commit;
every change to it, and every change to a value in configs/analysis.yaml, is
recorded here as a dated entry with a rationale.

The freeze commit is identified in FREEZE.md, together with the sha256 of each
normative file at that commit. Everything before it is protocol formation and
repair; everything after it is execution against frozen rules.

This file is a header and nothing else because no post-freeze amendment
exists, not because history was erased. The changes made during protocol
formation, including those that followed an independent audit's FAIL verdict,
are folded into PROTOCOL.md as protocol text and marked "Adopted before the
freeze commit"; the dated record of how each was reached is in the git history
that the freeze commit closes, reachable from the Stream D closure commits
named in FREEZE.md.

---

## Amendments

### A1, 2026-08-28. Which depth the pair-level parallax median reads.

PROTOCOL 3.2 defines pair-level parallax as the median of per-point baseline
over ground-truth depth across the pair's co-visible point set. It does not
name which view's depth enters the median. The implementation uses the
target-view depth of each co-visible point, read on the target grid where the
co-visible set is defined. The corrected Phase 3 run was produced under this
reading. This entry records the decision; no code or value changes.
Rationale: the co-visible set lives on the target grid, so the target-view
depth is the only per-point depth defined for every member of that set
without a second projection. Source: re-audit finding G-1.

### A2, 2026-08-28. Scope of the nonfinite-score rule.

PROTOCOL 3.2 permits exactly one nonfinite representation, the centered
columns of Mean-Feature rows, and says no other row may carry a nonfinite
score. Two clarifications of scope, both describing the shipped schema. First,
"score" means the agreement metrics the protocol defines. The coverage_mean
column is a splat-path diagnostic, undefined on per-point rows, and is
nonfinite there by design. Second, the centered intersection column of
Mean-Feature rows carries the same structural not-applicable as the other
centered Mean-Feature columns, under the same single representation. The
corrected Phase 3 run was produced under this reading. No code or value
changes. Source: re-audit finding G-2.

### A3, 2026-08-28. Phase 4 execution constants join configs/analysis.yaml.

PROTOCOL 4.5 states two numeric gates in prose: per-point scores under pure
rotation agree within 1e-5, and the forced-collision-order splat gate holds
within 1e-3. PROTOCOL's preamble requires gate tolerances to live in
configs/analysis.yaml, and the frozen file predates Phase 4 and does not carry
them. This amendment adds four keys, committed before any Phase 4 result
exists:

    rotation_gate_score_tol: 1.0e-5      transcribed from PROTOCOL 4.5
    rotation_gate_forced_tol: 1.0e-3     transcribed from PROTOCOL 4.5
    rotation_gate_coord_tol_px: 1.0e-3   new, defined here
    vggt_confidence_threshold: null      new, defined here

rotation_gate_coord_tol_px bounds the floating-point residue of the 4.5
coordinate identity. The geometry is exact; under pure rotation the lifted
depth cancels algebraically, so only rounding survives. The bound is set from
float32 pixel arithmetic at the 518 px frame size, about three orders of
magnitude above the observed residue and five below one pixel.

vggt_confidence_threshold freezes PROTOCOL 4.4's "any confidence threshold"
clause before results exist: null means no confidence gating, so Phase 4
validity is finite and positive depth alone. The exported confidence maps are
carried in evidence but gate nothing.

These keys sit outside the Phase 3 measurement identity. Adding them changes
the config and reporting digests and leaves the measurement digest untouched,
so the corrected Phase 3 parquet remains readable under PROTOCOL 3.12's rules.
Phase 4 run records carry their own identity over the fields Phase 4 consumes.

### A4, 2026-08-28. What the alignment scalar applies to, per path.

PROTOCOL 4.3 defines each alignment level's estimator but not its application
target, and 4.5's per-point gate presupposes an estimated-depth per-point
path without saying which frame's estimate that path lifts with. Frozen here,
before any Phase 4 result exists:

- Each alignment level yields one transform per pair. Level 1 is the
  leave-target-out scene scalar. Level 2 and the affine sensitivity are
  estimated from the pair's context image alone.
- The transform applies to whichever VGGT depth map serves as a transport
  input on a path: the context frame's map on the splat-and-pool path, and
  the target frame's map on the per-point path.
- The per-point path lifts the same Phase 3 target samples with the aligned
  VGGT depth of the target frame, read with the same arithmetic Phase 3 uses
  for ground truth, then warps and reads identically. The sample universe,
  eligibility filters, visibility buckets, and sample_id values remain the
  ground-truth-defined Phase 3 ones per 4.9; estimated depth changes only
  where the warp lands and which samples remain valid.

Rationale: the target-side lift is the only construction that preserves the
sample_id universe 3.2 requires every intersection to operate on, and it is
symmetric with Phase 3, where the per-point correspondence is also computed
from target-side depth. The target-exclusion invariant of 4.3 concerns
ground-truth depth entering estimators and is unaffected: the target frame's
VGGT estimate is estimator output, not ground truth, and all alignment levels
remain oracle calibration diagnostics. Under pure rotation the correspondence
is depth-free, so this choice is invisible to the 4.5 gate, which is the
property that makes the gate a pure correctness control.

### A5, 2026-08-29. Operationalization of the 4.8 localization masks.

PROTOCOL 4.8 names the masks and puts their thresholds in the config but
does not fix the measure. Frozen here, before any Phase 4 outcome is viewed,
implemented in lot.phase4:

- Depth boundary: the central-difference gradient magnitude of ground-truth
  target depth exceeds depth_boundary_gradient_threshold times the local
  depth. The threshold is relative because depth spans an order of magnitude
  across the scenes and the co-visibility tolerance is relative for the same
  reason. Pixels bordering invalid depth count as boundary. The mask is
  dilated by depth_boundary_dilation_px. A patch is a boundary patch when it
  contains any dilated boundary pixel; the dilation radius, not a majority
  rule, controls the band width.
- Texture: grayscale is the RGB channel mean scaled to [0, 1]; the statistic
  is the mean central-difference gradient magnitude over each 14 px patch;
  low texture means the statistic falls below texture_gradient_threshold.
- Both masks come from ground truth and the rendered RGB only. Estimated
  depth never defines a category, per PROTOCOL 4.9.

Also frozen here, from the same underdetermination family: the per-point
scored set (validity 5d) requires the estimated warp to land, meaning
positive depth in the context camera and a location inside the context
sampling box; the transported fraction reported per 4.4 is the scored
fraction of the ground-truth co-visible set per path; and the 4.4 identity
of surviving sets across the multiplicative levels is asserted on the
transport-validity notion (finite and positive depth), which is the notion
positive scaling provably cannot change. Landing can change with scale when
translation is nonzero, so the scored sets are reported as coverage, never
asserted equal.

### A6, 2026-08-29. Global depth-convention application.

Where authoritative model source establishes the depth-output convention
under PROTOCOL 4.1, that convention applies globally to every frame produced
by the pinned checkpoint and run. Per-scene or per-frame consistency-diagnostic
verdicts may not control conversion. The secant regression remains diagnostic
evidence only when source authority is definitive.

Why this needs saying rather than being left implicit. PROTOCOL 4.1 runs its
deterministic test "for the first rotation-program frame of every scene",
which is a per-scene computation, and it does not state in the same breath
that the decision the test informs is single. The implementation read that
phrasing literally and selected a conversion per scene, so one checkpoint
could have been given planar semantics in some scenes and ray-distance
semantics in others inside one table. A depth convention is a property of the
network's output semantics; it cannot vary by scene, frame, camera program,
camera angle, or dataset. The frozen text arguably entails as much, but it
did not say it, and what it did say produced the defect. Making the global
application referee-visible is the point of this entry.

This amendment does not alter the convention decision threshold, does not
redefine the regression, does not claim the observed data selected a
convention, and introduces no new outcome-dependent diagnostic. The threshold
depth_convention_slope_threshold is unchanged at 0.05.

The corresponding implementation change is a conformance bug fix, not a
method change: the run establishes one convention from source, every scene
reads it, no diagnostic verdict reaches the conversion path, and a permanent
test asserts the invariant against a deliberate case in which the per-scene
diagnostic verdicts disagree. The raw regression outputs are retained as
evidence of the diagnostic's limitation.

### A7, 2026-08-30. Pure-rotation forced rasterization structure.

PROTOCOL 4.5 intends the forced splat gate to separate reprojection from
discrete rasterization and collision effects, stating that its construction
"keeps missingness and collision ordering fully separated". Diagnostics on
the first gates run showed the implementation did not achieve that
separation: continuous coordinate differences well inside the
already-passing coordinate tolerance can cross a discrete floor(u + 0.5)
landing-cell boundary by one float32 ulp, changing candidate membership
before the forced winner rule is applied. The forced arm then silently
drops the winner, renormalizes the cell, and reports the loss as a score
residual. On the 2026-08-30 gates array all 18 scenes breached the 1e-3
forced tolerance this way, at residuals between 1.0e-3 and 2.9e-3; the
diagnosed pair (apartment_0 rotation_001 to rotation_008, level scene) had
exactly one flipped pixel at a boundary margin of 3.052e-05 px, one float32
ulp, and that single dropped pixel produced the entire 1.586e-3 residual
through a five-patch cell. Zero-baseline synthetic reproductions produce
exactly zero residual, so the invariant itself did not fail; its
implementation was incomplete.

Therefore, on the common-valid sample set, the forced gate freezes
Oracle-Transport's complete discrete rasterization structure for both arms:
each sample's target-cell assignment, per-cell candidate membership, and
the collision winner ordering. The existing coordinate gate continues to
test continuous reprojection independently, and the existing forced-score
tolerance rotation_gate_forced_tol is unchanged at 1e-3. No numerical
threshold moves under this amendment.

Ordinary estimated-depth rasterization remains unforced and is reported
separately. Because the unforced difference is now proven to contain both
landing-cell assignment changes and winner-order changes, the quantity
PROTOCOL 4.5 and Figure 2 call the collision-ordering tax is reported under
the umbrella name unforced rasterization tax, decomposed into a
landing-assignment component and a collision-ordering component that
telescope to the umbrella. Landing-cell flips are persisted as a
non-gating diagnostic per pair and level: the flip count, the flipped
fraction of the shared kept set, the maximum continuous coordinate
residual, the minimum distance of a flipped sample to its rasterization
boundary, and the count of affected cells. A real rasterization bug, a
convention mismatch, or a resize error moves coordinates by half pixels
and floods these counts; one-ulp float instability shows as isolated
flips at vanishing margins.

The implementation change is confined to the isolated forced-gate
diagnostic path. The frozen transport operator is untouched, and the
run-time assertion that the diagnostic copy with forcing disabled
reproduces the frozen plan remains in force. A permanent test drives a
deliberately boundary-adjacent sample across a landing boundary and
asserts the frozen structure pools identically to its donor while the
pre-amendment membership rule loses the winner, and that the flip is
counted at its vanishing margin. The 18-scene breach evidence is retained
under validation/evidence/phase4/.
