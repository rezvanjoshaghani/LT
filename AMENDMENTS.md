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
