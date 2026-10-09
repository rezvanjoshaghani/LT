# The four Phase 5 modes: implementation record

Recorded 2026-10-09. No mode has run on real artifacts. The integration gate
has not yet run on Borah. No outcome on real data informed anything here.

## What each mode does

`overfit`. Draws the frozen tiny subset from fold 0's training scenes: eight
pairs, round-robin over the three regimes and the training scenes, taking the
first pair in frozen order that has any supported sample. Nothing in the choice
reads an outcome. It fits the subset with `run_tiny_overfit_gate` and writes a
receipt bound to the integration receipt, pass or fail. A failure exits 1, and
the receipt blocks every later mode.

`train`. One (fold, seed) task per array index. Plans are computed once per
pair: the primary support depends only on aligned context depth, ground truth,
and the cameras, so the full-image visibility work runs once rather than once
per pass. Each pass draws a fresh permutation seeded by the training seed and
the pass index. The checkpoint is saved through a temporary name and a rename.
A JSON record beside it carries its sha256, the training-config digest, the
config digest, the commit, and the plan census. A previous run's checkpoint and
record are archived as `*.superseded.N.*` first.

`controls`. Pose and depth shuffles over each fold's whole validation set, for
every seed, against one shared baseline. Examples are streamed and each
shuffled example takes its moved fields from a donor built beside it, so memory
stays near two batches. Each shuffle reports how many exchanges changed
nothing.

`evaluate`. One test scene per array task, through the model of the fold that
held it out, every seed scored separately. For every pair: the primary support
from Context-Lift and ground truth; TL-Reference and the splat arm recomputed
through Phase 4's code and reconciled with Phase 4's persisted rows; the region
masks recomputed and reconciled; then every score on its fixed population. One
parquet per scene, written once, with the run record inside it.

Every mode passes through `lot.phase5.require_receipts` before it reads
anything. A non-primary level is refused by `overfit` always, and by the other
modes until every test scene has its primary parquet.

## Defects found while building, all fixed

1. The formulation diagnostic compared TL-Reference's prediction for a cell
   center against the target read at Context-Lift's landing location, and
   counted TL-Reference once per landing sample. It now works at cell level.
   Both arms are scored against the cell's own target at its Phase 3 sample,
   one weight per common cell. Context-Lift's supported vectors in a cell are
   pooled as a normalized mean, the pooled-output form PROTOCOL 3.7 already
   defines. This follows the frozen decision that V_form intersects the two
   estimators on the target patch cell. The population is unchanged. The
   defect was latent because nothing produced TL-Reference predictions before
   `evaluate` existed.
2. The depth shuffle moved the aligned depth and left the context validity
   mask behind, although the mask is computed from the depth alone. The two now
   move together. A control also refuses any shuffle that changes the
   supervised sample count.
3. `tests/test_phase5_depth_equivalence.py` compared a calibration field named
   `scale`, which does not exist, under a `hasattr` guard. Only `affine_failed`
   was ever compared. It now compares every field the dataclass declares,
   including the ratio array.
4. The first version of `controls` shuffled inside fixed chunks of 64
   consecutive validation pairs, to bound memory. Consecutive pairs often share
   a context frame, so a depth shuffle could hand a pair its own depth map back.
   It now draws one derangement over the whole set, the frozen control's
   population, by streaming. A test asserts the streamed result equals the
   in-memory control bit for bit.

## Two test expectations that were wrong

The first end-to-end tests of `evaluate` asserted two thresholds that had no
analytic basis. Both failed. Neither failure was a code defect.

Context-Lift was expected to beat No-Warp-Copy on every supported translation
pair, and to exceed 0.9. On 5 degree rotations it scored 0.842 against 0.916.
The geometry is exact. The shortfall is the landing read, and it is recorded
as a finding in `landing_read_asymmetry.md`. The test now checks the landings
against an independent float64 reprojection, checks the geometry against the
fixture's analytic field, and pins the read asymmetry.

TL-Reference's formulation score was expected to exceed 0.9. TL-Reference reads
the context grid off-grid, so on this fixture it pays the same kind of
interpolation residual on the context side. The test now checks the identity
that does hold: TL-Reference on aligned depth equals TL-Reference on ground
truth, cell set exactly and score within 1e-4.

## Why 28 of 92 fixture pairs carry support

The fixture writes one depth map for every frame. That map is consistent across
views for sideways translation, and nearly so for 5 degree rotation. All 20
sideways pairs and all 8 rotation pairs at 5 degrees carry support. None of the
52 axial or diagonal translation pairs does, and none of the 12 rotation pairs
at 10 degrees or more. On those pairs no landing passes the 1.5 percent
co-visibility rule. Phase 4's per-point arm scores zero cells on exactly the
same pairs. So the empty support is a property of the fixture, shared by both
estimators.

The same shared map makes the fixture's depth shuffle a no-op: every aligned
context map is the same array. `controls` now counts unchanged exchanges, and a
test asserts all of them are unchanged on this fixture and the degradation is
exactly zero.

## Verification

- `tests/test_phase5_modes.py` drives all four modes end to end on three
  synthetic scenes written to disk, with real Phase 3 and Phase 4 runs on the
  test scene. Evaluation reconciled all 92 pairs against the Phase 4 parquet:
  worst per-point residual 0.0, worst splat residual 0.0.
- The command line refuses every mode without its receipts, and refuses an
  undeclared level.
- The launcher's submission path was exercised with a stub `sbatch`: account,
  partition, output pattern, array range, and passed-through arguments arrive
  intact, including the empty case under `nounset`.
- Full suite at commit: 598 passed, 3 skipped. The skips need encoder weights
  (two) or Habitat-Sim with Replica (one), none of them present off the
  cluster.

## What has not run

Nothing here has touched real artifacts. The next step is
`./scripts/run_phase5.sh check` on Borah, and nothing after it runs until it
passes.
