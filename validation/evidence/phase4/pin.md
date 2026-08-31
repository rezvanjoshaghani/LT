# Phase 4 input-state pin (execution plan, Stream G, step 1)

Recorded 2026-08-29, before any VGGT depth is consumed by Phase 4. After this
pin, repository code and configuration are fixed for Phase 4: no source,
test, analysis, transport, launch-template, or .gitignore change until Phase 4
closes. The `check` mode of scripts/run_phase4.sh re-verifies every line of
this file on the cluster before anything runs.

## Commits

- Normative freeze commit: `d4ed1017bd2daca2871da28900b5b4a6a7ff92b6`
  (FREEZE.md; Stream D closure commits precede it).
- Provenance-hardening commit: `0992f7f` (evidence jobs refuse a dirty
  worktree; the SLURM-log ignore rule was already committed at `4787937`).
- Validation package commit: `aa977f4` (Stream F re-audit, pass with
  findings, preserved before any Phase 4 work).
- Phase 4 execution commits. The measurement code was introduced at `b964c42`
  (lot.phase4, lot.phase4_report, the A3 config keys, tests, launch scripts,
  and the validator 2.3 script) and this pin was written at `6ac138d`. Three
  commits follow, none of which touch a measurement path, and the runs are
  made at the last of them:

      96c66a3  cluster readiness. lot/phase4_report.py's paired scene
               bootstrap was re-pooling every record per replicate, 60 s per
               pooled cell; it now sums per-scene (sum, count) pairs, verified
               equal to the pooled-records reference to 3.6e-15 with the
               replicate still recomputing each quantity whole. Also the smoke
               mode's CUDA guard, the gates array mode, and the runbook.
      ea0730b  mode bits on the two launch scripts, content unchanged.
      this one bookkeeping: the inspect mode read vggt.__file__, which is
               None for a namespace package, plus this record.

  src/lot/phase4.py is byte-identical to its `b964c42` form at every one of
  them, so nothing that decides a row has moved since the pin was written.
  The Phase 4 measurement digest below is unchanged across all four commits,
  and every run record carries the commit it actually ran at.

## Frozen artifacts

sha256 of each normative file as committed at the freeze commit, verified
against FREEZE.md at pin time and re-verified by `check` on the cluster:

    517cc4924f8b770c2d27c9f9fecb761c634a920fd21a77ab44e92fe28473eee4  PROTOCOL.md
    bddd31e9d294778f10848028e4755ba56e6ea58f1e633b57eb4d09b29bbb4493  AMENDMENTS.md
    91f3a82ff066bcb029b9df7b9ff9c3da4bba31ea9604d1d17cea48a660b440e6  configs/analysis.yaml
    6e4897ff98b2061a4e8c544dfd30126e8b0b5cb561d4b8e306400f210a7c452e  VALIDATION.md

PROTOCOL.md and VALIDATION.md are byte-identical to their frozen blobs at the
execution commit. AMENDMENTS.md and configs/analysis.yaml differ from their
frozen blobs exactly by the enumerated amendments below, which is the
amendment mechanism working as designed; their live sha256 at the execution
commit:

    392ccc5fcc03c0c3ce89c049de50e9614790337ce815735b8bfb6db87bd7f1d3  AMENDMENTS.md
    41990264fe5c08685b954b93f237a26fb85e2282a073ee80b5551dd136d6dfe0  configs/analysis.yaml

## Post-freeze amendments in force

A1 (parallax median reads target-view depth), A2 (nonfinite-score scope),
A3 (the four Phase 4 execution keys), A4 (alignment application and the
per-point estimated lift), A5 (4.8 mask operationalizations and the 5d
landing rule). All predate any Phase 4 result.

## Identities

- Phase 3 measurement digest: `27244e6481d521159e513f2ea8799482`
  (unchanged by A3; the corrected Phase 3 parquet loads under the amended
  config, verified).
- Phase 4 measurement digest: `1579714398feff4771a9981e5f427c8a`.
- Full config digest at pin: `146ae78d35fff7425131b5e793c56022`;
  reporting digest: `949d14c5c88d5d8c5514ba6ab1db3ee9`.

## Input artifacts

- Corrected Phase 3 evaluation outputs (the frozen population Phase 4
  reuses): aggregate sha256
  `7de0ae525087806c7a7c1e147691934d67f57e7bb9dbedf61e596ee6fd6bc9a6` over 42
  files in sorted-path order; per-file digests in
  validation/evidence/reaudit/outputs_pin.txt.
- DINOv2 feature cache: per-scene `features_digest` values as carried in the
  18 pinned Phase 3 run records (weights fingerprint
  `1159bd1e21ae359a232648228319ab05`; checkpoint declared unpinnable per
  3.12; hub code revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`).
  Evaluation re-verifies each scene's bytes against its digest before
  scoring.
- VGGT depth cache (Phase 2 export, 518 x 518 with confidence): weights
  fingerprint `b9e29c09ce793d15bf3abdc14048838a`, weights revision
  `860abec7937da0a4c03c41d3c269c366e82abdf9`, inference code revision
  `a288dd0f14786c93483e45524328726ab7b1b4ce`. Per-scene `depth_digest`
  values live in the cache metadata on the cluster; `check` re-reads them
  and every Phase 4 run record carries them forward.
- Mean vector: reused from outputs/experiment_zero with its provenance
  record and vector digest; the loader refuses a mismatch.
- Manifest-set content hash, computed on Borah 2026-08-29 and recorded here
  rather than only in gitignored evidence, closing the deferral that re-audit
  dated note 1 left open:

      99bf3e938ffeeb9a2c82a919ee37824c632a539901fb783fe37064e1f14b9046

  over 18 manifests, as the sha256 of the sorted
  `sha256sum data/replica_renders/*/manifest.json` listing. The per-file
  listing is outputs/phase4_rung1/evidence/manifest_files.txt.

## Structural guarantees asserted before the run

- Target-frame ground truth cannot enter any Phase 4 calibration estimator:
  Level 1 excludes the target by construction with a per-record assertion,
  Level 2 and affine read the context image only, and the suite pins both
  (tests/test_phase4.py). PROTOCOL 4.3.
- The forced-collision machinery cannot change ordinary transport: it is an
  isolated copy whose forcing-disabled form is asserted equal to the frozen
  plan at run time and in the suite.
- The 4.5 gate tolerances, the confidence rule, and the mask
  operationalizations are config- and amendment-resident; nothing is
  introduced or loosened at run time.

## Verified on Borah, 2026-08-29

`./scripts/run_phase4.sh check` passed at `ea0730b`: frozen blobs verified
against FREEZE.md, all five amendments listed, clean worktree, all four
inputs present, both caches re-read and valid across 18 scenes (DINOv2
`1159bd1e21ae`, VGGT `b9e29c09ce79` at revision `860abec7...` and code
`a288dd0f...`, matching the pins above), Phase 3 measurement digest
`27244e6481d521159e513f2ea8799482` and Phase 4 measurement digest
`1579714398feff4771a9981e5f427c8a` both as recorded, suite 262 passed and 3
skipped. Frame counts corroborate the Phase 1 record independently: 17 scenes
at 288 plus frl_apartment_2 at 240 is 5,136.

Depth convention, PROTOCOL 4.1's primary authority: established as planar
camera-z from the installed VGGT source. `vggt/utils/geometry.py`'s
`depth_to_cam_coords_points`, which is what VGGT's own
`unproject_depth_map_to_point_map` uses to turn `predictions["depth"]` into
points, assigns `z_cam = depth_map` directly with `x_cam` and `y_cam` scaled
by depth over focal length, term for term the same as `lot.geometry.unproject`.
A ray-distance head would have to rescale along the ray there and does not.
Evidence: outputs/phase4_rung1/evidence/vggt_source_inspection.txt and, from
the corrected implementation, source_authority.json.

## Re-pin after the depth-convention closure, 2026-08-29

The first `convention` run stopped: the secant regression disagreed with the
source on 5 of 18 scenes. Diagnosing it showed the regression is unstable
within a single scene across camera rotations, and that the implementation
was selecting the conversion per scene. The closure changed pinned Phase 4
measurement code, so the state is re-pinned here.

- Previous pinned HEAD: `31227b5`.
- Bug-fix and closure commit: the commit containing this section.
- Normative freeze commit: unchanged, `d4ed1017bd2daca2871da28900b5b4a6a7ff92b6`.
- Applicable amendment: A6, global depth-convention application.
- Convention decision: `planar_z`, authority `source`, no cosine conversion.
- Threshold `depth_convention_slope_threshold`: unchanged at 0.05.
- Discarded outputs: none. The stop fired before any alignment level ran, so
  no Phase 4 eval parquet, table, or figure existed. The historical
  `convention_report.json` from the stopped run is preserved unmodified; the
  corrected stage writes `secant_diagnostic.json`, `source_authority.json`,
  and `convention_record.json` under new names rather than overwriting it.
- Tests: 267 passed, 3 skipped, including five new convention tests. The
  invariant test drives a deliberate case in which per-scene diagnostic
  verdicts disagree and asserts the applied convention stays globally
  identical, plus that the planar-z path applies no conversion where the
  conversion would have moved the map materially.
- Unrelated Phase 4 code: unchanged. The diff touches only the convention
  path in `src/lot/phase4.py`; transport, alignment, masking, binning,
  scoring, and feature code are untouched, and `src/lot/phase4_report.py` is
  not modified.

## Re-pin after the Phase 4 code review, 2026-08-29

A second review against the closure commit found ten defects, all fixed
before any evaluation ran, so nothing was discarded. The state is re-pinned
at the commit containing this section; no Phase 4 eval parquet, table, or
figure exists yet.

What changed in measurement code (`src/lot/phase4.py`):

- Phase 3 inheritance is row-level, not name-level: per pair, the recomputed
  validity masks must equal the persisted Phase 3 masks bit for bit and the
  recomputed Oracle and No-Warp ceilings must reproduce the recorded scores
  within `PHASE3_SCORE_RECON_TOL = 1e-5`, with the worst residual carried in
  the run record. Phase 3's feature-cache digest and measurement identity are
  compared before any pair is, and the Phase 3 source identity travels in
  every Phase 4 run record.
- The frozen 5a validity rule is applied to the depth maps themselves
  (invalid pixels become NaN), so a nonnull confidence threshold would govern
  transport and scoring, not only calibration. Null threshold: no behavior
  change.
- The multiplicative-level identity is asserted on the boolean sets, not
  their counts.
- The forced-collision gate checks raw and centered cosine; the unforced arm
  runs on the same common source population as the forced one, so the
  collision-ordering tax carries ordering only, not missingness; the forced
  scores are persisted for Figure 2.
- Run records carry the feature-encoder identity triple, the mean-vector
  digest, the manifest digest, and the convention record fields; resume
  refuses on any of them; rows carry the configured encoder.

What changed in reporting code (`src/lot/phase4_report.py`): the report
binds to the active Phase 4 measurement digest; refuses duplicate rows,
partial matched arms, and mask-mismatched arms while counting legitimately
empty scored sets; the cross-path disclosure uses the persisted intersection
columns with a paired scene bootstrap on dM and includes the selection
differential; affine accounting is per scope and an all-failed scope still
emits its accounting row; localization contrasts intersect their arm
populations per pair and carry their own support; Figure 1 shows the Phase 3
reference ceiling; Figure 2 plots the forced-order control; cell summaries
carry the secondary camera-pair bootstrap and the comparison-weighted
diagnostic, and the reporting-digest note is printed when reporting values
moved.

Tolerances: no frozen threshold changed. `PHASE3_SCORE_RECON_TOL` is a new
reconciliation bound for a new provenance check, an order of magnitude above
the ledger's observed 2e-7 reconstruction and unrelated to any gate.

Tests: 274 passed, 3 skipped, including tampered-mask, tampered-score,
stale-provenance, partial-arm, and masked-validity regression cases.

## Cluster execution order (all modes refuse a dirty worktree)

    ./scripts/run_phase4.sh check
    ./scripts/run_phase4.sh inspect          # VGGT depth semantics, evidence
    ./scripts/run_phase4.sh convention [doc-verdict]
    ./scripts/run_phase4.sh gates            # PROTOCOL 4.5, stop on failure
    sbatch --array 0-17 scripts/phase4.sbatch configs/phase4.yaml
    ./scripts/run_phase4.sh figures
    ./scripts/run_phase4.sh smoke            # GPU node, permanent 3.1 tests
    ./scripts/run_phase4.sh validate23       # Stream F closure, Addendum E

The validator 2.3 script was rehearsed on this machine against a synthetic
scene evaluated by the real pipeline: 280 of 280 rows reproduced with
bit-identical masks and every metric within 7.6e-7 of the frozen 1e-4
tolerance (validation/borah_check_2_3.py docstring records the independence
boundary).

## Validator 2.3 arithmetic-contract fix, 2026-08-29

The full apartment_0 audit returned FAIL with a proven mechanism: 918 of 930
pairs reproduced bit-identically, and 12 pairs each differed by one sample
sitting on a decision boundary. The diagnosed case (cell 1365) had its warp coordinate one float32 ulp outside the
sampling-box edge, margin -3.052e-05 px against edge 510.5, which is exactly
the spacing of float32 at that magnitude. Every one of the 24 metric
failures was Neighbor-Patch on those same near-edge coordinates: a one-ulp
option coordinate flips the admissible set, the hash ranks a different
number of options, and the direction lands elsewhere.

Root cause, in the validator and not the pipeline: three mask-deciding
computations still ran outside the run's arithmetic contract. The pose
inverse was computed in float64 and rounded (the run inverts the float32
matrix with the float32 block formula), the depth-at-center bilinear read
ran in float64 and rounded (the run interpolates in float32), and the
[N, 3] point transform went through numpy's matmul (the run's kernel is
torch's). Each lands one ulp away from the run at some coordinates; only
coordinates that straddle a boundary become visible, which is why 98.7
percent of pairs still matched.

Fix, entirely inside validation/borah_check_2_3.py: every computation that
decides a mask now runs under the run's declared contract, float32 with the
run's operation order, through torch kernels where rounding is at the
library's discretion. Covisibility, the one-surface test, the sampling-box
warp, the neighbour options, the splat landing chain including z-buffer
ties, and the per-patch covisible fraction are all mirrored; the formulas
remain this script's own, written from the protocol text, per VALIDATION.md
ground rule 4. Score accumulation is unchanged: independent float64 under
the frozen 1e-4. No tolerance moved, and no mask slack was added; the mask
comparison stays bit-for-bit.

Evidence the fix holds and the check keeps its teeth, both in
tests/test_borah_check_2_3.py against a parquet written by the real
evaluator on the analytic scene:

- audit_scene returns PASS with zero mask and count mismatches over every
  scored pair, all metrics within 1e-4.
- The mirrored chain is bit-identical to the pipeline per pair: covisible
  mask, per-point and splat selections, every warp coordinate, the
  admissible neighbour sets, and the Neighbor-Patch locations, compared
  with array equality on float32 bits.
- Teeth check (run once, not committed): re-injecting the float64 bilinear
  makes the bitwise test fail at 3.8e-06 max drift, the same ulp mechanism
  as the Borah failure.

One clarification the test surfaced: the analytic fixture stores float64
depth on disk, and evaluate_scene casts every depth map to geometry_dtype
float32 on read, so the run's chain is float32 regardless of storage. The
validator applies the same cast. On Borah the stored renders are float32
and the cast is a no-op.

Tests: 276 passed, 3 skipped.

Next on Borah: git pull, then rerun
    ./scripts/run_phase4.sh validate23
in full. If the twelve pairs now reproduce bit-for-bit, 2.3 closes as PASS
with no residual to classify. Any remaining disagreement gets --diagnose
and a severity classification under the frozen VALIDATION.md rules, decided
with the user, before Phase 4 resumes at Stream H.

Rerun verdict, 2026-08-29: PASS at c1bed4b. 930 pairs, 9300 rows, zero
mask and count mismatches, max metric abs diff 5.1e-07 against 1e-4. The
twelve boundary pairs reproduce bit-for-bit under the corrected contract.
Stream F is closed and the STOP on Phase 4 is lifted; the closure report
is validation/evidence/phase4/closure.md.

## Re-pin after Amendment A7, 2026-08-30

The first gates array (job 3193085, HEAD 1eaf1601) breached the 4.5
forced-collision-order gate on all 18 scenes, at residuals between 1.0e-3
and 2.9e-3 scattered across levels. The instrumented decomposition of the
first breaching pair proved the mechanism: exact-zero baseline, bitwise
Oracle reproduction at two of four levels, and at the breaching levels
exactly one source pixel landing one cell over at a boundary margin of
3.052e-05 px, one float32 ulp, whose dropped winner renormalized a
five-patch cell into the whole 1.586e-3 residual. Every lost winner was a
flipped pixel, so the forced-key machinery was clean; the construction
was incomplete. Evidence preserved verbatim under
validation/evidence/phase4/gates_breach_2026-08-30/.

Amendment A7 resolves it by freezing Oracle-Transport's complete discrete
rasterization structure for both forced arms: cell assignment, candidate
membership, and winner ordering. The gated score comparison is now the
true invariant 4.5 promises and reads exactly 0.0; rotation_gate_forced_tol
stays 1e-3, untouched. The pre-A7 membership arm survives as the midpoint
of the tax decomposition: the unforced difference is reported as the
unforced rasterization tax, split into a landing-assignment component and
a collision-ordering component that telescope to the umbrella, and
landing-cell flips are persisted per pair and level as a non-gating
diagnostic (count, fraction, affected cells, max continuous coordinate
residual, min boundary margin). The frozen transport operator is
untouched and the forcing-disabled identity assertion remains in force.

Code: splat_plan_detail gains forced_structure (exclusive with the
membership mode) and exposes per-pixel landings, continuous coordinates,
and boundary margins; the gate block runs four arms (oracle, frozen
structure, membership midpoint, unforced) and hard-fails if the frozen
structure produces weights that differ from its donor. New permanent
test: a deliberately boundary-adjacent sample driven across a landing
boundary, where the frozen structure pools identically to the donor while
the membership rule loses every winner, flips are counted at their
vanishing margins, tampered structures naming unkept sources are refused,
and cross-grid structures are refused.

Tests: 277 passed, 3 skipped.

Expected on the rerun: gates PASS 18 of 18 with forced residuals exactly
zero, flip counts of order zero to a few per pair reported in the
evidence, and the coordinate and per-point score gates unchanged.

## Reporting-layer bootstrap vectorization, 2026-08-30

The 18-scene evaluation array completed and the figures step was
intractable: 2,221,455 rows, 266,808 paired records, and roughly 16,675
camera pairs. The camera-pair bootstrap ran a Python loop over units per
replicate, at 153 s per pooled cell measured at that shape, which put the
whole reporting step at hours. The scene bootstrap over 18 units was
0.1 s and never mattered.

The bootstrap is now vectorized: per-unit sums and counts become arrays
once per cell, a block of replicates becomes a multiplicity matrix, and
sum-of-sums over sum-of-counts is two matrix products. Measured 0.55 s
for the 16,675-unit cell, 278 times faster.

Nothing about the resample changed. The draw is still successive
integers(0, n, size=n) from a generator seeded with bootstrap_seed; a
batched call reproduces that stream bit for bit, pinned by a test at
three sizes. The matmul accumulates in a different order from sequential
Python addition, so replicate means agree to floating-point rounding
rather than bit for bit; the same test pins the agreement at 1e-9
relative against a reference loop, and a second test pins that an
all-empty cell resamples to nan rather than to zero over zero. Point
estimates are untouched: they still come from pooled_means. No table,
figure, or threshold existed before this change, so nothing published
moved.

weighted_means was also restructured from one pass per field to one pass
over the records. Each field still accumulates in record order, so that
one is bit identical.

Tests: 279 passed, 3 skipped.

## Addendum E closed, 2026-08-30

Both closure items pass. Validator 2.3: PASS on apartment_0, 930 pairs,
9300 rows, zero mask and count mismatches, worst metric residual 5.1e-07
against the frozen 1e-4. Encoder smoke on a GPU node inside an
allocation: 3 passed, the PROTOCOL 3.1 permanent real-weight tests,
DINOv2 grid orientation and shape, VGGT sees one frame at a time, and
VGGT batching does not mix frames. Evidence:
outputs/phase4_rung1/evidence/encoder_smoke.txt and
validation/evidence/reaudit/borah_check_2_3.json.

The first smoke attempt failed for a reason outside the repository: srun
inherited the interactive session's task count and started sixteen copies
of the suite on one device, which exhausted its memory before VGGT
loaded. Rerun with an explicit single task, it passes. The runbook now
carries the flag.

Phase 4 rung 1 is complete. PLAN.md's acceptance for this phase is the
first two ladder rungs plotted together and an error-localization
visualization; both exist as
outputs/phase4_rung1/figures/phase4_figure1_tax_vs_parallax.png and
phase4_figure4_localization.png, regenerable from the tables alone.
Results and the two recorded anomalies are in FINDINGS.md under Phase 4
rung 1.
