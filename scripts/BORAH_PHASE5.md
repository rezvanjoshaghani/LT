# Borah runbook: Phase 5, rung 2

Every command runs from the repository root on Borah. Each mode refuses a dirty
worktree, so sync and commit before starting. The pinned execution state is
`validation/evidence/phase5/pin.md`, and the design correction that defines the
phase's comparator is `validation/evidence/phase5/design_correction.md`.

Read the design correction first. Phase 5 does not compare Predict-with-Depth
against the accepted Phase 4 per-point Transport-Only. That estimator lifts the
target frame's estimated depth under Amendment A4, which the predictor is
forbidden to see, so the comparison would confound a learned-versus-explicit
cost with an information asymmetry. The headline comparator is
**Context-Lift Transport-Only**, which uses context-frame depth only. The Phase 4
estimator is retained as **TL-Reference**, a formulation diagnostic.

## 0. Environment

Set once per shell. Account and partition are never hard-coded anywhere.

    export SLURM_ACCOUNT=<account>
    export SLURM_PARTITION=<partition>
    export LOT_ENV=lot-encode            # the Phase 2/3/4 env, unchanged
    export MAMBA_ROOT_PREFIX=/bsuscratch/$USER/micromamba

Inputs that must already be present, all of them Phase 4's:
`data/replica_renders`, `cache/features` (both encoders, the VGGT cache
including its depth export), `outputs/experiment_zero` (the corrected Phase 3
run, whose global mean vector Phase 5 reuses), and `outputs/phase4_rung1` (the
accepted Phase 4 run, whose context-image-scale scored-cell masks define the
secondary operational support and the formulation-diagnostic population).

Phase 5 re-encodes nothing. It consumes the existing caches.

## 1. Verify the pin and the inputs

    ./scripts/run_phase5.sh check

Verifies the four frozen blobs against FREEZE.md at the freeze commit,
enumerates the amendments, asserts a clean tree, re-reads both caches against
their recorded digests, confirms the Phase 3 and Phase 4 measurement
identities, recomputes the manifest-set hash, prints the fold digest and the
architecture and training-config digests, and runs the suite.

Expect: frozen blobs verified; A1 to A7 listed; Phase 3 measurement digest
`27244e6481d521159e513f2ea8799482`; Phase 4 measurement digest
`1579714398feff4771a9981e5f427c8a`; fold digest
`25f0c03f72d58cc8e3ff2d8d4123241f6459d4ed50d3c303c5dc108bb0e19865`; suite
green.

`check` is also where the cluster-resident pin fields are filled: the per-scene
DINOv2 and VGGT cache digests, the aligned context-depth digest, and the Phase 4
artifact hashes. It writes them to
`outputs/phase5_rung2/evidence/pin_cluster.txt`, which is the file that closes
the deferrals `validation/evidence/phase5/pin.md` records.

A frozen-blob mismatch, a cache-digest mismatch, or a measurement-identity
mismatch stops everything.

## 2. Validate the new comparator on real geometry

    ./scripts/run_phase5.sh gates

Stream R step 6 on the real pure-rotation regime, not only on the analytic
scenes the suite covers. For every rotation pair, Context-Lift Transport-Only's
landing must equal the analytic rotational homography within the already-frozen
`rotation_gate_coord_tol_px`, and must be unchanged when the depth map it is
given is replaced by a different one, because a zero-translation map is depth
free.

This is the gate that licenses using the comparator scientifically. A failure
is a stop: it would mean the forward map is wrong, and every headline number
would be measuring that instead of the science.

Expect: PASS on all 18 scenes, residuals far below the tolerance, and exact
invariance to the substituted depth.

## 3. The tiny-subset overfit gate

    ./scripts/run_phase5.sh tiny

Stream U step 14. Eight pairs from training scenes only, spanning the three
camera regimes, at the frozen threshold of centered cosine 0.98.

A failure is a stop and the phase does not continue. The reason is not that a
failing model is uninteresting; it is that Phase 5's whole question is whether
learning the transformation costs accuracy, and a trunk that cannot fit eight
pairs would answer a different question. The gate makes the later comparison
interpretable rather than making it look good.

## 4. Train

    sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
           --array 0-8 scripts/phase5_train.sbatch configs/phase5.yaml

Nine tasks: three folds by three seeds. Each task trains on its fold's nine
training scenes, selects its checkpoint on its three validation scenes by
validation centered cosine, and never touches the six test scenes. The seal is
asserted inside the training loop, so a leaked scene raises rather than
training.

Nothing about a test scene may influence anything here. If a training run is
restarted, changed, or retuned in response to a test metric, that run is
invalid under Stream S step 9 and must be discarded rather than reported.

## 5. Input-use controls

    ./scripts/run_phase5.sh controls

Stream U step 17, on validation scenes. Pose shuffle and depth shuffle against
one shared baseline.

These are diagnostics. A shuffle that barely moves the score is reported exactly
as observed, and is evidence about whether the network uses that input. It is
never a reason to retrain, and it is never a threshold anything must pass.

## 6. Evaluate

    ./scripts/run_phase5.sh evaluate

Streams V and W. For every test scene, through the model of the fold that held
it out, at each of the three seeds separately.

Three populations, fixed before any scoring and never merged:

- the primary per-point support, decided by Context-Lift Transport-Only, shared
  by CL-Transport, Predict-with-Depth, and No-Warp-Copy;
- the formulation support, its intersection with the Phase 4 target-lift set on
  the target patch cell, used only for TL-Reference against CL-Transport;
- the accepted Phase 4 context-image-scale splat-pool scored cells, used
  unchanged for the secondary operational comparison.

The predictor cannot narrow the primary support. A nonfinite prediction on a
supported sample is scored as a model failure and counted, not dropped.

## 7. Tables and figures

    ./scripts/run_phase5.sh report

Stream AC. The headline table, the formulation-reference table labelled as a
diagnostic rather than an estimand, the operational splat-pool table, and the
five figures. Every figure regenerates from the tables alone.

## 8. Acceptance

    ./scripts/run_phase5.sh accept

Stream AD step 45, re-derived from the shipped artifacts rather than asserted.
Every condition is checked, and any failure is a stop that forbids interpreting
the model-versus-explicit difference.

## What must not happen

The two mistakes this phase is most exposed to, both of which the code refuses
rather than merely discouraging:

- Substituting the Phase 4 target-lift score into the headline learned gap.
  `lot.phase5_estimands.assert_not_target_lift_headline` raises. That
  substitution would reintroduce exactly the confound the redesign removed.
- Letting the predictor's output influence the support it is scored on. The
  support is built from the explicit comparator and ground truth before any
  prediction is read, and a test drives an all-nonfinite predictor through the
  path to confirm the support does not move.

## Implementation status, honestly stated

At the time this runbook was written, the following are implemented and covered
by the suite: the context-lift comparator and its three validation gates, the
folds, the predictor with both boundaries enforced by AST audit, the training
loop with the overfit gate and the shuffle controls, the paired scene bootstrap
proven equal to Phase 4's, the estimand layer, and the scoring core with its
three supports.

The orchestration entry point `lot.phase5` that these modes call, which reads
the caches and the Phase 4 parquets and drives the pieces above, is the
remaining work. It cannot be validated off the cluster, because none of its
inputs exist off the cluster; expect to debug it against real cache shapes on
the first `check` run.
