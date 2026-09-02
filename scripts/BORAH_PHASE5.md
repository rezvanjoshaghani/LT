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

## 1. The integration gate

    ./scripts/run_phase5.sh check

A hard gate with no training side effects, and the only thing permitted to run
first. It verifies the frozen blobs against FREEZE.md, asserts a clean tree,
prints the identities, then runs sixteen steps that connect the frozen design
to the real artifacts: resolve and hash them, verify the real schemas against
what the frozen loaders assume, build real scene inputs across three scene
families, build one real example per regime and assert no forbidden field is
reachable, check the landing-location semantics the headline rests on, build
the primary support and prove an all-nonfinite predictor cannot move it, check
the formulation cells, re-derive the splat-pool symmetry audit against the
shipped source, run the real-data geometry checks, verify the folds against the
real inventory, dry-run one real batch through forward, loss, and backward,
probe resources against the frozen batch, prove the test seal, and complete the
deferred pin.

A failure prints the failing step, the exact evidence, and a classification:
`missing_artifact`, `implementation_bug`, or `frozen_design_mismatch`. Only the
last may justify touching the frozen design, and then only as an amendment with
a written rationale.

Expect: frozen blobs verified; A1 to A7 listed; Phase 3 measurement digest
`27244e6481d521159e513f2ea8799482`; Phase 4 measurement digest
`1579714398feff4771a9981e5f427c8a`; fold digest
`25f0c03f72d58cc8e3ff2d8d4123241f6459d4ed50d3c303c5dc108bb0e19865`; suite
green.

`check` is also where the cluster-resident pin fields are filled: the per-scene
DINOv2 and VGGT cache digests, the aligned context-depth digest, and the Phase 4
artifact hashes. It writes them to
`outputs/phase5_rung2/evidence/pin_cluster.json`, which is the file that closes
the deferrals `validation/evidence/phase5/pin.md` records.

A frozen-blob mismatch, a cache-digest mismatch, or a measurement-identity
mismatch stops everything.

## 2. The tiny-subset overfit gate

    ./scripts/run_phase5.sh overfit

Refuses to run until `check` has passed, read from the gate's own receipt
rather than from memory.

Stream U step 14. Eight pairs from training scenes only, spanning the three
camera regimes, at the frozen threshold of centered cosine 0.98.

A failure is a stop and the phase does not continue. The reason is not that a
failing model is uninteresting; it is that Phase 5's whole question is whether
learning the transformation costs accuracy, and a trunk that cannot fit eight
pairs would answer a different question. The gate makes the later comparison
interpretable rather than making it look good.

## 3. Train

    ./scripts/run_phase5.sh train

which submits

    sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
           --array 0-8 scripts/phase5_train.sbatch configs/phase5.yaml

and refuses unless both the integration gate and the overfit gate have passed,
each read from its own receipt rather than from memory.

Nine tasks: three folds by three seeds. Each task trains on its fold's nine
training scenes, selects its checkpoint on its three validation scenes by
validation centered cosine, and never touches the six test scenes. The seal is
asserted inside the training loop, so a leaked scene raises rather than
training.

Nothing about a test scene may influence anything here. If a training run is
restarted, changed, or retuned in response to a test metric, that run is
invalid under Stream S step 9 and must be discarded rather than reported.

## 4. Input-use controls

    ./scripts/run_phase5.sh controls

Stream U step 17, on validation scenes. Pose shuffle and depth shuffle against
one shared baseline.

These are diagnostics. A shuffle that barely moves the score is reported exactly
as observed, and is evidence about whether the network uses that input. It is
never a reason to retrain, and it is never a threshold anything must pass.

## 5. Evaluate

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

## 6. Tables and figures

    ./scripts/run_phase5.sh tables
    ./scripts/run_phase5.sh figures

Stream AC. The headline table, the formulation-reference table labelled as a
diagnostic rather than an estimand, the operational splat-pool table, and the
five figures. Every figure regenerates from the tables alone.

## 7. Acceptance

    ./scripts/run_phase5.sh acceptance

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

Implemented and covered by the suite: the context-lift comparator and its three
validation gates, the folds, the predictor with both boundaries enforced by AST
audit, the training loop with the overfit gate and the shuffle controls, the
paired scene bootstrap proven equal to Phase 4's, the estimand layer, the
scoring core with its three supports, and the sixteen-step integration gate
with its stop machinery.

`check` is implemented and runs. On a machine without the Borah artifacts it
stops at step 2 with classification `missing_artifact`, naming each absent path
and why the phase needs it, which is the correct behaviour and is how the stop
machinery was verified off the cluster.

Steps 3 through 16 have never executed against real artifacts. They are written
against the shapes Phase 4's own code produces, and the whole point of the gate
is that this is an assumption until it runs. Expect step 3 or step 5 to be
where a real mismatch first appears.

Not yet implemented: `overfit`, `train`, `controls`, and `evaluate` beyond their
data assembly, and `tables`, `figures`, and `acceptance` entirely. Those last
three refuse with an explanation rather than a stack trace, deliberately: their
inputs are the evaluation records the gate exists to make trustworthy, so
building them before the gate passes would be building on an unverified
foundation.
