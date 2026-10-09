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

## How the modes after `check` run

`overfit`, `train`, `controls`, and `evaluate` are submitted to SLURM by the
launcher rather than run on the login node. Each job's output is written under
`outputs/phase5_rung2/evidence/`, named by mode and job id. Account and
partition come from `SLURM_ACCOUNT` and `SLURM_PARTITION` only.

The launcher checks each mode's receipts before it submits, so a refusal costs
no queue time. The entry point checks them again inside the job, through
`lot.phase5.require_receipts`. That second check is the real boundary: a job
submitted by hand with `sbatch` is refused exactly as a launched one is, and a
job cannot run past a gate that stopped standing while it waited in the queue.

The templates are `scripts/phase5_train.sbatch` for the nine training tasks and
`scripts/phase5_job.sbatch` for the other three modes. Both take the config and
pass any further arguments to the entry point.

## 2. The tiny-subset overfit gate

    ./scripts/run_phase5.sh overfit

Refuses to run until `check` has passed, read from the gate's own receipt
rather than from memory. Its own receipt records the digest of that gate
receipt, so an overfit result cannot be carried across to a run the gate never
examined; rerunning `check` invalidates it and the overfit gate must be rerun.

The job prints `TINY-SUBSET OVERFIT GATE: PASS` or `FAIL`, the centered cosine
reached, and the eight pairs, and writes the verdict to
`outputs/phase5_rung2/evidence/tiny_overfit.json`. A rerun archives the earlier
receipt as `*.superseded.N.*` rather than overwriting it. A FAIL receipt blocks
every later mode, because verification requires a recorded pass. The gate runs
only at the primary alignment level.

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
           --output outputs/phase5_rung2/evidence/train_%A_%a.txt \
           --array 0-8 scripts/phase5_train.sbatch configs/phase5.yaml

and refuses unless both the integration gate and the overfit gate have passed,
each read from its own receipt rather than from memory.

Each task writes `checkpoints/{level}/fold{f}_seed{s}.pt` under the run
directory, with a JSON record beside it that carries the checkpoint's sha256,
the training-config digest, the commit, and the plan census. Evaluation and the
controls load a checkpoint only through that record and refuse one whose hash,
fold, seed, or config digest disagrees. Retraining a task archives the previous
checkpoint and record as `*.superseded.N.*` rather than overwriting them.

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

Run it after all nine training tasks have finished. It loops over every fold and
seed in one job and writes `input_use_controls_{level}.json` to the evidence
directory, archiving any earlier result rather than overwriting it. A missing
checkpoint is listed in that file and the job exits nonzero, so a partial result
is never mistaken for a complete one.

## 5. Evaluate

    ./scripts/run_phase5.sh evaluate

Streams V and W. For every test scene, through the model of the fold that held
it out, at each of the three seeds separately.

The launcher submits one array task per test scene, eighteen in all, with the
range read from the frozen folds. Each task writes
`eval/{level}/{scene}.parquet` under the run directory once, with the run
record inside the file: commit, digests, checkpoint hashes, the Phase 4 parquet
hash, and the reconciliation audit. A scene that already has its parquet
reports `exists` and is not recomputed, so a failed task is resubmitted alone:

    sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
           --array <index> scripts/phase5_job.sbatch configs/phase5.yaml evaluate

For every pair, TL-Reference and the splat arm are recomputed through Phase 4's
code and reconciled with the accepted Phase 4 rows: masks bit for bit, scores
within 1e-5. The region masks are reconciled the same way. Any disagreement, or
a Phase 4 pair that Phase 5 would not evaluate, stops the scene with
`ReferenceMismatch` rather than writing a record.

Three populations, fixed before any scoring and never merged:

- the primary per-point support, decided by Context-Lift Transport-Only, shared
  by CL-Transport, Predict-with-Depth, and No-Warp-Copy;
- the formulation support, its intersection with the Phase 4 target-lift set on
  the target patch cell, used only for TL-Reference against CL-Transport;
- the accepted Phase 4 context-image-scale splat-pool scored cells, used
  unchanged for the secondary operational comparison.

The predictor cannot narrow the primary support. A nonfinite prediction on a
supported sample is scored as a model failure and counted, not dropped.

Read `validation/evidence/phase5/landing_read_asymmetry.md` before interpreting
the headline. Under the frozen landing-location read, a predictor that returns
the true target grid scores exactly one, while Context-Lift with exact geometry
is capped by the interpolation residual of the target features. That cap is a
property of the frozen rule, not an implementation defect. Its size on DINOv2
features is unmeasured.

## 5a. Sensitivity and diagnostic levels

    ./scripts/run_phase5.sh train --level affine
    ./scripts/run_phase5.sh controls --level affine
    ./scripts/run_phase5.sh evaluate --level affine

The same sequence at a predeclared non-primary level: `affine` for sensitivity,
`none` for the systems diagnostic. The entry point refuses any undeclared
level, and refuses a non-primary level until every test scene has its primary
parquet. The config also asks for the primary result to be interpreted first.
Code cannot check that, so it is the operator's step.

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
machinery was verified off the cluster. On the cluster, step 2 compares every
scene's live feature, depth, and manifest digests against the accepted Phase 4
run records and stops on any disagreement; step 4 recomputes the aligned
context depth for every scene and records its digest; step 7 runs the
cross-path producer on a real pair. Receipts are bound to all of it by
content, so a later run cannot inherit a verdict about different bytes.

Steps 3 through 16 have never executed against real artifacts. They are written
against the shapes Phase 4's own code produces, and the whole point of the gate
is that this is an assumption until it runs. Expect step 3 or step 5 to be
where a real mismatch first appears.

`overfit`, `train`, `controls`, and `evaluate` are implemented.
`tests/test_phase5_modes.py` runs all four end to end on synthetic scenes
written to disk, with real Phase 3 and Phase 4 runs on the test scene, so
evaluation reconciles against a genuine Phase 4 parquet. On that fixture the
recomputed TL-Reference and splat arms match Phase 4 with a worst residual of
zero. The CLI refuses every mode without its receipts. None of the four has run
on real artifacts.

Not yet implemented: `tables`, `figures`, and `acceptance`. They refuse with an
explanation rather than a stack trace, deliberately: their inputs are the
evaluation records the gate exists to make trustworthy, so building them before
the gate passes would be building on an unverified foundation.
