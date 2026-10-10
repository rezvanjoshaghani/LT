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

Set in every new shell, from the repository root. Account and partition are
never hard-coded anywhere.

    export SLURM_ACCOUNT=<account>
    export SLURM_PARTITION=<partition>
    export LOT_ENV=lot-encode            # the Phase 2/3/4 env, unchanged
    export MAMBA_ROOT_PREFIX=/bsuscratch/$USER/micromamba

`sacctmgr show associations where user=$USER format=account%30,partition%30`
lists your accounts, and `sinfo -o "%P %G %l"` lists each partition's GPUs and
time limit.

Choose a partition that holds a single GPU type, and use that same partition
for every Phase 5 step. Two reasons. The integration gate measures whether the
frozen training batch fits in the memory of the GPU it runs on, so it must run
on the GPU type training will use. And the frozen training configuration allows
TF32, which some GPU generations have and others lack, so training tasks spread
over mixed hardware would not share one numerical contract. A partition that
mixes GPU types breaks both.

Inputs that must already be present, all of them Phase 4's:
`data/replica_renders`, `cache/features` (both encoders, the VGGT cache
including its depth export), `outputs/experiment_zero` (the corrected Phase 3
run, whose global mean vector Phase 5 reuses), and `outputs/phase4_rung1` (the
accepted Phase 4 run, whose context-image-scale scored-cell masks define the
secondary operational support and the formulation-diagnostic population).

Phase 5 re-encodes nothing. It consumes the existing caches.

## 1. The integration gate

    srun --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
         --ntasks=1 --cpus-per-task=4 --mem=48G --gres=gpu:1 \
         --time=04:00:00 --pty ./scripts/run_phase5.sh check

Run it inside a GPU allocation as above, never on the login node. Without a GPU
the gate dry-runs the training batch on the CPU, and step 13 cannot measure
whether the batch fits in GPU memory, so the gate would pass without that test.
`--ntasks=1` is explicit because an allocation that inherits a larger task
count runs one copy of the gate per task; that once happened to the Phase 4
smoke run. The time limit is an estimate, since the gate has not yet run on
real artifacts.

A hard gate with no training side effects, and the only thing permitted to run
first. It verifies the frozen blobs against FREEZE.md, asserts a clean tree,
prints the identities, then runs seventeen steps that connect the frozen design
to the real artifacts: resolve and hash them, verify the real schemas against
what the frozen loaders assume, build real scene inputs across three scene
families, build one real example per regime and assert no forbidden field is
reachable, check the landing-location semantics the headline rests on, build
the primary support and prove an all-nonfinite predictor cannot move it, check
the formulation cells, re-derive the splat-pool symmetry audit against the
shipped source, run the real-data geometry checks, verify the folds against the
real inventory, dry-run one real batch through forward, loss, and backward,
probe resources against the frozen batch, prove the test seal, check
Context-Lift and TL-Reference against the analytic rotational homography on
every rotation pair of all 18 scenes, and complete the deferred pin. The
rotation check is step 17. It runs before step 15, the pin, so the pin is
written only after every substantive check has passed, and a gate that stops
at step 17 leaves the earlier pin in place. Step 16, the verdict, stays last.

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

`lock` is the exception. It reads no scene and trains nothing, so it runs
where the launcher runs. Its output is kept in the same directory.

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
the training-config digest, the config digest, the commit, and the plan census.
The record also carries its provenance: the sha256 of the integration and
overfit receipts that licensed the task, the scenes each role actually planned
from, and when it was written, in UTC. Evaluation and the controls load a
checkpoint only through that record and refuse one whose hash, fold, seed,
training-config digest, or config digest disagrees. Retraining a task archives
the previous checkpoint and record as `*.superseded.N.*` rather than
overwriting them.

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
is never mistaken for a complete one. The file names the receipts that licensed
it and every checkpoint it loaded, by sha256, and when it was written. Each
result names the validation scenes it planned from.

## 5. The checkpoint lock

    ./scripts/run_phase5.sh lock

`validation/evidence/phase5/reporting_rules.md` section 7. Run it after the
controls and before evaluate. It refuses unless both gates stand.

It checks four things. Every checkpoint loads through its training record,
under the same checks evaluation applies. Every training record names this
level, its own fold and seed, this commit, this config digest, and the frozen
training settings. Every training record and the controls file name, as their
licence, the integration and overfit receipts the lock binds, so training and
the controls ran under exactly those receipts. The controls file of the level
carries this commit and these digests, lists no missing checkpoint, holds one
result for each fold and seed and no other, and names each checkpoint by the
sha256 it has now. Every problem found is listed, and nothing is written.

When every check holds, it writes `checkpoint_lock_{level}.json` to the
evidence directory. The receipt carries the commit and the digests, the sha256
of the integration and overfit receipts, and the path and sha256 of the nine
checkpoints, the nine training records, and the controls file. A rerun
archives the earlier lock as `*.superseded.N.*`. Each run's output is kept
beside it as `checkpoint_lock_{level}_{UTC time}.txt`.

`evaluate` refuses without the lock of its level. The launcher verifies the
lock before it submits, and the entry point verifies it again inside every
job. Verification hashes every file the lock names, at the path evaluation
will read it from. Retraining a checkpoint or rerunning the controls after the
lock breaks it, and evaluate refuses until the lock is written again.
Rerunning either gate breaks it too, and then writing the lock again is not
enough: the lock refuses training records and a controls file licensed by
other receipts. Training and the controls must first run again under the new
receipts. Every evaluation record names the lock by sha256 among the receipts
that licensed it.

## 6. Evaluate

    ./scripts/run_phase5.sh evaluate

Streams V and W. For every test scene, through the model of the fold that held
it out, at each of the three seeds separately. It refuses unless the
checkpoint lock of its level stands and still matches the live files.

The launcher submits one array task per test scene, eighteen in all, with the
range read from the frozen folds. Each task writes
`eval/{level}/{scene}.parquet` under the run directory once, with the run
record inside the file: commit, digests, checkpoint hashes, the Phase 4 parquet
hash, and the reconciliation audit. The record also carries the sha256 of the
receipts that licensed the task, the checkpoint lock among them, each seed's
training record by sha256, the scene's aligned-depth digest, the pairs with no
arm, the environment, and when it was written. A scene that already has its
parquet reports `exists` and is not recomputed, so a failed task is resubmitted
alone:

    sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
           --array <index> scripts/phase5_job.sbatch configs/phase5.yaml evaluate

A parquet is resumed only when its run record names this scene, level, fold,
seeds, and config digest, and the checkpoints and training records the lock
binds, by sha256. Any other file at that path is refused, recorded as an
error, and left as it is. Move it aside to evaluate the scene.

Every attempt writes a start to `evidence/evaluation_ledger/` before any work,
and a close when it ends: `written`, `exists`, or `error` with its message.
Both name one attempt id. The entry point turns SIGTERM, which SLURM sends at
the time limit, on `scancel`, and on preemption, into an error close. An
attempt killed outright, by SIGKILL, the out-of-memory killer, or a lost node,
leaves its start alone, and reads as unfinished. Each record is its own file,
created exclusively, because the eighteen tasks run at once on a network file
system, where appends from several nodes can interleave or be lost.
`reporting_rules.md` section 7 names one appended file,
`evaluation_ledger.jsonl`. `lot.phase5_modes.build_evaluation_ledger` renders
the directory to that file, one line per attempt, and must run once, after
every task has ended. Whether this reading of section 7 stands is the user's
decision to record before commit E.

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
property of the frozen rule, not an implementation defect.

Every parquet also carries the pre-registered landing-offset diagnostic,
`validation/evidence/phase5/landing_offset_diagnostic.md`, which sizes that cap
on DINOv2 features. It lifts with ground-truth context depth on its own support,
groups the scores by landing offset, and writes its columns on the `all` rows.
It is model free and never enters the headline.

## 6a. Sensitivity and diagnostic levels

    ./scripts/run_phase5.sh train --level affine
    ./scripts/run_phase5.sh controls --level affine
    ./scripts/run_phase5.sh lock --level affine
    ./scripts/run_phase5.sh evaluate --level affine

The same sequence at a predeclared non-primary level: `affine` for sensitivity,
`none` for the systems diagnostic. The entry point refuses any undeclared
level, and refuses a non-primary level until every test scene has its primary
parquet. The config also asks for the primary result to be interpreted first.
Code cannot check that, so it is the operator's step.

## 7. Tables and figures

    ./scripts/run_phase5.sh tables
    ./scripts/run_phase5.sh figures

Stream AC. The headline table, the formulation-reference table labelled as a
diagnostic rather than an estimand, the operational splat-pool table, and the
five figures. Every figure regenerates from the tables alone.

The landing-offset diagnostic's cells sit beside the headline, under the
interpretation rules its pre-registration fixes: the read deficit is reported
next to delta_learn_pp and never subtracted from it.

## 8. Acceptance

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
scoring core with its three supports, and the seventeen-step integration gate
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

Steps 1 through 16 passed on Borah at `360d5eac`, as `reporting_rules.md`
records. Step 17 was added after that run and has not yet executed against
real artifacts.

`overfit`, `train`, `controls`, `lock`, and `evaluate` are implemented.
`tests/test_phase5_modes.py` runs the four job modes end to end on synthetic
scenes written to disk, with real Phase 3 and Phase 4 runs on the test scene,
so evaluation reconciles against a genuine Phase 4 parquet. On that fixture the
recomputed TL-Reference and splat arms match Phase 4 with a worst residual of
zero. It also runs the controls and then the lock through the command line on
the fixture's checkpoints, and the verifier accepts the lock written. The CLI
refuses every mode without its receipts, and refuses evaluate without a lock
that matches the live files. None of the five has run on real artifacts.

The outcome and wording code of `reporting_rules.md` decision 2 is implemented
in `lot.phase5_report` and covered by `tests/test_phase5_report.py`: the call
of each supported cell as outcome 46, 47, or 48, the near-zero qualifier, the
outcome 49 and metric-sensitive flags, the landing-offset flags, and the
measured outcome per regime with the pooled row labelled a summary. Decision 1
freezes it at commit E.

Not yet implemented: the `tables`, `figures`, and `acceptance` modes, the rest
of Streams AC and AD. They refuse with an explanation rather than a stack
trace. Decision 1 of `reporting_rules.md` is what holds the chain back: the
reporting code of its section 8 is built and committed first, and the chain
then runs once, at one commit E. So the chain, from `check` through
`evaluate`, does not start until these modes are committed.
