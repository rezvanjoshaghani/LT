# Borah runbook: Phase 4, rung 1

Every command runs from the repository root on Borah. Each mode refuses a
dirty worktree, so sync and commit before starting; the pinned execution
state is `validation/evidence/phase4/pin.md`.

The order below is the execution order and it is enforced by the code, not
only by this document: evaluation refuses to start without the convention
report, and the gates are the PROTOCOL 4.5 stop condition that precedes any
interpretation of translation or orbit results.

## 0. Environment

Set once per shell. Account and partition are never hard-coded anywhere in
the repository.

    export SLURM_ACCOUNT=<account>
    export SLURM_PARTITION=<partition>
    export LOT_ENV=lot-encode            # the Phase 2/3 env, unchanged
    export MAMBA_ROOT_PREFIX=/bsuscratch/$USER/micromamba  # the root holding lot-encode

`LOT_DINOV2_REVISION` and `LOT_VGGT_REVISION` are needed only by the caching
job and by the `smoke` mode. Phase 4 consumes the existing caches and does
not re-encode anything.

Inputs that must already be present: `data/replica_renders`, `cache/features`
(both encoders, the VGGT cache including its depth export), and
`outputs/experiment_zero` (the corrected Phase 3 run, whose mean vector and
pair sample Phase 4 reuses).

## 1. Verify the pin and the inputs

    ./scripts/run_phase4.sh check

Verifies the four frozen blobs against FREEZE.md at the freeze commit,
enumerates the post-freeze amendments, prints the live hashes, asserts a
clean tree, re-reads both caches against their recorded digests, prints the
Phase 3 and Phase 4 measurement identities, and runs the suite.

Expect: frozen blobs verified; amendments A1 to A7 listed; Phase 3
measurement digest `27244e6481d521159e513f2ea8799482`; Phase 4 measurement
digest `1579714398feff4771a9981e5f427c8a`; 277 passed, 3 skipped.

On Linux the live sha256 of PROTOCOL.md and VALIDATION.md equal their frozen
values; AMENDMENTS.md and configs/analysis.yaml differ by exactly the five
amendments, which is the mechanism working. A frozen-blob mismatch stops
everything.

Also record the manifest-set hash, which the pin defers to cluster time:

    sha256sum data/replica_renders/*/manifest.json | sort | sha256sum

Save the output into `outputs/phase4_rung1/evidence/manifest_set_hash.txt`.

## 2. Establish the VGGT depth convention

PROTOCOL 4.1 gives the installed source or documentation primary authority
and makes the regression a consistency check. Look first:

    ./scripts/run_phase4.sh inspect

This writes the depth-related lines of the installed VGGT head and geometry
modules to `outputs/phase4_rung1/evidence/vggt_source_inspection.txt`. Read
it and decide whether the depth head emits planar camera-z depth or ray
distance.

Then run the deterministic test, passing what the source established:

    ./scripts/run_phase4.sh convention planar_z

Substitute `ray_distance` if that is what the source says. If the source is
not definitive, omit the argument and let the regression stand alone:

    ./scripts/run_phase4.sh convention

Expect a verdict per scene and `unanimous: True`. A material disagreement
between the documented convention and the regression stops the run and
records both, per step 4. Ray distance is converted to planar z by the
frozen cosine rule before any alignment level runs.

## 3. The pure-rotation correctness gates

PROTOCOL 4.5. Nothing downstream may be interpreted until these pass.

    sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
           --array 0-17 scripts/phase4.sbatch configs/phase4.yaml gates

One task per scene. Each writes `outputs/phase4_rung1/evidence/gates_<scene>.json`
and prints the three maxima. For a small setup all 18 scenes can run in one
process instead, which is slower:

    ./scripts/run_phase4.sh gates

Expect, at every alignment level: coordinate residual under 1e-3 px, score
residual under 1e-5, forced-order residual under 1e-3, which under
Amendment A7 is an exact identity and should read 0.0. Landing-cell flips
are reported per pair as a non-gating diagnostic; isolated flips at
vanishing boundary margins are float rasterization instability, while a
flood of them would indicate a real convention or resize error. A breach raises
`Phase4GateError` naming the scene, frames, level, sample_id, both
coordinates, both scores, and the residual, and the job exits nonzero. That
is a stop: find the pipeline bug, do not look at translation or orbit.

Check every task before continuing:

    grep -l "gates PASS" slurm-lot-phase4-*.out | wc -l

## 4. The full evaluation

    sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
           --array 0-17 scripts/phase4.sbatch configs/phase4.yaml

One scene per task, `--resume` safe: a finished scene is skipped only when
its stored run record matches this run, and a mismatch refuses rather than
mixing populations. Each task writes `outputs/phase4_rung1/eval/<scene>.parquet`
with its run record, plus `evidence/audit_<scene>.json` carrying the
per-pair Level 1 target-exclusion audit and the gate evidence.

The gates run inline here too, so a rotation-pair breach stops the task even
if step 3 was skipped.

## 5. Tables and figures

    ./scripts/run_phase4.sh figures

Writes `outputs/phase4_rung1/tables/phase4_ladder.parquet`,
`phase4_bins.parquet`, `phase4_near_zero.json`, and the four required
figures. Outputs are never overwritten; delete the run directory to rebuild.

The reporting layer refuses a run whose records carry a `-dirty` commit, and
refuses unpinned encoder provenance.

## 6. Addendum E: the closure work, same cluster period

The permanent real-weight encoder tests, inside a GPU allocation. The mode
refuses to run without a visible CUDA device, because these tests fall back
to CPU and would otherwise run VGGT-1B on a login node:

    srun --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
         --gres=gpu:1 --time=00:30:00 --pty ./scripts/run_phase4.sh smoke

Then the remaining Stream F item, the independent pixels-to-rows
reproduction of apartment_0 against the corrected Phase 3 parquet:

    ./scripts/run_phase4.sh validate23

Roughly 900 pairs reconstructed from the manifests, depth, and cached
features by validator code that shares no transform, mask, pooling, or
scoring path with `lot`. Expect `verdict: PASS` with bit-identical masks and
metric differences far inside 1e-4. It writes
`validation/evidence/reaudit/borah_check_2_3.json` and exits nonzero on
failure. This is a long single-process job; for a smoke pass first, run it
with a cap:

    PYTHONPATH=src python validation/borah_check_2_3.py \
        --renders data/replica_renders --cache cache/features \
        --eval-dir outputs/experiment_zero/eval --scene apartment_0 \
        --max-pairs 20 --out /tmp/check23_smoke.json

A material reproduction failure is governed by VALIDATION.md's severity
rules, not waived because the check was previously unverified.

## 7. Optional: retire R-1 at the run-record level

The historical Stream D run records carry a `-dirty` marker that cannot be
proven benign after the fact. A clean canonical rerun of the Phase 3
evaluation layer over the existing caches would retire it. It does not
change any Phase 4 input and must not overwrite the existing run:

    mv outputs/experiment_zero outputs/experiment_zero_historical
    ./scripts/run_stream_d.sh evaluate
    ./scripts/run_stream_d.sh counts
    ./scripts/run_stream_d.sh figures

If skipped, the reproducibility statement stands on the frozen re-derivation
chain, the Stream F audit, and the step 6 result.

## What to watch

- `convention`: verdicts unanimous across 18 scenes.
- `gates_*.json`: the three maxima, orders of magnitude inside tolerance.
- `audit_<scene>.json`: `affine_failed` pairs, and the Level 1 scene scale.
  VGGT depth is roughly metric, so scalars far from 1 are worth a note
  before figures are read.
- Run records: `n_est_outside`, the estimated-depth splat cells falling
  outside the Phase 3 set, and the per-level scored counts.
- Nothing in this sequence tunes anything. A tax that is flat, reversed, or
  large everywhere is a valid rung 1 result if the gates pass.
