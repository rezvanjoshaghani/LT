# Phase 5 evaluation and reporting rules: pre-registration

Recorded 2026-10-10. The integration gate has passed on Borah at
`360d5eac7243b367140da550bbe909b24005a731`. No Phase 5 model has trained at
scale, no checkpoint is locked, and no Phase 5 test result exists. Nothing in
this file was written after any Phase 5 outcome existed.

## The decision

A mapping of the revised Phase 5 specification against the code found work
that must exist before training and evaluation, and five open questions. They
were put to the user on 2026-10-10 with a recommendation each. The user's
reply, verbatim:

> go

That accepts all five recommendations, recorded as decisions 1 to 5 below,
and the design choices listed after them, which were stated as following the
design unless the user objected.

## Decision 1. One commit for the whole chain

The additions in section 7 and the reporting code in section 8 are built and
committed first. The chain then runs once, at one commit E:

    check -> overfit -> train -> controls -> lock -> evaluate

Training waits for the build. Receipts bind the commit and the config
digest, so every step of the chain is licensed by receipts at E.

The tables, figures, and acceptance modes may run at a later commit R. They
are licensed by the evaluated run's own provenance, not by receipts at R.
One class of files is frozen at E: `src/lot/phase5_estimands.py`,
`src/lot/paired_bootstrap.py`, the outcome and wording code in
`src/lot/phase5_report.py`, and this file. Any difference in that class
between E and R fails acceptance. Any other change after E to a reporting
file must be named, with its reason, in
`validation/evidence/phase5/post_evaluation_changes.md`, or acceptance fails.
No file that decides what is measured may change between E and R.

## Decision 2. The near-zero rule and how outcomes are called

### The trigger

PROTOCOL 3.9 scopes its discipline to "an interpreted effect no larger than
this tolerance", and the specification's step 35 applies it when the magnitude
is at most 0.003 "under the relevant path". The near-zero wording is therefore
engaged when, and only when, the reported effect's own estimate has magnitude
at most `path_agreement_tolerance`, 0.003. The two cross-path terms and their
paired difference are always shown beside it, engaged or not.

The previous code also engaged when either cross-path term fell in the band.
It could print "no claim of advantage" for a clearly nonzero headline gap
whose splat-path term happened to sit near zero, which made the
specification's outcome 49 unreachable. That is replaced.

### The wording, when engaged

The terms are the two cross-path estimates, computed on the cells both paths
share, with their paired intervals. In order:

1. The terms differ in sign, or either interval includes zero: "no claim of
   advantage; the effect is at the scale of evaluation-path choice". This is
   PROTOCOL 3.9's veto and it is tested first.
2. Both terms are inside the band: "small, sign-consistent effect".
3. Exactly one term is inside the band: "effect licensed but its size is
   path-sensitive; the two evaluation paths do not agree about whether it
   sits inside the operator band".
4. Neither term is inside the band: "effect licensed; inside the operator band
   on its own support but outside it on both paths' common cells". The
   previous code printed wording 3 here, which said the paths disagree when
   they agree.

A quantity with no second path by construction, the formulation gap:

- interval clear of zero: "within the operator band on its only path; its
  size is not certified by a second path";
- interval including zero: "no measurable difference at the reported scale;
  this quantity has no second evaluation path by construction".

The previous code printed the second sentence in both cases, which claimed no
measurable difference for an interval that excludes zero.

Equivalence is never claimed, under any branch.

### How outcomes are called (specification steps 46 to 49)

Each supported cell is classified from the reported gap's own paired scene
interval, under centered cosine. Raw cosine is classified beside it.

- The interval for delta_learn_pp excludes zero and the estimate is positive:
  outcome 46, explicit context-lift wins.
- The interval excludes zero and the estimate is negative: outcome 48, the
  learned transformation wins. It is reported directly.
- The interval includes zero: outcome 47, no measurable gap at the reported
  scale. It is never called equivalence.
- An engaged near-zero wording travels with the cell as a qualifier. It does
  not change the outcome.
- An unsupported cell, below the frozen support thresholds, is shown with its
  counts and is not classified.
- Outcome 49 is flagged when the per-point cell is 46 and the operational
  delta_learn_sp cell, classified on its own population by the same rule, is
  47 or 48. The cross-path terms are shown beside it. The two statements are
  never collapsed.
- A sign disagreement between centered and raw in a supported cell is flagged
  "metric-sensitive". It is descriptive.
- Trends with rotation angle or parallax are described from the bin
  estimates. No slope test is registered.

The landing-offset rules of `landing_offset_diagnostic.md` apply unchanged.
Their flags are reported only where both the delta_learn_pp cell and the
read_deficit cell are supported.

The verdict's "measured outcome" is the per-regime classification: one
outcome per regime row and metric, plus the pooled row, which is labelled a
summary. Claims are made per regime.

## Decision 3. The oracle-context-depth diagnostic is not run

Stream AF is optional and had to be declared before outcomes. It is
forfeited. It would need a new level and nine more training runs, and the
landing-offset diagnostic already gives a ground-truth-depth reference for
Context-Lift.

## Decision 4. Affine after the primary result

The affine context-depth sensitivity runs after the primary result is fixed
and interpreted, as findings item 13. It is not an acceptance condition. It
needs its own nine training runs under the same chain. The overfit gate is
not rerun at that level, because it runs once, at the primary level. The
native-scale diagnostic is not planned.

## Decision 5. A stable validation curve, and "essentially no effect"

PLAN.md makes "training converges with a stable validation curve" a Phase 5
acceptance criterion. A training run's curve is stable when:

- every validation score is finite;
- at least two validations ran;
- early stopping fired, or the last validation score is within 0.003 of the
  best.

All nine training runs must be stable. This is an acceptance condition.

A shuffle control whose degradation has magnitude at most 0.003 is labelled
"essentially no effect", per specification step 17. The label is
descriptive. A control is never a threshold anything must pass. Each label
travels with the control's count of exchanges that changed nothing.

## 6. Design choices, stated with the decisions

- **Metric.** Centered cosine is primary, with raw cosine beside it. Each L2
  companion goes in its own table, with no near-zero wording, because the
  band is calibrated on cosine.
- **Cells.** One row per regime, plus a pooled row labelled a summary.
  Rotation pairs are binned by the frozen rotation edges, translation pairs by
  the frozen parallax edges. Orbit pairs are reported in joint rotation by
  parallax cells only, never marginalized. Every row carries the visibility
  bucket, which is co-visible.
- **Unit and interval.** A cell's estimate is the unweighted mean over camera
  pairs of pair-level values. Its interval is the paired scene bootstrap with
  the frozen resamples, seed, and confidence, and it carries its replicate
  count. The comparison-weighted value is shown beside it as a diagnostic.
- **Seeds.** Each pair's three seed records are collapsed to one record
  before any cell is computed. Predictor fields are the seed mean. Explicit
  fields must be identical across seeds, and a mismatch stops the run. Seed
  spread is reported in its own columns and never widens the scene interval.
  Per-seed results are composite estimates per seed index, over all 18 test
  scenes. Each test scene is scored by the model of the fold that held it out.
- **Floors.** No-Warp-Copy sits beside every reported metric, Mean-Feature
  beside every raw metric. On the formulation support the two floors are
  added by this commit; they are levels, not margins, and add no interpreted
  effect.
- **Regions.** Boundary and interior, low and high texture, on the
  per-point and splat paths, with the paired contrasts boundary minus interior
  and low minus high texture. A contrast is computed on pairs present in both
  regions, carries its interval, and gets no near-zero wording.
- **Landing-offset cells** mirror the headline cells exactly, so the read
  deficit sits in the same cell as delta_learn_pp. Mean-Feature cells exist
  under raw cosine only.
- **Three rungs.** The Phase 3 Oracle-Transport ceiling, the Phase 4
  estimated-geometry result under its own estimator, and delta_learn_pp are
  reported separately and never folded into one residual.
- **Figures** are drawn from the tables alone. Unsupported cells are greyed
  and show their counts.
- **Outputs** are written once. Tables and figures are built in a staging
  directory and published by one rename. A rebuild moves the previous output
  aside and deletes nothing.

## 7. Additions that must exist before the chain runs

- **The pure-rotation gate across the regime.** Specification step 6 asks for
  Context-Lift, TL-Reference, and the analytic rotational homography to agree
  across the pure-rotation regime before Context-Lift is used scientifically.
  The integration gate checked one pair, without TL-Reference. A new step,
  run inside `check`, covers every rotation-regime pair of all 18 scenes at
  the primary level:
  - Context-Lift landings against the homography, on landed samples;
  - Context-Lift landings with the depth map replaced, on landed samples;
  - TL-Reference's context read locations against the inverse homography, on
    its landed samples.

  "Common-valid" is read as: each estimator is compared with the homography on
  its own valid samples. The two estimators index different samples, so they
  agree with each other exactly when both agree with the homography. The
  tolerance is the frozen 1e-3 px, plus, per sample, the displacement the
  pair's recorded camera translation can cause: focal length times
  translation over the sample's depth in the receiving camera. The frozen
  rotation-position bound, 1e-6 m, allows that translation. A pair whose
  translation exceeds the bound stops the gate.
- **The checkpoint lock.** A `lock` mode runs after controls. It verifies all
  nine checkpoints against their records, the controls file, and the training
  records' identity, then writes a lock receipt binding all of them by sha256.
  `evaluate` refuses without a lock receipt that still matches the live files.
- **The evaluation ledger.** Every evaluation attempt appends a line to
  `evaluation_ledger.jsonl`. Acceptance requires one evaluation per scene and
  level.
- **Provenance in the records.** Training records, the controls file, and
  every evaluation parquet name the sha256 of the receipts that licensed them
  and carry a UTC timestamp. Training records list the scenes actually planned
  in each role. Evaluation parquets also carry the training records' hashes,
  each scene's aligned-depth digest, the pairs with no arm, and the
  environment. Evaluation refuses a checkpoint whose training record names a
  different config digest.
- **Floors on the formulation support**, as in section 6.

## 8. The reporting code

Streams AC and AD are implemented under the rules above and committed at E,
before any outcome exists. Acceptance re-derives each of the specification's
18 conditions from the shipped artifacts. It adds three: the stable
validation curve of decision 5, the headline figures present and built from
the current tables, and the full suite green at the reporting commit.
