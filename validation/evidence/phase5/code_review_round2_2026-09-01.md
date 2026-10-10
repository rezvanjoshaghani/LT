# Phase 5 code review, round 2, 2026-09-01, and its disposition

Second review of the Phase 5 branch, against base `7726aef`, reviewed tip
`9150b56`. It re-audited the ten round-one findings and reviewed the receipt,
scoring, estimand, training, and test code added at `5f3c11b`. The round-one
repairs were confirmed. Five new findings, all verified and all fixed.

## Disposition

Findings 2 and 3 were reproduced exactly as reported before anything changed:
the near-zero branch returned "path-sensitive" on a sign-reversing pair, and
`cfg.digest()` was unchanged by every one of `feature_encoder`,
`depth_encoder`, `phase4_dir`, `renders_root`, `cache_root`, `seed`,
`controls`, and `sensitivity_alignment_levels`.

**1, the two-path disclosure computed on different populations.** The most
consequential of the five. The disclosure was comparing a quantity on
`V_P5_pp` against its counterpart on `V_sp`, which mixes an operator
difference with a selection difference, and its `path_difference` was a bare
subtraction of two point estimates with no interval. PROTOCOL 3.9 requires
every term recomputed on the cross-path common-valid cell set before
differencing, with the difference recomputed inside each replicate of one
paired scene bootstrap.

Fixed by adding a `CROSS_PATH` population. `lot.phase5_score.score_cross_path`
re-scores both paths on the target cells they share, and those columns live in
one field tuple so a single scene draw serves both paths and their difference.
A test gives the own-population columns a large opposite-signed gap and the
intersection columns a small agreeing one, and asserts the disclosure reads the
latter.

**2, the veto ordering.** PROTOCOL 3.9's third clause, "a difference in sign,
or an interval that includes zero, licenses no claim of advantage", is stated
unconditionally, so it is a veto over the other two branches rather than a
fallback after them. Testing the exactly-one-in-band branch first let a
sign-reversing cell be reported as merely path-sensitive. The band question is
now asked only to decide whether the near-zero discipline is engaged at all.

**3, receipt identity coverage.** The configuration digest now covers every
field except `output_root` and `experiment_name`, as an allowlist of
exclusions, and the receipt additionally binds the resolved artifact paths the
gate recorded. The SLURM worker derives its evidence directory from the config
it was given rather than hard-coding one, so it cannot read another run's
receipts.

**4, the tiny-overfit gate's optional requirements.** The fold, the pair count,
and the regime set are now required arguments checked before the optimizer is
constructed. The gate's verdict is quoted as evidence that the frozen
eight-pair, three-regime subset passed, so reporting the realized count
afterwards was not sufficient: it described whatever was supplied rather than
constraining it.

**5, controls outside the seal.** `run_input_use_controls` now requires the
fold and asserts the validation role before evaluating anything.

Regression coverage is in tests/test_phase5_estimands.py,
tests/test_phase5_train.py, tests/test_phase5_config.py, and
tests/test_phase5_receipt.py. Suite at the fix: 486 passed, 3 skipped.

## Note on the configuration digest

The frozen digest changed as a result of finding 3, from
`19d903812e4c...` to `9e3508606bca...`. No Phase 5 result existed under either
value, and the change widens what the identity covers rather than altering any
frozen experimental value. The pin records both and why.

## The reviewer's own environment limitation, recorded

The round-two verification could not run `bash -n` on the two launch scripts:
the installed WSL launcher returned `E_ACCESSDENIED`. That is an environment
limitation and not a verdict on the scripts. Both were syntax-checked here
after the fixes, and both pass.

---

## Findings, verbatim

```
PHASE 5 CODE REVIEW - ROUND 2
============================

Date: 2026-09-01
Repository: D:\LT
Branch: repair/validation-streams-abc
Reviewed tip: 9150b56
Comparison point: 7726aef

SUMMARY
-------

The second review re-audited the ten original findings after commit 5f3c11b
and reviewed the newly added receipt, scoring, estimand, training, and test
code. The mean-vector and dataset API calls, ragged batching, device transfer,
empty-stream stop, and L2 record schema are repaired. Five actionable issues
remain.


1. [P1] TWO-PATH DISCLOSURE IS STILL COMPUTED ON DIFFERENT POPULATIONS

File: src/lot/phase5_estimands.py
Lines: 409-417

evaluate_quantity computes the per-point interval after filtering records on
n_primary, then independently computes the splat counterpart after filtering
on n_splat. Those are V_P5_pp and V_sp, not the cross-path common-valid cell
set required by PROTOCOL 3.9. The records at this layer contain path-level
pair averages and counts, not the sample_id sets or validity masks needed to
recompute both paths on their intersection.

The resulting path_difference at line 305 is only the subtraction of two
point estimates. It has no interval, and the difference is not recomputed
inside one paired scene bootstrap replicate. Selection differences can
therefore change the sign or size attributed to evaluation-path choice and
can feed an invalid wording decision.

Required correction: persist or carry the contributing sample identities,
intersect the paths by sample_id, recompute every term on that intersection,
and bootstrap both estimates and their difference from the same scene draw.


2. [P1] THE ONE-IN-BAND BRANCH OVERRIDES THE PROTOCOL'S SIGN/INTERVAL VETO

File: src/lot/phase5_estimands.py
Lines: 284-295

The code checks pp_in != sp_in before it checks sign agreement and whether
both intervals exclude zero. PROTOCOL 3.9 states that a sign difference or an
interval containing zero licenses no claim of advantage. Those are vetoes,
including when exactly one estimate is inside the 0.003 band.

Reproduction:
  per-point  estimate +0.001, interval [+0.0005, +0.0015]
  splat-pool estimate -0.020, interval [-0.0300, -0.0100]

Current result:
  "effect licensed but its size is path-sensitive"

Required result:
  no claim of advantage, because the paths reverse sign.

Required correction: apply the sign-disagreement and interval-includes-zero
veto before the exactly-one-in-band wording branch.


3. [P1] RECEIPTS ARE NOT BOUND TO THE COMPLETE EXPERIMENT IDENTITY

Files:
  src/lot/phase5_receipt.py, line 63
  src/lot/phase5.py, lines 95-114
  scripts/phase5_train.sbatch, line 65

current_identity treats cfg.digest() as the configuration identity, but that
digest covers only model, training, tiny_overfit, primary_alignment_level,
and a version. It excludes fields that change the actual gated data or
evaluation, including feature_encoder, depth_encoder, renders_root,
cache_root, phase4_dir, the pair-subsampling seed, sensitivity levels, and
controls. The verifier also ignores the artifact hashes recorded by the gate.

Targeted checks confirmed that changing each of feature_encoder,
depth_encoder, phase4_dir, renders_root, controls, or seed leaves cfg.digest()
unchanged. A same-commit alternate configuration can therefore reuse a PASS
receipt even though the gate never examined its inputs. The SLURM worker also
hard-codes outputs/phase5_rung2/evidence despite accepting an arbitrary config
path, increasing the chance that it reads another run's receipt.

Required correction: bind the receipt to a complete content identity for all
scientifically meaningful Phase 5 configuration fields and to the resolved
artifact/pin identity. Derive or explicitly pass the receipt directory for
the supplied config. Output relocation fields may remain excluded if they are
the only excluded fields.


4. [P1] THE TINY-OVERFIT GATE'S FROZEN SUBSET REQUIREMENTS ARE OPTIONAL

File: src/lot/train.py
Lines: 524-547

The prior finding that the tiny gate accepts arbitrary scenes is not fully
fixed. fold defaults to None, and the role assertion only runs when callers
choose to supply it. Existing successful tests call the gate without a fold.
The function also does not require exactly the frozen eight pairs or require
the rotation, translation, and orbit regimes; it merely reports the count and
observed regimes after training.

The gate can therefore PASS on an easier, smaller, or test-derived subset and
still be presented as evidence that the frozen eight-pair, three-regime gate
passed.

Required correction: make the fold and frozen subset specification mandatory
at the gate boundary, assert every scene is in fold.train, require the exact
pair count, and require the configured regime set before constructing the
optimizer.


5. [P2] VALIDATION-ONLY INPUT CONTROLS ARE OUTSIDE THE SPLIT SEAL

File: src/lot/train.py
Lines: 645-662

run_input_use_controls documents its examples as validation pairs, and the
launcher states that evaluate is the only mode allowed to touch test scenes.
The function accepts no fold and performs no role assertion before evaluating
the baseline and shuffled examples. Training and checkpoint selection now
enforce positive split membership, but this adjacent entry point can still
consume training or sealed test scenes.

Required correction: require the fold and assert every control example is in
fold.val before either baseline or shuffled evaluation.


KNOWN INCOMPLETENESS
--------------------

The branch remains intentionally non-runnable end to end: modes overfit,
train, controls, and evaluate still exit at src/lot/phase5.py lines 472-475.
That known status is not counted as a new finding, but it means the new worker
receipt enforcement and the full scientific pipeline could not be exercised
as one completed run.
```

## Verification, verbatim

```
PHASE 5 CODE REVIEW - ROUND 2 VERIFICATION
==========================================

Date: 2026-09-01
Repository: D:\LT
Branch: repair/validation-streams-abc
Reviewed tip: 9150b56
Comparison point: 7726aef
Starting working tree: clean

SCOPE REVIEWED
--------------

  - Re-audit of all ten findings in the first Phase 5 review
  - Changes in commits 5f3c11b and 9150b56
  - Phase 5 configuration and cluster launch guards
  - Gate-receipt production, parsing, and identity verification
  - Ragged batching, CUDA transfer, training roles, tiny overfit, and controls
  - Raw/centered cosine and L2 scoring records
  - Estimand populations, paired bootstrap, and near-zero disclosure
  - Added and changed Phase 5 tests

AUTOMATED VERIFICATION
----------------------

Command:
  python -m pytest tests/test_phase5_estimands.py tests/test_phase5_receipt.py
      tests/test_phase5_score.py tests/test_phase5_train.py -q
Result:
  91 passed in 18.40 seconds

Command:
  python -m pytest -q
Result:
  471 passed, 3 skipped in 153.09 seconds

Command:
  python -m compileall -q src tests
Result:
  PASS

Command:
  git diff --check 7726aef..HEAD
Result:
  PASS

Command:
  bash -n scripts/run_phase5.sh
  bash -n scripts/phase5_train.sbatch
Result:
  NOT RUN. The installed WSL launcher returned E_ACCESSDENIED before Bash
  started. This is an environment limitation, not a script verdict.

TARGETED ADVERSARIAL CHECKS
---------------------------

1. Near-zero branch precedence

Input:
  per-point  PathEstimate(0.001, 0.0005, 0.0015)
  splat-pool PathEstimate(-0.02, -0.03, -0.01)

Observed wording:
  "effect licensed but its size is path-sensitive; the two evaluation paths
   do not agree about whether it sits inside the operator band"

Expected from PROTOCOL 3.9:
  no claim of advantage, because the paths differ in sign.

2. Configuration-receipt identity coverage

For each field below, dataclasses.replace was used to alter a loaded
Phase5Config and compare the resulting cfg.digest() with the original:

  feature_encoder  unchanged digest: True
  depth_encoder    unchanged digest: True
  phase4_dir       unchanged digest: True
  renders_root     unchanged digest: True
  controls         unchanged digest: True
  seed             unchanged digest: True

3. Cross-path population audit

Observed:
  _interval_for filters per-point records on n_primary > 0.
  _interval_for filters splat records on n_splat > 0 in a separate call.
  near_zero_disclosure subtracts the two independent point estimates.

Missing:
  sample_id/mask intersection, common-valid recomputation, and a paired
  bootstrap distribution for the path difference.

4. Split-boundary audit

Observed:
  run_tiny_overfit_gate accepts fold=None and validates no scene role in that
  case. It does not assert the frozen pair count or regime set.
  run_input_use_controls accepts no fold and performs no validation-role check.

REVIEW ARTIFACT EFFECT
----------------------

No source or test files were modified. This review added only the two plain
text evidence files named in this directory.
```
