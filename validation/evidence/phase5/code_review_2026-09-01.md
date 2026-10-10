# Phase 5 code review, 2026-09-01, and its disposition

Review of the Phase 5 branch against base `ff52d13`, reviewed tip `7726aef`.
Ten findings, all verified against the source and all fixed at `5f3c11b`.
Preserved verbatim below rather than summarized, on the same principle the
Phase 4 breach evidence is preserved: the record of what was wrong is worth
more than a claim that it was fixed.

## Disposition

Every finding was confirmed before any code changed. None was dismissed.
Four of the ten (1, 2, and the two halves of 4) were calls written against
signatures assumed rather than read, which is exactly the class of error the
Borah integration gate exists to surface; the review found them first,
before any cluster time was spent.

Two findings deserve naming because the fix changed behaviour rather than
only correctness. Finding 6: PROTOCOL 3.9's near-zero rule is explicitly
two-path, and the single-path implementation could license the strongest
wording while the operational path reversed sign; the second path is now a
required argument. Finding 3: the integration gate had been truncating every
example to the shortest supervision length, which made its shape probe pass
while the real training path could not batch differing support at all. The
gate now pushes a real ragged batch through the real path.

Regression coverage for all ten is in tests/test_phase5_train.py,
tests/test_phase5_score.py, tests/test_phase5_estimands.py, and
tests/test_phase5_receipt.py. Suite at the fix: 471 passed, 3 skipped.

---

## Findings, verbatim

```
CODE REVIEW FINDINGS
====================

Review target: repair/validation-streams-abc
Comparison base: ff52d13

1. [P1] Mean-vector loader is called with the wrong signature

File: src/lot/phase5_gate.py
Line: 163

load_or_build_mean_vector requires (cache_root, encoder, scenes, out_dir),
but this call supplies three arguments in a different order. With real
artifacts, integration-gate step 5 will stop with TypeError before exercising
the remaining checks.


2. [P1] Dataset helpers use obsolete positional APIs

Files:
  src/lot/phase5.py, lines 361-363
  src/lot/phase5_gate.py, lines 168-169

load_scene_pairs expects (renders_root, scene, config=...), while
subsample_by_stratum expects an integer max_per_stratum. These calls pass an
AnalysisConfig in the scene and count positions, respectively, so both the
gate and normal example iterator fail on real data.


3. [P1] Pair-dependent supervision cannot be batched

File: src/lot/train.py
Lines: 185-187

build_example produces one query per supported landing, so M varies by camera
pair. Stacking these tensors raises when two pairs have different support
counts. The integration gate conceals this by truncating every example to the
shortest length; padding and an unsupported mask are described there but never
implemented.


4. [P1] CUDA training leaves every batch on CPU

File: src/lot/train.py
Line: 312

The model moves to device, but examples created from NumPy and cache tensors
remain on CPU, and _collate performs no device transfer. Calling train_fold
with device="cuda" therefore fails at the first forward pass. The gate avoids
this by manually moving its synthetic batch, so it does not test the actual
training path.


5. [P1] Checkpoint selection does not enforce the validation split

File: src/lot/train.py
Lines: 353-354

Training checks only that its batch excludes test scenes; it does not require
membership in fold.train. Validation is consumed without any role check, so
test scenes can silently influence checkpoint selection, and validation scenes
can enter optimization. The tiny-overfit entry point similarly accepts
arbitrary scenes without a fold.


6. [P1] Near-zero wording ignores the required second path

File: src/lot/phase5_estimands.py
Lines: 211-228

PROTOCOL 3.9 licenses "small, sign-consistent effect" only after comparing
both evaluation paths on their common-valid population, with both intervals
excluding zero. This function sees one estimate and can issue that wording
even when the splat path disagrees or reverses sign.


7. [P2] Phase 5 omits required L2 companion metrics

File: src/lot/phase5_score.py
Lines: 66-85

The frozen protocol requires raw and centered cosine with L2 companions for
every scored record. Phase 5 computes and propagates cosine only; no Phase 5
source or test contains an L2 implementation, so its eventual result schema
will be protocol-incomplete.


8. [P1] Gate receipts are not bound to the current run identity

File: scripts/run_phase5.sh
Lines: 106-109 and 124-127

The launcher checks only the passed field. It does not compare the receipt's
commit, configuration digest, fold digest, measurement digest, or environment
with the current job. A clean later commit or changed configuration can
therefore reuse a stale PASS receipt.


9. [P1] Direct SLURM submission bypasses both gates

File: scripts/phase5_train.sbatch
Lines: 60-65

The worker checks only for a clean tree before starting training; it never
validates the integration or overfit receipts. Since the documented
configuration also shows direct sbatch usage, the true execution boundary does
not enforce the claimed prerequisites.


10. [P2] An empty training iterator loops forever

File: src/lot/train.py
Lines: 330-337

When train_examples yields nothing, the loop repeatedly recreates an empty
iterator without advancing step or raising. A fold with no usable examples
will occupy a cluster job indefinitely instead of producing a classified stop.


KNOWN INCOMPLETENESS
====================

The branch is not runnable end-to-end. Modes overfit, train, controls, and
evaluate all exit intentionally at src/lot/phase5.py lines 442-446, as the
runbook acknowledges.
```

## Verification, verbatim

```
CODE REVIEW VERIFICATION
========================

Repository: D:\LT
Branch: repair/validation-streams-abc
Review base: ff52d13
Reviewed tip: 7726aef
Working tree at review time: clean

Scope reviewed:
  - Seven local commits after ff52d13
  - Approximately 7,102 added lines across 25 files
  - Phase 5 configuration and cluster orchestration
  - Context-lift geometry and support construction
  - Predictor architecture and training loop
  - Scoring, estimands, and paired bootstrap
  - Integration-gate checks and receipt enforcement
  - Added Phase 5 tests

Verification performed:
  - python -m pytest -q
    Result: 439 passed, 3 skipped in 153.40 seconds

  - python -m compileall -q src tests validation
    Result: PASS

  - git diff --check ff52d13..HEAD
    Result: PASS

  - Reproduced heterogeneous-batch failure in lot.train.batch_loss
    Result: torch.stack rejected query tensors of shapes [5, 2] and [4, 2]

  - Invoked lot.phase5 with --mode train
    Result: exited because the mode is not implemented

No source files were modified as part of the review.
```
