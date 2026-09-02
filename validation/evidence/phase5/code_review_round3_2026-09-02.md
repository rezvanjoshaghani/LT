# Phase 5 code review, round 3, 2026-09-02, and its disposition

Third review of the Phase 5 branch, against base `9150b56`, reviewed tip
`cbef973`. It confirmed the round-two repairs and found five new issues, all
verified and all fixed. Finding 1 was reproduced numerically before any code
changed: 0.332 of path difference from collision multiplicity alone, with both
operators agreeing at every cell.

## Disposition

**1, the cross-path arms used different weighting units.** The most
consequential. The per-point arm scored every context sample landing in a
common cell while the splat arm scored the cell once, so a cell receiving two
samples counted twice on one path and once on the other. The target patch
cell is now the atomic unit on both arms: the supported per-point predictions
and targets landing in a common cell are pooled to one vector each before
scoring, which is PROTOCOL 3.7's output-level rule for pooled outputs, and both
arms then average over the same cells with one weight per cell. The common
cells, the per-point mask, and the samples-per-cell counts travel with the
record so an aggregate can prove what it rests on. The reviewer's collision
case, rebuilt with landings on exact patch centers, now gives a path
difference of exactly zero, and the integration gate runs the producer on a
real pair and stops if pooling loses or duplicates a sample.

**2, the gate verified a probe subset and compared nothing.** The pin had
described comparisons the implementation did not make. `verify_scene_identities`
now runs over every scene the folds name, reads the accepted identity out of
each Phase 4 parquet's own run record, compares the live feature-cache digest,
depth-cache digest, and manifest digest against it, hashes the parquet, and
stops on any disagreement as a frozen-design mismatch. The aligned context
depth, which Phase 4 never persisted, is recomputed through Phase 4's own code
for every scene and its digest recorded. The loaded mean vector is checked
against the digest Phase 4 centered with.

**3, receipts compared paths, not content.** The binding is now by content.
Every file artifact is re-hashed; every per-scene identity the gate recorded is
recomputed from the live inputs and compared; `mean_vector_dir` and
`phase4_convention` are bound; a receipt without an artifact block, or without
a scene-identity block, or naming an artifact this run cannot resolve, or
missing a bound field, is refused with the reason named. The test the reviewer
identified as not testing what its name claimed was replaced with one that
exercises the real comparison with only the loaders stubbed.

**4, the two-path disclosure covered one quantity.** Every interpreted effect
is now declared: each has a disclosure pair on the cross-path population, or is
declared single-path by construction, which only the formulation diagnostic
is. An interpreted effect with neither raises rather than falling back to a
one-path wording. The margins are disclosed against their operational
counterparts through common-support No-Warp columns, and the operational gap
is paired with the same pair as the headline.

**5, the cross-path record dropped the L2 companions.** All six arms carry all
four PROTOCOL 3.7 columns, and a producer-level test pins them to
`lot.evaluate.value_agreement` on the pooled vectors, to float32 rounding
because that function casts to float32 and the producer keeps float64.

Regression coverage is in tests/test_phase5_cross_path.py (new),
tests/test_phase5_estimands.py, tests/test_phase5_receipt.py, and
tests/test_phase5_gate.py. Suite at the fix: 518 passed, 3 skipped.

## What this round says about the pin

Until this fix the pin's "cluster-resident, filled by check" section described
a comparison the code did not perform. The text was written as a specification
and the implementation lagged it; that gap is exactly what a review is for, and
it is recorded here rather than smoothed over. The section is now accurate.

## The reviewer's own note, carried forward

The review recorded that `score_cross_path` had no direct unit test and was not
called anywhere in `src`, so producer-level defects could survive a green
suite. Both are addressed: the producer has its own test module, and the
integration gate calls it on a real pair at step 7.

---

## Findings, verbatim

```
PHASE 5 CODE REVIEW - ROUND 3
============================

Date: 2026-09-02
Repository: D:\LT
Branch: repair/validation-streams-abc
Reviewed tip: cbef973
Comparison point: 9150b56
Functional fix commit: a0e106a

SUMMARY
-------

This review re-audited the five round-two findings after a0e106a and reviewed
the new common-support scoring, receipt binding, configuration identity,
training guards, tests, and evidence changes. The sign/interval veto ordering,
complete configuration digest, mandatory tiny-gate arguments, and validation
role on input-use controls are fixed. Five actionable issues remain.


1. [P1] THE CROSS-PATH ARMS USE DIFFERENT WEIGHTING UNITS

File: src/lot/phase5_score.py
Lines: 531-540

cross_path_cells chooses common target cells, but score_cross_path then sends
every per-point sample landing in those cells to score_primary while sending
one value per cell to score_splat_pool. If several context samples land in one
target cell, the per-point arm weights that cell several times and the splat
arm weights it once. This is not a common observational population and can
manufacture a path difference from collision multiplicity alone.

Adversarial reproduction with two common cells:

  per-point samples per cell: 2 and 1
  per-cell explicit effects:  +1 and -1 on both paths
  per-cell learned effects:     0 and  0 on both paths

Observed:

  per-point effect          0.3333333333333333
  splat-pool effect         0.0
  reported path difference 0.3333333333333333

The operators agree exactly at each common cell; only their weights differ.
The record compounds the problem by storing n_intersect as the number of cells
while the per-point term actually averages more samples, and by discarding the
common cell/sample mask, so the aggregate cannot later prove or reconstruct
its support.

Required correction: put both arms on one atomic unit and one weighting rule.
For example, pool the per-point predictions and targets to one normalized value
per common cell before scoring, or read the splat field at each common
per-point identity. Persist the exact common mask/identities in the aggregated
record and test a collision case with unequal samples per cell.


2. [P1] THE INTEGRATION GATE DOES NOT VERIFY THE ARTIFACT SET THE PIN CLAIMS

Files:
  src/lot/phase5_gate.py, lines 97-101
  src/lot/phase5_check.py, lines 205-218
  validation/evidence/phase5/pin.md, lines 104-119

The pin states that check verifies feature and depth digests for all 18 scenes,
the recomputed aligned-depth digest, and every Phase 4 evaluation artifact,
comparing them with the accepted Phase 4 values and refusing a mismatch.

The implementation calls hash_scene_artifacts only for PROBE_SCENES, which is
three scenes. That helper hashes only each probe manifest and Phase 4 parquet;
it does not hash the feature cache, depth cache, or recomputed aligned maps. It
also only records the hashes and never compares them with an accepted identity.
The other 15 scenes and the inputs named by the pin can therefore drift while
the gate still returns PASS.

Required correction: derive and compare the frozen identities for all 18
scenes and every cluster-resident artifact named by the pin. A mismatch must be
a gate stop, not merely evidence copied into a receipt.


3. [P1] RECEIPT ARTIFACT BINDING COMPARES PATHS, NOT CONTENT

File: src/lot/phase5_receipt.py
Lines: 120-138

_artifact_problems rebuilds a mapping of artifact names to resolved paths and
compares only those strings. It ignores the sha256 and byte counts that the
gate records, ignores probe_scene_hashes entirely, and excludes
mean_vector_dir and phase4_convention from BOUND_ARTIFACT_FIELDS. It also skips
an artifact when its current path is missing instead of rejecting the receipt.

Consequently, changing bytes under the same path after the gate leaves receipt
verification green. A direct check also showed that a receipt carrying wrong
paths and stale hashes for mean_vector_dir and phase4_convention produces no
artifact problems. The new test named
test_the_artifact_check_compares_the_paths_the_gate_recorded uses
mean_vector_dir, which is not in the bound-field tuple, so it passes without
testing the behavior its name claims.

Required correction: require the complete artifact identity block for an
integration receipt, recompute and compare every frozen hash/digest, include
all required artifacts, and treat a missing current artifact or missing
identity field as a failure.


4. [P2] TWO-PATH DISCLOSURE IS IMPLEMENTED ONLY FOR THE HEADLINE GAP

File: src/lot/phase5_estimands.py
Lines: 422-435

DISCLOSURE_PAIR contains only delta_learn_pp. Other reported score-space
effects with real counterparts, including cl_margin versus
sp_transport_margin, predict_margin versus sp_predict_margin, and the
operational delta_learn_sp, are handed to near_zero_disclosure with no second
path. A small margin is therefore labeled "single path" even though both paths
exist.

For example, a per-point cl_margin of 0.002 currently receives
"no measurable difference at the reported scale, single path" without
consulting a splat margin that can be far outside the 0.003 band or reverse
sign. PROTOCOL 3.9 applies its two-path wording to any interpreted effect no
larger than the tolerance, not only to one quantity name.

Required correction: add the common-support No-Warp fields and mappings needed
to disclose both margins, pair the operational gap in the reverse direction,
and define explicitly which reported quantities are interpreted effects. Do
not silently fall back to a single-path wording when a counterpart exists.


5. [P2] THE NEW CROSS-PATH RECORD DROPS THE REQUIRED L2 COMPANIONS

Files:
  src/lot/phase5_score.py, lines 460-468
  src/lot/phase5_estimands.py, lines 88-93

CrossPathScores and INTERSECTION_FIELDS carry raw and centered cosine only.
They omit raw and centered L2 for every explicit and learned arm. This
reintroduces the same schema defect fixed after round one: the frozen record
contract requires raw and centered cosine with their unit-normalized L2
companions for every scored record.

Required correction: compute and persist x_*_l2_raw and x_*_l2_centered for all
four cross-path arms, include them in the intersection field set, and add a
producer-level test that compares the values with lot.evaluate.value_agreement.


KNOWN INCOMPLETENESS
--------------------

Modes overfit, train, controls, and evaluate remain intentionally unavailable
at src/lot/phase5.py lines 472-475. The new score_cross_path helper is not
called anywhere in src yet and has no direct unit test; the estimand tests
inject synthetic x_* columns instead. This is not counted as a separate
finding because the runbook already declares the end-to-end pipeline
incomplete, but it is why producer-level defects survive a green suite.
```

## Verification, verbatim

```
PHASE 5 CODE REVIEW - ROUND 3 VERIFICATION
==========================================

Date: 2026-09-02
Repository: D:\LT
Branch: repair/validation-streams-abc
Reviewed tip: cbef973
Comparison point: 9150b56
Starting working tree: clean

SCOPE REVIEWED
--------------

  - Functional fix commit a0e106a
  - Evidence-only consolidation commit cbef973
  - Re-audit of all five round-two findings
  - Cross-path score production and estimand consumption
  - Near-zero disclosure coverage and paired path-difference bootstrap
  - Full Phase5Config digest coverage
  - Gate receipt identity and resolved-artifact checks
  - Tiny-overfit and validation-control split guards
  - New and changed Phase 5 tests and pin claims

AUTOMATED VERIFICATION
----------------------

Command:
  python -m pytest tests/test_phase5_config.py tests/test_phase5_estimands.py
      tests/test_phase5_receipt.py tests/test_phase5_score.py
      tests/test_phase5_train.py -q
Result:
  118 passed in 18.96 seconds

Command:
  python -m pytest -q
Result:
  486 passed, 3 skipped in 158.89 seconds

Command:
  python -m compileall -q src tests
Result:
  PASS

Command:
  git diff --check 9150b56..HEAD
Result:
  PASS

TARGETED ADVERSARIAL CHECKS
---------------------------

1. Cross-path weighting

A synthetic ContextLiftMap placed three supported context samples into two
common target cells, with multiplicities 2 and 1. The explicit and learned
predictions were identical between paths at each cell.

Observed CrossPathScores:
  x_cl_raw            0.3333333333333333
  x_predict_raw       0.0
  x_sp_transport_raw  0.0
  x_sp_predict_raw    0.0

Observed path difference:
  0.3333333333333333

Expected when both paths use the common cells with equal weights:
  0.0

2. Configuration digest coverage

Changing every previously omitted scientific field now changes cfg.digest():

  feature_encoder               unchanged digest: False
  depth_encoder                 unchanged digest: False
  phase4_dir                    unchanged digest: False
  renders_root                  unchanged digest: False
  cache_root                    unchanged digest: False
  controls                      unchanged digest: False
  seed                          unchanged digest: False
  sensitivity_alignment_levels unchanged digest: False

Only the declared relocation fields remain excluded:

  output_root                   unchanged digest: True
  experiment_name               unchanged digest: True

This round-two finding is fixed.

3. Artifact receipt coverage

Input receipt artifact block:
  mean_vector_dir with a wrong path and stale sha256
  phase4_convention with a wrong path and stale sha256

Observed _artifact_problems result:
  []

Static tracing confirmed that current artifact content hashes and
probe_scene_hashes are never compared.

4. Near-zero counterpart coverage

Observed from evaluate_quantity on small margins:

  cl_margin      splat_pool: None, wording: single path
  predict_margin splat_pool: None, wording: single path
  delta_learn_sp splat_pool: None

Only delta_learn_pp appears in DISCLOSURE_PAIR.

5. Test coverage trace

No test calls score_cross_path or cross_path_cells. The new estimand tests
populate x_* values synthetically and therefore verify the bootstrap consumer,
not the producer that constructs the common-support values.

REVIEW ARTIFACT EFFECT
----------------------

No source or test files were modified. This review added only the two plain
text evidence files named in this directory.
```
