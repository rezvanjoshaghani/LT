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
