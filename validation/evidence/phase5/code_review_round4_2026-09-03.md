# Phase 5 code review, round 4, 2026-09-03, and its disposition

Fourth review of the Phase 5 branch, run against `ff52d13..cb8eae2` at high
effort. Ten findings, all verified against the source before any code changed,
and all fixed. Suite at the fix: 543 passed, 3 skipped.

## The pattern this round exposes

Three of the ten were introduced by the round-three fixes themselves, one day
earlier. That is the fourth consecutive round in which repairs made under review
pressure created the next round's defects, and it is the most useful thing this
review says. The three were:

- the overfit receipt made unverifiable by the round-three receipt hardening;
- the near-zero trigger moved onto the intersection terms by the round-three
  cross-path work;
- the cross-path disclosure terms swept into the near-zero rule's scope by the
  same change.

Each was a local, plausible edit that satisfied the finding in front of it and
broke something one layer away. The countermeasure applied here is not more
care: it is that every one of the three now has a regression test written from
the failure rather than from the fix, so the same class of edit fails the suite
next time.

## Disposition

**1, the overfit receipt could never verify (P1).** `verify()` applied the
integration gate's artifact and per-scene bindings to every receipt, and the
overfit receipt has neither block. Both gates could pass and training would
still refuse to start, with an error naming the wrong gate. Receipts now carry a
`kind`. The integration receipt is bound to the inputs it hashed; the overfit
receipt is bound to the identity and to the digest of the integration receipt
that licensed it, so it cannot be carried across to a run the gate never
examined, and rerunning the gate invalidates it. An unlabelled receipt defaults
to the stricter kind. `stamp_receipt` is now the one function that writes the
identity, so the shape writers produce and the shape `verify` expects cannot
drift.

**2, the near-zero trigger read the wrong number (P1).** The round-three work
overwrote the per-point term with the cross-path intersection estimate, and the
trigger read that. A headline gap of 0.0015, inside the frozen 0.003 band, would
print with no disclosure whenever its intersection terms fell outside it.
PROTOCOL 3.9 scopes the discipline to "an interpreted effect no larger than this
tolerance", which is the effect the cell reports; the two path *terms* are what
the wording branches read. Both are now passed separately, and the cell's own
estimate travels in the disclosure under `reported`.

**3, gate step 6 compared a value with itself (P1).** The check that
CL-Transport and the predictor are scored at one location built both sides from
the same `uv_t` tensor, so its stop was unreachable and the step reported PASS
regardless of what `build_example` did. It now rebuilds the support the way
`build_example` does and compares the example's own `query_patch_coords` against
an independently derived expectation, reporting the maximum coordinate residual.
A source-level test pins the comparison so the tautology cannot silently return.

**4, disclosure terms told they had no second path (P1).**
`SCORE_SPACE_QUANTITIES` was every quantity in `QUANTITY_POPULATION`, which swept
in the cross-path terms and the path differences. `path_difference_learn`, which
is itself the difference between the two paths, would have been labelled as
having none. The near-zero rule is now scoped to `INTERPRETED_EFFECTS`, since all
three of its licensed sentences are claims about an advantage; absolute levels
and disclosure terms are reported with their intervals and no near-zero wording.

**5, an absent input classified as a design mismatch (P1).** A missing Phase 4
parquet was folded into "the live inputs disagree", classified
`frozen_design_mismatch`, whose documented remedy is an amendment; the actual
remedy is copying a file. A missing cache escaped as an uncaught loader
exception and was reported as an implementation bug. Absent inputs are now
collected separately and stop as `missing_artifact`, before the disagreement
stop.

**6, a depth archive decompressed to read one JSON field (P2).**
`verify_scene_identities` and the receipt verifier both called
`load_depth_archive`, which digests and fully materializes the archive, only to
read `meta['depth_digest']` — the value `load_cache_meta` returns from the small
meta.json, and which `load_depth_archive` itself reads there first. Roughly 11 GB
of reads per 18-scene pass, paid twice in the gate and again at every SLURM task
start. Both now read the meta.

**7, Phase 4's depth preparation was copied (P2).** `build_scene_inputs`
inlined Phase 4's per-frame sequence while its docstring claimed the depth was
Phase 4's "by construction". Phase 4 is pinned and cannot be refactored to share
the code, so the two are now held together by evidence instead: the sequence
lives in `prepare_frame_depth`, and `tests/test_phase5_depth_equivalence.py`
transcribes Phase 4's inline steps from its own source and requires bit-for-bit
agreement, with a further test that fails if Phase 4 stops calling the
functions the transcription assumes. Same discipline `paired_bootstrap` uses.

**8, the frozen metric re-implemented without its cast (P2).**
`score_all_metrics` normalized in whatever dtype it was handed, while
`value_agreement` casts to float32 first. That gave float64 under the suite and
float32 in the shipped path, and would have given fp16 NaNs if the cache tensors
were ever passed directly. The cast is now applied, so the comparability the
record claims with the Phase 3 and Phase 4 tables rests on identical arithmetic.

**9, the documented run order could not be executed (P2).**
`configs/phase5.yaml` and the sbatch header instructed modes `gates`, `tiny` and
`report`, none of which exist since the rename at `fbefe1d`. Corrected to the
implemented sequence, with a note that Stream R's rotation validation runs
inside `check` at step 10 rather than as its own mode.

**10, receipts and the cluster pin overwrote in place (P2).** CLAUDE.md requires
"never overwrite existing outputs", and both were written unconditionally, so a
rerun destroyed the record of the run that gated training. `write_once` moves any
existing file aside to a numbered sibling first, which keeps every earlier
receipt and still lets a rerun proceed.

## What the review could not cover

Three of the eight finder angles, including two of the three correctness angles,
were lost to a session rate limit partway through. The surviving candidates were
verified individually against the source, and a targeted pass covered the ground
the failed angles would have walked, but line-by-line correctness coverage this
round is thinner than the effort level nominally provides. This is recorded so a
later reader does not mistake ten findings for an exhaustive sweep.
