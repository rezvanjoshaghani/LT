# Phase 5 input-state pin (Stream P, step 2)

Recorded 2026-08-30, before any Phase 5 model is trained and before any Phase 5
test metric exists. The `check` mode of `scripts/run_phase5.sh` re-verifies
every line of this file on the cluster before anything runs, and fills the
entries this machine cannot compute.

The design correction this phase rests on is
`validation/evidence/phase5/design_correction.md`, written and committed before
this pin.

## Gate: is Phase 5 allowed to start

All three conditions of Stream P step 2 are satisfied.

- Phase 4 accepted, 2026-08-30. FINDINGS.md "Phase 4 rung 1"; all five
  acceptance conditions re-derived from the shipped artifacts by
  `scripts/phase4_acceptance_check.py`, which passed all five.
- Validator 2.3 PASS. apartment_0, 930 pairs, 9,300 rows, zero mask and zero
  count mismatches, worst metric residual 5.07e-07 against the frozen 1e-4.
- No unresolved Phase 4 correctness blocker. The two recorded Phase 4 anomalies
  (low-texture surfaces paying less rather than more, and the orbit
  conditioning) are findings against PLAN.md's stated expectations, recorded
  and not tuned. Neither is a correctness defect and neither gates Phase 5.

## Commits

- Normative freeze commit: `d4ed1017bd2daca2871da28900b5b4a6a7ff92b6`.
- Accepted Phase 4 implementation: `28d0b96` (Phase 4 accepted, Addendum E
  closed), with the acceptance conditions closed on evidence at `ff52d13`.
- Phase 5 implementation HEAD at pin time: `ff52d13`, working tree clean,
  suite 279 passed and 3 skipped. Phase 5 code is added after this pin; every
  Phase 5 run record carries the commit it actually ran at.

## Frozen artifacts, verified on this machine

sha256 of each normative file as committed at the freeze commit:

    517cc4924f8b770c2d27c9f9fecb761c634a920fd21a77ab44e92fe28473eee4  PROTOCOL.md
    bddd31e9d294778f10848028e4755ba56e6ea58f1e633b57eb4d09b29bbb4493  AMENDMENTS.md
    91f3a82ff066bcb029b9df7b9ff9c3da4bba31ea9604d1d17cea48a660b440e6  configs/analysis.yaml
    6e4897ff98b2061a4e8c544dfd30126e8b0b5cb561d4b8e306400f210a7c452e  VALIDATION.md

Live at `ff52d13`:

    517cc4924f8b770c2d27c9f9fecb761c634a920fd21a77ab44e92fe28473eee4  PROTOCOL.md   (identical)
    4ae0d0e720bb47e25c389a3527148b52601d724ef1ad993146ea923a40279708  AMENDMENTS.md (A1-A7)
    41990264fe5c08685b954b93f237a26fb85e2282a073ee80b5551dd136d6dfe0  configs/analysis.yaml
    6e4897ff98b2061a4e8c544dfd30126e8b0b5cb561d4b8e306400f210a7c452e  VALIDATION.md (identical)

PROTOCOL.md and VALIDATION.md are byte-identical to their frozen blobs.
AMENDMENTS.md and configs/analysis.yaml differ from the freeze exactly by
amendments A1 to A7, which is the amendment mechanism working. The live
analysis.yaml hash is unchanged from the value the Phase 4 pin recorded at its
execution commit, so no analysis constant has moved since Phase 4 ran.

## Identities carried forward

- Phase 3 measurement digest: `27244e6481d521159e513f2ea8799482`.
- Phase 4 measurement digest: `1579714398feff4771a9981e5f427c8a`.
- Phase 3 evaluation outputs, the frozen population Phase 4 reused: aggregate
  sha256 `7de0ae525087806c7a7c1e147691934d67f57e7bb9dbedf61e596ee6fd6bc9a6`
  over 42 files in sorted-path order.
- Camera-manifest set hash, computed on Borah 2026-08-29 for Phase 4:
  `99bf3e938ffeeb9a2c82a919ee37824c632a539901fb783fe37064e1f14b9046`
  over 18 manifests. `check` recomputes it and refuses a mismatch.

## Encoder provenance, carried forward from the Phase 4 pin

- DINOv2 ViT-B/14: weights fingerprint `1159bd1e21ae359a232648228319ab05`,
  hub code revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, checkpoint
  declared unpinnable per PROTOCOL 3.12.
- VGGT: weights fingerprint `b9e29c09ce793d15bf3abdc14048838a`, weights
  revision `860abec7937da0a4c03c41d3c269c366e82abdf9`, inference code revision
  `a288dd0f14786c93483e45524328726ab7b1b4ce`.
- Depth convention: `planar_z`, authority `source`, no cosine conversion.
  Amendment A6 applies it globally.

## Phase 5 identities frozen here

- Scene-split (fold) digest: `25f0c03f72d58cc8e3ff2d8d4123241f6459d4ed50d3c303c5dc108bb0e19865`
  over the three folds in `src/lot/phase5_folds.py`, asserted against the
  executable rule by `tests/test_phase5_folds.py`.
- Architecture and training-config digest:
  `19d903812e4c366693717e48e81dba938cd5b87e94ead1d90bab8a38cba03ce1`
  over the model, training, tiny-overfit, and primary-level sections of
  configs/phase5.yaml, asserted by tests/test_phase5_config.py. It excludes
  the output paths, because moving a directory does not change what was
  measured. Predictor parameter count 16,680,960, also asserted.

## Cluster-resident, filled by `run_phase5.sh check`

These cannot be computed on the workstation this pin was written on, because
the inputs live on Borah. `check` computes each one, compares it against the
value the Phase 4 run recorded, and refuses to proceed on any mismatch.

- DINOv2 feature-cache per-scene `features_digest` values, 18 scenes.
- VGGT depth-cache per-scene `depth_digest` values, 18 scenes.
- Context-image-scaled context-depth artifact digest. Phase 5 does not persist
  a second copy of the aligned depth: it recomputes it by calling Phase 4's own
  `frame_calibration` and `aligned_depth`, so the values are identical to
  Phase 4's by construction rather than by comparison. `check` records the
  digest of the recomputed maps per scene, and the run records carry it.
- Phase 4 evaluation artifact hashes, per scene, for the parquets whose
  context-image-scale scored-cell masks define the secondary operational
  support and the formulation-diagnostic population.

## Primary geometry condition

Phase 4 Level 2, context-image oracle-scaled VGGT depth, applied to the
**context** frame's map, for both headline methods. Estimated by
`lot.phase4.frame_calibration` from the pair's context image alone. Neither
headline method receives target-frame ground-truth depth or target-frame VGGT
depth at any point.

Predeclared secondary conditions, run only after the primary result is frozen
and interpreted: context-image affine calibration (sensitivity), and native
unaligned context VGGT depth (optional systems diagnostic). Neither replaces
the primary.

## What is not yet pinned

Model checkpoint hashes, evaluation artifact hashes, and the realized fold and
seed identities are recorded at closure, in the Phase 5 verdict, because they
do not exist yet.
