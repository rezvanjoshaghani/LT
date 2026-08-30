# Phase 4 input-state pin (execution plan, Stream G, step 1)

Recorded 2026-08-29, before any VGGT depth is consumed by Phase 4. After this
pin, repository code and configuration are fixed for Phase 4: no source,
test, analysis, transport, launch-template, or .gitignore change until Phase 4
closes. The `check` mode of scripts/run_phase4.sh re-verifies every line of
this file on the cluster before anything runs.

## Commits

- Normative freeze commit: `d4ed1017bd2daca2871da28900b5b4a6a7ff92b6`
  (FREEZE.md; Stream D closure commits precede it).
- Provenance-hardening commit: `0992f7f` (evidence jobs refuse a dirty
  worktree; the SLURM-log ignore rule was already committed at `4787937`).
- Validation package commit: `aa977f4` (Stream F re-audit, pass with
  findings, preserved before any Phase 4 work).
- Phase 4 execution commits. The measurement code was introduced at `b964c42`
  (lot.phase4, lot.phase4_report, the A3 config keys, tests, launch scripts,
  and the validator 2.3 script) and this pin was written at `6ac138d`. Three
  commits follow, none of which touch a measurement path, and the runs are
  made at the last of them:

      96c66a3  cluster readiness. lot/phase4_report.py's paired scene
               bootstrap was re-pooling every record per replicate, 60 s per
               pooled cell; it now sums per-scene (sum, count) pairs, verified
               equal to the pooled-records reference to 3.6e-15 with the
               replicate still recomputing each quantity whole. Also the smoke
               mode's CUDA guard, the gates array mode, and the runbook.
      ea0730b  mode bits on the two launch scripts, content unchanged.
      this one bookkeeping: the inspect mode read vggt.__file__, which is
               None for a namespace package, plus this record.

  src/lot/phase4.py is byte-identical to its `b964c42` form at every one of
  them, so nothing that decides a row has moved since the pin was written.
  The Phase 4 measurement digest below is unchanged across all four commits,
  and every run record carries the commit it actually ran at.

## Frozen artifacts

sha256 of each normative file as committed at the freeze commit, verified
against FREEZE.md at pin time and re-verified by `check` on the cluster:

    517cc4924f8b770c2d27c9f9fecb761c634a920fd21a77ab44e92fe28473eee4  PROTOCOL.md
    bddd31e9d294778f10848028e4755ba56e6ea58f1e633b57eb4d09b29bbb4493  AMENDMENTS.md
    91f3a82ff066bcb029b9df7b9ff9c3da4bba31ea9604d1d17cea48a660b440e6  configs/analysis.yaml
    6e4897ff98b2061a4e8c544dfd30126e8b0b5cb561d4b8e306400f210a7c452e  VALIDATION.md

PROTOCOL.md and VALIDATION.md are byte-identical to their frozen blobs at the
execution commit. AMENDMENTS.md and configs/analysis.yaml differ from their
frozen blobs exactly by the enumerated amendments below, which is the
amendment mechanism working as designed; their live sha256 at the execution
commit:

    392ccc5fcc03c0c3ce89c049de50e9614790337ce815735b8bfb6db87bd7f1d3  AMENDMENTS.md
    41990264fe5c08685b954b93f237a26fb85e2282a073ee80b5551dd136d6dfe0  configs/analysis.yaml

## Post-freeze amendments in force

A1 (parallax median reads target-view depth), A2 (nonfinite-score scope),
A3 (the four Phase 4 execution keys), A4 (alignment application and the
per-point estimated lift), A5 (4.8 mask operationalizations and the 5d
landing rule). All predate any Phase 4 result.

## Identities

- Phase 3 measurement digest: `27244e6481d521159e513f2ea8799482`
  (unchanged by A3; the corrected Phase 3 parquet loads under the amended
  config, verified).
- Phase 4 measurement digest: `1579714398feff4771a9981e5f427c8a`.
- Full config digest at pin: `146ae78d35fff7425131b5e793c56022`;
  reporting digest: `949d14c5c88d5d8c5514ba6ab1db3ee9`.

## Input artifacts

- Corrected Phase 3 evaluation outputs (the frozen population Phase 4
  reuses): aggregate sha256
  `7de0ae525087806c7a7c1e147691934d67f57e7bb9dbedf61e596ee6fd6bc9a6` over 42
  files in sorted-path order; per-file digests in
  validation/evidence/reaudit/outputs_pin.txt.
- DINOv2 feature cache: per-scene `features_digest` values as carried in the
  18 pinned Phase 3 run records (weights fingerprint
  `1159bd1e21ae359a232648228319ab05`; checkpoint declared unpinnable per
  3.12; hub code revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`).
  Evaluation re-verifies each scene's bytes against its digest before
  scoring.
- VGGT depth cache (Phase 2 export, 518 x 518 with confidence): weights
  fingerprint `b9e29c09ce793d15bf3abdc14048838a`, weights revision
  `860abec7937da0a4c03c41d3c269c366e82abdf9`, inference code revision
  `a288dd0f14786c93483e45524328726ab7b1b4ce`. Per-scene `depth_digest`
  values live in the cache metadata on the cluster; `check` re-reads them
  and every Phase 4 run record carries them forward.
- Mean vector: reused from outputs/experiment_zero with its provenance
  record and vector digest; the loader refuses a mismatch.
- Manifest-set content hash, computed on Borah 2026-08-29 and recorded here
  rather than only in gitignored evidence, closing the deferral that re-audit
  dated note 1 left open:

      99bf3e938ffeeb9a2c82a919ee37824c632a539901fb783fe37064e1f14b9046

  over 18 manifests, as the sha256 of the sorted
  `sha256sum data/replica_renders/*/manifest.json` listing. The per-file
  listing is outputs/phase4_rung1/evidence/manifest_files.txt.

## Structural guarantees asserted before the run

- Target-frame ground truth cannot enter any Phase 4 calibration estimator:
  Level 1 excludes the target by construction with a per-record assertion,
  Level 2 and affine read the context image only, and the suite pins both
  (tests/test_phase4.py). PROTOCOL 4.3.
- The forced-collision machinery cannot change ordinary transport: it is an
  isolated copy whose forcing-disabled form is asserted equal to the frozen
  plan at run time and in the suite.
- The 4.5 gate tolerances, the confidence rule, and the mask
  operationalizations are config- and amendment-resident; nothing is
  introduced or loosened at run time.

## Verified on Borah, 2026-08-29

`./scripts/run_phase4.sh check` passed at `ea0730b`: frozen blobs verified
against FREEZE.md, all five amendments listed, clean worktree, all four
inputs present, both caches re-read and valid across 18 scenes (DINOv2
`1159bd1e21ae`, VGGT `b9e29c09ce79` at revision `860abec7...` and code
`a288dd0f...`, matching the pins above), Phase 3 measurement digest
`27244e6481d521159e513f2ea8799482` and Phase 4 measurement digest
`1579714398feff4771a9981e5f427c8a` both as recorded, suite 262 passed and 3
skipped. Frame counts corroborate the Phase 1 record independently: 17 scenes
at 288 plus frl_apartment_2 at 240 is 5,136.

Depth convention, PROTOCOL 4.1's primary authority: established as planar
camera-z from the installed VGGT source. `vggt/utils/geometry.py`'s
`depth_to_cam_coords_points`, which is what VGGT's own
`unproject_depth_map_to_point_map` uses to turn `predictions["depth"]` into
points, assigns `z_cam = depth_map` directly with `x_cam` and `y_cam` scaled
by depth over focal length, term for term the same as `lot.geometry.unproject`.
A ray-distance head would have to divide by the secant of the pixel angle
there and does not. The deterministic secant regression of 4.1 is therefore
run as a consistency check, with the verdict passed in as `--doc-verdict
planar_z`. Evidence: outputs/phase4_rung1/evidence/vggt_source_inspection.txt.

## Cluster execution order (all modes refuse a dirty worktree)

    ./scripts/run_phase4.sh check
    ./scripts/run_phase4.sh inspect          # VGGT depth semantics, evidence
    ./scripts/run_phase4.sh convention [doc-verdict]
    ./scripts/run_phase4.sh gates            # PROTOCOL 4.5, stop on failure
    sbatch --array 0-17 scripts/phase4.sbatch configs/phase4.yaml
    ./scripts/run_phase4.sh figures
    ./scripts/run_phase4.sh smoke            # GPU node, permanent 3.1 tests
    ./scripts/run_phase4.sh validate23       # Stream F closure, Addendum E

The validator 2.3 script was rehearsed on this machine against a synthetic
scene evaluated by the real pipeline: 280 of 280 rows reproduced with
bit-identical masks and every metric within 7.6e-7 of the frozen 1e-4
tolerance (validation/borah_check_2_3.py docstring records the independence
boundary).
