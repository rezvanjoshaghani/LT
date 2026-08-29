# VALIDATION_REPORT.md

Re-audit of this repository against the frozen protocol, executed per the frozen
VALIDATION.md. This report supersedes, and does not modify, the previous audit
report (verdict FAIL at commit `c5e50f9`), which is preserved in git history:
its final revision is the blob with sha256
`4a9666208689d30036d6262c86710fb6d24541865c56e9ef3c1e30fcda5d7221`, last touched
at commit `3420e4e`. The validator did not write, fix, or modify anything under
`src/` or `tests/`. New validator code from this audit is under `validation/`
with the `reaudit_` prefix; new mutated copies under `validation/mutants/ra_*`;
raw evidence under `validation/evidence/reaudit/`. Pre-existing files under
`validation/` are the previous audit's artifacts and the implementation's
committed evidence (the path-agreement ledger and path-margin tables); none were
modified.

**Verdict: recorded at the end of this report after all checks.**

---

## 0. Kickoff record

### K1. Normative artifact verification against FREEZE.md — PASS

First action, per VALIDATION.md and FREEZE.md. Hashes are of the file content
as git stores it.

```bash
for f in PROTOCOL.md AMENDMENTS.md configs/analysis.yaml VALIDATION.md; do
  git show d4ed1017bd2daca2871da28900b5b4a6a7ff92b6:$f | sha256sum
  git show HEAD:$f | sha256sum
done
```

| file | FREEZE.md sha256 | frozen blob | HEAD blob |
|---|---|---|---|
| PROTOCOL.md | `517cc492…73eee4` | match | match |
| AMENDMENTS.md | `bddd31e9…bb4493` | match | match |
| configs/analysis.yaml | `91f3a82f…b440e6` | match | match |
| VALIDATION.md | `6e4897ff…7c452e` | match | match |

All four match FREEZE.md exactly, at the freeze commit and at HEAD. No blocker
at the gate. AMENDMENTS.md is a header with no entries, so per FREEZE.md the
protocol stands exactly as frozen.

### K2. Git state at kickoff

- HEAD: `4729e3abf9ab9dbcb6bdd55efb7a45795a401ff5` (the bookkeeping commit
  FREEZE.md names), branch `repair/validation-streams-abc`.
- Working tree at kickoff: **clean** (`git status --porcelain=v1` empty).
- Freeze commit `d4ed101` is an ancestor of HEAD; the four normative files are
  identical at the freeze commit and at HEAD.
- FREEZE.md's claim that no source, test, evaluation, evidence-generation, or
  analysis code changed between the Stream D closure commits and the freeze was
  verified: `git diff --stat 2acbfbd d4ed101 -- src/ tests/ configs/` is empty.
- Every evidence entry below cites HEAD `4729e3a` unless stated otherwise.

### K3. Data-artifact pinning (the kickoff content hashes)

VALIDATION.md requires content hashes of the rendered scenes under `data/`, the
frozen feature cache under `cache/`, and the evaluation outputs under
`outputs/`, because git does not track them.

**Ordering rule** (recorded per VALIDATION.md): files are visited in sorted-path
order; paths are relative to the repository root with forward slashes; each
file's sha256 is computed over its raw bytes; the aggregate is the sha256 of
the concatenation of `"<hex-digest><2 spaces><relative-path>\n"` lines in that
same sorted order. Any other ordering is not a reproduction.

- **outputs/** (42 files, the complete Stream D evaluation outputs, figures and
  tables): aggregate sha256
  `7de0ae525087806c7a7c1e147691934d67f57e7bb9dbedf61e596ee6fd6bc9a6`.
  Per-file digests: `validation/evidence/reaudit/outputs_pin.txt`.
- **data/ and cache/ are not present on this machine.** They reside on the
  Borah cluster, which is not reachable non-interactively from this audit
  session (ssh requires interactive authentication; verified refused in
  BatchMode). Consequences and the partial substitute are in K5 and the
  Unverified list. The feature cache is pinned indirectly and per scene: every
  evaluation run record carries, per encoder, the cache `features_digest` (a
  content hash over the stored arrays computed when the cache was written and
  re-verified against the bytes at evaluation time by
  `_SceneCache`/`validate_feature_cache(check_digest=True)`), the
  `weights_fingerprint`, `weights_revision`, and `code_revision`. All 36
  (scene, encoder) digests are recorded in the run records inside the pinned
  outputs above, so the cache identity the run consumed is pinned even though
  the cache bytes are not locally hashable.

### K4. Run provenance of the artifacts under audit

All 18 per-scene run records agree on: `eval_version: 4`, `seed: 0`,
`git_commit: 61c9e99e3f7ad027bc79c64302e87e8283a1c117-dirty`,
`analysis_config_digest: ca4da48246e3c01b6cade9c42b328a94`,
`analysis_measurement_digest: 27244e6481d521159e513f2ea8799482`,
`analysis_reporting_digest: a497a98e44359e73e3a9cabb3cb481f5`, one weights
fingerprint per encoder (`1159bd1e…` DINOv2, `b9e29c09…` VGGT), one weights
revision per encoder (DINOv2: the explicit `unpinnable: torch.hub checkpoint
URL is unversioned` declaration PROTOCOL 3.12 permits, with hub `code_revision`
`7764ea0f…`; VGGT: `860abec7…` with code revision `a288dd0f…`). One encoder
identity across all scenes, as 3.12 requires.

Three provenance facts require explicit treatment:

1. **The run commit is an ancestor of the freeze, and the row-producing code is
   frozen unchanged.** `git diff 61c9e99 d4ed101 -- src/` touches only
   `analysis_config.py` (+2 field declarations), `figures.py`,
   `path_ledger.py` (new), `path_margins.py` (new), and tests.
   `evaluate.py`, `correspondence.py`, `transport.py`, `geometry.py`,
   `visibility.py`, `encoders.py`, `datasets.py`, `sample_identity.py` are
   byte-identical between the run commit and the freeze commit. The parquet
   under audit was therefore produced by exactly the frozen row-producing code.
   The post-run code (ledger, margins, figures changes) produced the post-run
   evidence and the shipped figures/tables, and stabilized at `d4f9679`, before
   the closure commits; nothing changed after them (K2).
2. **The `-dirty` marker.** `lot.evaluate.git_commit()` appends `-dirty`
   whenever `git status --porcelain` prints anything, which includes untracked
   files. At run time (2026-08-27, before commit `4787937` added `slurm-*.out`
   to .gitignore) the cluster worktree necessarily carried untracked SLURM logs,
   which alone triggers the marker. That is the benign explanation; it cannot
   be proven from here what else, if anything, was dirty on the cluster
   checkout, and the marker cannot distinguish untracked logs from tracked
   modifications. The analysis config is bound by content digest (below), so a
   config edit could not hide; source edits are not covered by any digest.
   `lot.figures.read_eval_dir` prints exactly this caveat when analysing the
   run. Recorded as **FINDING R-1 (major)** — see Findings.
3. **Config digest linkage.** The frozen `configs/analysis.yaml`'s measurement
   digest, computed by the frozen reader, is
   `27244e6481d521159e513f2ea8799482` — **exactly equal** to the measurement
   digest all 18 run records carry. The run's measurement identity is the
   frozen config's measurement identity, which is what PROTOCOL 3.12 requires
   for a report to be built from the run. The full-config digest differs
   (`f7595c5c…` now vs `ca4da482…` recorded) and the reporting digest differs
   (`24ade4ef…` vs `a497a98e…`), both explained exactly by the two
   ledger-tolerance keys (`ledger_recon_tol`, `ledger_closure_tol`) added
   post-run and pre-freeze at `d4f9679`, which sit outside `MEASUREMENT_FIELDS`.
   A textual diff (`git diff 61c9e99 d4ed101 -- configs/analysis.yaml`) confirms
   no measurement-side value changed. PROTOCOL 3.4/3.12 permit reporting-side
   changes; `read_eval_dir` gates on the measurement digest and notes the
   reporting difference. Consistent.

Timeline consistency (file mtimes and commits): support counts 08-27 14:50
(counts step precedes any outcome view per the runbook), ledger evidence
08-28 03:27, figures 03:55–03:57, results table 04:02, margins evidence 14:29,
closure and freeze commits 14:33–14:36. Consistent with the documented order.

### K5. What cannot be executed from this machine

`data/` (renders: RGB, depth, manifests, frame stats) and `cache/` (frozen
features) exist only on the cluster; interactive-only authentication blocks a
remote audit session. Everything below that requires them is listed as
unverified, not passed, in the sign-off. Checks that could run on a surrogate
(synthetic scenes driven through the real pipeline) were run and are labelled
surrogate. The evaluation parquet, figures, tables, ledger report, cut points,
and margins evidence are all local and were audited directly.

### Dated notes (VALIDATION.md instructs the validator to record these)

**Note 1, 2026-08-28 — artifact location.** VALIDATION.md's own first-note
requirement: the artifact-location gap the earlier audit found is resolved by
the kickoff content hashes of K3 for `outputs/`, and by the per-scene cache
digests carried in the pinned run records for `cache/`. For `data/` no local
pin is possible; the manifests' content is pinned only transitively (the run
records do not hash the renders). This residual gap is recorded here rather
than silently absorbed.

**Note 2, 2026-08-28 — where dated notes live.** VALIDATION.md says its own
ambiguity notes are "recorded as dated notes at the bottom" of itself, but
FREEZE.md freezes VALIDATION.md by content hash and any edit would break K1.
Resolution: dated notes are recorded in this report instead; VALIDATION.md is
left byte-identical to its frozen hash. Ambiguity recorded as a protocol-gap
note (minor).

**Note 3, 2026-08-28 — rotation angle formula.** VALIDATION.md 1.2 and 3.5
check for "arccos of (trace−1)/2 with the argument clamped" and a
clamp-removal mutant. PROTOCOL 3.2 states the arccos-with-clamp definition, but
PROTOCOL 3.12 (adopted before the freeze) states "Rotation magnitude is
computed from the skew term against the trace term, not from the trace alone",
and the implementation follows 3.12 (`atan2` of skew magnitude vs trace term,
`geometry.py:206-231`). The two frozen texts describe the same mathematical
quantity with different numerics; 3.12 is the more specific and later-settled
text and controls. Consequences: 1.2 is audited as "the 3.12 formula, correctly
implemented, with no domain-error exposure"; mutant 3.5 is executed against the
bug class the clamp guards (arccos domain overshoot) by mutating
`rotation_angle_deg` to an unclamped `acos((trace−1)/2)`. Both deviations from
VALIDATION.md's letter are recorded here as required by its own ground rule 5.

---

## Part 1: Static conformance audit

Audited at HEAD `4729e3a`; `src/` is byte-identical to the freeze commit.

### 1.1 Pose convention — PASS

`relative_pose` is defined once, [geometry.py:140](src/lot/geometry.py:140):
`invert_se3(T_world_from_target) @ T_world_from_context`, with the docstring
"This is the only place the formula is written."

```bash
grep -n "invert_se3\|linalg.inv\|relative_pose" src/lot/*.py | grep -v geometry.py
```

Every consumer imports it: [datasets.py:190](src/lot/datasets.py:190),
[evaluate.py:1030](src/lot/evaluate.py:1030),
[path_ledger.py:217](src/lot/path_ledger.py:217). The two other `invert_se3`
call sites ([correspondence.py:240](src/lot/correspondence.py:240),
[visibility.py:91](src/lot/visibility.py:91)) invert an already-computed
relative transform to get `T_context_from_target`; neither re-derives the
relative pose from world poses. The only `linalg.inv` is the intrinsics inverse
inside `rotation_homography` ([geometry.py:158](src/lot/geometry.py:158)), not a
pose inverse. **No duplicate.**

### 1.2 Rotation angle — PASS under PROTOCOL 3.12 (see dated note 3)

[geometry.py:206](src/lot/geometry.py:206) `rotation_angle_deg`: float64,
`sine = ‖R − Rᵀ‖_F / (2√2)`, `cosine = (trace(R) − 1)/2`,
`degrees(atan2(sine, cosine))`. This is exactly PROTOCOL 3.12's "skew term
against the trace term". It is total on its domain (no arccos domain error to
clamp), agrees with the 3.2 geodesic definition mathematically on [0, 180],
resolves small angles to full precision (which 3.12 requires so the
zero-rotation bin is a statement about the camera, not arithmetic), and
`datasets.pair_quantities` computes it from `T_target_from_context[:3, :3]`
([datasets.py:196](src/lot/datasets.py:196)). Independent numeric check in
2.1's harness: agreement with an independent implementation at 0, 1e-7, 7.5,
33, 179 degrees, and finite output on a near-identity rotation scaled by
(1 + 1e-13). Mutation coverage of the arccos-overshoot bug class: Part 3, 3.5.

### 1.3 Pixel-to-patch mapping — PASS

Defined once at [encoders.py:45](src/lot/encoders.py:45)
(`(uv_px + 0.5) / patch_size − 0.5`), inverse beside it at
[encoders.py:74](src/lot/encoders.py:74). `sample_features_bilinear`
([encoders.py:142](src/lot/encoders.py:142)) routes through it;
`patch_cell_index` ([encoders.py:53](src/lot/encoders.py:53)) rounds the same
mapping, and its docstring records that the earlier inline rewrites in
correspondence.py and evaluate.py were removed. Grep for a second copy of the
constant found none. `align_corners` appears nowhere in `src/`; the single hit
in `tests/test_one_path_pipeline.py:103` uses `F.interpolate(...,
align_corners=True)` to synthesize a smooth random test fixture, not to sample
features — not a violation of "no align_corners in feature sampling"; noted.

### 1.4 Transport — PASS

[transport.py:77](src/lot/transport.py:77) `transport_plan`: splat at pixel
resolution over the full `pixel_grid`; z-buffer `scatter_reduce_(amin)` at
pixel level ([transport.py:135](src/lot/transport.py:135)) with deterministic
tie-averaging within a 1e-6 relative epsilon; pooling to patches afterwards;
coverage = supported-pixel fraction per target patch
([transport.py:168](src/lot/transport.py:168)); rows of the weight matrix sum
to one where anything landed and zero in holes, so empty patches stay zero.
Disoccluded locations are not scored: the splat record set is
`covisible_per_patch >= min_covisible_fraction` AND `coverage > 0`
([evaluate.py:552-555](src/lot/evaluate.py:552)). The weight-matrix form vs
per-pixel splat equivalence is asserted by
`test_weighted_form_matches_the_pixel_level_splat` and re-verified
independently in Part 2 (surrogate).

### 1.5 Nulls — PASS (the blocker condition does not fire)

All three location controls read from the same context feature map
([correspondence.py:324-351](src/lot/correspondence.py:324)) and differ only in
read location:

- **Neighbor-Patch reads one patch away from the correct correspondence
  location** (`options = uv_warp[:, None, :] + offsets`,
  [correspondence.py:255](src/lot/correspondence.py:255)) and is transported
  identically on the splat path (same plan weights applied to a one-patch
  shifted source, [evaluate.py:691](src/lot/evaluate.py:691)). VALIDATION 1.5's
  blocker (the other reading, one patch from the same image coordinate) does
  **not** fire. The direction is drawn hash-deterministically from the
  record's `sample_id` among the in-bounds axis-aligned unit offsets
  ([correspondence.py:73-90](src/lot/correspondence.py:73), salt
  `NEIGHBOR_PATCH_SALT`), matching the frozen 3.6 text (which superseded the
  fixed-direction wording pre-freeze). Border records with no in-bounds offset
  are omitted and counted (`neighbor_omitted`). The admissible set is the
  intersection of both paths' rules so one record has one direction on both
  paths ([correspondence.py:257-272](src/lot/correspondence.py:257),
  [evaluate.py:544-546](src/lot/evaluate.py:544)).
- **No-Warp-Copy** reads at the same image coordinate as the target location
  (`uv_context_no_warp=uv_target.clone()`,
  [correspondence.py:316](src/lot/correspondence.py:316)).
- **Random-Patch** is an integer patch of the same context image, never across
  scenes (`derived_draw(ids, RANDOM_PATCH_SALT, ctx_patches_h * ctx_patches_w)`
  over the context grid, [correspondence.py:307](src/lot/correspondence.py:307);
  indexed, never interpolated, [correspondence.py:350](src/lot/correspondence.py:350)),
  hash-deterministic per record per the frozen 3.6.
- `sample_id` (PROTOCOL 3.2) exists and is load-bearing:
  [sample_identity.py](src/lot/sample_identity.py) derives a 64-bit id from
  scene, context frame, target frame, and half-pixel-quantized target
  coordinates via full-width blake2b + SplitMix64; uniqueness asserted per pair
  ([evaluate.py:352](src/lot/evaluate.py:352)); both paths index one universe
  (target patch grid) so records intersect by construction
  ([evaluate.py:378](src/lot/evaluate.py:378)); rows persist the validity
  bitmask (`sample_mask`) as 3.2 permits for pair-aggregated storage.

### 1.6 Masks — PASS

`visibility_masks` takes only ground-truth depth of both views
([visibility.py:71](src/lot/visibility.py:71)); grep for
estimated/vggt/predicted depth in the module: no hits. The co-visibility
tolerance is read lazily from the normative config
([visibility.py:51](src/lot/visibility.py:51)), not hard-coded; callers pass
`config.covisible_relative_depth_tol` explicitly
([evaluate.py:485](src/lot/evaluate.py:485)). The context depth map is read
with nearest sampling with the documented z-buffer rationale. The blocker
condition of VALIDATION 1.6 does not fire, and the prior audit's MINOR-1
(hard-coded tolerance) is repaired.

### 1.7 Metrics — PASS

`agreement_metrics` ([evaluate.py:159](src/lot/evaluate.py:159)) computes raw
cosine and centered cosine (plus L2 companions) for every record. "Centered"
subtracts **the encoder's global mean vector** — the same object as the
Mean-Feature floor — from both vectors immediately before unit-normalization
and cosine ([evaluate.py:151-156](src/lot/evaluate.py:151)), which is exactly
PROTOCOL 3.7's output-level definition. The vector is built by
`dataset_mean_vector` ([evaluate.py:188](src/lot/evaluate.py:188)): one global
[C] vector per encoder, averaged over all frames and positions of the
**training split** (`mean_vector_scenes` defaults to the train split,
[evaluate.py:906-912](src/lot/evaluate.py:906)), never a position-conditioned
map — matching 3.6/3.7 as frozen. The same vector is the Mean-Feature
prediction on both paths ([evaluate.py:792](src/lot/evaluate.py:792),
[evaluate.py:837](src/lot/evaluate.py:837)). Centered Mean-Feature is recorded
nonfinite, no epsilon-regularized substitute
([evaluate.py:159-181](src/lot/evaluate.py:159)). Margins are never baked into
rows; `figures.paired_records` computes them at analysis time and verifies the
variant and floor share the same persisted `sample_mask` before subtracting
([figures.py:484-587](src/lot/figures.py:484)). All five variants are scored on
the path's common valid record set (per-point: the sampler's selected set;
splat: the splat mask), so differences are paired by construction with the mask
as proof ([evaluate.py:764-864](src/lot/evaluate.py:764)). The prior audit's
BLOCKER-2 and MAJOR-1 are repaired. The mean vector's provenance is bound by
digest of inputs and of its own bytes
([evaluate.py:274-345](src/lot/evaluate.py:274)).

### 1.8 Schema — PASS (wide-metric form as frozen in 3.2)

Confirmed on the shipped parquet (Part 4): rows are long in variant and path,
wide in metric, and carry scene_id, context/target frame ids, camera regime,
variant, path, continuous `rotation_deg`, continuous `parallax`, contributing
count `n`, and the named metric columns — exactly the fields PROTOCOL 3.2
enumerates, plus the intersection columns and persisted `sample_mask` that 3.2's
pairing identity requires. **No bin labels in rows** (`parquet` columns contain
no `*_bin`; `figures.assign_bins` refuses rows that already carry labels,
[figures.py:437-453](src/lot/figures.py:437)). Bin edges live only in the
committed config; `datasets.py` reads the same config for sampling strata
(deliberately separate stratum edges, 3.4-adopted). The per-pair parallax is
computed as PROTOCOL 3.2 defines: median of baseline over ground-truth depth
across the pair's co-visible point set — implemented as
`baseline / median(depth_target[covisible])`
([evaluate.py:449-462](src/lot/evaluate.py:449)), with the documented
equivalence (baseline constant per pair) and the correct population (the
co-visible set, not the whole frame; MAJOR-4 repaired). The whole-frame proxy
survives only as a sampling covariate that never reaches rows
([datasets.py:14-20](src/lot/datasets.py:14)); run records document this.
Protocol gap (minor): 3.2 does not name which view's depth enters the median;
the implementation uses the target-view depth of each co-visible point. See
Findings (G-1).

### 1.9 Encoders — PASS

- DINOv2 preprocessing: ImageNet mean/std ([encoders.py:172](src/lot/encoders.py:172),
  applied at [encoders.py:257-260](src/lot/encoders.py:257)); no resize or crop
  by design, with the documented rationale that the renders are whole-patch
  sized and a resize would invalidate the manifest intrinsics (deliberate,
  documented deviation from the letter of "official resize"; carried over as a
  note from the prior audit).
- Frozen: `model.eval()`, `requires_grad_(False)` on load
  ([encoders.py:311-318](src/lot/encoders.py:311)); every call under
  `torch.inference_mode()` ([encoders.py:326-331](src/lot/encoders.py:326)).
- VGGT builds a length-one sequence axis: `views = x[:, None]`
  ([encoders.py:438](src/lot/encoders.py:438)); tokens captured from the
  aggregator by forward hook; `_squeeze_view_and_channel` asserts the view axis
  is length one ([encoders.py:485-496](src/lot/encoders.py:485)).
- The single-frame test runs ungated (`test_vggt_sees_one_frame_at_a_time`,
  [test_encoder_cache.py:450](tests/test_encoder_cache.py:450), asserts one
  aggregator call with sequence length 1). The batch-equality and
  grid-orientation tests are gated on `LOT_ENCODER_SMOKE`
  ([test_encoder_cache.py:376](tests/test_encoder_cache.py:376),
  [test_encoder_cache.py:465](tests/test_encoder_cache.py:465)) and the gate is
  now wired: `scripts/cache_features.sbatch:91-92` exports the variable and runs
  exactly those tests **before** caching, under `set -euo pipefail`, so a
  caching job that produced the caches this run consumed can only have run with
  those tests passing (they cannot skip: the variable is set). This repairs the
  prior MAJOR-15. Direct cluster logs are not in the repository (slurm logs
  untracked); the pass is therefore evidenced indirectly (caches exist whose
  meta records the pins this job requires) and the direct confirmation is
  listed as unverified-from-here (Part 5).
- Provenance: fingerprints hash every parameter
  ([encoders.py:264](src/lot/encoders.py:264)); revisions must be full 40-hex
  SHAs (`require_full_sha`, [encoders.py:640](src/lot/encoders.py:640));
  DINOv2's checkpoint unpinnability is declared explicitly per 3.12.

### 1.10 Reproducibility — PASS

Seeds from config (`cfg.seed` threads into stratified subsampling,
[evaluate.py:1014](src/lot/evaluate.py:1014); the correspondence layer is
hash-deterministic and consumes no RNG at all,
[correspondence.py:26-29](src/lot/correspondence.py:26); bootstrap seed in the
analysis config). Configs for the completed runs are committed
(`configs/experiment_zero.yaml`, `configs/cache_features_*.yaml`). Outputs are
never overwritten: `write_rows` refuses ([evaluate.py:1228](src/lot/evaluate.py:1228));
resume validates the stored run record against this run and refuses a mismatch
([evaluate.py:1319-1342](src/lot/evaluate.py:1319)); figures/table refuse
existing outputs and stage atomically ([figures.py:1707-1798](src/lot/figures.py:1707));
the one `replace=True` is the counts view, with the documented rationale, and
the results table is never replaced. Re-verified live on the synthetic
determinism run (Part 4.3).

Part 1 verdict: no blocker fires; every prior Part-1 blocker/major (BLOCKER-1,
BLOCKER-2, BLOCKER-3 schema half, MAJOR-1..15) is verifiably repaired in the
frozen source. Findings from Part 1: R-1 (run made from a dirty worktree,
K4), G-1 (parallax depth-view gap), plus notes.

---

## Part 2: Independent numerical re-derivation

Independent implementation: `validation/independent.py` (written from the
protocol text by the previous audit, committed, imports no `lot`; re-used and
re-run unmodified) driven by `validation/check_geometry.py`, plus the new
`validation/reaudit_forensics.py` and `validation/reaudit_claims.py` (pure
pandas/numpy/yaml, no `lot` imports). Full output:
`validation/evidence/reaudit/check_geometry_rerun.txt`, `forensics.json`,
`claims.json`.

### 2.1 Geometry cross-check — PASS (surrogate; real-data case unverifiable here)

```bash
python validation/check_geometry.py > validation/evidence/reaudit/check_geometry_rerun.txt
# 27 passed, 0 failed, of 27 checks
```

No manifests are on this machine, so the three pairs are synthetic, one per
regime shape, with deliberately different context and target intrinsics; 20
random pixels each are unprojected/transformed/projected independently and
through `lot`. Max disagreement: rotation 1.4e-14 px, translation 0.0 px,
orbit 2.8e-14 px, against the 0.1 px tolerance; the relative-pose matrices
agree to 2.2e-16, and reversing the pose direction moves the answer by 56 px,
so the check is not vacuous. No systematic offset. The same-check on real
manifest pairs requires `data/` and is listed unverified.

### 2.2 Homography check — PASS (surrogate)

The general two-intrinsics homography `K_tgt @ R @ inv(K_ctx)` was computed
independently and compared against the full depth-based reprojection over the
whole pixel grid at three unrelated depths: max disagreement 5.7e-14 px, exact
zero translation, and the single-K shortcut would have been wrong by 2.9 px,
which is why the general form is used. Real-data case: unverifiable here.

### 2.3 One-scene reproduction — the centerpiece is UNVERIFIED (inputs on the cluster); surrogate PASS; aggregate-layer substitute PASS

The designated centerpiece — recompute every Experiment Zero row for
apartment_0 from the frozen caches and manifests — cannot run on this machine:
`cache/` and `data/` exist only on Borah (K5). What was done instead, and what
it does and does not establish:

1. **Surrogate re-derivation (PASS).** The independent implementation of
   geometry, co-visibility, splat, z-buffer, pooling, coverage, and scoring was
   re-run against the frozen `lot` on synthetic two-plane scenes across all
   three regime shapes: co-visible masks agree on all 9604 pixels in every
   regime, transported features to 1.8e-07, coverage to float32 epsilon, with
   exact-zero and exact-one coverage present and no NaNs. This establishes
   convention-correctness of the machinery, not the shipped numbers.
2. **Row-derived aggregate reproduction on the real parquet (PASS, see 2.4 and
   Part 5).** Everything computable from the shipped rows was recomputed
   independently and matches exactly, including the entire shipped results
   table bit-for-bit. This validates every layer from the persisted rows up.
   The step it cannot validate is rows-from-pixels on the real scenes — that
   remains cluster-bound, partially covered by the implementation's own
   committed ledger evidence (reconstruction of recorded scores from stored
   inputs to 2e-7/6e-7 over all 33,772 comparisons; report.json, committed),
   which is implementation self-evidence, not an independent rederivation.

### 2.4 Aggregate reproduction — PASS on the corrected run; historical side impossible

Which definitions produced the parquet under audit: the frozen ones —
`eval_version: 4` rows produced by code byte-identical to the freeze commit
(K4), under the frozen measurement identity (digest match, K4). The
**normative check** therefore ran against the corrected parquet: with
independent pandas code (`reaudit_claims.py`), every quantitative value claimed
in FINDINGS' corrected sections and in the shipped tables was recomputed.
Results in Part 5; headline: the shipped
`outputs/experiment_zero/tables/experiment_zero.parquet` (436 rows) reproduces
**exactly** — every numeric column including all bootstrap intervals is
bit-equal to my regeneration from the frozen code, and my fully independent
recomputation of every value/margin estimate in the primary analyses matches to
float precision (396 rows compared, 0 mismatches). `support_counts.parquet`
reproduces exactly (124 rows).

The historical (pre-correction) parquet does not exist in this repository or on
this machine — it was never synced (prior audit, BLOCKER-4) and was superseded.
Reproducing its aggregates under its own definitions is therefore impossible
here; its numbers (+0.140 etc., 29,196 / 233,536) remain historical
observations exactly as 2.4 instructs, and no bridge recomputations were
needed: the corrected-run consistency was established directly. Listed as
unverifiable, not assumed.

### 2.5 Floors — PASS on everything locally checkable

The centered Mean-Feature scores are nonfinite for all 67,544 Mean-Feature
rows (both paths, both centered columns, and the centered intersect column);
**zero finite centered Mean-Feature values exist** — the pre-correction
signature is absent (`forensics.json: hygiene`). Raw Mean-Feature is present on
both paths as the same single global vector by construction
([evaluate.py:792](src/lot/evaluate.py:792), [evaluate.py:837](src/lot/evaluate.py:837)),
and the row-absence-vs-NA representation is exactly one representation, used
consistently (rows present, centered columns NaN — the 3.2-frozen choice).
Recomputing the global mean vector itself requires the caches (unverified
here); its provenance chain is pinned (input digests + vector digest,
[evaluate.py:274-345](src/lot/evaluate.py:274)).

### 2.6 Depth conventions — PASS on the classifier and the independent 4.1 procedure; manifest metadata unverifiable here

```bash
python validation/check_depth_convention.py > validation/evidence/reaudit/check_depth_convention_rerun.txt
# 7 conformant, 2 flagged (both Phase-4 readiness), of 9
```

- The renderer's classifier is correct on synthetic raw output of known
  convention in all four combinations (fronto-parallel and tilted, both
  conventions), and its decision thresholds now defer to the committed config
  (the prior MINOR-2 is repaired). Re-running it on real raw renders, and
  confirming `metadata.depth_convention` across the 18 manifests, requires
  `data/`: unverified here (the frozen `validate_manifest` refuses manifests
  without a resolved planar-z conviction, so the evaluation run having passed
  is indirect evidence).
- PROTOCOL 4.1's VGGT secant regression was implemented independently from the
  protocol text and classifies both synthetic conventions correctly (slope
  −0.0000 vs +0.83). There is still no implementation in `src/` and no
  resampling code — Phase 4 has not begun; both flags are readiness notes
  (carried as N-1), not conformance failures. The VGGT-documentation route was
  not resolvable offline and remains open for Phase 4.

---

## Part 4: Output forensics

Driver: `validation/reaudit_forensics.py` (no `lot` imports); evidence
`validation/evidence/reaudit/forensics.json`. All checks ran on the real
Stream D parquet (18 scenes, pinned in K3).

### 4.1 Record accounting — PASS, exact reconciliation

Derivation stated before observation (from PROTOCOL 3.2 plus one inspected
file): one record is one (camera pair, encoder, path, variant); metric is a
column dimension (wide), not a row dimension; five variants exist on **both**
paths; centered Mean-Feature is present-with-NaN, not absent, so it adds no
count term; Neighbor-Patch border cases drop samples inside a record, never
records, so they add no term either (and the run counted 0 omissions); a pair
scorable on one path contributes 5 rows instead of 10. Expected records =
Σ_scenes Σ_encoders (10·both + 5·pp_only + 5·sp_only) with the counters taken
from the run records, which are independent of the rows.

Observed: **expected 337,720 = observed 337,720, exact**, with per-scene,
per-encoder counter reconciliation exact for all 36 (scene, encoder) groups,
every (pair, encoder, path) group carrying exactly the five frozen variants,
and 67,544 groups = 2 paths × 33,772 (pair, encoder) comparisons =
2 × (16,895 considered − 9 dropped-unscorable) × 2 encoders / 2. FINDINGS'
33,772 is confirmed as the (pair, encoder) comparison count, every one scored
on both paths. The historical 29,196 / 233,536 cannot be reconciled because
that table no longer exists anywhere; per 2.4 they stay historical
observations, and the corrected run reconciles exactly with **no** unexplained
row difference (the earlier 32-row question is moot for the corrected schema:
neighbor omissions were 0 and no structural omissions exist).

### 4.2 Population checks — PASS on everything the rows can answer

- Distinct scored pairs 16,886: orbit 9,188, rotation 4,108, translation 3,590.
  Design-parameter cross-check against the render programs needs manifests
  (unverified here); the stratum-cap invariant was checked instead: rebuilding
  the sampling strata from the frozen stratum edges and the whole-frame proxy
  (both recoverable from rows) gives 473 strata, max 40 pairs each — exactly
  the frozen cap, never exceeded.
- **Rotation-program pairs: baseline_m is exactly 0.0 and parallax exactly 0.0
  for all 4,108 pairs** — the 3.3 construction claim holds literally in the
  manifest read-backs, well inside `rotation_position_bound_m`, and every
  rotation pair sits in the zero-parallax bin.
- **Translation-program pairs: rotation_deg is exactly 0.0 for all 3,590
  pairs** — inside `translation_rotation_bound_deg` (1e-7°) and the
  zero-rotation bin. 0 pairs violate the manifest bound.
- The forbidden interval (0, 0.025) contains **0 translation pairs** (the 3.4
  assertion holds on the reported statistic); orbit pairs in that interval: 0
  (legitimate but empty at the realized geometry).
- Binning re-derived independently from the config text: every value lands in
  a bin; the rotation regime appears only in the zero-parallax bin and the
  translation regime only in the zero-rotation bin.

### 4.3 Hygiene — PASS

- **Grain uniqueness:** 0 duplicates at
  (scene, context_frame_id, target_frame_id, encoder, path, variant) over
  337,720 rows.
- **Nonfinite audit:** nonfinite scores exist in exactly the single permitted
  representation — the centered columns (and centered-intersect column) of
  Mean-Feature rows, all 67,544 of them, and nowhere else. Raw cosine/l2:
  0 nonfinite. Intersect scores: 0 rows with empty intersection, 0 nonfinite
  with support. `coverage_mean` is NaN on per-point rows (coverage is a
  splat-path property, not a protocol-defined score; recorded as decided
  reading G-2, minor protocol gap).
- **Mask-level consistency (the 3.2 pairing identity, verifiable from rows
  alone):** for all 67,544 (pair, encoder, path) groups the five variants carry
  one identical persisted mask; popcount(mask) == n in all cases;
  popcount(pp_mask AND splat_mask) == n_intersect for all 33,772 comparisons;
  coverage_difference equals popcount(own AND NOT other) on every row.
  0 violations.
- **Determinism** (`validation/reaudit_determinism.py`): the real pipeline over
  a synthetic scene, evaluated twice from the same inputs and once from a
  freshly rebuilt scene directory: 450 rows, field-by-field identical
  (NaN==NaN for the permitted representation) in both comparisons; `write_rows`
  overwrite refusal confirmed live. Real-scene determinism requires the caches:
  covered indirectly by the hash-deterministic sampler design (no RNG in the
  correspondence layer) and listed unverified directly.

---

## Part 5: Claim-by-claim verification

Scope: FINDINGS' claim-bearing sections are the corrected (Stream D) sections —
"Methods notes, corrected Phase 3", "Path agreement, attributed", and
"Phase 3: Experiment Zero, corrected verdict". The 2026-08-24 verdict section
is explicitly withdrawn and non-citable ("Nothing here may be cited"), so its
numbers are audited only as what the withdrawal notice says they are. Phase 1
and Phase 2 sections are claims about artifacts on the cluster; what the rows
can corroborate was checked, the rest is inventoried unverifiable-from-here.

Driver: `validation/reaudit_claims.py`; evidence
`validation/evidence/reaudit/claims.json`. 48 checks; 45 pass; the 3 failures
are Findings F-1, F-2, F-3 below. Highlights, every one recomputed
independently from the parquet and config alone:

- **3.9 gate:** signed aggregate **+0.000115 raw, +0.000175 centered** over
  33,772 comparisons vs the frozen 0.003 — exact match to FINDINGS, to the
  ledger's signed T2, and to the frozen pipeline's own printed gate output
  (regeneration log). Dispersion diagnostics: mean |d| 0.00304 / 0.00417,
  median 0.00072 / 0.00139 — match.
- **Dispersion structure:** by rotation bin, zero-rotation 0.00389 (claim
  0.0039), 50-plus 0.00082 (claim 0.0008); dispersion falls as the common set
  grows (Spearman −0.157). Composition note N-4: the zero-rotation pool is
  3,590 translation + 1,227 orbit pairs, so "which are the translation
  programme" is imprecise prose; the number is right.
- **DINOv2 headline series** (cross-path common-set basis, which is FINDINGS'
  stated dual-path methodology): rotation raw margins 0-10°
  +0.23556/+0.23517 and 50+ +0.15965/+0.15866; translation raw 0.025-0.05
  +0.05677/+0.05690 and 0.4+ +0.12645/+0.12271 — all match the quoted
  +0.2356/+0.2352, +0.1596/+0.1587, +0.0568/+0.0569, +0.1264/+0.1227.
- **DINOv2 beats No-Warp-Copy in every supported cell of both primary
  analyses, both metrics, both paths** — confirmed, zero failures. Centered
  moves every DINOv2 primary-cell margin up — confirmed. "Changes no
  ordering" — fails for exactly one near-tied adjacent pair on the splat path
  (F-2).
- **VGGT rotation series, centered:** all six bins, both paths, match the
  quoted values exactly; monotone decreasing; raw traces the same path at a
  tenth the magnitude (+0.0097 → −0.0519); the margin is negative from 20°
  upward on both paths and both metrics, with max path disagreement 0.00101
  (the claim "within 0.001" holds at its own printed precision; strictly
  exceeded by 1.4e-5 in one bin — noted, N-5).
- **Near-zero machinery:** 232 supported cells; 27 flagged; all 27 VGGT raw
  (no DINOv2 cell in the band — confirmed); cases 22 / 1 / 4; the four
  not-robust cells are exactly the four named orbit oracle cells; the four
  individually quoted cells match to 5 decimals with the right case labels.
  **My fully independent margins table (point estimates, scene-bootstrap
  intervals with the frozen seed, and case labels) equals the shipped
  `path_margin_differences.parquet` on all 248 rows with max value difference
  0.0 and zero case disagreements** — but the FINDINGS sentence splitting the
  22 as "sixteen localization gaps, six Oracle margins" contradicts both my
  table and the shipped table, which say 18 and 4 (F-1).
- **One-patch cost range:** supported DINOv2 cells span [0.0351, 0.1370]
  (per-point [0.0355, 0.1370]); the claim "between 0.036 and 0.137" verifies at
  rounding precision.
- **Path differences on DINOv2 quantities:** max 0.0066 (claim ≤ 0.013 ✓),
  81.8% below 0.002 ("less than 0.002 on most" ✓), max ratio 4.2% of the
  effect ("under 10 percent in every supported cell" ✓), 68.2% under 2% ("the
  large majority" ✓).
- **Materiality sentence:** fails under the inclusive reading (F-3).
- **Ledger claims** (verified against the committed
  `path_agreement_ledger/report.json`; the underlying 27M per-cell rows are
  cluster-only): verdict PASS with empty stop list over 33,772 comparisons per
  metric; closure 7.4e-16 / 6.6e-16 ("7e-16" ✓); T3 exactly 0.0; reconstruction
  max |T1| 2.0e-7, max |T4| 6.3e-7 vs the frozen 1e-4 (claim "2e-7 and 6e-7"
  ✓; the "five hundred times inside" phrase is exact for T1 and generous for
  T4 at 158× — N-6); preflight bit-mismatches all zero; boundary contrast
  10.9%/12.2%/40.6%/30.5% (claims 11/12/41/31 ✓) with every interval excluding
  zero and 86.1% of cells tripping the flag (claim 86% ✓); norm contrast absent
  for three of four (intervals straddle zero) and present for VGGT raw at
  −53.8% of level (claim −54% ✓) with Spearman +0.55 ✓; per-cell |c| 0.01298
  DINOv2 vs 0.00124 VGGT raw (claim 0.0130 vs 0.0012 ✓).
- **Methods-note calibration sentence:** "less than 4 percent of ... the 0.072
  one-patch localization cost" — 0.003/0.072 = 4.17%, strictly not less than
  4% (F-4, minor). The same note's governing re-check ("flagged for amendment
  if it exceeds roughly 10 percent") holds against the corrected margins:
  0.003 / 0.0351 = 8.5% < 10%.
- **Scoping claims:** the length-one sequence axis is asserted by an ungated
  test that runs and passes in the baseline suite
  ([test_encoder_cache.py:450](tests/test_encoder_cache.py:450)); the
  batch-equality test against real weights is wired to run, ungated, before
  every caching job under `set -e` (`scripts/cache_features.sbatch:91-92`), so
  the caches this run consumed are indirect evidence it passed on the cluster;
  the direct cluster log is not in the repository — listed unverified.
- **Corrected analogs of frozen illustrative observations** (not pass/fail
  targets): PROTOCOL 3.7's motivating example (withdrawn-run 0.967 vs 0.964)
  reads 0.9528 vs 0.9536 pooled in the corrected run — the saturated-scale
  story stands, the constants are stale (N-2). PROTOCOL 3.8's "roughly 0.80 to
  0.54" ceiling decay reads 0.807 → 0.515 corrected — monotone decay confirmed,
  endpoint constant stale (N-2).
- **Row-level corroborations of Phase 1/2 claims:** 18 scenes (13 train,
  5 test) ✓; 107 viewpoints = 17×6 + frl_apartment_2's 5 ✓; 5,001 distinct
  frames used, consistent with 5,078 usable of 5,136 ✓ (the totals themselves
  need manifests). The remaining Phase 1/2 quantities (48 frames per
  viewpoint, navmesh recomputation, depth-convention metadata per scene,
  throughput, norm-concentration diagnostics 0.9095/0.4226, cache shapes and
  channel counts) resolve against manifests and caches on the cluster:
  inventoried **unverifiable from this machine**, not assumed.
- The historical first-run section's numbers (7,300 pairs, the saturated VGGT
  band, 0.265 vs 0.2518, and every value in the withdrawn verdict) are
  historical-run claims whose artifacts no longer exist; the withdrawal notice
  and 2.4's governing paragraph make them non-normative. Not verifiable,
  and not required to be.

---

## Part 3: Mutation tests

Harness: `validation/reaudit_mutate.py` + `validation/reaudit_run_one.py`;
validator-defined test `validation/reaudit_test_overshoot.py`; evidence
`validation/evidence/reaudit/mutation_report.json`. Each mutant is a **fresh
copy of the current frozen `src/lot`, `tests`, and `configs`** under
`validation/mutants/ra_*` (the pre-existing mutant directories are the previous
audit's committed artifacts and were not touched). Each runs in its own
subprocess; the driver imports `lot` first, prints `lot.__file__`, and
hard-exits with code 97 unless it resolves inside that mutant's directory — no
run returned 97, and the control's provenance line is in the stored evidence,
so every kill below was measured on the mutant's own code. `--maxfail=25`
truncates the failure list for heavily-failing mutants; a kill needs one.

| id | mutation | result | verdict |
|---|---|---|---|
| ra_control | none | **244 passed, 3 skipped** (242 baseline + the 2 validator overshoot tests; skips are the three documented gates) | **GREEN** |
| ra_3.1 | `relative_pose` returns context-from-target | 16 failed, incl. `test_relative_pose_definition_and_identity`, `test_composing_two_transforms_equals_direct`, the two-plane and correspondence tests | **KILLED**, by the tests 3.1 names |
| ra_3.2 | pixel-to-patch mapping shifted +0.5 patch | 25 failed, incl. `test_pixel_to_patch_mapping_formula`, `test_feature_sampling_at_patch_centers_is_exact`, splat/transport placement and cross-path read tests | **KILLED**, by the tests 3.2 names |
| ra_3.3 | z-buffer disabled (all splats win, occlusion unresolved) | 5 failed, incl. **`test_occlusion_resolves_by_depth_not_by_write_order[1]` and `[-1]`** | **KILLED — by the occlusion test itself.** The prior audit's MINOR-6 (the old occlusion test asserted only on the zbuffer diagnostic) is repaired: the suite now has an occlusion test that fires on the pooled features |
| ra_3.3b | farthest-wins z-buffer, zbuffer diagnostic left correct | same 5 failures | **KILLED** — feature-level occlusion covered independently of the diagnostic |
| ra_3.4 | `unproject` treats depth as ray distance | 25 failed, incl. reprojection, correspondence, and pipeline tests | **KILLED**, by the tests 3.4 names |
| ra_3.5 | `rotation_angle_deg` replaced by **unclamped acos of the trace term** (the bug class the 3.2-clamp guards; see dated note 3) | 20 failed: ordinary pose arithmetic in the suite raises `math domain error` (the overshoot condition is exercised), `test_rotation_angle_resolves_below_the_zero_rotation_tolerance` fails (the 3.12 resolution requirement), and the validator-defined `test_overshot_trace_argument_yields_finite_angle` fails, while passing on the control | **KILLED**, suite and validator test both |

### 3.6 Target leak — PASS (signature present, unambiguous)

`validation/reaudit_semantic.py` (fresh `ra_sem_*` mutants of the frozen
source; the committed probe scene with surface-attached features; provenance
line per run). Substituting the true target feature into Oracle-Transport's
per-point prediction moves the score from +0.941 to **+0.999999 (min over
pairs +0.999999166)** in raw cosine and from +0.938 to **+1.000000 (min
+0.999999762)** centered — the cosine-of-itself signature within 1e-6, raw and
centered, with no partial inflation. The shipped path is not quietly leaking:
its baseline sits far from 1 while the deliberate leak pins it there.

### 3.7 Correspondence shuffle — PASS (identity is load-bearing)

Within-pair permutation of the warp locations (read-location set unchanged,
pairing destroyed), on the single probe scene, so pair-level bootstrap per
VALIDATION 3.7's own provision: baseline paired Oracle-over-No-Warp margin
+0.1343 [+0.0637, +0.2125] raw (+0.1430 centered) over 40 pairs; shuffled
−0.7022 [−0.7709, −0.6314] (−0.7605 centered). **623% / 632% of the margin
destroyed, intervals disjoint** — far past the ≥50% criterion. The full-scene-
set version of this control requires the caches: unverified here, with the
single-scene fallback exercised exactly as the frozen text permits.

---

## Findings

### Blockers

**None.** The hash gate passed; the run's measurement identity equals the
frozen config's; no Part 1 blocker condition fired (Neighbor-Patch reads one
patch from the correct correspondence; visibility touches no estimated depth);
sample identity exists and is verifiably load-bearing; the corrected-run
accounting closes exactly.

### Majors

**R-1 (major) — the Stream D evaluation ran from a worktree marked dirty.**
All 18 run records carry `git_commit: 61c9e99…-dirty`. The marker fires on any
`git status --porcelain` output including untracked files, and untracked SLURM
logs are structurally guaranteed at run time (the job templates write
`slurm-*.out` into the repo root; they were gitignored only afterwards at
`4787937`), so the benign explanation is strongly implied — but a dirty flag
cannot distinguish untracked logs from uncommitted source edits, so the
run-to-source link has one unprovable step. Mitigation, verified: the
row-producing modules are byte-identical between the run commit and the freeze;
the analysis config is bound by content digest; and the committed ledger
evidence reconstructs every recorded score from the caches and manifests
through the closure-era (= frozen) code to 2e-7/6e-7 with zero bit-level
preflight mismatches over all 33,772 comparisons, which re-derives the rows'
populations and scores under provable code. The residual exposure is small and
explicitly bounded, but the direct proof (a clean tree at run time) does not
exist, and the protocol's provenance posture ("locked means retrievable")
warrants recording it at this severity. Resolution path: none retroactively;
for Phase 4, run evaluations from clean commits (the `-dirty` refusal could be
promoted from a warning to a gate).

**F-1 (major) — a FINDINGS sentence contradicts its own evidence table.**
"Sixteen are localization gaps … Six are Oracle margins" (the split of the 22
both-in-band cells). Both my independent recomputation and the shipped
`validation/evidence/path_margin_differences.parquet` say **18 localization
gaps and 4 Oracle margins**. Every other number in that section verifies,
including the 27/232, the 22/1/4 case split, and all four individually quoted
cells to five decimals; the licensed wording is identical for both quantities,
so no conclusion changes — but the sentence fails its query and Part 5's
frozen rule grades that major. Fix: erratum via AMENDMENTS.md-adjacent record
(FINDINGS is not frozen; correct the two words and cite this finding).

**F-3 (major) — the materiality sentence's arithmetic doesn't hold under its
natural reading.** "The observed operator agreement, +0.000115 raw and
+0.000175 centered, is between 3 and 5 percent of even the smallest
interpreted effect." The smallest effect carrying a licensed claim (a
both-in-band cell) is 0.0013, giving **8.9% and 13.5%**. The sentence's
arithmetic matches only the single one-in-band cell (0.00374 → 3.1% and 4.7%),
i.e. the smallest effect whose direction-claim is licensed with path-sensitive
magnitude. Under that narrow reading the sentence is true; the reading is not
the one the words most naturally carry, and the 22 both-in-band cells are
explicitly "interpreted" by the same section. Fix: restate with the intended
referent (or the inclusive ratios); no conclusion turns on it because the
band's cells' claims are jointly licensed by both paths by construction.

### Minors

- **F-2 (minor)** — "The centered metric moves every value up and changes no
  ordering": the first half verifies everywhere; the second fails for exactly
  one adjacent near-tie on the translation splat path (0.1-0.2 vs 0.4+:
  raw 0.1217 < 0.1227, centered 0.1382 > 0.1347). Per-point orderings and all
  rotation orderings are unchanged.
- **F-4 (minor)** — methods-note calibration sentence: "less than 4 percent of
  … the 0.072 one-patch localization cost" is 0.003/0.072 = **4.17%**. The
  governing 10% re-check against corrected margins holds (8.5%).
- **G-1 (minor, protocol gap)** — PROTOCOL 3.2's parallax ("median of per-point
  baseline over ground-truth depth across the pair's co-visible point set")
  does not name which view's depth; the implementation uses the target-view
  depth of each co-visible point ([evaluate.py:449](src/lot/evaluate.py:449)).
  Record the decision in AMENDMENTS.md.
- **G-2 (minor, protocol gap)** — `coverage_mean` is NaN on per-point rows
  (coverage is a splat-path property). PROTOCOL 3.2's "no other row may carry a
  nonfinite score" is read as covering scores, which coverage is not; the
  reading should be recorded. Same reading applies to the centered-intersect
  column of Mean-Feature rows (structural N/A, consistent with 3.2's one
  representation).

### Notes

- **N-1** — PROTOCOL 4.1's secant regression and nearest-neighbor resampling
  have no implementation yet; Phase 4 has not begun (readiness, not
  conformance). The validator's independent implementation of the secant
  procedure is demonstrated correct on both synthetic conventions and is
  available for the Phase 4 cross-check.
- **N-2** — Two illustrative constants inside frozen PROTOCOL prose predate the
  corrected run: 3.7's motivating example (0.967 vs 0.964; corrected pooled
  analog 0.9528 vs 0.9536 — the saturated-scale point stands, now with a
  slightly negative pooled margin) and 3.8's "roughly 0.80 to 0.54" (corrected
  0.807 → 0.515 — monotone decay confirmed). PROTOCOL is never edited; worth a
  clarifying AMENDMENTS entry when one is next written.
- **N-3** — VALIDATION.md instructs appending dated notes to itself while
  FREEZE.md pins it by hash; resolved by keeping it byte-frozen and recording
  notes here (dated note 2).
- **N-4** — the zero-rotation dispersion pool is 3,590 translation + 1,227
  orbit pairs; FINDINGS' "which are the translation programme" is imprecise
  prose around a correct number (0.00389).
- **N-5** — "agree there to within 0.001" holds at FINDINGS' own printed
  precision; the strict maximum is 0.00101 (centered, 20-30°).
- **N-6** — "five hundred times inside the frozen 1e-4 tolerance" is exact for
  the per-point reconstruction (2.0e-7) and generous for the splat side
  (6.3e-7 ≈ 158×).
- **N-7** — the shipped figures were rendered by matplotlib 3.10.9; this
  audit's environment has 3.10.3. Regeneration from the frozen code and the
  shipped parquet reproduces the results table **bit-exactly** (436 rows, every
  numeric column including all bootstrap intervals) and all four figures with
  identical dimensions but version-different pixels. PLAN Phase 8's
  byte-identical criterion will need a pinned plotting environment.
- **N-8** — VALIDATION.md 1.2/3.5's arccos-clamp wording is superseded by
  PROTOCOL 3.12's skew-vs-trace mandate, which the implementation follows;
  audited accordingly (dated note 3), with the overshoot bug class still
  mutation-covered (ra_3.5).

---

## Unverified — listed, not assumed

Everything below requires `data/` and `cache/` on the Borah cluster, which
this session cannot reach non-interactively (K5):

1. VALIDATION 2.3's centerpiece: the independent row-by-row reproduction of
   apartment_0 from the frozen caches, manifests, and persisted sample
   identities. The aggregate layer above the rows is fully and exactly
   verified (2.4, Part 5); the pixels-to-rows step is covered only by the
   implementation's own committed ledger evidence, which is self-evidence.
2. 2.1/2.2 on real manifest pairs (surrogates passed).
3. Independent recomputation of the global mean vector and the raw
   Mean-Feature scores from the caches (representation and object-identity
   checks passed from rows and source).
4. The renderer depth-convention probe on real raw output, and
   `metadata.depth_convention` across the 18 manifests (classifier verified on
   ground truth; `validate_manifest` refuses unresolved conventions, and the
   run passed it — indirect).
5. Manifest-derived population design: 5,136 = 107 × 48, per-regime frame
   totals 1,381/1,790/1,907, usable 5,078, per-scene depth statistics, navmesh
   recomputation records. (Row-level corroborations passed: 18 scenes 13/5,
   107 viewpoints with frl_apartment_2 = 5, frames-used ⊆ usable.)
6. The direct cluster log of the real-weights batch-equality and
   grid-orientation tests (wiring verified: they run ungated before every
   caching job under `set -e`; the caches' existence is indirect evidence).
7. The per-cell ledger parquets (~27M rows) behind the committed
   `report.json`; the report's every quoted number was verified against it,
   and the margins table was reproduced independently and exactly, but the
   cells themselves are cluster-only.
8. Real-scene byte-determinism of evaluation (synthetic-scene determinism
   passed; the sampler is RNG-free by design).
9. Content pinning of `data/` (dated note 1); `cache/` is pinned per scene via
   the digests carried in the pinned run records.

---

## Sign-off

**Verdict: pass with findings — no blockers; 3 majors (R-1, F-1, F-3); 4
minors (F-2, F-4, G-1, G-2); 8 notes.**

VALIDATION.md's three-value vocabulary (pass / pass with minor findings /
fail) does not name this grade; recording the mismatch per ground rule 5
rather than rounding in either direction. The operational consequence the
frozen text attaches to a verdict — "Phase 4 implementation does not start
until blockers are resolved" — is not triggered: there are no blockers. The
majors are two erratum-level prose corrections in FINDINGS (F-1, F-3), whose
fixes change no conclusion and should be recorded and re-verified against the
queries in `validation/evidence/reaudit/claims.json` before FINDINGS is cited,
and one provenance flag (R-1) that is unresolvable retroactively and should be
closed going forward by refusing dirty-tree evaluation runs.

What this audit established positively, in one paragraph: the four normative
artifacts verify against FREEZE.md at the freeze commit and at HEAD; the
frozen source conforms to the frozen protocol on every Part 1 item, with all
four blockers and all fifteen majors of the previous audit verifiably
repaired; the corrected run's parquet was produced by byte-identical-to-frozen
code under the frozen measurement identity, its accounting closes exactly at
337,720 rows = 33,772 comparisons × 2 paths × 5 variants with zero
mask-consistency violations, its populations honor every regime and binning
invariant including exact-zero baselines for rotation pairs and exact-zero
rotations for translation pairs, and the only nonfinite values are the single
permitted representation; the 3.9 gate passes on the frozen statistic at the
frozen tolerance (+0.000115 raw, +0.000175 centered) and reproduces
independently; the shipped results table and support counts reproduce exactly
(the table bit-for-bit, bootstrap intervals included); the dual-path margins
table reproduces independently to zero difference with identical
classifications; of 48 preregistered claim checks, 45 pass and the 3 failures
are the prose findings above; all five prescribed mutants plus a supplementary
one are killed under proven provenance with a green control, including the
occlusion bug class the previous audit found uncovered; and both semantic
negative controls show their preregistered signatures decisively. The
validator signs off only on what was checked; everything in the Unverified
list is unverified, not assumed.

---

## Reproducing this audit

```bash
# kickoff hashes: FREEZE.md's own loop, plus validation/evidence/reaudit/outputs_pin.txt
python validation/reaudit_forensics.py       # Part 4 + population/mask checks
python validation/reaudit_claims.py          # Part 2.4 + Part 5 claim ledger
python validation/check_geometry.py          # 2.1/2.2/2.3 surrogates (prior harness, re-run)
python validation/check_depth_convention.py  # 2.6 (prior harness, re-run)
python validation/reaudit_determinism.py     # 4.3 determinism + overwrite refusal
python validation/reaudit_mutate.py          # Part 3 mutants (fresh copies, ra_*)
python validation/reaudit_semantic.py        # 3.6 / 3.7 controls (fresh ra_sem_*)
PYTHONPATH=src python -m lot.figures --eval-dir outputs/experiment_zero/eval \
    --out-dir validation/evidence/reaudit/regen   # regeneration comparison
```

| evidence file (validation/evidence/reaudit/) | contents |
|---|---|
| `outputs_pin.txt` | per-file and aggregate sha256 of outputs/ (K3) |
| `forensics.json` | Part 4 accounting, hygiene, masks, populations |
| `claims.json` | the 48-check claim ledger with every recomputed value |
| `mutation_report.json` | Part 3 run tails and return codes |
| `semantic_controls.json` | 3.6/3.7 raw aggregates |
| `check_geometry_rerun.txt`, `check_depth_convention_rerun.txt` | prior-harness re-runs |
| `regen/` | frozen-code regeneration of figures and table |
