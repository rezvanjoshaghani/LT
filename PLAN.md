# PLAN.md

Phases in dependency order. Each phase lists what to build, its acceptance criteria, and the expected outcome in plain terms. Do not start a phase until the previous phase's acceptance criteria pass. If a result contradicts the expectation, stop and write up what was observed; in this project a clean surprise is a result, not a failure.

## Phase 0: Geometry core and harness

Build: geometry.py, visibility.py, transport.py, correspondence.py, and their tests. No downloads, no encoders, pure numpy and torch.

Tests that must pass:
1. Round trip: random 3D points project and unproject to identity; composing two transforms equals the direct transform.
2. Two-plane analytic scene (a front plane partially occluding a back plane, known depths, known cameras): transported features land exactly at analytically computed locations; the occluded strip in the visibility mask matches the analytic answer within one patch.
3. Pure in-place rotation: transport output equals the homography warp of the input, confirming the depth-independent special case.
4. Coverage: exactly 1.0 in fully supported patches, 0.0 in holes, fractional on boundaries, no NaNs anywhere.
5. Correspondence sampler: on the analytic scene it returns exact ground-truth pairs, and the nulls (no-warp copy, spatial neighbor, random patch, mean feature) are constructed correctly.

Acceptance: full test suite green.

## Phase 1: Replica rendering

Build: render_replica.py with three camera programs per scene, plus a per-scene manifest (frame id, pose as T_world_from_camera, K, rgb path, depth path, regime tag, program parameters).

Camera programs:
- In-place rotation: fixed position, yaw and pitch sweeps at fixed angular steps.
- Translation: lateral and forward moves sized to hit target parallax bins (parallax measured as baseline over median scene depth).
- Orbit: rotation around a scene anchor point at two radii.

Also in this phase: a depth-convention test. Render a wall-facing view and check whether depth is constant across the wall (planar z-depth) or grows toward the corners (euclidean ray distance). Convert to planar z-depth if needed and record the finding in the manifest metadata.

Acceptance: one scene rendered end to end; a QC contact sheet of RGB and depth per regime; manifest loads and validates; depth convention test resolved. Then render 18 scenes (13 train, 5 test) as a batch job.

Expected outcome: a few hundred context-target pairs per regime per scene, tens of thousands of pairs total.

## Phase 2: Encoder caching

Build: encoders.py wrappers for DINOv2 ViT-B/14 and VGGT, a caching script, and VGGT depth export for later phases.

Acceptance: features cached as fp16 [C, Hp, Wp] for every rendered frame of the pilot scene, with a throughput estimate for the full set; bilinear sampling at continuous locations tested against manual interpolation on a small tensor.

## Phase 3: Experiment Zero, the transportability test (no training)

Build: evaluate.py support for value-level transportability. Warp context features to the target camera with ground-truth depth; compare warped values against the target's own features at co-visible points; report cosine and L2 as margins over the No-Warp-Copy floor; run per-point and through the full splat-and-pool path; stratify by parallax bin and regime; run for DINOv2 and VGGT features.

Acceptance: one figure (margin versus parallax, per encoder, per path) and one table; a three-sentence written verdict naming the encoder chosen for all later phases.

Expected outcome in plain terms: at small viewpoint changes, warped features should match the target far better than unwarped copies do, and the advantage should shrink as viewpoint change grows. VGGT features are expected to decay slower than DINOv2 because they were trained to match content across views. If no encoder holds a useful margin anywhere, stop: the paper's headline becomes "frozen features do not behave like scene properties," and the plan gets rewritten around that result.

## Phase 4: Geometry-estimation cost (no training)

Build: Transport-Only with VGGT estimated depth, evaluated identically to Phase 3, side by side with Oracle-Transport.

Acceptance: the ladder's first two rungs plotted together; an error-localization visualization showing where estimated-depth transport loses accuracy.

Expected outcome: a modest overall drop from Oracle-Transport, concentrated at depth edges and low-texture surfaces. A large drop is still a valid result; it means depth estimation is the tax paid for structure, and the paper says so with a number.

## Phase 5: Predict-with-Depth versus Transport-Only (first training run)

Build: predictors.py (a small transformer or CNN trunk over cached features, relative pose encoded as input tokens or channels, depth channels included), train.py, and evaluation on the five held-out scenes, co-visible regions only, stratified by parallax and regime.

Baseline discipline: match parameter count and training budget across the two learned variants; train against the frozen target features of the encoder chosen in Phase 3; fixed seeds; no tuning on test scenes.

Acceptance: training converges with a stable validation curve; the stratified comparison figure exists.

Expected outcome: roughly a tie with Transport-Only at small parallax and in-place rotation, with Transport-Only pulling ahead as parallax grows, because large parallax is exactly where depth-dependent re-mapping is required. If the learned predictor keeps pace everywhere, that is the scale-beats-bias side winning at small scale, and it is equally reportable.

## Phase 6: Predict-Everything (second training run)

Build: the identical trunk with depth withheld, same budget, same evaluation.

Expected outcome: clearly worse than Predict-with-Depth at medium and large parallax, since it must infer geometry from appearance before applying it. A tie means depth input is redundant at this scale, which is itself worth reporting.

## Phase 7: Disocclusion slice (no new training)

Build: evaluation of both trained predictors on regions the context view never saw, against the Mean-Feature and No-Warp-Copy floors. Transport-Only is reported as coverage only in this slice, never scored on holes.

Expected outcome: both predictors land close together and well below their co-visible performance, because nothing observed can help there and both are guessing from priors. The informative number is the height above the Mean-Feature floor. Differences shrinking here after being large in co-visible regions is the cleanest demonstration that geometry's contribution lives entirely in the transportable part of the problem.

## Phase 8: Figures and numbers

Build: figures.py producing the three paper figures from outputs/eval/*.parquet alone: the error ladder, margin versus parallax with all variants and floors, and the co-visible versus disoccluded contrast. Export a single results table for the paper.

Acceptance: deleting all figure files and rerunning figures.py reproduces them byte-identically apart from timestamps.
