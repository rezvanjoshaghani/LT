# PROTOCOL.md

Status: frozen as of the pre-registration commit. This document is the referee for Phases 3 and 4. The git commit of this file is the pre-registration record. After freezing, changes are recorded only as dated entries in the companion document AMENDMENTS.md, with a rationale; this file is never edited. Where implementation and this document disagree, the disagreement is a validation finding, not a choice.

All numeric constants this protocol references (bin edges, the co-visibility depth tolerance, sample-support cutoffs, epsilon_margin, gate tolerances, boundary dilation radius, texture thresholds, the depth-convention decision threshold) live in configs/analysis.yaml at the pre-registration commit. That file is part of this protocol, and changing any value in it is an amendment.

Method names are used exactly as defined in CLAUDE.md: Predict-Everything, Predict-with-Depth, Transport-Only, Oracle-Transport, Transport-then-Predict, Transport-then-Refine, and the floors No-Warp-Copy and Mean-Feature. Letter labels are never used.

## Phase 3. Rung 0: Oracle-Transport and representation transportability

Goal: measure the maximum feature transportability available from the representation itself when geometry is exact. This phase answers: if the correct physical correspondence between two views is known, does the frozen encoder assign sufficiently similar feature values to the same physical surface? It does not test depth estimation or learned viewpoint prediction.

### 3.1 Frozen encoders

Primary encoder: DINOv2 ViT-B/14, locked for all downstream rungs. VGGT final-aggregator features remain a diagnostic encoder for Experiment Zero only.

All encoder features are frozen, computed independently per image with identical preprocessing and resolution, with no joint context-target encoding and no target leakage. The VGGT scope statement for the paper is: independently encoded, single-frame, final-aggregator VGGT features, with sequence-axis and batch-axis cross-image mixing both ruled out experimentally. The single-frame shape test and the batch-equality test remain in the suite permanently.

### 3.2 Evaluation rows store continuous geometry

Rows are long in variant and path and wide in metric. Each scored record carries at least: scene_id, context_frame_id, target_frame_id, camera_program, variant, path (per_point or splat_pool), rotation_deg, parallax, the count of contributing samples, and named metric columns for raw cosine and centered cosine with their l2 companions. This wide-metric form is the implementation's schema, adopted here before the freeze commit because the difference from a long metric-name-and-value layout is bookkeeping only. Centered Mean-Feature's structural not-applicable is represented as a nonfinite value in the centered columns of Mean-Feature rows, and that is the single permitted representation; no other row may carry a nonfinite score.

rotation_deg is the geodesic angle of the relative rotation, arccos of clip((trace(R_target_from_context) - 1) / 2, -1, 1), stored in degrees. The clamp is mandatory.

parallax at pair level is the median of per-point baseline over ground-truth depth across the pair's co-visible point set. This statistic is the binning variable; per-point parallax is not separately stratified in this protocol.

Bin labels never appear in rows. Bin edges live only in a committed analysis config applied by the figures code.

Pairing identity: every physical correspondence carries a deterministic sample_id, derived from scene, context frame, target frame, and the target-side sample coordinates. All intersections this protocol references, paired differences, surviving sets, matched ceilings, and common-valid gates, operate on sample_id, never on camera-pair membership or equal counts. Per-point records carry sample_id directly; where storage is pair-aggregated, the exact contributing sample_id set or its validity bitmask is persisted per record, and paired aggregates are recomputed from mask intersections. The Random-Patch hash of 3.6 draws on sample_id, which completes its definition.

### 3.3 Camera regimes are separate experimental controls

In-place rotation: translation is exactly zero by construction and is asserted from the manifest. This regime isolates representation change under orientation without parallax and is the sole source of the primary rotation-angle analysis.

Translation: little or no rotation; the sole source of the primary parallax analysis.

Orbit: rotation and parallax vary together. Orbit is an interaction regime, analyzed only in the joint rotation-by-parallax view. Orbit pairs never appear as points on the primary rotation curve or the primary parallax curve.

### 3.4 Binning, support, and uncertainty, frozen before Phase 4

Rotation bins: fixed equal-width 10-degree bins from 0 to 50 with a retained 50-plus overflow bin. Quantile bins are not used. If Phase 3 sample counts alone show insufficient support, edges are widened once, using counts only, never outcome values, and then locked. The final config is committed before any Phase 4 result exists.

Parallax bins: the adopted edges (0.025, 0.05, 0.1, 0.2, 0.4, overflow) live in the same config, plus the zero bin for exact-zero-parallax pairs. For translation-program pairs, the open interval from 0 to 0.025 is asserted empty by the program's design floor, and the assertion is enforced rather than silently absorbed by a bin; orbit pairs may legitimately fall in that interval, and the (0, 0.025) bin exists for them in the joint analysis. Bin intervals are closed on the right: a value equal to an edge belongs to the lower bin, matching the implementation and frozen here.

Support: every bin reports n_scenes, n_camera_pairs, and n_feature_comparisons. The support decision depends primarily on independent camera pairs and scene coverage, not raw comparison counts. Bins below the support threshold in the normative analysis config remain plotted, are greyed, show their n, and are never used for headline claims.

Uncertainty: bootstrap resampling at the scene level as the primary interval, camera-pair level as the secondary. Never bootstrap individual points or patches.

### 3.5 Experiment Zero procedure

For each valid co-visible physical point: extract the frozen context feature, lift with ground-truth depth and cameras, reproject into the target view, and compare the transported value against the frozen target feature at the same physical correspondence. Both the direct per-correspondence path and the full splat-and-pool pipeline path are run.

### 3.6 Nulls, frozen definitions

The three location controls, Neighbor-Patch, No-Warp-Copy, and Random-Patch, read from the same context feature map and differ only in read location, with image, encoder, and scene held fixed. Mean-Feature reads no location and is defined last.

Oracle-Transport reads the context feature at the correct correspondence location and transports it.

Neighbor-Patch reads one patch away from the correct correspondence location and transports it identically. It measures localization sharpness around the correct content. The offset direction is drawn hash-deterministically from the record's sample_id among the in-bounds axis-aligned unit offsets, which makes it reproducible per record, unbiased across directions, and defined at image borders. The audit established that the pre-correction run drew directions randomly per sample, so no single fixed direction exists to transcribe; this rule supersedes the earlier fixed-direction wording, adopted before the freeze commit. Where the offset patch falls outside the patch grid, the record is omitted and the omission is counted and documented.

No-Warp-Copy reads the context feature at the same image coordinate as the target location, without transport. It measures the position prior.

Random-Patch reads a random patch from the same context image, never across scenes or the dataset. Draws are deterministic: the patch index derives from a fixed hash of the record's sample_id as defined in 3.2, so the same record receives the same null regardless of batching or execution order. The completed run's realized draws are preserved in its evaluation records, so its aggregates remain reproducible independent of this rule.

Mean-Feature is a single global D-dimensional mean vector per encoder, averaged over all training-split frames and all spatial positions, so the floor and the centering statistics never adapt to the evaluated frames; it is compared against target features through each path. Position-conditioned mean maps are explicitly not used: subtracting a per-position mean would remove exactly the stationary positional component that the position-indexed VGGT finding measures. No-Warp-Copy remains the sole positional control.

### 3.7 Metrics, frozen definitions

Two metrics are computed for every record: raw cosine, and centered cosine, defined as cosine after subtracting the encoder's global mean vector (the same vector as the Mean-Feature floor) from both vectors, identically for source and target. Centered scoring is defined at the output level: the mean vector is subtracted from the final compared vectors, per-point samples or pooled outputs on scored cells, immediately before cosine. For the splat-and-pool path this coincides with centering before pooling only because pooled outputs on scored cells are normalized weighted means, with weights summing to one; that normalization is part of the transport contract, and a validation test asserts that the two centering orders agree.

One exception follows from these definitions: the Mean-Feature prediction is the mean vector itself, so its centered form is the zero vector and its centered cosine is undefined. Mean-Feature is therefore reported for raw cosine only and recorded as not applicable under centering, and No-Warp-Copy is the primary floor for centered results. No implementation may substitute an epsilon-regularized zero vector to manufacture a centered Mean-Feature score. Centering and margins are distinct operations and are never conflated: centering transforms the vectors before scoring; margins subtract a floor's score after scoring. The headline derived quantity is the margin of a variant over No-Warp-Copy within the same metric, path, and bin. All derived differences between variants, margins, one-patch-off costs, and taxes alike, are paired: computed on the intersection of the compared variants' valid records, so method differences never mix with selection differences. Within a path, the implementation scores all variants on the common valid record set, in practice the Neighbor-valid set, so ladder absolutes share one population and every difference is paired by construction, with the persisted mask as proof; per-variant omission counts are reported beside Figure A in place of differing n. Adopted before the freeze commit. No metric is ever presented without its floor. The motivating example is recorded: VGGT's raw Oracle-Transport cosine of 0.967 reads as excellent in isolation and becomes +0.003 beside its 0.964 floor; without the floor the conclusion reverses.

Verification note: the completed run shows the Mean-Feature floor differing by path (0.477 per-point versus 0.571 splat-and-pool, raw cosine), and it also reports finite centered Mean-Feature values (0.247 per-point, 0.428 splat-and-pool for DINOv2), which is impossible under this section's paired definitions. At least one of the implemented centering or Mean-Feature definitions therefore differs from this protocol. Validation must identify which, and the Phase 3 aggregates must be re-run under the frozen definitions before Phase 4.

### 3.8 Oracle ceiling

For each regime and bin, the Oracle-Transport score is the empirical representation ceiling for that cell, never 1.0. The observed decay of the DINOv2 centered ceiling from roughly 0.80 to 0.54 across parallax bins is itself a Phase 3 finding: frozen feature values become increasingly non-equivariant under larger viewpoint change even with exact geometry and correspondence. All later rungs are evaluated against these per-cell ceilings.

### 3.9 Operational transport check

The direct per-correspondence path and the splat-and-pool path must agree within the established tolerance of 0.003, computed on the cross-path intersection of valid records via sample_id, with the cross-path coverage difference reported beside it; agreement on differing populations would mix operator difference with selection difference. Adopted before the freeze commit. Interpretation, one sentence in the paper: the implemented transport path essentially reaches the representational ceiling, so downstream losses are not materially caused by the splat-and-pool operator.

### 3.10 Required figures and tables

Figure A: the null ladder per encoder (Random-Patch, Mean-Feature, No-Warp-Copy, Neighbor-Patch, Oracle-Transport), raw and centered, except Mean-Feature, which appears in raw only per 3.7.

Figure B: DINOv2 ceiling and No-Warp-Copy floor versus parallax, translation regime, with the margin between them visible. The floor curve is mandatory; a ceiling plotted alone reproduces the raw-cosine mistake this protocol exists to prevent.

Figure C: the same paired curves versus rotation angle, in-place rotation regime only.

Figure D: orbit joint analysis, rotation by parallax, as a heatmap or small multiples. Orbit is never collapsed into either marginal.

### 3.11 Phase 3 acceptance

Correctness criteria, all required: rotation and parallax stored continuously in rows; regime-aware stratification implemented as in 3.3; bin config and support rules committed before Phase 4; direct and splat-and-pool paths agree within tolerance; encoder leakage tests pass; DINOv2 ViT-B/14 locked.

Findings, expected but not required: Oracle-Transport beats No-Warp-Copy in every supported DINOv2 bin, and the margin stays positive as viewpoint difficulty increases. If either had failed, Phase 3 would still be an accepted measurement with a different headline, per the stop-on-anomaly rule.

## Phase 4. Rung 1: the estimated-geometry tax

Goal: replace only exact depth with VGGT estimated depth and quantify the additional transport loss caused by imperfect geometry. Question: given a representation whose oracle transportability is known, how much of that available signal is lost when exact geometry is replaced by estimated geometry?

### 4.1 Freeze everything except the depth source

Identical to Phase 3: DINOv2 features, context-target pairs, cameras, intrinsics, masks, rasterization, splat-and-pool, metrics, floors, bins, regimes, support rules. Ground-truth depth is replaced by VGGT estimated depth under the alignment ladder of 4.3, and nothing else changes.

Two additional frozen items. First, resampling: VGGT depth is produced at VGGT's processing resolution and is brought to render resolution by nearest-neighbor resampling. Bilinear resampling is forbidden because interpolation across depth discontinuities manufactures intermediate depths that exist on no surface, exactly at the boundaries this phase measures. Second, convention: before any alignment level runs, a deterministic depth-convention test executes. For the first rotation-program frame of every scene, compute the per-pixel ratio of resampled VGGT depth to ground-truth depth over valid pixels and regress it against the secant of each pixel's angle from the optical axis. A near-zero slope indicates planar z-depth; a positive slope tracking the secant indicates ray distance, converted by multiplying by the cosine of that angle. The decision threshold lives in the normative config, no frame or region is chosen by inspection, and the outcome is recorded in the run metadata. The test is a decision heuristic, not a proof: VGGT's own radial error patterns could imitate or mask the secant relationship. If the VGGT source or documentation independently establishes the depth head's convention, that establishes it, and the regression becomes a consistency check.

### 4.2 VGGT remains the depth estimator

The Phase 3 finding about VGGT's final-aggregator tokens concerns feature-value transportability and says nothing about depth quality. VGGT is not removed from the pipeline.

### 4.3 Depth-alignment ladder

All alignment levels are oracle calibration diagnostics: they consult ground-truth depth that no deployed system would have, and the ladder decomposes the estimated-geometry tax rather than simulating a deployable pipeline. The one inviolable rule is target exclusion: the evaluated pair's target-frame ground truth never contributes to the estimator applied to that pair, because the target side is not an input to the task. This is enforced by the per-record invariant test below, not by prose.

Level 0, no alignment: native predicted scale; the complete uncalibrated tax.

Level 1, leave-target-out scene oracle scale: for each pair, a single multiplicative scalar, the median of ground-truth depth over VGGT depth pooled over the valid pixels of all the scene's frames except the pair's target frame. This estimator consults frames that are neither the pair's context nor its target, so it is an oracle calibration diagnostic, not an input-available estimator, and it is named accordingly; what it cleanly measures is scene-level scale ambiguity. Without the target exclusion, a scene-wide scalar would leak every frame's ground truth into the pairs that target it. The scalars vary slightly across pairs within a scene, by the removal of one frame. A test asserts, for every evaluated record, that its target frame contributes zero pixels to the scale estimator used for that record. Separates global scene-scale ambiguity from structural error.

Level 2, context-image oracle scale: the same estimator computed from the pair's own context image only. It satisfies the target-exclusion invariant trivially. Removes per-frame scale drift, leaving primarily structural depth error.

Sensitivity, context-image affine (scale plus shift): per context image, ordinary least squares over the pre-alignment-valid pixels (finite and positive in both VGGT and ground-truth depth), unweighted, with no confidence weighting and no outlier clipping, minimizing the squared residual of s times VGGT depth plus b against ground-truth depth. The scale s is unconstrained; a fitted nonpositive s marks that image's affine row as failed, and the failure is reported rather than silently skipped. Validity is re-evaluated after applying the fit, per 4.4. This row is reported separately and never as the primary method. It shows how much additional error a calibration permitted to repair an offset removes.

The Level 1 and Level 2 medians are computed over the same pre-alignment-valid pixel definition.

### 4.4 Validity and coverage accounting

VGGT will produce invalid or unusable depth at some pixels where ground truth is valid, so the transported point set under estimated depth is a subset of the ground-truth co-visible set, and the missing points are not random. To avoid population drift between rungs, Phase 4 reports two numbers per bin and alignment level: transport quality on the successfully transported subset, and transported fraction of the ground-truth co-visible set. No single scalar summarizes both. The validity rule (finite, positive, any confidence threshold) is frozen before results exist, is identical across alignment levels, and is applied after alignment; this matters for the affine sensitivity row, where a fitted shift can produce nonpositive depths. A consequence worth enforcing: because positive scaling cannot change whether a depth is finite and positive, and the confidence threshold is scale independent, the surviving set is identical across the no-alignment, scene-oracle, and context-image-oracle levels, and any coverage difference among those three is an implementation error. Only the affine row may legitimately differ. Coverage relative to the ground-truth co-visible set is a property of the estimator, reported, never a bug by itself.

### 4.5 Mandatory correctness gate: pure rotation

Under zero translation, projection is scale-invariant in depth: lifting a pixel with any positive depth and reprojecting through the rotation lands on the same pixel, since the composition reduces to the depth-free homography K_target R K_context inverse, the single-K form when the two intrinsics coincide. Estimated depth therefore cannot move any correspondence in the in-place rotation regime. This is a pipeline invariant, not a hypothesis.

Consequently, before any Phase 4 result is interpreted, the gate runs on the intersection of the valid point sets of Oracle-Transport and the tested alignment level. The hard invariant: on that intersection, per-point correspondences must be identical and per-point scores must agree within 1e-5, because no projected coordinate can move under pure rotation. For splat-and-pool, one more effect exists: discrete collisions, where two samples land in the same output cell, are resolved by rotated depth, and wrong estimated depths can legitimately flip which sample wins even though neither location moved. Collisions are contests, not ties. The splat-and-pool gate therefore runs with a fixed collision ordering constructed as follows: intersect the valid samples of both methods, build Oracle-Transport's winner ordering using only those common-valid samples, and run both transport paths under that fixed ordering. Agreement within 1e-3 is then a true invariant and any failure is a pipeline bug. Constructing the ordering on the intersection matters because Oracle's winner at a cell can be a sample the estimated-depth run does not possess, and forcing an absent winner is meaningless; this construction keeps missingness and collision ordering fully separated. Ordinary unforced z-buffering is then run and its difference from the forced result is reported as the collision-ordering tax, not gated; under pure rotation the reprojection tax is exactly zero, so this difference isolates the collision-ordering tax purely, which no other regime can separate. The name is deliberate: under pure rotation these collisions are artifacts of discretizing a homography at finite patch resolution, not physical visibility changes. In translation and orbit, depth ordering does correspond to genuine occlusion, so ordering effects there are part of the ordinary tax rather than separately isolable. Coverage differences in this regime are reported separately. They do not violate the reprojection invariant, which applies only on the common-valid subset: rotation makes projection depth independent, not the estimator's validity independent of its own output. The 4.4 rule applies here as everywhere: coverage must be identical across the three multiplicative alignment levels, and a difference among them is an implementation error, while coverage relative to Oracle-Transport is a validity property of the estimator, reported, not gated. If the hard invariant or the forced-order gate fails, stop; do not interpret translation or orbit results; find the pipeline bug. Likely causes: mismatched intrinsics, resize or crop convention errors, coordinate-frame errors, validity-filter differences between depth sources, rasterization or visibility logic improperly depending on depth magnitude, or convention mismatch between ground-truth and VGGT depth.

### 4.6 Primary metrics

For each alignment level, the surviving set is the subset of the ground-truth co-visible points that remains valid under that level. All Phase 4 comparisons are subset matched: the matched ceiling is the Oracle-Transport score recomputed on that same surviving set, and the tax in each supported bin is the matched ceiling minus the estimated-depth score, within metric and path. The full-population Phase 3 ceiling appears in figures as the representation reference ceiling but is never subtracted from a score computed on a different population. The selection differential, the full-population ceiling minus the matched ceiling per bin, is reported so the difficulty of the dropped points is a measured quantity rather than an avoided confound. The retained transportable fraction is the estimated-depth margin over the floor divided by the matched-ceiling margin over the floor, both on the surviving set, and it is suppressed where the matched margin is below epsilon_margin, a denominator-stability constant in the normative config, distinct from the 3.4 sample-support thresholds. Interpretation: of the signal actually available to transport in this cell, what fraction survives estimated geometry.

### 4.7 Regime predictions

In-place rotation: the reprojection tax is exactly zero at every alignment level by the 4.5 invariant; the collision-ordering tax is reported and expected to be small.

Translation: the main experiment. The tax grows as parallax grows, because parallax is where projection is depth-sensitive.

Orbit: analyzed jointly. After controlling for parallax, rotation should not add a depth-estimation tax. Rotation can still lower the Phase 3 representational ceiling; that is a different effect and the two are never conflated: Phase 3 non-equivariance may grow with rotation, while the Phase 4 tax tracks depth sensitivity.

### 4.8 Error localization, pre-registered

Depth boundaries: a fixed discontinuity mask from the gradient magnitude of ground-truth depth, with a fixed dilation, splitting results into depth-boundary and depth-interior. Expected: the boundary tax exceeds the interior tax.

Texture: a fixed RGB gradient or variance measure with thresholds set before inspecting Phase 4 scores, splitting low from high texture. Expected: larger estimated-depth error in weak texture.

Optional, only if supported: stratification by ground-truth range.

All masks and thresholds are defined before Phase 4 outcomes are viewed.

### 4.9 Masks remain ground truth

Evaluation categories are defined from ground-truth geometry only. Co-visible means the target surface point was visible in at least one context view; disoccluded means visible in none. Estimated depth may cause transport failure; it never redefines whether a point was physically observable.

### 4.10 Required tables and figures

Table 1: the alignment ladder overall (Oracle-Transport, then VGGT at no alignment, scene scale, image scale, affine sensitivity), each with matched-subset score, margin over floor, matched tax, selection differential, retained fraction, and transported fraction.

Figure 1: the tax versus parallax, translation regime, one curve per alignment level.

Figure 2: the pure-rotation identity check under forced collision order, all variants overlapping, with the unforced collision-ordering tax shown beside it. Small, but included as the correctness control.

Figure 3: orbit interaction, rotation by parallax, plotting the tax.

Figure 4: error localization, depth interior versus boundary, plus texture if supported.

### 4.11 Phase 4 acceptance

Correctness, all required: the pure-rotation gate passes at every alignment level; masks, floors, bins, pairs, and transport path identical to Phase 3; no target-frame ground truth used in alignment; validity rule frozen and uniform; unsupported bins greyed per the pre-registered rule.

Interpretability, all required: alignment levels form the scale-versus-structure decomposition; results reported against Phase 3 per-cell ceilings; quality and coverage reported as the 4.4 pair, with the tax subset matched per 4.6.

Finding, expected but not required: a modest additional loss relative to Oracle-Transport, concentrated where parallax makes projection depth-sensitive and particularly at geometric discontinuities. A large tax everywhere is still a valid rung 1 result. Phase 4 is never tuned until it matches the expectation.

## What rungs 0 and 1 establish together

Rung 0, the representation tax: even with perfect geometry, frozen feature values are not perfectly persistent across views, and the tax grows with viewpoint difficulty. Rung 1, the geometry-estimation tax: after accounting for the representation ceiling, replacing exact geometry with estimated geometry introduces an independently measurable additional loss. Phase 5 then asks the central question on clean ground: given the same geometric information, what extra cost or benefit comes from making a neural predictor learn the viewpoint transformation instead of computing it? That is where Predict-with-Depth versus Transport-Only becomes meaningful; before rungs 0 and 1, those errors would have been mixed together.

## Optional, non-blocking extension

Cache VGGT intermediate aggregator layers (4, 11, 17, 23) and repeat Experiment Zero per layer. If transportability decreases monotonically with aggregator depth, the mechanistic claim upgrades to: geometry-grounded aggregation progressively converts surface-attached visual features into coordinate-indexed, view-covariant ones. This extension never delays Phase 4.

## Amendments

Amendments are recorded in the companion document AMENDMENTS.md, which is normative and takes precedence over this document wherever the two differ. This file is never edited after the freeze commit.