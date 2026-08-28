# Findings

Running record of empirical findings, in phase order. Anomalies are results
in this project. Each entry names the evidence trail.

## Phase 1: Replica rendering (accepted 2026-08-20)

- Habitat-Sim's depth sensor returns planar z-depth, not euclidean ray
  distance. All 18 scenes probed `planar_z` with confident margins. No
  conversion was applied. Stored depth is planar z by measurement, not by
  assumption. Evidence: `metadata.depth_convention` in every scene
  manifest and `probes/classification.json` per scene.
- Several Replica scans sit a few degrees off gravity alignment (+y).
  A fronto-parallel constancy test for the depth convention failed on 9 of
  18 scenes because probe floors are slightly sloped planes. Classifying
  by robust residual around a fitted plane resolves all 18. Evidence:
  per-probe fitted spreads in the manifests; commit ef6229d.
- frl_apartment_0, frl_apartment_1, and frl_apartment_5 ship
  `habitat/mesh_semantic.navmesh` files inconsistent with `mesh.ply`:
  their navigable points stand outside the rendered shell (straight-down
  floor views were 0 to 23 percent valid). The renderer now verifies every
  navmesh by floor visibility and recomputes it from the scene mesh on
  mismatch. These three scenes record `metadata.navmesh: recomputed`; the
  other 15 kept their shipped navmesh. Evidence: commit 7068edf and the
  navmesh field across manifests.
- Batch totals: 18 of 18 manifests validate; 5136 frames = 107 viewpoints
  x 48 frames. One scene accepted 5 of 6 viewpoints under the depth
  quality filter, which the design permits.

## Phase 1 review, before starting Phase 2 (2026-08-21)

A review of the Phase 0 core found defects that the analytic scenes could not
reach. All disparities in the two-plane scene are whole pixels, so every
reprojected location lands on a pixel center. That is exactly where an
interpolated read and a nearest read agree, so no test could see the
difference. tests/scenes.py now also carries a sub-pixel two-plane scene whose
edges and disparities both fall between pixels.

Fixed, each with a regression test:

- The co-visibility referee read the context depth map by bilinear
  interpolation. A depth map is a z-buffer. Interpolating across an occlusion
  edge returns a depth that lies on no surface. Background points the context
  camera really sees were therefore labelled disoccluded in a band about one
  pixel wide along every occlusion edge, and points in the depth gap could be
  labelled co-visible. At the smallest parallax bin the true disocclusion
  strip is only about ten pixels wide, so this bucket was contaminated most
  where the study cares most. The referee now reads the z-buffer with nearest
  sampling.
- The frustum test used pixel-center bounds while transport splats over the
  full pixel extent, so the two disagreed on a half-pixel border band. Both
  now use the physical extent.
- Correspondence sampling in patch_center mode interpolated the target depth
  across the four pixels around a patch center without checking they share a
  surface. A center on a depth edge was lifted to a point in mid air, and its
  warp missed both surfaces by about half a patch at the rendered resolution.
  Those centers are now dropped.
- The neighbour null could hand torch.multinomial an all-zero row on images
  under three patches wide. Such candidates are now dropped before sampling.
- check_intrinsics accepted zero and negative focal lengths, which mirror or
  collapse an image axis silently. It now requires positive focal lengths.

Open, recorded rather than fixed:

- Translation and orbit frames carry no per-frame quality check. Only the base
  view of each viewpoint passes the depth filter, and the derived frames move
  up to 0.4 times the median depth away with physics disabled. Pilot QC shows
  the effect: room_0 vp02 orbit_000 spans 0.04 to 6.46 m, and every room_0
  vp05 frame sits under 1.7 m. Phase 3 pair selection should filter on
  per-frame depth statistics rather than trust every rendered frame.
  Closed before Phase 3, but not the way it was framed. See the next entry.

## Phase 3 preparation: the frame filter was measuring the wrong thing

Applying the base view's own standard to every frame rejected 1208 of 5136
frames, 23.5 percent. The per-regime rates are the finding:

    rotation     1124/1391 pass, 19.2 percent rejected
    translation  1479/1819 pass, 18.7 percent rejected
    orbit        1325/1926 pass, 31.2 percent rejected

Rotation frames cannot exhibit the hazard the filter was built for. An
in-place rotation keeps the camera at the base viewpoint's position, and that
position already passed the viewpoint filter, so the camera is not inside
geometry. Yet rotation was rejected at 19.2 percent, indistinguishable from
translation's 18.7. A filter that rejects a fifth of the frames it cannot
possibly be right about is not measuring what it claims to.

What it was actually rejecting is pitch sweeps. A pitch of plus or minus 15
degrees at 1.5 m eye height looks at floor or ceiling about a metre away, and
the median depth band of 1.5 to 8 m throws those out. They are ordinary views
of real geometry.

Gating on median depth would also have damaged the study. Parallax is baseline
over median scene depth, so median depth is the study's main independent
variable. Rejecting close views removes the large disparities where
depth-dependent re-mapping matters most, and rejecting distant views removes
the near-homography regime. The filter would have thinned both ends of the
error ladder and left the middle, which is exactly where Transport-Only and a
learned predictor are least distinguishable.

The gate now tests only whether a frame is a view of the scene at all: enough
valid depth that the camera is not pointed into unscanned space, and enough
clearance in the central crop that the lens is not buried in a surface. Median
depth is measured and recorded for stratification, never used to reject.

The sidecar stores measurements only, with no verdict written into it. A
stored verdict outlives the reasoning behind it, which is how this mistake
would have propagated silently into every later phase. usable_frame_ids
applies the current policy to the stored measurements, so a policy change
costs nothing and re-reads no depth files.

Under the corrected gate 5078 of 5136 frames are usable, 98.9 percent:

    rotation     1381/1391 usable,  10 rejected
    translation  1790/1819 usable,  29 rejected
    orbit        1907/1926 usable,  19 rejected

The ordering is now the one the physics predicts, which is the check that the
gate measures what it claims. Rotation never moves the camera and loses the
fewest frames. Translation moves it furthest, up to 0.4 times the median
depth, and loses the most. Orbit sits between. The ten rotation losses are
views into unscanned space, where too little of the depth map is valid, not
cameras inside geometry.

Scene scale, for Phase 3 stratification. Per-frame median depth has a p50 of
1.9 m across the 18 scenes, ranging from 1.5 to 3.0 m by scene, with per-scene
p05 near 1.0 m and p95 between 1.9 and 5.6 m. These are close interiors. Since
parallax is baseline over median depth, the translation program's 0.4 target
means baselines of roughly 0.7 m, which is large relative to the rooms. Pairs
at the top of the parallax range will be genuinely hard.
- Manifests written before this review can contain bare Infinity and NaN
  tokens, from probe views that came out ambiguous. Python reads them, strict
  JSON readers do not. New manifests write null instead. Check the batch with
  `python -c "import json,sys;[json.loads(open(p).read(),parse_constant=lambda c:sys.exit(p)) for p in sys.argv[1:]]" data/replica_renders/*/manifest.json`.
- transport allocates a full feature vector per splatted pixel, which is
  824 MB per buffer at 518 px with 768 channels. Phase 3 needs the
  weight-matrix form before it runs over tens of thousands of pairs.
  Closed before Phase 3. Every pixel of a source patch carries that patch's one
  feature vector, so the splat accumulates scalar weights from source patch to
  target patch and mixes the features once with a small matmul. The weight
  matrix is 7.5 MB at the 37 by 37 grids in use. Runtime is now nearly
  independent of feature width: 68 ms at 768 channels and 78 ms at 2048, where
  the old form would have needed about 6.6 GB per call at VGGT's width. A test
  compares the result against the direct per-pixel splat on a random depth map,
  where splats collide and tie in ways the analytic scenes never produce.
- The per-pair pixel pipeline runs in float64 because manifest intrinsics and
  poses load as float64. Measured cost is 40 percent on CPU and far more on a
  consumer GPU.

## Methods notes, corrected Phase 3 (Stream D)

The 0.003 path-agreement tolerance was calibrated from the pilot run's
aggregate discrepancy between direct per-point evaluation and splat-and-pool
evaluation, and retained prospectively because it is less than 4 percent of
the smallest effect interpreted in the study, the 0.072 one-patch localization
cost. The tolerance gates the statistic it was calibrated on, the signed
aggregate over pairs; per-pair dispersion between the two paths is reported
beside it and judged by the path-agreement ledger
(validation/evidence/path_agreement_ledger/), whose reconstruction and closure
tolerances are frozen in configs/analysis.yaml. The materiality ratio is
recomputed against the corrected margins when they exist and flagged for
amendment if it exceeds roughly 10 percent.

### Path agreement, attributed (ledger verdict PASS, 2026-08-27)

The ledger ran over all 33,772 comparisons of the corrected run, both metrics,
and its stop list is empty. Evidence, cut points, and the report are under
validation/evidence/path_agreement_ledger/.

The preflight established that the two paths score the same object. Every read
that is a lookup into the same cached array agrees bit for bit across the
paths: the targets, the No-Warp-Copy source, and the Random-Patch source. The
targets being bit identical means the operator difference needs no split into
prediction-side and target-side pooling, because there is no target-side
pooling to separate. The per-point unit is one sample per eligible target patch
centre and the splat unit is one pooled target cell, so the common set is in
one-to-one correspondence and both paths average it uniformly at one over n.

The decomposition is exact rather than approximate. The four-term identity
closes to 7e-16, which is float64 epsilon. The weighting gap T3 is exactly
zero, so there is no unfrozen aggregation rule to document. Reconstruction of
the recorded scores from their own stored inputs is accurate to 2e-7 on the
per-point path and 6e-7 on the splat path, five hundred times inside the frozen
1e-4 tolerance, so the recorded numbers are what the recomputation says they
are. The whole of the recorded discrepancy therefore falls on T2, the operator
difference on the common cells.

Signed bias at the aggregation level every reported number uses is negligible:
+0.000115 raw and +0.000175 centered, against a 0.003 tolerance. The two
operators do not disagree about the answer. What they carry is dispersion:
mean per-pair absolute difference 0.0030 raw and 0.0042 centered, median 0.0007
and 0.0014, so the typical pair agrees several times better than the tolerance
and a heavy tail lifts the mean. The dispersion falls as the common set grows,
which is the signature of per-cell noise averaging down rather than of a
systematic offset.

Dispersion tracks occlusion, not rotation. Pairs at zero rotation, which are
the translation programme at its full range of baselines, carry the largest
dispersion at 0.0039 raw. Pairs beyond 50 degrees, which are in-place rotation
with no baseline at all, carry the smallest at 0.0008. Rotation enters only
where the orbit programme ties it to baseline.

Two preregistered mechanism contrasts were run on the per-cell operator
difference, with scene-level bootstrap intervals, and both are reported here
whatever they showed.

The boundary contrast is supported for both encoders and both metrics, with
every interval excluding zero. Cells whose bilinear read taps context patches
whose median depths differ by more than the co-visibility tolerance show a
larger operator difference than cells whose read footprint is flat: 11 percent
of the level for DINOv2 raw, 12 percent centered, 41 percent for VGGT raw and
31 percent centered. The flag is loose by construction, since it fires on any
depth variation across the four tapped patches and 86 percent of cells trip it
in furnished rooms, so the statement it supports is that a read footprint
spanning depth variation disagrees more than a flat one, not that sharp
occlusion edges specifically do. The mechanism is nonetheless the expected one:
the per-point read is bilinear over patch vectors with no depth test, while the
pooled read composites only z-buffer survivors, so the two answer differently
exactly where the footprint straddles surfaces.

The norm contrast is absent for three of the four cases. DINOv2 raw, DINOv2
centered, and VGGT centered all have intervals spanning zero, and the study
claims no norm mechanism for them. It is present for VGGT raw alone, where the
bottom quartile of centered target norm shows an operator difference 54 percent
below the level, with a Spearman correlation of +0.55 between quartile index
and scene-level mean. That asymmetry is consistent with the encoder finding
already on record: VGGT's raw cosine is dominated by a shared direction, so
cells whose target sits near the global mean give both paths the same nearly
saturated value and leave nothing to disagree about, while cells with
distinctive content do not. The contrast is absent under centering, which
removes that shared direction, and absent for DINOv2, whose centered target
norms are tightly concentrated and offer little to stratify on.

The per-cell operator difference is an order of magnitude larger for DINOv2
than for VGGT on the raw metric, 0.0130 against 0.0012, in proportion to how
sharply each encoder's patch features differ from their neighbours. That is a
property of the representations, not of the transport implementation.

Nothing in this section is a Phase 3 result. The results section below remains
withdrawn until Stream D's figures are produced and audited.

## Phase 3: Experiment Zero, verdict (WITHDRAWN 2026-08-24)

> **These numbers are withdrawn and are not a result.** An independent audit
> (VALIDATION_REPORT.md, verdict FAIL, at commit c5e50f9) found four blockers
> and fifteen majors in the code that produced them, and three subsequent
> reviews of the repair found more. The definitions have since changed in ways
> that move every figure below: Neighbor-Patch draws a different direction,
> scoring is on each path's common valid set, sample identity is a full-width
> digest so every hash-derived null differs, the reported estimand is the
> unweighted pair mean rather than the comparison-weighted one, and the rotation
> angle is measured differently near zero. The corrected run, Stream D, has not
> been executed.
>
> The section is kept, unedited below this notice, because it is what the audit
> was against and because withdrawing a claim silently is worse than leaving it
> visible with its status attached. Nothing here may be cited, and the
> qualitative reading is not carried forward either: whether DINOv2 transports
> and VGGT does not is a question the corrected run reopens.

### Withdrawn text, as it stood at acceptance (2026-08-21)

DINOv2 ViT-B/14 is the encoder for all later phases. Its patch features behave
like properties of the surface they sit on: transporting them with ground-truth
geometry beats copying them unwarped by 0.152 cosine on centered features, and
landing one patch off the correct location costs 0.072, so the agreement is
sharply localized rather than a general similarity between any two features of
the same room. VGGT's last-layer aggregator tokens do the opposite: their
overall margin is -0.0001, and resolved by rotation angle it falls from +0.055
between 5 and 10 degrees to -0.279 beyond 40, where copying a feature from the
same image position beats moving it to the same surface by a wide margin, which
is what a position-indexed quantity looks like and not what a surface property
looks like.

The full numbers, pooled over everything, centered, per-point path:

    encoder        Random  Mean-Feature  No-Warp  Neighbor  Oracle  margin
    dinov2_vitb14  0.0852        0.2368   0.4769    0.5486  0.6285  +0.1516
    vggt_1b        0.1646        0.3193   0.7516    0.7316  0.7514  -0.0001

These are from the run stratified on both axes of viewpoint change. The earlier
parallax-only run gave +0.1395 and +0.0129 on a sample half the size weighted
differently; per parallax bin the two runs agree to better than 0.015
everywhere, so the estimates are not sampling artifacts. The pooled figures
moved because the two-axis strata draw far more large-viewpoint-change pairs,
which is also why VGGT's pooled margin fell to exactly zero.

Reading VGGT's row across is the whole story. Its floor sits at 0.787 while its
correct answer sits at 0.800, and its one-patch-off null sits between them at
0.778, below the floor. A feature whose value barely changes when you move a
patch, and barely improves when you move it to the right place, is not encoding
which surface is there. It behaves like slowly varying scene context indexed by
image position, which is what the aggregator is for. Warping it actively hurts
in the two regimes with the largest image displacement, pure rotation at -0.046
and the largest parallax bin at -0.013, because moving a position-indexed
quantity to a geometrically correct location destroys the positional agreement
that made the unwarped copy match.

Geometry's value is a hump, not a slope, and an earlier reading here was wrong.
Resolved by rotation angle, DINOv2 centered on the per-point path:

    angle    ceiling  floor   margin      VGGT ceiling  floor   margin
    5-10      0.7711  0.4893  +0.2818           0.8804  0.8252  +0.0553
    10-20     0.7025  0.3705  +0.3319           0.7570  0.7386  +0.0184
    20-40     0.5594  0.2798  +0.2796           0.5453  0.6373  -0.0920
    40+       0.4124  0.2228  +0.1896           0.2595  0.5384  -0.2788

The mechanism is in the two columns beside the margin. From 5-10 to 10-20 the
floor falls faster than the ceiling, 0.119 against 0.068, so the margin grows.
Past 20 degrees the ceiling falls faster, so it shrinks. Transport is worth most
where the unwarped copy has already collapsed and correct transport has not yet.

This corrects what was recorded after the first run, that transport's advantage
grows with viewpoint change and contradicts PLAN's expectation that it shrinks.
The parallax axis alone showed only the rising side of the hump and the first
hint of the turn, +0.1476 at 0.2-0.4 falling to +0.1428 at 0.4+. PLAN's
expectation is right beyond the peak and wrong before it. Only the finer
stratification made the shape visible, which is the argument for having done it
before Phase 4 rather than after.

The same table makes VGGT's mechanism unmistakable. Beyond 40 degrees, copying
a feature from the same image position scores 0.538 while moving it to the same
surface scores 0.260. Transport is 0.279 worse than doing nothing. That is not
a small negative to be explained away; it is a large monotonic effect, and the
pooled -0.0001 was the average of a strong trend through zero. A representation
whose values are better predicted by where they sit in the frame than by what
surface they are on is position indexed, and this is the measurement that says
so directly.

Why the inversion happens, and why it is the useful result. The August 18
hypothesis was that VGGT should transport best, having been trained to match
content across views. The opposite followed from what its final tokens are for.
They exist to predict pointmaps, which are 3D coordinates in a reference frame,
and a coordinate is a function of the camera. Training pressure toward
coordinates produces view-covariant values. Surface identity is precisely the
view-invariant part. So the strongest correspondence machine available produces
final-layer values that barely transport, which makes VGGT an existence proof
that correspondence rank and transportable metric are different properties of a
representation.

The raw-versus-centered contrast is the vindication of the floors. Raw cosine
reads VGGT's Oracle-Transport at 0.967. Reported without the No-Warp-Copy floor
beside it, that is a number anyone would publish as excellent transportability.
The floor is what turns it into a margin of 0.003, and the diagnostic is what
turns that into a saturated scale. A metric without its floor would have
inverted the paper's conclusion.

Scope of the claim, both confirmed against the harness rather than assumed:

- VGGT features come from single-frame forward passes. The wrapper builds a
  sequence of length one, so the aggregator's cross-frame global attention has
  nothing to mix. Had a context and its target ever been handed over as one
  sequence, the context tokens would have already seen the target and the
  measurement would be contaminated. Pinned by
  test_vggt_sees_one_frame_at_a_time, with a cluster-gated companion that
  checks the real model's batch axis does not couple images either.
- Random-Patch is drawn inside the context image, not across the dataset, and
  is read from the same context feature map as No-Warp-Copy and Neighbor-Patch.
  All three predictions differ only in where they read: same image, same
  encoder, same scene. That is what makes "a random place in the other view of
  this room" at 0.176 the right comparison against "the same place" at 0.787
  and "one patch off" at 0.778, and what supports reading VGGT's tokens as
  slowly varying across the image and stable across views at fixed position.
  Pinned by test_every_null_reads_the_context_map_and_differs_only_in_where.

This is a statement about the last aggregator layer, which PLAN's Phase 2 chose
because it is what VGGT's own heads consume. Earlier layers and the depth head
were not tested and could behave differently. In particular VGGT contains a
DINO encoder, which would presumably transport like DINO, so all paper text
must keep the claim scoped to the last aggregator layer. The stronger version
is available cheaply: cache a mid-aggregator layer for the same frames and add
a row. If transportability decays monotonically with aggregator depth, the
finding becomes that geometry supervision progressively converts
surface-attached values into position-indexed ones. Optional, not blocking.

Consequences for the rest of the ladder, recorded here so later phases are read
correctly:

- The Oracle-Transport numbers are the reference ceiling every later figure
  plots against, and the ceiling itself decays with parallax, from 0.80 to 0.54
  centered. Representation non-equivariance is real and grows with viewpoint
  change. That is rung 0's own finding, and Phase 5's expectations must be read
  against the decaying ceiling within each bin, never against 1.0.
- VGGT stays in the pipeline untouched as the Phase 4 depth estimator. This
  finding concerns its tokens and says nothing about its depth. Nothing here
  should be read as a reason to remove VGGT.
- The splat-and-pool path matches the per-point path to within 0.003
  everywhere, so the operational ceiling equals the representational one and
  the pipeline costs essentially nothing. One sentence in the paper, no more.

Centering did not change what DINOv2 says, which is the reason to trust it. The
margin moves from 0.123 to 0.140, the growth with parallax from 0.065-to-0.134
to 0.075-to-0.147, and the zero bin from 0.248 to 0.288. Every ordering and
every trend survives. For VGGT centering quadrupled the margin, from 0.003 to
0.013, and left it an order of magnitude short.

The two paths agree to 0.001 in the centered reading as well, 0.6642 per-point
against 0.6647 splat-and-pool. The splat, z-buffer, and pooling machinery costs
nothing measurable, so the later phases measure representations rather than the
implementation.

Closed before Phase 4. Rotation was one parallax stratum by construction, so
the zero bin pooled 7.5 to 60 degrees of yaw and its margin of 0.288 was an
average across that whole range. A viewpoint change has two components that are
not interchangeable: the baseline decides how much depth-dependent re-mapping
is needed, and the rotation decides how far a surface point travels across the
image. Every pair is now binned on both, and the stratum is scene, regime,
parallax bin, and rotation bin.

Binning on one axis alone collapses one whole regime into a single cell,
whichever axis is chosen: in-place rotation has no baseline, pure translation
has no rotation. Orbit is the case that shows why both are needed, since it
moves on both axes at once and the two are correlated through the orbit radius.

The effect on sampling, per scene at a cap of 40 per stratum: rotation goes
from 1 stratum and 40 pairs to 4 strata and 160, orbit from 4 strata and 160 to
13 strata and 478, translation is unchanged at 5 strata and 200. The run grows
from 7200 pairs to 15084 and from about 9 minutes to about 19. Phase 4 inherits
this stratification, which is why it landed before Phase 4 rather than after:
doing it later would have meant re-running both rungs.

## Phase 3: Experiment Zero, first run and what it exposed (2026-08-21)

The first run scored 7300 pairs across 18 scenes. DINOv2 produced an ordered,
interpretable ladder. VGGT produced margins near zero. The second result was a
fault in the metric, not in the encoder, and finding that is the main outcome
of the run.

Cosine on raw features cannot resolve VGGT. Measured on the caches, VGGT puts
0.9095 of a feature's norm in a single direction shared by every patch, against
DINOv2's 0.4226. Two vectors that each spend 91 percent of their length on one
axis are forced above cosine 0.83 whatever the remaining 9 percent says, which
is exactly the band every VGGT number fell in: Random-Patch 0.857,
Mean-Feature 0.921, No-Warp-Copy 0.964, Oracle-Transport 0.967. The margin of
0.003 measures the width of a saturated scale.

The representation is not empty. After subtracting the dataset mean, two random
patches of one VGGT frame agree at 0.226, against 0.088 for DINOv2 centered.
There is more local structure under VGGT's offset than under DINOv2's. The
constant was hiding it.

The results table now carries both readings for every row, raw and centered,
and the centering subtracts one global mean vector shared by every encoder,
method, and pair. It is deliberately not the position-dependent mean map:
positional structure is real information about how rooms are laid out, it is
what the Mean-Feature floor exists to measure, and removing it would delete a
floor rather than clean a metric.

Two cross-checks that the pipeline itself is sound. Experiment Zero's
Random-Patch null for DINOv2 scored 0.265 while the independent cache
diagnostic put two random patches of a frame at 0.2518; two separate code paths
agreeing to that tolerance is evidence the measurement is real. And the
per-point and splat-and-pool paths agreed to within 0.001 in every cell, so the
resampling, z-buffering, and pooling cost essentially nothing, and later phases
will be measuring representations rather than the splat.

Two expectations in PLAN did not survive, both worth reporting. PLAN expected
transport's advantage to shrink as viewpoint change grows. For DINOv2 it grows:
the margin runs from 0.065 at the smallest parallax to 0.134 at the largest,
because the unwarped copy collapses faster than the warped one does. Geometry
helps most exactly where it is needed. And the zero-parallax bin holds DINOv2's
lowest absolute cosine, 0.699, with its largest margin, 0.248. Parallax is
baseline over depth and is identically zero for every rotation pair, so that
bin pools 7.5 to 60 degrees of yaw, which sweeps a surface point across
two thirds of a 90 degree field of view. It is not a small viewpoint change; it
is a large image displacement with no baseline. Rotation needs strata by angle,
which the first run did not have.

## Phase 2: encoder caching (2026-08-21)

- DINOv2 ViT-B/14 returns patch tokens in row-major order, so reshaping
  [B, N, C] to [B, Hp, Wp, C] puts image rows on the first grid axis. This was
  measured, not assumed. A bright stripe drawn across the image rows peaks at
  patch row 17 of 37 with a flat column profile, and the same stripe drawn down
  the columns peaks at patch column 17 with a flat row profile. A transposed
  reshape would keep the shape and the channel count, pass every other check in
  the suite, and mirror every warp in Phase 3. Evidence: the gated test
  test_dinov2_grid_orientation_and_shape.
- Throughput at 518 px, batch 8, on an RTX 2080 Super: the model alone runs at
  21 frames per second, the whole cache path at 8.9. The difference is PNG
  decode and host to device transfer, not the model. Moving the uint8 to float
  conversion and the normalization onto the GPU took the end to end rate from
  7.0 to 8.9. The full 5136 frame set is therefore about 10 minutes per
  encoder, at 2.1 MB per frame, so roughly 11 GB per encoder.
- No high-norm outlier tokens appeared on synthetic probes. The largest token
  norm was within 1.1 times the median for both dinov2_vitb14 and
  dinov2_vitb14_reg. That does not settle the question on real Replica frames,
  where large flat surfaces are common, so the register variant is registered
  as an encoder and Phase 3 can compare the two directly.
- Pilot acceptance passed on Borah. room_0 cached 288 frames, 6 viewpoints of
  48, as fp16 [768, 37, 37], and validated against its manifest frame by frame.
  The 37 by 37 grid confirms the encoder saw the 518 px frames the manifest
  declares, which is the assumption every Phase 3 warp rests on.
- The DINOv2 batch is complete and accepted. All 18 scenes cached and
  validated: 5136 frames, every one fp16 [768, 37, 37], frame counts matching
  the manifests, including frl_apartment_2 at 240 for the 5 viewpoints Phase 1
  accepted there. Total wall time was about 3.5 minutes on one L40.
- Per-scene rates ranged from 18.6 to 57.9 frames per second, a threefold
  spread with no relation to scene content or frame count, while the aggregate
  was 25.9. The job is bound by reading and decoding PNGs from shared scratch,
  not by the GPU, so the spread tracks filesystem contention. This matters for
  Phase 3 planning: encoding is cheap and repeatable, and any pipeline that
  re-reads the rendered PNGs will see the same variance, whereas one that reads
  the cached features will not.
- VGGT is accepted on the same 18 scenes. Its aggregator returns 2048 channels
  on the same 37 by 37 grid, and the estimated depth exported for Phase 4 is
  [518, 518] per frame with a confidence map beside it. All 5136 frames
  validate. The run took 13.6 minutes at 6.30 frames per second.
- The two encoders confirm the bottleneck reading. VGGT held 6.32 to 6.45
  frames per second across every scene, a spread of two percent, where DINOv2
  varied threefold over the same PNGs on the same filesystem. VGGT is heavy
  enough to be compute bound, so filesystem variance hides behind the GPU;
  DINOv2 is light enough that the filesystem is the whole story. Only
  apartment_0 broke VGGT's uniformity, at 5.41, because it also paid for
  loading the 5 GB of weights.
- VGGT's wrapper originally ran the aggregator twice per batch, once for the
  patch tokens and once inside the model's own forward for depth, and measured
  2.02 frames per second. Taking the tokens with a forward hook instead
  removed the duplicate trunk pass and reached 6.30, a factor of 3.1. The
  aggregator therefore dominates the cost even more than the head; the fix is
  in the wrapper, not in any setting.
- Method choice worth naming in the writeup: the VGGT features are the last
  aggregator layer, which is the representation VGGT's own heads consume.
  Every layer is exposed, so this is a choice rather than a constraint, and
  Phase 3's verdict on which encoder to carry forward rests partly on it. The
  estimated depth is in VGGT's own scale, not meters, and Phase 4 must align
  it and say how.
