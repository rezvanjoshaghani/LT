# Findings

> Frozen at commit d4ed1017bd2daca2871da28900b5b4a6a7ff92b6. The four
> normative artifacts and their sha256 values are recorded in FREEZE.md.
> Everything before that commit is protocol formation and repair;
> everything after it is execution against frozen rules.

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
evaluation, and retained prospectively because it is about 4.2 percent of the
smallest effect interpreted in the study at calibration time, the 0.072
one-patch localization cost. Recomputed against the corrected margins, the
ratio is 8.5 percent of the smallest supported one-patch cost, 0.035, which
stays under the 10 percent amendment flag below. The tolerance gates the statistic it was calibrated on, the signed
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

## Phase 4 methodology: the depth-convention diagnostic is confounded (2026-08-29)

PROTOCOL 4.1's deterministic secant regression produced scene- and
view-dependent classifications within fixed-checkpoint rotation sequences. In
office_1 the verdict runs from ray_distance to planar_z across the eight
in-place-rotation frames of one viewpoint; in room_0 it runs the other way
across the same kind of sequence; in hotel_0 the regression statistic moves
strongly with the fraction and spatial distribution of valid ground-truth
pixels, from -1.07 at 48 percent valid to -0.14 at 98 percent.

A depth convention is a property of the network's output semantics under a
fixed checkpoint. It cannot change because the camera yaw changed. The
supported conclusion is therefore about the diagnostic and not about VGGT:
the secant procedure exhibits strong scene- and view-dependent
field-angle-correlated structure, is confounded by the evaluated residual and
ground-truth-validity population, and cannot reliably identify the depth
convention on these data. PROTOCOL 4.1 anticipated exactly this when it
called the test a decision heuristic rather than a proof.

The authoritative VGGT source establishes planar camera-z semantics, so the
run uses planar-z globally and retains the secant outputs as diagnostic
evidence only. The decision threshold was not changed; changing it until the
flags disappeared would have converted a documented limitation into an
invisible one. Evidence: outputs/phase4_rung1/evidence/secant_diagnostic.json,
source_authority.json, and convention_record.json.

What is deliberately not claimed here is that VGGT has a strong intrinsic
radial depth error. This diagnostic does not isolate that. Candidate
contributors include VGGT depth error, spatially nonuniform ground-truth
missingness, scene geometry, the depth and range distribution, heteroscedastic
error, and their interaction with field angle. Where estimated-geometry error
actually localizes is what the preregistered Phase 4 structural analyses of
4.8 are for. Any later angle or radial analysis is exploratory and non-gating
unless it is frozen elsewhere first.

A conformance defect surfaced with it and is recorded in AMENDMENTS.md as A6:
the implementation selected the conversion per scene, so one checkpoint could
have carried two depth semantics inside one table. No Phase 4 scientific
output had been produced when the stop fired, so nothing required discarding.

## Phase 4 rung 1: the estimated-geometry tax (2026-08-30)

Run: 18 scenes, 16,884 camera pairs, 2,221,455 rows, 266,808 paired
records, evaluated on the inherited Phase 3 population and reconciled row
by row against it. The 4.5 gates passed on all 18 scenes under Amendment
A7 with the forced identity gap exactly zero at every alignment level.
Numbers are centered cosine; the tables carry raw as well, and both
evaluation paths. Evidence: outputs/phase4_rung1/tables/, and
scripts/phase4_acceptance_check.py re-verifies the acceptance conditions
from the shipped artifacts.

Phase 4 accepted 2026-08-30. PLAN.md asks for the ladder's first two
rungs plotted together and an error-localization visualization; both
exist and regenerate from the tables alone. Addendum E closed with it:
validator 2.3 reproduced apartment_0 row by row (930 pairs, zero mask and
count mismatches, worst residual 5.1e-07 against 1e-4), and the PROTOCOL
3.1 real-weight encoder tests passed on GPU.

A tax is reported here against the transportable signal it is a tax on,
not as a bare score difference. The margin available in a cell is the
matched Oracle ceiling minus the matched No-Warp-Copy floor, and the
retained fraction is the share of that margin the estimated-depth method
keeps. An absolute tax of 0.03 means different things against a margin of
0.12 and a margin of 0.5, and the ladder's cells differ in exactly that.

### The pure-rotation control is the cleanest result in the phase

In-place rotation pays a depth tax of exactly 0.0000 at every alignment
level, on 4,108 camera pairs, with retained fraction 1.0000 and
transported fraction 1.0000, and it stays exactly zero in every rotation
bin from 0-10 through 50-plus degrees. The A7 forced identity gap is
exactly zero at all four levels. The splat path reads -0.0001 uniformly,
the unforced rasterization tax of A7.

This is not only an implementation check. It establishes the conceptual
decomposition the study rests on: depth enters through translational
parallax, rather than different depth numbers producing different feature
scores by some other route. It is the correctness statement to lead with.

### Native VGGT depth is bad, and almost all of it is scale

Scene-level oracle scale removes about 84 percent of the native tax under
translation (0.2690 to 0.0444 per-point, 0.2548 to 0.0405 splat) and
about 78 percent under orbit (0.2425 to 0.0500, 0.2431 to 0.0527). At no
alignment the retained fraction is negative, -1.24 for translation
per-point: transporting features with natively scaled VGGT depth scores
below the No-Warp-Copy floor, so it is worse than not transporting at
all. Reporting raw estimated depth without calibration would therefore
overstate the failure of estimated geometry by a large factor, and most
of what it measured would be scale ambiguity rather than structure.

### After calibration a real residual remains, and its size depends on which question is asked

Translation, after context-image scale: the per-point tax is 0.0305
[0.0259, 0.0354] against a margin of 0.1209, which retains 75 percent;
the splat-pool tax is 0.0113 [0.0080, 0.0148] against a margin of 0.1233,
which retains 91 percent. Under orbit the split is wider: per-point 0.0497
retaining 64 percent, splat-pool 0.0143 retaining 90 percent. Affine
preserves the split (0.0296 against 0.0087 under translation).

The two paths ask related but different questions. The per-point path
asks how wrong individual transported correspondences are. The splat path
asks how much of that error survives the rasterize-and-pool operator that
a system would actually run. The depth is identical in both; the operator
is more tolerant than the correspondence. That is a result about operator
robustness, and it should not be described as better depth or summarized
into one number for the phase.

### The parallax structure is the main scientific figure

Translation, per-point, after context-image scale, by parallax bin:
0.0039, 0.0060, 0.0166, 0.0641, 0.0747. The affine row is nearly
identical: 0.0028, 0.0049, 0.0150, 0.0643, 0.0729. There is a clear
transition between the 0.1-0.2 and 0.2-0.4 bins, 0.0166 to 0.0641.

So the pooled 0.0305 averages two regimes rather than describing one. At
low parallax the calibrated tax is nearly negligible; at high parallax it
is not, and calling the overall figure small would hide that. Read with
the rotation control, the causal structure is strong: zero tax where
projection does not depend on depth, and a tax that grows sharply where
it does.

### Affine adds almost nothing, so the residual is structural

Adding a fitted shift to the per-image scale moves the per-point tax from
0.0305 to 0.0296 under translation and from 0.0497 to 0.0488 under orbit.
On the splat path the relative change is larger but the absolute change
is about 0.0025. Once a per-image multiplicative scale is fixed, an
additive bias explains essentially none of the remaining correspondence
error: what is left is structural depth error, not a calibration the
ladder failed to apply.

### Depth boundaries carry more of the tax, after image-level calibration

Boundary minus interior is +0.0051 [+0.0020, +0.0081] under translation
and +0.0146 [+0.0107, +0.0186] under orbit at context-image scale, with
affine essentially the same. The one exception is translation at scene
scale, +0.0028 [-0.0009, +0.0065], whose interval includes zero, so the
claim is about image-level calibration and not about every level. This is
where depth error is expected to break correspondence: occlusion
boundaries, thin geometry, sharp transitions.

### Anomaly against PLAN.md: low-texture surfaces pay less, not more

PLAN.md expects the drop concentrated at depth edges and low-texture
surfaces. Low texture goes the other way, with intervals excluding zero:
low-texture minus high-texture is -0.0120 [-0.0143, -0.0094] under
translation and -0.0283 [-0.0330, -0.0241] under orbit at context-image
scale. High-texture regions carry more feature-transport tax.

Nothing was tuned in response, and the result is not evidence that VGGT
depth is worse in textured regions. What is measured is feature-transport
tax, not metric depth error. The reading that fits is

    transport tax = geometric error times local feature sensitivity,

because DINOv2 features on a blank wall barely change under a two-pixel
misregistration while features on a detailed edge change substantially.
If that is right, the contrast localizes where the metric can see error
rather than where depth estimation is worst, which is a different and
more interesting statement than the preregistered one. It remains a
hypothesis: testing it needs depth error itself, or reprojection
displacement, measured by texture class. That analysis is frozen nowhere
and would be exploratory and non-gating.

### The orbit expectation is not supported at this conditioning

PROTOCOL 4.7 predicts that after controlling for parallax, rotation adds
little further depth-estimation tax. The joint grid shows substantial
residual structure instead. In the 0.4-plus parallax column at
context-image scale the tax runs 0.0324, 0.0708, 0.0808, 0.0807, 0.0887
as rotation rises, and scene scale runs 0.0289, 0.0400, 0.0684, 0.0781,
0.0871.

The supported conclusion is about the conditioning, not about rotation:
pair-level median parallax does not fully explain the orbit geometry tax,
and the preregistered expectation of little residual rotation dependence
is not supported by this grid. It does not license the conclusion that
rotation causes additional depth tax. The conditioning variable is a
single scalar, median baseline over depth, and within a bin, especially
the open-ended top bin, pairs still differ in true parallax magnitude,
translation direction, depth distribution, visibility, and where the
geometry sits in frame. The sign of the rotation trend reverses between
adjacent parallax columns, which is what residual within-cell composition
looks like. No bin edge was moved after seeing this grid. A continuous
model in parallax and rotation would say more and would be exploratory.

### Selection and coverage do not explain the result

After calibration the selection differential is a few thousandths:
translation at image scale is -0.0021 per-point and -0.0002 splat, orbit
-0.0025 and -0.0002. The matched Oracle ceiling barely moves when
estimated-depth validity filters the population, so the tax is not an
artifact of estimated depth discarding the difficult points. This is what
the matched-ceiling machinery was built for and it worked.

Coverage rises with calibration rather than trading against quality: the
transported fraction goes from roughly 0.76-0.86 natively to roughly
0.93-0.995 after calibration. One distinction must be kept explicit,
because the numbers invite the wrong reading. The transported fraction is
the landing-dependent scored set, which Amendment A5 permits to move with
scale under nonzero translation, since scaling depth moves where a point
lands. The 4.4 invariant is asserted on the source transport-valid set,
finite and positive depth, which positive scaling provably cannot change;
the evaluation layer asserted set equality per pair at run time and the
acceptance check re-verifies the persisted counts. Reading 0.8064 to
0.9660 to 0.9677 as a Step 10 failure would be reading the wrong set.

### Accounting, stated precisely

The affine fit is computed once per context image, before any evaluation
path, so its failures are path independent; the run records carry that
count as affine_failed_pairs. The ladder's per-scope column is a
different quantity, pairs the scope scored at some level and not at
affine, which is path dependent because a level can have no scored cells
on one path; it reads 236 on the per-point path and 215 on the splat
path. An earlier version of this file called those fit failures, which
was wrong.

The near-zero disclosure flags 225 cells, 130 not robust, 92 small and
sign-consistent under both paths, 3 path-sensitive in magnitude. That
total should not be quoted as a summary: it mixes the pure-rotation
zeros, which the protocol predicts as an invariant, with genuinely
ambiguous small effects such as low-parallax calibrated tax. The
acceptance check decomposes it by regime, level, and quantity, and that
decomposition is what belongs in any write-up.

## Phase 3: Experiment Zero, corrected verdict (Stream D, 2026-08-27)

Eighteen scenes, two frozen encoders, 33,772 comparisons, both evaluation
paths. PROTOCOL 3.9 passed on the statistic and at the tolerance it was
frozen with: signed aggregate +0.000115 raw and +0.000175 centered against
0.003. The path-agreement ledger passed with an empty stop list, attributing
the whole recorded per-path discrepancy to the operator difference on the
common cells. Every quantity below is reported under both paths, and no claim
rests on a difference the two paths do not share.

### DINOv2 transports, and the agreement is localized

Transporting DINOv2 patch features with ground-truth geometry beats copying
them unwarped in every supported cell of both primary analyses, under both
metrics and both evaluation paths. In-place rotation, raw cosine: +0.2356
per-point and +0.2352 splat-and-pool between 0 and 10 degrees, falling to
+0.1596 and +0.1587 beyond 50. Translation, raw cosine: +0.0568 and +0.0569
in the 0.025 to 0.05 parallax bin, rising to +0.1264 and +0.1227 above 0.4.
The centered metric moves every value up. It changes no ordering on the
per-point path. On the splat path one adjacent near-tie flips, the 0.1-0.2 and
0.4+ translation bins, whose raw margins differ by 0.001.

Landing one patch away from the correct location costs between 0.036 and
0.137 depending on the cell, again under both paths. The agreement is
therefore a property of the surface a feature sits on rather than a general
similarity between any two features of the same room, and it is that property
under either operator.

The two paths differ by at most 0.013 on any DINOv2 quantity in the table, and
by less than 0.002 on most. Where a ratio is meaningful, which is everywhere
for this encoder because every estimate is far above the operator gate, the
path difference is under 10 percent of the effect in every supported cell and
under 2 percent in the large majority.

No DINOv2 cell falls in the near-zero band. This is stated because the check
was preregistered to look: 27 of 232 supported cells are flagged, and all 27
belong to VGGT under raw cosine.

### VGGT does not, and past 20 degrees of rotation it is worse than not transporting

VGGT's last-layer aggregator tokens behave in the opposite way, and the
rotation series is monotone. Centered cosine, Oracle margin over No-Warp-Copy,
per-point and splat-and-pool: +0.0572 and +0.0568 between 0 and 10 degrees,
+0.0184 and +0.0178 between 10 and 20, then -0.0556 and -0.0566 between 20 and
30, -0.1428 and -0.1436 between 30 and 40, -0.2306 and -0.2307 between 40 and
50, and -0.3333 and -0.3337 beyond 50. Raw cosine traces the same path at a
tenth the magnitude, from +0.0097 to -0.0519.

Both evaluation paths preserve the negative margin at every rotation bin from
20 degrees upward, in both metrics, and agree there to within 0.001.
Geometrically transporting late VGGT features to their correct location is
worse than retaining their original image-coordinate placement. This is not a
statement about depth or about the transport implementation, which the ledger
accounts for exactly; it is a statement about what those tokens encode.

### Effects at the scale of the operator gate

Twenty-seven supported cells carry an estimate no larger than PROTOCOL 3.9's
0.003 tolerance on at least one path. All are VGGT, all under raw cosine, and
each is reported below with both path estimates and their difference. A
tolerance is a bound on operator disagreement, not a certificate of an effect
smaller than itself, so nothing here is certified by the gate; what is claimed
is restricted to what both paths show.

Twenty-two of the 27 have both paths inside the band, one sign, and both
intervals clear of zero. Eighteen are localization gaps, for which the reading
is that Oracle transport adds only a small, sign-consistent improvement over
Neighbor-Patch: for instance VGGT orbit at 20 to 30 degrees and parallax above
0.4, +0.00180 per-point and +0.00176 splat-and-pool, difference +0.00004. Four
are Oracle margins, read as a small, sign-consistent positive transport margin
under both evaluation paths, quantitatively close to the No-Warp floor: VGGT
translation in the 0.2 to 0.4 parallax bin gives +0.00239 and +0.00256,
difference -0.00017.

One cell has exactly one path inside the band: VGGT orbit at 10 to 20 degrees
and parallax above 0.4, Oracle margin +0.00374 per-point and +0.00296
splat-and-pool, difference +0.00078. The reading is a positive transport
margin under both evaluation paths whose magnitude is path-sensitive. No
claim of smallness or of proximity to the floor is made for it.

Four cells fail the conditions, either because the two paths disagree in sign
or because an interval includes zero, and for these there is no robust
transport advantage; the measured effect is at the scale of evaluation-path
choice. They are VGGT orbit Oracle margins at 30 to 40 degrees with parallax
0.2 to 0.4 and above 0.4, and at zero rotation with parallax 0.2 to 0.4 and
above 0.4. The zero-rotation, 0.2 to 0.4 cell is the clearest of them: -0.00019
per-point against +0.00050 splat-and-pool, a sign that does not survive the
choice of operator.

### Materiality of the operator tolerance, resolved

The preregistered materiality check compared PROTOCOL 3.9's fixed 0.003
tolerance against the smallest effect the study interprets and found the two
comparable: VGGT's near-zero translation and orbit effects are of the same
order as the gate meant to bound operator disagreement. The tolerance was not
re-derived. A gate rescaled by the effects it exists to be independent of
would no longer be independent of them, and choosing its value after seeing
which side the data falls on is the practice this study forbids elsewhere.

The resolution is disclosure rather than recalibration. Those effects are not
certified by the tolerance. They are reported under both validated evaluation
paths, with both estimates and their difference shown, and the interpretation
is restricted to content common to the two. Aggregate path bias is +0.000115
raw and +0.000175 centered against the 0.003 tolerance. Relative to the
smallest claim-carrying effect in this section, 0.0013, that bias is roughly
9 to 13 percent; relative to the smallest effect interpreted outside the band,
0.0037, it is roughly 3 to 5 percent. Near-zero claims are therefore reported
under both paths rather than certified by the operator gate, and the four
cells where a claim would have turned on the choice of path are reported as
carrying no robust advantage.

Corrections, 2026-08-28, from the re-audit (VALIDATION_REPORT.md): the split
of the 22 both-in-band cells above previously read sixteen localization gaps
and six Oracle margins; the evidence table says eighteen and four, and the
text now matches it (finding F-1). The materiality paragraph previously
claimed the operator agreement was 3 to 5 percent of even the smallest
interpreted effect; that ratio holds only against the one path-sensitive cell
at 0.0037, and the text now states both referents (finding F-3). Two smaller
wording fixes from the same audit: the calibration note's "less than 4
percent" is 4.2 percent (finding F-4), and the centering claim now names the
one splat-path near-tie ordering flip (finding F-2). No number in the tables
or evidence changed; these are corrections to prose that misdescribed it.

Evidence: validation/evidence/path_margin_differences.parquet and its readable
summary; validation/evidence/path_agreement_ledger/ for the decomposition and
the mechanism contrasts.

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
