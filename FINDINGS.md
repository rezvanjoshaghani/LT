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
- Manifests written before this review can contain bare Infinity and NaN
  tokens, from probe views that came out ambiguous. Python reads them, strict
  JSON readers do not. New manifests write null instead. Check the batch with
  `python -c "import json,sys;[json.loads(open(p).read(),parse_constant=lambda c:sys.exit(p)) for p in sys.argv[1:]]" data/replica_renders/*/manifest.json`.
- transport allocates a full feature vector per splatted pixel, which is
  824 MB per buffer at 518 px with 768 channels. Phase 3 needs the
  weight-matrix form before it runs over tens of thousands of pairs.
- The per-pair pixel pipeline runs in float64 because manifest intrinsics and
  poses load as float64. Measured cost is 40 percent on CPU and far more on a
  consumer GPU.

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
- The measured rate on an L40 is 19.69 frames per second, against 8.9 on the
  local RTX 2080 Super, so the full 5136 frame set is about 4.3 minutes per
  encoder rather than 10. The host side still sets the floor: the local model
  alone ran at 21 frames per second, and the L40 end to end lands just under
  that, so PNG decode and transfer, not the GPU, now bound the job.
