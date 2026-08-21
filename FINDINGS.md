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
