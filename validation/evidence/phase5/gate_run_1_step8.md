# First real run of the integration gate: stop at step 8

Recorded 2026-10-09. No Phase 5 model has trained and no Phase 5 result exists.

## The run

The gate ran on Borah, on an L40 node of partition `gpu-l40`, at commit
`6f9a3016ff4ae33367a9dfd57b56f7ffc53a9158`. That is Borah's merge of GitHub's
`b1aa1fc` with Borah's two unpushed Phase 4 evidence commits. The run's
identities matched the frozen ones: config digest
`8d731c5e1dabe8764c40f3599037de530ae39de9f2537c1675e068f3c909b1f8`, fold digest
`25f0c03f72d58cc8e3ff2d8d4123241f6459d4ed50d3c303c5dc108bb0e19865`, measurement
digest `27244e6481d521159e513f2ea8799482`, torch 2.5.1 with CUDA 12.1.

Steps 1 to 7 passed on real artifacts. Step 8 stopped:

    [FAIL] step 8: formulation support V_form on target cells
    classification implementation_bug
    message        landing cells fall outside the target grid: [1369]
    evidence:      n_cells: 1369

The gate's own record of the run is `outputs/phase5_rung2/evidence/
integration_gate.json` on Borah. A rerun moves it aside rather than deleting it.

## Diagnosis

Step 8 mapped every context patch to a target cell, including patches that
never landed. `lot.encoders.patch_cell_index` does not bound-check. A patch
that lands just past the right edge of the 37 by 37 grid gets column 37, which
wraps into the first cell of the next row and looks valid. Only the bottom
row's overflow leaves the grid, as cell 1369. That is why the stop named a
single cell.

The check was wrong, not the data. A landed sample cannot leave the grid,
because the landing rule keeps it inside the patch-center box.

No score was affected. Every scoring caller of `patch_cell_index` in
`lot.phase5_score` masks to the supported samples before a cell is read:
`formulation_support`, `formulation_cells`, `score_formulation`,
`region_masks`, `cross_path_cells`, and `score_cross_path`.

## A second instance of the same mistake, found before it fired

Step 10's pure-rotation check compared context-lift with the analytic
homography over every context patch. An unlanded ray can sit nearly parallel
to the target image plane, where its projection runs to tens of thousands of
pixels and float32 rounding alone exceeds the 1e-3 px tolerance. The tolerance
was set from float32 arithmetic at the 518 px frame size. On a 518 px frame
with a 90 degree field of view, the residual over every patch was:

| rotation | every patch (px) | landed samples (px) |
|---|---|---|
| 10 deg | 9.2e-5 | 6.1e-5 |
| 30 deg | 2.4e-4 | 9.2e-5 |
| 45 deg | 4.7e-2 | 6.1e-5 |
| 60 deg | 6.3e-2 | 9.2e-5 |
| 75 deg | 1.07 | 6.1e-5 |

The camera programs reach rotations past 45 degrees, so step 10 would have
stopped falsely whenever the gate's probe rotation pair was a large one.

## The fix

- Step 8 maps only supported samples to cells. It now also does what its
  specification asks and the first version did not: TL-Reference for the same
  pair is recomputed through Phase 4's code, reconciled with the accepted
  Phase 4 row, and intersected with the Context-Lift cells by
  `formulation_cells` and `score_formulation`, the functions evaluate runs.
  It reports the Context-Lift support, the TL-Reference cells, and their
  intersection. A target cell holding two TL-Reference samples stops it as
  ambiguous.
- Step 10's rotation check reads landed samples only. A NaN residual now fails
  it, where the first version's `>` comparison would have let one pass.
- Both checks moved into module-level functions,
  `lot.phase5_gate.formulation_support_evidence` and
  `lot.phase5_gate.rotation_gate_evidence`, so the suite drives them.

## Tests, written from the failure

- `tests/test_phase5_modes.py`: on the synthetic world, a supported pair whose
  unlanded patches map outside the grid, which is the Borah condition, passes
  step 8. Its report equals the formulation evaluate recorded for that pair.
  A tampered Phase 4 mask and a doubled TL-Reference cell each stop step 8.
- `tests/test_phase5_gate.py`: a 60 degree rotation, whose all-patch residual
  exceeds the tolerance, passes step 10 on its landed samples. A wrong
  rotation and a depth-dependent landing each still stop it.
- Both files also check that the gate's closures call the tested functions.

## What happens next

The gate is rerun at the commit that carries this fix. Its receipt binds that
commit, so nothing from this run carries over.
