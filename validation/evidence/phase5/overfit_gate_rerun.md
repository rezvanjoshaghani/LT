# The amended overfit gate passed

Recorded 2026-10-15. The nine training tasks were submitted after this pass,
as SLURM array 3362608. No evaluation has run, so no Phase 5 test result
exists.

## The run

The chain restarted at commit E,
`a7f3a64eb145104cbbb7f2f403347b0894bea3ca`. That commit carries the amendment
recorded in `overfit_gate_amendment.md`.

The integration gate ran first, as job 3362571, in an `srun` allocation on the
gpu-l40 partition. All 17 steps passed. It printed these identities:

- config digest
  `8d731c5e1dabe8764c40f3599037de530ae39de9f2537c1675e068f3c909b1f8`
- fold digest
  `25f0c03f72d58cc8e3ff2d8d4123241f6459d4ed50d3c303c5dc108bb0e19865`
- measurement digest `27244e6481d521159e513f2ea8799482`

The config digest is the one the gate carried before the amendment. Its
receipt replaced the previous one, which was kept as
`integration_gate.superseded.4.json`.

The amended overfit gate ran next, as job 3362599. `./scripts/run_phase5.sh
overfit` submitted it only after verifying the new gate receipt against this
commit and configuration.

## The result

The job's output, verbatim:

```
clean worktree at commit a7f3a64eb145104cbbb7f2f403347b0894bea3ca
TINY-SUBSET OVERFIT GATE: PASS
  reached centered cosine 0.9800 with dropout off against 0.98 in 1661 steps
  best training-batch value with dropout on 0.9720, reported only
  rotation     apartment_2 apartment_2_vp00_rotation_000 -> apartment_2_vp00_rotation_006
  translation  frl_apartment_1 frl_apartment_1_vp00_translation_000 -> frl_apartment_1_vp00_translation_001
  orbit        frl_apartment_2 frl_apartment_2_vp00_orbit_000 -> frl_apartment_2_vp00_orbit_002
  rotation     frl_apartment_5 frl_apartment_5_vp00_rotation_000 -> frl_apartment_5_vp00_rotation_006
  translation  office_0 office_0_vp00_translation_000 -> office_0_vp00_translation_001
  orbit        office_1 office_1_vp00_orbit_000 -> office_1_vp00_orbit_001
  rotation     office_4 office_4_vp00_rotation_000 -> office_4_vp00_rotation_004
  translation  room_1 room_1_vp00_translation_000 -> room_1_vp00_translation_001
receipt written to outputs/phase5_rung2/evidence/tiny_overfit.json
```

The receipt holds the unrounded values. Writing it moved the receipt it
replaced aside to a numbered `tiny_overfit.superseded.N.json` sibling, as
`lot.phase5_check.write_once` does for every receipt. The failed receipt of
job 3362298 is therefore kept.

## Against what was written before the rerun

- `overfit_gate_amendment.md` was committed at E, before this run. It expected
  a pass near step 1700. The diagnostic replay read 0.9766 with dropout off at
  step 1500 and 0.9802 at step 1700. The gate crossed 0.98 at step 1661,
  between those readings.
- The gate stops at the first step whose score reaches the threshold. Its
  reported value therefore sits at the threshold by construction. It does not
  measure how far the fit would have gone. In the replay, the dropout-off
  score kept rising, to 0.9832 at step 2000.
- The best dropout-on value through step 1661 was 0.9720. The replay read
  0.9721 with dropout on at step 1700.
- Both numbers are in line with the replay. Nothing here suggests that
  scoring after every step moved the training trajectory.

The result departs from nothing the amendment stated. No anomaly is recorded.

## What follows

The two receipts license training at E. The nine (fold, seed) tasks were
submitted as array 3362608, with a time limit of two days per task.
