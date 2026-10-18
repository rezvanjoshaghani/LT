# The chain at acc86ba

Recorded 2026-10-17. The chain restarted at commit E,
`acc86ba178958053dcf5d56e9050962d86e52528`. That commit carries the amendment
recorded in `stability_rule_amendment.md`. No evaluation has run, so no Phase
5 test result exists.

## The integration gate

The gate ran as job 3384236, in an `srun` allocation on the gpu-l40
partition. All 17 steps passed. It printed these identities, each the same as
at `a7f3a64`:

- config digest
  `8d731c5e1dabe8764c40f3599037de530ae39de9f2537c1675e068f3c909b1f8`
- fold digest
  `25f0c03f72d58cc8e3ff2d8d4123241f6459d4ed50d3c303c5dc108bb0e19865`
- measurement digest `27244e6481d521159e513f2ea8799482`

Its receipt replaced the previous one, which was kept as
`integration_gate.superseded.5.json`.

## The overfit gate

The amended overfit gate ran as job 3385109. Its output, verbatim:

```
clean worktree at commit acc86ba178958053dcf5d56e9050962d86e52528
TINY-SUBSET OVERFIT GATE: PASS
  reached centered cosine 0.9801 with dropout off against 0.98 in 1672 steps
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

The run at `a7f3a64`, recorded in `overfit_gate_rerun.md`, drew the same eight
pairs. It crossed 0.98 at step 1661, and its best dropout-on value was also
0.9720. The code that trains and scores the gate, the subset, the seed, and the
schedule are the same at both commits. The 11-step difference is consistent
with nondeterministic GPU arithmetic, which the gate does not control. Both
crossings lie between the diagnostic replay's readings of 0.9766 at step 1500
and 0.9802 at step 1700.

## Training

The nine training tasks were submitted after this pass, as array 3387760,
with the template's limit of eight hours per task.
