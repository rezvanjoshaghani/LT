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
with the template's limit of eight hours per task. All nine completed with
exit code 0, on gpu111 and gpu117, in 58 to 64 minutes each. Every record
names commit `acc86ba` and device cuda. Retraining moved the checkpoints and
records trained at `a7f3a64` aside as numbered siblings, as
`lot.phase5_check.supersede` does.

sacct reports the peak resident memory in KiB. It was 15.5 to 15.8 GiB per
task, against the 48 GiB requested. Read the same way, the `a7f3a64` runs
peaked at 15.6 to 15.9 GiB. `stability_rule_amendment.md` gave those as 16.3
to 16.6 GB, which counts each KiB as 1,000 bytes.

Decision 5 was applied on Borah to the nine training records, with the frozen
functions of `lot.phase5_outcomes`, a tolerance of 0.003, and a window of 10
validations. The gap from the last score to the best is computed here from
the printed values.

| run | steps | best step | best | last | last below best | level gap | early stopping | as amended | as registered |
|---|---|---|---|---|---|---|---|---|---|
| fold0_seed0 | 20000 | 16000 | 0.66615 | 0.66385 | 0.00230 | 0.00005 | no | stable | stable |
| fold0_seed1 | 20000 | 17000 | 0.66887 | 0.66678 | 0.00209 | 0.00000 | no | stable | stable |
| fold0_seed2 | 20000 | 19000 | 0.66692 | 0.66519 | 0.00173 | 0.00000 | no | stable | stable |
| fold1_seed0 | 20000 | 19000 | 0.68260 | 0.68061 | 0.00199 | 0.00000 | no | stable | stable |
| fold1_seed1 | 19500 | 14500 | 0.68122 | 0.67558 | 0.00564 | 0.00185 | yes | stable | stable |
| fold1_seed2 | 20000 | 16000 | 0.68179 | 0.67846 | 0.00333 | 0.00000 | no | stable | not stable |
| fold2_seed0 | 20000 | 15500 | 0.68792 | 0.68604 | 0.00188 | 0.00069 | no | stable | stable |
| fold2_seed1 | 20000 | 16500 | 0.69013 | 0.68797 | 0.00216 | 0.00000 | no | stable | stable |
| fold2_seed2 | 20000 | 16000 | 0.69124 | 0.69044 | 0.00080 | 0.00000 | no | stable | stable |

All nine runs are stable under decision 5 as amended, so acceptance condition
19 can pass. Under the rule as registered, fold1_seed2 is again not stable:
its last score is 0.00333 below its best. That verdict is reported and decides
nothing.

Against the runs at `a7f3a64`, which used the same seeds:

- every best validation score moved by at most 0.00053;
- seven of the nine best steps are the same;
- fold1_seed0 set its best at step 19000 this time, not 12500, so it ran to
  the step limit instead of stopping at step 17500;
- fold2_seed0 set its best at step 15500, not 15000, so its patience did not
  run out at step 20000;
- fold1_seed2 again set its best at step 16000 and again failed the rule as
  registered.

The training code, the data order, and the seeds are the same at both
commits. These differences are consistent with nondeterministic GPU
arithmetic.
