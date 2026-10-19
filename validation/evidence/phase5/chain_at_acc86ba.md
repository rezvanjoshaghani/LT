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

## The input-use controls

The controls ran as job 3387865, on gpu111, over each fold's validation
scenes. They wrote `input_use_controls_image.json`, sha256
`49b5353351a9ecbe95df7f8a5bdbf8f6327c0aeb4f797399ab07510f25bd9995`. Each
degradation is the shared baseline's validation centered cosine less the
shuffled one. Each count is the exchanges that changed nothing.

| run | pairs | pose shuffle | pose unchanged | depth shuffle | depth unchanged |
|---|---|---|---|---|---|
| fold0_seed0 | 2752 | +0.1957 | 1 | +0.0312 | 5 |
| fold0_seed1 | 2752 | +0.1983 | 1 | +0.0325 | 5 |
| fold0_seed2 | 2752 | +0.1992 | 1 | +0.0313 | 5 |
| fold1_seed0 | 2795 | +0.2141 | 2 | +0.0153 | 4 |
| fold1_seed1 | 2795 | +0.2079 | 2 | +0.0134 | 4 |
| fold1_seed2 | 2795 | +0.2092 | 2 | +0.0141 | 4 |
| fold2_seed0 | 2873 | +0.2039 | 4 | +0.0164 | 8 |
| fold2_seed1 | 2873 | +0.2064 | 4 | +0.0159 | 8 |
| fold2_seed2 | 2873 | +0.2048 | 4 | +0.0151 | 8 |

These are diagnostics, reported as observed. Every model loses about 0.20
when its pose input is shuffled, and 0.013 to 0.033 when its depth input is
shuffled. No degradation is within 0.003 of zero, so no control is labelled
"essentially no effect". The three seeds of each fold agree to within 0.007
on pose and 0.002 on depth.

## The checkpoint lock

`./scripts/run_phase5.sh lock` passed at 2026-10-18T17:52:10Z and wrote
`checkpoint_lock_image.json`. It binds the integration and overfit receipts,
the controls file above, and these checkpoints and training records, by
sha256:

| run | checkpoint | training record |
|---|---|---|
| fold0_seed0 | `8b0361f9655ee935cb4de306d398041024e364e6b72f1da2c6cc5fcf5f0f7d80` | `fed12e14d19c60cb38c7f21562f2911a0b06190d99ac2a8e12a322529720558a` |
| fold0_seed1 | `0fd21c0f9278da936cac0d4d33aefb775f8355da58dd699eeb21d0919abaadbe` | `e201cabbc2bde39d563aba2223caecf220b2c92e508531686c5188bdc3921425` |
| fold0_seed2 | `4133ad3e1dd5c579596e71565dfa15a49a4ea13300d59e2d2f79cbfcab247cbd` | `2f737115630526a0e0f3ecb1f6bcbd9e62d822ad0d6db0e7c7ef44066ae3266e` |
| fold1_seed0 | `6bac0916f5b9f7a64518ac38a29192f23a13c3b03647e282990f37cd55363620` | `04134a92c984706b26be045b657d123c5f0258c07d8479929a0e9a01d0169326` |
| fold1_seed1 | `3dedd728128d77c9e38a710634a62137bfa2534c0ca779100dd974e2a14a2053` | `5769889423a738352225ca8c68c2835d51b7f0b4860f9ac7720cff936481eeba` |
| fold1_seed2 | `6137361139cf5b21ea7ea0686d78fa512e221db755bea6bdd38b18216e6647fc` | `2bad1ac6642deab9809aee2f331253bd983845c2e70630c6ff7faf8a6bd11793` |
| fold2_seed0 | `89d717b510b0bb17e90b3607a2e40d120cc7ed4d40b56a26d6e3db546593fa12` | `ce2727907a043c9e613ea0205dc4ef125985e5dbc9b4753b5fc30580759021e3` |
| fold2_seed1 | `54966c5bd5e20b04c605399c921df3c5db51481a444e64a80aedd5eeea9a24c2` | `fe8f96d8d983fce7e083695f6265c8142f73e3aeacf1f5742af350875d7140d4` |
| fold2_seed2 | `e0a20d78e9fd77456440ca4264e06903a3149a1c66d39c1db25e217ec9f283ae` | `897c40d404cc1e80e92395bef8024cb2c3a65959a15166a340e9adfce69a3923` |

Its output is kept as `checkpoint_lock_image_20261018T175210Z.txt`. No
evaluation had run when the lock was written.

## Evaluation

`./scripts/run_phase5.sh evaluate` verified the gate receipts and the lock,
then submitted the 18 test scenes as array 3388663, with a limit of one day
per task. Borah's quality-of-service limit ran at most eight at once. All 18
completed with exit code 0, on gpu111, gpu114, and gpu117, in 12 to 17
minutes each, with a peak resident memory of 2.2 to 2.5 GiB. No task failed
or was resubmitted, so each scene was evaluated once.

The 18 parquets and their run records were written to
`outputs/phase5_rung2/eval/image/` on 2026-10-18. No test result was read
before this section was written.

## Acceptance

The tables, the figures, and acceptance ran at R = E, `acc86ba`, as jobs
3388726, 3388727, and 3388728 on the bsudfq partition. Each started only
after the one before it passed. The tables and the figures passed.

The first acceptance run returned NOT SATISFIED. Conditions 8, 11, and 21
failed, and the measured outcome was withheld. The other 18 conditions
passed. All three failures were one test,
`test_a_fit_masked_by_dropout_passes_the_amended_gate` in
`tests/test_phase5_train.py`, the only failure in the suite at R: 1 failed,
1429 passed, 4 skipped. Conditions 8 and 11 read that file's results, and
condition 21 reads the suite's.

The test was written with the overfit amendment at `a7f3a64`. It assumed that
in a 60-step run of a tiny model with dropout 0.5, at seed 0, the best
dropout-off score beats the best dropout-on score. Where it was written, that
held by 0.0032. On the Borah CPU node it did not: the best dropout-off score
was 0.43629, and the best dropout-on score 0.43768. The premise also fails for
two of the first eight seeds where the test was written, so it was never a
property of the gate. The other three overfit-gate tests check what the gate
reports, that its scoring leaves training unchanged, and where it stops. All
three passed on Borah.

The test now takes the first of 16 seeds whose run shows the effect by at
least 0.001, and checks the amended gate on that seed. Its assertions about
the gate are unchanged, and no measured code changed.
`post_evaluation_changes.md` names the change with its reason, as acceptance
requires of a test file whose results it reads. Writing that record exposed a
fixture in `tests/test_phase5_acceptance.py` that copied it into the
repository its tests build to stand for E. The fixture now leaves the record
out, and the record names that change too. The tables and figures stand,
because only neutral paths changed after they were built. Acceptance reruns
at the commit that carries the change, and the first verdict is kept as a
numbered sibling.

The first run's output named the failed conditions and withheld the measured
outcome. Under condition 16 it also printed how many interpreted-effect cells
carry each near-zero wording. The change reads nothing from them.
