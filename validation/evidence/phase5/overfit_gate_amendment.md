# The overfit gate scores with dropout off: an amendment after a failure

Recorded 2026-10-11 local time, 2026-10-12 UTC, before the gate is rerun. No Phase 5 model has trained at
scale and no Phase 5 test result exists. This amendment was decided after the
gate failed and after a diagnostic showed what a dropout-off score would have
been. That order is stated here so that no reader has to infer it.

## The failure

The tiny-subset overfit gate ran on Borah at commit
`1e7ce8ff95dc1537013f3901d24e187aa2822b07`, on an NVIDIA L40S, job 3362298.
Its receipt, `outputs/phase5_rung2/evidence/tiny_overfit.json`, stamped
2026-10-12T03:23:32Z, records:

- `passed`: false
- `reached_centered_cosine`: 0.9763194918632507
- `threshold`: 0.98
- `steps`: 2000
- `device`: cuda
- the frozen subset: eight pairs from fold 0's training scenes, three rotation,
  three translation, and two orbit, with 618 to 1,285 supported samples each

The gate trained in train mode and reported the best training-batch centered
cosine. Dropout, 0.1 in the frozen architecture, was active in that forward
pass.

## The diagnostic

`validation/phase5_overfit_diagnostic.py`, committed at `4b2e92a`, replayed the
gate's training with the same subset, seed, optimizer, clipping, and steps. It
ran on Borah as job 3362310, on an NVIDIA L40S, from a checkout at the same
commit. Every 50 steps it scored the batch twice: as the gate does, with
dropout on, and in eval mode, with dropout off. The eval forward draws no
random numbers, so the training trajectory was the gate's own. Its best
dropout-on value, 0.9762, matched the receipt's 0.9763 up to GPU
nondeterminism.

| step | dropout on | dropout off |
|---|---|---|
| 1000 | 0.9480 | 0.9592 |
| 1500 | 0.9680 | 0.9766 |
| 1700 | 0.9721 | 0.9802 |
| 2000 | 0.9762 | 0.9832 |

With dropout off at step 2000, every pair scored at least 0.9818, the lowest
being `room_1`, translation. The regime means were rotation 0.9845, orbit
0.9839, and translation 0.9821. Dropout cost between 0.007 and 0.012
throughout. Both curves were still rising slowly at step 2000.

## The decision

The options were put to the user on 2026-10-11, local time. The first, as
offered:

> Recommended: change only how the gate scores. After each training step,
> score the eight pairs with dropout off, and pass on reaching 0.98 within the
> same 2,000 steps. Everything else stays frozen: the 0.98 threshold, the eight
> pairs, the seed, the step count, the optimizer and learning rate, and dropout
> during training. This is a change made after seeing the failure, so I'd
> record it openly. The failed receipt, this diagnostic, and your decision get
> saved verbatim before the gate reruns.

The second was to keep the gate as frozen, so that the stop would stand and
Phase 5 would halt with training adequacy not established. The user's reply,
verbatim:

> go with 1

## The rationale

The specification's step 14 asks whether the model can "overfit a tiny
sample", and the gate's own code says it establishes that "the supervised
mapping is representable by this trunk under this optimizer". A score taken
with dropout active is a sample of a stochastic network, not the fit of the
network. Everywhere else in Phase 5 the model is scored with dropout off:
validation, checkpoint selection, the input-use controls, and the test
evaluation. The amended gate scores the fit the same way.

## What changes

- After every optimizer step, the gate scores the eight pairs in eval mode,
  with dropout off and no gradient, and records the pooled centered cosine.
- It passes when that score reaches 0.98 within 2000 steps, and stops there.
- Its receipt reports the dropout-off score as `reached_centered_cosine`, the
  best dropout-on score beside it as `train_mode_centered_cosine`, and the
  scoring rule as `scoring: eval_mode`. Acceptance requires that rule.
- Every evaluation run record embeds both values and the rule with the
  verdict. The adequacy table reports them as `tiny_overfit_scoring` and
  `tiny_overfit_train_mode_centered_cosine`, beside the existing
  `tiny_overfit_reached_centered_cosine`.

## What does not change

The threshold of 0.98, the eight pairs and how they are drawn, the seed, the
2000 steps, the optimizer, its learning rate and weight decay, gradient
clipping, the architecture, dropout during training, and every training,
evaluation, and reporting rule. The eval-mode forward draws no random numbers,
so training follows exactly the trajectory the frozen gate followed. The
config digest is unchanged.

## What follows

The failed receipt stays on Borah, archived by the rerun rather than
overwritten. The chain restarts at the commit that carries this amendment:
`check`, then the amended overfit gate, then training. By the diagnostic, the
gate should pass near step 1700.
