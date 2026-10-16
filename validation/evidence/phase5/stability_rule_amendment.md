# The stable-curve rule reads the curve's level: an amendment after a failure

Recorded 2026-10-15, before the chain is rerun. Nine Phase 5 models have
trained, at commit `a7f3a64`. No control and no evaluation has run on them,
so no Phase 5 test result exists. This amendment was decided after one of the
nine runs failed decision 5's rule, and after the rule had been applied to all
nine curves. That order is stated here so that no reader has to infer it.

This is the second amendment of a rule Phase 5 registered. The first,
`overfit_gate_amendment.md`, changed how the overfit gate scores. Both were
decided after a failure and before any test result existed.

## The rule as registered

Decision 5 of `reporting_rules.md` makes PLAN.md's "training converges with a
stable validation curve" an acceptance condition. A run's curve is stable when
every validation score is finite, at least two validations ran, and early
stopping fired, or the last validation score is within 0.003 of the best. All
nine runs must be stable. `lot.phase5_outcomes.stable_validation_curve`
applied the rule, and acceptance condition 19 required it of every run.

## The failure

The nine training tasks ran on Borah as array 3362608, at commit
`a7f3a64eb145104cbbb7f2f403347b0894bea3ca`, licensed by the receipts that
`overfit_gate_rerun.md` records. All nine completed with exit code 0, on
cuda, in 53 to 60 minutes each, with a peak resident memory of 16.3 to 16.6
GB. Validation ran every 500 steps. The early-stopping patience is 10
validations.

The frozen rule was run on the nine training records on Borah. The level gap
in the table is the highest mean of any 10 consecutive validation scores less
the mean of the last 10, computed there from the same records.

| run | validations | steps | best step | best | last | last below best | level gap | as registered |
|---|---|---|---|---|---|---|---|---|
| fold0_seed0 | 40 | 20000 | 16000 | 0.66647 | 0.66435 | 0.00212 | 0.00002 | stable |
| fold0_seed1 | 40 | 20000 | 17000 | 0.66869 | 0.66697 | 0.00172 | 0.00000 | stable |
| fold0_seed2 | 40 | 20000 | 19000 | 0.66702 | 0.66520 | 0.00183 | 0.00000 | stable |
| fold1_seed0 | 35 | 17500 | 12500 | 0.68266 | 0.68040 | 0.00226 | 0.00023 | stable, early stopping |
| fold1_seed1 | 39 | 19500 | 14500 | 0.68115 | 0.67547 | 0.00568 | 0.00138 | stable, early stopping |
| fold1_seed2 | 40 | 20000 | 16000 | 0.68168 | 0.67825 | 0.00342 | 0.00025 | not stable |
| fold2_seed0 | 40 | 20000 | 15000 | 0.68772 | 0.68559 | 0.00213 | 0.00076 | stable, early stopping |
| fold2_seed1 | 40 | 20000 | 16500 | 0.68960 | 0.68757 | 0.00202 | 0.00003 | stable |
| fold2_seed2 | 40 | 20000 | 16000 | 0.69131 | 0.69065 | 0.00066 | 0.00000 | stable |

fold2_seed0's patience ran out at its last validation, step 20000, so early
stopping fired as training reached its step limit.

fold1_seed2 did not stop early, and its last score was 0.00342 below its best.
Under the rule as registered, condition 19 would fail, and Phase 5 could not be
accepted.

## What the curves show

- fold1_seed2 had levelled off. Over its last 10 validations its scores
  scatter with a standard deviation of about 0.0015, read from the curves
  printed to three decimals. Its best, 0.68168 at step 16000, is one high
  reading about 0.003 above that level. The mean of its last 10 validations is
  0.00025 below the highest mean of any 10 consecutive validations.
- The rule judged two similar endings in opposite ways. fold1_seed1 ended
  0.00568 below its best and counted as stable, because early stopping fired.
  fold1_seed2 ended 0.00342 below its best and did not. Its high reading at
  step 16000 reset the patience count and left eight validations before the
  step limit, so the rule fell back on comparing the last reading with that
  peak.
- Across all nine runs, the largest level gap is 0.00138.

## The decision

The options were put to the user on 2026-10-15. The first, as offered:

> Recommended: change the rule, then rerun the chain. The curve would be
> judged by its level, not by single readings. It counts as stable if early
> stopping fired, or if the average of the last 10 validations is within 0.003
> of the best average of any 10 in a row. The 0.003 tolerance, the
> finite-score check and the all-nine requirement stay. The current rule's
> verdict would still be reported alongside.
>
> Cost: the rule sits in phase5_outcomes.py:434, which is frozen at E, so this
> needs a new commit and a full rerun. That means check, overfit, all nine
> training runs, then the rest: about a day on the cluster. The current
> checkpoints would be kept but never evaluated.
>
> The downside: this would be the second rule change in a week, again after a
> failure. I've also already seen that every current run passes the new rule;
> the largest gap is 0.0016. Both changes would be written up with their
> original verdicts.

The 0.0016 was read from the curves printed to three decimals. The exact
values above, which came back with the user's reply, put the largest level gap
at 0.00138.

The second, as offered:

> Keep the rule and go straight to controls, lock and evaluation. Acceptance
> then fails on this condition. The results still go into the tables, but
> Phase 5 isn't accepted. Phase 6 could only start if you override CLAUDE.md's
> rule that each phase must pass acceptance first.

Retraining fold1_seed2 at the same commit until a curve passed, extending
training, and loosening the 0.003 were ruled out as options, as was routing a
new rule through the acceptance code to avoid a new commit. The user's reply,
verbatim:

> go with 1

## The rationale

The rule's second branch compares one reading with the maximum of up to 40.
On a plateau, readings scatter, and the maximum sits above the plateau by
about two standard deviations, close to the 0.003 tolerance. That branch
therefore fails curves that have levelled off, depending on where one high
reading fell. PLAN.md asks whether training converges with a stable curve. A
mean over the patience window answers that from the curve's level, and a
comparison of levels is not moved by one reading.

The amended rule is not only more lenient. A curve whose level collapsed and
whose last single reading recovered passes the rule as registered. It fails
the amended rule.

## What changes

- A run's curve is stable when every validation score is finite, at least two
  validations ran, and early stopping fired, or the mean of its last 10
  validations is within 0.003 of the highest mean of any 10 consecutive
  validations.
- 10 is the frozen early-stopping patience, read from the training
  configuration. Both branches therefore judge the last patience window.
- A run with fewer validations than one window is judged by the rule as
  registered. No completed Phase 5 run that did not stop early can have so
  few, since the frozen schedule validates 40 times in 20,000 steps.
- `lot.phase5_outcomes.stable_validation_curve` applies the amended rule.
  `stable_validation_curve_as_registered` keeps the rule as registered, and
  `validation_level_gap` computes the level gap.
- Acceptance condition 19 requires all nine runs to be stable under the
  amended rule. It reports each run's verdict under the rule as registered
  beside it, with the last score, the best, and the level gap. That verdict
  decides nothing.
- The adequacy table reports `stable_validation_curve` under the amended rule,
  with `stable_validation_curve_as_registered` and `validation_level_gap`
  beside the existing `validation_last_minus_best`.

## What does not change

The 0.003 tolerance, the patience, the validation schedule, early stopping,
checkpoint selection, training, the training configuration and its digest,
the config digest, the requirement that all nine runs be stable, every other
acceptance condition, and every estimand, outcome rule, and wording.
`reporting_rules.md` is not edited. Its decision 5 stands as registered, and
this note amends its rule.

## What follows

The rule sits in `src/lot/phase5_outcomes.py`, which is frozen at E with
`reporting_rules.md`. The chain therefore restarts at the commit that carries
this amendment: check, the overfit gate, and the nine training runs, then the
controls, the lock, and evaluation. Retraining moves the checkpoints and
records trained at `a7f3a64` aside as numbered siblings. They stay on Borah
and are never evaluated. The retrained runs are judged by the amended rule,
and their verdicts under the rule as registered are reported beside it.
