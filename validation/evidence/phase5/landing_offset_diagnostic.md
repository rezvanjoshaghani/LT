# Landing-offset diagnostic: pre-registration

Recorded 2026-10-09. At this commit, `check`, the overfit gate, training, and
evaluation have not run on real data. No Phase 5 result on real data exists.

## The decision

`landing_read_asymmetry.md` offered three responses to the landing read
asymmetry. The user chose the second on 2026-10-09. Their words, verbatim:

> yes, push it and go with option 2

Option 2 is a model-free diagnostic that sizes the landing read on real data
without touching any estimand. It had two candidates. This is the one
recommended to the user before the decision: grouping Context-Lift's scores by
how far each landing falls from a patch center. The splat-grid candidate is not
built.

## What it measures

Under the frozen per-point read, a predictor's grid is read with the target's
own bilinear weights, so its ceiling is one at every landing. Context-Lift
carries one patch vector, which an off-grid read of the target grid cannot
reproduce. The diagnostic measures how much Context-Lift loses to that read on
DINOv2 features.

Exact geometry does not depend on how far a landing falls from a patch center.
The read does. So the change in Context-Lift's score across offset groups sizes
the read.

## Definition, frozen here

1. Condition. Context-Lift Oracle-Transport: Context-Lift Transport-Only lifted
   with ground-truth context depth instead of aligned depth, so depth error
   cannot mix into the measurement. It is a reference condition, as
   Oracle-Transport is, and never a method under comparison.
2. Support. Its landed samples that are ground-truth evaluable, by the same
   rule as V_P5_pp. It does not depend on the alignment level.
3. Offset. The distance from the landing to the nearest target patch center, in
   patch units, under the frozen pixel-to-patch mapping. It is computed in
   float64 from the run's float32 landings. Its range is 0 to sqrt(2) / 2.
4. Bins. Upper edges at 0.1, 0.2, 0.3, 0.4, and 0.5 patch, closed on the right
   as PROTOCOL 3.4 fixes, and one open bin above 0.5. They are fixed width over
   the reachable range and were chosen from geometry alone. In the first bin
   the read puts at least 86 percent of its weight on the nearest patch. Under
   uniformly placed landings the six bins would hold 3.1, 9.4, 15.7, 22.0,
   28.3, and 21.5 percent of samples.
5. Arms. Each is scored per bin and on the whole support, against the frozen
   per-point target, which is the target grid read bilinearly at the landing.
   - Context-Lift Oracle-Transport, the context patch's own vector, with all
     four PROTOCOL 3.7 columns.
   - No-Warp-Copy, the context map read at the landing, with all four columns.
   - Mean-Feature, the centering vector, with raw cosine and raw L2 only, the
     only columns PROTOCOL 3.7 defines for it.
6. Records. Every evaluate parquet carries the columns on its region `all`
   rows, identical for every seed. Region rows carry zero counts and no scores.
   The columns are `offset_n_{b0..b5,all}`,
   `offset_{cl_oracle,nowarp}_{raw,centered,l2_raw,l2_centered}_{label}`, and
   `offset_meanfeat_{raw,l2_raw}_{label}`.

## Reported cells, frozen here

Each cell goes through `lot.phase5_estimands` with the unit and interval of
every Phase 5 cell. The estimate is the unweighted mean over camera pairs of
pair-level values. The interval is the scene-level bootstrap with the frozen
resamples, seed, and confidence. Each cell carries n_replicates and its support
counts.

- Per offset bin k: `cl_oracle_offset_b{k}`, `nowarp_offset_b{k}`,
  `cl_oracle_offset_margin_b{k}`, and, under raw cosine only,
  `meanfeat_offset_b{k}`. The population is the pairs with at least one sample
  in bin k.
- The same cells on the whole diagnostic support, labelled `all`.
- `read_deficit`: Context-Lift Oracle-Transport in the near-grid bin minus the
  same on the whole support, within each pair, over the pairs with at least
  one near-grid sample. It is paired by construction, because one scene draw
  serves both terms.
- The cells are stratified as the headline is, by regime and parallax bin.
  Centered cosine is primary, with raw beside it. They are reported once, from
  the primary level's records, because the diagnostic does not depend on the
  level. Seeds do not enter.
- None of these is an interpreted effect, so PROTOCOL 3.9's near-zero wording
  does not apply to them.

## Interpretation, frozen here

- The read deficit is reported beside delta_learn_pp in the same cell, in the
  same units, with its interval. It is never subtracted from delta_learn_pp. No
  corrected headline is reported.
- Where Predict-with-Depth leads, so delta_learn_pp is below zero, and the read
  deficit's upper interval end is at least the size of that lead, the text
  says the lead in that cell is within the size of the landing read. The lead
  is then not attributed to learning the transformation.
- Where Context-Lift leads, the text says the read works against Context-Lift,
  so its lead is if anything understated.
- The deficit is measured with ground-truth depth, and the headline uses
  aligned depth. The text states this beside the two numbers.
- The curve across bins is read for its shape only. A decline with offset is
  the read. The bins are not randomized: within a cell, offset can covary with
  image position and content, and the text says so.
- A negative read deficit contradicts the mechanism the diagnostic was built to
  measure. It is reported as an anomaly and not interpreted, per CLAUDE.md.

## What did not change

The headline estimand, every support, every comparator, the information-access
rules, the folds, the architecture, the training, and every existing reporting
rule are unchanged.

The config digest moved, by design. The diagnostic's specification is part of
what a run measures, so the gate receipts must be bound to it. Removing only
the new `landing_offset` section reproduces the previous digest,
`9e3508606bcaedb6068e95807aa27ea706d2d38b77f8c2df763d8f28968a0eca`, exactly.
The new digest is
`8d731c5e1dabe8764c40f3599037de530ae39de9f2537c1675e068f3c909b1f8`.

## Where it lives

- `configs/phase5.yaml`, section `landing_offset`, validated by
  `lot.phase5.landing_offset_edges`.
- `lot.phase5_score`: `landing_offsets`, `offset_bins`, `score_landing_offset`.
- `lot.phase5_modes.evaluate_scene`: the ground-truth lift and its support,
  per pair.
- `lot.phase5_estimands`: the fields, the populations, and the quantities.
- `tests/test_phase5_landing_offset.py`: offsets, bins, per-bin scores on
  hand-placed patches, the Mean-Feature columns, the pairing of the deficit,
  the populations, and the config.
- `tests/test_phase5_modes.py`: the records on the end-to-end fixture, with
  bins and scores checked against an independent float64 reprojection.

Stream AC, which is not built yet, must render these cells beside the
headline. Nothing else here waits on it.
