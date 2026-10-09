# Mean-Feature floor on the Phase 5 records

Recorded 2026-10-09. At this commit, `check`, the overfit gate, training, and
evaluation have not run on real data. No Phase 5 result on real data exists.

## The gap

CLAUDE.md requires every reported metric to be accompanied by the No-Warp-Copy
and Mean-Feature floors. PROTOCOL 3.7 defines Mean-Feature as the frozen global
mean vector. It is reported under raw cosine only and recorded as not
applicable under centering. Phases 3 and 4 report it on both paths.

The Phase 5 per-pair records carried No-Warp-Copy on every support. They
carried Mean-Feature only inside the landing-offset diagnostic. The primary,
splat-pool, and cross-path records had no Mean-Feature columns. This change
closes that gap against CLAUDE.md and PROTOCOL 3.7.

## The change

- Each record scores Mean-Feature on its own support. The floor predicts the
  frozen mean vector at every target. The targets are the ones the record's
  other arms are scored against.
- Primary per-point record: `meanfeat_raw` and `meanfeat_l2_raw`. The targets
  are the target grid read at each landing.
- Splat-pool record: `sp_meanfeat_raw` and `sp_meanfeat_l2_raw`. The targets
  are the scored cells' own features.
- Cross-path record: `x_meanfeat_raw` and `x_meanfeat_l2_raw` on the per-point
  arm, against the pooled landing reads. `x_sp_meanfeat_raw` and
  `x_sp_meanfeat_l2_raw` on the splat arm, against the cells' own features.
  The two arms score different targets. So each arm carries its own floor, as
  each carries its own No-Warp-Copy.
- Only raw cosine and raw L2 are written. The centered columns do not exist.
  No epsilon-regularized zero vector stands in for them. The landing-offset
  diagnostic already uses this representation.
- An empty support gives NaN, as it does for every arm.
- `lot.phase5_estimands` registers the eight columns. It adds four cells, under
  raw cosine only: `mean_feature` on per_point, `sp_mean_feature` on
  splat_pool, and `x_mean_feature` and `x_sp_mean_feature` on cross_path. They
  are floors, not interpreted effects. PROTOCOL 3.9's near-zero rule does not
  apply to them.

## What did not change

The headline estimand, every existing quantity, every support, every
comparator, and the near-zero logic are unchanged. `configs/phase5.yaml` is
untouched, so the config digest has not moved. The gate and the receipt bind
no record columns, so neither needed a change.

Every pre-existing column and cell was compared against the parent commit,
`fa30da6`, by running the same inputs through both trees. The comparison
covered the 52 pre-existing fields of the three records on the analytic
two-plane scene. It also covered 194 cells, with their intervals and
disclosures, from the synthetic estimand records under two scene spreads. All
were bitwise identical.

## Where it lives

- `lot.phase5_score`: `_mean_feature` and `_raw_prefixed`, called by
  `score_primary`, `score_splat_pool`, and `score_cross_path`.
- `lot.phase5_estimands`: the fields, the four cells, and their populations.
- `tests/test_phase5_score.py`: exact raw cosines and L2 against the mean on a
  hand-built grid, for each record. One landing reads a blend of two cells, so
  the test tells the landing read apart from the cell's own vector. It also
  covers empty supports, and the scorer writing exactly the registered columns.
- `tests/test_phase5_estimands.py`: the registry, the raw-only cells and their
  populations, and the near-zero rule not applying.
- `tests/test_phase5_modes.py`: the columns on every evaluate row, finite
  exactly when the record's support is not empty, and identical across seeds.
  The primary floor also matches an independent float64 reprojection.

Stream AC is not built yet. CLAUDE.md requires it to show these cells beside
the metrics they accompany.
