# AMENDMENTS.md

Companion to PROTOCOL.md. PROTOCOL.md is never edited after the freeze commit;
every change to it, and every change to a value in configs/analysis.yaml, is
recorded here as a dated entry with a rationale.

Two kinds of entry appear below.

Pre-freeze entries record changes made before the freeze commit. They are not
amendments in the strict sense, since there was nothing yet to amend, and they
are listed anyway. A protocol that arrived at its frozen text through a series
of undocumented edits is not a pre-registration, whatever its header says. What
makes the freeze meaningful is that the trail into it is visible.

Post-freeze entries are amendments proper. None exist yet.

The freeze commit is the last commit that touches PROTOCOL.md,
configs/analysis.yaml, VALIDATION.md, or this file before any Phase 4 result
exists. After it, this file is the only place either document changes.

---

## Pre-freeze

### 2026-08-24: seven edits to PROTOCOL.md during validation

Made by the author while validation was in progress, before the freeze commit.
Each resolved an ambiguity that validation surfaced, and each is a narrowing:
none loosened a requirement.

1. The evaluation record schema is wide in metric. Raw and centered cosine are
   columns of one row rather than separate rows, so a metric cannot be compared
   against a differently populated set of the same records.
2. The Mean-Feature vector and the centering statistic are averaged over the
   training split only, never over evaluated frames.
3. The Neighbor-Patch direction is drawn by a hash of the record's sample_id,
   so it does not depend on batching or execution order.
4. The empty interval asserted for translation-program pairs is scoped to
   (0, 0.025), and orbit pairs may legitimately fall in it.
5. Bin intervals are closed on the right, matching the implementation.
6. Section 3.7 scores every variant on the path's common valid set, so a
   margin is paired by construction rather than by a later check.
7. Section 3.9 compares the two paths on their cross-path intersection, matched
   by sample_id.

### 2026-08-25: zero_parallax_tol raised from 1e-9 to 1e-5

Also adds min_expected_median_depth_m: 1.0 to configs/analysis.yaml, with a
load-time assertion that rotation_position_bound_m divided by it does not
exceed zero_parallax_tol.

The two constants were mutually inconsistent. A 1e-6 m position spread among
in-place rotation frames, which is the read-back bound the protocol allows, is
a parallax of 5e-7 at 2 m depth. That is five hundred times the 1e-9 tolerance,
so pure-rotation pairs would have failed the exact-zero test that puts them in
the zero bin, and PROTOCOL 3.3's primary rotation curve would have lost the
regime it is made of.

The tolerance now sits above the largest parallax the position bound can
produce at the shallowest depth the render filter admits, and the assertion
holds the two together, so neither can be changed alone into an inconsistent
pair again.

### 2026-08-25: the reported estimand is the unweighted mean of pair-level means

PROTOCOL does not state how cell aggregates weight their records, and the
implementation and the completed Phase 3 tables disagreed: the tables weighted
each pair by its comparison count.

PROTOCOL 3.4 does fix the unit, twice. Support "depends primarily on
independent camera pairs and scene coverage, not raw comparison counts", and
the bootstrap is over scenes and camera pairs with points and patches excluded
by name. Weighting a pair by how many of its correspondences survived makes the
point the unit of the point estimate while the interval around it keeps the
pair, so the two describe different quantities.

The unweighted mean is the estimand. The comparison-weighted value is reported
beside it as a diagnostic column, because the weighting is not neutral: a
pair's comparison count is largely set by how much of the target the context
still sees, so within a bin the weight rises with the easier geometry.

Consequence: the corrected Phase 3 figures will not match the numbers in the
completed run, for this reason as well as the definitional corrections.

### 2026-08-26: translation_rotation_bound_deg added, then set to 1.0e-7

The mirror of rotation_position_bound_m on the other regime's other axis.
PROTOCOL 3.3 makes translation the sole source of the primary parallax curve
because that regime holds rotation at exactly zero, but nothing checked the
manifests for it. A translation frame whose stored orientation drifted would
put an unlabelled rotation into the marginal that exists to exclude it, and
nothing downstream could see it, because the regime tag is what routes a pair
onto the curve.

Translation frames share one orientation by construction, so this is a
read-back tolerance on the stored pose, not a design allowance.

Set at 1.0e-4 first, which was inconsistent with zero_rotation_tol_deg at
1.0e-6: a translation pair at the manifest bound would have passed validation
and then been binned outside the zero-rotation bin its own regime defines. A
load-time assertion now holds the bound at or below the tolerance, and the
value is 1.0e-7. The measured pairwise residual over the camera programs is
exactly zero, since the translation program copies the rotation block
unchanged.

The residual is measured over every unordered pair, not against the first
frame. Orientations at 0, +9e-5 and -9e-5 degrees are each within 9e-5 of the
first while the worst actual pair is 1.8e-4, and the pair is what becomes a
camera pair downstream.

### 2026-08-26: PROTOCOL 3.9's gated statistic is the mean per-pair absolute difference

3.9 states that the two paths must agree within the tolerance, without naming
the statistic the tolerance applies to. Three readings were tried.

The largest single pair's difference. Rejected: the 0.003 tolerance was
established on the completed run's pooled per-path scores, so gating the
maximum applies a number to a statistic it was never measured against.

The absolute difference of the two pooled means. Rejected: it destroys the
measurement. A pair where the per-point path reads high and a pair where it
reads low cancel exactly, so two pairs disagreeing by 0.2 in opposite
directions give an aggregate of zero and a passing gate.

The mean of the per-pair absolute differences. Adopted: paired, an aggregate,
and unable to cancel. The median, maximum, and count of pairs above tolerance
are reported beside it as diagnostics.

Both metrics are gated, raw and centered. Centering subtracts a fixed vector
from both sides but does not act equally on the two paths, since the splat
path's pooled output is a weighted mean over a cell and the per-point path's is
a single sample. A gate reading only the raw column would certify a centered
table it never looked at, and the centered metric is the one the VGGT reading
rests on.

### 2026-08-26: rotation_angle_deg is computed from the skew term, not the trace alone

Not a config change, and it moves reported numbers, so it is recorded here.

The angle was acos of (trace - 1) / 2. That is correct mathematics and poor
arithmetic near identity, where the cosine is 1 minus something of order theta
squared: float64 rounding of the trace puts a floor of about 8.5e-7 degrees on
what the formula can resolve. zero_rotation_tol_deg is 1.0e-6, so the boundary
of the zero-rotation bin sat below the noise of the quantity being binned, and
the bin was a statement about arithmetic rather than about the camera.

It is now atan2 of the skew magnitude against the trace term. The two agree
mathematically. The skew term is linear in theta near zero, so small angles
resolve to full precision, and atan2 stays stable near 180 degrees where the
cosine is flat again. Measured relative error is at most 2.4e-16 from 1e-9
degrees to 180.

Consequence: rotation_deg changes by at most a part in 1e15 for the angles the
programs actually produce, and changes materially only for angles at or below
the old noise floor, which is exactly the population the zero bin is about.

---

## Post-freeze

None.
