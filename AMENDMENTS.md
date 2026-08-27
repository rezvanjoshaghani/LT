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

### 2026-08-26: encoder revisions are recorded, and pinnable

LOT_DINOV2_REVISION and LOT_VGGT_REVISION pin the checkpoints;
scripts/pin_encoder_revisions.py resolves both and prints them. VGGT's
inference implementation is a third artifact and is installed at a commit
rather than a branch, since the same state dict run through different code
produces different features. All three are recorded in every cache and travel
into the evaluation run record.

Concrete values are not written here yet. They are resolved on the cluster
immediately before the caching jobs, and recording a value that has not been
used would be a claim rather than a record. The entry is completed when the
caches are rebuilt.

### 2026-08-26: the analysis config has a measurement identity and a reporting identity

A report was not bound to the config that produced the run it reads, so a
different co-visibility tolerance, sampling cap, or manifest bound would have
produced a different report from the same parquet without complaint.

Requiring the whole config to match is wrong in the other direction, because
PROTOCOL 3.4 has the support thresholds set from realized counts after the run.
That edit is a reporting change by construction, and forbidding it would forbid
the documented workflow.

So the config carries two digests. The measurement digest covers the values that
decide what the rows contain, and the analysis refuses a run whose measurement
digest differs. The reporting digest covers bin edges, support thresholds,
bootstrap settings and gate tolerances; a difference there is reported, not
refused. Both are recorded per scene.

### 2026-08-26: the sampling design, the edge convention, and the design floor are measurement values

Three additions to configs/analysis.yaml, all pre-freeze, all forced by the
same defect found from three directions: a value that decides which pairs are
drawn or gated was classified as a reporting value, so a permitted post-run
reporting edit could silently change the sample or an evaluation-time gate
while two runs compared as one measurement.

stratum_parallax_edges and stratum_rotation_edges_deg freeze the strata the
pair sample is drawn within, separately from the reporting bins, which PROTOCOL
3.4 permits to be widened once from counts after the run. They hold the same
numbers as the reporting edges today; the point is that one set can now move
without the other.

bin_right_closed moves into the measurement identity. It governs the stratum
labels as well as the reporting bins, so flipping it moves which pairs a capped
stratum draws; and PROTOCOL 3.4 froze the convention outright, so no post-run
edit to it was permitted anyway.

translation_parallax_design_floor: 0.025 is the floor the evaluation-time
assertion reads. It previously read the first reporting edge, so widening that
edge to 0.1, a permitted reporting edit, would have made evaluation reject at
0.08 under one identity what it accepted under an equal one. The floor is the
translation program's own design property and does not move with reporting.

### 2026-08-26: unpinned encoder provenance refuses a report

PROTOCOL locks the encoders. Recording that a run was unpinned, and warning
about it, is not a lock: the complete Phase 3 table and figures could be
generated from unpinned caches. The reporting layer now refuses provenance
whose weights_revision or code_revision reads unpinned or unknown, the caching
job refuses to start without LOT_DINOV2_REVISION and LOT_VGGT_REVISION set, and
cross-scene aggregation compares the full encoder identity, weights fingerprint
plus both revisions, rather than the fingerprint alone.

The one accepted exception is a declaration, not an omission: DINOv2's
checkpoint bytes are served from an unversioned URL and its loader records
weights_revision as "unpinnable: ..." while pinning the hub ref it can pin.
A declared impossibility is provenance; an unset variable is not.

Tightened the same day: a pin is a full 40-hex commit hash, validated
positively at every gate, the encoder loaders, the caching job, and the
reporting layer. The first version listed the bad strings instead, and a
blocklist accepts everything it did not think of: "main" passed it, and a
branch name is a moving ref, the opposite of a pin wearing the shape of one.
Each sidecar's provenance must also cover exactly the encoders its run
declares, because an empty mapping exists, passes a presence check, and gates
nothing. The mean vector's record now binds the vector's own bytes and its
build verifies each archive against the digest recorded at write time, closing
the two in-place replacements that input-side provenance cannot see.

---

## Post-freeze

None.
