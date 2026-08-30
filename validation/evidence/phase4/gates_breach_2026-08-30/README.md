# Gates array breach evidence, 2026-08-30

The first Phase 4 pure-rotation gates array (SLURM job 3193085, Borah HEAD
1eaf16014e261aa8690542ea3b2a47d697df9973, clean worktree) breached the
PROTOCOL 4.5 forced-collision-order gate on all 18 scenes. The per-scene
error lines are in phase4gateerrors.txt, verbatim from the job logs. The
instrumented decomposition of the first breaching pair is in
diagnose_apartment_0_rotation_001_008.txt, verbatim from
validation/diagnose_forced_gate.py run at the same checkout.

Mechanism, proven by the diagnostic: the pair's baseline is exactly zero;
levels none and image reproduce Oracle bitwise; at level scene exactly one
source pixel lands one cell over, at a rasterization-boundary margin of
3.052e-05 px, which is one float32 ulp at that coordinate magnitude. That
pixel is the only lost winner (zero non-flipped losses, so the forced-key
machinery itself was clean), and the single five-patch cell it touches
carries the entire 1.586e-3 residual through pool renormalization. All 18
scene breaches are marginal, 1.0e-3 to 2.9e-3, scattered across levels
none, scene, image, and affine: a per-scene float lottery over which
(pair, level) first puts one pixel on a boundary, not a convention,
intrinsics, or frame error, all of which would move coordinates by half
pixels and flood the counts.

Resolution: Amendment A7 (AMENDMENTS.md), which freezes Oracle-Transport's
complete discrete rasterization structure for both forced arms and reports
landing-cell flips as a non-gating diagnostic. No numerical threshold
changed. This directory preserves what the pre-A7 construction measured.
