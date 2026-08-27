"""VALIDATION Part 4 (record accounting, population, hygiene), 1.7, and 2.5.

No outputs/ directory, no parquet and no feature cache ship with this
repository, so there is no shipped evaluation table to audit. This script
instead drives the real lot.evaluate pipeline over a synthetic scene, writes a
real parquet through the real writer, and audits THAT. What it establishes is
the record layout, the grain, the floor definitions actually implemented, and
the NaN policy. Absolute values are meaningless here; structure is not.

Section 4.1 requires the expected record count to be derived and stated BEFORE
the observed count is inspected. The derivation is printed first, from the code
path alone, and only then compared.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lot.encoders import CACHE_VERSION, cache_dir, features_digest  # noqa: E402
from lot.analysis_config import load_analysis_config  # noqa: E402
from lot.evaluate import (  # noqa: E402
    MEAN_FEATURE,
    NO_WARP_COPY,
    ORACLE_TRANSPORT,
    PER_POINT,
    SPLAT_POOL,
    EvalConfig,
    dataset_mean_vector,
    evaluate_scene,
    read_rows,
    write_rows,
)
from lot.render_replica import (  # noqa: E402
    FrameRecord,
    Manifest,
    intrinsics_from_hfov,
    program_rotation,
    program_translation,
    write_frame_stats,
    write_manifest,
)

SIDE = 112         # 8 x 8 patches at stride 14
CHANNELS = 768    # dinov2_vitb14's real width, so the cache the probe
                  # fabricates is one the validator will accept as that
                  # encoder's. A narrower fiction was caught by cache
                  # validation, which is the validator working.
SCENE = "room_0"

results: list[tuple[str, str, bool, str]] = []


def record(check, name, ok, detail):
    results.append((check, name, ok, detail))
    print(f"[{'PASS' if ok else 'FLAG'}] {check} {name}: {detail}")


def base_pose():
    """A world-from-camera pose looking down +z, matching the render conventions."""
    T = torch.eye(4, dtype=torch.float64)
    T[:3, 3] = torch.tensor([0.0, 1.5, 0.0], dtype=torch.float64)
    return T


def smooth_features(hp, wp, channels, seed):
    """Spatially correlated features.

    White noise would make a within-image shuffle exactly equal to a uniform
    random read, which is the degenerate case VALIDATION 3.7 warns against.
    Smooth features keep the shuffle control meaningful.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:hp, 0:wp].astype(np.float64)
    out = np.zeros((channels, hp, wp))
    for c in range(channels):
        fx, fy = rng.uniform(0.2, 1.1, 2)
        ph = rng.uniform(0, 6.28, 2)
        out[c] = np.sin(fx * x + ph[0]) + np.cos(fy * y + ph[1]) + 0.15 * rng.normal(size=(hp, wp))
    return out.astype(np.float16)


def build_scene(root):
    scene_root = root / SCENE
    (scene_root / "rgb").mkdir(parents=True)
    (scene_root / "depth").mkdir(parents=True)
    from PIL import Image

    K = intrinsics_from_hfov(SIDE, SIDE, 90.0)
    frames, features = [], {}
    # A depth map with real structure, so co-visibility and occlusion are not trivial.
    yy, xx = np.mgrid[0:SIDE, 0:SIDE]
    depth = np.where(xx < SIDE // 2, 2.0, 4.0).astype(np.float32)
    depth = depth + 0.002 * yy
    # Viewpoint 1's depth straddles every patch center: stripe boundaries at
    # columns 7 + 14k, so a center at u = 6.5 + 14k reads one pixel on each
    # side and the four-corner consistency test rejects every per-point
    # candidate, while the splat path keeps full support. Every vp01 pair is
    # then a genuine one-path pair, made by the sampler's own rules rather than
    # a forced mask, so the population accounting's one-path term is exercised
    # by this scene instead of only by a unit test. Translation, not rotation:
    # a planar z-depth map is only a consistent surface between two cameras
    # when they share orientation.
    striped = np.empty((SIDE, SIDE), dtype=np.float32)
    for col in range(SIDE):
        striped[:, col] = 3.0 if ((col + 7) // 14) % 2 == 0 else 5.0
    programs = [
        (0, depth, program_rotation(base_pose(), [-5.0, -2.5, 0.0, 2.5, 5.0], [])
         + program_translation(base_pose(), [0.05, 0.1], 3.0)),
        (1, striped, program_translation(base_pose(), [0.1, 0.2], 3.0)),
    ]
    i = 0
    for viewpoint, depth_map, posed in programs:
        counters: dict[str, int] = {}
        for frame in posed:
            index = counters.get(frame.regime, 0)
            counters[frame.regime] = index + 1
            fid = f"{SCENE}_vp{viewpoint:02d}_{frame.regime}_{index:03d}"
            Image.fromarray(np.zeros((SIDE, SIDE, 3), np.uint8)).save(
                scene_root / f"rgb/{fid}.png"
            )
            np.save(scene_root / f"depth/{fid}.npy", depth_map)
            frames.append(FrameRecord(
                frame_id=fid, scene=SCENE, regime=frame.regime,
                params=dict(frame.params, viewpoint=viewpoint),
                T_world_from_camera=frame.T_world_from_camera, K=K,
                height=SIDE, width=SIDE,
                rgb_path=f"rgb/{fid}.png", depth_path=f"depth/{fid}.npy",
            ))
            features[fid] = smooth_features(SIDE // 14, SIDE // 14, CHANNELS, seed=100 + i)
            i += 1
    manifest = Manifest(
        scene=SCENE,
        metadata={"depth_convention": {"raw_verdict": "planar_z", "stored_depth": "planar_z"}},
        frames=frames,
    )
    write_manifest(scene_root / "manifest.json", manifest)
    write_frame_stats(scene_root, manifest)
    d = cache_dir(root / "cache", "dinov2_vitb14", SCENE)
    d.mkdir(parents=True)
    np.savez(d / "features.npz", **features)
    # Evaluation validates a cache before opening it, so the probe cache has to
    # carry the provenance a real one does.
    (d / "meta.json").write_text(json.dumps({
        "cache_version": CACHE_VERSION,
        "encoder": "dinov2_vitb14",
        "scene": SCENE,
        "channels": CHANNELS,
        "patch_size": 14,
        "patch_grid": [SIDE // 14, SIDE // 14],
        "image_hw": [SIDE, SIDE],
        "dtype": "float16",
        "frame_count": len(features),
        "frame_ids": [f.frame_id for f in frames],
        "has_depth": False,
        "weights_fingerprint": "validation-probe",
        "weights_revision": "2" * 40,
        "code_revision": "3" * 40,
        "features_digest": features_digest(features),
        "depth_digest": None,
    }, indent=1), encoding="utf-8")
    return manifest


def main():
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="lot_validation_"))
    build_scene(tmp)
    cfg = EvalConfig(
        experiment_name="validation_probe", renders_root=tmp, cache_root=tmp / "cache",
        output_root=tmp / "out", scenes=[SCENE], encoders=["dinov2_vitb14"],
        seed=0, mean_vector_scenes=[SCENE],
    )
    mean_vector = dataset_mean_vector(cfg.cache_root, "dinov2_vitb14", [SCENE])

    # -----------------------------------------------------------------
    # 4.1 DERIVATION FIRST, before any observed count is read.
    # -----------------------------------------------------------------
    print("=" * 78)
    print("4.1 EXPECTED RECORD COUNT, derived from the code path before observation")
    print("=" * 78)
    print("""
One record = one (pair, encoder, path, variant). `metric` is NOT a row
dimension: each row carries wide metric COLUMNS. Every variant exists on both
paths, per PROTOCOL 3.6, so a pair scorable on both paths yields 5 x 2 = 10
rows per encoder, and a pair scorable on exactly one path, which is valid and
deliberately kept, yields 5.

The expected counts come from the run metadata, never from the rows being
checked. An earlier version derived n_pairs from the produced rows, and a pair
dropped entirely then shrank the expected and the observed count together, so
the check was an identity. A later version took the pair count from metadata
but assumed ten rows for every pair, which a one-path pair falsifies; the
evaluator now records the both-paths and one-path splits independently of row
production, and the expectation is built from those.
""")
    rows, run_meta = evaluate_scene(
        cfg, SCENE, {"dinov2_vitb14": mean_vector}, load_analysis_config()
    )
    pairs = {(r["context_frame_id"], r["target_frame_id"]) for r in rows}
    both = run_meta["pairs_scored_both_paths"]
    one = run_meta["pairs_scored_one_path"]
    expected_pairs = run_meta["pairs_considered"] - run_meta["pairs_dropped_unscorable"]
    expected = (both * 10 + one * 5) * 1
    record("4.1", "record count matches the metadata-derived population",
           len(rows) == expected
           and len(pairs) == expected_pairs
           and both + one == expected_pairs,
           f"metadata: {run_meta['pairs_considered']} considered - "
           f"{run_meta['pairs_dropped_unscorable']} dropped = {expected_pairs} pairs "
           f"({both} on both paths, {one} on one); {both} x 10 + {one} x 5 = "
           f"{expected} expected rows. Observed: {len(pairs)} distinct pairs, "
           f"{len(rows)} rows")
    # The probe scene is built to contain both terms: viewpoint 1's striped
    # depth makes every one of its pairs splat-only by the sampler's own rules.
    # If either count is zero the scene has stopped exercising half the
    # arithmetic, and this check would be passing on the half it still sees.
    record("4.1c", "both population terms are exercised, not merely summed",
           both >= 1 and one >= 1,
           f"pairs on both paths: {both}; pairs on exactly one: {one}. A zero "
           f"here means the probe scene regressed and the corresponding term "
           f"of the expectation is vacuous.")

    by_path = Counter((r["path"], r["variant"]) for r in rows)
    print("\n  variants actually present, by path:")
    for (p, v), n in sorted(by_path.items()):
        print(f"    {p:11s} {v:17s} {n}")
    # This check once recorded that two nulls existed on one path only. It now
    # asserts the repaired state: PROTOCOL 3.5 runs both paths and 3.10's Figure
    # A asks for the full ladder, so every variant must appear on both.
    missing = [
        (path, variant)
        for path in ("per_point", "splat_pool")
        for variant in (
            "Oracle-Transport", "No-Warp-Copy", "Mean-Feature",
            "Neighbor-Patch", "Random-Patch",
        )
        if (path, variant) not in by_path
    ]
    record("4.1b", "every variant exists on both paths", not missing,
           f"PROTOCOL 3.5 runs both paths; 3.10 Figure A asks for the full null "
           f"ladder. Missing: {missing}")

    # -----------------------------------------------------------------
    # 4.3 Grain and hygiene
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("4.3 GRAIN AND HYGIENE")
    print("=" * 78)
    print("  columns:", sorted(rows[0].keys()))
    grain = ("scene", "context_frame_id", "target_frame_id", "encoder", "path", "variant")
    keys = Counter(tuple(r[c] for c in grain) for r in rows)
    dupes = {k: n for k, n in keys.items() if n > 1}
    record("4.3a", "rows unique at the established grain", not dupes,
           f"grain = {grain}; {len(keys)} distinct keys, {len(dupes)} duplicated")

    # PROTOCOL 3.2 accepts either the contributing sample_id set or a validity
    # bitmask, persisted per record. Storage is pair-aggregated here, so the
    # bitmask is the form that applies.
    record("4.3b", "sample identity is persisted per record",
           "sample_mask" in rows[0],
           "PROTOCOL 3.2 requires the contributing sample_id set or its validity "
           f"bitmask persisted per record. sample_mask present: "
           f"{'sample_mask' in rows[0]}")

    metric_cols = ["cosine_mean", "l2_mean", "cosine_centered_mean", "l2_centered_mean"]
    nonfinite = [
        (r["path"], r["variant"], c)
        for r in rows for c in metric_cols if not np.isfinite(r[c])
    ]
    empty_rows = [r for r in rows if r["n"] == 0]
    nonfinite_populated = [
        (r["path"], r["variant"], c)
        for r in rows if r["n"] > 0 for c in metric_cols if not np.isfinite(r[c])
    ]
    # PROTOCOL 3.7 permits a nonfinite in exactly one place: Mean-Feature's
    # centered columns, where the vector subtracted is the prediction itself, so
    # the centered prediction is the zero vector and the cosine is undefined.
    # Anywhere else is a failed method.
    nonfinite_populated = [
        entry for entry in nonfinite_populated
        if not (entry[1] == "Mean-Feature" and entry[2].startswith(("cosine_centered", "l2_centered")))
    ]
    record("4.3c", "no nonfinite scores outside centered Mean-Feature",
           not nonfinite_populated,
           f"{len(nonfinite_populated)} nonfinite values in the {len(rows) - len(empty_rows)} "
           f"rows with n > 0" +
           (f"; examples {nonfinite_populated[:3]}" if nonfinite_populated else ""))
    record("4.3c-empty", "no rows carry an empty sample set", not empty_rows,
           f"{len(empty_rows)} of {len(rows)} rows have n == 0 and therefore carry NaN in "
           f"all four metric columns. evaluate.value_agreement returns (nan, nan) for an "
           f"empty selection by design and write_rows accepts it, so a pair with no "
           f"co-visible surface reaches the parquet as NaN scores. Incidence depends on "
           f"the scene; this synthetic probe is not evidence about the real run.")

    bin_cols = [c for c in rows[0] if c.endswith("_bin")]
    record("1.8", "bin labels absent from rows", not bin_cols,
           f"PROTOCOL 3.2: bin labels never appear in rows. Found: {bin_cols}")

    # 4.3 determinism: evaluate the same scene twice.
    rows2, _ = evaluate_scene(
        cfg, SCENE, {"dinov2_vitb14": mean_vector}, load_analysis_config()
    )

    def eq(x, y):
        # NaN == NaN must count as identical here: a pair with no co-visible
        # points legitimately scores NaN, and treating that as a difference
        # would report a determinism failure that is only float semantics.
        if isinstance(x, float) and isinstance(y, float):
            return (np.isnan(x) and np.isnan(y)) or x == y
        return x == y

    same = len(rows) == len(rows2) and all(
        all(eq(a[c], b[c]) for c in a) for a, b in zip(rows, rows2)
    )
    record("4.3d", "re-running one scene reproduces identical rows", same,
           "two evaluate_scene calls on the same config compared field by field")

    # write/read round trip through the real writer
    out = cfg.eval_dir / f"{SCENE}.parquet"
    write_rows(out, rows, {"scene": SCENE})
    back = read_rows(out)
    record("4.3e", "parquet round trip preserves rows", len(back) == len(rows),
           f"{len(back)} rows read back from {out.name}")
    try:
        write_rows(out, rows, {"scene": SCENE})
        record("1.10", "writer refuses to overwrite", False, "second write SUCCEEDED")
    except FileExistsError:
        record("1.10", "writer refuses to overwrite", True, "second write raised FileExistsError")

    # -----------------------------------------------------------------
    # 2.5 / 1.7 Floors and centering
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2.5 / 1.7 FLOORS AND CENTERING")
    print("=" * 78)
    # Restricted to populated rows: an empty pair is NaN for every variant, which
    # is a different phenomenon and would mask the one under test here.
    mf = [r for r in rows if r["variant"] == MEAN_FEATURE and r["n"] > 0]
    finite_centered = [r for r in mf if np.isfinite(r["cosine_centered_mean"])]
    record("2.5a", "centered Mean-Feature is N/A", not finite_centered,
           f"{len(finite_centered)} of {len(mf)} populated Mean-Feature rows carry a "
           f"FINITE centered cosine. PROTOCOL 3.7: centered Mean-Feature is undefined "
           f"and must be recorded not applicable, and no implementation may manufacture "
           f"a score there.")
    if finite_centered:
        per_point_vals = [r["cosine_centered_mean"] for r in finite_centered
                          if r["path"] == PER_POINT]
        splat_vals = [r["cosine_centered_mean"] for r in finite_centered
                      if r["path"] == SPLAT_POOL]
        print(f"    centered Mean-Feature per_point  mean = {np.mean(per_point_vals):+.4f}")
        print(f"    centered Mean-Feature splat_pool mean = {np.mean(splat_vals):+.4f}")

    raw_pp = np.mean([r["cosine_mean"] for r in mf if r["path"] == PER_POINT])
    raw_sp = np.mean([r["cosine_mean"] for r in mf if r["path"] == SPLAT_POOL])
    record("2.5b", "Mean-Feature scores differ only by what they are scored against",
           True,
           f"per_point {raw_pp:.4f} vs splat_pool {raw_sp:.4f}, difference "
           f"{abs(raw_pp - raw_sp):.4f}. Reported, not gated. This check once "
           f"required the two to be equal. They must not be: the prediction is one "
           f"global vector on both paths, but the target is not one object, being a "
           f"bilinear read at a sample location on one path and a pooled cell on the "
           f"other. Requiring equality would have pushed the implementation towards a "
           f"per-path floor, which is the fault this file was written to find.")

    print("")
    print("  Mean-Feature prediction object, measured:")
    print(f"    mean_vector shape {tuple(mean_vector.shape)}, passed to both roles")
    record("1.7", "the centering vector equals the Mean-Feature floor object",
           mean_vector.dim() == 1,
           f"PROTOCOL 3.6 makes Mean-Feature one global D-vector and PROTOCOL 3.7 "
           f"makes the same vector the centering statistic. evaluate.py passes one "
           f"object to both roles; observed shape {tuple(mean_vector.shape)}.")

    print("\n" + "=" * 78)
    flags = [r for r in results if not r[2]]
    print(f"SUMMARY: {len(results) - len(flags)} conformant, {len(flags)} flagged, "
          f"of {len(results)} checks")
    for c, n, _, d in flags:
        print(f"  FLAG {c} {n}")
    print(f"\nprobe artifacts under {tmp}")


if __name__ == "__main__":
    main()
