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

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lot.encoders import cache_dir  # noqa: E402
from lot.evaluate import (  # noqa: E402
    MEAN_FEATURE,
    NO_WARP_COPY,
    ORACLE_TRANSPORT,
    PER_POINT,
    SPLAT_POOL,
    EvalConfig,
    dataset_mean_feature_map,
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
CHANNELS = 16
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
    posed = program_rotation(base_pose(), [-5.0, -2.5, 0.0, 2.5, 5.0], [])
    posed += program_translation(base_pose(), [0.05, 0.1], 3.0)
    frames, features = [], {}
    counters: dict[str, int] = {}
    # A depth map with real structure, so co-visibility and occlusion are not trivial.
    yy, xx = np.mgrid[0:SIDE, 0:SIDE]
    depth = np.where(xx < SIDE // 2, 2.0, 4.0).astype(np.float32)
    depth = depth + 0.002 * yy
    for i, frame in enumerate(posed):
        index = counters.get(frame.regime, 0)
        counters[frame.regime] = index + 1
        fid = f"{SCENE}_vp00_{frame.regime}_{index:03d}"
        Image.fromarray(np.zeros((SIDE, SIDE, 3), np.uint8)).save(scene_root / f"rgb/{fid}.png")
        np.save(scene_root / f"depth/{fid}.npy", depth)
        frames.append(FrameRecord(
            frame_id=fid, scene=SCENE, regime=frame.regime,
            params=dict(frame.params, viewpoint=0),
            T_world_from_camera=frame.T_world_from_camera, K=K,
            height=SIDE, width=SIDE,
            rgb_path=f"rgb/{fid}.png", depth_path=f"depth/{fid}.npy",
        ))
        features[fid] = smooth_features(SIDE // 14, SIDE // 14, CHANNELS, seed=100 + i)
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
    return manifest


def main():
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="lot_validation_"))
    build_scene(tmp)
    cfg = EvalConfig(
        experiment_name="validation_probe", renders_root=tmp, cache_root=tmp / "cache",
        output_root=tmp / "out", scenes=[SCENE], encoders=["dinov2_vitb14"],
        max_pairs_per_stratum=40, points_per_pair=64, seed=0,
        mean_feature_scenes=[SCENE],
    )
    mean_map = dataset_mean_feature_map(cfg.cache_root, "dinov2_vitb14", [SCENE])

    # -----------------------------------------------------------------
    # 4.1 DERIVATION FIRST, before any observed count is read.
    # -----------------------------------------------------------------
    print("=" * 78)
    print("4.1 EXPECTED RECORD COUNT, derived from the code path before observation")
    print("=" * 78)
    print("""
One record = one (pair, encoder, path, variant). `metric` is NOT a row
dimension: each row carries four metric COLUMNS (cosine_mean, l2_mean,
cosine_centered_mean, l2_centered_mean). PROTOCOL 3.2 asks for long format
carrying "metric name, metric value"; the implementation stores metrics wide.

Variants present per path, read from evaluate._VARIANT_NAMES and
evaluate_pair_for_encoder:
  per_point  : Oracle-Transport, No-Warp-Copy, Neighbor-Patch, Random-Patch,
               Mean-Feature                                        = 5
  splat_pool : Oracle-Transport, No-Warp-Copy, Mean-Feature        = 3
               (Neighbor-Patch and Random-Patch do not exist on this path)

So expected rows = n_pairs * n_encoders * 8, with no structural omissions:
Neighbor-Patch border records are not omitted per record, because the sampler
drops out-of-box candidates before sampling rather than emitting a short row,
and centered Mean-Feature is NOT absent: it is emitted as a finite column.
""")
    rows = evaluate_scene(cfg, SCENE, {"dinov2_vitb14": mean_map})
    pairs = {(r["context_frame_id"], r["target_frame_id"]) for r in rows}
    expected = len(pairs) * 1 * 8
    record("4.1", "record count matches the derivation", len(rows) == expected,
           f"{len(pairs)} pairs x 1 encoder x 8 = {expected} expected, {len(rows)} observed")

    by_path = Counter((r["path"], r["variant"]) for r in rows)
    print("\n  variants actually present, by path:")
    for (p, v), n in sorted(by_path.items()):
        print(f"    {p:11s} {v:17s} {n}")
    record("4.1b", "Neighbor-Patch and Random-Patch absent from splat_pool",
           ("splat_pool", "Neighbor-Patch") not in by_path
           and ("splat_pool", "Random-Patch") not in by_path,
           "PROTOCOL 3.5 runs both paths; 3.10 Figure A asks for the full null "
           "ladder. Two of five nulls exist on one path only.")

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

    record("4.3b", "sample_id column absent", "sample_id" not in rows[0],
           "PROTOCOL 3.2 requires a deterministic sample_id per correspondence, and "
           "for pair-aggregated storage the contributing sample_id set or validity "
           "bitmask persisted per record. Neither column exists.")

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
    record("4.3c", "no nonfinite scores among populated rows (n > 0)",
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
    rows2 = evaluate_scene(cfg, SCENE, {"dinov2_vitb14": mean_map})

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
    write_rows(out, rows)
    back = read_rows(out)
    record("4.3e", "parquet round trip preserves rows", len(back) == len(rows),
           f"{len(back)} rows read back from {out.name}")
    try:
        write_rows(out, rows)
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
    record("2.5b", "Mean-Feature raw score is path independent",
           abs(raw_pp - raw_sp) < 1e-6,
           f"per_point {raw_pp:.4f} vs splat_pool {raw_sp:.4f}, difference "
           f"{abs(raw_pp - raw_sp):.4f}. A single global vector compared through both "
           f"paths cannot differ; a differing value means the two paths use different "
           f"Mean-Feature objects.")

    # What object is each path's Mean-Feature prediction?
    print("\n  Mean-Feature prediction objects, read from the source:")
    print("    per_point  : correspondence.gather_value_pairs -> "
          "features_context.mean(dim=(1,2))  == the PER-CONTEXT-IMAGE mean vector")
    print("    splat_pool : evaluate.evaluate_pair_for_encoder -> "
          "mean_feature_map [C,Hp,Wp]        == a POSITION-CONDITIONED mean MAP")
    print("    centering  : evaluate.evaluate_pair_for_encoder -> "
          "mean_feature_map.mean(dim=(1,2))  == a global vector")
    spread = float(mean_map.std(dim=(1, 2)).mean())
    record("1.7", "the centering vector equals the Mean-Feature floor object",
           False,
           f"centering subtracts mean_map.mean(dim=(1,2)) (a global [C] vector) while "
           f"splat_pool Mean-Feature predicts mean_map itself (a [C,Hp,Wp] map, mean "
           f"per-channel spatial sd {spread:.4f}). PROTOCOL 3.7 requires them to be "
           f"the same vector, which is exactly why centered Mean-Feature comes out "
           f"finite instead of undefined.")

    print("\n" + "=" * 78)
    flags = [r for r in results if not r[2]]
    print(f"SUMMARY: {len(results) - len(flags)} conformant, {len(flags)} flagged, "
          f"of {len(results)} checks")
    for c, n, _, d in flags:
        print(f"  FLAG {c} {n}")
    print(f"\nprobe artifacts under {tmp}")


if __name__ == "__main__":
    main()
