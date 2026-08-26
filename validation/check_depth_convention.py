"""VALIDATION 2.6: depth conventions.

Two separate procedures are in play and they must not be conflated.

(a) The RENDERER convention test (PLAN Phase 1). Implemented as
    render_replica.classify_depth_convention. Re-run here on synthetic raw
    output whose convention is known by construction, so the classifier is
    checked against ground truth rather than against itself.

(b) The VGGT convention test (PROTOCOL 4.1). A different procedure entirely:
    for the first rotation-program frame of every scene, regress the per-pixel
    ratio of resampled VGGT depth to ground-truth depth against the secant of
    each pixel's angle from the optical axis. Implemented independently below
    from the protocol text. There is no implementation in src/ to compare it
    against, and no VGGT depth in the repository to run it on, so it is
    demonstrated on synthetic depth of both conventions instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

import independent as ind  # noqa: E402
from lot.render_replica import classify_depth_convention  # noqa: E402

results = []


def record(check, name, ok, detail):
    results.append((check, name, ok, detail))
    print(f"[{'PASS' if ok else 'FLAG'}] {check} {name}: {detail}")


H = W = 128
K = np.array([[70.0, 0, (W - 1) / 2], [0, 70.0, (H - 1) / 2], [0, 0, 1]], dtype=np.float64)


def ray_norm_map(h, w, K):
    """sec(angle from the optical axis) per pixel, which is ||ray|| / z."""
    uv = ind.pixel_grid(h, w)
    x = (uv[..., 0] - K[0, 2]) / K[0, 0]
    y = (uv[..., 1] - K[1, 2]) / K[1, 1]
    return np.sqrt(x * x + y * y + 1.0)


def wall(h, w, K, tilt=(0.0, 0.0), z0=3.0):
    """Planar z-depth of a plane, optionally tilted, as an affine function of x, y."""
    uv = ind.pixel_grid(h, w)
    x = (uv[..., 0] - K[0, 2]) / K[0, 0]
    y = (uv[..., 1] - K[1, 2]) / K[1, 1]
    return z0 / (1.0 - tilt[0] * x - tilt[1] * y)


# ---------------------------------------------------------------------------
# (a) the renderer's classifier, against depth whose convention is known
# ---------------------------------------------------------------------------
print("=== 2.6a renderer depth-convention classifier vs known ground truth ===")
sec = ray_norm_map(H, W, K)
Kt = torch.from_numpy(K)

for label, tilt in (("fronto-parallel", (0.0, 0.0)), ("tilted 6 deg", (0.10, 0.06))):
    planar = wall(H, W, K, tilt)
    euclid = planar * sec           # the same surface stored as ray distance
    for truth, dmap in (("planar_z", planar), ("euclidean_ray", euclid)):
        out = classify_depth_convention(torch.from_numpy(dmap), Kt)
        record("2.6a", f"{label}, stored as {truth}", out["verdict"] == truth,
               f"verdict={out['verdict']!r} spread_planar={out['spread_planar']:.3e} "
               f"spread_euclidean={out['spread_euclidean']:.3e}")

# The thresholds the verdict turns on are function defaults in source.
import inspect  # noqa: E402

# This recorded a failure when the thresholds were literals in source and the
# config file did not exist. It measures now: the signature defaults must be
# None, meaning the value is resolved from the committed config at call time,
# and that file must carry the keys.
sig = inspect.signature(classify_depth_convention)
threshold_names = ("slope_threshold", "flat_tol", "margin", "center_crop")
defaults = {
    k: v.default for k, v in sig.parameters.items()
    if k in threshold_names and v.default is not inspect._empty
}
deferred = [k for k, v in defaults.items() if v is None]

from lot.analysis_config import DEFAULT_CONFIG_PATH, load_analysis_config  # noqa: E402

config_exists = Path(DEFAULT_CONFIG_PATH).is_file()
config_keys = []
if config_exists:
    cfg_values = load_analysis_config().as_dict()
    config_keys = [
        k for k in (
            "depth_convention_slope_threshold", "depth_convention_flat_tol",
            "depth_convention_margin", "depth_convention_center_crop",
        )
        if k in cfg_values
    ]
record("2.6a-cfg", "decision thresholds come from a committed config",
       config_exists and len(config_keys) == 4 and len(deferred) == len(defaults),
       f"{DEFAULT_CONFIG_PATH.name} exists: {config_exists}; it carries "
       f"{len(config_keys)} of 4 depth-convention keys; {len(deferred)} of "
       f"{len(defaults)} signature thresholds defer to it ({defaults}).")

# ---------------------------------------------------------------------------
# (b) PROTOCOL 4.1's secant regression, implemented independently
# ---------------------------------------------------------------------------
print("\n=== 2.6b PROTOCOL 4.1 secant regression, independent implementation ===")


def secant_regression(depth_estimated, depth_gt, K, slope_threshold=0.05):
    """PROTOCOL 4.1 verbatim.

    Compute the per-pixel ratio of resampled estimated depth to ground-truth
    depth over valid pixels and regress it against the secant of each pixel's
    angle from the optical axis. A near-zero slope indicates planar z-depth; a
    positive slope tracking the secant indicates ray distance.
    """
    s = ray_norm_map(*depth_gt.shape, K).reshape(-1)
    ratio = (depth_estimated / depth_gt).reshape(-1)
    ok = np.isfinite(ratio) & np.isfinite(s) & (depth_gt.reshape(-1) > 0)
    A = np.stack([np.ones(ok.sum()), s[ok]], axis=1)
    coef, *_ = np.linalg.lstsq(A, ratio[ok], rcond=None)
    intercept, slope = coef
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "verdict": "ray_distance" if slope > slope_threshold else "planar_z",
    }


gt = wall(H, W, K, (0.05, 0.03))
for truth, est in (("planar_z", gt * 0.83), ("ray_distance", gt * 0.83 * sec)):
    out = secant_regression(est, gt, K)
    record("2.6b", f"secant regression classifies {truth}", out["verdict"] == truth,
           f"slope={out['slope']:+.4f} intercept={out['intercept']:+.4f} "
           f"verdict={out['verdict']!r}")

src_text = "\n".join(
    p.read_text(encoding="utf-8", errors="ignore") for p in (ROOT / "src" / "lot").glob("*.py")
)
has_secant = "secant" in src_text.lower() or "sec(" in src_text
record("2.6b-impl", "the PROTOCOL 4.1 secant regression exists in src/", has_secant,
       "no occurrence of a secant regression against a VGGT/GT depth ratio anywhere in "
       "src/lot. PROTOCOL 4.1 requires it to run before any alignment level.")

# Match calls, not prose. The words "resize" and "interpolate" appear in this
# codebase only inside docstrings that state nothing is resized, and counting
# those as call sites would report the opposite of the truth.
import re  # noqa: E402

resample_calls = [
    line.strip() for line in src_text.splitlines()
    if re.search(r"\b(F\.interpolate|torch\.nn\.functional\.interpolate|grid_sample|"
                 r"\.resize\(|cv2\.resize|Image\.resize|zoom\()", line)
]
record("2.6b-resample", "VGGT depth resampling to render resolution exists",
       bool(resample_calls),
       f"{len(resample_calls)} actual resampling calls in src/lot (the words resize and "
       f"interpolate occur only in docstrings stating that nothing is resized). PROTOCOL "
       f"4.1 requires nearest-neighbor resampling of VGGT depth to render resolution and "
       f"forbids bilinear. No resampling exists, so the rule is so far neither obeyed nor "
       f"broken; FINDINGS records VGGT depth exported at 518x518, the render resolution, "
       f"in which case the step may be a no-op, but nothing in code asserts that.")

print("\n=== SUMMARY ===")
flags = [r for r in results if not r[2]]
print(f"{len(results) - len(flags)} conformant, {len(flags)} flagged, of {len(results)}")
for c, n, _, _ in flags:
    print(f"  FLAG {c} {n}")
