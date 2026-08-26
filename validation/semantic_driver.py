"""Driver run inside a mutant directory for VALIDATION 3.6 and 3.7.

Usage: semantic_driver.py <mutant_dir> <label>

Puts the mutant's src first on sys.path, proves the loaded lot package is the
mutant's, builds the probe scene, evaluates it, and prints per-variant
aggregates plus the paired Oracle-over-No-Warp-Copy margin with a pair-level
bootstrap interval. VALIDATION 3.7 permits pair-level uncertainty when the
control is run on a single scene, which is the case here.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

mutant = Path(sys.argv[1]).resolve()
label = sys.argv[2]

sys.path.insert(0, str(mutant / "src"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

import lot  # noqa: E402

resolved = Path(lot.__file__).resolve()
if mutant not in resolved.parents:
    print(f"VOID: loaded {resolved}, not the mutant's copy", file=sys.stderr)
    raise SystemExit(2)
print(f"PROVENANCE lot.__file__ = {resolved} | inside mutant dir = True")

import probe_scene  # noqa: E402

root = Path(tempfile.mkdtemp(prefix=f"lot_sem_{label}_"))
probe_scene.build_scene(root, feature_mode="surface")
rows = probe_scene.evaluate_probe(root)

PER_POINT = "per_point"
ORACLE, FLOOR = "Oracle-Transport", "No-Warp-Copy"

# Paired margins: one per (pair, path, metric), computed on the rows of that pair.
by_pair = defaultdict(dict)
for r in rows:
    if r["n"] == 0:
        continue
    by_pair[(r["context_frame_id"], r["target_frame_id"], r["path"])][r["variant"]] = r

out = {"label": label, "metrics": {}}
for metric in ("cosine_mean", "cosine_centered_mean"):
    per_variant = defaultdict(list)
    margins = defaultdict(list)
    for (_c, _t, path), variants in by_pair.items():
        for v, row in variants.items():
            if np.isfinite(row[metric]):
                per_variant[(path, v)].append(row[metric])
        o, f = variants.get(ORACLE), variants.get(FLOOR)
        if o and f and np.isfinite(o[metric]) and np.isfinite(f[metric]):
            margins[path].append(o[metric] - f[metric])

    entry = {"variants": {}, "paired_margin": {}}
    for (path, v), vals in sorted(per_variant.items()):
        entry["variants"][f"{path}|{v}"] = {"mean": float(np.mean(vals)), "n": len(vals),
                                            "min": float(np.min(vals)), "max": float(np.max(vals))}
    rng = np.random.default_rng(7)
    for path, vals in sorted(margins.items()):
        a = np.array(vals)
        boot = np.array([rng.choice(a, a.size, replace=True).mean() for _ in range(2000)])
        entry["paired_margin"][path] = {
            "mean": float(a.mean()), "n_pairs": int(a.size),
            "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
        }
    out["metrics"][metric] = entry

print("RESULT " + json.dumps(out))
