"""Re-audit Part 3, semantic negative controls 3.6 and 3.7, against frozen src.

Reuses the committed probe scene and driver (validation/probe_scene.py,
validation/semantic_driver.py), but builds fresh mutants of the CURRENT source
under validation/mutants/ra_sem_* and writes evidence only under
validation/evidence/reaudit/. Verdict criteria are VALIDATION.md 3.6/3.7's.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "validation"
MUTANTS = HERE / "mutants"
OUT = HERE / "evidence" / "reaudit"
OUT.mkdir(parents=True, exist_ok=True)


def build(name: str) -> Path:
    target = MUTANTS / name
    if target.exists():
        shutil.rmtree(target)
    (target / "src").mkdir(parents=True)
    shutil.copytree(ROOT / "src" / "lot", target / "src" / "lot",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "configs", target / "configs")
    return target


def patch(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"anchor not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def run(d: Path, label: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(HERE / "semantic_driver.py"), str(d), label],
        cwd=str(d), capture_output=True, text=True, timeout=3600,
    )
    prov = [l for l in proc.stdout.splitlines() if l.startswith("PROVENANCE")]
    res = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    if not prov or not res:
        raise SystemExit(f"{label}: void or failed\n{proc.stdout[-2500:]}\n{proc.stderr[-2500:]}")
    print(f"  {prov[0]}")
    return json.loads(res[0][len("RESULT "):])


def main() -> None:
    print("baseline (unmutated current source), surface-attached features")
    base = run(build("ra_sem_baseline"), "baseline")

    d = build("ra_3.6_leak_target")
    patch(
        d / "src" / "lot" / "correspondence.py",
        '        "warp": sample_features_bilinear(features_context, samples.uv_context_warp, patch_size),',
        '        "warp": sample_features_bilinear(features_target, samples.uv_target, patch_size),',
    )
    leak = run(d, "leak")

    d = build("ra_3.7_shuffle")
    patch(
        d / "src" / "lot" / "correspondence.py",
        "    return CorrespondenceSamples(\n        sample_id=ids,",
        "    if uv_warp.shape[0] > 1:\n"
        "        _perm = np.argsort(derived_draw(ids, SELECTION_SALT, 1 << 62), kind='stable')\n"
        "        uv_warp = uv_warp[torch.from_numpy(_perm.copy()).to(uv_warp.device)]\n"
        "    return CorrespondenceSamples(\n        sample_id=ids,",
    )
    shuffle = run(d, "shuffle")

    (OUT / "semantic_controls.json").write_text(
        json.dumps({"baseline": base, "leak": leak, "shuffle": shuffle}, indent=2)
    )

    print("\n3.6 target leak, signature: per-point Oracle cosine == 1 within 1e-6, raw and centered")
    ok36 = True
    for metric in ("cosine_mean", "cosine_centered_mean"):
        k = "per_point|Oracle-Transport"
        b = base["metrics"][metric]["variants"][k]
        l = leak["metrics"][metric]["variants"][k]
        hit = abs(l["mean"] - 1.0) < 1e-6 and abs(l["min"] - 1.0) < 1e-6
        ok36 &= hit
        print(f"  {metric:24s} baseline {b['mean']:+.6f} -> leaked {l['mean']:+.9f} "
              f"(min {l['min']:+.9f})  {'SIGNATURE PRESENT' if hit else 'ABSENT'}")
    print(f"  3.6 verdict: {'PASS' if ok36 else 'FAIL'}")

    print("\n3.7 shuffle, criterion: >=50% of the paired Oracle margin destroyed, degradation clear")
    print("    (single scene, so pair-level bootstrap per VALIDATION 3.7)")
    ok37 = True
    for metric in ("cosine_mean", "cosine_centered_mean"):
        b = base["metrics"][metric]["paired_margin"]["per_point"]
        s = shuffle["metrics"][metric]["paired_margin"]["per_point"]
        destroyed = (b["mean"] - s["mean"]) / b["mean"] if b["mean"] else float("nan")
        clear = s["ci_hi"] < b["ci_lo"]
        usable = b["mean"] > 0.01 and b["ci_lo"] > 0.0
        hit = usable and destroyed >= 0.5 and clear
        ok37 &= hit
        print(f"  {metric:24s} baseline {b['mean']:+.4f} [{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}] "
              f"n={b['n_pairs']} | shuffled {s['mean']:+.4f} [{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}] "
              f"| destroyed {destroyed:.0%} disjoint={clear} -> {'MET' if hit else 'NOT MET'}")
    print(f"  3.7 verdict: {'PASS' if ok37 else 'FAIL'}")
    print(f"\nevidence -> {OUT / 'semantic_controls.json'}")


if __name__ == "__main__":
    main()
