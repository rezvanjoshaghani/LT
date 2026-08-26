"""VALIDATION 3.6 and 3.7: semantic negative controls.

These are not mutation kills. No existing test is expected to fail. They
deliberately violate a scientific assumption and check that the pre-registered
pathological signature appears. A failure means the analysis is unexpectedly
insensitive to the violation.

Both are run against the synthetic probe scene of validation/probe_scene.py,
because the repository ships no rendered data. That limits 3.7: scene-level
bootstrap is impossible with one scene, and VALIDATION 3.7 anticipates this by
allowing pair-level uncertainty in that case.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mutate import build, patch  # noqa: E402


def m36_leak_target(d: Path) -> str:
    """3.6 Substitute the true target feature into Oracle-Transport's prediction."""
    patch(
        d / "src" / "lot" / "correspondence.py",
        '    mean = features_context.to(out["warp"].dtype).mean(dim=(1, 2))',
        '    # CONTROL 3.6: the prediction IS the target. Raw cosine must become\n'
        '    # exactly 1, and centered cosine must too wherever it is defined.\n'
        '    out["warp"] = out["target"].clone()\n'
        '    mean = features_context.to(out["warp"].dtype).mean(dim=(1, 2))',
    )
    return "Oracle-Transport per-point prediction replaced by the true target feature"


def m37_shuffle_correspondences(d: Path) -> str:
    """3.7 Permute the warp locations within a pair."""
    patch(
        d / "src" / "lot" / "correspondence.py",
        "    return CorrespondenceSamples(\n        uv_target=uv_target,\n        uv_context_warp=uv_warp,",
        "    # CONTROL 3.7: correspondence identity destroyed by permuting the warp\n"
        "    # locations within this pair. The set of read locations is unchanged;\n"
        "    # only which target each is paired with changes.\n"
        "    if uv_warp.shape[0] > 1:\n"
        "        perm = torch.randperm(uv_warp.shape[0], generator=generator).to(uv_warp.device)\n"
        "        uv_warp = uv_warp[perm]\n"
        "    return CorrespondenceSamples(\n        uv_target=uv_target,\n        uv_context_warp=uv_warp,",
    )
    return "warp locations permuted within each pair"


def run(d: Path, label: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(HERE / "semantic_driver.py"), str(d), label],
        cwd=str(d), capture_output=True, text=True,
    )
    prov = [l for l in proc.stdout.splitlines() if l.startswith("PROVENANCE")]
    res = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    if not prov:
        raise SystemExit(f"{label}: provenance missing, run is void\n{proc.stderr[-2000:]}")
    if not res:
        raise SystemExit(f"{label}: no result\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    print(f"  {prov[0]}")
    return json.loads(res[0][len("RESULT "):])


def main():
    print("=" * 78)
    print("Building baseline and the two semantic controls")
    print("=" * 78)
    base = run(build("3.6_baseline_unmutated"), "baseline")

    d = build("3.6_leak_target")
    bug36 = m36_leak_target(d)
    leak = run(d, "leak")

    d = build("3.7_shuffle_correspondences")
    bug37 = m37_shuffle_correspondences(d)
    shuffle = run(d, "shuffle")

    (HERE / "evidence").mkdir(exist_ok=True)
    (HERE / "evidence" / "semantic_controls.json").write_text(
        json.dumps({"baseline": base, "leak": leak, "shuffle": shuffle}, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("3.6 TARGET LEAK -- pre-registered signature: per-point Oracle-Transport")
    print("    cosine becomes 1 within 1e-6, raw and (where defined) centered.")
    print(f"    bug: {bug36}")
    print("=" * 78)
    ok36 = True
    for metric in ("cosine_mean", "cosine_centered_mean"):
        k = "per_point|Oracle-Transport"
        b = base["metrics"][metric]["variants"][k]
        l = leak["metrics"][metric]["variants"][k]
        hit = abs(l["mean"] - 1.0) < 1e-6 and abs(l["min"] - 1.0) < 1e-6
        ok36 &= hit
        print(f"  {metric:24s} baseline {b['mean']:+.6f}  leaked {l['mean']:+.6f} "
              f"(min {l['min']:+.9f}, max {l['max']:+.9f}, n={l['n']})  "
              f"-> {'SIGNATURE PRESENT' if hit else 'SIGNATURE ABSENT'}")
    print(f"  verdict: {'PASS -- the leak is unambiguous and the shipped path is not leaking' if ok36 else 'FAIL -- ambiguous inflation, not a clean control'}")

    print("\n" + "=" * 78)
    print("3.7 CORRESPONDENCE SHUFFLE -- pre-registered kill criterion: shuffling")
    print("    destroys at least half the paired Oracle-over-No-Warp-Copy margin,")
    print("    with the degradation statistically clear.")
    print(f"    bug: {bug37}")
    print("=" * 78)
    ok37 = True
    for metric in ("cosine_mean", "cosine_centered_mean"):
        for path in ("per_point",):
            b = base["metrics"][metric]["paired_margin"][path]
            s = shuffle["metrics"][metric]["paired_margin"][path]
            destroyed = (b["mean"] - s["mean"]) / b["mean"] if b["mean"] else float("nan")
            clear = s["ci_hi"] < b["ci_lo"]
            # Precondition. "Destroys at least half the margin" is only a test
            # when there is a margin to destroy. If the baseline Oracle margin
            # is not clearly positive, the control is inconclusive rather than
            # failed, and saying otherwise would report a property of the probe
            # scene as a property of the pipeline.
            usable = b["mean"] > 0.01 and b["ci_lo"] > 0.0
            hit = usable and destroyed >= 0.5 and clear
            ok37 &= hit
            print(f"  {metric:24s} {path}")
            print(f"    baseline margin {b['mean']:+.4f}  95% CI [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]  n_pairs={b['n_pairs']}")
            print(f"    shuffled margin {s['mean']:+.4f}  95% CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]  n_pairs={s['n_pairs']}")
            print(f"    fraction of margin destroyed = {destroyed:.1%}; intervals disjoint = {clear}")
            if not usable:
                print("    -> INCONCLUSIVE: baseline margin is not clearly positive, so "
                      "there is no margin for the shuffle to destroy")
            else:
                print(f"    -> {'CRITERION MET' if hit else 'CRITERION NOT MET'}")
    print(f"  verdict: {'PASS -- correspondence identity is load bearing' if ok37 else 'NOT ESTABLISHED -- see the per-line notes above'}")
    print(f"\nevidence: {HERE / 'evidence' / 'semantic_controls.json'}")


if __name__ == "__main__":
    main()
