"""Re-audit Part 3: mutation tests against the frozen source.

Fresh copy of src/lot + tests + configs per mutant under validation/mutants/ra_*.
Each mutant runs in its own subprocess via reaudit_run_one.py, which imports lot
first, records lot.__file__, and refuses to run unless it resolves inside the
mutant directory. Evidence: validation/evidence/reaudit/mutation_report.json.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTANTS = ROOT / "validation" / "mutants"
OUT = ROOT / "validation" / "evidence" / "reaudit"
OUT.mkdir(parents=True, exist_ok=True)


def build(name: str) -> Path:
    target = MUTANTS / name
    if target.exists():
        shutil.rmtree(target)
    (target / "src").mkdir(parents=True)
    shutil.copytree(ROOT / "src" / "lot", target / "src" / "lot",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "tests", target / "tests",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "configs", target / "configs")
    return target


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"mutation anchor not found in {path}:\n{old}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def mutate(name: str) -> Path:
    t = build(name)
    lot = t / "src" / "lot"
    if name == "ra_control":
        pass
    elif name == "ra_3.1_relative_pose_direction":
        edit(lot / "geometry.py",
             "    return invert_se3(T_world_from_target) @ T_world_from_context",
             "    return invert_se3(T_world_from_context) @ T_world_from_target")
    elif name == "ra_3.2_patch_mapping_half_patch":
        edit(lot / "encoders.py",
             "    return (uv_px + 0.5) / patch_size - 0.5",
             "    return (uv_px + 0.5) / patch_size")
    elif name == "ra_3.3_zbuffer_disabled":
        # Occlusion resolution removed: every kept splat wins, all surfaces mix.
        edit(lot / "transport.py",
             "    winners = z_keep <= zbuffer[lin] * (1 + TIE_RELATIVE_EPS)",
             "    winners = torch.ones_like(z_keep, dtype=torch.bool)")
    elif name == "ra_3.3b_farthest_wins":
        # The z-buffer prefers the farthest surface; the zbuffer diagnostic
        # output is left as recorded so the kill must come from features.
        edit(lot / "transport.py",
             '    zbuffer.scatter_reduce_(0, lin, z_keep, reduce="amin", include_self=True)\n'
             "    winners = z_keep <= zbuffer[lin] * (1 + TIE_RELATIVE_EPS)",
             '    zbuffer.scatter_reduce_(0, lin, z_keep, reduce="amin", include_self=True)\n'
             "    far = torch.full_like(zbuffer, -torch.inf)\n"
             '    far.scatter_reduce_(0, lin, z_keep, reduce="amax", include_self=True)\n'
             "    winners = z_keep >= far[lin] * (1 - TIE_RELATIVE_EPS)")
    elif name == "ra_3.4_unproject_ray_distance":
        edit(lot / "geometry.py",
             "    x = (uv[..., 0] - K[0, 2]) * depth / K[0, 0]\n"
             "    y = (uv[..., 1] - K[1, 2]) * depth / K[1, 1]\n"
             "    return torch.stack((x, y, depth), dim=-1)",
             "    dx = (uv[..., 0] - K[0, 2]) / K[0, 0]\n"
             "    dy = (uv[..., 1] - K[1, 2]) / K[1, 1]\n"
             "    z = depth / torch.sqrt(1 + dx * dx + dy * dy)\n"
             "    return torch.stack((dx * z, dy * z, z), dim=-1)")
    elif name == "ra_3.5_unclamped_acos":
        # The arccos-overshoot bug class VALIDATION 3.5 targets: replace the
        # 3.12 atan2 form with an UNCLAMPED acos of the trace term.
        edit(lot / "geometry.py",
             "    M = R.to(torch.float64)\n"
             "    sine = float(torch.linalg.matrix_norm(M - M.mT)) / (2.0 * math.sqrt(2.0))\n"
             "    cosine = (float(torch.diagonal(M).sum()) - 1.0) / 2.0\n"
             "    return math.degrees(math.atan2(sine, cosine))",
             "    M = R.to(torch.float64)\n"
             "    cosine = (float(torch.diagonal(M).sum()) - 1.0) / 2.0\n"
             "    return math.degrees(math.acos(cosine))")
    else:
        raise SystemExit(f"unknown mutant {name}")
    return t


NAMES = [
    "ra_control",
    "ra_3.1_relative_pose_direction",
    "ra_3.2_patch_mapping_half_patch",
    "ra_3.3_zbuffer_disabled",
    "ra_3.3b_farthest_wins",
    "ra_3.4_unproject_ray_distance",
    "ra_3.5_unclamped_acos",
]


def main() -> None:
    results = {}
    for name in NAMES:
        target = mutate(name)
        extra = []
        if name in ("ra_control", "ra_3.5_unclamped_acos"):
            # The validator-defined overshoot test runs beside the suite for the
            # clamp mutant and the control (it must pass on the control).
            shutil.copy(ROOT / "validation" / "reaudit_test_overshoot.py",
                        target / "tests" / "test_ra_overshoot.py")
            extra = ["tests/test_ra_overshoot.py"]
        cmd = [sys.executable, str(ROOT / "validation" / "reaudit_run_one.py"), str(target)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(target), timeout=1800)
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        results[name] = {
            "returncode": proc.returncode,
            "tail": tail,
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-5:]),
        }
        print(f"=== {name}: rc={proc.returncode} ===")
        print(tail)
        print()
    (OUT / "mutation_report.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
