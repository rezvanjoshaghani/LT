"""VALIDATION Part 3: mutation tests.

Each mutant is a fresh, complete copy of src/lot AND tests/ under
validation/mutants/<name>/, so it is a self-contained mini repository. This
layout is required, not cosmetic: tests/conftest.py inserts
`Path(conftest).parents[1] / "src"` at sys.path[0], which would shadow any
mutant placed on PYTHONPATH and silently test the original package. Copying
the suite alongside the mutated package makes that same conftest line resolve
to the mutant's own src.

Each mutant runs in its own fresh subprocess via run_one.py, which imports lot,
prints and asserts the resolved lot.__file__ lies inside the mutant directory,
and only then calls pytest.main() in that same process. A run without that
provenance line is void.

Nothing here writes to src/ or tests/.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MUTANTS = Path(__file__).resolve().parent / "mutants"


def build(name: str) -> Path:
    """Fresh copy of src/lot and tests/ under validation/mutants/<name>/."""
    d = MUTANTS / name
    if d.exists():
        shutil.rmtree(d)
    (d / "src").mkdir(parents=True)
    shutil.copytree(ROOT / "src" / "lot", d / "src" / "lot",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "tests", d / "tests",
                    ignore=shutil.ignore_patterns("__pycache__"))
    # Three tests load the shipped yaml by a path relative to the repo root
    # (test_shipped_configs_load, test_shipped_config_loads,
    # test_repo_configs_load). Without configs/ the control copy fails those
    # three for a reason that has nothing to do with any mutant, and a red
    # control makes every kill meaningless.
    shutil.copytree(ROOT / "configs", d / "configs")
    return d


def patch(path: Path, old: str, new: str) -> None:
    """Exact single-occurrence replacement. Raises if it is not unique."""
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"{path}: pattern occurs {text.count(old)} times, expected 1:\n{old}")
    path.write_text(text.replace(old, new), encoding="utf-8")


# ---------------------------------------------------------------------------
# The five mutants of VALIDATION 3.1 - 3.5
# ---------------------------------------------------------------------------

def m31_relative_pose_direction(d: Path) -> str:
    """3.1 Swap the relative transform direction (context-from-target)."""
    patch(
        d / "src" / "lot" / "geometry.py",
        "    return invert_se3(T_world_from_target) @ T_world_from_context",
        "    return invert_se3(T_world_from_context) @ T_world_from_target",
    )
    return "geometry.relative_pose now returns T_context_from_target"


def m32_patch_mapping_half_patch(d: Path) -> str:
    """3.2 Shift the pixel-to-patch mapping by half a patch."""
    patch(
        d / "src" / "lot" / "encoders.py",
        "    return (uv_px + 0.5) / patch_size - 0.5\n",
        "    return (uv_px + 0.5) / patch_size - 0.5 + 0.5\n",
    )
    return "encoders.pixel_to_patch_coords shifted by +0.5 patch"


def m33_zbuffer_disabled(d: Path) -> str:
    """3.3 Disable the z-buffer: last write wins instead of nearest depth."""
    patch(
        d / "src" / "lot" / "transport.py",
        '    zbuffer.scatter_reduce_(0, lin, z_keep, reduce="amin", include_self=True)\n'
        "    winners = z_keep <= zbuffer[lin] * (1 + TIE_RELATIVE_EPS)",
        '    zbuffer.scatter_reduce_(0, lin, z_keep, reduce="amin", include_self=True)\n'
        "    # MUTANT 3.3: no occlusion resolution. The last splat to touch a\n"
        "    # target pixel wins, whatever its depth.\n"
        "    last = torch.full_like(zbuffer, -1.0)\n"
        "    last.scatter_(0, lin, torch.arange(lin.numel(), dtype=last.dtype, device=last.device))\n"
        "    winners = last[lin] == torch.arange(lin.numel(), dtype=last.dtype, device=last.device)",
    )
    return "transport z-buffer bypassed; last splat written to a pixel wins"


def m34_unproject_ray_distance(d: Path) -> str:
    """3.4 Unproject treating depth as euclidean ray distance, not planar z."""
    patch(
        d / "src" / "lot" / "geometry.py",
        "    x = (uv[..., 0] - K[0, 2]) * depth / K[0, 0]\n"
        "    y = (uv[..., 1] - K[1, 2]) * depth / K[1, 1]\n"
        "    return torch.stack((x, y, depth), dim=-1)",
        "    # MUTANT 3.4: depth read as euclidean ray distance. The ray is\n"
        "    # normalized to unit length first, so the recovered z is\n"
        "    # depth / ||ray|| rather than depth.\n"
        "    ex = (uv[..., 0] - K[0, 2]) / K[0, 0]\n"
        "    ey = (uv[..., 1] - K[1, 2]) / K[1, 1]\n"
        "    norm = torch.sqrt(ex * ex + ey * ey + 1.0)\n"
        "    return torch.stack((ex * depth / norm, ey * depth / norm, depth / norm), dim=-1)",
    )
    return "geometry.unproject treats depth as ray distance"


def m35_no_arccos_clamp(d: Path) -> str:
    """3.5 Remove the arccos clamp."""
    patch(
        d / "src" / "lot" / "geometry.py",
        "    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))",
        "    return math.degrees(math.acos(cosine))",
    )
    return "geometry.rotation_angle_deg arccos clamp removed"


def m33b_farthest_wins(d: Path) -> str:
    """3.3b Invert occlusion for the pooled features only.

    The returned zbuffer stays the true per-pixel minimum, so this isolates one
    question: does any test assert that the POOLED FEATURES respect occlusion,
    as opposed to asserting on the zbuffer diagnostic?
    """
    patch(
        d / "src" / "lot" / "transport.py",
        "    winners = z_keep <= zbuffer[lin] * (1 + TIE_RELATIVE_EPS)",
        "    # MUTANT 3.3b: the FARTHEST splat wins each target pixel.\n"
        "    zfar = torch.full_like(zbuffer, -torch.inf)\n"
        '    zfar.scatter_reduce_(0, lin, z_keep, reduce="amax", include_self=True)\n'
        "    winners = z_keep >= zfar[lin] * (1 - TIE_RELATIVE_EPS)",
    )
    return "occluded (farthest) surface wins each target pixel; zbuffer output unchanged"


def m33c_zbuffer_output_amax(d: Path) -> str:
    """3.3c Full inversion: the returned zbuffer becomes the maximum depth too."""
    patch(
        d / "src" / "lot" / "transport.py",
        '    zbuffer.scatter_reduce_(0, lin, z_keep, reduce="amin", include_self=True)\n'
        "    winners = z_keep <= zbuffer[lin] * (1 + TIE_RELATIVE_EPS)",
        "    zbuffer = torch.full_like(zbuffer, -torch.inf)\n"
        '    zbuffer.scatter_reduce_(0, lin, z_keep, reduce="amax", include_self=True)\n'
        "    winners = z_keep >= zbuffer[lin] * (1 - TIE_RELATIVE_EPS)",
    )
    return "z-buffer fully inverted: farthest wins and the returned buffer holds it"


MUTANTS_SPEC = [
    ("3.1_relative_pose_direction", m31_relative_pose_direction,
     ["tests/test_geometry.py", "tests/test_correspondence.py", "tests/test_transport.py",
      "tests/test_evaluate.py", "tests/test_visibility.py", "tests/test_datasets.py"]),
    ("3.2_patch_mapping_half_patch", m32_patch_mapping_half_patch,
     ["tests/test_encoders.py", "tests/test_transport.py", "tests/test_correspondence.py",
      "tests/test_evaluate.py"]),
    ("3.3_zbuffer_disabled", m33_zbuffer_disabled,
     ["tests/test_transport.py", "tests/test_evaluate.py"]),
    ("3.3b_farthest_wins_features_only", m33b_farthest_wins,
     ["tests/test_transport.py", "tests/test_evaluate.py"]),
    ("3.3c_zbuffer_output_inverted", m33c_zbuffer_output_amax,
     ["tests/test_transport.py", "tests/test_evaluate.py"]),
    ("3.4_unproject_ray_distance", m34_unproject_ray_distance,
     ["tests/test_geometry.py", "tests/test_transport.py", "tests/test_correspondence.py",
      "tests/test_visibility.py", "tests/test_evaluate.py"]),
    ("3.5_no_arccos_clamp", m35_no_arccos_clamp,
     ["tests/test_geometry.py", "tests/test_datasets.py"]),
]


def run(mutant_dir: Path, targets: list[str], extra: list[str] | None = None) -> dict:
    """Run the targeted tests in a fresh subprocess rooted at the mutant."""
    cmd = [sys.executable, str(Path(__file__).parent / "run_one.py"), str(mutant_dir)] + targets
    if extra:
        cmd += extra
    proc = subprocess.run(cmd, cwd=str(mutant_dir), capture_output=True, text=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def main() -> None:
    MUTANTS.mkdir(parents=True, exist_ok=True)
    report = []

    # Control: an unmutated copy must be green, otherwise a kill proves nothing.
    d = build("3.0_control_unmutated")
    out = run(d, ["tests"])
    report.append({"mutant": "3.0_control_unmutated", "bug": "none (control)",
                   "targets": ["tests"], **out})

    for name, apply, targets in MUTANTS_SPEC:
        d = build(name)
        bug = apply(d)
        out = run(d, targets)
        report.append({"mutant": name, "bug": bug, "targets": targets, **out})

    # 3.5 additionally gets a validator-defined test for the overshoot trigger,
    # because VALIDATION 3.5 states plainly that clean inputs cannot kill it.
    d = MUTANTS / "3.5_no_arccos_clamp"
    shutil.copy(Path(__file__).parent / "validator_test_clamp.py",
                d / "tests" / "test_validator_clamp.py")
    out = run(d, ["tests/test_validator_clamp.py"])
    report.append({"mutant": "3.5_no_arccos_clamp (validator test)",
                   "bug": "arccos clamp removed", "targets": ["tests/test_validator_clamp.py"],
                   **out})

    # And the same validator test against the unmutated package, which must pass.
    d = build("3.5_control_with_validator_test")
    shutil.copy(Path(__file__).parent / "validator_test_clamp.py",
                d / "tests" / "test_validator_clamp.py")
    out = run(d, ["tests/test_validator_clamp.py"])
    report.append({"mutant": "3.5_control_with_validator_test",
                   "bug": "none (control)", "targets": ["tests/test_validator_clamp.py"], **out})

    path = Path(__file__).parent / "evidence" / "mutation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("MUTATION SUMMARY")
    print("=" * 78)
    for r in report:
        prov = [l for l in r["stdout"].splitlines() if l.startswith("PROVENANCE")]
        tail = [l for l in r["stdout"].splitlines()
                if " passed" in l or " failed" in l or " error" in l]
        control = "control" in r["mutant"]
        killed = r["returncode"] != 0
        verdict = ("GREEN" if not killed else "RED") if control else \
                  ("KILLED" if killed else "SURVIVED")
        print(f"\n{r['mutant']}")
        print(f"  bug      : {r['bug']}")
        print(f"  targets  : {' '.join(r['targets'])}")
        print(f"  {prov[0] if prov else 'PROVENANCE MISSING -- RUN IS VOID'}")
        print(f"  result   : {tail[-1] if tail else r['stderr'].strip()[:200]}")
        print(f"  verdict  : {verdict}")
    print(f"\nfull evidence: {path}")


if __name__ == "__main__":
    main()
