"""Phase 5 Borah integration gate. No training side effects.

The Phase 5 scientific design is frozen. This module changes none of it. Its one
job is to connect that frozen design to the real cluster-resident artifacts and
find out, before any training time is spent, whether the design's assumptions
about artifact shapes and indexing conventions are true.

Every step either passes with recorded evidence or stops. A stop names the
failing step, prints the exact evidence, and classifies the cause as one of

    missing_artifact        an input is absent or ambiguous on this machine
    implementation_bug      the code is wrong about something it controls
    frozen_design_mismatch  the frozen design disagrees with the real artifacts

The classification matters because the three have different remedies. A missing
artifact is fixed by pointing at the right path. An implementation bug is fixed
in code. Only the third may justify touching the frozen design, and even then
the change is an amendment with a written rationale, never a silent edit.

No step trains. Step 12 runs one forward pass, one loss, and one backward pass
on a real batch to prove the pipeline is finite end to end, then discards the
gradients; nothing is optimized, no checkpoint is written, and the gate asserts
afterwards that no checkpoint appeared.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

MISSING_ARTIFACT = "missing_artifact"
IMPLEMENTATION_BUG = "implementation_bug"
FROZEN_DESIGN_MISMATCH = "frozen_design_mismatch"

# Forbidden model inputs, from the Phase 5 leakage rule. Checked by name against
# whatever the batch actually carries, so a field added later is caught too.
FORBIDDEN_BATCH_FIELDS = (
    "target_rgb", "rgb_target",
    "features_target", "target_features",
    "depth_target", "target_depth", "est_target", "target_aligned",
    "pointmap", "target_pointmap",
    "flow", "optical_flow",
    "covisible", "co_visible", "covisible_mask",
    "correspondence", "landing", "uv_target",
)


class GateStop(RuntimeError):
    """A gate step failed. Carries the evidence and the classification."""

    def __init__(self, step: str, message: str, classification: str,
                 evidence: dict[str, Any] | None = None):
        super().__init__(message)
        self.step = step
        self.message = message
        self.classification = classification
        self.evidence = evidence or {}


@dataclasses.dataclass
class StepResult:
    step: str
    title: str
    passed: bool
    evidence: dict[str, Any]

    def as_row(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path, pattern: str = "**/*") -> tuple[str, int]:
    """Digest of a directory's file contents, in sorted-path order.

    Returns (digest, file_count). Sorted so the value does not depend on the
    filesystem's enumeration order, which differs between machines and would
    otherwise make the same tree hash differently on Borah and elsewhere.
    """
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    outer = hashlib.sha256()
    for path in files:
        outer.update(path.relative_to(root).as_posix().encode("utf-8"))
        outer.update(b"\0")
        outer.update(sha256_file(path).encode("ascii"))
        outer.update(b"\n")
    return outer.hexdigest(), len(files)


def environment_identity() -> dict[str, Any]:
    """What ran this. Recorded so a later rerun can be compared against it."""
    identity = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "node": platform.node(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        identity["cuda"] = torch.version.cuda
        identity["gpu"] = torch.cuda.get_device_name(0)
        identity["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return identity


# ---------------------------------------------------------------------------
# Step 2: resolve and hash the real artifacts
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RequiredArtifact:
    name: str
    path: Path
    kind: str          # "dir" or "file"
    why: str


def required_artifacts(cfg: Any) -> list[RequiredArtifact]:
    """Every input the frozen Phase 5 design needs, with why it needs it.

    Paths come from the config, never from a development assumption about where
    things sit. A path that does not resolve is a stop rather than a fallback,
    because a fallback would silently measure something other than what the pin
    names.
    """
    return [
        RequiredArtifact(
            "renders_root", Path(cfg.renders_root), "dir",
            "camera manifests, RGB, and ground-truth depth for all 18 scenes",
        ),
        RequiredArtifact(
            "cache_features", Path(cfg.cache_root), "dir",
            "frozen DINOv2 features and the VGGT depth export",
        ),
        RequiredArtifact(
            "mean_vector_dir", Path(cfg.mean_vector_dir), "dir",
            "the Phase 3 global mean vector, the frozen centering statistic",
        ),
        RequiredArtifact(
            "phase4_dir", Path(cfg.phase4_dir), "dir",
            "the accepted Phase 4 run: per-point and splat-pool supports",
        ),
        RequiredArtifact(
            "phase4_convention", Path(cfg.phase4_dir) / "evidence" / "convention_record.json",
            "file", "the single depth convention (A6), read rather than re-decided",
        ),
        RequiredArtifact(
            "phase4_eval", Path(cfg.phase4_dir) / "eval", "dir",
            "per-scene Phase 4 parquets defining V_form and the splat support",
        ),
    ]


def step2_resolve_artifacts(cfg: Any) -> StepResult:
    found: dict[str, Any] = {}
    missing: list[dict[str, str]] = []
    for artifact in required_artifacts(cfg):
        path = artifact.path
        if not path.exists():
            missing.append(
                {"name": artifact.name, "path": str(path), "why": artifact.why}
            )
            continue
        if artifact.kind == "dir" and not path.is_dir():
            missing.append(
                {"name": artifact.name, "path": str(path),
                 "why": f"{artifact.why} (exists but is not a directory)"}
            )
            continue
        entry: dict[str, Any] = {"path": str(path.resolve())}
        if artifact.kind == "file":
            entry["sha256"] = sha256_file(path)
            entry["bytes"] = path.stat().st_size
        found[artifact.name] = entry

    if missing:
        raise GateStop(
            "2", "required Phase 5 artifacts are missing or not resolvable",
            MISSING_ARTIFACT,
            {"missing": missing, "found": sorted(found)},
        )
    return StepResult("2", "resolve real Borah artifacts", True, {"artifacts": found})


def verify_scene_identities(
    cfg: Any, scenes: Sequence[str], step: str = "2"
) -> dict[str, Any]:
    """Compare every scene's live inputs against the accepted Phase 4 identity.

    This is what the pin says the gate does, for all eighteen scenes and not a
    probe subset, and it compares rather than merely records: a live value that
    disagrees with what the accepted Phase 4 run recorded is a stop.

    Per scene, the accepted identity is read from the Phase 4 parquet's own run
    record, which carries the feature-cache digest, the depth-cache digest, the
    manifest digest, and the mean-vector digest that produced the accepted
    rows. The live feature and depth caches are opened and their recorded
    digests compared to it; the manifest is re-hashed with Phase 4's own
    function and compared; the parquet itself is hashed so the receipt binds
    the bytes the supports were read from.

    The aligned context depth has no accepted stored value to compare against,
    because Phase 4 never persisted the maps, only the calibrations. It is
    recomputed through Phase 4's own code and its digest recorded, which is the
    most a gate can do for a derived artifact and is what the pin promises.
    """
    from .encoders import load_cache_meta
    from .evaluate import read_run_metadata
    from .phase4 import load_depth_archive, manifest_digest

    per_scene: dict[str, Any] = {}
    mismatches: list[dict[str, Any]] = []
    for scene in scenes:
        entry: dict[str, Any] = {}
        parquet = Path(cfg.phase4_dir) / "eval" / f"{scene}.parquet"
        if not parquet.exists():
            mismatches.append({"scene": scene, "field": "phase4_parquet",
                               "problem": f"missing: {parquet}"})
            continue
        accepted = read_run_metadata(parquet) or {}
        entry["phase4_parquet_sha256"] = sha256_file(parquet)
        entry["phase4_parquet_bytes"] = parquet.stat().st_size
        entry["phase4_commit"] = accepted.get("git_commit")

        feature_meta = load_cache_meta(cfg.cache_root, cfg.feature_encoder, scene)
        live_features = feature_meta.get("features_digest")
        entry["features_digest"] = live_features
        if live_features != accepted.get("features_digest"):
            mismatches.append({
                "scene": scene, "field": "features_digest",
                "accepted": accepted.get("features_digest"), "live": live_features,
            })

        depth_meta = load_depth_archive(cfg.cache_root, cfg.depth_encoder, scene)["meta"]
        live_depth = depth_meta.get("depth_digest")
        entry["depth_digest"] = live_depth
        if live_depth != accepted.get("depth_digest"):
            mismatches.append({
                "scene": scene, "field": "depth_digest",
                "accepted": accepted.get("depth_digest"), "live": live_depth,
            })

        live_manifest = manifest_digest(Path(cfg.renders_root) / scene)
        entry["manifest_digest"] = live_manifest
        if live_manifest != accepted.get("manifest_digest"):
            mismatches.append({
                "scene": scene, "field": "manifest_digest",
                "accepted": accepted.get("manifest_digest"), "live": live_manifest,
            })
        entry["accepted_mean_vector_digest"] = accepted.get("mean_vector_digest")
        per_scene[scene] = entry

    if mismatches:
        raise GateStop(
            step,
            f"{len(mismatches)} live input(s) disagree with the accepted Phase 4 "
            "identity; the gate would be verifying different bytes from the ones "
            "the accepted supports were read from",
            FROZEN_DESIGN_MISMATCH,
            {"mismatches": mismatches, "scenes_checked": len(per_scene)},
        )
    return per_scene


def aligned_depth_digest(inputs: Any) -> str:
    """Digest of one scene's recomputed aligned context depth, frame order fixed."""
    digest = hashlib.sha256()
    for frame_id in sorted(inputs.est_maps):
        digest.update(str(frame_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(inputs.est_maps[frame_id]).tobytes())
        calibration = inputs.calibrations[frame_id]
        digest.update(repr(dataclasses.astuple(calibration)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_mean_vector_identity(
    center: Any, per_scene: dict[str, Any], step: str = "5"
) -> dict[str, Any]:
    """The centering statistic must be the one the accepted rows were scored with."""
    from .evaluate import vector_digest

    live = vector_digest(center.detach().cpu().numpy())
    accepted = {
        scene: entry.get("accepted_mean_vector_digest")
        for scene, entry in per_scene.items()
    }
    disagreeing = {s: a for s, a in accepted.items() if a is not None and a != live}
    if disagreeing:
        raise GateStop(
            step, "the loaded mean vector is not the one the accepted Phase 4 "
            "rows were centered with",
            FROZEN_DESIGN_MISMATCH,
            {"live": live, "accepted_by_scene": disagreeing},
        )
    return {"mean_vector_digest": live, "scenes_agreeing": len(accepted)}


# ---------------------------------------------------------------------------
# Step 3: verify the real schemas
# ---------------------------------------------------------------------------

def describe_tensor(value: Any) -> dict[str, Any]:
    """Shape, dtype, and finiteness of whatever a loader actually returned."""
    if isinstance(value, torch.Tensor):
        finite = bool(torch.isfinite(value).all()) if value.is_floating_point() else True
        return {
            "type": "torch.Tensor", "shape": list(value.shape),
            "dtype": str(value.dtype), "device": str(value.device),
            "all_finite": finite,
        }
    if isinstance(value, np.ndarray):
        finite = bool(np.isfinite(value).all()) if value.dtype.kind == "f" else True
        return {
            "type": "numpy.ndarray", "shape": list(value.shape),
            "dtype": str(value.dtype), "all_finite": finite,
        }
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def check_schema(
    name: str, value: Any, expected_shape: Sequence[int | None] | None = None,
    expected_dtypes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare a real artifact against what the frozen loaders assume.

    A None entry in expected_shape means "any size on this axis". A mismatch is
    a frozen-design mismatch and not an implementation bug, because the code is
    doing what the design says and the design is what disagrees with the data.
    """
    described = describe_tensor(value)
    if expected_shape is not None:
        shape = described.get("shape")
        if shape is None or len(shape) != len(expected_shape):
            raise GateStop(
                "3", f"{name}: rank {shape} does not match the frozen "
                f"expectation {list(expected_shape)}",
                FROZEN_DESIGN_MISMATCH, {"observed": described},
            )
        for axis, (got, want) in enumerate(zip(shape, expected_shape)):
            if want is not None and got != want:
                raise GateStop(
                    "3", f"{name}: axis {axis} is {got}, frozen design expects {want}",
                    FROZEN_DESIGN_MISMATCH, {"observed": described,
                                             "expected_shape": list(expected_shape)},
                )
    if expected_dtypes is not None and described.get("dtype") not in expected_dtypes:
        raise GateStop(
            "3", f"{name}: dtype {described.get('dtype')} is not one of "
            f"{list(expected_dtypes)}",
            FROZEN_DESIGN_MISMATCH, {"observed": described},
        )
    return described


# ---------------------------------------------------------------------------
# Step 5 and 12: the leakage assertion on a real batch
# ---------------------------------------------------------------------------

def assert_no_forbidden_fields(batch: Any, where: str) -> dict[str, Any]:
    """Nothing reachable through the predictor's batch may be target content.

    Checked against the object the model is actually handed, by name, so a field
    someone adds later is caught by the same assertion rather than needing a new
    one. Naming is the only handle available here, which is why lot.predictors
    also enforces the boundary structurally through its signature.
    """
    if dataclasses.is_dataclass(batch):
        names = [f.name for f in dataclasses.fields(batch)]
    elif isinstance(batch, dict):
        names = sorted(batch)
    else:
        names = [n for n in dir(batch) if not n.startswith("_")]

    offenders = sorted(
        {name for name in names for bad in FORBIDDEN_BATCH_FIELDS if bad in name}
    )
    if offenders:
        raise GateStop(
            "5", f"{where}: forbidden target-content fields reachable in the batch: "
            f"{offenders}",
            IMPLEMENTATION_BUG,
            {"offending_fields": offenders, "all_fields": names},
        )
    return {"fields": names, "forbidden_found": []}


def model_visible_fields() -> tuple[str, ...]:
    """Exactly what reaches the network, read off the forward signature."""
    import inspect

    from .predictors import PredictWithDepth

    params = list(inspect.signature(PredictWithDepth.forward).parameters)
    return tuple(p for p in params if p != "self")


# ---------------------------------------------------------------------------
# Step 9: splat-pool symmetry, re-verified statically at runtime
# ---------------------------------------------------------------------------

def splat_symmetry_evidence() -> dict[str, Any]:
    """Re-derive the audit that let Phase 5 reuse Phase 4's operational arm.

    Repeated here rather than trusted from the design document, because the
    document records a conclusion and this records the file, the lines, and the
    argument names as they are at the commit actually running.
    """
    import ast
    import inspect

    from . import phase4

    source_path = Path(inspect.getfile(phase4))
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()

    transport_calls: list[dict[str, Any]] = []
    target_depth_uses: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in ("transport_plan", "apply_transport_plan", "splat_plan_detail"):
                first = node.args[0] if node.args else None
                transport_calls.append({
                    "line": node.lineno,
                    "call": name,
                    "first_arg": ast.unparse(first) if first is not None else None,
                    "source": lines[node.lineno - 1].strip(),
                })
        if isinstance(node, ast.Attribute) and node.attr in ("est_target",):
            target_depth_uses.append({
                "line": node.lineno, "source": lines[node.lineno - 1].strip(),
            })

    # A transport call whose depth argument mentions the target estimate would
    # break the symmetry the secondary comparison depends on.
    tainted = [
        call for call in transport_calls
        if call["first_arg"] and (
            "est_target" in call["first_arg"] or "target_aligned" in call["first_arg"]
        )
    ]
    if tainted:
        raise GateStop(
            "9", "the operational splat-pool arm consumes target-frame estimated "
            "depth; the secondary comparison is not information symmetric and a "
            "context-lift splat-pool comparator must be defined before it is used",
            FROZEN_DESIGN_MISMATCH,
            {"tainted_calls": tainted, "all_transport_calls": transport_calls},
        )
    return {
        "source_file": str(source_path),
        "transport_calls": transport_calls,
        "target_estimate_uses": target_depth_uses,
        "verdict": "context-side only; no transport call receives the target estimate",
    }


# ---------------------------------------------------------------------------
# Step 11: folds against the real scene inventory
# ---------------------------------------------------------------------------

def check_folds_against_inventory(renders_root: Path) -> dict[str, Any]:
    from .phase5_folds import REPLICA_SCENES, fold_digest, frozen_folds

    present = sorted(
        p.name for p in Path(renders_root).iterdir()
        if p.is_dir() and (p / "manifest.json").exists()
    )
    expected = sorted(REPLICA_SCENES)
    absent = sorted(set(expected) - set(present))
    unexpected = sorted(set(present) - set(expected))
    if absent:
        raise GateStop(
            "11", f"scenes named by the frozen folds are absent on disk: {absent}",
            MISSING_ARTIFACT, {"absent": absent, "present": present},
        )
    if unexpected:
        raise GateStop(
            "11", f"scenes exist on disk that the frozen folds do not name: "
            f"{unexpected}. The split hash covers a different inventory than the "
            "one present, so fold membership is not well defined.",
            FROZEN_DESIGN_MISMATCH, {"unexpected": unexpected, "expected": expected},
        )

    folds = frozen_folds()
    test_counts: dict[str, int] = {scene: 0 for scene in expected}
    for fold in folds:
        overlap_tv = set(fold.train) & set(fold.val)
        overlap_tt = set(fold.train) & set(fold.test)
        overlap_vt = set(fold.val) & set(fold.test)
        if overlap_tv or overlap_tt or overlap_vt:
            raise GateStop(
                "11", f"fold {fold.index} has overlapping splits",
                IMPLEMENTATION_BUG,
                {"train_val": sorted(overlap_tv), "train_test": sorted(overlap_tt),
                 "val_test": sorted(overlap_vt)},
            )
        for scene in fold.test:
            test_counts[scene] += 1

    wrong = {scene: n for scene, n in test_counts.items() if n != 1}
    if wrong:
        raise GateStop(
            "11", f"scenes are not test exactly once: {wrong}",
            IMPLEMENTATION_BUG, {"test_counts": test_counts},
        )
    return {
        "n_scenes": len(present),
        "fold_digest": fold_digest(folds),
        "folds": [
            {"index": f.index, "train": list(f.train), "val": list(f.val),
             "test": list(f.test)}
            for f in folds
        ],
    }


# ---------------------------------------------------------------------------
# Step 14: the test seal
# ---------------------------------------------------------------------------

def check_test_seal() -> dict[str, Any]:
    """Prove the seal by trying to break it, not by reading the code.

    Three claims: training accepts its own scenes, checkpoint selection accepts
    validation scenes, and a test scene reaching the training loop raises. The
    third is the one that matters, and it is demonstrated rather than asserted.
    """
    from .phase5_folds import frozen_folds
    from .train import assert_fold_is_sealed

    fold = frozen_folds()[0]
    evidence: dict[str, Any] = {"fold": fold.index}

    assert_fold_is_sealed(fold, fold.train, "gate: train sees train")
    evidence["train_accepts_train"] = True
    assert_fold_is_sealed(fold, fold.val, "gate: selection sees validation")
    evidence["selection_accepts_val"] = True

    try:
        assert_fold_is_sealed(fold, [fold.test[0]], "gate: test must not reach training")
    except ValueError as error:
        evidence["test_rejected"] = True
        evidence["test_rejection_message"] = str(error)
    else:
        raise GateStop(
            "14", "a test scene reached the training loop without raising; the "
            "sealed test set is not enforced and every trained model would be "
            "invalid under Stream S step 9",
            IMPLEMENTATION_BUG, evidence,
        )
    return evidence


# ---------------------------------------------------------------------------
# Step 13: the resource probe
# ---------------------------------------------------------------------------

def resource_probe(
    forward: Callable[[], Any], backward: Callable[[Any], None], device: str
) -> dict[str, Any]:
    """Time one forward and one backward, and record peak memory.

    Nothing is optimized. The numbers exist to answer one question before the
    array is submitted: does the frozen batch configuration fit. If it does not,
    the gate stops and reports the mismatch rather than quietly shrinking the
    effective batch, which would change the frozen training configuration in
    response to a resource limit.
    """
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    output = forward()
    if use_cuda:
        torch.cuda.synchronize()
    forward_s = time.perf_counter() - start

    start = time.perf_counter()
    backward(output)
    if use_cuda:
        torch.cuda.synchronize()
    backward_s = time.perf_counter() - start

    probe = {
        "device": device,
        "forward_seconds": round(forward_s, 4),
        "backward_seconds": round(backward_s, 4),
    }
    if use_cuda:
        probe["gpu"] = torch.cuda.get_device_name(0)
        probe["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        probe["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        probe["total_memory_bytes"] = int(
            torch.cuda.get_device_properties(0).total_memory
        )
    return probe


def assert_no_checkpoint_written(run_dir: Path, before: set[Path]) -> dict[str, Any]:
    """The gate must have no training side effects. Verified, not promised."""
    after = set(Path(run_dir).glob("**/*.pt")) if Path(run_dir).exists() else set()
    created = sorted(str(p) for p in after - before)
    if created:
        raise GateStop(
            "12", f"the integration gate wrote checkpoints: {created}. The gate "
            "must have no training side effects.",
            IMPLEMENTATION_BUG, {"created": created},
        )
    return {"checkpoints_before": len(before), "checkpoints_after": len(after)}


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GateReport:
    passed: bool
    steps: list[StepResult]
    failure: dict[str, Any] | None
    environment: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "passed": self.passed,
                "environment": self.environment,
                "steps": [s.as_row() for s in self.steps],
                "failure": self.failure,
            },
            indent=2, sort_keys=True, default=str,
        )


def run_steps(steps: Iterable[tuple[str, str, Callable[[], dict[str, Any]]]]) -> GateReport:
    """Run gate steps in order, stopping at the first failure.

    Stopping rather than collecting every failure is deliberate. A later step
    that runs on an artifact an earlier step could not verify would produce
    evidence about nothing, and a list of ten consequential failures obscures
    the one that caused them.
    """
    results: list[StepResult] = []
    environment = environment_identity()
    for number, title, action in steps:
        try:
            evidence = action()
        except GateStop as stop:
            results.append(StepResult(stop.step, title, False, stop.evidence))
            return GateReport(
                passed=False, steps=results,
                failure={
                    "step": stop.step, "title": title, "message": stop.message,
                    "classification": stop.classification, "evidence": stop.evidence,
                },
                environment=environment,
            )
        except Exception as error:  # noqa: BLE001
            # An unexpected exception is an implementation bug by definition:
            # the gate is supposed to know what can fail and say so.
            results.append(StepResult(number, title, False, {"error": repr(error)}))
            return GateReport(
                passed=False, steps=results,
                failure={
                    "step": number, "title": title, "message": str(error),
                    "classification": IMPLEMENTATION_BUG,
                    "evidence": {"exception": repr(error)},
                },
                environment=environment,
            )
        results.append(StepResult(number, title, True, evidence))
    return GateReport(passed=True, steps=results, failure=None, environment=environment)


def format_report(report: GateReport) -> str:
    lines: list[str] = []
    for step in report.steps:
        mark = "PASS" if step.passed else "FAIL"
        lines.append(f"  [{mark}] step {step.step}: {step.title}")
    lines.append("")
    if report.passed:
        lines.append("PHASE 5 BORAH INTEGRATION GATE: PASS")
        lines.append("Next allowed action: tiny-subset overfit gate.")
    else:
        failure = report.failure or {}
        lines.append("PHASE 5 BORAH INTEGRATION GATE: FAIL")
        lines.append(f"  failing step   {failure.get('step')}: {failure.get('title')}")
        lines.append(f"  classification {failure.get('classification')}")
        lines.append(f"  message        {failure.get('message')}")
        evidence = failure.get("evidence") or {}
        if evidence:
            lines.append("  evidence:")
            for key, value in sorted(evidence.items()):
                rendered = json.dumps(value, indent=2, default=str).splitlines()
                lines.append(f"    {key}: {rendered[0]}")
                lines.extend(f"    {line}" for line in rendered[1:])
        lines.append("")
        lines.append("Training is not permitted until this gate passes.")
    return "\n".join(lines)
