"""The Borah integration gate's machinery: does it actually stop, and on what.

A gate that cannot fail is not a gate. These tests drive each stop condition the
gate is responsible for and check that it fires, names the right step, and
classifies the cause correctly, because the classification is what tells a
reader whether to fix a path, fix the code, or open an amendment.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from lot.phase5_check import (
    FROZEN_DESIGN_MISMATCH,
    IMPLEMENTATION_BUG,
    MISSING_ARTIFACT,
    GateReport,
    GateStop,
    StepResult,
    assert_no_checkpoint_written,
    assert_no_forbidden_fields,
    check_folds_against_inventory,
    check_schema,
    check_test_seal,
    describe_tensor,
    environment_identity,
    format_report,
    model_visible_fields,
    run_steps,
    sha256_tree,
    splat_symmetry_evidence,
    step2_resolve_artifacts,
)
from lot.phase5_folds import REPLICA_SCENES, frozen_folds


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

def test_run_steps_stops_at_the_first_failure():
    ran: list[str] = []

    def ok(name):
        def action():
            ran.append(name)
            return {"name": name}
        return action

    def boom():
        ran.append("b")
        raise GateStop("2", "no artifacts", MISSING_ARTIFACT, {"missing": ["x"]})

    report = run_steps([
        ("1", "first", ok("a")),
        ("2", "second", boom),
        ("3", "third", ok("c")),
    ])
    assert not report.passed
    assert ran == ["a", "b"], "a later step ran after a stop"
    assert report.failure["step"] == "2"
    assert report.failure["classification"] == MISSING_ARTIFACT
    assert report.failure["evidence"] == {"missing": ["x"]}
    assert len(report.steps) == 2


def test_an_unexpected_exception_is_classified_as_an_implementation_bug():
    """The gate is meant to know what can fail. A surprise is its own bug."""
    def surprise():
        raise KeyError("frame_id")

    report = run_steps([("1", "first", surprise)])
    assert not report.passed
    assert report.failure["classification"] == IMPLEMENTATION_BUG
    assert "frame_id" in report.failure["evidence"]["exception"]


def test_a_passing_run_reports_the_permitted_next_action():
    report = run_steps([("1", "first", lambda: {"ok": True})])
    assert report.passed
    text = format_report(report)
    assert "PHASE 5 BORAH INTEGRATION GATE: PASS" in text
    assert "Next allowed action: tiny-subset overfit gate." in text


def test_a_failing_run_prints_step_classification_and_evidence():
    def boom():
        raise GateStop("7", "support moved", IMPLEMENTATION_BUG, {"before": 10, "after": 9})

    text = format_report(run_steps([("7", "support", boom)]))
    assert "PHASE 5 BORAH INTEGRATION GATE: FAIL" in text
    assert "failing step   7" in text
    assert IMPLEMENTATION_BUG in text
    assert "support moved" in text
    assert "before" in text and "after" in text
    assert "Training is not permitted until this gate passes." in text


def test_the_report_serializes():
    report = run_steps([("1", "first", lambda: {"n": 3})])
    payload = json.loads(report.to_json())
    assert payload["passed"] is True
    assert payload["steps"][0]["evidence"]["n"] == 3
    assert "environment" in payload


# ---------------------------------------------------------------------------
# Step 2, exercised against a real (local) filesystem
# ---------------------------------------------------------------------------

def test_missing_artifacts_stop_with_paths_and_reasons(tmp_path):
    from lot.phase5 import load_phase5_config

    cfg = dataclasses.replace(
        load_phase5_config(Path("configs/phase5.yaml")),
        renders_root=str(tmp_path / "nope"),
        cache_root=str(tmp_path / "also_nope"),
        mean_vector_dir=str(tmp_path / "gone"),
        phase4_dir=str(tmp_path / "absent"),
    )
    with pytest.raises(GateStop) as caught:
        step2_resolve_artifacts(cfg)
    stop = caught.value
    assert stop.step == "2"
    assert stop.classification == MISSING_ARTIFACT
    names = {m["name"] for m in stop.evidence["missing"]}
    assert {"renders_root", "cache_features", "phase4_dir"} <= names
    # Every missing entry says why the phase needs it, not only that it is gone.
    assert all(m["why"] for m in stop.evidence["missing"])


def test_a_file_where_a_directory_belongs_is_a_missing_artifact(tmp_path):
    from lot.phase5 import load_phase5_config

    impostor = tmp_path / "renders"
    impostor.write_text("not a directory", encoding="utf-8")
    cfg = dataclasses.replace(
        load_phase5_config(Path("configs/phase5.yaml")), renders_root=str(impostor)
    )
    with pytest.raises(GateStop) as caught:
        step2_resolve_artifacts(cfg)
    assert "not a directory" in json.dumps(caught.value.evidence)


# ---------------------------------------------------------------------------
# Step 3, the schema comparison
# ---------------------------------------------------------------------------

def test_schema_shape_mismatch_is_a_frozen_design_mismatch():
    with pytest.raises(GateStop) as caught:
        check_schema("dino", torch.zeros(768, 37, 37), expected_shape=(768, 40, 40))
    assert caught.value.classification == FROZEN_DESIGN_MISMATCH
    assert "axis 1" in caught.value.message


def test_schema_rank_mismatch_is_reported_as_rank():
    with pytest.raises(GateStop) as caught:
        check_schema("dino", torch.zeros(768, 37), expected_shape=(768, 37, 37))
    assert "rank" in caught.value.message


def test_schema_dtype_mismatch_stops():
    with pytest.raises(GateStop) as caught:
        check_schema("depth", torch.zeros(4, 4, dtype=torch.float64),
                     expected_dtypes=("torch.float32",))
    assert caught.value.classification == FROZEN_DESIGN_MISMATCH


def test_schema_accepts_a_wildcard_axis():
    described = check_schema("depth", torch.zeros(518, 518), expected_shape=(None, 518))
    assert described["shape"] == [518, 518]


def test_describe_tensor_reports_finiteness():
    assert describe_tensor(torch.zeros(3))["all_finite"] is True
    assert describe_tensor(torch.tensor([float("nan")]))["all_finite"] is False


# ---------------------------------------------------------------------------
# Steps 5 and 12, the leakage assertion
# ---------------------------------------------------------------------------

def test_a_forbidden_field_on_the_batch_is_an_implementation_bug():
    @dataclasses.dataclass
    class Leaky:
        features_context: int = 0
        target_depth: int = 0

    with pytest.raises(GateStop) as caught:
        assert_no_forbidden_fields(Leaky(), "test batch")
    assert caught.value.classification == IMPLEMENTATION_BUG
    assert "target_depth" in caught.value.evidence["offending_fields"]


def test_the_real_training_example_carries_no_forbidden_field():
    from lot.train import TrainingExample

    names = [f.name for f in dataclasses.fields(TrainingExample)]
    fake = TrainingExample(**{
        n: (torch.zeros(1) if "features" in n or "depth" in n or "camera" in n
            or "coords" in n or "centered" in n or "support" in n or "valid" in n
            else "x")
        for n in names
    })
    evidence = assert_no_forbidden_fields(fake, "real example")
    assert evidence["forbidden_found"] == []


def test_model_visible_fields_are_context_and_camera_only():
    fields = model_visible_fields()
    assert fields == (
        "features_context", "depth_context_aligned", "camera", "context_valid"
    )


# ---------------------------------------------------------------------------
# Step 9, run against the real Phase 4 source
# ---------------------------------------------------------------------------

def test_splat_symmetry_holds_in_the_real_phase4_source():
    """Not a fixture. This parses the shipped lot/phase4.py at this commit."""
    evidence = splat_symmetry_evidence()
    assert evidence["verdict"].startswith("context-side only")
    calls = evidence["transport_calls"]
    assert calls, "the audit found no transport calls at all, so it proves nothing"
    for call in calls:
        assert "est_target" not in (call["first_arg"] or "")
        assert "target_aligned" not in (call["first_arg"] or "")
    # The target estimate is used somewhere in Phase 4 (the per-point path), so
    # a blanket absence would mean the audit was looking at the wrong file.
    assert evidence["target_estimate_uses"], "expected per-point uses to exist"


# ---------------------------------------------------------------------------
# Step 11, folds against an inventory
# ---------------------------------------------------------------------------

def _inventory(root: Path, scenes) -> Path:
    for scene in scenes:
        (root / scene).mkdir(parents=True, exist_ok=True)
        (root / scene / "manifest.json").write_text("{}", encoding="utf-8")
    return root


def test_folds_verify_against_a_complete_inventory(tmp_path):
    evidence = check_folds_against_inventory(_inventory(tmp_path, REPLICA_SCENES))
    assert evidence["n_scenes"] == 18
    assert len(evidence["folds"]) == 3


def test_a_missing_scene_is_a_missing_artifact(tmp_path):
    with pytest.raises(GateStop) as caught:
        check_folds_against_inventory(_inventory(tmp_path, REPLICA_SCENES[:-1]))
    assert caught.value.step == "11"
    assert caught.value.classification == MISSING_ARTIFACT


def test_an_extra_scene_is_a_frozen_design_mismatch(tmp_path):
    """A scene the split hash does not cover makes fold membership undefined."""
    with pytest.raises(GateStop) as caught:
        check_folds_against_inventory(
            _inventory(tmp_path, list(REPLICA_SCENES) + ["office_5"])
        )
    assert caught.value.classification == FROZEN_DESIGN_MISMATCH
    assert "office_5" in caught.value.evidence["unexpected"]


# ---------------------------------------------------------------------------
# Step 14, the seal, demonstrated by trying to break it
# ---------------------------------------------------------------------------

def test_the_seal_is_demonstrated_not_asserted():
    evidence = check_test_seal()
    assert evidence["train_accepts_train"]
    assert evidence["selection_accepts_val"]
    assert evidence["test_rejected"]
    assert "sealed" in evidence["test_rejection_message"]


# ---------------------------------------------------------------------------
# Step 16, no side effects
# ---------------------------------------------------------------------------

def test_a_checkpoint_written_by_the_gate_is_a_stop(tmp_path):
    before: set[Path] = set()
    (tmp_path / "sneaky.pt").write_bytes(b"weights")
    with pytest.raises(GateStop) as caught:
        assert_no_checkpoint_written(tmp_path, before)
    assert caught.value.classification == IMPLEMENTATION_BUG
    assert "sneaky.pt" in json.dumps(caught.value.evidence)


def test_no_new_checkpoint_passes(tmp_path):
    (tmp_path / "existing.pt").write_bytes(b"weights")
    before = set(tmp_path.glob("**/*.pt"))
    evidence = assert_no_checkpoint_written(tmp_path, before)
    assert evidence["checkpoints_before"] == evidence["checkpoints_after"] == 1


# ---------------------------------------------------------------------------
# Hashing and environment
# ---------------------------------------------------------------------------

def test_tree_digest_is_order_independent_and_content_sensitive(tmp_path):
    for name in ("b.txt", "a.txt", "c.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    first, count = sha256_tree(tmp_path)
    assert count == 3
    assert sha256_tree(tmp_path)[0] == first
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    assert sha256_tree(tmp_path)[0] != first


def test_environment_identity_names_what_ran():
    identity = environment_identity()
    assert identity["python"] and identity["torch"]
    assert "cuda_available" in identity


# ---------------------------------------------------------------------------
# Round-three finding 2: every scene compared against the accepted identity
# ---------------------------------------------------------------------------

def _identity_env(monkeypatch, tmp_path, accepted, live_features, live_depth, live_manifest):
    """Stand up fake caches and a fake Phase 4 parquet record for one scene."""
    from lot import phase5_check as m
    import lot.encoders as enc
    import lot.phase4 as p4
    import lot.evaluate as ev

    eval_dir = tmp_path / "phase4" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "room_0.parquet").write_bytes(b"parquet-bytes")
    monkeypatch.setattr(ev, "read_run_metadata", lambda path: accepted)

    # Both digests now come from the small meta.json rather than from
    # decompressing the depth archive, so the stub answers per encoder.
    def cache_meta(root, encoder, scene):
        if encoder == "vggt_1b":
            return {"depth_digest": live_depth}
        return {"features_digest": live_features}

    monkeypatch.setattr(enc, "load_cache_meta", cache_meta)
    monkeypatch.setattr(p4, "manifest_digest", lambda root: live_manifest)

    @dataclasses.dataclass
    class Cfg:
        phase4_dir: str = str(tmp_path / "phase4")
        cache_root: str = str(tmp_path / "cache")
        feature_encoder: str = "dinov2_vitb14"
        depth_encoder: str = "vggt_1b"
        renders_root: str = str(tmp_path / "renders")

    return m, Cfg()


def test_matching_scene_identities_pass_and_are_recorded(monkeypatch, tmp_path):
    from lot.phase5_check import verify_scene_identities

    accepted = {"features_digest": "F", "depth_digest": "D", "manifest_digest": "M",
                "mean_vector_digest": "V", "git_commit": "abc"}
    _, cfg = _identity_env(monkeypatch, tmp_path, accepted, "F", "D", "M")
    out = verify_scene_identities(cfg, ["room_0"])
    assert out["room_0"]["features_digest"] == "F"
    assert out["room_0"]["accepted_mean_vector_digest"] == "V"
    assert out["room_0"]["phase4_parquet_bytes"] == len(b"parquet-bytes")


@pytest.mark.parametrize("field,live", [
    ("features_digest", ("X", "D", "M")),
    ("depth_digest", ("F", "X", "M")),
    ("manifest_digest", ("F", "D", "X")),
])
def test_a_live_input_disagreeing_with_phase4_is_a_stop(monkeypatch, tmp_path, field, live):
    from lot.phase5_check import verify_scene_identities

    accepted = {"features_digest": "F", "depth_digest": "D", "manifest_digest": "M"}
    _, cfg = _identity_env(monkeypatch, tmp_path, accepted, *live)
    with pytest.raises(GateStop) as caught:
        verify_scene_identities(cfg, ["room_0"])
    assert caught.value.classification == FROZEN_DESIGN_MISMATCH
    mismatch = caught.value.evidence["mismatches"][0]
    assert mismatch["field"] == field
    assert mismatch["accepted"] != mismatch["live"]


def test_a_missing_phase4_parquet_is_a_missing_artifact_not_a_design_mismatch(
    monkeypatch, tmp_path
):
    """The two failures have different remedies and must not be conflated.

    An absent parquet is fixed by copying a file. A frozen-design mismatch may
    require an amendment with a written rationale. Reporting the first as the
    second sends the reader to the wrong repair.
    """
    from lot.phase5_check import MISSING_ARTIFACT, verify_scene_identities

    accepted = {"features_digest": "F", "depth_digest": "D", "manifest_digest": "M"}
    _, cfg = _identity_env(monkeypatch, tmp_path, accepted, "F", "D", "M")
    with pytest.raises(GateStop) as caught:
        verify_scene_identities(cfg, ["room_0", "room_1"])
    assert caught.value.classification == MISSING_ARTIFACT
    absent = caught.value.evidence["absent"]
    assert any(a["scene"] == "room_1" and a["input"] == "phase4_parquet" for a in absent)
    # room_0's parquet exists and agrees, so it is not reported as absent.
    assert not any(a["scene"] == "room_0" for a in absent)


def test_an_unreadable_cache_is_a_missing_artifact_not_an_implementation_bug(
    monkeypatch, tmp_path
):
    """Without this the loader's exception reaches run_steps' catch-all."""
    import lot.encoders as enc

    from lot.phase5_check import MISSING_ARTIFACT, verify_scene_identities

    accepted = {"features_digest": "F", "depth_digest": "D", "manifest_digest": "M"}
    _, cfg = _identity_env(monkeypatch, tmp_path, accepted, "F", "D", "M")

    def absent_cache(root, encoder, scene):
        raise FileNotFoundError(f"no cache for {encoder}/{scene}")

    monkeypatch.setattr(enc, "load_cache_meta", absent_cache)
    with pytest.raises(GateStop) as caught:
        verify_scene_identities(cfg, ["room_0"])
    assert caught.value.classification == MISSING_ARTIFACT
    assert caught.value.evidence["absent"][0]["input"] == "cache_or_manifest"


def test_the_mean_vector_must_be_the_one_phase4_centered_with():
    from lot.evaluate import vector_digest
    from lot.phase5_check import verify_mean_vector_identity

    center = torch.arange(8, dtype=torch.float32)
    live = vector_digest(center.numpy())
    ok = verify_mean_vector_identity(
        center, {"a": {"accepted_mean_vector_digest": live}}
    )
    assert ok["mean_vector_digest"] == live
    with pytest.raises(GateStop) as caught:
        verify_mean_vector_identity(
            center, {"a": {"accepted_mean_vector_digest": "not-it"}}
        )
    assert caught.value.classification == FROZEN_DESIGN_MISMATCH


def test_aligned_depth_digest_is_deterministic_and_content_sensitive():
    from lot.phase5_check import aligned_depth_digest

    @dataclasses.dataclass
    class Calib:
        scale: float = 1.0
        affine_failed: bool = False

    class Inputs:
        def __init__(self, value):
            import numpy as np
            self.est_maps = {"f1": np.full((2, 2), value, dtype=np.float32)}
            self.calibrations = {"f1": Calib()}

    a = aligned_depth_digest(Inputs(1.0))
    assert a == aligned_depth_digest(Inputs(1.0))
    assert a != aligned_depth_digest(Inputs(2.0))


def test_the_gate_no_longer_stops_at_a_probe_subset():
    """The identity check is over every scene the folds name."""
    import inspect

    from lot import phase5_gate

    source = inspect.getsource(phase5_gate.run_integration_gate)
    assert "verify_scene_identities(cfg, REPLICA_SCENES" in source
    assert "hash_scene_artifacts" not in source


# ---------------------------------------------------------------------------
# Outputs are write-once: CLAUDE.md forbids overwriting them
# ---------------------------------------------------------------------------

def test_write_once_keeps_the_previous_file(tmp_path):
    from lot.phase5_check import write_once

    target = tmp_path / "evidence" / "integration_gate.json"
    first = write_once(target, "first")
    assert first["archived_previous"] is None
    assert target.read_text(encoding="utf-8") == "first"

    second = write_once(target, "second")
    assert target.read_text(encoding="utf-8") == "second"
    kept = Path(second["archived_previous"])
    assert kept.exists() and kept.read_text(encoding="utf-8") == "first"


def test_write_once_numbers_successive_supersessions(tmp_path):
    from lot.phase5_check import write_once

    target = tmp_path / "receipt.json"
    write_once(target, "a")
    write_once(target, "b")
    write_once(target, "c")
    kept = sorted(p.name for p in tmp_path.glob("receipt.superseded.*.json"))
    assert kept == ["receipt.superseded.1.json", "receipt.superseded.2.json"]
    assert target.read_text(encoding="utf-8") == "c"
    # Nothing was lost.
    bodies = {(tmp_path / name).read_text(encoding="utf-8") for name in kept}
    assert bodies == {"a", "b"}


def test_step_six_compares_the_example_against_an_independent_expectation():
    """The defect this guards: the check compared uv_t with itself.

    Both sides were built from the same tensor, so the GateStop was unreachable
    and build_example could have indexed lift.landed instead of the support, or
    applied the patch mapping twice, with the step still reporting PASS. The
    comparison must read the coordinates the example actually carries and an
    expectation derived independently from the lift and the support.
    """
    import inspect

    from lot import phase5_gate

    source = inspect.getsource(phase5_gate.run_integration_gate)
    step6 = source[source.index("def step6("):source.index("def step7(")]
    assert "example.query_patch_coords" in step6, "step 6 ignores the real example"
    assert "primary_support(" in step6, "step 6 does not rebuild the support"
    assert "context_lift_support(" in step6, "step 6 does not rebuild evaluability"
    # The tautology was a self-comparison of the landing coordinates.
    assert '"supervision_read_uv"' not in step6


# ---------------------------------------------------------------------------
# Step 10: the pure-rotation check reads landed samples only
# ---------------------------------------------------------------------------
#
# Measured before the first real run reached step 10. On a 518 px frame with a
# 90 degree field of view, the residual between context-lift and the analytic
# homography over every context patch passes the 1e-3 px tolerance from about
# 45 degrees of rotation: 0.047 px at 45, 1.07 px at 75. Landed samples stay
# near 6e-5 px. The excess is float32 rounding on unlanded rays that project
# tens of thousands of pixels out, where no score ever reads.

ROTATION_TOL_PX = 1.0e-3
SIDE = 518


def _rotation(degrees: float) -> torch.Tensor:
    import math

    a = math.radians(degrees)
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = torch.tensor([[math.cos(a), 0.0, math.sin(a)],
                              [0.0, 1.0, 0.0],
                              [-math.sin(a), 0.0, math.cos(a)]], dtype=torch.float32)
    return T


def _rotation_case(T: torch.Tensor, substitute_T: torch.Tensor | None = None):
    from lot.context_lift import context_lift_map, rotation_homography_landing
    from lot.render_replica import intrinsics_from_hfov

    K = intrinsics_from_hfov(SIDE, SIDE, 90.0).to(torch.float32)
    hw = (SIDE, SIDE)
    depth = torch.full(hw, 2.5) + torch.linspace(0.0, 1.0, SIDE)[None, :]
    lift = context_lift_map(depth, K, K, T, hw, hw)
    analytic = rotation_homography_landing(K, K, T, hw, dtype=torch.float32)
    substituted = context_lift_map(
        torch.full_like(depth, 7.0), K, K, T if substitute_T is None else substitute_T, hw, hw
    )
    return lift, analytic, substituted


def test_step_ten_reads_only_landed_samples_at_large_rotation():
    from lot.phase5_gate import rotation_gate_evidence

    lift, analytic, substituted = _rotation_case(_rotation(60.0))
    # What the first version compared, and would have stopped on.
    assert float((lift.uv_target - analytic).abs().max()) > ROTATION_TOL_PX
    evidence = rotation_gate_evidence(lift, analytic, substituted, ROTATION_TOL_PX)
    assert evidence["n_landed_samples"] == int(lift.landed.sum())
    assert 0 < evidence["n_landed_samples"] < evidence["n_context_patches"]
    assert evidence["max_homography_residual_px"] <= ROTATION_TOL_PX


def test_step_ten_still_stops_a_wrong_rotation():
    """Comparing against the inverse rotation must fail on the landed samples."""
    from lot.context_lift import rotation_homography_landing
    from lot.phase5_gate import rotation_gate_evidence
    from lot.render_replica import intrinsics_from_hfov

    T = _rotation(20.0)
    lift, _, substituted = _rotation_case(T)
    K = intrinsics_from_hfov(SIDE, SIDE, 90.0).to(torch.float32)
    wrong = rotation_homography_landing(K, K, _rotation(-20.0), (SIDE, SIDE), dtype=torch.float32)
    with pytest.raises(GateStop) as stop:
        rotation_gate_evidence(lift, wrong, substituted, ROTATION_TOL_PX)
    assert stop.value.step == "10"
    assert stop.value.classification == IMPLEMENTATION_BUG


def test_step_ten_still_stops_a_depth_dependent_landing():
    """A substituted depth that moves a landing means the map is not depth free."""
    from lot.phase5_gate import rotation_gate_evidence

    T = _rotation(20.0)
    translated = T.clone()
    translated[0, 3] = 0.2
    lift, analytic, moved = _rotation_case(T, substitute_T=translated)
    with pytest.raises(GateStop) as stop:
        rotation_gate_evidence(lift, analytic, moved, ROTATION_TOL_PX)
    assert stop.value.step == "10"
    assert "depth" in stop.value.message


def test_step_ten_runs_the_tested_function():
    import inspect

    from lot import phase5_gate

    source = inspect.getsource(phase5_gate.run_integration_gate)
    step10 = source[source.index("def step10"):source.index("def step11")]
    assert "rotation_gate_evidence(" in step10
    assert "(lift.uv_target - analytic)" not in step10


# ---------------------------------------------------------------------------
# Step 17: the pure-rotation gate across the whole rotation regime
# ---------------------------------------------------------------------------
#
# reporting_rules.md section 7. Every rotation pair of all 18 scenes, at the
# primary level. Context-Lift is compared with the analytic homography, and
# with itself under a substituted depth map. TL-Reference's context read
# locations are compared with the inverse homography at the Phase 3 target
# samples. Each comparison runs on its own estimator's landed samples. The
# per-sample limit is the frozen tolerance plus focal length times the pair's
# translation over the sample's depth in the receiving camera.

POSITION_BOUND_M = 1.0e-6
COMPARISONS = ("context_lift", "depth_substitution", "tl_reference")
WHERE = "room_0 ctx -> tgt level image"


def _translated(T: torch.Tensor, x_m: float) -> torch.Tensor:
    moved = T.clone()
    moved[0, 3] = x_m
    return moved


def _target_samples(count: int = 600) -> torch.Tensor:
    """Phase 3 style target sample coordinates: continuous, inside the frame."""
    generator = torch.Generator().manual_seed(5)
    return torch.rand((count, 2), generator=generator, dtype=torch.float32) * (SIDE - 1)


def _tl_reads(depth_target: torch.Tensor, K: torch.Tensor, T: torch.Tensor,
              uv_target: torch.Tensor):
    """TL-Reference's read locations, through the primitives Phase 4 calls."""
    from lot.correspondence import _in_box, _sampling_box
    from lot.encoders import PATCH_SIZE, sample_map_bilinear
    from lot.geometry import invert_se3, project, transform_points, unproject

    read = sample_map_bilinear(depth_target, uv_target)
    valid = torch.isfinite(read) & (read > 0)
    safe = torch.where(valid, read, torch.ones_like(read))
    points = transform_points(invert_se3(T), unproject(uv_target, safe, K))
    uv, z = project(points, K)
    landed = valid & (z > 0) & _in_box(uv, _sampling_box((SIDE, SIDE), PATCH_SIZE))
    return uv, z, landed.numpy()


def _rotation_pair(T: torch.Tensor, *, lift_T: torch.Tensor | None = None,
                   substitute_T: torch.Tensor | None = None,
                   tl_T: torch.Tensor | None = None, near_m: float | None = None):
    """Keyword arguments for rotation_pair_evidence on one analytic 518 px pair.

    T is the pair's recorded relative transform. Each estimator can be built
    with a different transform, which is how a wrong mapping is injected.
    near_m puts both depth maps at about that depth.
    """
    from lot.context_lift import context_lift_map
    from lot.render_replica import intrinsics_from_hfov

    K = intrinsics_from_hfov(SIDE, SIDE, 90.0).to(torch.float32)
    hw = (SIDE, SIDE)
    ramp = torch.linspace(0.0, 1.0, SIDE)
    if near_m is None:
        depth_context = torch.full(hw, 2.5) + ramp[None, :]
        depth_target = torch.full(hw, 3.0) + 0.5 * ramp[:, None]
    else:
        depth_context = torch.full(hw, near_m) + 0.01 * ramp[None, :]
        depth_target = torch.full(hw, near_m) + 0.01 * ramp[:, None]
    lift = context_lift_map(depth_context, K, K, T if lift_T is None else lift_T, hw, hw)
    substituted = context_lift_map(
        torch.full_like(depth_context, 7.0), K, K,
        T if substitute_T is None else substitute_T, hw, hw,
    )
    uv_target = _target_samples()
    tl_uv, tl_z, tl_landed = _tl_reads(
        depth_target, K, T if tl_T is None else tl_T, uv_target
    )
    return {
        "lift": lift, "substituted": substituted,
        "tl_read_uv_context": tl_uv, "tl_read_depth_context": tl_z,
        "tl_landed": tl_landed, "uv_target_samples": uv_target,
        "K_context": K, "K_target": K, "T_target_from_context": T,
        "context_hw": hw, "tol_px": ROTATION_TOL_PX,
        "position_bound_m": POSITION_BOUND_M, "where": WHERE,
    }


@pytest.mark.parametrize("degrees", [5.0, 20.0, 45.0, 60.0])
def test_step_seventeen_passes_a_pure_rotation_on_each_estimators_landed_samples(degrees):
    from lot.phase5_gate import rotation_pair_evidence

    args = _rotation_pair(_rotation(degrees))
    evidence = rotation_pair_evidence(**args)
    assert evidence["pair"] == WHERE
    assert evidence["checked"] is True
    assert evidence["translation_norm_m"] == 0.0
    n_cl = int(args["lift"].landed.sum())
    n_tl = int(args["tl_landed"].sum())
    assert evidence["n_context_patches"] == args["lift"].landed.numel()
    assert evidence["n_tl_samples"] == args["uv_target_samples"].shape[0]
    assert evidence["context_lift"]["n_compared"] == n_cl > 0
    assert evidence["depth_substitution"]["n_compared"] == n_cl
    assert evidence["tl_reference"]["n_compared"] == n_tl > 0
    for name in COMPARISONS:
        assert evidence[name]["max_residual_px"] <= ROTATION_TOL_PX
        assert evidence[name]["max_translation_allowance_px"] == 0.0
        assert evidence[name]["min_headroom_px"] > 0.0


def test_step_seventeen_ignores_unlanded_samples_at_large_rotation():
    """The unlanded residual exceeds the tolerance at 60 degrees and is not read."""
    from lot.context_lift import rotation_homography_landing
    from lot.phase5_gate import rotation_pair_evidence

    args = _rotation_pair(_rotation(60.0))
    analytic = rotation_homography_landing(
        args["K_context"], args["K_target"], args["T_target_from_context"],
        args["context_hw"], dtype=torch.float32,
    )
    assert float((args["lift"].uv_target - analytic).abs().max()) > ROTATION_TOL_PX
    assert 0 < int(args["lift"].landed.sum()) < args["lift"].landed.numel()
    evidence = rotation_pair_evidence(**args)
    assert evidence["context_lift"]["max_residual_px"] <= ROTATION_TOL_PX


def test_step_seventeen_stops_a_translation_above_the_bound():
    from lot.phase5_gate import rotation_pair_evidence

    T = _translated(_rotation(20.0), 2.0e-6)
    with pytest.raises(GateStop) as stop:
        rotation_pair_evidence(**_rotation_pair(T))
    assert stop.value.step == "17"
    assert stop.value.classification == FROZEN_DESIGN_MISMATCH
    assert WHERE in stop.value.message
    assert stop.value.evidence["pair"] == WHERE
    assert stop.value.evidence["translation_norm_m"] == pytest.approx(2.0e-6, rel=1e-6)


def test_step_seventeen_stops_a_context_lift_with_the_wrong_rotation():
    from lot.phase5_gate import rotation_pair_evidence

    T = _rotation(20.0)
    with pytest.raises(GateStop) as stop:
        rotation_pair_evidence(**_rotation_pair(T, lift_T=_rotation(-20.0)))
    assert stop.value.step == "17"
    assert stop.value.classification == IMPLEMENTATION_BUG
    assert "Context-Lift" in stop.value.message
    assert WHERE in stop.value.message
    assert stop.value.evidence["comparison"] == "context_lift"
    worst = stop.value.evidence["worst_sample"]
    assert worst["residual_px"] > worst["limit_px"]
    assert f"sample {worst['index']}" in stop.value.message


def test_step_seventeen_stops_a_tl_reference_with_the_wrong_rotation():
    from lot.phase5_gate import rotation_pair_evidence

    T = _rotation(20.0)
    with pytest.raises(GateStop) as stop:
        rotation_pair_evidence(**_rotation_pair(T, tl_T=_rotation(-20.0)))
    assert stop.value.step == "17"
    assert stop.value.classification == IMPLEMENTATION_BUG
    assert "TL-Reference" in stop.value.message
    assert WHERE in stop.value.message
    assert stop.value.evidence["comparison"] == "tl_reference"


def test_step_seventeen_stops_a_depth_dependent_landing():
    from lot.phase5_gate import rotation_pair_evidence

    T = _rotation(20.0)
    with pytest.raises(GateStop) as stop:
        rotation_pair_evidence(**_rotation_pair(T, substitute_T=_translated(T, 0.2)))
    assert stop.value.step == "17"
    assert stop.value.classification == IMPLEMENTATION_BUG
    assert "depth" in stop.value.message
    assert stop.value.evidence["comparison"] == "depth_substitution"


def test_a_tiny_translation_within_the_bound_passes_thanks_to_the_allowance():
    """At 0.12 m, 0.9e-6 m of translation moves a landing by about 2e-3 px.

    That exceeds the bare 1e-3 px tolerance on every comparison, so without
    the allowance this genuine pure-rotation pair would stop the gate.
    """
    from lot.phase5_gate import rotation_pair_evidence

    T = _translated(_rotation(10.0), 0.9e-6)
    evidence = rotation_pair_evidence(**_rotation_pair(T, near_m=0.12))
    assert evidence["translation_norm_m"] == pytest.approx(0.9e-6, rel=1e-6)
    for name in COMPARISONS:
        assert evidence[name]["n_compared"] > 0
        assert evidence[name]["max_residual_px"] > ROTATION_TOL_PX, name
        assert evidence[name]["max_translation_allowance_px"] > 0.0
        assert evidence[name]["min_headroom_px"] > 0.0


def test_the_allowance_is_not_a_blanket_pass():
    """A hundred times the translation, injected into Context-Lift, still stops."""
    from lot.phase5_gate import rotation_pair_evidence

    T = _translated(_rotation(10.0), 0.9e-6)
    with pytest.raises(GateStop) as stop:
        rotation_pair_evidence(
            **_rotation_pair(T, lift_T=_translated(T, 0.9e-4), near_m=0.12)
        )
    assert stop.value.classification == IMPLEMENTATION_BUG
    assert stop.value.evidence["comparison"] == "context_lift"


def test_a_pair_where_nothing_lands_is_recorded_and_not_checked():
    from lot.phase5_gate import rotation_pair_evidence

    evidence = rotation_pair_evidence(**_rotation_pair(_rotation(150.0)))
    assert evidence["checked"] is False
    for name in COMPARISONS:
        assert evidence[name]["n_compared"] == 0
        assert evidence[name]["max_residual_px"] is None


def test_a_scene_with_rotation_pairs_but_none_checkable_stops():
    from lot.phase5_gate import rotation_pair_evidence, rotation_scene_summary

    blind = rotation_pair_evidence(**_rotation_pair(_rotation(150.0)))
    with pytest.raises(GateStop) as stop:
        rotation_scene_summary("room_0", [blind], [])
    assert stop.value.step == "17"
    assert stop.value.classification == FROZEN_DESIGN_MISMATCH
    assert "room_0" in stop.value.message
    with pytest.raises(GateStop) as stop:
        rotation_scene_summary("room_0", [], [WHERE])
    assert stop.value.classification == FROZEN_DESIGN_MISMATCH


def test_a_scene_without_rotation_pairs_is_recorded_not_stopped():
    from lot.phase5_gate import rotation_scene_summary

    summary = rotation_scene_summary("room_0", [], [])
    assert summary["n_rotation_pairs"] == 0
    assert summary["n_pairs_checked"] == 0


def test_the_regime_summary_stops_on_zero_rotation_pairs_overall():
    from lot.phase5_gate import rotation_regime_summary, rotation_scene_summary

    empty = rotation_scene_summary("room_0", [], [])
    with pytest.raises(GateStop) as stop:
        rotation_regime_summary({"room_0": empty, "room_1": empty},
                                ROTATION_TOL_PX, POSITION_BOUND_M)
    assert stop.value.step == "17"
    assert stop.value.classification == FROZEN_DESIGN_MISMATCH
    with pytest.raises(GateStop):
        rotation_regime_summary({}, ROTATION_TOL_PX, POSITION_BOUND_M)


def test_the_regime_summary_names_the_worst_pair_and_the_largest_translation():
    from lot.phase5_gate import (
        rotation_pair_evidence,
        rotation_regime_summary,
        rotation_scene_summary,
    )

    def pair(degrees, where, x_m=0.0, near_m=None):
        args = _rotation_pair(_translated(_rotation(degrees), x_m), near_m=near_m)
        args["where"] = where
        return rotation_pair_evidence(**args)

    quiet = pair(5.0, "room_0 a -> b level image")
    loud = pair(10.0, "room_1 c -> d level image", x_m=0.9e-6, near_m=0.12)
    blind = pair(150.0, "room_1 e -> f level image")
    per_scene = {
        "room_0": rotation_scene_summary("room_0", [quiet], []),
        "room_1": rotation_scene_summary("room_1", [loud, blind], []),
        "room_2": rotation_scene_summary("room_2", [], []),
    }
    assert per_scene["room_1"]["n_rotation_pairs"] == 2
    assert per_scene["room_1"]["n_pairs_checked"] == 1
    assert per_scene["room_1"]["n_pairs_nothing_landed"] == 1
    summary = rotation_regime_summary(per_scene, ROTATION_TOL_PX, POSITION_BOUND_M)
    assert summary["n_rotation_pairs"] == 3
    assert summary["n_pairs_checked"] == 2
    assert summary["max_translation_norm_m"] == pytest.approx(0.9e-6, rel=1e-6)
    assert summary["max_translation_pair"] == "room_1 c -> d level image"
    for name in COMPARISONS:
        worst = summary["worst"][name]
        assert worst["pair"] == "room_1 c -> d level image"
        assert worst["max_residual_px"] == loud[name]["max_residual_px"]
        assert worst["min_headroom_px"] > 0.0
        assert summary["scenes"]["room_0"]["worst"][name]["pair"] == "room_0 a -> b level image"
    assert summary["tolerance_px"] == ROTATION_TOL_PX
    assert summary["rotation_position_bound_m"] == POSITION_BOUND_M
    json.dumps(summary)


def _gate_step_ids() -> list[str]:
    import ast
    import inspect
    import textwrap

    from lot import phase5_gate

    tree = ast.parse(textwrap.dedent(inspect.getsource(phase5_gate.run_integration_gate)))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run_steps"
    ]
    assert len(calls) == 1
    return [entry.elts[0].value for entry in calls[0].args[0].elts]


def test_step_seventeen_runs_before_the_pin_and_step_sixteen_stays_last():
    """Step 15 writes pin_cluster.json, moving any earlier pin aside. Running
    step 17 first means the pin is written only after every substantive check
    has passed, so a gate that stops at step 17 leaves the earlier pin in
    place. Step 16, the verdict, stays last."""
    assert _gate_step_ids() == [str(i) for i in range(1, 15)] + ["17", "15", "16"]


def test_step_seventeen_runs_the_tested_functions():
    """The closure must drive what these tests drive, over every scene."""
    import inspect

    from lot import phase5_gate

    source = inspect.getsource(phase5_gate.run_integration_gate)
    step17 = source[source.index("def step17"):source.index("return run_steps(")]
    assert "for scene in REPLICA_SCENES" in step17
    assert "build_scene_inputs(" in step17
    assert ".close()" in step17
    assert "rotation_scene_evidence(" in step17
    assert "rotation_regime_summary(" in step17
    assert '("17", "pure-rotation gate across the regime", step17)' in source

    scene = inspect.getsource(phase5_gate.rotation_scene_evidence)
    assert "phase5_scene_pairs(" in scene
    assert 'regime == "rotation"' in scene
    assert "cfg.primary_alignment_level" in scene
    assert "recompute_reference_arms(" in scene
    assert "rotation_pair_evidence(" in scene
    assert "rotation_scene_summary(" in scene

    pair = inspect.getsource(phase5_gate.rotation_pair_evidence)
    assert "rotation_homography_landing(" in pair
