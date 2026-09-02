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
    monkeypatch.setattr(enc, "load_cache_meta",
                        lambda root, encoder, scene: {"features_digest": live_features})
    monkeypatch.setattr(p4, "load_depth_archive",
                        lambda root, encoder, scene: {"meta": {"depth_digest": live_depth}})
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


def test_a_missing_phase4_parquet_is_reported_per_scene(monkeypatch, tmp_path):
    from lot.phase5_check import verify_scene_identities

    _, cfg = _identity_env(monkeypatch, tmp_path, {}, "F", "D", "M")
    with pytest.raises(GateStop) as caught:
        verify_scene_identities(cfg, ["room_0", "room_1"])
    problems = caught.value.evidence["mismatches"]
    assert any(p["scene"] == "room_1" and "missing" in p["problem"] for p in problems)


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
