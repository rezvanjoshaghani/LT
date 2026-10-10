"""A gate receipt is evidence about the state that produced it, and nothing else.

The failure this guards against is quiet: a green receipt from an earlier commit
sitting in the evidence directory while different code trains beside it. Checking
only the receipt's verdict would let that through, so every identity field it
carries has to match the run asking to proceed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lot.phase5_receipt import BOUND_FIELDS, current_identity, receipt_identity, verify

CONFIG = Path("configs/phase5.yaml")


def _receipt(tmp_path: Path, passed: bool = True, **overrides) -> Path:
    identity = {**current_identity(CONFIG), **overrides}
    report = {
        "passed": passed,
        "steps": [{"step": "1", "title": "identity", "passed": True,
                   "evidence": identity}],
        "failure": None if passed else {"step": "7", "message": "support moved"},
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _identity_problems(problems: list[str]) -> list[str]:
    """The identity-layer problems only; artifact binding is tested separately."""
    return [
        p for p in problems
        if "artifact" not in p and "identity block" not in p
    ]


def test_a_matching_receipt_passes_the_identity_layer(tmp_path):
    assert _identity_problems(verify(_receipt(tmp_path), CONFIG, "gate")) == []


def test_a_failed_receipt_is_refused_with_its_failing_step(tmp_path):
    problems = verify(_receipt(tmp_path, passed=False), CONFIG, "gate")
    assert problems
    assert "FAIL" in problems[0]
    assert "support moved" in problems[0]


@pytest.mark.parametrize("field", BOUND_FIELDS)
def test_every_bound_field_refuses_a_mismatch(field, tmp_path):
    """A stale receipt must not carry a later run over the line."""
    path = _receipt(tmp_path, **{field: "something-else"})
    problems = verify(path, CONFIG, "gate")
    assert problems, f"{field} was not checked"
    assert field in problems[0]
    assert "Rerun it" in problems[0]


def test_a_receipt_missing_an_identity_field_is_refused(tmp_path):
    identity = current_identity(CONFIG)
    identity.pop("commit")
    report = {"passed": True, "steps": [{"evidence": identity}]}
    path = tmp_path / "thin.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    problems = verify(path, CONFIG, "gate")
    assert any("carries no commit" in p for p in problems)


def test_identity_is_read_from_the_top_level_too(tmp_path):
    """A simpler receipt format must not silently skip the binding."""
    report = {"passed": True, **current_identity(CONFIG)}
    path = tmp_path / "flat.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert _identity_problems(verify(path, CONFIG, "gate")) == []


def test_an_unreadable_receipt_is_refused(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    problems = verify(path, CONFIG, "gate")
    assert problems and "could not be read" in problems[0]


def test_a_missing_receipt_is_refused(tmp_path):
    problems = verify(tmp_path / "absent.json", CONFIG, "gate")
    assert problems and "could not be read" in problems[0]


def test_receipt_identity_prefers_the_first_occurrence(tmp_path):
    report = {
        "commit": "top",
        "steps": [{"evidence": {"commit": "step", "config_digest": "d"}}],
    }
    identity = receipt_identity(report)
    assert identity["commit"] == "top"
    assert identity["config_digest"] == "d"


def test_current_identity_names_all_bound_fields():
    identity = current_identity(CONFIG)
    assert set(identity) == set(BOUND_FIELDS)
    assert all(identity[field] for field in BOUND_FIELDS)


# ---------------------------------------------------------------------------
# The receipt is bound to the content of the inputs the gate resolved
# ---------------------------------------------------------------------------

def _full_receipt(tmp_path: Path, artifacts: dict, scenes: dict, **identity) -> Path:
    report = {
        "passed": True,
        "steps": [
            {"step": "1", "evidence": {**current_identity(CONFIG), **identity}},
            {"step": "2", "evidence": {"artifacts": artifacts, "scene_identities": scenes}},
        ],
    }
    path = tmp_path / "full.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_a_receipt_without_an_artifact_block_cannot_license_training(tmp_path):
    """A verdict alone is not an integration receipt."""
    problems = verify(_receipt(tmp_path), CONFIG, "gate")
    assert any("no resolved-artifact block" in p for p in problems)


def test_a_receipt_without_scene_identities_is_refused(tmp_path):
    path = _full_receipt(tmp_path, artifacts={"x": {}}, scenes={})
    problems = verify(path, CONFIG, "gate")
    assert any("no per-scene identity block" in p for p in problems)


def test_every_required_artifact_must_be_recorded(tmp_path):
    from lot.phase5_receipt import BOUND_ARTIFACT_FIELDS

    assert "mean_vector_dir" in BOUND_ARTIFACT_FIELDS
    assert "phase4_convention" in BOUND_ARTIFACT_FIELDS
    path = _full_receipt(tmp_path, artifacts={"renders_root": {"path": "x"}}, scenes={"a": {}})
    problems = verify(path, CONFIG, "gate")
    # Every bound artifact this run can resolve and the receipt omits is named.
    from lot.phase5 import load_phase5_config
    cfg = load_phase5_config(CONFIG)
    if Path(cfg.mean_vector_dir).exists():
        assert any("does not record mean_vector_dir" in p for p in problems)


def test_an_artifact_this_run_cannot_resolve_is_a_failure_not_a_skip(tmp_path):
    from lot.phase5 import load_phase5_config

    cfg = load_phase5_config(CONFIG)
    if Path(cfg.phase4_dir).exists():
        pytest.skip("phase4_dir resolves here; the skip path cannot be exercised")
    path = _full_receipt(
        tmp_path,
        artifacts={"phase4_dir": {"path": "/elsewhere/phase4_rung1"}},
        scenes={"a": {}},
    )
    problems = verify(path, CONFIG, "gate")
    assert any("phase4_dir is not present" in p for p in problems)


def test_a_file_artifact_that_changed_under_its_path_is_refused(tmp_path):
    """Same path, different bytes: the binding is by content."""
    import dataclasses

    from lot.phase5 import load_phase5_config
    from lot.phase5_check import sha256_file

    convention = tmp_path / "phase4" / "evidence" / "convention_record.json"
    convention.parent.mkdir(parents=True)
    convention.write_text('{"convention": "planar_z"}', encoding="utf-8")
    (tmp_path / "phase4" / "eval").mkdir()
    cfg_path = tmp_path / "phase5.yaml"
    base = load_phase5_config(CONFIG)
    moved = dataclasses.replace(base, phase4_dir=str(tmp_path / "phase4"))
    import yaml
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["phase4_dir"] = str(tmp_path / "phase4")
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    good = {
        "phase4_convention": {
            "path": str(convention.resolve()), "sha256": sha256_file(convention),
        }
    }
    path = _full_receipt(tmp_path, artifacts=good, scenes={"a": {}},
                         config_digest=moved.digest())
    before = [p for p in verify(path, cfg_path, "gate") if "phase4_convention" in p]
    assert before == [], before

    convention.write_text('{"convention": "ray_distance"}', encoding="utf-8")
    after = [p for p in verify(path, cfg_path, "gate") if "phase4_convention" in p]
    assert after and "changed under its path" in after[0]


def test_scene_identity_fields_are_compared_when_recomputable(tmp_path, monkeypatch):
    """The real comparison runs; only the loaders it calls are stubbed."""
    import lot.encoders as enc
    import lot.phase4 as p4
    import lot.phase5_check as chk
    from lot import phase5_receipt as m
    from lot.render_replica import REPLICA_SCENES

    def cache_meta(root, encoder, scene):
        if encoder == "vggt_1b":
            return {"depth_digest": "d"}
        return {"features_digest": "f"}

    monkeypatch.setattr(enc, "load_cache_meta", cache_meta)
    monkeypatch.setattr(p4, "manifest_digest", lambda root: "m")
    monkeypatch.setattr(chk, "sha256_file", lambda path: "p")

    # A real parquet path is stat()ed for its byte count, so give every scene one.
    from lot.phase5 import load_phase5_config
    import dataclasses, yaml
    eval_dir = tmp_path / "phase4" / "eval"
    eval_dir.mkdir(parents=True)
    for scene in REPLICA_SCENES:
        (eval_dir / f"{scene}.parquet").write_bytes(b"x")
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["phase4_dir"] = str(tmp_path / "phase4")
    cfg_path = tmp_path / "phase5.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_phase5_config(cfg_path)

    live = {"features_digest": "f", "depth_digest": "d", "manifest_digest": "m",
            "phase4_parquet_sha256": "p", "phase4_parquet_bytes": 1}
    scenes = {scene: dict(live) for scene in REPLICA_SCENES}
    scenes["room_0"]["depth_digest"] = "stale"
    report = {"passed": True, "steps": [
        {"step": "2", "evidence": {"artifacts": {"x": {}}, "scene_identities": scenes}},
    ]}
    problems = m._scene_identity_problems(report, cfg, "gate")
    assert any("room_0 depth_digest moved" in p for p in problems), problems
    # Every other scene agreed, so room_0 is the only mismatch reported.
    assert sum("moved since the gate ran" in p for p in problems) == 1


def test_a_scene_missing_from_the_receipt_is_named(tmp_path, monkeypatch):
    from lot import phase5_receipt as m
    from lot.render_replica import REPLICA_SCENES

    scenes = {scene: {} for scene in REPLICA_SCENES if scene != "office_3"}
    report = {"steps": [{"evidence": {"scene_identities": scenes}}]}
    monkeypatch.setattr(m, "BOUND_SCENE_FIELDS", ())
    from lot.phase5 import load_phase5_config
    problems = m._scene_identity_problems(report, load_phase5_config(CONFIG), "gate")
    assert any("office_3" in p and "did not verify" in p for p in problems)


def test_receipt_scene_identities_merge_step2_and_step4_blocks():
    from lot.phase5_receipt import receipt_scene_identities

    report = {"steps": [
        {"evidence": {"scene_identities": {"a": {"features_digest": "f"}}}},
        {"evidence": {"scene_identities": {"a": {"aligned_depth_digest": "z"}}}},
    ]}
    merged = receipt_scene_identities(report)
    assert merged["a"] == {"features_digest": "f", "aligned_depth_digest": "z"}


# ---------------------------------------------------------------------------
# Receipt kinds: the overfit receipt must be verifiable
# ---------------------------------------------------------------------------

def _overfit_receipt(tmp_path: Path, gate_path: Path, **identity) -> Path:
    from lot.phase5_receipt import KIND_OVERFIT, stamp_receipt

    report = stamp_receipt(
        {"passed": True, "reached_centered_cosine": 0.991},
        CONFIG, KIND_OVERFIT, gate_receipt=gate_path,
    )
    report.update(identity)
    path = tmp_path / "tiny_overfit.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _gate_file(tmp_path: Path, body: str = "gate") -> Path:
    path = tmp_path / "integration_gate.json"
    path.write_text(json.dumps({"passed": True, "body": body}), encoding="utf-8")
    return path


def test_an_overfit_receipt_verifies_without_an_artifact_block(tmp_path):
    """The defect this guards: applying the gate's bindings to every receipt
    made the overfit receipt impossible to verify, so both gates could pass and
    training would still refuse to start."""
    from lot.phase5_receipt import KIND_OVERFIT

    gate = _gate_file(tmp_path)
    receipt = _overfit_receipt(tmp_path, gate)
    assert verify(receipt, CONFIG, "overfit", kind=KIND_OVERFIT, gate_receipt=gate) == []


def test_an_overfit_receipt_is_bound_to_the_gate_that_licensed_it(tmp_path):
    from lot.phase5_receipt import KIND_OVERFIT

    gate = _gate_file(tmp_path)
    receipt = _overfit_receipt(tmp_path, gate)
    # Rerunning the gate changes its receipt, which invalidates the binding.
    gate.write_text(json.dumps({"passed": True, "body": "rerun"}), encoding="utf-8")
    problems = verify(receipt, CONFIG, "overfit", kind=KIND_OVERFIT, gate_receipt=gate)
    assert any("ran under a different integration gate" in p for p in problems)


def test_an_overfit_receipt_without_its_gate_binding_is_refused(tmp_path):
    from lot.phase5_receipt import KIND_OVERFIT

    gate = _gate_file(tmp_path)
    path = tmp_path / "unbound.json"
    path.write_text(
        json.dumps({"passed": True, "kind": KIND_OVERFIT, **current_identity(CONFIG)}),
        encoding="utf-8",
    )
    problems = verify(path, CONFIG, "overfit", kind=KIND_OVERFIT, gate_receipt=gate)
    assert any("does not record the integration receipt" in p for p in problems)


def test_a_receipt_of_the_wrong_kind_is_refused(tmp_path):
    from lot.phase5_receipt import KIND_INTEGRATION, KIND_OVERFIT

    gate = _gate_file(tmp_path)
    receipt = _overfit_receipt(tmp_path, gate)
    problems = verify(
        receipt, CONFIG, "gate", kind=KIND_INTEGRATION, gate_receipt=gate
    )
    assert any("required here" in p for p in problems)


def test_an_unlabelled_receipt_defaults_to_the_stricter_kind(tmp_path):
    """A receipt that does not say what it is gets the binding that refuses more."""
    from lot.phase5_receipt import KIND_INTEGRATION, receipt_kind

    assert receipt_kind({}) == KIND_INTEGRATION
    assert receipt_kind({"kind": "nonsense"}) == KIND_INTEGRATION
    problems = verify(_receipt(tmp_path), CONFIG, "gate")
    assert any("no resolved-artifact block" in p for p in problems)


def test_stamp_receipt_refuses_an_overfit_stamp_without_a_gate(tmp_path):
    from lot.phase5_receipt import KIND_OVERFIT, stamp_receipt

    with pytest.raises(ValueError, match="must be bound to the integration receipt"):
        stamp_receipt({"passed": True}, CONFIG, KIND_OVERFIT, gate_receipt=None)
    with pytest.raises(ValueError, match="unknown receipt kind"):
        stamp_receipt({"passed": True}, CONFIG, "something-else")


def test_stamp_receipt_supplies_the_identity_every_receipt_must_carry(tmp_path):
    from lot.phase5_receipt import BOUND_FIELDS, KIND_INTEGRATION, stamp_receipt

    stamped = stamp_receipt({"passed": True}, CONFIG, KIND_INTEGRATION)
    assert stamped["kind"] == KIND_INTEGRATION
    for field in BOUND_FIELDS:
        assert stamped[field] == current_identity(CONFIG)[field]


# ---------------------------------------------------------------------------
# Every receipt records when it was stamped, reporting_rules.md section 7
# ---------------------------------------------------------------------------

# ISO 8601 in UTC, always to the microsecond, with the offset written out.
UTC_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")


def test_stamp_receipt_records_when_it_was_stamped_in_utc(tmp_path):
    from datetime import datetime, timedelta, timezone

    from lot.phase5_receipt import KIND_INTEGRATION, KIND_OVERFIT, stamp_receipt

    gate = _gate_file(tmp_path)
    before = datetime.now(timezone.utc)
    stamps = [
        stamp_receipt({"passed": True}, CONFIG, KIND_INTEGRATION)["stamped_utc"],
        stamp_receipt({"passed": True}, CONFIG, KIND_OVERFIT,
                      gate_receipt=gate)["stamped_utc"],
    ]
    after = datetime.now(timezone.utc)
    for stamp in stamps:
        assert UTC_STAMP.match(stamp), stamp
        moment = datetime.fromisoformat(stamp)
        assert moment.utcoffset() == timedelta(0)
        assert before <= moment <= after


def test_a_stamp_time_already_in_a_report_is_replaced(tmp_path):
    """The stamp says when this receipt was written, not what the report claimed."""
    from lot.phase5_receipt import KIND_INTEGRATION, stamp_receipt

    stamped = stamp_receipt(
        {"passed": True, "stamped_utc": "1999-01-01T00:00:00.000000+00:00"},
        CONFIG, KIND_INTEGRATION,
    )
    assert not stamped["stamped_utc"].startswith("1999")


@pytest.mark.parametrize("stamped", [True, False])
def test_verify_accepts_an_overfit_receipt_with_or_without_its_stamp_time(
    tmp_path, stamped
):
    """The time is recorded and never bound, so an older receipt still verifies."""
    from lot.phase5_receipt import KIND_OVERFIT

    gate = _gate_file(tmp_path)
    receipt = _overfit_receipt(tmp_path, gate)
    report = json.loads(receipt.read_text(encoding="utf-8"))
    assert UTC_STAMP.match(report["stamped_utc"])
    if not stamped:
        del report["stamped_utc"]
        receipt.write_text(json.dumps(report), encoding="utf-8")
    assert verify(receipt, CONFIG, "overfit", kind=KIND_OVERFIT, gate_receipt=gate) == []


@pytest.mark.parametrize("stamped", [True, False])
def test_verify_accepts_an_integration_receipt_with_or_without_its_stamp_time(
    tmp_path, stamped
):
    from lot.phase5_receipt import KIND_INTEGRATION, stamp_receipt

    report = stamp_receipt({"passed": True}, CONFIG, KIND_INTEGRATION)
    assert UTC_STAMP.match(report["stamped_utc"])
    if not stamped:
        del report["stamped_utc"]
    path = tmp_path / "integration_gate.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert _identity_problems(verify(path, CONFIG, "gate", kind=KIND_INTEGRATION)) == []


# ---------------------------------------------------------------------------
# The checkpoint lock, reporting_rules.md section 7
# ---------------------------------------------------------------------------
#
# The lock receipt is bound by sha256 to the integration receipt and to the
# overfit receipt. It also names every file evaluation reads before it touches
# a test scene: each (fold, seed)'s checkpoint and training record, and the
# controls file of the level. The verifier only hashes those files, so here
# they hold arbitrary bytes. tests/test_phase5_modes.py locks real checkpoints.

LOCK_LEVEL = "image"


def config_with_outputs_at(tmp_path: Path) -> Path:
    """The shipped configuration, with its outputs under tmp_path.

    output_root is outside the config digest, so the identity is the shipped
    configuration's own.
    """
    import yaml

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["output_root"] = str(tmp_path / "outputs")
    path = tmp_path / "phase5.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def write_locked_files(config_path: Path, level: str = LOCK_LEVEL) -> dict:
    """Stand-ins for every file a lock binds, where a run under config_path reads them.

    Returns the lock's file block: each checkpoint and training record under
    its (fold, seed) key, and the controls file, each with its path and sha256.
    """
    from lot.phase5 import load_phase5_config
    from lot.phase5_check import sha256_file
    from lot.phase5_folds import frozen_folds
    from lot.phase5_modes import checkpoint_lock_files
    from lot.train import training_config_from

    cfg = load_phase5_config(config_path)
    files = checkpoint_lock_files(
        cfg, level, frozen_folds(), training_config_from(cfg.training).seeds
    )

    def entry(path: Path, body: str) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return {"path": str(path), "sha256": sha256_file(path)}

    return {
        "checkpoints": {key: entry(path, f"checkpoint {key}")
                        for key, path in files["checkpoints"].items()},
        "training_records": {key: entry(path, f"record {key}")
                             for key, path in files["training_records"].items()},
        "controls": entry(files["controls"], "controls"),
    }


def _lock(tmp_path: Path, level: str = LOCK_LEVEL) -> dict:
    """A stamped lock receipt over stand-in files, and the receipts it is bound to."""
    from lot.phase5_receipt import KIND_LOCK, stamp_receipt

    config = config_with_outputs_at(tmp_path)
    gate = _gate_file(tmp_path)
    overfit = _overfit_receipt(tmp_path, gate)
    report = stamp_receipt(
        {"passed": True, "level": level, **write_locked_files(config, level)},
        config, KIND_LOCK, gate_receipt=gate, overfit_receipt=overfit,
    )
    path = tmp_path / f"checkpoint_lock_{level}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return {"lock": path, "config": config, "gate": gate, "overfit": overfit,
            "report": report}


def _verify_lock(lock: dict, level: str | None = LOCK_LEVEL, **overrides) -> list[str]:
    from lot.phase5_receipt import KIND_LOCK

    arguments = {"kind": KIND_LOCK, "gate_receipt": lock["gate"],
                 "overfit_receipt": lock["overfit"], "level": level, **overrides}
    return verify(lock["lock"], lock["config"], "lock", **arguments)


def _rewrite(lock: dict, report: dict) -> None:
    lock["lock"].write_text(json.dumps(report), encoding="utf-8")


def test_a_lock_receipt_verifies(tmp_path):
    """Bound to both receipts by sha256. It names a checkpoint and a training
    record for each of the frozen folds by the frozen seeds, nine of each, and
    the controls file."""
    from lot.phase5_check import sha256_file
    from lot.phase5_receipt import (
        GATE_RECEIPT_DIGEST,
        KIND_LOCK,
        OVERFIT_RECEIPT_DIGEST,
        RECEIPT_KINDS,
    )

    assert KIND_LOCK in RECEIPT_KINDS
    lock = _lock(tmp_path)
    report = lock["report"]
    assert report["kind"] == KIND_LOCK
    assert report[GATE_RECEIPT_DIGEST] == sha256_file(lock["gate"])
    assert report[OVERFIT_RECEIPT_DIGEST] == sha256_file(lock["overfit"])
    assert UTC_STAMP.match(report["stamped_utc"])
    for field in BOUND_FIELDS:
        assert report[field] == current_identity(lock["config"])[field]
    assert len(report["checkpoints"]) == len(report["training_records"]) == 9
    assert _verify_lock(lock) == []


@pytest.mark.parametrize("group,key,name", [
    ("checkpoints", "fold1_seed2", "checkpoint fold1_seed2"),
    ("training_records", "fold2_seed0", "training record fold2_seed0"),
    ("controls", None, "controls file"),
])
def test_a_file_changed_after_locking_fails_verify(tmp_path, group, key, name):
    """Evaluation refuses a lock that no longer matches the live files, and
    the one file that changed is the one named."""
    lock = _lock(tmp_path)
    entry = lock["report"][group] if key is None else lock["report"][group][key]
    path = Path(entry["path"])
    path.write_bytes(path.read_bytes() + b"x")
    problems = _verify_lock(lock)
    assert len(problems) == 1, problems
    assert name in problems[0] and "changed since the lock was written" in problems[0]


def test_a_locked_file_that_is_gone_fails_verify(tmp_path):
    lock = _lock(tmp_path)
    Path(lock["report"]["checkpoints"]["fold0_seed1"]["path"]).unlink()
    problems = _verify_lock(lock)
    assert len(problems) == 1, problems
    assert "checkpoint fold0_seed1" in problems[0] and "absent" in problems[0]


def test_a_lock_bound_to_a_superseded_gate_receipt_fails(tmp_path):
    """Rerunning check moves the old gate receipt aside and writes a new one,
    through write_once. The lock still names the old receipt's sha256."""
    from lot.phase5_check import write_once

    lock = _lock(tmp_path)
    outcome = write_once(lock["gate"], json.dumps({"passed": True, "body": "rerun"}))
    assert outcome["archived_previous"]
    problems = _verify_lock(lock)
    assert any("ran under a different integration gate" in p for p in problems), problems


def test_a_lock_bound_to_a_superseded_overfit_receipt_fails(tmp_path):
    from lot.phase5_check import write_once

    lock = _lock(tmp_path)
    write_once(lock["overfit"], json.dumps({"passed": True, "body": "rerun"}))
    problems = _verify_lock(lock)
    assert any("ran under a different overfit gate" in p for p in problems), problems


@pytest.mark.parametrize("field", ["gate_receipt_sha256", "overfit_receipt_sha256"])
def test_a_lock_without_one_of_its_bindings_is_refused(tmp_path, field):
    lock = _lock(tmp_path)
    report = dict(lock["report"])
    del report[field]
    _rewrite(lock, report)
    problems = _verify_lock(lock)
    assert any(f"({field})" in p and "does not record" in p for p in problems), problems


@pytest.mark.parametrize("field", BOUND_FIELDS)
def test_a_lock_receipt_is_bound_to_the_run_identity(tmp_path, field):
    lock = _lock(tmp_path)
    _rewrite(lock, {**lock["report"], field: "something-else"})
    problems = _verify_lock(lock)
    assert any(field in p and "moved since the receipt was written" in p
               for p in problems), problems


def test_a_lock_for_another_level_licenses_nothing_at_this_one(tmp_path):
    lock = _lock(tmp_path)
    problems = _verify_lock(lock, level="affine")
    assert len(problems) == 1, problems
    assert "'image'" in problems[0] and "'affine'" in problems[0]


def test_a_lock_must_name_exactly_the_files_this_run_reads(tmp_path):
    """The files are located from the configuration, the frozen folds, and the
    configured seeds. A lock that omits one does not bind it, and a lock that
    names another binds nothing this run reads."""
    lock = _lock(tmp_path)
    report = json.loads(lock["lock"].read_text(encoding="utf-8"))
    report["checkpoints"]["fold3_seed0"] = report["checkpoints"].pop("fold2_seed2")
    _rewrite(lock, report)
    problems = _verify_lock(lock)
    assert any("does not record the checkpoint fold2_seed2" in p for p in problems), problems
    assert any("fold3_seed0" in p and "does not read" in p for p in problems), problems
    assert len(problems) == 2, problems


def test_verifying_a_lock_needs_both_receipts_and_the_level(tmp_path):
    """A missing argument is the caller's error, and it is reported, not skipped."""
    lock = _lock(tmp_path)
    assert any("no integration receipt was supplied" in p
               for p in _verify_lock(lock, gate_receipt=None))
    assert any("no overfit receipt was supplied" in p
               for p in _verify_lock(lock, overfit_receipt=None))
    assert any("no alignment level was supplied" in p
               for p in _verify_lock(lock, level=None))


def test_stamp_receipt_refuses_a_lock_without_both_receipts(tmp_path):
    from lot.phase5_receipt import KIND_LOCK, stamp_receipt

    gate = _gate_file(tmp_path)
    overfit = _overfit_receipt(tmp_path, gate)
    with pytest.raises(ValueError, match="must be bound to the integration receipt"):
        stamp_receipt({"passed": True}, CONFIG, KIND_LOCK, overfit_receipt=overfit)
    with pytest.raises(ValueError, match="must be bound to the overfit receipt"):
        stamp_receipt({"passed": True}, CONFIG, KIND_LOCK, gate_receipt=gate)
    with pytest.raises(ValueError, match="must be bound to the overfit receipt"):
        stamp_receipt({"passed": True}, CONFIG, KIND_LOCK, gate_receipt=gate,
                      overfit_receipt=tmp_path / "absent.json")


def test_the_command_line_verifier_checks_a_lock(tmp_path, capsys):
    """The launcher's early check goes through this entry point."""
    from lot.phase5_receipt import main

    lock = _lock(tmp_path)
    arguments = [
        "--receipt", str(lock["lock"]), "--config", str(lock["config"]),
        "--label", "the checkpoint lock", "--kind", "lock",
        "--gate-receipt", str(lock["gate"]),
        "--overfit-receipt", str(lock["overfit"]), "--level", LOCK_LEVEL,
    ]
    main(arguments)
    assert "PASS" in capsys.readouterr().out
    checkpoint = Path(lock["report"]["checkpoints"]["fold0_seed0"]["path"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    with pytest.raises(SystemExit) as stop:
        main(arguments)
    assert stop.value.code == 1
    assert "checkpoint fold0_seed0 changed" in capsys.readouterr().err
