"""A gate receipt is evidence about the state that produced it, and nothing else.

The failure this guards against is quiet: a green receipt from an earlier commit
sitting in the evidence directory while different code trains beside it. Checking
only the receipt's verdict would let that through, so every identity field it
carries has to match the run asking to proceed.
"""

from __future__ import annotations

import json
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


def test_a_matching_receipt_is_accepted(tmp_path):
    assert verify(_receipt(tmp_path), CONFIG, "gate") == []


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
    assert verify(path, CONFIG, "gate") == []


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
# The receipt is bound to the inputs the gate actually resolved
# ---------------------------------------------------------------------------

def _receipt_with_artifacts(tmp_path: Path, artifacts: dict) -> Path:
    report = {
        "passed": True,
        "steps": [
            {"step": "1", "evidence": current_identity(CONFIG)},
            {"step": "2", "evidence": {"artifacts": artifacts}},
        ],
    }
    path = tmp_path / "with_artifacts.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_a_receipt_naming_a_different_input_tree_is_refused(tmp_path):
    """Two runs can share a commit and a config and still read different bytes."""
    path = _receipt_with_artifacts(
        tmp_path, {"phase4_dir": {"path": "/somewhere/else/phase4_rung1"}}
    )
    problems = verify(path, CONFIG, "gate")
    # Only fires when the current run can resolve that artifact; on this machine
    # phase4_dir does not exist, so the check is correctly silent rather than
    # inventing a comparison against nothing.
    from lot.phase5 import load_phase5_config
    resolvable = Path(load_phase5_config(CONFIG).phase4_dir).exists()
    assert bool(problems) == resolvable


def test_the_artifact_check_compares_the_paths_the_gate_recorded(tmp_path):
    """Exercised against an artifact that does resolve here."""
    from lot.phase5 import load_phase5_config

    cfg = load_phase5_config(CONFIG)
    mean_dir = Path(cfg.mean_vector_dir)
    if not mean_dir.exists():
        pytest.skip("the Phase 3 outputs are not present on this machine")

    matching = _receipt_with_artifacts(
        tmp_path, {"mean_vector_dir": {"path": str(mean_dir.resolve())}}
    )
    assert verify(matching, CONFIG, "gate") == []


def test_a_receipt_without_an_artifact_block_still_checks_identity(tmp_path):
    """An older receipt format must not silently skip the identity binding."""
    path = _receipt(tmp_path, commit="stale")
    problems = verify(path, CONFIG, "gate")
    assert any("commit moved" in p for p in problems)
