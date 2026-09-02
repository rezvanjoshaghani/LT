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
