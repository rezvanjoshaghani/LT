"""Provenance for the Phase 5 reporting modes, reporting_rules.md decision 1.

The chain runs once, at commit E. The tables, figures, and acceptance modes
run at a later commit R. They are licensed by the evaluated run's own
provenance, never by receipts at R.

These tests build a complete evaluated run on disk, as the chain writes it,
and then damage one thing at a time. Each damage must be refused, and the
refusal must name the field that is wrong. The parquets are written with
lot.evaluate.write_rows, and their run records by
lot.phase5_modes.evaluation_metadata, the writer the evaluate mode uses. So
the reader is tested against the real record layout. Receipts, checkpoints,
training records, the controls file, and the lock hold synthetic content in
the places the chain writes them. Only the integration receipt's binding to
the cluster inputs is replaced, because tests/test_phase5_receipt.py covers
it.

Code ancestry is tested on temporary git repositories.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

import lot.phase5_modes as modes
import lot.phase5_provenance as prov
import lot.phase5_receipt as receipt_module
from lot.analysis_config import load_analysis_config
from lot.evaluate import read_rows, read_run_metadata, write_rows
from lot.phase5 import load_phase5_config
from lot.phase5_check import sha256_file, write_once
from lot.phase5_folds import fold_digest, fold_of_test_scene, frozen_folds
from lot.phase5_modes import (
    checkpoint_lock_path,
    checkpoint_path,
    controls_path,
    controls_payload,
    evaluation_metadata,
    fold_seed_key,
    training_record_path,
)
from lot.phase5_provenance import ProvenanceError, require_evaluated_run

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "phase5.yaml"
LEVEL = "image"
FOLDS = frozen_folds()
# One test scene of each fold, so every fold's models and controls are read.
SCENES = [fold.test[0] for fold in FOLDS]
# The commit the synthetic chain ran at. Never a real commit of this checkout.
E = "e" * 40
STAMP = "2026-10-10T09:00:00.000000+00:00"

OVERFIT_VERDICT = {
    "passed": True, "reached_centered_cosine": 0.9917, "threshold": 0.98, "steps": 1460,
    "n_pairs": 8, "regimes": ["rotation", "translation", "orbit"], "fold": 0,
    "level": LEVEL, "seed": 0,
    "subset": [{"scene": FOLDS[0].train[0], "context_frame_id": "000",
                "target_frame_id": "004", "regime": "rotation", "n_supported": 812}],
}


# ---------------------------------------------------------------------------
# A complete evaluated run, as the chain writes it
# ---------------------------------------------------------------------------

def _json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")
    return path


def _stamp(payload: dict, kind: str, identity: dict, **digests) -> dict:
    """A receipt as lot.phase5_receipt.stamp_receipt writes it, at this identity."""
    return {**payload, "kind": kind, **identity, **digests, "stamped_utc": STAMP}


def _training_record(cfg, fold, seed, checkpoint: Path, commit: str, gates: dict) -> dict:
    """A training record in the layout lot.phase5_modes.run_train_task writes.

    Its best validation is the best entry of its own history, at best_step,
    as lot.train records it.
    """
    history = [[500 * i, 0.5 + 0.01 * i + 0.01 * seed] for i in range(1, 8)]
    history.append([4000, history[-1][1] - 0.005])
    return {
        "fold": fold.index, "seed": seed,
        "train_scenes": list(fold.train), "val_scenes": list(fold.val),
        "test_scenes": list(fold.test),
        "steps_run": 4000, "best_step": 3500,
        "best_validation_centered_cosine": history[6][1],
        "parameter_count": 16680960, "training_config_digest": "7" * 64,
        "stopped_early": True,
        "history": history,
        "level": LEVEL, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "n_train_examples": 120, "n_val_examples": 30, "census": [],
        "superseded": {"checkpoint": None, "record": None},
        "config_digest": cfg.digest(), "commit": commit, "licence": dict(gates),
        "train_scenes_planned": list(fold.train), "val_scenes_planned": list(fold.val),
        "written_utc": "2026-10-10T08:00:00.000000+00:00",
    }


def build_run(tmp_path: Path, monkeypatch, commit: str = E,
              receipt_commit: str | None = None, training_commit: str | None = None,
              overfit_passed: bool = True) -> SimpleNamespace:
    """Every artifact the chain writes, for SCENES at LEVEL, evaluated at commit.

    receipt_commit and training_commit, when given, are the commits the
    receipts and the training records name instead.
    """
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["output_root"] = str(tmp_path / "outputs")
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "phase5.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_phase5_config(config)
    analysis = load_analysis_config()
    seeds = [int(seed) for seed in raw["training"]["seeds"]]
    run_dir, evidence = Path(cfg.run_dir), Path(cfg.evidence_dir)

    identity = {"commit": commit, "config_digest": cfg.digest(),
                "fold_digest": fold_digest(FOLDS),
                "measurement_digest": analysis.measurement_digest()}
    stamped = {**identity, "commit": receipt_commit or commit}
    gate = _json(evidence / "integration_gate.json",
                 _stamp({"passed": True, "steps": [], "failure": None}, "integration",
                        stamped))
    overfit = _json(evidence / "tiny_overfit.json",
                    _stamp({**OVERFIT_VERDICT, "passed": overfit_passed}, "overfit",
                           stamped, gate_receipt_sha256=sha256_file(gate)))
    gates = {"integration_gate": sha256_file(gate), "tiny_overfit": sha256_file(overfit)}

    checkpoints: dict[str, Path] = {}
    records: dict[str, Path] = {}
    for fold in FOLDS:
        for seed in seeds:
            key = fold_seed_key(fold.index, seed)
            ckpt = checkpoint_path(run_dir, LEVEL, fold.index, seed)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_bytes(f"the weights of {key}".encode("utf-8"))
            checkpoints[key] = ckpt
            records[key] = _json(
                training_record_path(run_dir, LEVEL, fold.index, seed),
                _training_record(cfg, fold, seed, ckpt, training_commit or commit, gates),
            )

    results = {
        key: {"n_pairs": 40, "val_scenes_planned": ["x"],
              "pose_shuffle": {"baseline_centered_cosine": 0.6,
                               "shuffled_centered_cosine": 0.4, "degradation": 0.2,
                               "n_samples": 900, "n_unchanged": 0},
              "depth_shuffle": {"baseline_centered_cosine": 0.6,
                                "shuffled_centered_cosine": 0.59, "degradation": 0.01,
                                "n_samples": 900, "n_unchanged": 3}}
        for key in checkpoints
    }
    controls = _json(controls_path(evidence, LEVEL), controls_payload(
        LEVEL, results, [], identity, gates,
        {key: sha256_file(path) for key, path in checkpoints.items()},
        written_utc="2026-10-10T08:30:00.000000+00:00",
    ))

    def entry(path: Path) -> dict:
        return {"path": str(path), "sha256": sha256_file(path)}

    lock = _json(checkpoint_lock_path(evidence, LEVEL), _stamp(
        {"passed": True, "level": LEVEL, "folds": [fold.index for fold in FOLDS],
         "seeds": seeds,
         "checkpoints": {key: entry(path) for key, path in checkpoints.items()},
         "training_records": {key: entry(path) for key, path in records.items()},
         "controls": entry(controls)},
        "lock", stamped, gate_receipt_sha256=gates["integration_gate"],
        overfit_receipt_sha256=gates["tiny_overfit"],
    ))
    licence = {**gates, lock.stem: sha256_file(lock)}

    # The writer records the commit it runs at. Here that is the chain's E.
    monkeypatch.setattr(modes, "git_commit", lambda: commit)
    center = torch.linspace(-1.0, 1.0, 768)
    for scene in SCENES:
        phase4 = tmp_path / "phase4" / f"{scene}.parquet"
        phase4.parent.mkdir(parents=True, exist_ok=True)
        phase4.write_bytes(f"phase 4 rows of {scene}".encode("utf-8"))
        audit = {"pairs": 2, "evaluated": 2, "no_arm": 0, "worst_per_point_residual": 0.0,
                 "worst_splat_residual": 0.0, "no_arm_pairs": []}
        meta = evaluation_metadata(
            cfg, analysis, scene, fold_of_test_scene(scene, FOLDS), LEVEL, center, run_dir,
            seeds, {"metadata": {"git_commit": "4" * 40}}, phase4, audit,
            licence=licence, aligned_depth_digest="a" * 64,
        )
        rows = [{"scene": scene, "context_frame_id": "000", "target_frame_id": "001",
                 "seed": seed, "region": "all", "delta_learn_pp": 0.01 * seed}
                for seed in seeds]
        write_rows(eval_path(run_dir, scene), rows, meta)

    # The integration receipt's binding to the cluster inputs is
    # tests/test_phase5_receipt.py's subject. Every other binding runs.
    monkeypatch.setattr(receipt_module, "_artifact_problems", lambda *args: [])
    return SimpleNamespace(
        cfg=cfg, config=config, analysis=analysis, run_dir=run_dir, evidence=evidence,
        seeds=seeds, licence=licence, gates=gates, gate=gate, overfit=overfit,
        lock=lock, controls=controls, checkpoints=checkpoints, records=records,
        commit=commit,
    )


def eval_path(run_dir: Path, scene: str) -> Path:
    return Path(run_dir) / "eval" / LEVEL / f"{scene}.parquet"


@pytest.fixture
def run(tmp_path, monkeypatch):
    return build_run(tmp_path, monkeypatch)


def require(run, **kwargs):
    arguments = {"expected_scenes": SCENES, "check_code": False, **kwargs}
    return require_evaluated_run(run.cfg, run.analysis, run.config, LEVEL, **arguments)


def refused(run, field: str, match: str | None = None, **kwargs) -> ProvenanceError:
    """The run is refused, and the refusal names field."""
    with pytest.raises(ProvenanceError) as error:
        require(run, **kwargs)
    assert field in error.value.fields, str(error.value)
    if match is not None:
        assert re.search(match, str(error.value)), str(error.value)
    return error.value


def rewrite(run, change, scenes=None) -> None:
    """Rewrite run records in place: same rows, the record changed by change."""
    for scene in scenes or SCENES:
        path = eval_path(run.run_dir, scene)
        rows, meta = read_rows(path), read_run_metadata(path)
        change(meta)
        path.unlink()
        path.with_suffix(".meta.json").unlink()
        write_rows(path, rows, meta)


# ---------------------------------------------------------------------------
# A complete run
# ---------------------------------------------------------------------------

def test_a_complete_run_yields_its_identity(run):
    identity = require(run)
    assert identity.level == LEVEL
    assert identity.commit == identity.training_commit == E
    assert identity.report_commit is None and identity.code_checked is False
    assert identity.scenes == tuple(SCENES)
    assert identity.folds == tuple(fold.index for fold in FOLDS)
    assert identity.seeds == tuple(run.seeds)
    assert identity.config_digest == run.cfg.digest()
    assert identity.fold_digest == fold_digest(FOLDS)
    assert identity.measurement_digest == run.analysis.measurement_digest()
    assert identity.analysis_reporting_digest == run.analysis.reporting_digest()
    assert identity.eval_version == modes.PHASE5_EVAL_VERSION
    assert identity.phase4_commit == "4" * 40
    assert identity.licence == run.licence
    assert identity.bootstrap == {
        "primary_unit": "scene", "resamples": run.analysis.bootstrap_resamples,
        "seed": run.analysis.bootstrap_seed, "confidence": run.analysis.bootstrap_confidence,
    }
    for scene in SCENES:
        assert identity.eval_paths[scene] == str(eval_path(run.run_dir, scene))
        assert identity.eval_sha256[scene] == sha256_file(eval_path(run.run_dir, scene))
    assert identity.checkpoints == {key: sha256_file(p) for key, p in run.checkpoints.items()}
    assert identity.training_records == {key: sha256_file(p) for key, p in run.records.items()}
    assert identity.controls_sha256 == sha256_file(run.controls)
    assert identity.receipts == {
        stem: {"path": str(path), "sha256": run.licence[stem]}
        for stem, path in (("integration_gate", run.gate), ("tiny_overfit", run.overfit),
                           (run.lock.stem, run.lock))
    }
    assert identity.reporting_changes == {}
    # The run records that were verified, so the tables never read them again.
    for scene in SCENES:
        assert identity.records[scene] == read_run_metadata(eval_path(run.run_dir, scene))
    assert not re.search(r"\brecords=", repr(identity))
    # Every file the run was read from, by its path under the run directory.
    inputs = identity.inputs
    assert inputs[f"eval/{LEVEL}/{SCENES[0]}.parquet"] == identity.eval_sha256[SCENES[0]]
    assert inputs[f"checkpoints/{LEVEL}/fold2_seed1.pt"] == identity.checkpoints["fold2_seed1"]
    assert (inputs[f"checkpoints/{LEVEL}/fold0_seed2.json"]
            == identity.training_records["fold0_seed2"])
    assert inputs[f"evidence/input_use_controls_{LEVEL}.json"] == identity.controls_sha256
    assert inputs["evidence/tiny_overfit.json"] == run.licence["tiny_overfit"]
    assert len(inputs) == len(SCENES) + 2 * len(run.checkpoints) + 1 + 3


def test_the_default_scenes_are_every_test_scene(run):
    """Without injection, completeness is checked against all eighteen."""
    error = refused(run, "scenes", "missing", expected_scenes=None)
    assert all(scene in str(error) for scene in FOLDS[0].test[1:])


# ---------------------------------------------------------------------------
# 1. Completeness
# ---------------------------------------------------------------------------

def test_a_missing_scene_is_refused(run):
    eval_path(run.run_dir, SCENES[1]).unlink()
    error = refused(run, "scenes", "missing")
    assert SCENES[1] in str(error)


def test_an_extra_scene_is_refused(run):
    extra = FOLDS[0].test[1]
    shutil.copyfile(eval_path(run.run_dir, SCENES[0]), eval_path(run.run_dir, extra))
    shutil.copyfile(eval_path(run.run_dir, SCENES[0]).with_suffix(".meta.json"),
                    eval_path(run.run_dir, extra).with_suffix(".meta.json"))
    rewrite(run, lambda meta: meta.update(scene=extra), scenes=[extra])
    error = refused(run, "scenes", "unexpected")
    assert extra in str(error) and error.fields == ("scenes",)


@pytest.mark.parametrize("scenes, message", [
    ([], "no scenes are expected"),
    ([SCENES[0], SCENES[0]], "repeat"),
], ids=["none", "repeated"])
def test_the_expected_scenes_are_a_caller_choice_checked_first(run, scenes, message):
    with pytest.raises(ValueError, match=message) as error:
        require(run, expected_scenes=scenes)
    assert not isinstance(error.value, ProvenanceError)


def test_an_unfinished_write_is_refused(run):
    partial = eval_path(run.run_dir, SCENES[0]).with_name(f"{SCENES[0]}.parquet.partial")
    partial.write_bytes(b"half a parquet")
    refused(run, "partial", re.escape(partial.name))


def test_a_parquet_whose_record_is_only_beside_it_is_refused(run):
    """The record must be inside the parquet. A sidecar alone would make the
    run unreadable from the evaluation parquets alone."""
    path = eval_path(run.run_dir, SCENES[2])
    rows = read_rows(path)
    path.unlink()
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert path.with_suffix(".meta.json").exists()
    refused(run, "run_record", SCENES[2])


@pytest.mark.parametrize("field, value", [
    ("scene", SCENES[1]),
    ("level", "affine"),
    ("fold", FOLDS[2].index),
    ("phase", 4),
], ids=["stem", "level", "fold", "phase"])
def test_a_record_that_does_not_describe_its_file_is_refused(run, field, value):
    rewrite(run, lambda meta: meta.update({field: value}), scenes=[SCENES[0]])
    error = refused(run, field)
    assert SCENES[0] in str(error)


def test_checkpoints_must_be_keyed_by_the_seeds(run):
    rewrite(run, lambda meta: meta["checkpoints"].pop("2"), scenes=[SCENES[0]])
    refused(run, "checkpoints", "keyed")


# ---------------------------------------------------------------------------
# 2. One identity, present then equal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", prov.IDENTITY_FIELDS)
def test_an_identity_field_one_record_lacks_is_refused(run, field):
    """Present, then equal. A field no record carries must not agree with itself."""
    rewrite(run, lambda meta: meta.pop(field), scenes=[SCENES[1]])
    error = refused(run, field, "absent")
    assert SCENES[1] in str(error)


DIFFERENT = {
    "commit": "f" * 40,
    "config_digest": "0" * 64,
    "fold_digest": "0" * 64,
    "measurement_digest": "0" * 32,
    "mean_vector_digest": "0" * 32,
    "seeds": [2, 1, 0],
    "phase4_commit": "5" * 40,
    "eval_version": 2,
    "analysis_reporting_digest": "0" * 32,
    "licence": None,
}


@pytest.mark.parametrize("field", prov.IDENTITY_FIELDS)
def test_a_run_mixing_identities_is_refused(run, field):
    def change(meta):
        if field == "licence":
            meta["licence"] = {**meta["licence"], "tiny_overfit": "0" * 64}
        else:
            meta[field] = DIFFERENT[field]

    rewrite(run, change, scenes=[SCENES[2]])
    refused(run, field, "2 values")


@pytest.mark.parametrize("commit, words", [
    (E + "-dirty", "uncommitted"),
    ("unknown", "unknown"),
    ("e" * 12, "full commit"),
], ids=["dirty", "unknown", "abbreviated"])
def test_a_commit_that_names_no_code_is_refused(run, commit, words):
    rewrite(run, lambda meta: meta.update(commit=commit))
    refused(run, "commit", words)


def test_a_run_record_of_another_layout_is_refused(run):
    rewrite(run, lambda meta: meta.update(eval_version=modes.PHASE5_EVAL_VERSION + 1))
    refused(run, "eval_version")


def test_an_evaluation_licensed_without_its_lock_is_refused(run):
    rewrite(run, lambda meta: meta["licence"].pop(run.lock.stem))
    refused(run, "licence", run.lock.stem)


# ---------------------------------------------------------------------------
# 3. Binding to the configuration and analysis reading the run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field, value", [
    ("measurement_digest", "0" * 32),
    ("config_digest", "0" * 64),
    ("fold_digest", "0" * 64),
    ("analysis_reporting_digest", "0" * 32),
    ("seeds", [2, 1, 0]),
])
def test_a_run_measured_under_another_identity_is_refused(run, field, value):
    rewrite(run, lambda meta: meta.update({field: value}))
    refused(run, field, "reading")


def test_the_configuration_object_must_be_the_file_it_names(run):
    """A caller error, not a defect of the run."""
    import dataclasses

    other = dataclasses.replace(run.cfg, seed=run.cfg.seed + 1)
    with pytest.raises(ValueError, match="config"):
        require_evaluated_run(other, run.analysis, run.config, LEVEL,
                              expected_scenes=SCENES, check_code=False)


def test_an_undeclared_level_is_refused(run):
    with pytest.raises(ValueError, match="not declared"):
        require_evaluated_run(run.cfg, run.analysis, run.config, "scene",
                              expected_scenes=SCENES, check_code=False)


# ---------------------------------------------------------------------------
# 4. The receipt chain at E
# ---------------------------------------------------------------------------

def test_a_receipt_moved_aside_by_a_rerun_is_located_by_its_sha256(run):
    """A gate rerun after evaluation supersedes the receipt that licensed E. The
    run is still licensed by that receipt, found among the superseded files."""
    write_once(run.overfit, json.dumps({"passed": True, "body": "a rerun at R"}))
    write_once(run.gate, json.dumps({"passed": True, "body": "a rerun at R"}))
    identity = require(run)
    for stem in ("integration_gate", "tiny_overfit"):
        located = Path(identity.receipts[stem]["path"])
        assert located.name == f"{stem}.superseded.1.json"
        assert sha256_file(located) == run.licence[stem]
    assert identity.inputs["evidence/tiny_overfit.superseded.1.json"] == (
        run.licence["tiny_overfit"])


def test_two_files_with_the_licensed_sha256_are_refused(run):
    copy = run.evidence / "tiny_overfit.superseded.4.json"
    shutil.copyfile(run.overfit, copy)
    error = refused(run, "receipts.tiny_overfit", "2 files")
    assert copy.name in str(error)


def test_a_licensed_receipt_that_is_gone_is_refused(run):
    run.lock.unlink()
    refused(run, f"receipts.{run.lock.stem}", "no file")


def test_locate_receipt_reads_current_and_superseded_files_only(run):
    stem = "tiny_overfit"
    assert prov.locate_receipt(run.evidence, stem, run.licence[stem]) == run.overfit
    decoy = run.evidence / "tiny_overfit_copy.json"
    shutil.copyfile(run.overfit, decoy)
    run.overfit.unlink()
    with pytest.raises(ProvenanceError, match="no file"):
        prov.locate_receipt(run.evidence, stem, run.licence[stem])


def test_receipts_are_verified_against_e_and_not_against_head(tmp_path, monkeypatch):
    """Receipts stamped at another commit did not license the run evaluated at E."""
    run = build_run(tmp_path, monkeypatch, receipt_commit="f" * 40)
    error = refused(run, "receipts.integration_gate", "evaluated run: " + E)
    assert {"receipts.tiny_overfit", f"receipts.{run.lock.stem}"} <= set(error.fields)


def test_a_failed_overfit_gate_licenses_nothing(tmp_path, monkeypatch):
    run = build_run(tmp_path, monkeypatch, overfit_passed=False)
    refused(run, "receipts.tiny_overfit", "FAIL")


def test_the_integration_receipt_is_bound_to_the_live_inputs(run, monkeypatch):
    monkeypatch.setattr(receipt_module, "_artifact_problems",
                        lambda *args: ["gate: renders_root changed under its path"])
    refused(run, "receipts.integration_gate", "renders_root")


def test_a_checkpoint_changed_after_the_lock_is_refused(run):
    path = run.checkpoints["fold1_seed2"]
    path.write_bytes(path.read_bytes() + b"x")
    refused(run, f"receipts.{run.lock.stem}", "fold1_seed2 changed since the lock")


# ---------------------------------------------------------------------------
# 5. The checkpoint chain and the embedded evidence
# ---------------------------------------------------------------------------

def test_a_checkpoint_the_lock_does_not_bind_is_refused(run):
    rewrite(run, lambda meta: meta["checkpoints"].update({"1": "0" * 64}),
            scenes=[SCENES[1]])
    error = refused(run, "checkpoints", "lock")
    assert SCENES[1] in str(error)


def test_a_training_record_the_lock_does_not_bind_is_refused(run):
    rewrite(run, lambda meta: meta["training_records"]["0"].update(sha256="0" * 64),
            scenes=[SCENES[0]])
    refused(run, "training_records", "lock")


def test_the_embedded_training_record_must_be_the_locked_one(run):
    def change(meta):
        meta["training_record_contents"]["2"]["best_step"] += 500

    rewrite(run, change, scenes=[SCENES[2]])
    refused(run, "training_record_contents", SCENES[2])


def test_training_at_another_commit_is_refused(tmp_path, monkeypatch):
    """Decision 1: one commit for the whole chain."""
    run = build_run(tmp_path, monkeypatch, training_commit="7" * 40)
    error = refused(run, "training_record_contents", "commit")
    assert "training_records" in error.fields


def test_the_embedded_controls_must_be_the_locked_ones(run):
    def change(meta):
        key = next(iter(meta["controls"]["results"]))
        meta["controls"]["results"][key]["n_pairs"] = 41

    rewrite(run, change, scenes=[SCENES[1]])
    refused(run, "controls", SCENES[1])


def test_the_embedded_overfit_verdict_must_be_the_licensed_one(run):
    rewrite(run, lambda meta: meta["overfit_verdict"].update(reached_centered_cosine=0.5),
            scenes=[SCENES[0]])
    refused(run, "overfit_verdict", SCENES[0])


@pytest.mark.parametrize("key", prov.EMBEDDED_FIELDS)
def test_a_record_without_its_embedded_evidence_is_refused(run, key):
    rewrite(run, lambda meta: meta.pop(key), scenes=[SCENES[2]])
    refused(run, key, SCENES[2])


# ---------------------------------------------------------------------------
# Reading the rows that were verified
# ---------------------------------------------------------------------------

def test_rows_are_read_only_from_the_bytes_that_were_verified(run):
    identity = require(run)
    path = eval_path(run.run_dir, SCENES[1])
    assert prov.read_scene_rows(identity, SCENES[1]) == read_rows(path)
    rows, meta = read_rows(path), read_run_metadata(path)
    rows[0]["delta_learn_pp"] = 0.5
    path.unlink()
    write_rows(path, rows, meta)
    with pytest.raises(ProvenanceError) as error:
        prov.read_scene_rows(identity, SCENES[1])
    assert "eval_sha256" in error.value.fields


# ---------------------------------------------------------------------------
# Output records and atomic writes
# ---------------------------------------------------------------------------

def test_an_output_record_names_the_run_and_every_input(run):
    from lot.phase5_check import environment_identity

    identity = require(run)
    record = prov.output_run_record(
        identity, "phase5_primary", {"tables/image/phase5_primary.parquet": "9" * 64},
        {"table": "phase5_primary"},
    )
    assert record["kind"] == "phase5_primary" and record["table"] == "phase5_primary"
    assert record["record_version"] == prov.OUTPUT_RECORD_VERSION
    assert record["level"] == LEVEL
    assert record["evaluation_commit"] == record["training_commit"] == E
    assert record["report_commit"] is None and record["code_checked"] is False
    for field in ("config_digest", "fold_digest", "measurement_digest",
                  "mean_vector_digest", "analysis_reporting_digest", "eval_version",
                  "phase4_commit"):
        assert record[field] == getattr(identity, field), field
    assert record["seeds"] == list(identity.seeds)
    assert record["scenes"] == list(SCENES)
    assert record["bootstrap"] == identity.bootstrap
    assert record["licence"] == identity.licence
    assert record["receipts"] == identity.receipts
    assert record["inputs"] == {**identity.inputs,
                                "tables/image/phase5_primary.parquet": "9" * 64}
    assert record["reporting_changes"] == {}
    assert record["environment"] == environment_identity()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00",
                        record["created_utc"])
    json.dumps(record, allow_nan=False)


def test_an_output_record_refuses_a_clash(run):
    identity = require(run)
    with pytest.raises(ValueError, match="kind"):
        prov.output_run_record(identity, "phase5_primary", {}, {"kind": "other"})
    named = f"eval/{LEVEL}/{SCENES[0]}.parquet"
    with pytest.raises(ValueError, match=re.escape(named)):
        prov.output_run_record(identity, "phase5_primary", {named: "0" * 64}, {})
    # The same input under the same hash is not a clash.
    prov.output_run_record(identity, "phase5_primary", {named: identity.inputs[named]}, {})


def _as_level(identity, level: str, **changes):
    """identity as a level run under the same chain carries it: its own lock only."""
    licence = {stem: sha for stem, sha in identity.licence.items()
               if stem in (prov.INTEGRATION_RECEIPT_STEM, prov.OVERFIT_RECEIPT_STEM)}
    licence[checkpoint_lock_path(Path("."), level).stem] = "1" * 64
    return dataclasses.replace(identity, level=level, licence=licence, **changes)


def test_a_level_under_the_primary_chain_is_bound_to_it(run):
    """reporting_rules.md decision 4: affine trains under the same chain, at E,
    under the one integration and the one overfit receipt. Its own lock is the
    only receipt it does not share with the primary level."""
    identity = require(run)
    primary = prov.output_run_record(identity, "phase5_tables_manifest")
    prov.require_primary_chain(primary, _as_level(identity, "affine"))


@pytest.mark.parametrize("changes, field", [
    ({"commit": "0" * 40}, "evaluation_commit"),
    ({"training_commit": "0" * 40}, "training_commit"),
    ({"config_digest": "0" * 64}, "config_digest"),
    ({"fold_digest": "0" * 64}, "fold_digest"),
    ({"measurement_digest": "0" * 64}, "measurement_digest"),
    ({"mean_vector_digest": "0" * 64}, "mean_vector_digest"),
    ({"analysis_reporting_digest": "0" * 64}, "analysis_reporting_digest"),
    ({"eval_version": 99}, "eval_version"),
    ({"phase4_commit": "5" * 40}, "phase4_commit"),
    ({"seeds": (0, 1)}, "seeds"),
    ({"folds": (0,)}, "folds"),
    ({"scenes": ("other",)}, "scenes"),
    ({"bootstrap": {"primary_unit": "scene", "resamples": 7}}, "bootstrap"),
])
def test_a_level_under_another_chain_is_refused(run, changes, field):
    identity = require(run)
    primary = prov.output_run_record(identity, "phase5_tables_manifest")
    with pytest.raises(ProvenanceError) as error:
        prov.require_primary_chain(primary, _as_level(identity, "affine", **changes))
    assert error.value.fields == (field,), str(error.value)
    assert "primary level's chain" in str(error.value)


@pytest.mark.parametrize("stem", ["integration_gate", "tiny_overfit"])
def test_a_level_licensed_by_another_gate_receipt_is_refused(run, stem):
    """The overfit gate runs once, at the primary level, so a level whose
    overfit or integration receipt is another one ran under another chain."""
    identity = require(run)
    primary = prov.output_run_record(identity, "phase5_tables_manifest")
    level = _as_level(identity, "affine")
    level = dataclasses.replace(level, licence={**level.licence, stem: "0" * 64})
    with pytest.raises(ProvenanceError) as error:
        prov.require_primary_chain(primary, level)
    assert error.value.fields == (f"licence.{stem}",), str(error.value)


def test_a_checked_level_is_not_bound_to_primary_tables_built_unchecked(run):
    identity = require(run)
    primary = prov.output_run_record(identity, "phase5_tables_manifest")
    assert primary["code_checked"] is False
    with pytest.raises(ProvenanceError) as error:
        prov.require_primary_chain(primary, _as_level(identity, "affine", code_checked=True))
    assert error.value.fields == ("code_checked",), str(error.value)


def test_the_staging_name_is_public_and_matches_only_a_staging_directory():
    """lot.phase5_acceptance skips a reporting build's staging directory by this
    pattern, and nothing else that merely holds .partial in its name."""
    staging = f"image.partial.{'a' * 32}"
    assert prov.STAGING_NAME.match(staging)["final"] == "image"
    for name in ("image.partial", "image.partial.old", f"image.partial.{'a' * 31}",
                 f"image.partial.{'A' * 32}", "image.old"):
        assert prov.STAGING_NAME.match(name) is None, name


def test_a_parquet_with_its_record_is_written_once_and_whole(tmp_path):
    path = tmp_path / "tables" / "phase5_primary.parquet"
    rows = [{"metric": "centered", "delta_learn_pp": 0.0125, "n_scenes": 18},
            {"metric": "raw", "delta_learn_pp": -0.003, "n_scenes": 18}]
    record = {"kind": "phase5_primary", "inputs": {"a": "b"}}
    prov.write_parquet_with_record(path, rows, record)
    assert read_rows(path) == rows
    assert read_run_metadata(path) == record
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]

    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        prov.write_parquet_with_record(path, rows[:1], record)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="no rows"):
        prov.write_parquet_with_record(tmp_path / "empty.parquet", [], record)


def test_a_failed_parquet_write_leaves_nothing(tmp_path, monkeypatch):
    def broken(table, where, *args, **kwargs):
        Path(where).write_bytes(b"half a parquet")
        raise OSError("the disk filled")

    monkeypatch.setattr(pq, "write_table", broken)
    path = tmp_path / "tables" / "phase5_primary.parquet"
    with pytest.raises(OSError, match="disk filled"):
        prov.write_parquet_with_record(path, [{"a": 1}], {"kind": "x"})
    assert list(path.parent.iterdir()) == []


# ---------------------------------------------------------------------------
# Publishing a directory
# ---------------------------------------------------------------------------

def _staging_with(final: Path, text: str) -> Path:
    staging = prov.new_staging_directory(final)
    (staging / "table.txt").write_text(text, encoding="utf-8")
    return staging


def test_a_staging_directory_is_published_by_one_rename(tmp_path):
    final = tmp_path / "tables" / "image"
    staging = _staging_with(final, "first")
    assert staging.parent == final.parent and staging.name.startswith("image.partial.")
    outcome = prov.publish_directory(staging, final)
    assert outcome == {"published": str(final), "superseded": None}
    assert not staging.exists()
    assert (final / "table.txt").read_text(encoding="utf-8") == "first"


def test_an_existing_output_is_refused_unless_superseded(tmp_path):
    final = tmp_path / "tables" / "image"
    prov.publish_directory(_staging_with(final, "first"), final)
    staging = _staging_with(final, "second")
    with pytest.raises(FileExistsError):
        prov.publish_directory(staging, final)
    assert (final / "table.txt").read_text(encoding="utf-8") == "first"
    assert (staging / "table.txt").read_text(encoding="utf-8") == "second"

    outcome = prov.publish_directory(staging, final, supersede=True)
    archived = final.with_name("image.superseded.1")
    assert outcome == {"published": str(final), "superseded": str(archived)}
    assert (archived / "table.txt").read_text(encoding="utf-8") == "first"
    assert (final / "table.txt").read_text(encoding="utf-8") == "second"

    prov.publish_directory(_staging_with(final, "third"), final, supersede=True)
    assert (final.with_name("image.superseded.2") / "table.txt").read_text(
        encoding="utf-8") == "second"
    assert (final / "table.txt").read_text(encoding="utf-8") == "third"


def test_publishing_needs_a_sibling_staging_directory(tmp_path):
    final = tmp_path / "tables" / "image"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(ValueError, match="parent"):
        prov.publish_directory(elsewhere, final)
    with pytest.raises(FileNotFoundError):
        prov.publish_directory(final.with_name("image.partial.absent"), final)
    assert not final.exists()


def test_a_staged_output_publishes_on_success_and_leaves_nothing_on_failure(tmp_path):
    final = tmp_path / "tables" / "image"
    with pytest.raises(RuntimeError, match="a table failed"):
        with prov.staged_output(final) as staged:
            (staged.path / "table.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("a table failed")
    assert not final.exists() and list(final.parent.iterdir()) == []

    with prov.staged_output(final) as staged:
        (staged.path / "table.txt").write_text("whole", encoding="utf-8")
    assert staged.outcome == {"published": str(final), "superseded": None}
    assert (final / "table.txt").read_text(encoding="utf-8") == "whole"

    # An existing output is refused before any work, unless superseded.
    with pytest.raises(FileExistsError):
        with prov.staged_output(final):
            raise AssertionError("the body must not run")
    with prov.staged_output(final, supersede=True) as staged:
        (staged.path / "table.txt").write_text("rebuilt", encoding="utf-8")
    assert staged.outcome["superseded"] == str(final.with_name("image.superseded.1"))


# ---------------------------------------------------------------------------
# Path classes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path, expected", [
    ("src/lot/phase5_estimands.py", "frozen_at_e"),
    ("src/lot/paired_bootstrap.py", "frozen_at_e"),
    ("src/lot/phase5_outcomes.py", "frozen_at_e"),
    ("validation/evidence/phase5/reporting_rules.md", "frozen_at_e"),
    ("src/lot/phase5_modes.py", "measurement"),
    ("src/lot/context_lift.py", "measurement"),
    ("src/lot/sample_identity.py", "measurement"),
    ("src/lot/phase4.py", "measurement"),
    ("configs/phase5.yaml", "measurement"),
    ("configs/analysis.yaml", "measurement"),
    ("PROTOCOL.md", "measurement"),
    ("AMENDMENTS.md", "measurement"),
    ("VALIDATION.md", "measurement"),
    ("FREEZE.md", "measurement"),
    ("src/lot/phase5_report.py", "reporting"),
    ("src/lot/phase5_figures.py", "reporting"),
    ("src/lot/phase5_acceptance.py", "reporting"),
    ("src/lot/phase5_provenance.py", "reporting"),
    ("scripts/run_phase5.sh", "reporting"),
    ("scripts/phase5_readout.py", "reporting"),
    ("tests/test_phase5_modes.py", "neutral"),
    ("FINDINGS.md", "neutral"),
    ("scripts/BORAH_PHASE5.md", "neutral"),
    ("validation/evidence/phase5/post_evaluation_changes.md", "neutral"),
    ("validation/evidence/phase5/verdict.json", "neutral"),
    ("src/lot/phase5_gate.py", "other"),
    ("src/lot/phase5_check.py", "other"),
    ("src/lot/phase5_receipt.py", "other"),
    ("src/lot/figures.py", "other"),
    ("scripts/phase4_readout.py", "other"),
    ("scripts/phase5_job.sbatch", "other"),
    ("pyproject.toml", "other"),
    ("validation/mutants/3.0_control_unmutated/configs/analysis.yaml", "other"),
])
def test_every_path_falls_in_one_class(path, expected):
    """The first class that names a path wins: frozen at E, measurement,
    reporting, neutral. A path in none of them is other, and stops."""
    assert prov.classify_path(path) == expected


def test_every_named_measurement_file_exists():
    for pattern in prov.MEASUREMENT_PATHS:
        matches = [p for p in REPO.glob(pattern) if p.is_file()]
        assert matches, f"{pattern} names nothing in the repository"


def _lot_imports(path: Path, everywhere: bool) -> set[str]:
    """The lot modules a file imports, at module level or everywhere."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = ast.walk(tree) if everywhere else tree.body
    found: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:
                found.add(node.module.split(".")[0])
            elif node.level == 1:
                found.update(alias.name for alias in node.names)
            elif node.module and node.module.startswith("lot."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            found.update(alias.name.split(".")[1] for alias in node.names
                         if alias.name.startswith("lot."))
    return found


def test_nothing_the_evaluation_imports_is_a_reporting_or_neutral_path():
    """Every module evaluation reads is in a class whose change after E stops.

    The walk follows module-level imports from lot.phase5_modes, where the
    evaluate mode lives, and every import inside phase5_modes itself.
    """
    source = REPO / "src" / "lot"
    seen: set[str] = set()
    queue = ["phase5_modes"]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        queue.extend(_lot_imports(source / f"{name}.py", everywhere=name == "phase5_modes"))
    assert {"phase5", "phase5_score", "phase5_reference", "phase5_folds", "context_lift",
            "sample_identity", "predictors", "train", "evaluate", "phase4"} <= seen
    loose = {name: prov.classify_path(f"src/lot/{name}.py") for name in sorted(seen)}
    loose = {name: cls for name, cls in loose.items() if cls in ("reporting", "neutral")}
    assert loose == {}


def test_the_classes_are_read_back_from_the_module_source():
    """Code ancestry reads the classes of commit E from E's copy of this module."""
    source = Path(prov.__file__).read_text(encoding="utf-8")
    assert prov.classes_from_source(source) == {
        "frozen_at_e": prov.FROZEN_AT_E,
        "measurement": prov.MEASUREMENT_PATHS,
        "reporting": prov.REPORTING_PATHS,
        "neutral": prov.NEUTRAL_PATHS,
    }
    widened = source + "\nMEASUREMENT_PATHS = " + repr(prov.MEASUREMENT_PATHS + ("X.md",))
    assert prov.classes_from_source(widened)["measurement"][-1] == "X.md"
    with pytest.raises(ValueError, match="NEUTRAL_PATHS"):
        prov.classes_from_source(source + "\nNEUTRAL_PATHS = tuple(sorted(NEUTRAL_PATHS))")


def test_the_module_text_uses_no_em_dash_and_no_letter_method_labels():
    source = Path(prov.__file__).read_text(encoding="utf-8")
    assert "\u2014" not in source
    assert not re.search(r"\bmethod [ABC]\b", source)
    assert source.isascii()


def test_the_post_evaluation_listing_reads_each_path_with_its_reason():
    text = "\n".join([
        "# Changes to reporting files after commit E",
        "",
        "Prose that names `src/lot/phase5_figures.py` outside a list lists nothing.",
        "",
        "- `src/lot/phase5_report.py`: a table label named the wrong method.",
        "* `scripts/run_phase5.sh` - print the run identity before the tables.",
        "- `src/lot/phase5_acceptance.py`",
        "- `./src/lot/phase5_report.py`: a second change, to a column order.",
    ])
    assert prov.post_evaluation_listing(text) == {
        "src/lot/phase5_report.py":
            "a table label named the wrong method.; a second change, to a column order.",
        "scripts/run_phase5.sh": "print the run identity before the tables.",
        "src/lot/phase5_acceptance.py": "",
    }


# ---------------------------------------------------------------------------
# 6. Code ancestry, on temporary git repositories
# ---------------------------------------------------------------------------

# The smallest tree that holds one file of every class.
BASE_TREE = {
    "src/lot/phase5_estimands.py": "# estimands\n",
    "src/lot/paired_bootstrap.py": "# bootstrap\n",
    "src/lot/phase5_outcomes.py": "# outcomes\n",
    "validation/evidence/phase5/reporting_rules.md": "# rules\n",
    "src/lot/phase5_modes.py": "# modes\n",
    "src/lot/phase5_report.py": "# tables\n",
    "src/lot/phase5_gate.py": "# gate\n",
    "configs/phase5.yaml": "seed: 0\n",
    "PROTOCOL.md": "# protocol\n",
    "tests/test_something.py": "# a test\n",
    "FINDINGS.md": "# findings\n",
}
PROVENANCE = "src/lot/phase5_provenance.py"
LISTING = "validation/evidence/phase5/post_evaluation_changes.md"


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                          text=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def write_tree(repo: Path, files: dict[str, str | None]) -> None:
    """Write each file, or remove it where the content is None."""
    for relative, text in files.items():
        path = repo / relative
        if text is None:
            path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


def commit(repo: Path, files: dict[str, str | None] | None = None,
           message: str = "a change") -> str:
    write_tree(repo, files or {})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def make_repo(tmp_path: Path, files: dict[str, str | None] | None = None) -> tuple[Path, str]:
    """A repository whose first commit is E: the base tree, this module, and files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for key, value in (("user.name", "Phase 5 test"),
                       ("user.email", "phase5@example.invalid"),
                       ("core.autocrlf", "false"), ("commit.gpgsign", "false"),
                       ("core.hooksPath", "no-hooks")):
        git(repo, "config", key, value)
    source = Path(prov.__file__).read_text(encoding="utf-8")
    tree = {**BASE_TREE, PROVENANCE: source, **(files or {})}
    tree = {path: text for path, text in tree.items() if text is not None}
    return repo, commit(repo, tree, "E")


def test_a_report_at_e_itself_passes(tmp_path):
    repo, e = make_repo(tmp_path)
    assert prov.verify_code_ancestry(e, repo) == (e, {})


def test_neutral_changes_after_e_pass(tmp_path):
    repo, e = make_repo(tmp_path)
    head = commit(repo, {"tests/test_something.py": "# a better test\n",
                         "FINDINGS.md": "# findings, Phase 5\n",
                         "validation/evidence/phase5/verdict.md": "# verdict\n"})
    assert prov.verify_code_ancestry(e, repo) == (head, {})


@pytest.mark.parametrize("path, field", [
    ("src/lot/phase5_modes.py", "measurement_paths"),
    ("configs/phase5.yaml", "measurement_paths"),
    ("PROTOCOL.md", "measurement_paths"),
    ("src/lot/phase5_estimands.py", "frozen_at_e"),
    ("src/lot/phase5_outcomes.py", "frozen_at_e"),
    ("validation/evidence/phase5/reporting_rules.md", "frozen_at_e"),
    ("src/lot/phase5_gate.py", "other_paths"),
])
def test_a_change_that_no_listing_can_license_is_refused(tmp_path, path, field):
    repo, e = make_repo(tmp_path)
    commit(repo, {path: "# changed after E\n", LISTING: f"- `{path}`: tried anyway.\n"})
    with pytest.raises(ProvenanceError) as error:
        prov.verify_code_ancestry(e, repo)
    assert error.value.fields == (field,), str(error.value)
    assert path in str(error.value)


def test_an_unlisted_reporting_change_is_refused(tmp_path):
    repo, e = make_repo(tmp_path)
    commit(repo, {"src/lot/phase5_report.py": "# tables, relabelled\n"})
    with pytest.raises(ProvenanceError) as error:
        prov.verify_code_ancestry(e, repo)
    assert error.value.fields == ("reporting_paths",)
    assert "src/lot/phase5_report.py" in str(error.value)


def test_a_listed_reporting_change_passes_with_its_reason(tmp_path):
    repo, e = make_repo(tmp_path)
    head = commit(repo, {
        "src/lot/phase5_report.py": "# tables, relabelled\n",
        LISTING: "# Post-evaluation changes\n\n"
                 "- `src/lot/phase5_report.py`: a column label named the wrong method.\n",
    })
    assert prov.verify_code_ancestry(e, repo) == (
        head, {"src/lot/phase5_report.py": "a column label named the wrong method."}
    )


def test_a_reporting_change_listed_without_its_reason_is_refused(tmp_path):
    repo, e = make_repo(tmp_path)
    commit(repo, {"src/lot/phase5_report.py": "# tables, relabelled\n",
                  LISTING: "- `src/lot/phase5_report.py`\n"})
    with pytest.raises(ProvenanceError, match="reason") as error:
        prov.verify_code_ancestry(e, repo)
    assert error.value.fields == ("reporting_paths",)


def test_a_rename_names_both_paths(tmp_path):
    """A measurement file moved to a neutral path is a measurement change."""
    repo, e = make_repo(tmp_path)
    git(repo, "mv", "src/lot/phase5_modes.py", "tests/phase5_modes_moved.py")
    commit(repo)
    with pytest.raises(ProvenanceError) as error:
        prov.verify_code_ancestry(e, repo)
    assert "measurement_paths" in error.value.fields
    assert "src/lot/phase5_modes.py" in str(error.value)


@pytest.mark.parametrize("dirt", [
    {"notes.txt": "untracked\n"},
    {"FINDINGS.md": "# edited, not committed\n"},
], ids=["untracked", "modified"])
def test_a_dirty_worktree_is_refused(tmp_path, dirt):
    repo, e = make_repo(tmp_path)
    write_tree(repo, dirt)
    with pytest.raises(ProvenanceError) as error:
        prov.verify_code_ancestry(e, repo)
    assert error.value.fields == ("worktree",)


def test_e_must_be_an_ancestor_of_head(tmp_path):
    repo, base = make_repo(tmp_path)
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    git(repo, "checkout", "-q", "-b", "side")
    e = commit(repo, {"FINDINGS.md": "# measured on a side branch\n"})
    git(repo, "checkout", "-q", branch)
    commit(repo, {"FINDINGS.md": "# reported on the main line\n"})
    with pytest.raises(ProvenanceError, match="not an ancestor") as error:
        prov.verify_code_ancestry(e, repo)
    assert error.value.fields == ("commit",)


def test_e_must_exist(tmp_path):
    repo, _ = make_repo(tmp_path)
    with pytest.raises(ProvenanceError, match="does not exist") as error:
        prov.verify_code_ancestry("0" * 40, repo)
    assert error.value.fields == ("commit",)


def test_a_frozen_file_absent_at_e_is_refused(tmp_path):
    """Decision 1 freezes the outcome code at E. A class member that does not
    exist at E froze nothing, so the class is not what decision 1 asks."""
    repo, e = make_repo(tmp_path, {"src/lot/phase5_outcomes.py": None})
    with pytest.raises(ProvenanceError, match="phase5_outcomes.py") as error:
        prov.verify_code_ancestry(e, repo)
    assert error.value.fields == ("frozen_at_e",)


def test_e_must_carry_its_path_classes(tmp_path):
    """The reporting code, this module with it, is committed before the chain."""
    repo, e = make_repo(tmp_path, {PROVENANCE: None})
    commit(repo, {PROVENANCE: Path(prov.__file__).read_text(encoding="utf-8"),
                  LISTING: f"- `{PROVENANCE}`: added after E.\n"})
    with pytest.raises(ProvenanceError, match="absent at E") as error:
        prov.verify_code_ancestry(e, repo)
    assert "classification" in error.value.fields


def test_the_classes_at_e_still_bind_after_e(tmp_path):
    """A path E's classes call measurement stays measurement, whatever R says.

    At E this module also named FINDINGS.md as a measurement path. R restores
    the ordinary classes, listed as a reporting change, and edits FINDINGS.md.
    The edit is refused, so a post-evaluation edit to the classes cannot
    license a change E's classes forbid.
    """
    source = Path(prov.__file__).read_text(encoding="utf-8")
    strict = (source + "\nMEASUREMENT_PATHS = "
              + repr(prov.MEASUREMENT_PATHS + ("FINDINGS.md",)) + "\n")
    repo, e = make_repo(tmp_path, {PROVENANCE: strict})
    commit(repo, {PROVENANCE: source, "FINDINGS.md": "# findings, edited at R\n",
                  LISTING: f"- `{PROVENANCE}`: restore the ordinary classes.\n"})
    with pytest.raises(ProvenanceError) as error:
        prov.verify_code_ancestry(e, repo)
    assert error.value.fields == ("measurement_paths",)
    assert "FINDINGS.md" in str(error.value)


def test_the_reporting_modes_check_code_ancestry_by_default(tmp_path, monkeypatch):
    """require_evaluated_run, through to the repository the run names as E."""
    repo, e = make_repo(tmp_path)
    run = build_run(tmp_path / "run", monkeypatch, commit=e)
    identity = require(run, check_code=True, repo_root=repo)
    assert identity.code_checked is True
    assert identity.report_commit == e and identity.reporting_changes == {}

    commit(repo, {"src/lot/phase5_report.py": "# tables, relabelled\n"})
    refused(run, "reporting_paths", check_code=True, repo_root=repo)
    head = commit(repo, {LISTING: "- `src/lot/phase5_report.py`: relabelled a column.\n"})
    identity = require(run, check_code=True, repo_root=repo)
    assert identity.report_commit == head
    assert identity.reporting_changes == {"src/lot/phase5_report.py": "relabelled a column."}
    record = prov.output_run_record(identity, "phase5_primary", {}, {})
    assert record["report_commit"] == head and record["code_checked"] is True
    assert record["reporting_changes"] == identity.reporting_changes
