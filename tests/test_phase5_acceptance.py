"""Phase 5 acceptance, Stream AD: every condition re-derived from the artifacts.

reporting_rules.md section 8. Acceptance re-derives the specification's
eighteen conditions from the shipped artifacts, and adds three: the stable
validation curve of decision 5, the headline figures present and built from
the current tables, and the full suite green at the reporting commit.

A complete synthetic artifact set is built once, in the layout the chain
writes it. A temporary git repository holds the code, the documents, and the
tests at commit E. The gate, overfit, and lock receipts, nine training records with
their checkpoints, the controls file, three evaluation parquets with their run
records, and the evaluation ledger are written where the modes write them. The
run records come from lot.phase5_modes.evaluation_metadata itself. The tables
and figures are published by the tables and figures modes, with the code
checked against the repository. Every condition must pass on it.

Each mutation then damages one thing, as a real defect would, and exactly the
condition that owns it must fail. A mutation of an upstream artifact rebuilds
everything downstream of it consistently, so the one thing the test names is
the only thing wrong. A rebuild reuses the base tables' rows, which do not
change, recomputes the two tables that read the embedded training records, and
draws placeholder figures.

Three things are injected. The integration receipt's binding to the cluster
inputs is replaced, as tests/test_phase5_provenance.py replaces it. The full
suite and the Phase 4 acceptance check are passed in, because running them
here would recurse into this suite and needs the cluster's Phase 4 run.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import lot.phase5_acceptance as acceptance  # noqa: E402
import lot.phase5_figures as figures  # noqa: E402
import lot.phase5_modes as modes  # noqa: E402
import lot.phase5_receipt as receipt_module  # noqa: E402
import lot.phase5_report as report  # noqa: E402
import test_phase5_report as report_tests  # noqa: E402
from lot.evaluate import read_rows, read_run_metadata, vector_digest, write_rows  # noqa: E402
from lot.phase5 import load_phase5_config, predictor_config_from  # noqa: E402
from lot.phase5_check import (  # noqa: E402
    check_test_seal,
    sha256_file,
    splat_symmetry_evidence,
    utc_timestamp,
)
from lot.phase5_folds import FROZEN_FOLD_DIGEST, fold_digest, fold_of_test_scene  # noqa: E402
from lot.phase5_gate import rotation_regime_summary, rotation_scene_summary  # noqa: E402
from lot.phase5_modes import (  # noqa: E402
    checkpoint_lock_path,
    checkpoint_path,
    controls_path,
    controls_payload,
    evaluation_metadata,
    fold_seed_key,
    training_record_path,
)
from lot.phase5_provenance import write_parquet_with_record  # noqa: E402
from lot.predictors import build_predictor  # noqa: E402
from lot.render_replica import REPLICA_SCENES  # noqa: E402
from lot.train import training_config_from  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "phase5.yaml"
LEVEL = "image"
FAST = report_tests.FAST
FOLDS = report_tests.FOLDS
SCENES = list(report_tests.SCENES)
SEEDS = list(report_tests.SEEDS)
REGIMES = ("rotation", "translation", "orbit")
# The Phase 4 commit the synthetic Phase 4 run records.
P4 = "4" * 40
CENTER = torch.linspace(-1.0, 1.0, 768)
MEAN_VECTOR = vector_digest(CENTER.numpy())
PNG = b"\x89PNG\r\n\x1a\n"
ALIGNED = {scene: hashlib.sha256(f"aligned depth of {scene}".encode()).hexdigest()
           for scene in REPLICA_SCENES}
FORWARD = ["features_context", "depth_context_aligned", "camera", "context_valid"]
EXAMPLE_FIELDS = ["scene", "context_frame_id", "target_frame_id", "regime",
                  "features_context", "depth_context_aligned", "camera", "context_valid",
                  "query_patch_coords", "target_centered", "support"]
PARAMETERS = 16680960
LETTER_LABEL = re.compile(r"(?<![A-Za-z0-9_])(method|Method) [ABC](?![A-Za-z0-9_])")

# What commit E holds: the code, the configurations, the documents, and the
# tests, whose results at R are evidence only for the tests registered at E.
REPO_GLOBS = ("src/lot/*.py", "configs/*.yaml", "validation/evidence/phase5/*.md",
              "tests/*.py")
REPO_DOCUMENTS = ("PROTOCOL.md", "AMENDMENTS.md", "VALIDATION.md", "FREEZE.md",
                  "FINDINGS.md", "PLAN.md", ".gitattributes", ".gitignore")


# ---------------------------------------------------------------------------
# A repository, and the Phase 4 run the evaluation reconciled against
# ---------------------------------------------------------------------------

def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                          text=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def commit_files(repo: Path, files: dict[str, str], message: str) -> str:
    for relative, text in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def make_repository(root: Path) -> tuple[Path, str]:
    """A repository whose one commit, E, holds the code and the documents.

    Files are copied from this checkout with LF line ends, the bytes git stores,
    so the frozen documents hash to FREEZE.md's values.
    """
    repo = root / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    for key, value in (("user.name", "Phase 5 test"),
                       ("user.email", "phase5@example.invalid"),
                       ("core.autocrlf", "false"), ("commit.gpgsign", "false"),
                       ("core.hooksPath", "no-hooks")):
        git(repo, "config", key, value)
    files: dict[str, str] = {}
    for pattern in REPO_GLOBS:
        for path in sorted(REPO.glob(pattern)):
            files[path.relative_to(REPO).as_posix()] = path.read_bytes().replace(
                b"\r\n", b"\n").decode("utf-8")
    for name in REPO_DOCUMENTS:
        files[name] = (REPO / name).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    return repo, commit_files(repo, files, "E")


def make_phase4(root: Path) -> Path:
    """The accepted Phase 4 run's parquets, each carrying its run record."""
    phase4 = root / "phase4"
    for scene in SCENES:
        write_rows(phase4 / "eval" / f"{scene}.parquet", [{"scene": scene, "level": LEVEL}],
                   {"git_commit": P4, "mean_vector_digest": MEAN_VECTOR})
    return phase4


# ---------------------------------------------------------------------------
# The receipts, in the layout the gate and the modes write them
# ---------------------------------------------------------------------------

def rotation_pair(scene: str, index: int) -> dict:
    """One pair's step 17 evidence, as lot.phase5_gate.rotation_pair_evidence returns it."""
    comparison = {"n_compared": 800, "max_residual_px": 6.0e-5,
                  "max_translation_allowance_px": 2.0e-4, "min_headroom_px": 9.0e-4}
    return {"pair": f"{scene} rotation_{index:02d}_c -> rotation_{index:02d}_t level image",
            "translation_norm_m": 2.0e-7, "n_context_patches": 1369, "n_tl_samples": 900,
            "context_lift": dict(comparison), "depth_substitution": dict(comparison),
            "tl_reference": dict(comparison), "checked": True}


def rotation_evidence() -> dict:
    """Step 17 over every scene: each evaluated scene holds the rows' rotation pairs."""
    per_scene = {}
    for scene in REPLICA_SCENES:
        count = report_tests.PER_REGIME if scene in SCENES else 5
        per_scene[scene] = rotation_scene_summary(
            scene, [rotation_pair(scene, index) for index in range(count)], [])
    return rotation_regime_summary(per_scene, FAST.rotation_gate_coord_tol_px,
                                   FAST.rotation_position_bound_m)


def integration_report(identity: dict, phase4: Path) -> dict:
    """The gate's report: seventeen steps, in the order the gate runs them."""
    steps: list[dict] = []

    def step(number: str, title: str, evidence: dict) -> None:
        steps.append({"step": number, "title": title, "passed": True, "evidence": evidence})

    live = {scene: sha256_file(phase4 / "eval" / f"{scene}.parquet") for scene in SCENES}
    identities = {
        scene: {"phase4_parquet_sha256": live.get(scene,
                                                  hashlib.sha256(scene.encode()).hexdigest()),
                "phase4_parquet_bytes": 1000, "phase4_commit": P4,
                "features_digest": "f" * 32, "depth_digest": "d" * 32,
                "manifest_digest": "e" * 64, "accepted_mean_vector_digest": MEAN_VECTOR,
                "aligned_depth_digest": ALIGNED[scene]}
        for scene in REPLICA_SCENES
    }
    step("1", "script identity and environment",
         {**identity, "primary_alignment_level": LEVEL, "seeds": SEEDS,
          "environment": {"python": "3.11.9"}})
    step("2", "resolve real Borah artifacts",
         {"artifacts": {"renders_root": {"path": "/scratch/renders"}},
          "scene_identities": identities})
    step("3", "verify real schemas",
         {"dino_features": {"type": "torch.Tensor", "shape": [768, 37, 37],
                            "dtype": "torch.float16", "all_finite": True}})
    step("4", "build real scene inputs, three families",
         {"scenes": {scene: {"n_frames": 48, "convention": "planar_z",
                             "aligned_depth_digest": ALIGNED[scene]} for scene in REPLICA_SCENES},
          "single_convention": "planar_z"})
    step("5", "one real example per regime, no forbidden fields",
         {"model_visible_fields": FORWARD,
          "per_regime": {regime: {"pair": "000 -> 004", "n_supported": 700,
                                  "fields": EXAMPLE_FIELDS} for regime in REGIMES}})
    step("6", "headline landing-location semantics",
         {"samples": [], "n_landed": 900, "n_supported": 700,
          "max_abs_coordinate_difference": 0.0})
    step("7", "primary support V_P5_pp and predictor invariance",
         {"counts": {"candidate_samples": 1369, "depth_valid": 1369, "landed": 1100,
                     "gt_evaluable": 760, "final_support": 700},
          "cl_centered": 0.62, "nowarp_centered": 0.41,
          "all_nonfinite_predictor_score": -1.0, "failures_counted": 700,
          "cross_path": {"n_intersect": 52, "max_samples_per_cell": 3,
                         "x_cl_centered": 0.6, "x_sp_transport_centered": 0.61}})
    step("8", "formulation support V_form on target cells",
         {"common_cells": 40, "tl_samples_per_common_cell": 1})
    step("9", "splat-pool information symmetry on real code", splat_symmetry_evidence())
    step("10", "real-data geometry checks",
         {"rotation": {"n_landed_samples": 1000, "n_context_patches": 1369,
                       "max_homography_residual_px": 6.0e-5, "tolerance_px": 1.0e-3,
                       "max_depth_substitution_shift_px": 4.0e-5},
          "translation": {"n_samples": 64,
                          "max_independent_reprojection_residual_px": 3.0e-5}})
    step("11", "frozen folds against the real scene inventory",
         {"n_scenes": len(REPLICA_SCENES), "fold_digest": fold_digest(FOLDS),
          "folds": [{"index": f.index, "train": list(f.train), "val": list(f.val),
                     "test": list(f.test)} for f in FOLDS]})
    step("12", "dry-run one real batch: forward, loss, backward",
         {"loss": 0.41, "parameter_count": PARAMETERS, "optimizer_constructed": False,
          "n_parameters_with_gradients": 137})
    step("13", "resource probe against the frozen batch",
         {"device": "cuda", "frozen_batch_pairs": 8})
    step("14", "test seal", {**check_test_seal(), "test_evaluation_requires_explicit_mode": True})
    step("17", "pure-rotation gate across the regime", rotation_evidence())
    step("15", "complete the deferred pin",
         {"written": "outputs/phase5_rung2/evidence/pin_cluster.json",
          "archived_previous": None, "keys": ["config_digest", "split_hash"]})
    step("16", "verdict and no-side-effect assertion",
         {"checkpoints_before": 0, "checkpoints_after": 0,
          "pin": "outputs/phase5_rung2/evidence/pin_cluster.json"})
    return {"passed": True, "environment": {"python": "3.11.9"}, "steps": steps,
            "failure": None}


def overfit_report() -> dict:
    """The overfit gate's verdict: eight training pairs of fold 0, every regime."""
    train = FOLDS[0].train
    return {
        "passed": True, "reached_centered_cosine": 0.9917, "threshold": 0.98, "steps": 1460,
        "n_pairs": 8, "regimes": list(REGIMES), "fold": 0, "level": LEVEL, "seed": 0,
        "subset": [{"scene": train[index % len(train)], "context_frame_id": f"{index:03d}",
                    "target_frame_id": f"{index + 4:03d}", "regime": REGIMES[index % 3],
                    "n_supported": 800} for index in range(8)],
    }


def stamp(payload: dict, kind: str, identity: dict, when: str, **digests) -> dict:
    """A receipt as lot.phase5_receipt.stamp_receipt writes it."""
    return {**payload, "kind": kind, **identity, **digests, "stamped_utc": when}


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Training records and their checkpoints
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def compact_state() -> dict:
    """The frozen architecture's state, every tensor a view of one stored value.

    Each tensor has the shape the real model has, so its parameter count is the
    frozen 16,680,960, while each checkpoint stays a few kilobytes on disk.
    """
    model = build_predictor(predictor_config_from(load_phase5_config(CONFIG), (518, 518)))
    return {name: torch.zeros((1,) * value.dim()).expand(value.shape)
            for name, value in model.state_dict().items()}


def training_history(fold: int, seed: int) -> list[list]:
    """A run that improved, then stopped after the frozen patience without improving."""
    best = 4 + fold + seed
    scores = [0.40 + 0.03 * index for index in range(best + 1)]
    scores += [scores[-1] - 0.0004 * (later + 1) for later in range(10)]
    return [[500 * (index + 1), score] for index, score in enumerate(scores)]


def training_record(cfg, fold, seed: int, checkpoint: Path, commit: str, gates: dict,
                    written: str) -> dict:
    """A training record in the layout lot.phase5_modes.run_train_task writes."""
    history = training_history(fold.index, seed)
    best_step, best = max(history, key=lambda entry: entry[1])
    return {
        "fold": fold.index, "seed": seed,
        "train_scenes": list(fold.train), "val_scenes": list(fold.val),
        "test_scenes": list(fold.test),
        "steps_run": history[-1][0], "best_step": best_step,
        "best_validation_centered_cosine": best, "parameter_count": PARAMETERS,
        "training_config_digest": training_config_from(cfg.training).digest(),
        "stopped_early": True, "history": history,
        "level": LEVEL, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "n_train_examples": 2600, "n_val_examples": 800,
        "census": [{"scene": scene, "level": LEVEL, "n_pairs": 300, "n_planned": 290,
                    "n_no_arm": 0, "n_empty_support": 10} for scene in fold.train + fold.val],
        "superseded": {"checkpoint": None, "record": None},
        "config_digest": cfg.digest(), "commit": commit, "licence": dict(gates),
        "train_scenes_planned": list(fold.train), "val_scenes_planned": list(fold.val),
        "written_utc": written,
    }


def control_result(fold) -> dict:
    return {
        "n_pairs": 800, "val_scenes_planned": list(fold.val),
        "pose_shuffle": {"baseline_centered_cosine": 0.6, "shuffled_centered_cosine": 0.4,
                         "degradation": 0.2, "n_samples": 90000, "n_unchanged": 0},
        "depth_shuffle": {"baseline_centered_cosine": 0.6, "shuffled_centered_cosine": 0.59,
                          "degradation": 0.01, "n_samples": 90000, "n_unchanged": 3},
    }


def placeholder_figures(published, directory, analysis, drawn) -> None:
    """Placeholder PNGs, for rebuilds whose figures do not need drawing."""
    for name in drawn:
        (Path(directory) / name).write_bytes(PNG + f"placeholder {name}".encode("utf-8"))


def reuse_tables(base_set):
    """build_tables, reusing the base rows and recomputing what reads the evidence.

    Every rebuild evaluates the same synthetic rows, so every cell table is the
    base's. The adequacy and history tables read the embedded training records,
    which a mutation may change, so they are computed again.
    """
    def build(prepared, analysis, *, adequacy=None, training=None, phase4=None):
        tables = dict(base_set.tables)
        tables[report.ADEQUACY_TABLE] = report.adequacy_table(
            adequacy, analysis, level=prepared.level, primary_level=prepared.primary_level,
            training=training)
        tables[report.HISTORY_TABLE] = report.validation_history_table(
            adequacy, level=prepared.level)
        return dataclasses.replace(base_set, tables=tables)
    return build


# ---------------------------------------------------------------------------
# One complete evaluated, reported run
# ---------------------------------------------------------------------------

def build_world(root: Path, repo: Path, commit: str, phase4: Path, *, hooks: dict | None = None,
                base_set=None, draw: bool = False) -> SimpleNamespace:
    """Every artifact the chain and the reporting modes write, at commit.

    hooks change an artifact in place before it is written, and everything
    downstream is derived from the changed artifact:

    - integration(report) and overfit(report), before each receipt is stamped;
    - checkpoint(state, fold, seed), before a checkpoint is saved;
    - training(record, fold, seed), after its checkpoint is saved, so the
      checkpoint keeps the selection training made;
    - controls(results), the controls file's results by fold and seed;
    - evaluation(record, scene), before a parquet is written.

    base_set is the base TableSet to reuse. Without it the tables are built in
    full. draw draws real figures, and placeholders are drawn otherwise.
    """
    hooks = hooks or {}
    root.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["output_root"] = str(root / "outputs")
    raw["phase4_dir"] = str(phase4)
    config = root / "phase5.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_phase5_config(config)
    run_dir, evidence = Path(cfg.run_dir), Path(cfg.evidence_dir)
    start = datetime.now(timezone.utc) - timedelta(days=1)

    def at(minutes: int) -> str:
        return utc_timestamp(start + timedelta(minutes=minutes))

    identity = {"commit": commit, "config_digest": cfg.digest(), "fold_digest": fold_digest(FOLDS),
                "measurement_digest": FAST.measurement_digest()}
    gate_report = integration_report(identity, phase4)
    if "integration" in hooks:
        hooks["integration"](gate_report)
    gate = write_json(evidence / "integration_gate.json",
                      stamp(gate_report, "integration", identity, at(0)))
    overfit_payload = overfit_report()
    if "overfit" in hooks:
        hooks["overfit"](overfit_payload)
    overfit = write_json(evidence / "tiny_overfit.json", stamp(
        overfit_payload, "overfit", identity, at(10), gate_receipt_sha256=sha256_file(gate)))
    gates = {"integration_gate": sha256_file(gate), "tiny_overfit": sha256_file(overfit)}

    checkpoints: dict[str, Path] = {}
    records: dict[str, Path] = {}
    for fold in FOLDS:
        for seed in SEEDS:
            key = fold_seed_key(fold.index, seed)
            history = training_history(fold.index, seed)
            best_step, best = max(history, key=lambda entry: entry[1])
            state = {"model": compact_state(), "step": best_step,
                     "validation_centered_cosine": best, "fold": fold.index, "seed": seed,
                     "training_config_digest": training_config_from(cfg.training).digest()}
            if "checkpoint" in hooks:
                hooks["checkpoint"](state, fold, seed)
            ckpt = checkpoint_path(run_dir, LEVEL, fold.index, seed)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, ckpt)
            record = training_record(cfg, fold, seed, ckpt, commit, gates,
                                     at(20 + 3 * fold.index + seed))
            if "training" in hooks:
                hooks["training"](record, fold, seed)
            checkpoints[key] = ckpt
            records[key] = write_json(training_record_path(run_dir, LEVEL, fold.index, seed),
                                      record)

    results = {fold_seed_key(f.index, s): control_result(f) for f in FOLDS for s in SEEDS}
    if "controls" in hooks:
        hooks["controls"](results)
    controls = write_json(controls_path(evidence, LEVEL), controls_payload(
        LEVEL, results, [], identity, gates,
        {key: sha256_file(path) for key, path in checkpoints.items()}, written_utc=at(60)))

    def entry(path: Path) -> dict:
        return {"path": str(path), "sha256": sha256_file(path)}

    lock = write_json(checkpoint_lock_path(evidence, LEVEL), stamp(
        {"passed": True, "level": LEVEL, "folds": [f.index for f in FOLDS], "seeds": SEEDS,
         "checkpoints": {key: entry(path) for key, path in checkpoints.items()},
         "training_records": {key: entry(path) for key, path in records.items()},
         "controls": entry(controls)},
        "lock", identity, at(70), gate_receipt_sha256=gates["integration_gate"],
        overfit_receipt_sha256=gates["tiny_overfit"]))
    licence = {**gates, lock.stem: sha256_file(lock)}

    ledger = evidence / "evaluation_ledger"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(modes, "git_commit", lambda: commit)
        for fold_index, (scene, (rows, audit)) in enumerate(report_tests.all_scene_rows().items()):
            fold = fold_of_test_scene(scene, FOLDS)
            assert fold.index == fold_index
            meta = evaluation_metadata(
                cfg, FAST, scene, fold, LEVEL, CENTER, run_dir, SEEDS,
                {"metadata": {"git_commit": P4}}, phase4 / "eval" / f"{scene}.parquet", audit,
                licence=licence, aligned_depth_digest=ALIGNED[scene])
            if "evaluation" in hooks:
                hooks["evaluation"](meta, scene)
            out = run_dir / "eval" / LEVEL / f"{scene}.parquet"
            attempt = modes.new_attempt_id()
            modes.record_evaluation_attempt(ledger, scene, LEVEL, "started", licence, out,
                                            attempt=attempt)
            write_rows(out, rows, meta)
            modes.record_evaluation_attempt(ledger, scene, LEVEL, "written", licence, out,
                                            attempt=attempt)

    captured: dict = {}
    with pytest.MonkeyPatch.context() as patch:
        if base_set is not None:
            patch.setattr(report, "build_tables", reuse_tables(base_set))
        else:
            original = report.build_tables

            def spy(*args, **kwargs):
                captured["set"] = original(*args, **kwargs)
                return captured["set"]

            patch.setattr(report, "build_tables", spy)
        report.run_tables(cfg, FAST, config, LEVEL, expected_scenes=SCENES, folds=FOLDS,
                          repo_root=repo, check_code=True,
                          phase4_reference=report_tests._fake_phase4)
    with pytest.MonkeyPatch.context() as patch:
        if not draw:
            patch.setattr(figures, "draw_figures", placeholder_figures)
        figures.run_figures(cfg, FAST, config, LEVEL, expected_scenes=SCENES, folds=FOLDS,
                            repo_root=repo, check_code=True)
    return SimpleNamespace(
        root=root, repo=repo, E=commit, phase4=phase4, cfg=cfg, config=config, run_dir=run_dir,
        evidence=evidence, licence=licence, gates=gates, gate=gate, overfit=overfit, lock=lock,
        controls=controls, checkpoints=checkpoints, records=records,
        table_set=captured.get("set", base_set),
    )


def copy_world(world: SimpleNamespace, root: Path) -> SimpleNamespace:
    """The world's outputs, copied under root, with a config that points at them."""
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(world.root / "outputs", root / "outputs")
    raw = yaml.safe_load(world.config.read_text(encoding="utf-8"))
    raw["output_root"] = str(root / "outputs")
    config = root / "phase5.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_phase5_config(config)
    assert cfg.digest() == world.cfg.digest()
    return SimpleNamespace(**{**vars(world), "root": root, "cfg": cfg, "config": config,
                              "run_dir": Path(cfg.run_dir), "evidence": Path(cfg.evidence_dir)})


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """One repository, one Phase 4 run, and the base world, with real figures."""
    patcher = pytest.MonkeyPatch()
    # The integration receipt's binding to the cluster inputs is
    # tests/test_phase5_receipt.py's subject. Every other binding runs.
    patcher.setattr(receipt_module, "_artifact_problems", lambda *args: [])
    try:
        root = tmp_path_factory.mktemp("acceptance")
        repo, commit = make_repository(root)
        phase4 = make_phase4(root)
        world = build_world(root / "base", repo, commit, phase4, draw=True)
        yield SimpleNamespace(repo=repo, E=commit, phase4=phase4, world=world)
    finally:
        patcher.undo()


def rebuild(shared, root: Path, **hooks) -> SimpleNamespace:
    """A world built again from the changed artifacts, everything downstream consistent."""
    return build_world(root, shared.repo, shared.E, shared.phase4, hooks=hooks,
                       base_set=shared.world.table_set)


# ---------------------------------------------------------------------------
# Running acceptance with the suite and the Phase 4 check injected
# ---------------------------------------------------------------------------

def green_suite(repo_root) -> dict:
    files = sorted({name for names in acceptance.SUITE_FILES.values() for name in names}
                   | {"tests/test_render_replica.py"})
    return {
        "command": ["python", "-m", "pytest", "-q"], "returncode": 0,
        "summary": "1300 passed, 3 skipped in 600.00s",
        "counts": {"passed": 1300, "failed": 0, "errors": 0, "skipped": 3},
        "files": {name: {"passed": 10, "failed": 0, "errors": 0, "skipped": 0} for name in files},
        "output_tail": ["1300 passed, 3 skipped in 600.00s"],
    }


def accepted_phase4(cfg, repo_root) -> dict:
    return {"command": ["python", "scripts/phase4_acceptance_check.py"], "returncode": 0,
            "stdout": ["All 5 acceptance conditions satisfied."], "stderr": []}


def accept(world, **kwargs):
    arguments = {"expected_scenes": SCENES, "folds": FOLDS, "repo_root": world.repo,
                 "suite_runner": green_suite, "phase4_check": accepted_phase4, **kwargs}
    return acceptance.evaluate_acceptance(world.cfg, FAST, world.config, LEVEL, **arguments)


def failed(verdict) -> set[int]:
    return {condition.number for condition in verdict.conditions if not condition.ok}


def notes(verdict, number: int) -> str:
    return "\n".join(verdict.conditions[number - 1].notes)


def only(verdict, number: int, *words: str) -> None:
    """Exactly condition number failed, and its notes say what failed."""
    assert failed(verdict) == {number}, acceptance.format_report(verdict)
    text = notes(verdict, number)
    for word in words:
        assert word in text, text


def overriding(world, path: str, change):
    """A source reader that changes one file at commit E, and reads git otherwise."""
    def reader(commit: str, relative: str) -> bytes | None:
        data = acceptance.git_blob(world.repo, commit, relative)
        if commit == world.E and relative == path and data is not None:
            changed = change(data.decode("utf-8"))
            assert changed != data.decode("utf-8"), "the mutation changed nothing"
            return changed.encode("utf-8")
        return data
    return reader


def repo_copy(world, root: Path, files: dict[str, str], message: str) -> SimpleNamespace:
    """The world with a copy of its repository, one commit after E."""
    repo = root / "repo"
    shutil.copytree(world.repo, repo)
    head = commit_files(repo, files, message)
    assert head != world.E
    return SimpleNamespace(**{**vars(world), "repo": repo})


def table_copies(shared, root: Path, change, redraw: bool) -> dict:
    """The base tables, copied and changed, with figures redrawn from them or copied."""
    world = shared.world
    tables_root, figures_root = root / "tables", root / "figures"
    shutil.copytree(world.run_dir / "tables" / LEVEL, tables_root / LEVEL)
    change(tables_root / LEVEL)
    if redraw:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(figures, "draw_figures", placeholder_figures)
            figures.run_figures(world.cfg, FAST, world.config, LEVEL, expected_scenes=SCENES,
                                folds=FOLDS, repo_root=world.repo, check_code=True,
                                tables_root=tables_root, figures_root=figures_root)
    else:
        shutil.copytree(world.run_dir / "figures" / LEVEL, figures_root / LEVEL)
    return {"tables_root": tables_root, "figures_root": figures_root}


def rewrite_table(directory: Path, name: str, rows_change=None, record_change=None) -> None:
    """Rewrite one published table in place, and name its new sha256 in MANIFEST.json."""
    path = directory / name
    rows, record = read_rows(path), read_run_metadata(path)
    if record_change is not None:
        record_change(record)
    if rows_change is not None:
        rows = rows_change([dict(row) for row in rows])
    path.unlink()
    write_parquet_with_record(path, rows, record)
    manifest_path = directory / report.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][name] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")


def rewrite_document(directory: Path, name: str, change) -> None:
    """Rewrite one published JSON output, and name its new sha256 in MANIFEST.json."""
    path = directory / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    manifest_path = directory / report.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][name] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# A complete artifact set passes, and its verdict names what step 52 asks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def verdict(shared):
    return accept(shared.world)


def test_a_complete_artifact_set_passes_every_condition(verdict):
    assert [c.number for c in verdict.conditions] == list(range(1, 22))
    assert [c.title for c in verdict.conditions] == [
        acceptance.CONDITION_TITLES[number] for number in range(1, 22)]
    assert failed(verdict) == set(), acceptance.format_report(verdict)
    assert verdict.passed is True and verdict.record["passed"] is True
    for condition in verdict.conditions:
        assert condition.notes, condition.number
        assert condition.evidence, condition.number


def test_every_condition_reads_artifacts_rather_than_asserting(verdict):
    """Each condition names what it read, so a reader can audit the re-derivation."""
    evidence = {c.number: c.evidence for c in verdict.conditions}
    assert evidence[1]["phase4_parquets"][SCENES[0]]["live"] is not None
    assert set(evidence[5]["rotation_pairs"]) == set(SCENES)
    assert evidence[8]["ledger"]["writers"].keys() == set(SCENES)
    assert len(evidence[11]["selection"]) == len(FOLDS) * len(SEEDS)
    assert evidence[14]["recomputed"], evidence[14]
    assert evidence[16]["entries_checked"] > 0
    assert len(evidence[19]["runs"]) == len(FOLDS) * len(SEEDS)
    assert evidence[21]["summary"] == green_suite(None)["summary"]


def test_without_scene_inputs_the_recount_is_said_not_to_have_run(verdict):
    text = notes(verdict, 12)
    assert "scene inputs" in text and "not available" in text


def test_the_verdict_names_what_step_52_records(shared, verdict):
    world, record = shared.world, verdict.record
    assert record["kind"] == "phase5_acceptance" and record["level"] == LEVEL
    assert record["evaluation_commit"] == shared.E
    assert record["report_commit"] == git(world.repo, "rev-parse", "HEAD")
    assert record["scene_split_hash"] == FROZEN_FOLD_DIGEST
    assert record["folds"] == [{"index": f.index, "train": list(f.train), "val": list(f.val),
                                "test": list(f.test)} for f in FOLDS]
    assert record["seeds"] == SEEDS
    assert record["config_digest"] == world.cfg.digest()
    assert record["training_config_digest"] == training_config_from(world.cfg.training).digest()
    blob = subprocess.run(["git", "cat-file", "blob", f"{shared.E}:src/lot/context_lift.py"],
                          cwd=world.repo, capture_output=True).stdout
    assert record["context_lift"]["sha256_at_e"] == hashlib.sha256(blob).hexdigest()
    assert record["checkpoints"] == {key: sha256_file(path)
                                     for key, path in world.checkpoints.items()}
    assert record["evaluation_artifacts"] == {
        scene: sha256_file(world.run_dir / "eval" / LEVEL / f"{scene}.parquet")
        for scene in SCENES}
    tables_manifest = json.loads(
        (world.run_dir / "tables" / LEVEL / "MANIFEST.json").read_text(encoding="utf-8"))
    figures_manifest = json.loads(
        (world.run_dir / "figures" / LEVEL / "MANIFEST.json").read_text(encoding="utf-8"))
    assert record["tables"]["files"] == tables_manifest["files"]
    assert record["figures"]["files"] == figures_manifest["files"]
    assert {stem: entry["sha256"] for stem, entry in record["receipts"].items()} == world.licence
    assert record["design_correction"]["commits"] == [shared.E]
    assert [c["number"] for c in record["conditions"]] == list(range(1, 22))
    assert all(c["ok"] and c["notes"] for c in record["conditions"])


def test_the_verdict_carries_the_measured_outcome_per_regime_and_metric(shared, verdict):
    outcome = verdict.record["measured_outcome"]
    assert [(row["metric"], row["scope"]) for row in outcome] == [
        (metric, scope) for metric in ("centered", "raw")
        for scope in ("rotation", "translation", "orbit", "pooled")]
    by_key = {(row["metric"], row["scope"]): row for row in outcome}
    assert by_key[("centered", "pooled")]["summary"] is True
    assert by_key[("centered", "pooled")]["row_label"] == "pooled over regimes (summary)"
    assert by_key[("centered", "rotation")]["outcome"] == "46"
    assert by_key[("centered", "orbit")]["outcome"] == "48"
    assert by_key[("centered", "translation")]["outcome_49"] is True
    published = read_rows(shared.world.run_dir / "tables" / LEVEL / report.MEASURED_OUTCOME_TABLE)
    assert [row["outcome"] for row in outcome] == [row["outcome"] for row in published]
    # Decision 2 shows the cross-path terms beside outcome 49 and never collapses
    # its two statements. So every published column is copied, whole.
    for mine, theirs in zip(outcome, published):
        for column, value in theirs.items():
            assert column in mine, column
            assert mine[column] == value or (
                isinstance(value, float) and math.isnan(value) and math.isnan(mine[column])
            ), column
    for column in ("x_delta_learn_pp_ci_low", "x_delta_learn_sp_ci_high",
                   "path_difference_learn_ci_replicates", "delta_learn_sp_ci_low",
                   "delta_learn_sp_supported", "read_deficit_ci_high", "read_deficit_supported",
                   "n_feature_comparisons"):
        assert column in by_key[("centered", "translation")], column
    assert all(row["visibility_bucket"] == "co-visible" for row in outcome)


def test_a_verdict_at_another_level_is_not_the_rung2_verdict(verdict):
    """Decision 4: a sensitivity level is not an acceptance condition, and step 52
    marks Rung 2 accepted from the primary result alone. A verdict at another
    level that holds every condition says so, names its role and the primary
    level, and claims no Rung 2 acceptance."""
    assert verdict.record["level_role"] == "primary"
    assert verdict.record["rung2_verdict"] is True
    assert "Rung 2 is accepted" in acceptance.format_report(verdict)
    record = {**verdict.record, "level": "affine", "level_role": "sensitivity",
              "rung2_verdict": False}
    other = acceptance.AcceptanceVerdict("affine", verdict.conditions, record)
    assert other.passed
    text = acceptance.format_report(other)
    assert "Rung 2 is accepted" not in text and "ACCEPTED" not in text
    assert "not the Rung 2 verdict" in text
    assert "sensitivity level affine" in text and f"primary level {LEVEL}" in text
    assert "Measured outcome, decision 2" not in text
    assert "Measured outcome at the sensitivity level affine" in text


def test_run_acceptance_writes_the_verdict_once_and_renders_the_ledger(shared, tmp_path):
    world = copy_world(shared.world, tmp_path / "copy")
    stream = io.StringIO()
    arguments = {"expected_scenes": SCENES, "folds": FOLDS, "repo_root": world.repo,
                 "suite_runner": green_suite, "phase4_check": accepted_phase4}
    status = acceptance.run_acceptance(world.cfg, FAST, world.config, LEVEL, stream=stream,
                                       **arguments)
    assert status == 0, stream.getvalue()
    path = acceptance.verdict_path(world.evidence, LEVEL)
    assert path == world.evidence / "acceptance_image.json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["passed"] is True and len(written["conditions"]) == 21
    ledger = world.evidence / modes.LEDGER_FILE
    lines = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert sorted(line["scene"] for line in lines) == sorted(SCENES)
    assert written["evaluation_ledger"]["sha256"] == sha256_file(ledger)
    text = stream.getvalue()
    assert "PHASE 5 ACCEPTANCE" in text and "21 of 21" in text

    # A second run keeps the first verdict and the first rendering.
    assert acceptance.run_acceptance(world.cfg, FAST, world.config, LEVEL,
                                     stream=io.StringIO(), **arguments) == 0
    assert (world.evidence / "acceptance_image.superseded.1.json").is_file()
    assert (world.evidence / "evaluation_ledger.superseded.1.jsonl").is_file()


def test_a_failing_condition_exits_one_and_its_verdict_is_written(shared, tmp_path):
    world = copy_world(shared.world, tmp_path / "copy")
    stream = io.StringIO()
    status = acceptance.run_acceptance(
        world.cfg, FAST, world.config, LEVEL, stream=stream, expected_scenes=SCENES,
        folds=FOLDS, repo_root=world.repo, suite_runner=red_suite,
        phase4_check=accepted_phase4)
    assert status == 1
    written = json.loads(acceptance.verdict_path(world.evidence, LEVEL).read_text(
        encoding="utf-8"))
    assert written["passed"] is False and written["failed_conditions"] == [21]
    assert "NOT SATISFIED" in stream.getvalue()


def test_a_failed_verdict_withholds_the_measured_outcome(shared):
    """Specification step 45: any failure is a stop. Step 52 records the measured
    outcome only when every condition holds. So a failed verdict names the
    conditions that failed, and neither records nor prints the outcome."""
    verdict = accept(shared.world, suite_runner=red_suite)
    assert failed(verdict) == {21}
    for field in ("measured_outcome", "findings"):
        block = verdict.record[field]
        assert isinstance(block, dict) and "withheld" in block, block
        assert block["failed_conditions"] == [21]
    text = acceptance.format_report(verdict)
    assert "NOT SATISFIED" in text and "withheld" in text
    assert "Measured outcome" not in text and "outcome 46" not in text


def test_the_verdict_reads_the_measured_outcome_through_the_bound_tables(shared):
    """The tables are read for the outcome only once they are bound to the
    evaluated run, as the conditions read them. Tables the run cannot claim give
    no outcome, even beside conditions that all hold."""
    world = shared.world
    ctx = acceptance.AcceptanceContext(
        world.cfg, FAST, world.config, LEVEL, expected_scenes=SCENES, folds=FOLDS,
        repo_root=world.repo, suite_runner=green_suite, phase4_check=accepted_phase4)

    def unbound():
        raise acceptance.Unavailable("the published tables are not the evaluated run's")

    ctx.tables = unbound
    assert ctx.published().tables[report.MEASURED_OUTCOME_TABLE]
    conditions = [acceptance.ConditionResult(number, title, True, ("held",), {})
                  for number, title in sorted(acceptance.CONDITION_TITLES.items())]
    record = acceptance._verdict_record(ctx, conditions)
    assert record["passed"] is True
    for field in ("measured_outcome", "findings"):
        assert set(record[field]) == {"unavailable"}, record[field]
        assert "not the evaluated run's" in record[field]["unavailable"]


def test_the_level_and_the_configuration_are_checked_first(shared):
    world = shared.world
    with pytest.raises(ValueError, match="not declared"):
        acceptance.evaluate_acceptance(world.cfg, FAST, world.config, "scene")
    other = dataclasses.replace(world.cfg, seed=7)
    with pytest.raises(ValueError, match="configuration"):
        acceptance.evaluate_acceptance(other, FAST, world.config, LEVEL)


# ---------------------------------------------------------------------------
# One mutation per condition, each failing exactly that condition
# ---------------------------------------------------------------------------

def test_1_a_failed_phase4_acceptance_check(shared):
    def failing(cfg, repo_root):
        return {**accepted_phase4(cfg, repo_root), "returncode": 1,
                "stdout": ["NOT SATISFIED: 1 of 5"]}

    only(accept(shared.world, phase4_check=failing), 1, "phase4_acceptance_check")


def test_1_an_evaluation_naming_a_phase4_commit_the_phase4_run_does_not(shared, tmp_path):
    world = rebuild(shared, tmp_path / "world",
                    evaluation=lambda meta, scene: meta.update(phase4_commit="5" * 40))
    only(accept(world), 1, "5" * 40)


def test_2_a_frozen_class_change_after_e(shared, tmp_path):
    path = "src/lot/phase5_estimands.py"
    text = (shared.repo / path).read_text(encoding="utf-8")
    world = repo_copy(shared.world, tmp_path, {path: text + "\n# Edited after E.\n"}, "edit")
    only(accept(world), 2, "frozen_at_e", path)


def test_2_an_unlisted_reporting_change_after_e(shared, tmp_path):
    path = "src/lot/phase5_figures.py"
    text = (shared.repo / path).read_text(encoding="utf-8")
    world = repo_copy(shared.world, tmp_path, {path: text + "\n# Edited after E.\n"}, "edit")
    only(accept(world), 2, "reporting_paths", path)


def test_2_a_design_record_edited_after_e(shared, tmp_path):
    path = "validation/evidence/phase5/design_correction.md"
    text = (shared.repo / path).read_text(encoding="utf-8")
    world = repo_copy(shared.world, tmp_path, {path: text + "\nA note added later.\n"}, "edit")
    only(accept(world), 2, "design_correction.md")


def test_3_a_target_depth_parameter_in_context_lift(shared):
    reader = overriding(shared.world, "src/lot/context_lift.py", lambda text: text.replace(
        "    patch_size: int = PATCH_SIZE,\n) -> ContextLiftMap:",
        "    patch_size: int = PATCH_SIZE,\n    depth_target: Tensor | None = None,\n"
        ") -> ContextLiftMap:"))
    only(accept(shared.world, source_reader=reader), 3, "depth_target")


HOLLOW_TEST = "def test_trivial():\n    assert True\n"


def test_3_and_5_a_test_file_hollowed_after_e(shared, tmp_path):
    """Conditions 3 and 5 read the suite's results for tests/test_context_lift.py
    at R. Tests are neutral to code ancestry, so a file replaced after E by one
    trivial test would still pass there. Its results are evidence only for the
    tests registered at E, so a change must be named with its reason."""
    world = repo_copy(shared.world, tmp_path, {"tests/test_context_lift.py": HOLLOW_TEST},
                      "hollow")
    verdict = accept(world)
    assert failed(verdict) == {3, 5}, acceptance.format_report(verdict)
    for number in (3, 5):
        text = notes(verdict, number)
        assert "tests/test_context_lift.py" in text and "post_evaluation_changes.md" in text


def test_a_test_file_change_named_with_its_reason_is_disclosed(shared, tmp_path):
    reason = "a test fixed for the cluster's numpy, with no assertion weakened"
    world = repo_copy(shared.world, tmp_path, {
        "tests/test_context_lift.py": HOLLOW_TEST,
        "validation/evidence/phase5/post_evaluation_changes.md":
            f"- `tests/test_context_lift.py`: {reason}\n",
    }, "a listed test change")
    verdict = accept(world)
    assert failed(verdict) == set(), acceptance.format_report(verdict)
    entry = verdict.conditions[2].evidence["suite_file_sha256"]["tests/test_context_lift.py"]
    assert entry["reason"] == reason and entry["at_e"] != entry["at_r"]


def test_a_test_helper_changed_after_e_fails_every_condition_it_serves(shared, tmp_path):
    """tests/scenes.py builds the analytic scenes, and tests/conftest.py puts the
    code on the path. Every condition that reads the suite rests on them."""
    text = (shared.repo / "tests" / "scenes.py").read_text(encoding="utf-8")
    world = repo_copy(shared.world, tmp_path, {"tests/scenes.py": text + "\n# changed after E\n"},
                      "scenes")
    verdict = accept(world)
    assert failed(verdict) == set(acceptance.SUITE_FILES), acceptance.format_report(verdict)
    assert "tests/scenes.py" in notes(verdict, 12)


def test_a_new_test_file_after_e_fails_nothing(shared, tmp_path):
    world = repo_copy(shared.world, tmp_path,
                      {"tests/test_a_later_note.py": "def test_note():\n    pass\n"}, "a note")
    verdict = accept(world)
    assert failed(verdict) == set(), acceptance.format_report(verdict)


def test_the_suite_files_read_at_r_are_es_own(verdict):
    for number, names in acceptance.SUITE_FILES.items():
        hashes = verdict.conditions[number - 1].evidence["suite_file_sha256"]
        assert set(hashes) == set(names) | set(acceptance.SUITE_SUPPORT), number
        for path, entry in hashes.items():
            assert entry["at_e"] is not None and entry["at_e"] == entry["at_r"], (number, path)
            assert entry["reason"] is None


def test_4_a_target_input_to_the_predictor(shared):
    reader = overriding(shared.world, "src/lot/predictors.py", lambda text: text.replace(
        "        context_valid: Tensor | None = None,\n    ) -> Tensor:",
        "        context_valid: Tensor | None = None,\n"
        "        features_target: Tensor | None = None,\n    ) -> Tensor:"))
    only(accept(shared.world, source_reader=reader), 4, "features_target")


def test_5_a_missing_rotation_evidence_item(shared, tmp_path):
    def drop(report_payload):
        for step in report_payload["steps"]:
            if step["step"] == "17":
                del step["evidence"]["scenes"][SCENES[1]]

    only(accept(rebuild(shared, tmp_path / "world", integration=drop)), 5, SCENES[1])


def test_6_model_inputs_fed_another_depth_than_the_lift(shared):
    other_depth = 'aligned_context_depth(inputs, ctx, "none")'
    reader = overriding(shared.world, "src/lot/phase5_modes.py", lambda text: text.replace(
        "visible = model_inputs(cfg, inputs, pair, cams, context_depth)",
        f"visible = model_inputs(cfg, inputs, pair, cams, {other_depth})"))
    only(accept(shared.world, source_reader=reader), 6, "model_inputs")


def test_6_an_aligned_depth_the_gate_did_not_record(shared, tmp_path):
    def other(meta, scene):
        if scene == SCENES[2]:
            meta["aligned_depth_digest"] = "0" * 64

    only(accept(rebuild(shared, tmp_path / "world", evaluation=other)), 6, SCENES[2])


def test_7_a_training_scene_evaluated_as_a_test_scene(shared, tmp_path):
    world = copy_world(shared.world, tmp_path / "copy")
    eval_dir = world.run_dir / "eval" / LEVEL
    leaked = next(scene for scene in FOLDS[0].train if scene not in SCENES)
    shutil.copyfile(eval_dir / f"{SCENES[0]}.parquet", eval_dir / f"{leaked}.parquet")
    only(accept(world), 7, leaked)


def test_7_controls_that_planned_from_a_test_scene(shared, tmp_path):
    """The planned-scene lists, not the fold constants, show what a run read."""
    leaked = FOLDS[1].test[0]

    def leak(results):
        results[fold_seed_key(1, 0)]["val_scenes_planned"].append(leaked)

    only(accept(rebuild(shared, tmp_path / "world", controls=leak)), 7, leaked, "fold1_seed0")


def test_8_a_checkpoint_changed_after_the_lock(shared, tmp_path):
    world = copy_world(shared.world, tmp_path / "copy")
    path = checkpoint_path(world.run_dir, LEVEL, 1, 2)
    state = torch.load(path, map_location="cpu", weights_only=False)
    name = next(iter(state["model"]))
    tensor = state["model"][name]
    state["model"][name] = torch.ones((1,) * tensor.dim()).expand(tensor.shape)
    torch.save(state, path)
    only(accept(world), 8, "fold1_seed2")


def test_8_a_second_evaluation_attempt(shared, tmp_path):
    world = copy_world(shared.world, tmp_path / "copy")
    scene = SCENES[1]
    out = world.run_dir / "eval" / LEVEL / f"{scene}.parquet"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(modes, "git_commit", lambda: world.E)
        attempt = modes.new_attempt_id()
        ledger = world.evidence / "evaluation_ledger"
        modes.record_evaluation_attempt(ledger, scene, LEVEL, "started", world.licence, out,
                                        attempt=attempt)
        modes.record_evaluation_attempt(ledger, scene, LEVEL, "written", world.licence, out,
                                        attempt=attempt)
    only(accept(world), 8, scene, "written")


def test_8_an_evaluation_moved_aside(shared, tmp_path):
    """A second evaluation hidden outside the evaluation directory is found."""
    world = copy_world(shared.world, tmp_path / "copy")
    hidden = world.run_dir / "eval" / f"{LEVEL}.old"
    hidden.mkdir()
    shutil.copyfile(world.run_dir / "eval" / LEVEL / f"{SCENES[0]}.parquet",
                    hidden / f"{SCENES[0]}.parquet")
    only(accept(world), 8, f"{LEVEL}.old")


def test_8_an_unreadable_parquet_under_the_run_directory(shared, tmp_path):
    """A file that cannot be read cannot be shown not to be another evaluation."""
    world = copy_world(shared.world, tmp_path / "copy")
    hidden = world.run_dir / "eval" / f"{LEVEL}.old"
    hidden.mkdir()
    (hidden / f"{SCENES[0]}.parquet").write_bytes(b"not a parquet")
    # A staging directory of an unfinished tables build holds reporting outputs only.
    staging = world.run_dir / "tables" / f"{LEVEL}.partial.{'0' * 32}"
    staging.mkdir()
    (staging / "phase5_primary.parquet").write_bytes(b"half a table")
    verdict = accept(world)
    only(verdict, 8, "cannot be read", f"{LEVEL}.old")
    assert ".partial." not in notes(verdict, 8)


@pytest.mark.parametrize("name", [f"{LEVEL}.partial", f"{LEVEL}.partial.old"])
def test_8_an_evaluation_moved_aside_under_a_partial_name(shared, tmp_path, name):
    """Only a reporting build's own staging directory, {level}.partial.<32 hex>
    under tables/ or figures/, is skipped. An evaluation moved aside under any
    other name holding .partial is found."""
    world = copy_world(shared.world, tmp_path / "copy")
    hidden = world.run_dir / "eval" / name
    hidden.mkdir()
    shutil.copyfile(world.run_dir / "eval" / LEVEL / f"{SCENES[0]}.parquet",
                    hidden / f"{SCENES[0]}.parquet")
    only(accept(world), 8, name, "another Phase 5 evaluation")


def test_9_an_overfit_subset_outside_the_training_scenes(shared, tmp_path):
    def leak(payload):
        payload["subset"][0]["scene"] = FOLDS[0].val[0]

    only(accept(rebuild(shared, tmp_path / "world", overfit=leak)), 9, FOLDS[0].val[0])


def test_10_a_training_config_digest_that_drifted(shared, tmp_path):
    drifted = "0" * 64

    def on_state(state, fold, seed):
        state["training_config_digest"] = drifted

    def on_record(record, fold, seed):
        record["training_config_digest"] = drifted

    world = rebuild(shared, tmp_path / "world", checkpoint=on_state, training=on_record)
    only(accept(world), 10, "training_config_digest")


def test_11_a_best_step_that_is_not_the_selected_checkpoint(shared, tmp_path):
    def move(record, fold, seed):
        if (fold.index, seed) == (0, 1):
            record["best_step"] -= 500

    only(accept(rebuild(shared, tmp_path / "world", training=move)), 11, "fold0_seed1")


def recount_from_rows(world, shift: int):
    """A recount that reads the stored count, moved by shift for one pair."""
    def recount(cfg, analysis, level, scene, pairs):
        rows = read_rows(world.run_dir / "eval" / LEVEL / f"{scene}.parquet")
        stored = {(r["context_frame_id"], r["target_frame_id"]): r["n_primary"]
                  for r in rows if r["region"] == "all" and r["seed"] == SEEDS[0]}
        return {pair: stored[pair] + (shift if index == 0 and scene == SCENES[0] else 0)
                for index, pair in enumerate(pairs)}
    return recount


def test_12_a_recount_that_agrees_passes(shared):
    verdict = accept(shared.world, recount=recount_from_rows(shared.world, 0))
    assert failed(verdict) == set(), acceptance.format_report(verdict)
    assert "recounted" in notes(verdict, 12)


def test_12_a_stored_support_the_recount_does_not_reproduce(shared):
    only(accept(shared.world, recount=recount_from_rows(shared.world, 1)), 12, "recount")


def test_13_a_support_reassigned_after_the_model_runs(shared):
    reader = overriding(shared.world, "src/lot/phase5_modes.py", lambda text: text.replace(
        '                )[0].to("cpu", torch.float32)\n            for region in REGIONS:',
        '                )[0].to("cpu", torch.float32)\n'
        '            support = support & torch.isfinite(predicted).all(dim=-1).any()\n'
        '            for region in REGIONS:'))
    only(accept(shared.world, source_reader=reader), 13, "support")


def test_14_an_interval_the_bootstrap_does_not_reproduce(shared, tmp_path):
    def widen(directory):
        def change(rows):
            for row in rows:
                if (row["metric"], row["analysis"], row["axis"]) == ("centered", "rotation", "all"):
                    row["delta_learn_pp_ci_low"] -= 0.002
            return rows
        rewrite_table(directory, report.PRIMARY_TABLE, rows_change=change)

    only(accept(shared.world, **table_copies(shared, tmp_path, widen, redraw=True)), 14,
         "delta_learn_pp")


def test_15_a_protocol_that_is_not_the_frozen_one(shared):
    reader = overriding(shared.world, "PROTOCOL.md", lambda text: text + "\nAn edit.\n")
    only(accept(shared.world, source_reader=reader), 15, "PROTOCOL.md")


def test_16_a_dropped_disclosure_entry(shared, tmp_path):
    def drop(directory):
        rewrite_document(directory, report.NEAR_ZERO_FILE,
                         lambda payload: payload["entries"].pop(3))

    only(accept(shared.world, **table_copies(shared, tmp_path, drop, redraw=False)), 16,
         "no disclosure entry")


def test_16_the_word_equivalent_in_a_table(shared, tmp_path):
    def claim(directory):
        def change(rows):
            rows[0]["table_label"] = "the two methods are equivalent"
            return rows
        rewrite_table(directory, report.L2_TABLE, rows_change=change)

    only(accept(shared.world, **table_copies(shared, tmp_path, claim, redraw=False)), 16,
         "equivalen")


def test_16_a_cell_below_support_worded_as_a_claim(shared, tmp_path):
    """PROTOCOL 3.4 keeps a cell below support out of every claim. One such cell's
    disclosure entry and its headline row are rewritten, consistently, to print
    the small, sign-consistent wording. Its terms are untouched."""
    from lot.phase5_estimands import WORDING_SMALL_SIGN_CONSISTENT

    def engage(directory):
        target: dict = {}

        def change_entries(payload):
            entry = next(e for e in payload["entries"] if e["table"] == report.PRIMARY_TABLE
                         and e["quantity"] == "delta_learn_pp" and e["supported"] is False)
            entry.update(near_zero=True, wording=WORDING_SMALL_SIGN_CONSISTENT)
            target.update({key: entry[key] for key in ("metric", "analysis", "axis", "bin")})

        def change_rows(rows):
            for row in rows:
                if all(row[key] == value for key, value in target.items()):
                    row.update(delta_learn_pp_near_zero=True,
                               delta_learn_pp_near_zero_wording=WORDING_SMALL_SIGN_CONSISTENT)
            return rows

        rewrite_document(directory, report.NEAR_ZERO_FILE, change_entries)
        rewrite_table(directory, report.PRIMARY_TABLE, rows_change=change_rows)

    only(accept(shared.world, **table_copies(shared, tmp_path, engage, redraw=True)), 16,
         "below support")


def test_16_prints_the_decomposition_by_regime_quantity_and_metric(verdict):
    """Design section 8: the supported cells' wordings are printed by regime,
    quantity, and metric, never as a bare total, as Phase 4's check printed them."""
    condition = verdict.conditions[15]
    decomposition = condition.evidence["decomposition"]
    assert decomposition
    for case, count in decomposition.items():
        wording, regime, quantity, metric = case.split(" | ")
        assert any(f"{regime} | {quantity} | {metric}:" in line and f"{wording!r} {count}" in line
                   for line in condition.notes), case


def test_17_a_splat_transport_fed_target_depth(shared):
    reader = overriding(shared.world, "src/lot/phase5_reference.py", lambda text: text.replace(
        "torch.from_numpy(context_aligned).to(torch_dtype),\n"
        "        K_context, K_target, T_target_from_context, target_hw,",
        "torch.from_numpy(target_aligned).to(torch_dtype),\n"
        "        K_context, K_target, T_target_from_context, target_hw,"))
    only(accept(shared.world, source_reader=reader), 17, "target_aligned")


def test_18_target_lift_in_the_headline_table(shared, tmp_path):
    def inject(directory):
        def change(rows):
            for row in rows:
                row["tl_reference"] = 0.7
            return rows
        rewrite_table(directory, report.PRIMARY_TABLE, rows_change=change)

    only(accept(shared.world, **table_copies(shared, tmp_path, inject, redraw=True)), 18,
         "tl_reference")


def test_19_an_unstable_validation_curve(shared, tmp_path):
    def unstable(record, fold, seed):
        if (fold.index, seed) == (2, 2):
            history = training_history(fold.index, seed)
            best = max(score for _, score in history)
            settings = training_config_from(shared.world.cfg.training)
            steps = range(history[-1][0] + 500, settings.max_steps + 1, 500)
            history += [[step, best - 0.05] for step in steps]
            record.update(history=history, stopped_early=False, steps_run=settings.max_steps)

    only(accept(rebuild(shared, tmp_path / "world", training=unstable)), 19, "fold2_seed2")


def test_20_headline_figures_drawn_from_an_earlier_headline_table(shared, tmp_path):
    def restamp(directory):
        rewrite_table(directory, report.PRIMARY_TABLE,
                      record_change=lambda record: record.update(
                          created_utc=utc_timestamp()))

    only(accept(shared.world, **table_copies(shared, tmp_path, restamp, redraw=False)), 20,
         report.PRIMARY_TABLE)


def red_suite(repo_root) -> dict:
    result = green_suite(repo_root)
    result["files"]["tests/test_render_replica.py"] = {"passed": 9, "failed": 1, "errors": 0,
                                                      "skipped": 0}
    return {**result, "returncode": 1, "summary": "1 failed, 1299 passed, 3 skipped",
            "counts": {"passed": 1299, "failed": 1, "errors": 0, "skipped": 3}}


def test_21_a_suite_that_is_not_green(shared):
    only(accept(shared.world, suite_runner=red_suite), 21, "1 failed")


def test_21_head_moves_away_from_r_while_acceptance_runs(shared, tmp_path):
    """R is HEAD when acceptance starts. A commit made before the suite runs
    leaves the suite and the code ancestry at another commit than the verdict
    names."""
    world = repo_copy(shared.world, tmp_path, {"tests/earlier_note.txt": "a note\n"}, "a note")

    def moving(cfg, repo_root):
        commit_files(world.repo, {"tests/later_note.txt": "another note\n"}, "another note")
        return accepted_phase4(cfg, repo_root)

    verdict = accept(world, phase4_check=moving)
    only(verdict, 21, "the verdict names R", "while acceptance ran", "not at R")
    assert verdict.record["report_commit"] != git(world.repo, "rev-parse", "HEAD")


# ---------------------------------------------------------------------------
# The pieces, on their own
# ---------------------------------------------------------------------------

def attempt(scene: str, status: str, *, sha: str | None = "s" * 64, at_start: str | None = None,
            start: str = "2026-10-10T10:00:00.000000+00:00", message: str | None = None,
            name: str = "1", commit: str = "e" * 40) -> dict:
    return {"attempt": name, "event": "evaluate", "scene": scene, "level": LEVEL,
            "licence": {"a": "b"}, "path": f"eval/{LEVEL}/{scene}.parquet", "commit": commit,
            "status": status, "message": message, "started_utc": start,
            "finished_utc": None if status == "unfinished" else start,
            "parquet_sha256_at_start": at_start,
            "parquet_sha256": None if status == "unfinished" else sha,
            "start_file": f"{name}.json", "close_file": None}


def ledger_problems(attempts: list[dict]) -> list[str]:
    problems, _ = acceptance.ledger_problems(
        attempts, level=LEVEL, scenes=["x"], live={"x": "s" * 64},
        written_utc={"x": "2026-10-10T10:30:00.000000+00:00"}, commit="e" * 40,
        licence={"a": "b"})
    return problems


def test_the_ledger_accepts_one_written_evaluation_per_scene():
    assert ledger_problems([attempt("x", "written")]) == []


def test_the_ledger_explains_an_error_that_wrote_nothing_and_a_resume():
    attempts = [attempt("x", "error", sha=None, message="OSError: lost the node", name="1"),
                attempt("x", "written", name="2", start="2026-10-10T10:10:00.000000+00:00"),
                attempt("x", "exists", at_start="s" * 64, name="3",
                        start="2026-10-10T11:00:00.000000+00:00"),
                # Killed outright after it found the live parquet: it could only resume.
                attempt("x", "unfinished", at_start="s" * 64, name="4",
                        start="2026-10-10T12:00:00.000000+00:00")]
    assert ledger_problems(attempts) == []


def test_the_ledger_takes_an_attempt_killed_after_its_write_as_the_writer():
    assert ledger_problems([attempt("x", "unfinished")]) == []
    killed = attempt("x", "error", message="SystemExit: evaluation stopped by SIGTERM")
    assert ledger_problems([killed]) == []


def test_the_ledger_explains_a_kill_that_a_resubmission_recovered():
    """An attempt killed outright, by the out-of-memory killer or a lost node,
    and then the resubmitted attempt the runbook prescribes. The resubmission
    found no parquet at its start and wrote the live one, so the attempt killed
    before it left nothing at the path. reporting_rules.md section 9 reports it
    as unfinished, and it fails nothing."""
    killed = attempt("x", "unfinished", start="2026-10-10T09:00:00.000000+00:00", name="1")
    problems, evidence = acceptance.ledger_problems(
        [killed, attempt("x", "written", name="2")], level=LEVEL, scenes=["x"],
        live={"x": "s" * 64}, written_utc={"x": "2026-10-10T10:30:00.000000+00:00"},
        commit="e" * 40, licence={"a": "b"})
    assert problems == []
    assert evidence["writers"] == {"x": {"attempt": "2", "status": "written"}}
    assert [entry["attempt"] for entry in evidence["explained_errors"]] == ["1"]


@pytest.mark.parametrize("attempts", [
    # Some attempt found another file at the path, so a file was moved aside,
    # and the attempt killed before the writer may have written it.
    [attempt("x", "unfinished", start="2026-10-10T09:00:00.000000+00:00", name="1"),
     attempt("x", "error", at_start="t" * 64, sha="t" * 64, message="ValueError: not this run's",
             start="2026-10-10T09:30:00.000000+00:00", name="2"),
     attempt("x", "written", name="3")],
    # The writer is an error close, which found no parquet only because the
    # signal landed after its write: it says nothing about what came before it.
    [attempt("x", "unfinished", start="2026-10-10T09:00:00.000000+00:00", name="1"),
     attempt("x", "error", message="SystemExit: evaluation stopped by SIGTERM", name="2")],
    # The writer itself found a parquet at its start, so the path was not clean.
    [attempt("x", "unfinished", start="2026-10-10T09:00:00.000000+00:00", name="1"),
     attempt("x", "written", at_start="t" * 64, name="2")],
], ids=["a-file-moved-aside", "writer-is-an-error-close", "writer-found-a-file"])
def test_the_ledger_still_refuses_a_kill_nothing_explains(attempts):
    problems = ledger_problems(attempts)
    assert any("unfinished" in problem for problem in problems), problems


@pytest.mark.parametrize("attempts, words", [
    ([attempt("x", "written", name="1"), attempt("x", "written", name="2")], "2 written"),
    ([attempt("x", "written", sha="t" * 64)], "not the live parquet"),
    ([attempt("x", "unfinished", start="2026-10-10T11:00:00.000000+00:00", name="1"),
      attempt("x", "written", name="2")], "unfinished"),
    ([attempt("x", "error", sha="t" * 64, message="boom", name="1"),
      attempt("x", "written", name="2")], "wrote a parquet"),
    ([attempt("x", "written", name="1"),
      attempt("x", "error", message="SystemExit: stopped", name="2",
              start="2026-10-10T10:20:00.000000+00:00")], "evaluated more than once"),
    ([attempt("x", "error", sha=None, message=None, name="1"),
      attempt("x", "written", name="2")], "no message"),
    ([], "no attempt"),
    ([attempt("x", "written", commit="f" * 40)], "commit"),
    ([attempt("x", "written"), attempt("y", "written", name="2")], "not an evaluation scene"),
], ids=["two-writes", "stale-write", "unexplained-kill", "error-wrote", "error-wrote-live",
        "silent-error", "nothing", "other-commit", "other-scene"])
def test_the_ledger_refuses_what_is_not_one_explained_evaluation(attempts, words):
    problems = ledger_problems(attempts)
    assert problems and any(words in problem for problem in problems), problems


def test_the_default_suite_runner_reads_a_real_pytest_run(tmp_path):
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_one.py").write_text(
        "def test_passes():\n    assert True\n\ndef test_fails():\n    assert False\n",
        encoding="utf-8")
    (project / "tests" / "test_two.py").write_text(
        "import pytest\n\n@pytest.mark.skip(reason='not here')\ndef test_skipped():\n    pass\n",
        encoding="utf-8")
    result = acceptance.run_suite(project)
    assert result["returncode"] == 1
    assert result["counts"] == {"passed": 1, "failed": 1, "errors": 0, "skipped": 1}
    assert result["files"]["tests/test_one.py"] == {"passed": 1, "failed": 1, "errors": 0,
                                                    "skipped": 0}
    assert result["files"]["tests/test_two.py"]["skipped"] == 1
    assert "1 failed" in result["summary"]
    assert result["command"][1:4] == ["-m", "pytest", "-q"]
    assert not list(project.glob(".pytest_cache"))


def test_the_default_phase4_check_runs_the_phase4_script_on_the_phase4_run(shared,
                                                                         monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"], seen["cwd"] = command, kwargs.get("cwd")
        return subprocess.CompletedProcess(command, 0,
                                           "All 5 acceptance conditions satisfied.\n", "")

    monkeypatch.setattr(acceptance.subprocess, "run", fake_run)
    result = acceptance.run_phase4_acceptance_check(shared.world.cfg, shared.repo)
    command = seen["command"]
    assert command[0] == sys.executable
    assert Path(command[1]) == shared.repo / "scripts" / "phase4_acceptance_check.py"
    assert command[command.index("--eval-dir") + 1] == str(shared.phase4 / "eval")
    assert command[command.index("--tables") + 1] == str(shared.phase4 / "tables")
    assert Path(command[command.index("--validator") + 1]) == (
        shared.repo / "validation" / "evidence" / "reaudit" / "borah_check_2_3.json")
    assert result["returncode"] == 0
    assert result["stdout"] == ["All 5 acceptance conditions satisfied."]


def test_the_default_recount_says_when_scene_inputs_are_absent(shared):
    with pytest.raises(acceptance.SceneInputsUnavailable, match="not available"):
        acceptance.recount_primary_support(shared.world.cfg, FAST, LEVEL, SCENES[0],
                                           [("rotation_00_c", "rotation_00_t")])


def test_batched_git_reads_equal_single_reads(shared, tmp_path):
    """One git process per commit reads exactly what one process per file reads."""
    world = repo_copy(shared.world, tmp_path, {
        "validation/evidence/phase5/design_correction.md": "edited after E\n",
        "src/lot/train.py": (shared.repo / "src/lot/train.py").read_text(encoding="utf-8")
        + "\n# edited after E\n"}, "edit")
    head = git(world.repo, "rev-parse", "HEAD")
    paths = list(acceptance.SOURCES_AT_E) + ["no/such/file.md", "src/lot"]
    batched = acceptance._Repository(world.repo, None)
    batched.prefetch(shared.E, paths)
    batched.prefetch(head, paths)
    batched.prefetch_touching(acceptance.PREREGISTRATION_DOCUMENTS + acceptance.TRAINING_SOURCES)
    for commit in (shared.E, head):
        for path in paths:
            assert batched.blob(commit, path) == acceptance.git_blob(world.repo, commit, path), path
    assert batched.blob(shared.E, "no/such/file.md") is None
    for path in acceptance.PREREGISTRATION_DOCUMENTS + acceptance.TRAINING_SOURCES:
        single = git(world.repo, "rev-list", "HEAD", "--", path).split()
        assert batched.commits_touching(path) == single, path
    assert batched.commits_touching("src/lot/train.py") == [head, shared.E]


def test_every_condition_has_a_title_and_the_suite_files_exist():
    assert sorted(acceptance.CONDITION_TITLES) == list(range(1, 22))
    for number, names in acceptance.SUITE_FILES.items():
        assert 1 <= number <= 21
        for name in names:
            assert (REPO / name).is_file(), name
    for name in acceptance.SUITE_SUPPORT:
        assert (REPO / name).is_file(), name


def test_the_module_text_uses_no_em_dash_and_no_letter_method_labels():
    source = Path(acceptance.__file__).read_text(encoding="utf-8")
    assert "\u2014" not in source
    assert not LETTER_LABEL.search(source)
    assert source.isascii()
    for title in acceptance.CONDITION_TITLES.values():
        assert "\u2014" not in title and not LETTER_LABEL.search(title)


def test_a_verdict_never_claims_equivalence(verdict):
    """The word appears only as the name of the depth-equivalence test file it cites."""
    text = json.dumps(verdict.record, default=str)
    assert "test_phase5_depth_equivalence.py" in text
    text = text.replace("test_phase5_depth_equivalence.py", "")
    assert not re.search("equivalen", text, re.IGNORECASE)
    measured = verdict.record["measured_outcome"]
    assert all(not isinstance(row["delta_learn_pp"], float) or math.isfinite(row["delta_learn_pp"])
               for row in measured)
