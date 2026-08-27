"""The path-agreement ledger, exercised end to end on a synthetic run.

The scene comes from test_one_path_pipeline: a flat viewpoint whose pairs score
on both paths and a striped viewpoint whose pairs are splat-only. The ledger
runs over the both-path pairs, reconstructs the recorded scores, closes the
identity, and writes its evidence; the stop conditions are then provoked one at
a time by tampering with what they guard.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lot.analysis_config import load_analysis_config
from lot.evaluate import EVAL_VERSION, dataset_mean_vector, evaluate_scene, write_rows
from lot.path_ledger import (
    CUTPOINTS_NAME,
    ledger_scene,
    per_cell_cosine,
    report,
    spearman,
)
from test_one_path_pipeline import ANALYSIS, SCENE, build_scene

import torch


def run_and_write(tmp_path):
    """Evaluate the synthetic scene and lay its outputs out as a real run."""
    from lot.evaluate import EvalConfig

    build_scene(tmp_path)
    cfg = EvalConfig(
        experiment_name="one_path", renders_root=tmp_path, cache_root=tmp_path / "cache",
        output_root=tmp_path / "out", scenes=[SCENE], encoders=["dinov2_vitb14"],
        seed=0, mean_vector_scenes=[SCENE],
    )
    mean = dataset_mean_vector(cfg.cache_root, "dinov2_vitb14", [SCENE])
    rows, metadata = evaluate_scene(cfg, SCENE, {"dinov2_vitb14": mean}, ANALYSIS)
    eval_dir = tmp_path / "out" / "one_path" / "eval"
    write_rows(eval_dir / f"{SCENE}.parquet", rows, {**metadata, "run_scenes": [SCENE]})

    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "experiment_name: one_path\n"
        f"renders_root: {tmp_path.as_posix()}\n"
        f"cache_root: {(tmp_path / 'cache').as_posix()}\n"
        f"output_root: {(tmp_path / 'out').as_posix()}\n"
        f"scenes: [{SCENE}]\n"
        "encoders: [dinov2_vitb14]\n",
        encoding="utf-8",
    )
    return config_path, eval_dir


def test_the_ledger_reconstructs_and_closes_on_a_real_run(tmp_path):
    config_path, _ = run_and_write(tmp_path)
    out = tmp_path / "ledger"
    facts = ledger_scene(config_path, SCENE, out, ANALYSIS)

    # Preflight: the both-path pairs are ledgered, the splat-only pairs are
    # skipped as having no common set, and every lookup agrees bit for bit.
    assert facts["pairs"] >= 1
    assert facts["pairs_skipped_empty_intersection"] >= 1
    for field in ("target_bit_mismatches", "no_warp_bit_mismatches",
                  "random_bit_mismatches", "n_intersect_mismatches",
                  "cell_order_mismatches"):
        assert facts[field] == 0, field

    summary = report(out, ANALYSIS)
    assert summary["verdict"] == "PASS"
    for metric in ("raw", "centered"):
        entry = summary[metric]
        # Reconstruction sits at dtype noise, far under the frozen tolerance.
        assert entry["max_abs_T1"] < ANALYSIS.ledger_recon_tol
        assert entry["max_abs_T4"] < ANALYSIS.ledger_recon_tol
        # T2 is then the whole recorded difference: the signed aggregate of
        # the recorded scores equals the signed aggregate of T2 to closure.
        assert entry["pairs"] == facts["pairs"]
    # The mechanism artifacts exist and the cut points were stored.
    assert (out / CUTPOINTS_NAME).exists()
    assert "dinov2_vitb14/raw" in summary["mechanism"]


def test_a_tampered_recorded_score_is_a_reconstruction_stop(tmp_path):
    config_path, eval_dir = run_and_write(tmp_path)
    import pyarrow.parquet as pq
    import pyarrow as pa

    table = pq.read_table(eval_dir / f"{SCENE}.parquet")
    rows = table.to_pylist()
    for row in rows:
        if row["variant"] == "Oracle-Transport" and row["path"] == "per_point":
            row["cosine_intersect_mean"] = row["cosine_intersect_mean"] + 0.01
            break
    tampered = pa.Table.from_pylist(rows).replace_schema_metadata(table.schema.metadata)
    (eval_dir / f"{SCENE}.parquet").unlink()
    pq.write_table(tampered, eval_dir / f"{SCENE}.parquet")

    out = tmp_path / "ledger"
    ledger_scene(config_path, SCENE, out, ANALYSIS)
    summary = report(out, ANALYSIS)
    assert summary["verdict"] == "STOP"
    assert any("T1 reconstruction" in item for item in summary["stop"])


def test_cut_points_are_preregistered_not_recomputed(tmp_path):
    """An existing cut-point file is reused, so reruns cannot move the split."""
    config_path, _ = run_and_write(tmp_path)
    out = tmp_path / "ledger"
    ledger_scene(config_path, SCENE, out, ANALYSIS)
    (out / "report.json").unlink(missing_ok=True)
    first = report(out, ANALYSIS)["cutpoints"]
    forged = {encoder: [0.0, 0.0, 0.0] for encoder in first}
    (out / CUTPOINTS_NAME).write_text(json.dumps(forged), encoding="utf-8")
    (out / "report.json").unlink()
    second = report(out, ANALYSIS)["cutpoints"]
    assert second == forged, "the stored file must win over recomputation"


def test_per_cell_cosine_matches_the_recording_arithmetic():
    generator = torch.Generator().manual_seed(3)
    a = torch.rand((40, 16), generator=generator)
    b = torch.rand((40, 16), generator=generator)
    center = torch.rand((16,), generator=generator)
    from lot.evaluate import value_agreement

    for c in (None, center):
        recorded, _ = value_agreement(a, b, center=c)
        assert float(per_cell_cosine(a, b, c).mean()) == pytest.approx(recorded, abs=1e-6)


def test_spearman_on_known_orderings():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert abs(spearman(x, np.array([2.0, 1.0, 4.0, 3.0]))) < 1.0
