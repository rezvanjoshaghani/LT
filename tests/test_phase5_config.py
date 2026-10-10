"""The Phase 5 config is frozen, loadable, and identifies what it must."""

from __future__ import annotations

from pathlib import Path

import pytest

from lot.analysis_config import load_analysis_config
from lot.phase5 import (
    Phase5Config,
    fold_and_seed_for_task,
    load_phase5_config,
    predictor_config_from,
)
from lot.phase5_folds import frozen_folds
from lot.predictors import build_predictor
from lot.train import TrainingConfig

CONFIG_PATH = Path("configs/phase5.yaml")

# The digest of the shipped architecture and training configuration. Frozen
# before any Phase 5 model was trained, and recorded in the Phase 5 pin. A
# change to any value it covers fails here, which is the point: the frozen
# configuration is what makes "not tuned on the test set" checkable rather than
# merely asserted.
#
# Moved once, on 2026-10-09, from 9e3508606bcaedb6068e95807aa27ea706d2d38b77f8c
# 2df763d8f28968a0eca, by the pre-registered landing-offset diagnostic, the only
# change. No Phase 5 gate, model, or result existed. The record is
# validation/evidence/phase5/landing_offset_diagnostic.md.
FROZEN_CONFIG_DIGEST = (
    "8d731c5e1dabe8764c40f3599037de530ae39de9f2537c1675e068f3c909b1f8"
)


def test_the_shipped_config_loads():
    cfg = load_phase5_config(CONFIG_PATH)
    assert cfg.experiment_name == "phase5_rung2"
    assert cfg.primary_alignment_level == "image"
    assert cfg.feature_encoder == "dinov2_vitb14"
    assert cfg.mean_vector_dir == "outputs/experiment_zero"
    assert cfg.phase4_dir == "outputs/phase4_rung1"


def test_the_config_digest_is_the_frozen_one():
    assert load_phase5_config(CONFIG_PATH).digest() == FROZEN_CONFIG_DIGEST


def test_the_digest_moves_with_any_value_it_covers():
    """Everything that decides what is measured, not only the architecture.

    The encoders, the input paths, the pair-subsampling seed, the sensitivity
    levels, and the controls all change what a gate would have had to verify, so
    a receipt bound to this digest must not survive any of them moving.
    """
    import dataclasses

    cfg = load_phase5_config(CONFIG_PATH)
    for field, value in (
        ("model", {**cfg.model, "d_model": 512}),
        ("training", {**cfg.training, "learning_rate": 1e-3}),
        ("tiny_overfit", {**cfg.tiny_overfit, "threshold_centered_cosine": 0.9}),
        ("primary_alignment_level", "affine"),
        ("feature_encoder", "dinov2_vitl14"),
        ("depth_encoder", "something_else"),
        ("renders_root", "elsewhere/renders"),
        ("cache_root", "elsewhere/cache"),
        ("mean_vector_dir", "elsewhere/mean"),
        ("phase4_dir", "elsewhere/phase4"),
        ("analysis_config", "elsewhere/analysis.yaml"),
        ("sensitivity_alignment_levels", ("none",)),
        ("diagnostic_alignment_levels", ()),
        ("controls", {"pose_shuffle": False}),
        ("landing_offset", {**cfg.landing_offset, "upper_edges_patch": [0.2, 0.3, 0.4, 0.5, 0.6]}),
        ("seed", 99),
    ):
        moved = dataclasses.replace(cfg, **{field: value})
        assert moved.digest() != cfg.digest(), field


def test_the_digest_ignores_only_where_outputs_are_written():
    """Relocation is the sole permitted exclusion, and the list is closed."""
    import dataclasses

    from lot.phase5 import Phase5Config

    cfg = load_phase5_config(CONFIG_PATH)
    assert Phase5Config.RELOCATION_FIELDS == ("output_root", "experiment_name")
    for field, value in (("output_root", "elsewhere"), ("experiment_name", "other")):
        moved = dataclasses.replace(cfg, **{field: value})
        assert moved.digest() == cfg.digest(), field

    # The exclusion list is an allowlist of exclusions, so a field added to the
    # config later is inside the identity without anyone remembering to add it.
    fields = {f.name for f in dataclasses.fields(cfg)}
    assert set(Phase5Config.RELOCATION_FIELDS) < fields
    assert len(fields) - len(Phase5Config.RELOCATION_FIELDS) >= 13


def test_unknown_config_keys_are_refused(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("experiment_name: x\nnot_a_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_phase5_config(path)


def test_the_frozen_training_values_match_the_dataclass_defaults():
    """The yaml and the dataclass must not disagree about what is frozen."""
    training = load_phase5_config(CONFIG_PATH).training
    defaults = TrainingConfig()
    assert training["optimizer"] == defaults.optimizer
    assert training["learning_rate"] == pytest.approx(defaults.learning_rate)
    assert training["weight_decay"] == pytest.approx(defaults.weight_decay)
    assert training["max_steps"] == defaults.max_steps
    assert training["warmup_steps"] == defaults.warmup_steps
    assert training["batch_pairs"] == defaults.batch_pairs
    assert training["validation_every_steps"] == defaults.validation_every_steps
    assert training["early_stopping_patience"] == defaults.early_stopping_patience
    assert training["checkpoint_selection"] == defaults.checkpoint_selection
    assert tuple(training["seeds"]) == defaults.seeds
    assert training["effective_batch_pairs"] == defaults.effective_batch_pairs


def test_the_recorded_parameter_count_is_the_built_model_s():
    cfg = load_phase5_config(CONFIG_PATH)
    model = build_predictor(predictor_config_from(cfg, (518, 518)))
    assert model.parameter_count() == cfg.model["parameter_count"]


def test_predictor_config_takes_its_grid_from_the_image_size():
    cfg = load_phase5_config(CONFIG_PATH)
    built = predictor_config_from(cfg, (518, 518))
    assert built.context_grid == (37, 37)
    assert built.target_grid == (37, 37)
    assert built.d_model == 384 and built.n_blocks == 6 and built.n_heads == 6


def test_task_index_maps_onto_fold_and_seed():
    seeds = (0, 1, 2)
    seen = set()
    for index in range(9):
        fold, seed = fold_and_seed_for_task(index, seeds)
        seen.add((fold.index, seed))
    assert len(seen) == 9, "every (fold, seed) must be trained exactly once"
    assert fold_and_seed_for_task(0, seeds)[0].index == 0
    assert fold_and_seed_for_task(8, seeds) == (frozen_folds()[2], 2)


def test_task_index_out_of_range_is_refused():
    with pytest.raises(ValueError, match="outside"):
        fold_and_seed_for_task(9, (0, 1, 2))
    with pytest.raises(ValueError, match="outside"):
        fold_and_seed_for_task(-1, (0, 1, 2))


def test_scene_scale_is_refused_as_a_phase5_condition():
    """Level 1 is a Phase 4 diagnostic and needs a leave-target-out estimator
    that has no place in a context-only Phase 5 condition."""
    from lot.phase5 import aligned_context_depth

    with pytest.raises(ValueError, match="not a Phase 5 condition"):
        aligned_context_depth(None, "frame", "scene")


def test_analysis_config_is_the_frozen_one():
    cfg = load_phase5_config(CONFIG_PATH)
    analysis = load_analysis_config(Path(cfg.analysis_config))
    # Phase 5 inherits Phase 3's measurement identity unchanged.
    assert analysis.measurement_digest() == "27244e6481d521159e513f2ea8799482"
    assert analysis.path_agreement_tolerance == 0.003
    assert analysis.bootstrap_seed == 20260825
    assert analysis.bootstrap_resamples == 1000
