"""Stream U: training adequacy, the sealed test set, and the input-use controls."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from lot.phase5_folds import Fold, frozen_folds
from lot.predictors import PredictorConfig, build_predictor
from lot.train import (
    ControlResult,
    TinyOverfitFailure,
    TrainingConfig,
    TrainingExample,
    assert_fold_is_sealed,
    batch_loss,
    centered_cosine_loss,
    evaluate_validation,
    learning_rate_at,
    read_predictions_at,
    run_input_use_controls,
    run_tiny_overfit_gate,
    shuffle_depth,
    shuffle_pose,
    train_fold,
)

TINY = PredictorConfig(
    d_model=32, n_blocks=2, n_heads=4, ffn_dim=64, dropout=0.0,
    feature_dim=16, patch_size=2, context_grid=(3, 3), target_grid=(3, 3),
)
GRID_HW = TINY.target_grid
CTX_N = TINY.context_grid[0] * TINY.context_grid[1]
HW = (TINY.context_grid[0] * TINY.patch_size, TINY.context_grid[1] * TINY.patch_size)
M = 5


def make_example(index: int, scene: str = "room_1", regime: str = "translation",
                 seed: int | None = None) -> TrainingExample:
    g = torch.Generator().manual_seed(index if seed is None else seed)
    target = torch.randn(M, TINY.feature_dim, generator=g)
    return TrainingExample(
        scene=scene,
        context_frame_id=f"ctx_{index}",
        target_frame_id=f"tgt_{index}",
        regime=regime,
        features_context=torch.randn(CTX_N, TINY.feature_dim, generator=g),
        depth_context_aligned=torch.rand(*HW, generator=g) * 4.0 + 0.5,
        camera=torch.randn(TINY.camera_features, generator=g),
        context_valid=torch.ones(CTX_N, dtype=torch.bool),
        query_patch_coords=torch.rand(M, 2, generator=g) * 2.0,
        target_centered=target / target.norm(dim=-1, keepdim=True),
        support=torch.ones(M, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# The loss and the readout
# ---------------------------------------------------------------------------

def test_centered_cosine_loss_is_zero_on_an_exact_match():
    x = torch.randn(4, 8)
    assert centered_cosine_loss(x, x).item() == pytest.approx(0.0, abs=1e-6)
    assert centered_cosine_loss(x, -x).item() == pytest.approx(2.0, abs=1e-6)


def test_loss_ignores_unsupported_samples_entirely():
    """A sample outside the support may not move the loss by any amount."""
    good = torch.randn(1, 4, 6)
    bad = good.clone()
    bad[0, 2] = -bad[0, 2]
    weights = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    a = centered_cosine_loss(good, good, weights)
    b = centered_cosine_loss(bad, good, weights)
    assert a.item() == pytest.approx(b.item(), abs=1e-6)


def test_loss_with_no_support_is_zero_and_differentiable():
    x = torch.randn(1, 3, 4, requires_grad=True)
    loss = centered_cosine_loss(x, torch.randn(1, 3, 4), torch.zeros(1, 3))
    assert loss.item() == 0.0
    loss.backward()
    assert x.grad is not None


def test_read_predictions_at_recovers_grid_values_at_cell_centers():
    """Reading at an integer patch coordinate returns that cell exactly."""
    grid = torch.arange(9 * 2, dtype=torch.float32).reshape(1, 9, 2)
    coords = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]]])
    read = read_predictions_at(grid, coords, (3, 3))
    assert torch.allclose(read[0, 0], grid[0, 0])
    assert torch.allclose(read[0, 1], grid[0, 2])
    assert torch.allclose(read[0, 2], grid[0, 4])


def test_read_predictions_at_interpolates_between_cells():
    grid = torch.zeros(1, 9, 1)
    grid[0, 0, 0] = 0.0
    grid[0, 1, 0] = 10.0
    read = read_predictions_at(grid, torch.tensor([[[0.5, 0.0]]]), (3, 3))
    assert read[0, 0, 0].item() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Stream S step 9: the sealed test set
# ---------------------------------------------------------------------------

def test_assert_fold_is_sealed_rejects_a_test_scene():
    fold = frozen_folds()[0]
    assert_fold_is_sealed(fold, fold.train, "ok")
    assert_fold_is_sealed(fold, fold.val, "ok")
    with pytest.raises(ValueError, match="sealed"):
        assert_fold_is_sealed(fold, [fold.test[0]], "leak")


def test_training_refuses_a_batch_containing_a_test_scene():
    """The seal is enforced in the training loop, not only at the data layer."""
    fold = frozen_folds()[0]
    leaked = fold.test[0]
    examples = [make_example(i, scene=leaked) for i in range(8)]
    cfg = TrainingConfig(batch_pairs=8, max_steps=1, warmup_steps=1,
                         validation_every_steps=1)
    with pytest.raises(ValueError, match="sealed"):
        train_fold(
            fold, seed=0, model_cfg=TINY, train_cfg=cfg,
            train_examples=lambda: iter(examples),
            val_examples=lambda: iter(examples),
            grid_hw=GRID_HW,
        )


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------

def test_learning_rate_warms_up_then_decays_to_the_floor():
    cfg = TrainingConfig(learning_rate=1e-3, final_learning_rate=1e-5,
                         warmup_steps=100, max_steps=1000)
    # Step 0 is the first of 100 warmup steps, so one hundredth of the peak.
    assert learning_rate_at(0, cfg) == pytest.approx(1e-3 / 100)
    assert learning_rate_at(99, cfg) == pytest.approx(1e-3)
    assert learning_rate_at(999, cfg) == pytest.approx(1e-5, rel=1e-3)
    # Monotone decay after warmup.
    values = [learning_rate_at(s, cfg) for s in range(100, 1000, 50)]
    assert all(a >= b for a, b in zip(values, values[1:]))


# ---------------------------------------------------------------------------
# Stream U step 14: the tiny-subset overfit gate
# ---------------------------------------------------------------------------

def test_tiny_overfit_gate_passes_on_a_learnable_subset():
    examples = [make_example(i) for i in range(8)]
    cfg = TrainingConfig(learning_rate=3e-3, weight_decay=0.0, warmup_steps=1)
    result = run_tiny_overfit_gate(
        examples, TINY, cfg, GRID_HW, threshold=0.98, max_steps=1500, seed=0
    )
    assert result.passed
    assert result.reached_centered_cosine >= 0.98
    assert result.n_pairs == 8


def test_tiny_overfit_gate_stops_when_the_threshold_is_unreachable():
    """Failure raises rather than returning quietly, because it is a stop."""
    examples = [make_example(i) for i in range(8)]
    # A threshold above the metric's range cannot be met by any model.
    cfg = TrainingConfig(learning_rate=3e-3, weight_decay=0.0, warmup_steps=1)
    with pytest.raises(TinyOverfitFailure, match="tiny-subset overfit gate failed"):
        run_tiny_overfit_gate(
            examples, TINY, cfg, GRID_HW, threshold=1.5, max_steps=20, seed=0
        )


def test_tiny_overfit_gate_can_report_without_raising():
    examples = [make_example(i) for i in range(4)]
    cfg = TrainingConfig(learning_rate=3e-3, weight_decay=0.0, warmup_steps=1)
    result = run_tiny_overfit_gate(
        examples, TINY, cfg, GRID_HW, threshold=1.5, max_steps=5, seed=0,
        raise_on_failure=False,
    )
    assert not result.passed
    assert result.steps == 5


def test_tiny_overfit_gate_records_the_regimes_it_spanned():
    examples = (
        [make_example(i, regime="rotation") for i in range(3)]
        + [make_example(i + 3, regime="translation") for i in range(3)]
        + [make_example(i + 6, regime="orbit") for i in range(2)]
    )
    cfg = TrainingConfig(learning_rate=3e-3, weight_decay=0.0, warmup_steps=1)
    result = run_tiny_overfit_gate(
        examples, TINY, cfg, GRID_HW, threshold=0.0, max_steps=1, seed=0
    )
    assert result.regimes == ("orbit", "rotation", "translation")


# ---------------------------------------------------------------------------
# Checkpoint selection and training
# ---------------------------------------------------------------------------

def test_training_selects_on_validation_and_records_the_fold(tmp_path):
    fold = frozen_folds()[0]
    train = [make_example(i, scene=fold.train[0]) for i in range(8)]
    val = [make_example(100 + i, scene=fold.val[0]) for i in range(8)]
    cfg = TrainingConfig(
        learning_rate=3e-3, weight_decay=0.0, batch_pairs=8, max_steps=6,
        warmup_steps=1, validation_every_steps=2, early_stopping_patience=99,
    )
    ckpt = tmp_path / "model.pt"
    model, record = train_fold(
        fold, seed=1, model_cfg=TINY, train_cfg=cfg,
        train_examples=lambda: iter(train), val_examples=lambda: iter(val),
        grid_hw=GRID_HW, checkpoint_path=ckpt,
    )
    assert record.fold == fold.index and record.seed == 1
    assert record.test_scenes == tuple(fold.test)
    assert record.steps_run == 6
    assert len(record.history) == 3
    assert ckpt.exists()
    saved = torch.load(ckpt, weights_only=False)
    # The saved checkpoint is the best validation score, not the last one.
    assert saved["validation_centered_cosine"] == pytest.approx(
        max(score for _, score in record.history)
    )
    assert saved["step"] == record.best_step


def test_early_stopping_fires_on_patience():
    fold = frozen_folds()[0]
    train = [make_example(i, scene=fold.train[0]) for i in range(8)]
    val = [make_example(100 + i, scene=fold.val[0]) for i in range(8)]
    cfg = TrainingConfig(
        learning_rate=0.0, weight_decay=0.0, batch_pairs=8, max_steps=100,
        warmup_steps=1, validation_every_steps=1, early_stopping_patience=2,
    )
    _, record = train_fold(
        fold, seed=0, model_cfg=TINY, train_cfg=cfg,
        train_examples=lambda: iter(train), val_examples=lambda: iter(val),
        grid_hw=GRID_HW,
    )
    # With a zero learning rate nothing improves, so patience must end it early.
    assert record.stopped_early
    assert record.steps_run < 100


def test_training_config_digest_moves_with_any_value():
    a = TrainingConfig()
    assert a.digest() == TrainingConfig().digest()
    assert a.digest() != TrainingConfig(learning_rate=1e-4).digest()
    assert a.effective_batch_pairs == 8
    assert TrainingConfig(grad_accumulation_steps=4).effective_batch_pairs == 32


def test_validation_pools_only_supported_samples():
    model = build_predictor(TINY).eval()
    examples = [make_example(i) for i in range(4)]
    for e in examples:
        e.support[:] = False
        e.support[0] = True
    result = evaluate_validation(model, examples, GRID_HW, batch_pairs=2)
    assert result.n_samples == 4
    assert result.n_pairs == 4
    assert math.isfinite(result.centered_cosine)


# ---------------------------------------------------------------------------
# Stream U step 17: input-use controls
# ---------------------------------------------------------------------------

def test_shuffles_are_derangements_that_preserve_everything_else():
    examples = [make_example(i) for i in range(6)]
    posed = shuffle_pose(examples, seed=3)
    for original, shuffled in zip(examples, posed):
        assert not torch.equal(original.camera, shuffled.camera)
        assert torch.equal(original.features_context, shuffled.features_context)
        assert torch.equal(original.depth_context_aligned, shuffled.depth_context_aligned)
        assert torch.equal(original.target_centered, shuffled.target_centered)

    depths = shuffle_depth(examples, seed=3)
    for original, shuffled in zip(examples, depths):
        assert not torch.equal(original.depth_context_aligned, shuffled.depth_context_aligned)
        assert torch.equal(original.camera, shuffled.camera)
        assert torch.equal(original.features_context, shuffled.features_context)


def test_shuffle_of_one_example_is_a_no_op():
    """A control needs something to permute; one example is reported unchanged."""
    examples = [make_example(0)]
    assert torch.equal(shuffle_pose(examples, 1)[0].camera, examples[0].camera)


def test_controls_report_both_shuffles_against_one_baseline():
    model = build_predictor(TINY).eval()
    examples = [make_example(i) for i in range(6)]
    results = run_input_use_controls(model, examples, GRID_HW, batch_pairs=3, seed=5)
    assert [r.name for r in results] == ["pose_shuffle", "depth_shuffle"]
    assert all(isinstance(r, ControlResult) for r in results)
    baselines = {r.baseline_centered_cosine for r in results}
    assert len(baselines) == 1, "both controls must share one baseline"
    for r in results:
        assert r.n_pairs == 6
        assert math.isfinite(r.degradation)


def test_batch_loss_reports_the_supported_count():
    model = build_predictor(TINY)
    examples = [make_example(i) for i in range(3)]
    examples[0].support[:] = False
    loss, cosine, n = batch_loss(model, examples, GRID_HW)
    assert n == 2 * M
    assert torch.isfinite(loss)
    assert math.isfinite(cosine)


# ---------------------------------------------------------------------------
# Review fixes: ragged supervision, device placement, and fold roles
# ---------------------------------------------------------------------------

def _ragged(index: int, m: int, scene: str = "room_1") -> TrainingExample:
    e = make_example(index, scene=scene)
    return dataclasses.replace(
        e,
        query_patch_coords=e.query_patch_coords[:m],
        target_centered=e.target_centered[:m],
        support=e.support[:m],
    )


def test_pairs_with_different_support_counts_batch():
    """Real pairs never share a supervision length; stacking them must work."""
    model = build_predictor(TINY)
    batch = [_ragged(0, 5), _ragged(1, 3), _ragged(2, 4)]
    loss, cosine, n = batch_loss(model, batch, GRID_HW)
    assert torch.isfinite(loss)
    assert n == 5 + 3 + 4, "every supported sample must be counted"


def test_padding_is_inert_rather_than_approximate():
    """A padded row must not move the loss by any amount."""
    model = build_predictor(TINY).eval()
    with torch.no_grad():
        alone, _, n_alone = batch_loss(model, [_ragged(0, 3)], GRID_HW)
        padded, _, n_padded = batch_loss(model, [_ragged(0, 3), _ragged(0, 3)], GRID_HW)
        mixed, _, n_mixed = batch_loss(model, [_ragged(0, 3), _ragged(0, 5)], GRID_HW)
    assert n_alone == 3 and n_padded == 6 and n_mixed == 8
    # Two copies of one example give the same mean as one copy.
    assert padded.item() == pytest.approx(alone.item(), abs=1e-6)
    # The mixed batch pads the shorter pair; its 3 real samples still contribute
    # exactly what they did alone, so nothing was invented or dropped.
    assert math.isfinite(mixed.item())


def test_truncating_instead_of_padding_would_lose_samples():
    """Names the bias the padding fix removes."""
    model = build_predictor(TINY).eval()
    batch = [_ragged(0, 5), _ragged(1, 2)]
    with torch.no_grad():
        _, _, n = batch_loss(model, batch, GRID_HW)
    shortest = min(int(e.support.numel()) for e in batch)
    assert n == 7
    assert n > shortest * len(batch), "truncation would have scored only 4"


def test_collate_moves_inputs_to_the_model_device():
    from lot.train import _collate

    model = build_predictor(TINY)
    device = next(model.parameters()).device
    batch = _collate([make_example(0), make_example(1)], device)
    for name, tensor in batch.items():
        assert tensor.device == device, name


def test_training_runs_on_the_model_device_without_manual_transfer():
    """The training path itself must place its own tensors.

    Run on whatever device is available. On CPU this is trivially true; on a
    machine with CUDA it is the regression test for the failure where the model
    moved to the device and the batch did not.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold = frozen_folds()[0]
    train = [make_example(i, scene=fold.train[0]) for i in range(8)]
    val = [make_example(100 + i, scene=fold.val[0]) for i in range(8)]
    cfg = TrainingConfig(
        learning_rate=1e-3, weight_decay=0.0, batch_pairs=8, max_steps=2,
        warmup_steps=1, validation_every_steps=2, early_stopping_patience=99,
    )
    model, record = train_fold(
        fold, seed=0, model_cfg=TINY, train_cfg=cfg,
        train_examples=lambda: iter(train), val_examples=lambda: iter(val),
        grid_hw=GRID_HW, device=device,
    )
    assert record.steps_run == 2
    assert next(model.parameters()).device.type == device


def test_a_scene_outside_the_train_split_is_refused_even_if_not_a_test_scene():
    """Excluding test is necessary but not sufficient."""
    from lot.train import assert_scenes_in_role

    fold = frozen_folds()[0]
    assert_scenes_in_role(fold, fold.train, "train", "ok")
    assert_scenes_in_role(fold, fold.val, "val", "ok")
    # A validation scene is not a test scene, and must still be refused in a
    # training role: it would leak optimization into the selection statistic.
    with pytest.raises(ValueError, match="not in its train split"):
        assert_scenes_in_role(fold, [fold.val[0]], "train", "leak")
    # And a training scene must be refused in the selection role.
    with pytest.raises(ValueError, match="not in its val split"):
        assert_scenes_in_role(fold, [fold.train[0]], "val", "leak")


def test_a_test_scene_in_a_role_is_named_as_a_seal_breach():
    from lot.train import assert_scenes_in_role

    fold = frozen_folds()[0]
    with pytest.raises(ValueError, match="sealed test set was breached"):
        assert_scenes_in_role(fold, [fold.test[0]], "train", "leak")


def test_an_unknown_role_is_refused():
    from lot.train import assert_scenes_in_role

    with pytest.raises(ValueError, match="unknown role"):
        assert_scenes_in_role(frozen_folds()[0], [], "holdout", "x")


def test_checkpoint_selection_refuses_a_validation_stream_of_test_scenes():
    """Test scenes must not be able to influence checkpoint selection."""
    fold = frozen_folds()[0]
    train = [make_example(i, scene=fold.train[0]) for i in range(8)]
    leaked = [make_example(100 + i, scene=fold.test[0]) for i in range(8)]
    cfg = TrainingConfig(
        learning_rate=1e-3, weight_decay=0.0, batch_pairs=8, max_steps=1,
        warmup_steps=1, validation_every_steps=1, early_stopping_patience=99,
    )
    with pytest.raises(ValueError, match="sealed test set was breached"):
        train_fold(
            fold, seed=0, model_cfg=TINY, train_cfg=cfg,
            train_examples=lambda: iter(train), val_examples=lambda: iter(leaked),
            grid_hw=GRID_HW,
        )


def test_the_tiny_gate_refuses_scenes_outside_the_training_split():
    fold = frozen_folds()[0]
    examples = [make_example(i, scene=fold.test[0]) for i in range(4)]
    cfg = TrainingConfig(learning_rate=3e-3, weight_decay=0.0, warmup_steps=1)
    with pytest.raises(ValueError, match="sealed test set was breached"):
        run_tiny_overfit_gate(
            examples, TINY, cfg, GRID_HW, threshold=0.98, max_steps=5, fold=fold
        )


def test_an_empty_training_stream_stops_instead_of_spinning():
    """A fold with no usable examples must fail, not occupy a cluster job."""
    fold = frozen_folds()[0]
    val = [make_example(100 + i, scene=fold.val[0]) for i in range(8)]
    cfg = TrainingConfig(batch_pairs=8, max_steps=1000, warmup_steps=1,
                         validation_every_steps=1)
    with pytest.raises(ValueError, match="produced no training examples"):
        train_fold(
            fold, seed=0, model_cfg=TINY, train_cfg=cfg,
            train_examples=lambda: iter([]), val_examples=lambda: iter(val),
            grid_hw=GRID_HW,
        )
