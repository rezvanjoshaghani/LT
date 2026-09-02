"""Phase 5 training: Stream U. Single GPU, cached features, frozen configuration.

Everything in this module that could influence a result is read from
configs/phase5.yaml, which was frozen before any Phase 5 model was trained.
Nothing here inspects a test scene. The fold object a run is given exposes only
train and val; a test scene reaching this module is a programming error and
raises rather than silently training on it.

The three adequacy instruments Stream U requires all live here.

The tiny-subset overfit gate establishes that the architecture and optimizer can
represent the supervised mapping at all. It is run before full training and its
threshold is frozen in the config. Its purpose is negative: without it, a
predictor that underperforms the explicit comparator could be underperforming
because transformation learning is genuinely hard, or because the trunk could
not fit anything, and the Phase 5 conclusion would not distinguish those.

The pose-shuffle and depth-shuffle controls ask whether the network uses the
inputs it is given. They are diagnostics and never acceptance thresholds: a
shuffle that changes little is reported as such, and is not permitted to become
a reason to retrain.

Checkpoint selection reads validation centered cosine and nothing else.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from .encoders import sample_map_bilinear
from .phase5_folds import Fold
from .predictors import PredictWithDepth, PredictorConfig, build_predictor

# The frozen loss. Centered cosine, following PROTOCOL 3.7's primary metric, so
# what is optimized and what is reported are the same quantity.
def centered_cosine_loss(
    predicted_centered: Tensor, target_centered: Tensor, weights: Tensor | None = None
) -> Tensor:
    """1 - centered cosine, averaged over supported samples.

    predicted_centered, target_centered: [..., C] already-centered features.
    weights: optional [...] float mask. Samples with zero weight contribute
        nothing; the mean is over the weight, so a pair with few supported
        samples does not silently dominate through a smaller denominator.
    """
    cosine = torch.nn.functional.cosine_similarity(
        predicted_centered, target_centered, dim=-1, eps=1e-8
    )
    loss = 1.0 - cosine
    if weights is None:
        return loss.mean()
    total = weights.sum()
    if float(total) == 0.0:
        return loss.sum() * 0.0
    return (loss * weights).sum() / total


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """The frozen training configuration, Stream U step 15."""

    optimizer: str = "adamw"
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8
    grad_clip_norm: float = 1.0
    batch_pairs: int = 8
    grad_accumulation_steps: int = 1
    max_steps: int = 20000
    warmup_steps: int = 500
    scheduler: str = "cosine"
    final_learning_rate: float = 3.0e-5
    precision: str = "fp32"
    allow_tf32: bool = True
    validation_every_steps: int = 500
    early_stopping_patience: int = 10
    checkpoint_selection: str = "best_validation_centered_cosine"
    seeds: tuple[int, ...] = (0, 1, 2)

    @property
    def effective_batch_pairs(self) -> int:
        return self.batch_pairs * self.grad_accumulation_steps

    def digest(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True, default=list)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def learning_rate_at(step: int, cfg: TrainingConfig) -> float:
    """Linear warmup then cosine decay to the frozen floor.

    Written as a pure function of the step so a resumed run cannot drift onto a
    different schedule from the one that produced its earlier steps.
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(cfg.warmup_steps, 1)
    if cfg.scheduler != "cosine":
        return cfg.learning_rate
    span = max(cfg.max_steps - cfg.warmup_steps, 1)
    progress = min((step - cfg.warmup_steps) / span, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.final_learning_rate + (cfg.learning_rate - cfg.final_learning_rate) * cosine


@dataclasses.dataclass
class TrainingExample:
    """One context-target pair, already reduced to what the model may see.

    Assembled outside this module by the Phase 5 data layer, which is the only
    place that touches caches and geometry. Everything here is either a
    permitted model input or a support mask used to place a loss.
    """

    scene: str
    context_frame_id: str
    target_frame_id: str
    regime: str
    features_context: Tensor          # [N_ctx, C] frozen DINOv2 patch features
    depth_context_aligned: Tensor     # [H, W] aligned context depth, meters
    camera: Tensor                    # [camera_features]
    context_valid: Tensor             # [N_ctx] bool
    # Supervision, at the context-lift landing locations. Support only: these
    # never enter the model, they decide where the loss is taken.
    query_patch_coords: Tensor        # [M, 2] target patch-grid coordinates
    target_centered: Tensor           # [M, C] centered target features
    support: Tensor                   # [M] bool


def _collate(
    examples: Sequence[TrainingExample], device: str | torch.device = "cpu"
) -> dict[str, Tensor]:
    """Stack the model inputs and move them to the device the model is on.

    The transfer belongs here rather than at the call sites. Leaving it to the
    caller meant train_fold could move the model to CUDA and then hand it CPU
    tensors, which fails at the first forward pass; the integration gate did not
    catch it because it moved its own batch by hand and so never exercised this
    path.
    """
    return {
        "features_context": torch.stack(
            [e.features_context for e in examples]
        ).to(device),
        "depth_context_aligned": torch.stack(
            [e.depth_context_aligned for e in examples]
        ).to(device),
        "camera": torch.stack([e.camera for e in examples]).to(device),
        "context_valid": torch.stack([e.context_valid for e in examples]).to(device),
    }


def _pad_supervision(
    examples: Sequence[TrainingExample], device: str | torch.device = "cpu"
) -> tuple[Tensor, Tensor, Tensor]:
    """Stack per-pair supervision of differing lengths, padding with no support.

    The number of supervised landings varies by camera pair, because it is the
    count of samples the explicit comparator could validly transport and ground
    truth could referee. Stacking those directly raises the moment two pairs
    disagree, which is every real batch.

    Padding is inert rather than approximate: a padded row carries support 0, and
    centered_cosine_loss divides by the summed support, so a padded entry moves
    neither the numerator nor the denominator. Truncating to the shortest pair
    instead, which is what the integration gate did to get a shape probe, would
    silently discard real supervised samples and bias every batch toward the
    pairs with the least support.
    """
    lengths = [int(e.support.numel()) for e in examples]
    width = max(lengths) if lengths else 0
    coords, targets, support = [], [], []
    for example in examples:
        pad = width - int(example.support.numel())
        q = example.query_patch_coords
        t = example.target_centered
        s = example.support
        if pad:
            q = torch.cat([q, q.new_zeros(pad, q.shape[-1])])
            t = torch.cat([t, t.new_zeros(pad, t.shape[-1])])
            s = torch.cat([s, s.new_zeros(pad, dtype=s.dtype)])
        coords.append(q)
        targets.append(t)
        support.append(s)
    return (
        torch.stack(coords).to(device),
        torch.stack(targets).to(device),
        torch.stack(support).to(device),
    )


def read_predictions_at(
    predicted_grid: Tensor, query_patch_coords: Tensor, grid_hw: tuple[int, int]
) -> Tensor:
    """Read a predicted target grid at continuous patch-grid coordinates.

    predicted_grid: [B, N, C] row-major over the target patch lattice.
    query_patch_coords: [B, M, 2] continuous (x, y) in patch-grid units.

    Uses sample_map_bilinear, the one reader CLAUDE.md keeps this project's
    patch-grid interpolation defined in, rather than a second implementation.
    Training and scoring therefore read the predicted field through identical
    arithmetic, so no train-versus-test readout mismatch can appear inside the
    gap this phase reports. The batch loop is over pairs, which is eight, and is
    invisible next to the transformer forward.
    """
    batch, _, channels = predicted_grid.shape
    grid_h, grid_w = grid_hw
    maps = predicted_grid.permute(0, 2, 1).reshape(batch, channels, grid_h, grid_w)
    return torch.stack(
        [sample_map_bilinear(maps[b], query_patch_coords[b]) for b in range(batch)]
    )


def batch_loss(
    model: PredictWithDepth,
    examples: Sequence[TrainingExample],
    grid_hw: tuple[int, int],
) -> tuple[Tensor, float, int]:
    """Loss and mean centered cosine over one batch's supported samples."""
    device = next(model.parameters()).device
    batch = _collate(examples, device)
    predicted = model(
        batch["features_context"],
        batch["depth_context_aligned"],
        batch["camera"],
        context_valid=batch["context_valid"],
    )
    queries, targets, support_mask = _pad_supervision(examples, device)
    support = support_mask.to(predicted.dtype)

    read = read_predictions_at(predicted, queries, grid_hw)
    loss = centered_cosine_loss(read, targets, weights=support)
    with torch.no_grad():
        cosine = torch.nn.functional.cosine_similarity(read, targets, dim=-1, eps=1e-8)
        n = int(support.sum())
        mean_cosine = float((cosine * support).sum() / support.sum()) if n else float("nan")
    return loss, mean_cosine, n


@dataclasses.dataclass
class ValidationResult:
    centered_cosine: float
    n_samples: int
    n_pairs: int


@torch.no_grad()
def evaluate_validation(
    model: PredictWithDepth,
    examples: Iterable[TrainingExample],
    grid_hw: tuple[int, int],
    batch_pairs: int,
) -> ValidationResult:
    """Mean centered cosine over validation pairs. The checkpoint statistic.

    Pooled over samples rather than averaged over pairs, deliberately and only
    here: this is a model-selection statistic, not a reported estimand. Every
    reported Phase 5 number uses the frozen pair-level estimand of PROTOCOL 3.4.
    """
    was_training = model.training
    model.eval()
    total, count, pairs = 0.0, 0, 0
    buffer: list[TrainingExample] = []

    def flush() -> None:
        nonlocal total, count, pairs
        if not buffer:
            return
        device = next(model.parameters()).device
        batch = _collate(buffer, device)
        predicted = model(
            batch["features_context"], batch["depth_context_aligned"],
            batch["camera"], context_valid=batch["context_valid"],
        )
        queries, targets, support_mask = _pad_supervision(buffer, device)
        support = support_mask.to(predicted.dtype)
        read = read_predictions_at(predicted, queries, grid_hw)
        cosine = torch.nn.functional.cosine_similarity(read, targets, dim=-1, eps=1e-8)
        total += float((cosine * support).sum())
        count += int(support.sum())
        pairs += len(buffer)
        buffer.clear()

    for example in examples:
        buffer.append(example)
        if len(buffer) == batch_pairs:
            flush()
    flush()
    if was_training:
        model.train()
    return ValidationResult(
        centered_cosine=total / count if count else float("nan"),
        n_samples=count,
        n_pairs=pairs,
    )


@dataclasses.dataclass
class TrainingRecord:
    """What a finished run reports, for the Stream AC training-adequacy table."""

    fold: int
    seed: int
    train_scenes: tuple[str, ...]
    val_scenes: tuple[str, ...]
    test_scenes: tuple[str, ...]
    steps_run: int
    best_step: int
    best_validation_centered_cosine: float
    parameter_count: int
    training_config_digest: str
    stopped_early: bool
    history: tuple[tuple[int, float], ...]


def assert_fold_is_sealed(fold: Fold, scenes: Sequence[str], where: str) -> None:
    """No test scene of this fold may appear in anything training consumes.

    Stream S step 9 makes the sealed test set a correctness condition rather
    than a habit, so it is asserted at every entry point instead of being left
    to the caller to remember.
    """
    leaked = sorted(set(scenes) & set(fold.test))
    if leaked:
        raise ValueError(
            f"{where}: fold {fold.index} test scenes {leaked} reached training; "
            "the test set is sealed and this run is invalid"
        )


def assert_scenes_in_role(
    fold: Fold, scenes: Sequence[str], role: str, where: str
) -> None:
    """Every scene must belong to the split whose role it is being used for.

    Excluding the test set is necessary but not sufficient. A scene absent from
    the fold entirely, or a validation scene appearing in an optimizer step, or
    a training scene appearing in checkpoint selection, all pass a test-only
    check and all corrupt the experiment in different directions: the first
    means the fold is not what the split hash says, the second leaks
    optimization into the selection statistic, and the third selects a
    checkpoint on data the model was fitted to.

    So membership is asserted positively, against the named role, rather than
    negatively against the test set alone.
    """
    allowed = {"train": set(fold.train), "val": set(fold.val)}.get(role)
    if allowed is None:
        raise ValueError(f"{where}: unknown role {role!r}; use 'train' or 'val'")
    intruders = sorted(set(scenes) - allowed)
    if intruders:
        in_test = sorted(set(intruders) & set(fold.test))
        detail = (
            f" and {in_test} are this fold's test scenes, so the sealed test set "
            "was breached and this run is invalid"
            if in_test else ""
        )
        raise ValueError(
            f"{where}: fold {fold.index} {role} role received scenes {intruders} "
            f"that are not in its {role} split{detail}"
        )


def train_fold(
    fold: Fold,
    seed: int,
    model_cfg: PredictorConfig,
    train_cfg: TrainingConfig,
    train_examples: Callable[[], Iterable[TrainingExample]],
    val_examples: Callable[[], Iterable[TrainingExample]],
    grid_hw: tuple[int, int],
    device: str = "cpu",
    checkpoint_path: Path | None = None,
    max_steps: int | None = None,
) -> tuple[PredictWithDepth, TrainingRecord]:
    """Train one (fold, seed) model. Checkpoint selection on validation only.

    train_examples and val_examples are called to obtain a fresh iterator, so a
    run that needs more steps than one pass simply asks for another pass.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if train_cfg.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_predictor(model_cfg).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        eps=train_cfg.adam_eps,
    )

    limit = train_cfg.max_steps if max_steps is None else max_steps
    best = -math.inf
    best_step = -1
    since_improvement = 0
    history: list[tuple[int, float]] = []
    stopped_early = False
    step = 0

    stream = iter(train_examples())
    buffer: list[TrainingExample] = []
    yielded_this_pass = 0
    while step < limit:
        try:
            buffer.append(next(stream))
            yielded_this_pass += 1
        except StopIteration:
            # A pass that produced nothing means the fold has no usable example
            # at all. Restarting the iterator would spin forever and hold a
            # cluster job open with no output, so it is a classified stop.
            if yielded_this_pass == 0:
                raise ValueError(
                    f"train_fold: fold {fold.index} seed {seed} produced no "
                    "training examples in a full pass over its training scenes "
                    f"{list(fold.train)}. Nothing can be optimized; this is a "
                    "data or support failure, not a training outcome."
                ) from None
            stream = iter(train_examples())
            yielded_this_pass = 0
            continue
        if len(buffer) < train_cfg.batch_pairs:
            continue

        assert_scenes_in_role(fold, [e.scene for e in buffer], "train", "train_fold")
        loss, _, _ = batch_loss(model, buffer, grid_hw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate_at(step, train_cfg)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        buffer.clear()
        step += 1

        if step % train_cfg.validation_every_steps == 0 or step == limit:
            # Checkpoint selection may see validation scenes and nothing else.
            # Asserted on the examples actually consumed, so a mislabelled
            # stream cannot quietly select on train or test data.
            def validation_stream():
                for example in val_examples():
                    assert_scenes_in_role(
                        fold, [example.scene], "val", "checkpoint selection"
                    )
                    yield example

            result = evaluate_validation(
                model, validation_stream(), grid_hw, train_cfg.batch_pairs
            )
            history.append((step, result.centered_cosine))
            if result.centered_cosine > best:
                best = result.centered_cosine
                best_step = step
                since_improvement = 0
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "step": step,
                            "validation_centered_cosine": best,
                            "fold": fold.index,
                            "seed": seed,
                            "training_config_digest": train_cfg.digest(),
                        },
                        checkpoint_path,
                    )
            else:
                since_improvement += 1
                if since_improvement >= train_cfg.early_stopping_patience:
                    stopped_early = True
                    break

    record = TrainingRecord(
        fold=fold.index,
        seed=seed,
        train_scenes=tuple(fold.train),
        val_scenes=tuple(fold.val),
        test_scenes=tuple(fold.test),
        steps_run=step,
        best_step=best_step,
        best_validation_centered_cosine=best if math.isfinite(best) else float("nan"),
        parameter_count=model.parameter_count(),
        training_config_digest=train_cfg.digest(),
        stopped_early=stopped_early,
        history=tuple(history),
    )
    return model, record


# ---------------------------------------------------------------------------
# Stream U step 14: the tiny-subset overfit gate
# ---------------------------------------------------------------------------

class TinyOverfitFailure(RuntimeError):
    """The architecture could not fit eight pairs. A stop, not a warning."""


@dataclasses.dataclass
class TinyOverfitResult:
    reached_centered_cosine: float
    threshold: float
    steps: int
    passed: bool
    n_pairs: int
    regimes: tuple[str, ...]


def run_tiny_overfit_gate(
    examples: Sequence[TrainingExample],
    model_cfg: PredictorConfig,
    train_cfg: TrainingConfig,
    grid_hw: tuple[int, int],
    threshold: float,
    max_steps: int,
    seed: int = 0,
    device: str = "cpu",
    raise_on_failure: bool = True,
    fold: Fold | None = None,
) -> TinyOverfitResult:
    """Fit the tiny subset and report whether the frozen threshold was reached.

    The subset is training scenes only, and when a fold is supplied that is
    asserted rather than assumed. Reaching the threshold says nothing about
    generalization and is not meant to: it says the supervised mapping is
    representable by this trunk under this optimizer, which is the precondition
    for reading anything scientific into a later underperformance.
    """
    if fold is not None:
        assert_scenes_in_role(
            fold, [e.scene for e in examples], "train", "tiny overfit gate"
        )
    torch.manual_seed(seed)
    model = build_predictor(model_cfg).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        eps=train_cfg.adam_eps,
    )
    best = -math.inf
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        loss, cosine, n = batch_loss(model, examples, grid_hw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        optimizer.step()
        if n and cosine > best:
            best = cosine
        if best >= threshold:
            break

    passed = best >= threshold
    result = TinyOverfitResult(
        reached_centered_cosine=best if math.isfinite(best) else float("nan"),
        threshold=threshold,
        steps=step + 1,
        passed=passed,
        n_pairs=len(examples),
        regimes=tuple(sorted({e.regime for e in examples})),
    )
    if not passed and raise_on_failure:
        raise TinyOverfitFailure(
            f"tiny-subset overfit gate failed: reached centered cosine "
            f"{result.reached_centered_cosine:.4f} against the frozen threshold "
            f"{threshold} after {result.steps} steps over {len(examples)} pairs. "
            "Phase 5 stops here: a predictor that cannot fit this sample cannot "
            "support a scientific reading of its underperformance at scale."
        )
    return result


# ---------------------------------------------------------------------------
# Stream U step 17: input-use controls
# ---------------------------------------------------------------------------

def shuffle_pose(
    examples: Sequence[TrainingExample], seed: int
) -> list[TrainingExample]:
    """Permute the camera vectors between examples, holding everything else fixed.

    A derangement where the sample allows one, so no example keeps its own
    camera by accident and dilutes the control.
    """
    return _permute_field(examples, "camera", seed)


def shuffle_depth(
    examples: Sequence[TrainingExample], seed: int
) -> list[TrainingExample]:
    """Permute the aligned context depth maps, holding features and cameras fixed."""
    return _permute_field(examples, "depth_context_aligned", seed)


def _permute_field(
    examples: Sequence[TrainingExample], field: str, seed: int
) -> list[TrainingExample]:
    n = len(examples)
    if n < 2:
        return list(examples)
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    for _ in range(64):
        rng.shuffle(order)
        if not np.any(order == np.arange(n)):
            break
    else:
        order = np.roll(np.arange(n), 1)
    return [
        dataclasses.replace(example, **{field: getattr(examples[order[i]], field)})
        for i, example in enumerate(examples)
    ]


@dataclasses.dataclass
class ControlResult:
    name: str
    baseline_centered_cosine: float
    shuffled_centered_cosine: float
    n_pairs: int

    @property
    def degradation(self) -> float:
        return self.baseline_centered_cosine - self.shuffled_centered_cosine


@torch.no_grad()
def run_input_use_controls(
    model: PredictWithDepth,
    examples: Sequence[TrainingExample],
    grid_hw: tuple[int, int],
    batch_pairs: int,
    seed: int,
) -> list[ControlResult]:
    """Pose and depth shuffles on validation pairs. Diagnostics, never thresholds.

    A shuffle that barely moves the score is reported exactly as observed. It is
    evidence about whether the network uses that input, and it is not a reason
    to retrain, retune, or reinterpret the headline gap.
    """
    baseline = evaluate_validation(model, examples, grid_hw, batch_pairs).centered_cosine
    results = []
    for name, shuffler in (("pose_shuffle", shuffle_pose), ("depth_shuffle", shuffle_depth)):
        shuffled = shuffler(examples, seed)
        score = evaluate_validation(model, shuffled, grid_hw, batch_pairs).centered_cosine
        results.append(
            ControlResult(
                name=name,
                baseline_centered_cosine=baseline,
                shuffled_centered_cosine=score,
                n_pairs=len(examples),
            )
        )
    return results
