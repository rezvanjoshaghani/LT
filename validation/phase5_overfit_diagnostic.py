"""Diagnose a failed tiny-subset overfit gate. A diagnostic, never a gate.

Written for the gate as it stood at 1e7ce8f. That gate
(lot.train.run_tiny_overfit_gate) trained on the frozen eight-pair subset with
the model in train mode and reported the best training-batch centered cosine.
Dropout was active in that forward pass, so the reported value mixed how well
the trunk fits the subset with dropout noise. The gate now scores with dropout
off, as validation/evidence/phase5/overfit_gate_amendment.md records. This
script replays the gate's training exactly, with the same subset, seed,
optimizer, clipping, and steps, and records two values every few steps:

    train_mode   the gate's own measure, dropout active, from the training forward
    eval_mode    the same batch scored with dropout off, a deterministic forward

It also records each pair's eval-mode cosine and each regime's. The eval
forward draws no random numbers, so the training trajectory is the gate's: the
best train-mode value should match the gate's receipt, up to GPU
nondeterminism.

It writes only to --out, which must lie outside the repository, so the
checkout stays clean and at the commit the run's receipts are bound to. It
changes nothing the gate decided. Its output informs a decision that belongs
to the user.

Run from the repository root:

    PYTHONPATH=src python <path>/phase5_overfit_diagnostic.py \\
        --config configs/phase5.yaml --out <dir outside the repo>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


def _pair_label(example: Any) -> str:
    return f"{example.scene} {example.context_frame_id} -> {example.target_frame_id}"


def _eval_cosines(model: Any, examples: Sequence[Any], grid: tuple[int, int]) -> tuple[float, list[float]]:
    """The pooled eval-mode cosine of the batch and each pair's own, dropout off."""
    from lot.train import batch_loss

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, pooled, _ = batch_loss(model, examples, grid)
        per_pair = [batch_loss(model, [example], grid)[1] for example in examples]
    if was_training:
        model.train()
    return pooled, per_pair


def diagnose(
    examples: Sequence[Any],
    model_cfg: Any,
    train_cfg: Any,
    grid: tuple[int, int],
    max_steps: int,
    seed: int,
    device: str,
    every: int = 50,
) -> dict[str, Any]:
    """Replay the gate's loop and record train-mode and eval-mode cosines.

    The training loop is lot.train.run_tiny_overfit_gate's, statement for
    statement, except that it never stops early and it scores the batch in eval
    mode at every `every` steps and at the end, rather than after every step.
    """
    from lot.predictors import build_predictor
    from lot.train import batch_loss

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
    rows: list[dict[str, Any]] = []
    best_train = -math.inf
    best_eval = -math.inf
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        loss, cosine, n = batch_loss(model, examples, grid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        optimizer.step()
        if n and cosine > best_train:
            best_train = cosine
        last = step + 1 == max_steps
        if (step + 1) % every == 0 or step == 0 or last:
            pooled, _ = _eval_cosines(model, examples, grid)
            best_eval = max(best_eval, pooled)
            rows.append({
                "step": step + 1,
                "train_mode": cosine,
                "eval_mode": pooled,
                "loss": float(loss.detach()),
            })
    pooled, per_pair = _eval_cosines(model, examples, grid)
    regimes: dict[str, list[float]] = {}
    for example, value in zip(examples, per_pair):
        regimes.setdefault(example.regime, []).append(value)
    return {
        "best_train_mode": best_train,
        "best_eval_mode": best_eval,
        "final_eval_mode": pooled,
        "curve": rows,
        "per_pair_final_eval": [
            {"pair": _pair_label(e), "regime": e.regime,
             "n_supported": int(e.support.sum()), "eval_mode": v}
            for e, v in zip(examples, per_pair)
        ],
        "per_regime_final_eval_mean": {
            regime: sum(values) / len(values) for regime, values in sorted(regimes.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/phase5.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--every", type=int, default=50)
    args = parser.parse_args(argv)

    repo = Path.cwd().resolve()
    out = args.out.expanduser().resolve()
    if out == repo or repo in out.parents:
        raise SystemExit(f"--out must lie outside the repository {repo}, so the "
                         "checkout stays clean")

    from lot.analysis_config import load_analysis_config
    from lot.evaluate import git_commit
    from lot.phase5 import (
        load_convention_record, load_phase5_config, materialize_example,
        phase5_mean_vector, predictor_config_from, scene_image_hw,
    )
    from lot.phase5_folds import frozen_folds
    from lot.phase5_modes import SceneStore, resolve_device, select_tiny_subset
    from lot.train import training_config_from

    cfg = load_phase5_config(args.config)
    analysis = load_analysis_config(Path(cfg.analysis_config))
    convention = load_convention_record(cfg)
    center = phase5_mean_vector(cfg)
    train_cfg = training_config_from(cfg.training)
    folds = frozen_folds()
    model_cfg = predictor_config_from(cfg, scene_image_hw(cfg, folds[0].train[0]))
    level = cfg.primary_alignment_level
    tiny = cfg.tiny_overfit
    device = resolve_device(args.device)

    with SceneStore(cfg, analysis, convention) as store:
        plans = select_tiny_subset(
            cfg, analysis, store, folds[0], level, int(tiny["n_pairs"]),
            tuple(tiny["regimes"]),
        )
        examples = [materialize_example(cfg, store.get(p.scene), p, center) for p in plans]

    result = diagnose(
        examples, model_cfg, train_cfg, tuple(model_cfg.target_grid),
        int(tiny["max_steps"]), int(tiny["seed"]), device, every=args.every,
    )
    result.update({
        "commit": git_commit(),
        "config_digest": cfg.digest(),
        "threshold": float(tiny["threshold_centered_cosine"]),
        "max_steps": int(tiny["max_steps"]),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dropout": model_cfg.dropout,
        "learning_rate": train_cfg.learning_rate,
    })
    out.mkdir(parents=True, exist_ok=True)
    (out / "overfit_diagnostic.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8"
    )

    print(f"commit {result['commit']}  device {device}  gpu {result['gpu']}")
    print(f"threshold {result['threshold']}  dropout {result['dropout']}  "
          f"learning rate {result['learning_rate']}  steps {result['max_steps']}")
    print(f"{'step':>6}{'train_mode':>12}{'eval_mode':>12}{'loss':>10}")
    for row in result["curve"]:
        print(f"{row['step']:>6}{row['train_mode']:>12.4f}{row['eval_mode']:>12.4f}{row['loss']:>10.4f}")
    print(f"best train mode {result['best_train_mode']:.4f}  best eval mode "
          f"{result['best_eval_mode']:.4f}  final eval mode {result['final_eval_mode']:.4f}")
    for item in result["per_pair_final_eval"]:
        print(f"  {item['regime']:<12}{item['eval_mode']:>8.4f}  n={item['n_supported']:<6}{item['pair']}")
    for regime, value in result["per_regime_final_eval_mean"].items():
        print(f"  regime {regime:<12} mean eval mode {value:.4f}")
    print(f"written to {out / 'overfit_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
