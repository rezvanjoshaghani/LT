"""Verify a Phase 5 gate receipt against the state that is about to run.

A receipt records that some earlier invocation passed. It is evidence about the
state that produced it and about nothing else. Checking only its `passed` field
would let a later commit, an edited configuration, a changed fold assignment, or
a different measurement identity inherit an older run's verdict, so a job could
train under code the gate never examined while pointing at a green receipt as
though it had.

Every field below therefore has to match the run asking to proceed. A mismatch
names the field, prints both values, and refuses. This module is also the single
place that answers "may this run proceed", so the shell launcher and the SLURM
worker ask the same question the same way rather than each implementing their
own weaker version.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# The identity a receipt must carry forward. Each is something that, if it
# changed, would mean the gate verified a different experiment.
BOUND_FIELDS = (
    "commit",
    "config_digest",
    "fold_digest",
    "measurement_digest",
)

# The resolved inputs the gate actually examined. A receipt that verified one
# feature cache or one accepted Phase 4 run is not evidence about another, and
# the configuration digest alone cannot see this: two runs can name the same
# paths while the bytes behind them differ. Checked when the receipt carries
# them, which every integration-gate receipt does.
BOUND_ARTIFACT_FIELDS = ("renders_root", "cache_features", "phase4_dir", "phase4_eval")


def receipt_identity(report: dict[str, Any]) -> dict[str, Any]:
    """Pull the identity out of a gate report, wherever the writer put it.

    The integration gate records it as step 1's evidence; a simpler receipt may
    record it at the top level. Both are read, so a receipt format can gain a
    field without this check silently starting to ignore it.
    """
    identity: dict[str, Any] = {
        key: report[key] for key in BOUND_FIELDS if key in report
    }
    for step in report.get("steps", []):
        evidence = step.get("evidence") or {}
        for key in BOUND_FIELDS:
            if key in evidence and key not in identity:
                identity[key] = evidence[key]
    return identity


def current_identity(config_path: Path) -> dict[str, Any]:
    from .analysis_config import load_analysis_config
    from .evaluate import git_commit
    from .phase5 import load_phase5_config
    from .phase5_folds import fold_digest, frozen_folds

    cfg = load_phase5_config(config_path)
    analysis = load_analysis_config(Path(cfg.analysis_config))
    return {
        "commit": git_commit(),
        "config_digest": cfg.digest(),
        "fold_digest": fold_digest(frozen_folds()),
        "measurement_digest": analysis.measurement_digest(),
    }


def verify(receipt_path: Path, config_path: Path, label: str) -> list[str]:
    """Return a list of human-readable problems. Empty means the receipt stands."""
    problems: list[str] = []
    try:
        report = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"{label}: receipt {receipt_path} could not be read: {error}"]

    if not report.get("passed"):
        failure = report.get("failure") or {}
        detail = (
            f" (failing step {failure.get('step')}: {failure.get('message')})"
            if failure else ""
        )
        problems.append(f"{label}: the recorded verdict is FAIL{detail}")

    recorded = receipt_identity(report)
    current = current_identity(config_path)
    for field in BOUND_FIELDS:
        if field not in recorded:
            problems.append(
                f"{label}: receipt carries no {field}, so it cannot be bound to "
                "this run; rerun the gate"
            )
            continue
        if recorded[field] != current[field]:
            problems.append(
                f"{label}: {field} moved since the receipt was written.\n"
                f"    receipt: {recorded[field]}\n"
                f"    current: {current[field]}\n"
                "    The gate verified a different state. Rerun it."
            )

    problems.extend(_artifact_problems(report, config_path, label))
    return problems


def _artifact_problems(
    report: dict[str, Any], config_path: Path, label: str
) -> list[str]:
    """Compare the inputs the gate resolved with the ones this run would read."""
    from .phase5 import load_phase5_config
    from .phase5_check import required_artifacts

    recorded = receipt_artifacts(report)
    if not recorded:
        return []
    cfg = load_phase5_config(config_path)
    current = {a.name: str(a.path.resolve()) for a in required_artifacts(cfg)
               if a.path.exists()}
    problems = []
    for name in BOUND_ARTIFACT_FIELDS:
        if name not in recorded:
            continue
        was = recorded[name].get("path") if isinstance(recorded[name], dict) else None
        now = current.get(name)
        if was is not None and now is not None and was != now:
            problems.append(
                f"{label}: the {name} the gate verified is not the one this run "
                f"would read.\n    receipt: {was}\n    current: {now}\n"
                "    Rerun the gate against the inputs you intend to use."
            )
    return problems


def receipt_artifacts(report: dict[str, Any]) -> dict[str, Any]:
    """The resolved-artifact block a gate receipt records at step 2."""
    for step in report.get("steps", []):
        evidence = step.get("evidence") or {}
        if "artifacts" in evidence:
            return evidence["artifacts"]
    return {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="verify a Phase 5 gate receipt")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5.yaml"))
    parser.add_argument("--label", default="gate")
    args = parser.parse_args(argv)

    problems = verify(args.receipt, args.config, args.label)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(
            f"{args.label}: refusing to proceed on a receipt that does not "
            "describe this run.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"{args.label}: PASS, bound to this commit and configuration")


if __name__ == "__main__":
    main()
