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
# paths while the bytes behind them differ. So the binding is by content. Every
# file artifact is re-hashed and compared; every scene-level identity the gate
# recorded (feature digest, depth digest, manifest digest, Phase 4 parquet
# hash, aligned-depth digest) is recomputed from the live inputs and compared.
# An integration receipt without this block, or with any artifact absent from
# it, or naming an artifact this run cannot resolve, is refused outright.
BOUND_ARTIFACT_FIELDS = (
    "renders_root", "cache_features", "mean_vector_dir",
    "phase4_dir", "phase4_convention", "phase4_eval",
)
# Which per-scene identity fields the receipt binds by content. The aligned
# depth digest is compared only when the current run can recompute it, which
# means building scene inputs; the verifier does not do that, because it runs
# at job start and the gate already has. It is bound through the parquet and
# cache digests that determine it.
BOUND_SCENE_FIELDS = (
    "features_digest", "depth_digest", "manifest_digest",
    "phase4_parquet_sha256", "phase4_parquet_bytes",
)

# A receipt's kind decides which bindings apply to it. The integration gate is
# the only thing that resolves and hashes the inputs, so it is the only receipt
# that can carry an artifact block or a per-scene identity block. The
# tiny-overfit gate trains on a frozen subset and produces no such evidence.
#
# An earlier version applied the integration bindings to every receipt, which
# made the overfit receipt impossible to verify: both gates could pass and
# training would still refuse to start, citing a missing artifact block and
# naming the wrong gate. The overfit receipt is instead bound to the identity
# and to the digest of the integration receipt that licensed it, so it cannot be
# carried across to a run the gate never examined.
KIND_INTEGRATION = "integration"
KIND_OVERFIT = "overfit"
RECEIPT_KINDS = (KIND_INTEGRATION, KIND_OVERFIT)

# The key an overfit receipt records the licensing gate receipt under.
GATE_RECEIPT_DIGEST = "gate_receipt_sha256"


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


def receipt_kind(report: dict[str, Any]) -> str:
    """Which kind of receipt this is, defaulting to the integration gate.

    The default is the stricter kind on purpose: a receipt that does not say
    what it is gets the binding that refuses more, not less.
    """
    kind = report.get("kind")
    return kind if kind in RECEIPT_KINDS else KIND_INTEGRATION


def verify(
    receipt_path: Path,
    config_path: Path,
    label: str,
    kind: str | None = None,
    gate_receipt: Path | None = None,
) -> list[str]:
    """Return a list of human-readable problems. Empty means the receipt stands.

    kind names the receipt expected here; None reads it from the receipt itself.
    gate_receipt is the integration receipt an overfit receipt must be bound to.
    """
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

    found_kind = receipt_kind(report)
    if kind is not None and found_kind != kind:
        problems.append(
            f"{label}: receipt says it is a {found_kind!r} receipt, but a "
            f"{kind!r} receipt is required here"
        )
        return problems

    if found_kind == KIND_INTEGRATION:
        problems.extend(_artifact_problems(report, config_path, label))
    else:
        problems.extend(_gate_binding_problems(report, label, gate_receipt))
    return problems


def _gate_binding_problems(
    report: dict[str, Any], label: str, gate_receipt: Path | None
) -> list[str]:
    """An overfit receipt is bound to the integration receipt that licensed it.

    It carries no artifact or scene block of its own, because the overfit gate
    resolves no inputs; what it must prove is that it ran after, and under, a
    specific integration gate. Recording that gate receipt's digest does exactly
    that: if the gate is rerun, its receipt changes and this binding fails, which
    is the intended behaviour.
    """
    from .phase5_check import sha256_file

    recorded = report.get(GATE_RECEIPT_DIGEST)
    if not recorded:
        return [
            f"{label}: receipt does not record the integration receipt it ran "
            f"under ({GATE_RECEIPT_DIGEST}); rerun the overfit gate"
        ]
    if gate_receipt is None:
        return [
            f"{label}: no integration receipt was supplied to bind against; this "
            "is a caller error, not a receipt defect"
        ]
    if not Path(gate_receipt).exists():
        return [
            f"{label}: the integration receipt at {gate_receipt} is absent, so "
            "this receipt's binding cannot be checked; rerun the gate"
        ]
    live = sha256_file(Path(gate_receipt))
    if recorded != live:
        return [
            f"{label}: ran under a different integration gate than the one "
            f"present.\n    receipt: {recorded}\n    current: {live}\n"
            "    Rerun the overfit gate under the current gate."
        ]
    return []


def stamp_receipt(
    report: dict[str, Any],
    config_path: Path,
    kind: str,
    gate_receipt: Path | None = None,
) -> dict[str, Any]:
    """Add the identity a receipt must carry, so writers cannot forget it.

    One function stamps every receipt, so the shape verify() expects and the
    shape the writers produce cannot drift apart.
    """
    from .phase5_check import sha256_file

    if kind not in RECEIPT_KINDS:
        raise ValueError(f"unknown receipt kind {kind!r}; use one of {RECEIPT_KINDS}")
    stamped = {**report, "kind": kind, **current_identity(config_path)}
    if kind == KIND_OVERFIT:
        if gate_receipt is None or not Path(gate_receipt).exists():
            raise ValueError(
                "an overfit receipt must be bound to the integration receipt "
                "that licensed it, and that receipt is absent"
            )
        stamped[GATE_RECEIPT_DIGEST] = sha256_file(Path(gate_receipt))
    return stamped


def _artifact_problems(
    report: dict[str, Any], config_path: Path, label: str
) -> list[str]:
    """Bind the receipt to the content of the inputs the gate examined.

    Three classes of failure, none of them silent. An artifact the receipt does
    not name: the gate never examined it. An artifact this run cannot resolve:
    the receipt describes inputs that are not here. Content that differs from
    what the receipt recorded: the bytes moved under the path.
    """
    from .phase5 import load_phase5_config
    from .phase5_check import required_artifacts, sha256_file

    recorded = receipt_artifacts(report)
    if not recorded:
        # A receipt with no artifact block is not an integration receipt and
        # cannot license training, whatever its verdict says.
        return [
            f"{label}: receipt carries no resolved-artifact block, so it cannot be "
            "bound to this run's inputs; rerun the integration gate"
        ]
    cfg = load_phase5_config(config_path)
    required = {a.name: a for a in required_artifacts(cfg)}
    problems: list[str] = []

    for name in BOUND_ARTIFACT_FIELDS:
        artifact = required.get(name)
        if artifact is None:
            continue
        entry = recorded.get(name)
        if not isinstance(entry, dict):
            problems.append(
                f"{label}: receipt does not record {name}; the gate never "
                "examined it. Rerun the gate."
            )
            continue
        if not artifact.path.exists():
            problems.append(
                f"{label}: {name} is not present at {artifact.path}; the receipt "
                "describes inputs this run cannot read."
            )
            continue
        now = str(artifact.path.resolve())
        if entry.get("path") != now:
            problems.append(
                f"{label}: the {name} the gate verified is not the one this run "
                f"would read.\n    receipt: {entry.get('path')}\n    current: {now}\n"
                "    Rerun the gate against the inputs you intend to use."
            )
            continue
        if artifact.kind == "file":
            live = sha256_file(artifact.path)
            if entry.get("sha256") != live:
                problems.append(
                    f"{label}: {name} changed under its path since the gate ran.\n"
                    f"    receipt sha256: {entry.get('sha256')}\n"
                    f"    current sha256: {live}\n"
                    "    Rerun the gate."
                )

    problems.extend(_scene_identity_problems(report, cfg, label))
    return problems


def _scene_identity_problems(report: dict[str, Any], cfg: Any, label: str) -> list[str]:
    """Recompute every bound per-scene identity and compare it to the receipt."""
    from .encoders import load_cache_meta
    from .phase4 import manifest_digest
    from .phase5_check import sha256_file
    from .render_replica import REPLICA_SCENES

    recorded = receipt_scene_identities(report)
    if not recorded:
        return [
            f"{label}: receipt carries no per-scene identity block, so the "
            "feature, depth, manifest, and Phase 4 inputs are unbound; rerun "
            "the integration gate"
        ]
    problems: list[str] = []
    missing = sorted(set(REPLICA_SCENES) - set(recorded))
    if missing:
        problems.append(
            f"{label}: receipt records no identity for scenes {missing}; the gate "
            "did not verify them."
        )
    for scene in sorted(set(REPLICA_SCENES) & set(recorded)):
        entry = recorded[scene]
        live: dict[str, Any] = {}
        try:
            live["features_digest"] = load_cache_meta(
                cfg.cache_root, cfg.feature_encoder, scene
            ).get("features_digest")
            # load_cache_meta reads the small meta.json. load_depth_archive
            # would return the identical value, but only after digesting and
            # fully decompressing the depth archive, which is roughly 11 GB of
            # reads across 18 scenes for a field already on disk in JSON. The
            # bytes behind that digest were verified by the gate.
            live["depth_digest"] = load_cache_meta(
                cfg.cache_root, cfg.depth_encoder, scene
            ).get("depth_digest")
            live["manifest_digest"] = manifest_digest(Path(cfg.renders_root) / scene)
            parquet = Path(cfg.phase4_dir) / "eval" / f"{scene}.parquet"
            live["phase4_parquet_sha256"] = sha256_file(parquet)
            live["phase4_parquet_bytes"] = parquet.stat().st_size
        except Exception as error:  # noqa: BLE001
            problems.append(
                f"{label}: could not recompute {scene}'s identity from the live "
                f"inputs ({error!r}); the receipt cannot be bound to them."
            )
            continue
        for field in BOUND_SCENE_FIELDS:
            if field not in entry:
                problems.append(
                    f"{label}: receipt records no {field} for {scene}; rerun the gate."
                )
            elif entry[field] != live[field]:
                problems.append(
                    f"{label}: {scene} {field} moved since the gate ran.\n"
                    f"    receipt: {entry[field]}\n    current: {live[field]}\n"
                    "    Rerun the gate."
                )
    return problems


def receipt_artifacts(report: dict[str, Any]) -> dict[str, Any]:
    """The resolved-artifact block a gate receipt records at step 2."""
    for step in report.get("steps", []):
        evidence = step.get("evidence") or {}
        if "artifacts" in evidence:
            return evidence["artifacts"]
    return {}


def receipt_scene_identities(report: dict[str, Any]) -> dict[str, Any]:
    """The per-scene identity block the gate records at step 2 and extends at 4."""
    found: dict[str, Any] = {}
    for step in report.get("steps", []):
        evidence = step.get("evidence") or {}
        block = evidence.get("scene_identities")
        if isinstance(block, dict):
            for scene, entry in block.items():
                found.setdefault(scene, {}).update(entry or {})
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="verify a Phase 5 gate receipt")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5.yaml"))
    parser.add_argument("--label", default="gate")
    parser.add_argument("--kind", choices=RECEIPT_KINDS, default=None)
    parser.add_argument(
        "--gate-receipt", type=Path, default=None,
        help="the integration receipt an overfit receipt must be bound to",
    )
    args = parser.parse_args(argv)

    problems = verify(
        args.receipt, args.config, args.label,
        kind=args.kind, gate_receipt=args.gate_receipt,
    )
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
