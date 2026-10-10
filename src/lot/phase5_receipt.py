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
from typing import Any, Callable

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
#
# The checkpoint lock of reporting_rules.md section 7 is the third kind. It is
# written after the controls and before evaluation, once per alignment level.
# Like the overfit receipt, it is bound to the identity and to the digest of the
# integration receipt. It is also bound to the digest of the overfit receipt.
# It carries one block of its own: the sha256 of every file evaluation reads
# before it touches a test scene. Those are each (fold, seed)'s checkpoint and
# training record, and the level's controls file. verify() hashes each of them
# again, at the path the current run would read it from.
KIND_INTEGRATION = "integration"
KIND_OVERFIT = "overfit"
KIND_LOCK = "lock"
RECEIPT_KINDS = (KIND_INTEGRATION, KIND_OVERFIT, KIND_LOCK)

# The keys under which a receipt names the receipts that licensed it, by
# sha256. An overfit or lock receipt names the integration receipt. A lock
# receipt also names the overfit receipt.
GATE_RECEIPT_DIGEST = "gate_receipt_sha256"
OVERFIT_RECEIPT_DIGEST = "overfit_receipt_sha256"

# What to rerun when a lock no longer verifies.
LOCK_RERUN = "the checkpoint lock"


def gate_steps(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The integration gate's step records: the mappings in the list under "steps".

    The overfit receipt uses the same key for its optimizer step count, an
    int, so the key holds gate steps only when it holds a list. Reading the
    count as a list raised TypeError, which stopped every mode after the
    overfit gate at its receipt check.
    """
    steps = report.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)]


def receipt_identity(report: dict[str, Any]) -> dict[str, Any]:
    """Pull the identity out of a gate report, wherever the writer put it.

    The integration gate records it as step 1's evidence; a simpler receipt may
    record it at the top level. Both are read, so a receipt format can gain a
    field without this check silently starting to ignore it.
    """
    identity: dict[str, Any] = {
        key: report[key] for key in BOUND_FIELDS if key in report
    }
    for step in gate_steps(report):
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
    overfit_receipt: Path | None = None,
    level: str | None = None,
) -> list[str]:
    """Return a list of human-readable problems. Empty means the receipt stands.

    Bound to HEAD: the receipt must carry the identity of the state about to
    run, as current_identity reads it from config_path and the checkout. Every
    mode of the chain asks this question. verify_against asks it of another
    identity, which is what the reporting modes need.

    kind names the receipt expected here; None reads it from the receipt itself.
    gate_receipt is the integration receipt an overfit or lock receipt must be
    bound to. overfit_receipt is the overfit receipt a lock receipt must be
    bound to. level is the alignment level a lock receipt must lock. The last
    two are read for a lock receipt only.
    """
    return _verify_receipt(
        receipt_path, config_path, label, lambda: current_identity(config_path),
        "current", kind, gate_receipt, overfit_receipt, level,
    )


def verify_against(
    receipt_path: Path,
    config_path: Path,
    label: str,
    expected: dict[str, Any],
    *,
    kind: str | None = None,
    gate_receipt: Path | None = None,
    overfit_receipt: Path | None = None,
    level: str | None = None,
    reference: str = "expected",
) -> list[str]:
    """verify, bound to a given identity instead of HEAD's.

    reporting_rules.md decision 1. The chain runs at one commit E, and the
    reporting modes may run later. They pass the identity of the evaluated
    run, commit E with its config, fold, and measurement digests, so each
    receipt is checked against the run it licensed and not against the
    reporting commit. Everything else is verify's: the verdict, the kind, the
    bindings between receipts by sha256, and the live inputs and files each
    kind binds by content.

    expected must name every field of BOUND_FIELDS; one left out would leave
    that field unbound, so it raises ValueError. reference names the expected
    identity in a problem, where verify says "current".
    """
    missing = [field for field in BOUND_FIELDS if field not in expected]
    if missing:
        raise ValueError(
            f"the identity to verify {label} against names no {missing}; every "
            f"field of {BOUND_FIELDS} must be bound"
        )
    identity = {field: expected[field] for field in BOUND_FIELDS}
    return _verify_receipt(
        receipt_path, config_path, label, lambda: identity,
        reference, kind, gate_receipt, overfit_receipt, level,
    )


def _verify_receipt(
    receipt_path: Path,
    config_path: Path,
    label: str,
    identity: Callable[[], dict[str, Any]],
    reference: str,
    kind: str | None,
    gate_receipt: Path | None,
    overfit_receipt: Path | None,
    level: str | None,
) -> list[str]:
    """The core of verify and verify_against.

    identity returns the identity the receipt must carry. It is called only
    after the receipt has been read, so a receipt that cannot be read is
    reported as such whatever the identity would have required. reference
    names that identity in a problem.
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
    current = identity()
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
                f"    {reference}: {current[field]}\n"
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
    elif found_kind == KIND_OVERFIT:
        problems.extend(_gate_binding_problems(report, label, gate_receipt))
    else:
        problems.extend(
            _gate_binding_problems(report, label, gate_receipt, rerun=LOCK_RERUN)
        )
        problems.extend(_overfit_binding_problems(report, label, overfit_receipt))
        problems.extend(_lock_file_problems(report, config_path, label, level))
    return problems


def _gate_binding_problems(
    report: dict[str, Any],
    label: str,
    gate_receipt: Path | None,
    rerun: str = "the overfit gate",
) -> list[str]:
    """An overfit receipt is bound to the integration receipt that licensed it.

    It carries no artifact or scene block of its own, because the overfit gate
    resolves no inputs; what it must prove is that it ran after, and under, a
    specific integration gate. Recording that gate receipt's digest does exactly
    that: if the gate is rerun, its receipt changes and this binding fails, which
    is the intended behaviour.

    A lock receipt is bound the same way. rerun names what must run again when
    the binding fails.
    """
    return _digest_binding_problems(
        report, label, GATE_RECEIPT_DIGEST, gate_receipt,
        source="integration", source_gate="gate", rerun=rerun,
    )


def _overfit_binding_problems(
    report: dict[str, Any], label: str, overfit_receipt: Path | None
) -> list[str]:
    """A lock receipt is also bound to the overfit receipt that licensed it.

    Training ran under that overfit receipt, and the lock binds training's
    checkpoints. The lock mode enforces this when it writes the lock: every
    training record and the controls file must name the integration and
    overfit receipts it binds as their licence. If the overfit gate is rerun,
    its receipt changes and this binding fails.
    """
    return _digest_binding_problems(
        report, label, OVERFIT_RECEIPT_DIGEST, overfit_receipt,
        source="overfit", source_gate="overfit gate", rerun=LOCK_RERUN,
    )


def _digest_binding_problems(
    report: dict[str, Any],
    label: str,
    field: str,
    source_receipt: Path | None,
    source: str,
    source_gate: str,
    rerun: str,
) -> list[str]:
    """One binding by digest: the receipt names, under field, the sha256 of the
    source receipt that licensed it, and that file must still hash to it.

    source names the source receipt's kind, and source_gate the step that
    writes it. rerun names what must run again when the binding fails.
    """
    from .phase5_check import sha256_file

    recorded = report.get(field)
    if not recorded:
        return [
            f"{label}: receipt does not record the {source} receipt it ran "
            f"under ({field}); rerun {rerun}"
        ]
    if source_receipt is None:
        return [
            f"{label}: no {source} receipt was supplied to bind against; this "
            "is a caller error, not a receipt defect"
        ]
    if not Path(source_receipt).exists():
        return [
            f"{label}: the {source} receipt at {source_receipt} is absent, so "
            f"this receipt's binding cannot be checked; rerun the {source_gate}"
        ]
    live = sha256_file(Path(source_receipt))
    if recorded != live:
        return [
            f"{label}: ran under a different {source} gate than the one "
            f"present.\n    receipt: {recorded}\n    current: {live}\n"
            f"    Rerun {rerun} under the current {source_gate}."
        ]
    return []


# How the lock's two keyed groups are named in a problem: by one file each.
LOCK_FILE_GROUPS = {"checkpoints": "checkpoint", "training_records": "training record"}


def _lock_file_problems(
    report: dict[str, Any], config_path: Path, label: str, level: str | None
) -> list[str]:
    """Re-hash every file a lock receipt binds, at the path this run reads it from.

    The files are located from the current configuration, the frozen folds,
    and the configured seeds, never from the paths the lock recorded. So the
    lock covers exactly the files evaluation will load, and a lock written for
    another level or another run directory binds nothing here. Each file must
    be present and still have the sha256 the lock recorded. A file this run
    reads and the lock does not name is refused, and so is an entry the lock
    names and this run does not read.

    The bytes are the binding, not the path, so a run directory moved whole
    still verifies. The recorded paths are kept as evidence for a reader.
    """
    from .phase5 import load_phase5_config
    from .phase5_check import sha256_file
    from .phase5_folds import frozen_folds
    from .phase5_modes import checkpoint_lock_files
    from .train import training_config_from

    if level is None:
        return [
            f"{label}: no alignment level was supplied to check the lock against; "
            "this is a caller error, not a receipt defect"
        ]
    if report.get("level") != level:
        return [
            f"{label}: the lock is for level {report.get('level')!r}, but this run "
            f"is at level {level!r}. A lock licenses only the level it locked."
        ]
    cfg = load_phase5_config(config_path)
    files = checkpoint_lock_files(
        cfg, level, frozen_folds(), training_config_from(cfg.training).seeds
    )

    problems: list[str] = []
    expected: list[tuple[str, Path, Any]] = []
    for group, name in LOCK_FILE_GROUPS.items():
        recorded = report.get(group)
        recorded = recorded if isinstance(recorded, dict) else {}
        unread = sorted(set(recorded) - set(files[group]))
        if unread:
            problems.append(
                f"{label}: the lock names {name}s this run does not read: {unread}"
            )
        expected += [
            (f"{name} {key}", path, recorded.get(key))
            for key, path in files[group].items()
        ]
    expected.append(("controls file", files["controls"], report.get("controls")))

    for name, path, entry in expected:
        if not isinstance(entry, dict) or not entry.get("sha256"):
            problems.append(
                f"{label}: the lock does not record the {name}, so it does not "
                f"bind a file evaluation reads; rerun {LOCK_RERUN}"
            )
        elif not Path(path).exists():
            problems.append(
                f"{label}: the locked {name} is absent at {path}; the lock "
                "describes a file this run cannot read"
            )
        else:
            live = sha256_file(Path(path))
            if live != entry["sha256"]:
                problems.append(
                    f"{label}: the {name} changed since the lock was written.\n"
                    f"    lock sha256:    {entry['sha256']}\n"
                    f"    current sha256: {live}\n"
                    "    Evaluation would read a file the lock never bound."
                )
    return problems


def stamp_receipt(
    report: dict[str, Any],
    config_path: Path,
    kind: str,
    gate_receipt: Path | None = None,
    overfit_receipt: Path | None = None,
) -> dict[str, Any]:
    """Add the identity a receipt must carry, so writers cannot forget it.

    One function stamps every receipt, so the shape verify() expects and the
    shape the writers produce cannot drift apart.

    The stamp also records when it was made, in UTC, as stamped_utc. That time
    is evidence for a reader and binds nothing. verify() does not read it, so a
    receipt written before the field existed still verifies.

    An overfit receipt records the sha256 of gate_receipt. A lock receipt
    records it too, and the sha256 of overfit_receipt. A receipt it must record
    and cannot read is refused.
    """
    from .phase5_check import sha256_file, utc_timestamp

    if kind not in RECEIPT_KINDS:
        raise ValueError(f"unknown receipt kind {kind!r}; use one of {RECEIPT_KINDS}")
    stamped = {
        **report, "kind": kind, **current_identity(config_path),
        "stamped_utc": utc_timestamp(),
    }
    if kind in (KIND_OVERFIT, KIND_LOCK):
        if gate_receipt is None or not Path(gate_receipt).exists():
            raise ValueError(
                f"the {kind} receipt must be bound to the integration receipt "
                "that licensed it, and that receipt is absent"
            )
        stamped[GATE_RECEIPT_DIGEST] = sha256_file(Path(gate_receipt))
    if kind == KIND_LOCK:
        if overfit_receipt is None or not Path(overfit_receipt).exists():
            raise ValueError(
                "the lock receipt must be bound to the overfit receipt that "
                "licensed it, and that receipt is absent"
            )
        stamped[OVERFIT_RECEIPT_DIGEST] = sha256_file(Path(overfit_receipt))
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
    for step in gate_steps(report):
        evidence = step.get("evidence") or {}
        if "artifacts" in evidence:
            return evidence["artifacts"]
    return {}


def receipt_scene_identities(report: dict[str, Any]) -> dict[str, Any]:
    """The per-scene identity block the gate records at step 2 and extends at 4."""
    found: dict[str, Any] = {}
    for step in gate_steps(report):
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
        help="the integration receipt an overfit or lock receipt must be bound to",
    )
    parser.add_argument(
        "--overfit-receipt", type=Path, default=None,
        help="the overfit receipt a lock receipt must be bound to",
    )
    parser.add_argument(
        "--level", default=None,
        help="the alignment level a lock receipt must lock",
    )
    args = parser.parse_args(argv)

    problems = verify(
        args.receipt, args.config, args.label,
        kind=args.kind, gate_receipt=args.gate_receipt,
        overfit_receipt=args.overfit_receipt, level=args.level,
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
