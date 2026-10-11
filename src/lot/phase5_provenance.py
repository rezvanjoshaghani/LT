"""Provenance for the Phase 5 reporting modes, reporting_rules.md decision 1.

The chain runs once, at one commit E:

    check -> overfit -> train -> controls -> lock -> evaluate

Its receipts bind that commit, so every step is licensed by receipts at E.
The tables, figures, and acceptance modes may run later, at a commit R. They
are licensed by the evaluated run's own provenance, never by receipts at R.
This module asks the question they ask before they read a row: was the run
evaluated at E licensed, complete, and coherent, and is the code at R still
the code that measured it?

require_evaluated_run answers it in six stages. A stage lists every problem
it finds, each named by its field. A stage with a problem stops the mode with
ProvenanceError, and the later stages do not run. Stages 2 and 3 are
reported together, because neither needs the other.

1. Completeness. The parquets under eval/{level} are exactly the evaluation
   scenes. No unfinished write is present. Each parquet carries its run
   record inside it, and the record names its own scene, the level, phase 5,
   and the fold that held the scene out, with one checkpoint per seed.
2. One identity. Every identity field is present in every record, then equal
   across them. Comparing only the values present would let a field that no
   record carries agree with itself. The commit is a full commit hash.
3. Binding. The run was measured under the configuration and the analysis
   that read it: the config, fold, measurement, and reporting digests, and
   the seeds.
4. Code ancestry. HEAD is clean, E is an ancestor of HEAD, and every path
   changed since E is allowed by its class, below. It is cheap, so it runs
   before the receipts are hashed again.
5. The receipt chain at E. The integration, overfit, and lock receipts are
   located by the sha256 the records name, among the current files and those
   a rerun moved aside. Each is verified against E's identity through
   lot.phase5_receipt.verify_against, never against HEAD's. The lock hashes
   every checkpoint, training record, and controls file it binds again.
6. The checkpoint chain. Each record names the checkpoints and training
   records the lock binds, and the lock binds the live files. The training
   records, controls entries, and overfit verdict a record embeds are the
   locked and licensed files' own content. Training and the controls ran at
   E under the receipts the evaluation names.

Path classes. Every repository path is in one class, the first that names it
in this order:

- frozen at E: decision 1's frozen class. Any difference between E and R
  stops.
- measurement: every file that decides what is measured. Any difference
  stops.
- reporting: the reporting code. A difference must be named, with its
  reason, in validation/evidence/phase5/post_evaluation_changes.md.
- neutral: tests, documents, and evidence. A difference is allowed.

A path in none of them stops, so a new kind of file is refused until someone
decides its class. The classes are read from this module's copy at E and at
HEAD as well as from the running copy, and a path takes the strictest class
any of them gives it. So an edit after E to this module cannot license a
change that E's classes forbid.

post_evaluation_changes.md names each change as a list item that starts with
the path in backticks, followed by its reason:

    - `src/lot/phase5_report.py`: a column label named the wrong method.

A non-primary level is reported beside the primary result, never in its
place. require_primary_chain asks whether its run is the primary run's in all
but its level: the same commit E, digests, seeds, and gate receipts, which
decision 4's one chain gives every level.

The output helpers write what the reporting modes produce: a run record for
every output, parquets that carry it inside them, and directories published
by one rename that never overwrite an earlier output.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import fnmatch
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .phase5_check import environment_identity, sha256_file, utc_timestamp
from .phase5_check import supersede as move_aside

# ---------------------------------------------------------------------------
# Path classes
# ---------------------------------------------------------------------------
#
# Patterns are POSIX paths relative to the repository root, matched with
# fnmatch, where * also crosses a slash. Each list is a literal tuple of
# strings, because code ancestry reads the lists of commit E from E's copy of
# this file without running it.

# reporting_rules.md decision 1: the estimands, the paired bootstrap, the
# outcome and wording code, and the reporting rules themselves.
FROZEN_AT_E: tuple[str, ...] = (
    "src/lot/phase5_estimands.py",
    "src/lot/paired_bootstrap.py",
    "src/lot/phase5_outcomes.py",
    "validation/evidence/phase5/reporting_rules.md",
)

# Every file that decides what is measured: the evaluation path and every
# module it reads, the configurations, and the frozen documents.
MEASUREMENT_PATHS: tuple[str, ...] = (
    "src/lot/phase5.py",
    "src/lot/phase5_modes.py",
    "src/lot/phase5_score.py",
    "src/lot/phase5_reference.py",
    "src/lot/phase5_folds.py",
    "src/lot/context_lift.py",
    "src/lot/sample_identity.py",
    "src/lot/predictors.py",
    "src/lot/train.py",
    "src/lot/encoders.py",
    "src/lot/geometry.py",
    "src/lot/visibility.py",
    "src/lot/transport.py",
    "src/lot/evaluate.py",
    "src/lot/phase4.py",
    "src/lot/datasets.py",
    "src/lot/correspondence.py",
    "src/lot/analysis_config.py",
    "src/lot/render_replica.py",
    "configs/*.yaml",
    "PROTOCOL.md",
    "AMENDMENTS.md",
    "VALIDATION.md",
    "FREEZE.md",
)

# The reporting code. A change after E is allowed when it is named, with its
# reason, in post_evaluation_changes.md.
REPORTING_PATHS: tuple[str, ...] = (
    "src/lot/phase5_report.py",
    "src/lot/phase5_figures.py",
    "src/lot/phase5_acceptance.py",
    "src/lot/phase5_provenance.py",
    "scripts/run_phase5.sh",
    "scripts/phase5*readout*.py",
)

# Tests, documents, and evidence. The frozen and measurement documents and
# reporting_rules.md are named above, and the first class that names a path
# wins.
NEUTRAL_PATHS: tuple[str, ...] = (
    "tests/*",
    "validation/evidence/*",
    "*.md",
)

FROZEN = "frozen_at_e"
MEASUREMENT = "measurement"
REPORTING = "reporting"
NEUTRAL = "neutral"
OTHER = "other"
# Each class and the list that defines it, in the order a path is matched.
CLASS_LISTS = {
    FROZEN: "FROZEN_AT_E",
    MEASUREMENT: "MEASUREMENT_PATHS",
    REPORTING: "REPORTING_PATHS",
    NEUTRAL: "NEUTRAL_PATHS",
}
# From the class that holds a path least to the one that holds it most. A
# path takes the strictest class any copy of the lists gives it.
STRICTNESS = (NEUTRAL, REPORTING, OTHER, MEASUREMENT, FROZEN)
# The problem field that names each class of path that may not change freely.
CLASS_FIELDS = {
    FROZEN: "frozen_at_e",
    MEASUREMENT: "measurement_paths",
    REPORTING: "reporting_paths",
    OTHER: "other_paths",
}

# Where this module and the listing of post-evaluation changes live.
PROVENANCE_MODULE = "src/lot/phase5_provenance.py"
POST_EVALUATION_CHANGES = "validation/evidence/phase5/post_evaluation_changes.md"
REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# The evaluated run
# ---------------------------------------------------------------------------

# Present in every run record, then equal across them.
IDENTITY_FIELDS = (
    "commit", "config_digest", "fold_digest", "measurement_digest",
    "mean_vector_digest", "seeds", "phase4_commit", "eval_version",
    "analysis_reporting_digest", "licence",
)
# What each run record embeds, lot.phase5_modes.embedded_evidence.
EMBEDDED_FIELDS = ("training_record_contents", "controls", "overfit_verdict")
# Where lot.phase5.require_receipts reads the two gate receipts, by stem.
INTEGRATION_RECEIPT_STEM = "integration_gate"
OVERFIT_RECEIPT_STEM = "tiny_overfit"

# The layout of output_run_record.
OUTPUT_RECORD_VERSION = 1

FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LISTED = re.compile(r"^\s*[-*+]\s+`(?P<path>[^`]+)`(?P<rest>.*)$")
_CONTEXT = "the evaluated run cannot license a report"


class ProvenanceError(ValueError):
    """A run, a receipt, or the code cannot license a report.

    problems holds (field, message) pairs. fields lists each field once, in
    the order its first problem was found.
    """

    def __init__(self, problems: Sequence[tuple[str, str]], context: str = _CONTEXT):
        self.problems = tuple((str(field), str(message)) for field, message in problems)
        super().__init__(
            context + ":\n"
            + "\n".join(f"  - {field}: {message}" for field, message in self.problems)
        )

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(field for field, _ in self.problems))


@dataclasses.dataclass(frozen=True)
class EvaluatedScene:
    """One evaluation parquet: its scene, path, sha256, and embedded run record."""

    scene: str
    path: Path
    sha256: str
    record: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class RunIdentity:
    """The evaluated run a report is licensed by, after every check passed.

    commit is E, where the chain ran. training_commit is the commit every
    training record names, which decision 1 makes E. report_commit is HEAD,
    R, when the code was checked, and None when it was not. The digests, the
    seeds, phase4_commit, eval_version, and licence are the run records' one
    identity. bootstrap is the frozen interval settings of the analysis that
    read it. receipts names each licensing receipt by file stem: the file it
    was found in and its sha256. checkpoints and training_records name each
    (fold, seed) the run evaluated by its fold_seed_key and the sha256 the
    lock binds. reporting_changes maps each reporting path changed since E to
    the reason post_evaluation_changes.md gives. inputs maps every file the
    run was read from to its sha256, by its path under the run directory.
    records holds each scene's run record as it was verified, with the
    training records, controls entries, and overfit verdict it embeds, so the
    tables never read a record twice. It is left out of repr and comparison.
    """

    level: str
    run_dir: str
    commit: str
    training_commit: str
    report_commit: str | None
    code_checked: bool
    config_digest: str
    fold_digest: str
    measurement_digest: str
    mean_vector_digest: str
    analysis_reporting_digest: str
    eval_version: int
    phase4_commit: str
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    scenes: tuple[str, ...]
    bootstrap: dict[str, Any]
    licence: dict[str, str]
    receipts: dict[str, dict[str, str]]
    eval_paths: dict[str, str]
    eval_sha256: dict[str, str]
    checkpoints: dict[str, str]
    training_records: dict[str, str]
    controls_sha256: str
    reporting_changes: dict[str, str]
    inputs: dict[str, str]
    records: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _canonical(value: Any) -> str:
    """One text per value, so nested records compare whole, NaN included."""
    return json.dumps(value, sort_keys=True, default=str)


def _short(value: Any, limit: int = 120) -> str:
    text = _canonical(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _normal(path: str) -> str:
    """A repository path as git prints it: POSIX, relative, no leading ./."""
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def _read_bound_json(path: Path) -> tuple[str, Any]:
    """A JSON file's sha256 and content, from one read of its bytes."""
    import hashlib

    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest(), json.loads(data.decode("utf-8"))


def _relative(path: Path, run_dir: Path) -> str:
    """A path under the run directory as a POSIX name, or whole if outside it."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(run_dir).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _embedded_record(path: Path) -> dict[str, Any] | None:
    """The run record inside a parquet, never the sidecar beside it.

    CLAUDE.md: every figure is regenerable from the evaluation parquets alone.
    A record that lives only in the sidecar would not survive that rule.
    """
    import pyarrow.parquet as pq

    from .evaluate import RUN_METADATA_KEY

    raw = (pq.read_schema(path).metadata or {}).get(RUN_METADATA_KEY)
    return None if raw is None else json.loads(raw.decode("utf-8"))


def _receipt_stems(level: str) -> dict[str, str]:
    """Each licensing receipt's file stem, by kind, for an evaluation at level."""
    from .phase5_modes import checkpoint_lock_path
    from .phase5_receipt import KIND_INTEGRATION, KIND_LOCK, KIND_OVERFIT

    return {
        KIND_INTEGRATION: INTEGRATION_RECEIPT_STEM,
        KIND_OVERFIT: OVERFIT_RECEIPT_STEM,
        KIND_LOCK: checkpoint_lock_path(Path("."), level).stem,
    }


# ---------------------------------------------------------------------------
# Classes of paths
# ---------------------------------------------------------------------------

def module_classes() -> dict[str, tuple[str, ...]]:
    """The path classes of the running copy of this module."""
    return {FROZEN: FROZEN_AT_E, MEASUREMENT: MEASUREMENT_PATHS,
            REPORTING: REPORTING_PATHS, NEUTRAL: NEUTRAL_PATHS}


def classify_path(path: str, classes: Mapping[str, Sequence[str]] | None = None) -> str:
    """The class of one repository path: the first class whose list matches it.

    path is POSIX and relative to the repository root, as git prints it.
    classes defaults to this module's own. Returns frozen_at_e, measurement,
    reporting, neutral, or other.
    """
    classes = module_classes() if classes is None else classes
    path = _normal(path)
    for name in CLASS_LISTS:
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in classes[name]):
            return name
    return OTHER


def _strictest(names: Sequence[str]) -> str:
    return max(names, key=STRICTNESS.index)


def classes_from_source(source: str) -> dict[str, tuple[str, ...]]:
    """The path classes a copy of this module defines, read without running it.

    Each list is the last top-level assignment to its name, the value Python
    would leave. A list that is absent, or assigned anything other than a
    literal tuple of strings, raises ValueError.
    """
    wanted = {name: cls for cls, name in CLASS_LISTS.items()}
    found: dict[str, tuple[str, ...]] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if not (isinstance(target, ast.Name) and target.id in wanted):
                continue
            try:
                literal = ast.literal_eval(value)
            except ValueError as error:
                raise ValueError(
                    f"{target.id} is not assigned a literal tuple of paths: {error}"
                ) from error
            if not (isinstance(literal, tuple)
                    and all(isinstance(item, str) for item in literal)):
                raise ValueError(f"{target.id} is not a tuple of path strings")
            found[target.id] = literal
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise ValueError(f"no assignment to {missing}")
    return {cls: found[name] for name, cls in wanted.items()}


def post_evaluation_listing(text: str) -> dict[str, str]:
    """The reporting changes post_evaluation_changes.md names, each with its reason.

    A change is a list item that starts with its path in backticks. Its reason
    is the rest of the line, after an optional colon or dash. A path named
    twice keeps both reasons. Every other line names nothing.
    """
    listed: dict[str, str] = {}
    for line in text.splitlines():
        match = _LISTED.match(line)
        if match is None:
            continue
        path = _normal(match["path"])
        reason = match["rest"].strip().lstrip(":-\u2013").strip()
        if path in listed:
            listed[path] = "; ".join(part for part in (listed[path], reason) if part)
        else:
            listed[path] = reason
    return listed


# ---------------------------------------------------------------------------
# Code ancestry
# ---------------------------------------------------------------------------

_ANCESTRY = "the code at HEAD cannot report the run evaluated at E"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    """git in root. Run with cwd rather than -C, which git 1.8.3 lacks."""
    try:
        return subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, timeout=300
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ProvenanceError(
            [("commit", f"git could not run in {root}: {error}")], _ANCESTRY
        ) from error


def _out(done: subprocess.CompletedProcess) -> str:
    return done.stdout.decode("utf-8", errors="replace")


def _err(done: subprocess.CompletedProcess) -> str:
    return done.stderr.decode("utf-8", errors="replace").strip()


def _show(root: Path, revision: str, path: str) -> str | None:
    """A file's text at a revision, or None when the revision does not hold it."""
    if _git(root, "cat-file", "-e", f"{revision}:{path}").returncode != 0:
        return None
    done = _git(root, "show", f"{revision}:{path}")
    if done.returncode != 0:
        raise ProvenanceError(
            [("commit", f"git show {revision}:{path} failed: {_err(done)}")], _ANCESTRY
        )
    return _out(done)


def _classes_at(
    root: Path, revision: str, name: str
) -> tuple[dict[str, tuple[str, ...]] | None, str | None, bool]:
    """This module's classes at a revision, read from git without running them.

    Returns the classes, or None with the reason they cannot be read, and
    whether the revision holds the file at all. name says which revision it
    is in a message.
    """
    source = _show(root, revision, PROVENANCE_MODULE)
    if source is None:
        return None, f"{PROVENANCE_MODULE} is absent at {name}", False
    try:
        return classes_from_source(source), None, True
    except (SyntaxError, ValueError) as error:
        return None, (f"the path classes in {PROVENANCE_MODULE} at {name} cannot be "
                      f"read: {error}"), True


def verify_code_ancestry(commit: str, repo_root: Path | None = None) -> tuple[str, dict[str, str]]:
    """Whether the code at HEAD may report a run evaluated at commit, E.

    HEAD must be clean. E must exist and be an ancestor of HEAD. Every path
    git diff names between E and HEAD, renames split into both paths, must be
    allowed by its class: none frozen at E, none measurement, none other, and
    every reporting path named with its reason in post_evaluation_changes.md
    at HEAD. Every frozen class member must exist at E, because a member that
    does not froze nothing. E's copy of this module must exist, because
    decision 1 commits the reporting code before the chain runs.

    repo_root defaults to the repository this module is in. Returns HEAD's
    commit and the reporting changes, each path with its reason. Every
    refusal raises ProvenanceError naming worktree, commit, classification,
    or the class of the paths.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    problems: list[tuple[str, str]] = []

    status = _git(root, "status", "--porcelain")
    if status.returncode != 0:
        raise ProvenanceError(
            [("worktree", f"git status failed in {root}: {_err(status)}")], _ANCESTRY
        )
    dirty = _out(status).splitlines()
    if dirty:
        problems.append((
            "worktree",
            f"HEAD has uncommitted changes, so no commit describes the code that "
            f"reports the run: {dirty[:5]}",
        ))
    head_done = _git(root, "rev-parse", "HEAD")
    if head_done.returncode != 0:
        raise ProvenanceError(
            problems + [("commit", f"HEAD cannot be read in {root}: {_err(head_done)}")],
            _ANCESTRY,
        )
    head = _out(head_done).strip()

    if not (isinstance(commit, str) and FULL_COMMIT.fullmatch(commit)):
        raise ProvenanceError(
            problems + [("commit", f"{commit!r} is not a full commit hash")], _ANCESTRY
        )
    if _git(root, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        raise ProvenanceError(problems + [(
            "commit",
            f"E {commit} does not exist in {root}; fetch it before reporting the run",
        )], _ANCESTRY)
    ancestor = _git(root, "merge-base", "--is-ancestor", commit, "HEAD")
    if ancestor.returncode != 0:
        detail = (
            f"E {commit} is not an ancestor of HEAD {head}, so the reporting code "
            "did not grow from the code that measured the run"
            if ancestor.returncode == 1
            else f"git merge-base failed: {_err(ancestor)}"
        )
        raise ProvenanceError(problems + [("commit", detail)], _ANCESTRY)

    # The classes of the running copy, of E's copy, and of HEAD's copy. In a
    # clean checkout HEAD's copy is the running one. E's may differ, and it
    # still binds.
    class_sets = [module_classes()]
    at_e, why, _ = _classes_at(root, commit, f"E {commit}")
    if at_e is None:
        problems.append(("classification", (
            f"{why}. Decision 1 commits the reporting code, this module with it, "
            "before the chain runs, so the classes of E must be readable"
        )))
    else:
        class_sets.append(at_e)
    current = [module_classes()]
    at_head, why, present = _classes_at(root, "HEAD", "HEAD")
    if at_head is not None:
        class_sets.append(at_head)
        current.append(at_head)
    elif present:
        problems.append(("classification", why))

    # Every frozen class member, as the code at HEAD names it, exists at E.
    frozen = sorted({path for classes in current for path in classes[FROZEN]})
    absent = [
        path for path in frozen
        if not any(ch in path for ch in "*?[")
        and _git(root, "cat-file", "-e", f"{commit}:{path}").returncode != 0
    ]
    if absent:
        problems.append((
            "frozen_at_e",
            f"{absent} are absent at E {commit}. Decision 1 freezes the outcome and "
            "wording code at E, and a class member absent at E froze nothing. Name "
            "the files that hold that code in FROZEN_AT_E",
        ))

    diff = _git(root, "diff", "--name-only", "--no-renames", "-z", commit, "HEAD", "--")
    if diff.returncode != 0:
        raise ProvenanceError(problems + [
            ("commit", f"git diff {commit} HEAD failed: {_err(diff)}")
        ], _ANCESTRY)
    changed = sorted({_normal(path) for path in _out(diff).split("\0") if path.strip()})
    by_class: dict[str, list[str]] = {}
    for path in changed:
        name = _strictest([classify_path(path, classes) for classes in class_sets])
        by_class.setdefault(name, []).append(path)

    if by_class.get(FROZEN):
        problems.append((CLASS_FIELDS[FROZEN], (
            f"changed since E: {by_class[FROZEN]}. Decision 1 freezes this class at E, "
            "and any difference fails"
        )))
    if by_class.get(MEASUREMENT):
        problems.append((CLASS_FIELDS[MEASUREMENT], (
            f"changed since E: {by_class[MEASUREMENT]}. No file that decides what is "
            "measured may change between E and the report"
        )))
    if by_class.get(OTHER):
        problems.append((CLASS_FIELDS[OTHER], (
            f"changed since E, and in no class: {by_class[OTHER]}. After E only "
            "neutral paths and named reporting paths may change"
        )))
    reporting = by_class.get(REPORTING, [])
    listed = post_evaluation_listing(_show(root, "HEAD", POST_EVALUATION_CHANGES) or "")
    unlisted = [path for path in reporting if path not in listed]
    unreasoned = [path for path in reporting if path in listed and not listed[path]]
    if unlisted:
        problems.append((CLASS_FIELDS[REPORTING], (
            f"changed since E and not named in {POST_EVALUATION_CHANGES}: {unlisted}"
        )))
    if unreasoned:
        problems.append((CLASS_FIELDS[REPORTING], (
            f"named in {POST_EVALUATION_CHANGES} without a reason: {unreasoned}"
        )))
    if problems:
        raise ProvenanceError(problems, _ANCESTRY)
    return head, {path: listed[path] for path in reporting}


# ---------------------------------------------------------------------------
# 1. Completeness
# ---------------------------------------------------------------------------

def read_evaluated_run(
    run_dir: Path,
    level: str,
    expected_scenes: Sequence[str],
    folds: Sequence[Any] | None = None,
) -> dict[str, EvaluatedScene]:
    """Every evaluation parquet of a level, with its run record, or a refusal.

    The stems of run_dir/eval/{level}/*.parquet must be expected_scenes
    exactly, and no unfinished write may be present. Each parquet must carry
    its run record inside it. The record must name phase 5, its own stem as
    its scene, this level, and the fold of folds that held the scene out, and
    key its checkpoints by its seeds. folds defaults to the frozen folds.

    Returns one EvaluatedScene per scene, in the order of expected_scenes,
    with each file's sha256. Raises ProvenanceError listing every problem.
    Rows are not read here: the reporting code reads one scene at a time,
    through read_scene_rows.
    """
    from .phase5_folds import fold_of_test_scene, frozen_folds

    folds = list(folds) if folds is not None else frozen_folds()
    expected = list(expected_scenes)
    if not expected:
        raise ValueError("no scenes are expected, so no run can be shown complete")
    if len(set(expected)) != len(expected):
        raise ValueError(f"the expected scenes repeat: {expected}")
    eval_dir = Path(run_dir) / "eval" / level
    if not eval_dir.is_dir():
        raise ProvenanceError([("scenes", f"no evaluation directory at {eval_dir}")])

    problems: list[tuple[str, str]] = []
    names = sorted(entry.name for entry in eval_dir.iterdir())
    partial = [name for name in names if ".partial" in name]
    if partial:
        problems.append((
            "partial",
            f"{eval_dir} holds unfinished writes {partial}. An interrupted write is "
            "not an evaluation, and the run is not complete",
        ))
    files = {path.stem: path for path in sorted(eval_dir.glob("*.parquet")) if path.is_file()}
    missing = [scene for scene in expected if scene not in files]
    extra = sorted(set(files) - set(expected))
    if missing:
        problems.append(("scenes", f"missing scene(s) {missing}: no parquet under {eval_dir}"))
    if extra:
        problems.append((
            "scenes",
            f"unexpected scene(s) {extra} under {eval_dir}. A directory holding more "
            "than the evaluation scenes is a different population",
        ))

    scenes: dict[str, EvaluatedScene] = {}
    for scene in expected:
        path = files.get(scene)
        if path is None:
            continue
        try:
            record = _embedded_record(path)
        except Exception as error:  # noqa: BLE001
            # Any failure to read the record refuses the scene. It is reported
            # with the others rather than raised alone.
            problems.append((
                "run_record",
                f"{path.name}: its run record cannot be read: {type(error).__name__}: {error}",
            ))
            continue
        if not isinstance(record, dict):
            problems.append((
                "run_record",
                f"{path.name} carries no run record inside it. A record only beside "
                "it would not survive the rule that reports come from the parquets alone",
            ))
            continue
        if record.get("phase") != 5:
            problems.append(("phase", f"{path.name}: its run record names phase "
                                      f"{record.get('phase')!r}, not 5"))
        if record.get("scene") != scene:
            problems.append(("scene", f"{path.name}: its run record names scene "
                                      f"{record.get('scene')!r}, not its stem {scene!r}"))
        if record.get("level") != level:
            problems.append(("level", f"{path.name}: its run record names level "
                                      f"{record.get('level')!r}, but it is under {level!r}"))
        try:
            held_out_by = fold_of_test_scene(scene, folds).index
        except ValueError:
            problems.append(("fold", f"{scene} is not a test scene of any fold"))
        else:
            if record.get("fold") != held_out_by:
                problems.append(("fold", (
                    f"{path.name}: its run record names fold {record.get('fold')!r}, but "
                    f"fold {held_out_by} held {scene} out"
                )))
        seeds = record.get("seeds")
        checkpoints = record.get("checkpoints")
        if seeds is None:
            problems.append(("seeds", f"{path.name}: absent from its run record"))
        elif not (isinstance(seeds, list) and seeds
                  and all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)
                  and len(set(seeds)) == len(seeds)):
            problems.append(("seeds", f"{path.name}: {_short(seeds)} is not a list of "
                                      "distinct seeds"))
        elif not (isinstance(checkpoints, dict)
                  and set(checkpoints) == {str(seed) for seed in seeds}):
            keys = sorted(checkpoints) if isinstance(checkpoints, dict) else checkpoints
            problems.append(("checkpoints", (
                f"{path.name}: its checkpoints are keyed {_short(keys)}, but its seeds "
                f"are {seeds}"
            )))
        scenes[scene] = EvaluatedScene(scene, path, sha256_file(path), record)
    if problems:
        raise ProvenanceError(problems)
    return scenes


# ---------------------------------------------------------------------------
# 2 and 3. One identity, bound to the configuration and analysis reading it
# ---------------------------------------------------------------------------

def _identity(scenes: Mapping[str, EvaluatedScene], level: str
              ) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    from .phase5_modes import PHASE5_EVAL_VERSION

    problems: list[tuple[str, str]] = []
    values: dict[str, Any] = {}
    for field in IDENTITY_FIELDS:
        absent = [scene for scene, entry in scenes.items() if entry.record.get(field) is None]
        if absent:
            problems.append((field, (
                f"absent from the run record of {len(absent)} scene(s), first "
                f"{absent[0]}. A record without it cannot be shown to belong with the "
                "others"
            )))
            continue
        groups: dict[str, list[str]] = {}
        for scene, entry in scenes.items():
            groups.setdefault(_canonical(entry.record[field]), []).append(scene)
        if len(groups) > 1:
            shown = "; ".join(
                f"{_short(json.loads(value))} in {members}" for value, members in groups.items()
            )
            problems.append((field, (
                f"takes {len(groups)} values across the scenes, so they are not one "
                f"run: {shown}"
            )))
            continue
        values[field] = next(iter(scenes.values())).record[field]

    commit = values.get("commit")
    if commit is not None and not (isinstance(commit, str) and FULL_COMMIT.fullmatch(commit)):
        if isinstance(commit, str) and commit.endswith("-dirty"):
            problems.append(("commit", (
                f"{commit!r}: the run was evaluated from a worktree with uncommitted "
                "changes, so no commit describes the code that measured it"
            )))
        elif commit == "unknown":
            problems.append(("commit", "unknown: the run was evaluated where its "
                                       "commit could not be read"))
        else:
            problems.append(("commit", f"{_short(commit)} is not a full commit hash"))
    version = values.get("eval_version")
    if version is not None and version != PHASE5_EVAL_VERSION:
        problems.append(("eval_version", (
            f"the run records have layout {version!r}, and this reader reads layout "
            f"{PHASE5_EVAL_VERSION}"
        )))
    licence = values.get("licence")
    if licence is not None:
        want = sorted(_receipt_stems(level).values())
        if not (isinstance(licence, dict) and sorted(licence) == want
                and all(isinstance(v, str) and SHA256.fullmatch(v) for v in licence.values())):
            problems.append(("licence", (
                f"names {_short(licence)}. An evaluation is licensed by exactly the "
                f"receipts {want}, each by sha256"
            )))
    return values, problems


def _binding(values: Mapping[str, Any], cfg: Any, analysis: Any,
             folds: Sequence[Any]) -> list[tuple[str, str]]:
    from .phase5_folds import fold_digest
    from .train import training_config_from

    expected = {
        "config_digest": (cfg.digest(), "configuration"),
        "fold_digest": (fold_digest(folds), "fold assignment"),
        "measurement_digest": (analysis.measurement_digest(), "analysis configuration"),
        "analysis_reporting_digest": (analysis.reporting_digest(), "analysis configuration"),
        "seeds": ([int(seed) for seed in training_config_from(cfg.training).seeds],
                  "configuration"),
    }
    return [
        (field, (
            f"the run records name {_short(values[field])}, but the {source} reading "
            f"them has {_short(want)}"
        ))
        for field, (want, source) in expected.items()
        if field in values and values[field] != want
    ]


# ---------------------------------------------------------------------------
# 5. The receipt chain at E
# ---------------------------------------------------------------------------

def locate_receipt(evidence_dir: Path, stem: str, sha256: str) -> Path:
    """The one receipt file under evidence_dir whose content has this sha256.

    The candidates are {stem}.json and each {stem}.superseded.N.json that
    lot.phase5_check.supersede leaves when a rerun moves a receipt aside. The
    sha256 decides. No match, or more than one, raises ProvenanceError naming
    receipts.{stem}.
    """
    evidence_dir = Path(evidence_dir)
    pattern = re.compile(rf"^{re.escape(stem)}(\.superseded\.\d+)?\.json$")
    candidates = sorted(
        path for path in evidence_dir.iterdir() if path.is_file() and pattern.match(path.name)
    ) if evidence_dir.is_dir() else []
    matches = [path for path in candidates if sha256_file(path) == sha256]
    field = f"receipts.{stem}"
    if not matches:
        raise ProvenanceError([(field, (
            f"no file among {[path.name for path in candidates]} under {evidence_dir} "
            f"has the licensed sha256 {sha256}"
        ))], "the receipt that licensed the run cannot be found")
    if len(matches) > 1:
        raise ProvenanceError([(field, (
            f"{len(matches)} files under {evidence_dir} have the licensed sha256 "
            f"{sha256}: {[path.name for path in matches]}. Which one licensed the run "
            "is ambiguous"
        ))], "the receipt that licensed the run cannot be found")
    return matches[0]


def _receipt_chain(cfg: Any, config_path: Path, level: str,
                   values: Mapping[str, Any]) -> dict[str, Path]:
    """Locate each licensing receipt and verify it against E's identity."""
    from .phase5_receipt import (
        BOUND_FIELDS,
        KIND_INTEGRATION,
        KIND_LOCK,
        KIND_OVERFIT,
        verify_against,
    )

    evidence = Path(cfg.evidence_dir)
    stems = _receipt_stems(level)
    licence = values["licence"]
    located: dict[str, Path] = {}
    problems: list[tuple[str, str]] = []
    for kind, stem in stems.items():
        try:
            located[kind] = locate_receipt(evidence, stem, licence[stem])
        except ProvenanceError as error:
            problems.extend(error.problems)
    if problems:
        raise ProvenanceError(problems)

    at_e = {field: values[field] for field in BOUND_FIELDS}
    checks = (
        (KIND_INTEGRATION, {}),
        (KIND_OVERFIT, {"gate_receipt": located[KIND_INTEGRATION]}),
        (KIND_LOCK, {"gate_receipt": located[KIND_INTEGRATION],
                     "overfit_receipt": located[KIND_OVERFIT], "level": level}),
    )
    for kind, bindings in checks:
        stem = stems[kind]
        for problem in verify_against(located[kind], config_path, f"the {stem} receipt",
                                      at_e, kind=kind, reference="evaluated run",
                                      **bindings):
            problems.append((f"receipts.{stem}", problem))
    if problems:
        raise ProvenanceError(problems)
    return {stems[kind]: path for kind, path in located.items()}


# ---------------------------------------------------------------------------
# 6. The checkpoint chain and the embedded evidence
# ---------------------------------------------------------------------------

def _checkpoint_chain(cfg: Any, level: str, scenes: Mapping[str, EvaluatedScene],
                      values: Mapping[str, Any], receipts: Mapping[str, Path]
                      ) -> dict[str, Any]:
    """Records bind the locked files, and embed the locked and licensed content."""
    from .phase5_modes import (
        OVERFIT_VERDICT_FIELDS,
        checkpoint_lock_path,
        checkpoint_path,
        controls_path,
        fold_seed_key,
        training_record_path,
    )

    run_dir, evidence = Path(cfg.run_dir), Path(cfg.evidence_dir)
    commit, digest = values["commit"], values["config_digest"]
    seeds, licence = list(values["seeds"]), values["licence"]
    gates = {stem: licence[stem] for stem in (INTEGRATION_RECEIPT_STEM, OVERFIT_RECEIPT_STEM)}
    lock = json.loads(receipts[checkpoint_lock_path(evidence, level).stem]
                      .read_text(encoding="utf-8"))

    def locked(group: str) -> dict[str, Any]:
        entries = lock.get(group) if isinstance(lock.get(group), dict) else {}
        return {key: (entry or {}).get("sha256") for key, entry in entries.items()}

    locked_checkpoints, locked_records = locked("checkpoints"), locked("training_records")
    locked_controls = (lock.get("controls") or {}).get("sha256")
    folds = sorted({entry.record["fold"] for entry in scenes.values()})
    keys = [fold_seed_key(fold, seed) for fold in folds for seed in seeds]
    problems: list[tuple[str, str]] = []

    # The live training records, read once each. The lock bound their bytes.
    live: dict[str, Any] = {}
    record_paths: dict[str, Path] = {}
    for fold in folds:
        for seed in seeds:
            key = fold_seed_key(fold, seed)
            path = training_record_path(run_dir, level, fold, seed)
            record_paths[key] = path
            try:
                sha, record = _read_bound_json(path)
            except (OSError, ValueError) as error:
                problems.append(("training_records", f"{key}: the training record at "
                                                     f"{path} cannot be read: {error}"))
                continue
            if sha != locked_records.get(key):
                problems.append(("training_records", (
                    f"{key}: the live training record has sha256 {sha}, but the lock "
                    f"binds {locked_records.get(key)}"
                )))
            live[key] = record
            record = record if isinstance(record, dict) else {}
            identity = {"level": level, "fold": fold, "seed": seed, "commit": commit,
                        "config_digest": digest,
                        "checkpoint_sha256": locked_checkpoints.get(key),
                        "licence": gates}
            for field, want in identity.items():
                if record.get(field) != want:
                    problems.append(("training_record_contents", (
                        f"{key}: the locked training record names {field} "
                        f"{_short(record.get(field))}, not {_short(want)}"
                    )))

    # The live controls file. The lock bound its bytes too.
    controls_file = controls_path(evidence, level)
    controls: Any = {}
    controls_sha = None
    try:
        controls_sha, controls = _read_bound_json(controls_file)
    except (OSError, ValueError) as error:
        problems.append(("controls", f"the controls file {controls_file} cannot be "
                                     f"read: {error}"))
    if controls_sha is not None and controls_sha != locked_controls:
        problems.append(("controls", (
            f"the live controls file has sha256 {controls_sha}, but the lock binds "
            f"{locked_controls}"
        )))
    controls = controls if isinstance(controls, dict) else {}
    for field, want in {"level": level, "commit": commit, "config_digest": digest,
                        "licence": gates, "missing": []}.items():
        if controls.get(field) != want:
            problems.append(("controls", (
                f"the locked controls file names {field} {_short(controls.get(field))}, "
                f"not {_short(want)}"
            )))
    control_results = controls.get("results") if isinstance(controls.get("results"), dict) else {}
    control_checkpoints = (controls.get("checkpoints")
                           if isinstance(controls.get("checkpoints"), dict) else {})
    for key in keys:
        if control_checkpoints.get(key) != locked_checkpoints.get(key):
            problems.append(("controls", (
                f"{key}: the controls ran on checkpoint {control_checkpoints.get(key)}, "
                f"but the lock binds {locked_checkpoints.get(key)}"
            )))

    # The overfit receipt the evaluation names, located by its sha256.
    overfit: Any = {}
    try:
        overfit_sha, overfit = _read_bound_json(receipts[OVERFIT_RECEIPT_STEM])
    except (OSError, ValueError) as error:
        overfit_sha = None
        problems.append(("overfit_verdict", (
            f"the overfit receipt {receipts[OVERFIT_RECEIPT_STEM]} cannot be read: {error}"
        )))
    if overfit_sha is not None and overfit_sha != licence[OVERFIT_RECEIPT_STEM]:
        problems.append(("overfit_verdict", (
            f"{receipts[OVERFIT_RECEIPT_STEM]} changed while it was read: its sha256 "
            f"is {overfit_sha}, and the licence names {licence[OVERFIT_RECEIPT_STEM]}"
        )))
    overfit = overfit if isinstance(overfit, dict) else {}
    verdict = {"sha256": licence[OVERFIT_RECEIPT_STEM],
               **{field: overfit.get(field) for field in OVERFIT_VERDICT_FIELDS}}

    for scene, entry in scenes.items():
        record = entry.record
        fold = record["fold"]
        fold_keys = [fold_seed_key(fold, seed) for seed in seeds]
        named_records = record.get("training_records")
        named_records = named_records if isinstance(named_records, dict) else {}
        for seed, key in zip(seeds, fold_keys):
            checkpoint = (record.get("checkpoints") or {}).get(str(seed))
            if checkpoint != locked_checkpoints.get(key):
                problems.append(("checkpoints", (
                    f"{scene}: seed {seed} was evaluated with checkpoint {checkpoint}, "
                    f"but the lock binds {locked_checkpoints.get(key)} for {key}"
                )))
            named = named_records.get(str(seed))
            named = named if isinstance(named, dict) else {}
            if named.get("sha256") != locked_records.get(key):
                problems.append(("training_records", (
                    f"{scene}: seed {seed} names training record {named.get('sha256')}, "
                    f"but the lock binds {locked_records.get(key)} for {key}"
                )))
            for field, want in (("commit", commit), ("config_digest", digest)):
                if named.get(field) != want:
                    problems.append(("training_records", (
                        f"{scene}: seed {seed}'s training record names {field} "
                        f"{_short(named.get(field))}, not the evaluated run's {_short(want)}"
                    )))

        for field in EMBEDDED_FIELDS:
            if record.get(field) is None:
                problems.append((field, f"{scene}: absent from its run record"))
        contents = record.get("training_record_contents")
        if isinstance(contents, dict):
            if set(contents) != {str(seed) for seed in seeds}:
                problems.append(("training_record_contents", (
                    f"{scene}: it embeds training records for seeds {sorted(contents)}, "
                    f"not {seeds}"
                )))
            for seed, key in zip(seeds, fold_keys):
                if key in live and _canonical(contents.get(str(seed))) != _canonical(live[key]):
                    problems.append(("training_record_contents", (
                        f"{scene}: seed {seed}'s embedded training record is not the "
                        f"locked record {key}"
                    )))
        elif contents is not None:
            problems.append(("training_record_contents", f"{scene}: not a mapping by seed"))
        block = record.get("controls")
        expected_block = {
            "sha256": controls_sha,
            "written_utc": controls.get("written_utc"),
            "results": {key: control_results.get(key) for key in fold_keys},
            "checkpoints": {key: control_checkpoints.get(key) for key in fold_keys},
        }
        if block is not None and _canonical(block) != _canonical(expected_block):
            problems.append(("controls", (
                f"{scene}: its embedded controls are not the locked controls file's "
                f"entries for fold {fold}"
            )))
        embedded_verdict = record.get("overfit_verdict")
        if embedded_verdict is not None and _canonical(embedded_verdict) != _canonical(verdict):
            problems.append(("overfit_verdict", (
                f"{scene}: its embedded overfit verdict is not the licensed receipt's"
            )))
    if problems:
        raise ProvenanceError(problems)

    return {
        "checkpoints": {key: locked_checkpoints[key] for key in keys},
        "training_records": {key: locked_records[key] for key in keys},
        "checkpoint_paths": {
            fold_seed_key(fold, seed): checkpoint_path(run_dir, level, fold, seed)
            for fold in folds for seed in seeds
        },
        "record_paths": record_paths,
        "controls_sha256": controls_sha,
        "controls_path": controls_file,
    }


# ---------------------------------------------------------------------------
# The question the reporting modes ask
# ---------------------------------------------------------------------------

def require_evaluated_run(
    cfg: Any,
    analysis: Any,
    config_path: Path,
    level: str,
    *,
    expected_scenes: Sequence[str] | None = None,
    folds: Sequence[Any] | None = None,
    repo_root: Path | None = None,
    check_code: bool = True,
) -> RunIdentity:
    """The evaluated run at level, if it licenses a report. Otherwise a stop.

    cfg must be the configuration at config_path: the receipts are verified
    against that file, so both must describe one configuration. level must be
    declared in it. expected_scenes defaults to every test scene of folds, and
    folds to the frozen folds. repo_root is the repository whose history holds
    E, by default the one this module is in.

    The six stages of the module docstring run in order. Each refusal raises
    ProvenanceError, naming every field that is wrong. check_code=False skips
    code ancestry. It exists only for tests, whose runs were not evaluated at
    a commit of a repository: a reporting mode never passes it.
    """
    from .phase5 import load_phase5_config
    from .phase5_folds import frozen_folds
    from .phase5_modes import evaluation_scenes

    declared = (cfg.primary_alignment_level, *cfg.sensitivity_alignment_levels,
                *cfg.diagnostic_alignment_levels)
    if level not in declared:
        raise ValueError(f"level {level!r} is not declared in the configuration: {declared}")
    if load_phase5_config(config_path).digest() != cfg.digest():
        raise ValueError(
            f"the configuration object is not the config at {config_path}. The receipts "
            "are verified against that file, so both must be one configuration"
        )
    folds = list(folds) if folds is not None else frozen_folds()
    expected = (list(expected_scenes) if expected_scenes is not None
                else evaluation_scenes(folds))
    run_dir = Path(cfg.run_dir)
    context = f"the evaluated run at level {level!r} cannot license a report"

    try:
        scenes = read_evaluated_run(run_dir, level, expected, folds)
        values, problems = _identity(scenes, level)
        problems += _binding(values, cfg, analysis, folds)
        if problems:
            raise ProvenanceError(problems)
        commit = values["commit"]
        head, changes = (verify_code_ancestry(commit, repo_root) if check_code
                         else (None, {}))
        receipts = _receipt_chain(cfg, config_path, level, values)
        chain = _checkpoint_chain(cfg, level, scenes, values, receipts)
    except ProvenanceError as error:
        raise ProvenanceError(error.problems, context) from None

    inputs: dict[str, str] = {}
    for entry in scenes.values():
        inputs[_relative(entry.path, run_dir)] = entry.sha256
    for key, sha in chain["checkpoints"].items():
        inputs[_relative(chain["checkpoint_paths"][key], run_dir)] = sha
    for key, sha in chain["training_records"].items():
        inputs[_relative(chain["record_paths"][key], run_dir)] = sha
    inputs[_relative(chain["controls_path"], run_dir)] = chain["controls_sha256"]
    for stem, path in receipts.items():
        inputs[_relative(path, run_dir)] = values["licence"][stem]

    return RunIdentity(
        level=level,
        run_dir=str(run_dir),
        commit=commit,
        training_commit=commit,
        report_commit=head,
        code_checked=bool(check_code),
        config_digest=values["config_digest"],
        fold_digest=values["fold_digest"],
        measurement_digest=values["measurement_digest"],
        mean_vector_digest=values["mean_vector_digest"],
        analysis_reporting_digest=values["analysis_reporting_digest"],
        eval_version=values["eval_version"],
        phase4_commit=values["phase4_commit"],
        seeds=tuple(int(seed) for seed in values["seeds"]),
        folds=tuple(sorted({entry.record["fold"] for entry in scenes.values()})),
        scenes=tuple(scenes),
        bootstrap={
            "primary_unit": "scene",
            "resamples": analysis.bootstrap_resamples,
            "seed": analysis.bootstrap_seed,
            "confidence": analysis.bootstrap_confidence,
        },
        licence=dict(values["licence"]),
        receipts={stem: {"path": str(path), "sha256": values["licence"][stem]}
                  for stem, path in receipts.items()},
        eval_paths={scene: str(entry.path) for scene, entry in scenes.items()},
        eval_sha256={scene: entry.sha256 for scene, entry in scenes.items()},
        checkpoints=dict(chain["checkpoints"]),
        training_records=dict(chain["training_records"]),
        controls_sha256=chain["controls_sha256"],
        reporting_changes=dict(changes),
        inputs=dict(sorted(inputs.items())),
        records={scene: entry.record for scene, entry in scenes.items()},
    )


def read_scene_rows(identity: RunIdentity, scene: str) -> list[dict[str, Any]]:
    """One scene's evaluation rows, read only from the bytes that were verified.

    Read through lot.evaluate.read_rows, which uses pyarrow's to_pylist and
    never pandas, so every value is a plain Python int or float. The file is
    hashed before and after the read, and both must be the sha256 the run
    identity recorded.
    """
    from .evaluate import read_rows

    if scene not in identity.eval_paths:
        raise ValueError(f"{scene!r} is not a scene of the run at level {identity.level!r}")
    path, want = Path(identity.eval_paths[scene]), identity.eval_sha256[scene]
    before = sha256_file(path)
    rows = read_rows(path) if before == want else None
    after = sha256_file(path) if rows is not None else before
    if before != want or after != want:
        raise ProvenanceError([("eval_sha256", (
            f"{scene}: {path} has sha256 {after}, but the run was verified with {want}"
        ))], "the rows are not the rows that were verified")
    return rows


# ---------------------------------------------------------------------------
# A non-primary level, beside the primary result
# ---------------------------------------------------------------------------

# The receipts every level shares. reporting_rules.md decision 4 runs the
# overfit gate once, at the primary level, so its receipt and the integration
# receipt it binds license every level. Each level has its own lock.
SHARED_RECEIPT_STEMS = (INTEGRATION_RECEIPT_STEM, OVERFIT_RECEIPT_STEM)


def require_primary_chain(primary_record: Mapping[str, Any], identity: RunIdentity) -> None:
    """A non-primary level ran under the primary level's chain, or a refusal.

    reporting_rules.md decision 4 trains the affine level under the same chain,
    and runs the overfit gate once, at the primary level. Findings item 13
    compares the levels. So a level is reported beside the primary result only
    when its run is the primary run's in all but its level, its models, and its
    lock. primary_record is the run record of the primary level's published
    tables. identity is the level's own RunIdentity, from
    require_evaluated_run.

    The level must name the primary run's commit E and training commit, its
    configuration, fold, measurement, centering, and reporting digests, its
    record layout and Phase 4 commit, its seeds, folds, scenes, and bootstrap
    settings, and the same integration and overfit receipts by sha256. Every
    receipt binds the commit it was stamped at, so a level run at a later
    commit ran under gates run again, and is refused even when no measurement
    file changed. A level whose code was checked is never reported beside
    primary tables built without that check. Raises ProvenanceError naming
    every field that differs.
    """
    values = {
        "evaluation_commit": identity.commit,
        "training_commit": identity.training_commit,
        "config_digest": identity.config_digest,
        "fold_digest": identity.fold_digest,
        "measurement_digest": identity.measurement_digest,
        "mean_vector_digest": identity.mean_vector_digest,
        "analysis_reporting_digest": identity.analysis_reporting_digest,
        "eval_version": identity.eval_version,
        "phase4_commit": identity.phase4_commit,
        "seeds": list(identity.seeds),
        "folds": list(identity.folds),
        "scenes": list(identity.scenes),
        "bootstrap": dict(identity.bootstrap),
    }
    problems = [
        (field, (f"the primary level's tables name {_short(primary_record.get(field))}, and "
                 f"level {identity.level!r} {_short(want)}"))
        for field, want in values.items() if primary_record.get(field) != want
    ]
    licence = primary_record.get("licence")
    licence = licence if isinstance(licence, Mapping) else {}
    for stem in SHARED_RECEIPT_STEMS:
        if licence.get(stem) != identity.licence.get(stem):
            problems.append((f"licence.{stem}", (
                f"the primary level ran under the {stem} receipt {licence.get(stem)}, and "
                f"level {identity.level!r} under {identity.licence.get(stem)}. One chain "
                "runs each gate once"
            )))
    if identity.code_checked and primary_record.get("code_checked") is not True:
        problems.append(("code_checked", (
            "the primary level's tables were built without checking the code that reported "
            f"them, and level {identity.level!r} was reported with its code checked"
        )))
    if problems:
        raise ProvenanceError(
            problems, f"level {identity.level!r} was not run under the primary level's chain")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def output_run_record(
    identity: RunIdentity,
    kind: str,
    inputs: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The run record every table, JSON output, figure manifest, and verdict carries.

    It names the output's kind and the evaluated run: E, the training commit,
    R when the code was checked, the run's digests, seeds, folds, scenes, the
    bootstrap settings, the licence, and the receipts. inputs maps every file
    read to its sha256: the run's own files, and those the caller names, such
    as the tables a figure was drawn from. reporting_changes says what changed
    in the reporting code since E, and why. created_utc and environment say
    when and where the output was made. extra adds the caller's own fields.

    An input named with two different hashes, or an extra field that would
    replace a field of the record, raises ValueError.
    """
    if not (isinstance(kind, str) and kind):
        raise ValueError(f"an output record needs a kind, not {kind!r}")
    named = dict(identity.inputs)
    for name, sha in (inputs or {}).items():
        if name in named and named[name] != sha:
            raise ValueError(
                f"the input {name!r} is named with two hashes, {named[name]} and {sha}"
            )
        named[name] = sha
    record = {
        "kind": kind,
        "record_version": OUTPUT_RECORD_VERSION,
        "phase": 5,
        "level": identity.level,
        "evaluation_commit": identity.commit,
        "training_commit": identity.training_commit,
        "report_commit": identity.report_commit,
        "code_checked": identity.code_checked,
        "config_digest": identity.config_digest,
        "fold_digest": identity.fold_digest,
        "measurement_digest": identity.measurement_digest,
        "mean_vector_digest": identity.mean_vector_digest,
        "analysis_reporting_digest": identity.analysis_reporting_digest,
        "eval_version": identity.eval_version,
        "phase4_commit": identity.phase4_commit,
        "seeds": list(identity.seeds),
        "folds": list(identity.folds),
        "scenes": list(identity.scenes),
        "bootstrap": dict(identity.bootstrap),
        "licence": dict(identity.licence),
        "receipts": {stem: dict(entry) for stem, entry in identity.receipts.items()},
        "inputs": dict(sorted(named.items())),
        "reporting_changes": dict(identity.reporting_changes),
        "created_utc": utc_timestamp(),
        "environment": environment_identity(),
    }
    clash = sorted(set(record) & set(extra or {}))
    if clash:
        raise ValueError(f"extra fields {clash} would replace fields of the run record")
    record.update(extra or {})
    return record


def write_parquet_with_record(
    path: Path, rows: Sequence[Mapping[str, Any]], record: Mapping[str, Any]
) -> None:
    """Write rows to a parquet that carries record inside it, atomically, once.

    The record goes under lot.evaluate.RUN_METADATA_KEY, as every evaluation
    parquet carries its own, so lot.evaluate.read_run_metadata reads it back.
    The table is written to a uniquely named temporary file beside path and
    renamed into place, so a reader sees no file or the whole file. An
    existing path is refused, and a failed write leaves nothing behind.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .evaluate import RUN_METADATA_KEY

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; outputs are written once")
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows])
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}),
        RUN_METADATA_KEY: json.dumps(dict(record), sort_keys=True).encode("utf-8"),
    })
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.partial")
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


# The name of a reporting build's staging directory, beside its final output:
# {final}.partial.<32 lowercase hex digits>. lot.phase5_acceptance skips these
# directories, and only these, when it searches for a hidden evaluation.
STAGING_NAME = re.compile(r"^(?P<final>.+)\.partial\.[0-9a-f]{32}$")


def new_staging_directory(final: Path) -> Path:
    """A fresh, empty directory beside final, to build final's contents in."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(f"{final.name}.partial.{uuid.uuid4().hex}")
    staging.mkdir()
    return staging


def publish_directory(staging: Path, final: Path, supersede: bool = False) -> dict[str, Any]:
    """Publish a finished staging directory as final, by one rename.

    staging must be a directory beside final, so the rename stays on one file
    system and is atomic. An existing final is refused unless supersede is
    given. Then it is first moved aside to a numbered sibling, through
    lot.phase5_check.supersede, so nothing is ever deleted. Returns where the
    output was published and where the earlier one went, or None.
    """
    staging, final = Path(staging), Path(final)
    if not staging.is_dir():
        raise FileNotFoundError(f"no staging directory at {staging}")
    if staging.resolve().parent != final.resolve().parent:
        raise ValueError(
            f"the staging directory {staging} must share a parent with {final}, so "
            "publishing is one rename on one file system"
        )
    archived = None
    if final.exists():
        if not supersede:
            raise FileExistsError(
                f"{final} exists; outputs are written once. Supersede it to rebuild, "
                "which moves it aside and deletes nothing"
            )
        archived = move_aside(final)
    os.replace(staging, final)
    return {"published": str(final), "superseded": archived}


@dataclasses.dataclass
class StagedOutput:
    """A directory being built. Write into path. outcome is set once published."""

    path: Path
    final: Path
    outcome: dict[str, Any] | None = None


def _discard_staging(staging: Path, final: Path) -> None:
    """Remove a failed staging directory, and only a staging directory of final."""
    match = STAGING_NAME.match(staging.name)
    if match is None or match["final"] != final.name or staging.parent != final.parent:
        raise ValueError(f"{staging} is not a staging directory of {final}")
    shutil.rmtree(staging, ignore_errors=True)


@contextlib.contextmanager
def staged_output(final: Path, *, supersede: bool = False) -> Iterator[StagedOutput]:
    """Build final in a staging directory and publish it by one rename.

    An existing final is refused before any work, unless supersede is given.
    When the block raises, or publishing fails, the staging directory is
    removed and final is left as it was. When it ends, the staging directory
    is published through publish_directory and its outcome is recorded.
    """
    final = Path(final)
    if final.exists() and not supersede:
        raise FileExistsError(
            f"{final} exists; outputs are written once. Supersede it to rebuild"
        )
    staged = StagedOutput(new_staging_directory(final), final)
    try:
        yield staged
        staged.outcome = publish_directory(staged.path, final, supersede=supersede)
    except BaseException:
        if staged.path.exists():
            _discard_staging(staged.path, final)
        raise
