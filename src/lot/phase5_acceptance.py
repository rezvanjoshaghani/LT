"""Phase 5 acceptance: Stream AD, specification step 45, re-derived from artifacts.

reporting_rules.md section 8. Acceptance re-derives each of the specification's
eighteen conditions from the shipped artifacts. It adds three: the stable
validation curve of decision 5, the headline figures present and built from
the current tables, and the full suite green at the reporting commit. No
condition is asserted. Each is one function that reads the artifacts it rests
on and returns whether it holds, its notes, and its evidence.

The run was evaluated at commit E. Acceptance runs at a later commit R, or at
E itself. Like the tables and figures modes, it is licensed by the run's own
provenance, never by receipts at R. Unlike them, it does not stop at the first
problem. Every artifact is read tolerantly, and each condition re-derives its
own part of the provenance. So a defect fails the condition that owns it, and
the verdict lists every failure.

What each condition reads:

 1. The Phase 4 acceptance check, rerun on the Phase 4 run. The Phase 4
    parquets and commit every evaluation record names, against the gate's
    record and the live Phase 4 run.
 2. The pre-registration records at E, and every commit that touched them.
    Code ancestry: the frozen class unchanged since E, every other reporting
    change named with its reason. The tables and figures built by the code
    at R.
 3. The comparator's signature and references in context_lift.py at E, and
    the headline lift's depth in evaluate_scene at E. Gate steps 6 and 10.
 4. The predictor's forward signature, references, and imports at E, what
    model_inputs builds, and what evaluate_scene hands the model. Gate steps
    5 and 12.
 5. Gate step 17 over every scene, its count of rotation pairs against the
    pairs the evaluation read, gate step 10, and the gate's step order.
 6. One context depth and one set of cameras for both methods in
    evaluate_scene at E, model_inputs at E, and each scene's aligned-depth
    digest against gate step 4.
 7. The frozen folds, the evaluated scenes and the fold that held each out,
    every fold digest, gate step 11, each training record's roles, and the
    scenes training, selection, and the controls actually planned from.
 8. Gate step 14, the lock verified against the live files, the checkpoint
    chain, the licences, the order the artifacts were written in, the
    evaluation ledger, and any other evaluation record under the run
    directory, outside a reporting build's own staging directories.
 9. The overfit receipt, located by the sha256 the run names, verified, and
    read against the frozen gate settings and fold 0's training scenes.
10. One commit E and one configuration everywhere, the training-config
    digest in every record and checkpoint, the frozen architecture and its
    parameter count read off every checkpoint, and no measurement file
    changed since E.
11. Each checkpoint's step and validation score against its training record,
    and each record's history against the frozen cadence and patience.
12. The Context-Lift arm and support fixed across seeds, the region
    partition, the support's definition in evaluate_scene at E, gate step 7,
    and an independent recount on a deterministic sample of pairs.
13. The support count identical across the seeds' models, failures counted
    on it, and the support fixed before the model runs in evaluate_scene at
    E. Gate step 7's failure count.
14. The bootstrap settings every table names, a replicate count beside every
    interval, and a sample of headline cells recomputed from the parquets.
15. The frozen documents at E against FREEZE.md and pin.md, the measurement
    and mean-vector digests, both metrics over the same cells, and the
    Mean-Feature floor under raw cosine only.
16. Every interpreted-effect cell's disclosure, rerun from its persisted
    terms, the licensed wordings, no wording on a cell below support, the
    absence of the forbidden claim word, and decision 2's outcomes
    recomputed from the headline cells. The supported cells' wordings are
    printed by regime, quantity, and metric, never as a bare total.
17. Gate step 9, the splat transport's depth in phase5_reference.py at E,
    and every scene's splat reconciliation.
18. The headline table's columns and strings, its gap against its two
    methods, the formulation table's label, the reference table's rung, and
    where the reporting code at R names the target-lift reference.
19. Decision 5's stable curve, from the training records each run record
    embeds.
20. The published figures against the current tables, and the headline
    figures among them.
21. The full suite, run in a subprocess at R, with HEAD unmoved and the
    worktree clean.

Several conditions also read the suite's results for the test files that
cover them, from that one run. Tests are neutral to code ancestry, so those
results count only for the tests registered at E. Each such file, and the
test-tree modules SUITE_SUPPORT names, must exist at E and be byte-identical
at R, or be named with its reason in post_evaluation_changes.md. Their sha256
at E and at R go into each condition's evidence.

The verdict is written once to evidence/acceptance_{level}.json through
lot.phase5_check.write_once, which keeps any earlier verdict. It names
whether acceptance passed, the level and its role, every condition with its
notes and evidence, E and R, the scene-split hash, the training-config hash,
the config digest, the Context-Lift source at E, the checkpoint, evaluation,
table, and figure hashes, the folds, the seeds, and the receipts. When every
condition holds, it records the measured outcome, decision 2's
classification per regime and metric, read from the tables bound to the run,
each published row whole. A failed verdict withholds it and names the failed
conditions, because any failure is a stop. Only the primary level's verdict
is specification step 52's Rung 2 verdict. A verdict at a sensitivity or
diagnostic level is reported beside it and says so, reporting_rules.md
decision 4. The evaluation ledger's attempt files are rendered to
evaluation_ledger.jsonl first, by lot.phase5_modes.build_evaluation_ledger.
Any FAIL exits 1.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .phase5_check import FORBIDDEN_BATCH_FIELDS, environment_identity, sha256_file
from .phase5_check import utc_timestamp, write_once
from .phase5_estimands import (
    INTERPRETED_EFFECTS,
    PER_POINT,
    PRIMARY_FIELDS,
    REGION_CONTRASTS,
    SINGLE_PATH_BY_CONSTRUCTION,
    SPLAT_POOL,
    TL_REFERENCE,
    WORDING_NOT_ESTIMABLE,
    WORDING_ONLY_PATH_CLEAR,
    WORDING_ONLY_PATH_INCLUDES_ZERO,
    WORDING_OUTSIDE_BAND,
    WORDING_OUTSIDE_ON_COMMON_CELLS,
    WORDING_PATH_SENSITIVE,
    WORDING_SMALL_SIGN_CONSISTENT,
    WORDING_VETO,
    PathEstimate,
    evaluate_quantity,
    near_zero_disclosure,
)
from .phase5_figures import (
    FIGURES_DIR,
    FIGURE_TABLES,
    HEADLINE_FIGURES,
    RUN_FIELDS,
    check_published_figures,
    read_published_tables,
)
from .phase5_folds import (
    FROZEN_FOLD_DIGEST,
    assert_scene_separation,
    fold_digest,
    fold_of_test_scene,
    frozen_folds,
)
from .phase5_gate import ROTATION_COMPARISONS, ROTATION_READING
from .phase5_modes import (
    LEDGER_EVENT,
    LEDGER_FILE,
    LEDGER_UNFINISHED,
    OVERFIT_VERDICT_FIELDS,
    PHASE5_EVAL_VERSION,
    REGIONS,
    build_evaluation_ledger,
    checkpoint_lock_path,
    checkpoint_path,
    controls_path,
    evaluation_attempts,
    evaluation_scenes,
    fold_seed_key,
    training_record_path,
)
from .phase5_outcomes import (
    METRICS,
    OUTCOME_WORDING,
    POOLED_SCOPE,
    SCOPES,
    WORDING_BELOW_SUPPORT,
    OutcomeAnomaly,
    call_outcome,
    outcome_49,
    stable_validation_curve,
)
from .phase5_provenance import (
    FROZEN_AT_E,
    FULL_COMMIT,
    NEUTRAL,
    POST_EVALUATION_CHANGES,
    REPO_ROOT,
    STAGING_NAME,
    ProvenanceError,
    classify_path,
    locate_receipt,
    post_evaluation_listing,
    verify_code_ancestry,
)
from .phase5_receipt import (
    GATE_RECEIPT_DIGEST,
    KIND_INTEGRATION,
    KIND_LOCK,
    KIND_OVERFIT,
    OVERFIT_RECEIPT_DIGEST,
    gate_steps,
    receipt_identity,
    receipt_scene_identities,
    verify_against,
)
from .phase5_report import (
    FORMULATION_TABLE,
    HEADLINE_DEFINITION,
    L2_TABLE,
    MANIFEST_FILE,
    MEAN_FEATURE_NAMES,
    MEAN_FEATURE_NOT_APPLICABLE,
    MEAN_FEATURE_REPORTED,
    MEASURED_OUTCOME_TABLE,
    NEAR_ZERO_FILE,
    PRIMARY_EFFECTS,
    PRIMARY_TABLE,
    REFERENCE_TABLE,
    REGION_GAP,
    REGION_TABLE,
    SCOPE_AXIS,
    SPLAT_EFFECTS,
    SPLAT_TABLE,
    TABLE_LABELS,
    TABLES_DIR,
    level_role,
)
from .render_replica import REPLICA_SCENES

# ---------------------------------------------------------------------------
# The conditions and what they read
# ---------------------------------------------------------------------------

ACCEPTANCE_VERSION = 1
VERDICT_KIND = "phase5_acceptance"
# Where the evaluate mode writes its attempt files, under the evidence directory.
LEDGER_DIRECTORY = "evaluation_ledger"

CONDITION_TITLES = {
    1: "Phase 4 accepted",
    2: "design correction and reporting rules recorded before test outcomes",
    3: "Context-Lift Transport-Only uses no target depth",
    4: "Predict-with-Depth uses no target depth or target content",
    5: "Context-Lift passes the pure-rotation and synthetic geometry validation",
    6: "both headline methods receive identical context depth and camera information",
    7: "scene-level train and test separation is exact",
    8: "the test set is sealed",
    9: "the tiny-overfit gate passes",
    10: "architecture and training frozen before test inspection",
    11: "checkpoint selection used validation only",
    12: "the primary comparison uses the fixed Context-Lift support",
    13: "Predict-with-Depth cannot shrink that support",
    14: "the paired scene bootstrap is used",
    15: "raw and centered definitions are frozen",
    16: "the near-zero disclosure is followed",
    17: "splat-pool symmetry verified before the operational comparison is used",
    18: "the Phase 4 target-lift result never enters the headline gap",
    19: "every training run has a stable validation curve",
    20: "the headline figures are present and built from the current tables",
    21: "the full suite is green at the reporting commit",
}

# The records made before any test outcome: the design correction, the two
# landing-read records, the Mean-Feature floor, and the reporting rules.
DESIGN_CORRECTION = "validation/evidence/phase5/design_correction.md"
PREREGISTRATION_DOCUMENTS = (
    DESIGN_CORRECTION,
    "validation/evidence/phase5/landing_read_asymmetry.md",
    "validation/evidence/phase5/landing_offset_diagnostic.md",
    "validation/evidence/phase5/mean_feature_floor.md",
    "validation/evidence/phase5/reporting_rules.md",
)
FREEZE_DOCUMENT = "FREEZE.md"
PIN_DOCUMENT = "validation/evidence/phase5/pin.md"
FINDINGS_DOCUMENT = "FINDINGS.md"
AMENDMENTS_DOCUMENT = "AMENDMENTS.md"
# Byte-identical to their frozen blobs, per FREEZE.md.
FROZEN_BY_FREEZE = ("PROTOCOL.md", "VALIDATION.md")
# Changed since the freeze by amendments only, to the values pin.md records.
AMENDED_SINCE_FREEZE = (AMENDMENTS_DOCUMENT, "configs/analysis.yaml")

CONTEXT_LIFT_SOURCE = "src/lot/context_lift.py"
MODES_SOURCE = "src/lot/phase5_modes.py"
PHASE5_SOURCE = "src/lot/phase5.py"
PREDICTORS_SOURCE = "src/lot/predictors.py"
REFERENCE_SOURCE = "src/lot/phase5_reference.py"
REPORT_SOURCE = "src/lot/phase5_report.py"
TRAINING_SOURCES = ("configs/phase5.yaml", "src/lot/predictors.py", "src/lot/train.py")
# Every file the conditions read at E.
SOURCES_AT_E = PREREGISTRATION_DOCUMENTS + (
    CONTEXT_LIFT_SOURCE, MODES_SOURCE, PHASE5_SOURCE, PREDICTORS_SOURCE, REFERENCE_SOURCE,
    FREEZE_DOCUMENT, PIN_DOCUMENT, FINDINGS_DOCUMENT,
) + FROZEN_BY_FREEZE + AMENDED_SINCE_FREEZE
PHASE4_CHECK_SCRIPT = "scripts/phase4_acceptance_check.py"
VALIDATOR_SUMMARY = "validation/evidence/reaudit/borah_check_2_3.json"

# The suite's files whose results each condition reads, from the one run of
# condition 21.
SUITE_FILES: dict[int, tuple[str, ...]] = {
    3: ("tests/test_context_lift.py",),
    4: ("tests/test_predictors.py",),
    5: ("tests/test_context_lift.py", "tests/test_phase5_gate.py"),
    6: ("tests/test_phase5_depth_equivalence.py",),
    7: ("tests/test_phase5_folds.py",),
    8: ("tests/test_phase5_train.py",),
    11: ("tests/test_phase5_train.py",),
    12: ("tests/test_phase5_score.py",),
    14: ("tests/test_paired_bootstrap.py",),
    16: ("tests/test_phase5_estimands.py", "tests/test_phase5_outcomes.py"),
}
# The test-tree modules those files run under. conftest.py puts the code on
# the import path, and scenes.py builds the analytic scenes that
# test_context_lift.py and test_phase5_score.py check against.
# test_phase5_outcomes.py imports test_phase5_estimands.py, which SUITE_FILES
# names already. Tests are neutral to code ancestry, so the suite's results at
# R are evidence only for the files registered at E: each must be E's own, or
# named with its reason in post_evaluation_changes.md.
SUITE_SUPPORT: tuple[str, ...] = ("tests/conftest.py", "tests/scenes.py")

# The comparator's frozen signature and the names its transport path may not
# reference. tests/test_context_lift.py holds the same lists for the code at R.
CONTEXT_LIFT_PARAMETERS = (
    "depth_context_aligned", "K_context", "K_target", "T_target_from_context",
    "context_hw", "target_hw", "patch_size",
)
FORBIDDEN_IN_TRANSPORT = (
    "depth_target", "target_depth", "est_target", "target_aligned", "pointmap", "flow",
    "correspondence", "features_target", "target_features",
)
TRANSPORT_DEPTH_NAMES = frozenset({"depth_context_aligned", "depth", "depth_valid",
                                   "depth_context"})
# The predictor's frozen inputs and the names its module may not reference.
# tests/test_predictors.py holds the same lists for the code at R.
FORWARD_PARAMETERS = ("features_context", "depth_context_aligned", "camera", "context_valid")
FORBIDDEN_PREDICTOR_INPUTS = (
    "target_rgb", "rgb_target", "features_target", "target_features", "depth_target",
    "target_depth", "target_pointmap", "pointmap", "optical_flow", "flow", "covisible",
    "co_visible", "correspondence", "landing", "uv_target",
)
FORBIDDEN_PREDICTOR_OPERATIONS = (
    "unproject", "project", "transform_points", "grid_sample", "transport_plan",
    "apply_transport_plan", "splat", "zbuffer", "z_buffer", "rotation_homography",
    "apply_homography", "visibility_masks", "sample_features_bilinear",
    "sample_map_bilinear", "inverse", "invert_se3",
)
PREDICTOR_LOCAL_IMPORTS = [("encoders", ("PATCH_SIZE",))]
# The camera quantities both headline methods read from one PairCameras.
CAMERA_FIELDS = ("K_context", "K_target", "T_target_from_context", "context_hw", "target_hw")
# The verdict gate step 9 writes when the splat arm is context-side only.
SPLAT_SYMMETRY_VERDICT = "context-side only; no transport call receives the target estimate"

# Decision 2's sentences: engaged on two paths, engaged on the only path, and
# printed when the wording is not engaged.
ENGAGED_TWO_PATH = (WORDING_VETO, WORDING_SMALL_SIGN_CONSISTENT, WORDING_PATH_SENSITIVE,
                    WORDING_OUTSIDE_ON_COMMON_CELLS)
ENGAGED_SINGLE_PATH = (WORDING_ONLY_PATH_CLEAR, WORDING_ONLY_PATH_INCLUDES_ZERO)
NOT_ENGAGED = (WORDING_OUTSIDE_BAND, WORDING_NOT_ESTIMABLE)
# The claim no output may make: no equivalence region was ever frozen.
CLAIM_WORD = re.compile("equivalen", re.IGNORECASE)

# The headline cells recomputed from the parquets: every scope row, under both
# metrics, for the gap and both margins. A tolerance, not bitwise equality,
# because a matrix product can differ in its last bits across machines.
RECOMPUTED_QUANTITIES = ("delta_learn_pp", "cl_margin", "predict_margin")
RECOMPUTE_TOLERANCE = 1e-12
# The columns that name one cell of a table beside its metric.
CELL_KEY_COLUMNS = ("analysis", "axis", "bin", "region", "path", "row_kind", "offset_label",
                    "population", "quantity", "seed", "scope")
# The metric pairs every table must cover alike.
METRIC_PAIRS = (("centered", "raw"), ("l2_centered", "l2_raw"))
# How many problems a condition's notes show before counting the rest.
SHOWN_PROBLEMS = 30


class Unavailable(RuntimeError):
    """An artifact a condition rests on cannot be read or cannot be verified."""


class SceneInputsUnavailable(RuntimeError):
    """The scene inputs a recount needs are not on this machine."""


@dataclasses.dataclass(frozen=True)
class ConditionResult:
    """One condition: whether it holds, what the reader should know, what was read."""

    number: int
    title: str
    ok: bool
    notes: tuple[str, ...]
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"number": self.number, "title": self.title, "ok": self.ok,
                "notes": list(self.notes), "evidence": self.evidence}


@dataclasses.dataclass
class AcceptanceVerdict:
    """Every condition's result, and the verdict record written from them."""

    level: str
    conditions: list[ConditionResult]
    record: dict[str, Any]

    @property
    def passed(self) -> bool:
        numbers = [condition.number for condition in self.conditions]
        return numbers == sorted(CONDITION_TITLES) and all(c.ok for c in self.conditions)

    def failed(self) -> list[int]:
        return [condition.number for condition in self.conditions if not condition.ok]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _short(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else _canonical(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return _number(value) and math.isfinite(value)


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _same(a: Any, b: Any) -> bool:
    """Equality with two NaNs equal."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _close(a: Any, b: Any, tolerance: float = RECOMPUTE_TOLERANCE) -> bool:
    if not (_number(a) and _number(b)):
        return False
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(float(a) - float(b)) <= tolerance


def _sha256(data: bytes | None) -> str | None:
    return None if data is None else hashlib.sha256(data).hexdigest()


def _utc(value: Any) -> datetime | None:
    """An instant from lot.phase5_check.utc_timestamp's format, or None."""
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


def _sign(value: float) -> int:
    return 0 if value == 0.0 else (1 if value > 0.0 else -1)


# ---------------------------------------------------------------------------
# The repository
# ---------------------------------------------------------------------------

def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    """git in root, with cwd rather than -C, which git 1.8.3 lacks."""
    return subprocess.run(["git", *args], cwd=str(root), capture_output=True, timeout=600)


def git_blob(repo_root: Path, commit: str, path: str) -> bytes | None:
    """A file's bytes as git stores them at a commit, or None where it holds none."""
    done = _git(Path(repo_root), "cat-file", "blob", f"{commit}:{path}")
    return done.stdout if done.returncode == 0 else None


class _Repository:
    """The repository acceptance runs in, read through git and memoized.

    reader replaces git_blob for the file contents at a commit, for tests.
    """

    def __init__(self, root: Path, reader: Callable[[str, str], bytes | None] | None):
        self.root = Path(root)
        self._reader = reader
        self._memo: dict[tuple, Any] = {}

    def _run(self, *args: str) -> str:
        try:
            done = _git(self.root, *args)
        except (OSError, subprocess.SubprocessError) as error:
            raise Unavailable(f"git could not run in {self.root}: {error}") from None
        if done.returncode != 0:
            message = done.stderr.decode("utf-8", errors="replace").strip()
            raise Unavailable(f"git {' '.join(args)} failed in {self.root}: {message}")
        return done.stdout.decode("utf-8", errors="replace")

    def _remember(self, key: tuple, compute: Callable[[], Any]) -> Any:
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    def blob(self, commit: str, path: str) -> bytes | None:
        def read() -> bytes | None:
            if self._reader is not None:
                return self._reader(commit, path)
            try:
                return git_blob(self.root, commit, path)
            except (OSError, subprocess.SubprocessError) as error:
                raise Unavailable(f"git could not read {path} at {commit}: {error}") from None
        return self._remember(("blob", commit, path), read)

    def prefetch(self, commit: str, paths: Sequence[str]) -> None:
        """Read many files at one commit through one git process, into the memo.

        Each file reads exactly as blob would read it alone. With a reader
        given, the reader is asked instead.
        """
        wanted = [path for path in dict.fromkeys(paths) if ("blob", commit, path) not in self._memo]
        if not wanted:
            return
        if self._reader is not None:
            for path in wanted:
                self.blob(commit, path)
            return
        request = "".join(f"{commit}:{path}\n" for path in wanted).encode("utf-8")
        try:
            done = subprocess.run(["git", "cat-file", "--batch"], cwd=str(self.root),
                                  input=request, capture_output=True, timeout=600)
        except (OSError, subprocess.SubprocessError):
            return
        if done.returncode != 0:
            return
        output, position = done.stdout, 0
        for path in wanted:
            end = output.find(b"\n", position)
            if end < 0:
                return
            header = output[position:end].split()
            position = end + 1
            if len(header) == 3 and header[1] == b"blob":
                size = int(header[2])
                self._memo[("blob", commit, path)] = output[position: position + size]
                position += size + 1
            elif header and header[-1] == b"missing":
                self._memo[("blob", commit, path)] = None
            else:
                # Not a file at this commit, a tree for instance. Read it alone.
                if len(header) == 3:
                    position += int(header[2]) + 1
                continue

    def text(self, commit: str, path: str) -> str | None:
        data = self.blob(commit, path)
        return None if data is None else data.decode("utf-8")

    def head(self) -> str:
        return self._remember(("head",), self.current_head)

    def current_head(self) -> str:
        return self._run("rev-parse", "HEAD").strip()

    def status(self) -> list[str]:
        return [line for line in self._run("status", "--porcelain").splitlines() if line.strip()]

    def is_ancestor(self, older: str, newer: str) -> bool:
        try:
            done = _git(self.root, "merge-base", "--is-ancestor", older, newer)
        except (OSError, subprocess.SubprocessError) as error:
            raise Unavailable(f"git could not run in {self.root}: {error}") from None
        if done.returncode in (0, 1):
            return done.returncode == 0
        message = done.stderr.decode("utf-8", errors="replace").strip()
        raise Unavailable(f"git merge-base {older} {newer} failed: {message}")

    def ancestors(self, commit: str) -> frozenset[str]:
        return self._remember(("ancestors", commit), lambda: frozenset(
            self._run("rev-list", commit).split()))

    def commits_touching(self, path: str) -> list[str]:
        """Every commit on HEAD's history whose change names path, newest first."""
        self.prefetch_touching([path])
        return self._memo[("touching", path)]

    def prefetch_touching(self, paths: Sequence[str]) -> None:
        """commits_touching for many paths, through one git log, into the memo.

        A merge commit names no file here. The commit that made a change is
        listed on whichever side of a merge it was made.
        """
        wanted = [path for path in dict.fromkeys(paths) if ("touching", path) not in self._memo]
        if not wanted:
            return
        text = self._run("log", "--format=commit %H", "--name-only", "--no-renames", "HEAD",
                         "--", *wanted)
        touched: dict[str, list[str]] = {path: [] for path in wanted}
        current = None
        for line in text.splitlines():
            if line.startswith("commit "):
                current = line.split()[1]
            elif line.strip() and current is not None:
                name = line.strip().replace("\\", "/")
                if name in touched and current not in touched[name]:
                    touched[name].append(current)
        for path, commits in touched.items():
            self._memo[("touching", path)] = commits

    def changed(self, older: str, newer: str) -> list[str]:
        text = self._run("diff", "--name-only", "--no-renames", "-z", older, newer, "--")
        return sorted({part.strip().replace("\\", "/") for part in text.split("\0")
                       if part.strip()})

    def blob_id(self, commit: str, path: str) -> str | None:
        try:
            return self._run("rev-parse", f"{commit}:{path}").strip()
        except Unavailable:
            return None


# ---------------------------------------------------------------------------
# The defaults behind the injection points
# ---------------------------------------------------------------------------

_SUMMARY = re.compile(r"\b\d+ (passed|failed|errors?|skipped|deselected)\b|no tests ran")


def _test_file(case: Any) -> str:
    """The test file a junit testcase belongs to, as a repository path."""
    if case.get("file"):
        return Path(case.get("file")).as_posix()
    parts = (case.get("classname") or case.get("name") or "").split(".")
    for index, part in enumerate(parts):
        if part.startswith("test_") or part.endswith("_test"):
            return "/".join(parts[: index + 1]) + ".py"
    return "/".join(parts) + ".py"


def _junit_results(path: Path) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Per test file and in total: passed, failed, errors, skipped."""
    blank = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    files: dict[str, dict[str, int]] = {}
    counts = dict(blank)
    for case in ElementTree.parse(path).getroot().iter("testcase"):
        if case.find("failure") is not None:
            outcome = "failed"
        elif case.find("error") is not None:
            outcome = "errors"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        slot = files.setdefault(_test_file(case), dict(blank))
        slot[outcome] += 1
        counts[outcome] += 1
    return files, counts


def run_suite(repo_root: Path) -> dict[str, Any]:
    """Run the full suite at repo_root in a subprocess, and read what it did.

    The run writes its junit report and its temporary files outside the
    repository and keeps no cache, so it leaves the worktree as it found it.
    Returns the command, the exit code, the summary line, the counts in total
    and per test file, and the last lines of its output.
    """
    root = Path(repo_root)
    with tempfile.TemporaryDirectory(prefix="phase5_acceptance_suite_") as scratch:
        junit = Path(scratch) / "suite.xml"
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   f"--junitxml={junit}", f"--basetemp={Path(scratch) / 'basetemp'}"]
        # The output is read for its summary only, so a byte the locale cannot
        # decode is replaced rather than allowed to fail the run.
        done = subprocess.run(command, cwd=str(root), capture_output=True, text=True,
                              errors="replace")
        if junit.is_file():
            files, counts = _junit_results(junit)
        else:
            files, counts = {}, {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    lines = done.stdout.splitlines()
    summary = next((line.strip().strip("=").strip() for line in reversed(lines)
                    if _SUMMARY.search(line)), "")
    return {
        "command": command[:6] + ["--junitxml=<temporary>", "--basetemp=<temporary>"],
        "returncode": done.returncode,
        "summary": summary,
        "counts": counts,
        "files": files,
        "output_tail": lines[-30:] + done.stderr.splitlines()[-10:],
    }


def run_phase4_acceptance_check(cfg: Any, repo_root: Path) -> dict[str, Any]:
    """Rerun scripts/phase4_acceptance_check.py on the accepted Phase 4 run.

    The Phase 4 run is the configuration's phase4_dir. A relative one is read
    from the working directory, as every mode reads it. The validator summary
    is the one scripts/BORAH_PHASE4.md section 6 writes into the repository.
    """
    root = Path(repo_root)
    phase4 = Path(cfg.phase4_dir)
    if not phase4.is_absolute():
        phase4 = Path.cwd() / phase4
    command = [sys.executable, str(root / PHASE4_CHECK_SCRIPT),
               "--eval-dir", str(phase4 / "eval"), "--tables", str(phase4 / "tables"),
               "--validator", str(root / VALIDATOR_SUMMARY)]
    done = subprocess.run(command, cwd=str(root), capture_output=True, text=True,
                          errors="replace")
    return {"command": command, "returncode": done.returncode,
            "stdout": done.stdout.splitlines(), "stderr": done.stderr.splitlines()[-20:]}


def recount_primary_support(cfg: Any, analysis: Any, level: str, scene: str,
                            pairs: Sequence[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """Each pair's primary support recounted from the scene inputs, at level.

    The count comes from lot.phase5.plan_example, which training uses: the
    Context-Lift landings from aligned context depth, refereed by ground
    truth, with no predictor in sight. Evaluation computes the same support
    inline, so this is a second path to the number each record carries.

    Raises SceneInputsUnavailable when the inputs are not on this machine:
    the Phase 4 convention record, the scene's manifest, or its feature or
    depth cache. Inputs that exist and fail to read raise as they fail.
    """
    from .encoders import CACHE_META_NAME, cache_dir
    from .phase5 import build_scene_inputs, load_convention_record, phase5_scene_pairs
    from .phase5 import plan_example
    from .render_replica import MANIFEST_NAME

    needed = {
        "the Phase 4 convention record": (Path(cfg.phase4_dir) / "evidence"
                                          / "convention_record.json"),
        "the scene manifest": Path(cfg.renders_root) / scene / MANIFEST_NAME,
        "the feature cache": cache_dir(cfg.cache_root, cfg.feature_encoder, scene)
        / CACHE_META_NAME,
        "the depth cache": cache_dir(cfg.cache_root, cfg.depth_encoder, scene) / CACHE_META_NAME,
    }
    absent = [f"{name} at {path}" for name, path in needed.items() if not Path(path).is_file()]
    if absent:
        raise SceneInputsUnavailable(
            f"the scene inputs of {scene} are not available here: {', '.join(absent)}")
    inputs = build_scene_inputs(cfg, analysis, scene, load_convention_record(cfg))
    try:
        by_pair = {(p.context_frame_id, p.target_frame_id): p
                   for p in phase5_scene_pairs(cfg, analysis, scene)}
        counts: dict[tuple[str, str], int] = {}
        for key in pairs:
            pair = by_pair.get(tuple(key))
            if pair is None:
                raise ValueError(f"{scene} {key} is not a pair of the frozen sample")
            plan = plan_example(cfg, analysis, inputs, pair, level)
            counts[tuple(key)] = 0 if plan is None else int(plan.support.sum())
        return counts
    finally:
        inputs.close()


def load_checkpoint_state(path: Path) -> dict[str, Any]:
    """A checkpoint as lot.train saved it, on the CPU."""
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _checkpoint_facts(state: Any) -> dict[str, Any]:
    """What acceptance reads from a checkpoint, without keeping its tensors."""
    if not isinstance(state, Mapping):
        raise Unavailable("the checkpoint does not hold a mapping")
    model = state.get("model")
    shapes = numel = None
    if isinstance(model, Mapping):
        shapes = {str(name): [int(size) for size in value.shape] for name, value in model.items()}
        numel = int(sum(int(value.numel()) for value in model.values()))
    return {
        "step": state.get("step"),
        "validation_centered_cosine": state.get("validation_centered_cosine"),
        "fold": state.get("fold"),
        "seed": state.get("seed"),
        "training_config_digest": state.get("training_config_digest"),
        "shapes": shapes,
        "numel": numel,
    }


# ---------------------------------------------------------------------------
# Reading source code at a commit
# ---------------------------------------------------------------------------

def _parse(source: str | None, what: str) -> ast.Module:
    if source is None:
        raise Unavailable(f"{what} is absent")
    try:
        return ast.parse(source)
    except SyntaxError as error:
        raise Unavailable(f"{what} does not parse: {error}") from None


def _function(tree: ast.Module, name: str, owner: str | None = None) -> ast.FunctionDef:
    """A top-level function, or a method of a top-level class, by name."""
    body: list[ast.stmt] = tree.body
    if owner is not None:
        classes = [node for node in body if isinstance(node, ast.ClassDef) and node.name == owner]
        if not classes:
            raise Unavailable(f"no class {owner}")
        body = classes[0].body
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise Unavailable(f"no function {name}" + (f" in class {owner}" if owner else ""))


def _parameters(function: ast.FunctionDef) -> list[str]:
    args = function.args
    return [arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs]


def _identifiers(node: ast.AST) -> set[str]:
    """Every identifier code references: names, attributes, arguments, imports."""
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
        elif isinstance(child, ast.arg):
            found.add(child.arg)
        elif isinstance(child, ast.keyword) and child.arg:
            found.add(child.arg)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                found.add(alias.asname or alias.name.split(".")[0])
    return found


def _names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _calls(node: ast.AST, name: str, plain: bool = False) -> list[ast.Call]:
    """Every call of name inside node, in line order. plain: called by bare name."""
    found = [child for child in ast.walk(node) if _call_name(child) == name
             and (not plain or isinstance(child.func, ast.Name))]
    return sorted(found, key=lambda call: (call.lineno, call.col_offset))


def _pairs(target: ast.AST, value: ast.AST) -> list[tuple[str, ast.AST]]:
    """The names a target binds, each with the value it receives."""
    if isinstance(target, ast.Name):
        return [(target.id, value)]
    if isinstance(target, (ast.Tuple, ast.List)):
        if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts):
            return [pair for t, v in zip(target.elts, value.elts) for pair in _pairs(t, v)]
        return [(name, value) for element in target.elts for name, _ in _pairs(element, value)]
    if isinstance(target, ast.Starred):
        return _pairs(target.value, value)
    return []


def _bindings(function: ast.AST, name: str) -> list[tuple[int, ast.AST]]:
    """Every value a name is bound to inside a function, with its line, in order.

    Assignments, augmented and annotated assignments, loop targets, and with
    targets all bind. A tuple assignment pairs element by element.
    """
    found: list[tuple[int, ast.AST]] = []
    for node in ast.walk(function):
        bound: list[tuple[ast.AST, ast.AST]] = []
        if isinstance(node, ast.Assign):
            bound = [(target, node.value) for target in node.targets]
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and node.value is not None:
            bound = [(node.target, node.value)]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bound = [(node.target, node.iter)]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            bound = [(item.optional_vars, item.context_expr) for item in node.items
                     if item.optional_vars is not None]
        elif isinstance(node, ast.NamedExpr):
            bound = [(node.target, node.value)]
        for target, value in bound:
            found += [(node.lineno, bound_value) for bound_name, bound_value in
                      _pairs(target, value) if bound_name == name]
    return sorted(found, key=lambda item: item[0])


def _first_argument(call: ast.Call, keyword: str) -> ast.AST | None:
    if call.args:
        return call.args[0]
    for item in call.keywords:
        if item.arg == keyword:
            return item.value
    return None


def _argument(call: ast.Call, index: int, keyword: str) -> ast.AST | None:
    if len(call.args) > index:
        return call.args[index]
    for item in call.keywords:
        if item.arg == keyword:
            return item.value
    return None


def _local_imports(tree: ast.Module) -> list[tuple[str, tuple[str, ...]]]:
    """The lot modules a module imports from, with the names, in order."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            local = (node.level or 0) > 0 or (node.module or "").startswith("lot")
            if local:
                found.append((node.module or "", tuple(alias.name for alias in node.names)))
    return found


def _enclosing_functions(tree: ast.Module, name: str) -> list[tuple[int, str | None]]:
    """Where a name is referenced: each line and the innermost function around it."""
    found: list[tuple[int, str | None]] = []

    def visit(node: ast.AST, inside: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Name) and child.id == name:
                found.append((child.lineno, inside))
            visit(child, inside)

    visit(tree, None)
    return found


def _literal_tuple(tree: ast.Module, name: str) -> tuple | None:
    """The last top-level literal assigned to a name, or None."""
    value = None
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) and node.value is not None else [])
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                try:
                    value = ast.literal_eval(node.value)
                except ValueError:
                    value = None
    return tuple(value) if isinstance(value, (tuple, list)) else None


# ---------------------------------------------------------------------------
# The evaluation ledger
# ---------------------------------------------------------------------------

def ledger_problems(
    attempts: Sequence[Mapping[str, Any]],
    *,
    level: str,
    scenes: Sequence[str],
    live: Mapping[str, str],
    written_utc: Mapping[str, str],
    commit: str,
    licence: Mapping[str, str],
) -> tuple[list[str], dict[str, Any]]:
    """One explained evaluation per scene and level, read from the attempts.

    attempts is lot.phase5_modes.evaluation_attempts' output. live maps each
    scene to its live parquet's sha256, and written_utc to its run record's
    written_utc. The rules, reporting_rules.md sections 7 and 9:

    - every attempt at the level is for an evaluation scene, at commit E,
      under the run's licence;
    - a scene has at most one written close, and it names the live parquet;
    - exactly one attempt is on record as having written the live parquet:
      its written close; else an error close that names the live parquet
      and found none at its start, which a signal landing after the write
      leaves; else one attempt killed outright whose start precedes the
      parquet's written_utc;
    - every other error attempt wrote nothing and records its message;
    - every other unfinished attempt found the live parquet at its start, so
      it could only resume. Or it started before a written close that found
      no parquet at its start, while no attempt found any other file at the
      path, so it left nothing there: the attempt was killed outright and
      then resubmitted, as the runbook prescribes. Any other is unexplained;
    - a resumed attempt found the live parquet.

    Returns the problems, and per scene the attempt that wrote its parquet.
    """
    problems: list[str] = []
    expected = list(scenes)
    at_level = [attempt for attempt in attempts if attempt.get("level") == level]
    for attempt in at_level:
        where = f"attempt {attempt.get('attempt')} of {attempt.get('scene')}"
        if attempt.get("scene") not in expected:
            problems.append(f"{where}: {attempt.get('scene')!r} is not an evaluation scene of "
                            f"level {level}")
        if attempt.get("event") != LEDGER_EVENT:
            problems.append(f"{where}: event {attempt.get('event')!r} is not {LEDGER_EVENT!r}")
        if attempt.get("commit") != commit:
            problems.append(f"{where}: it ran at commit {attempt.get('commit')}, not at E "
                            f"{commit}")
        if _canonical(attempt.get("licence")) != _canonical(dict(licence)):
            problems.append(f"{where}: it was licensed by {_short(attempt.get('licence'))}, not "
                            "by the receipts the run names")
    writers: dict[str, Any] = {}
    explained: list[dict[str, Any]] = []
    for scene in expected:
        mine = [attempt for attempt in at_level if attempt.get("scene") == scene]
        sha = live.get(scene)
        when = _utc(written_utc.get(scene))
        if not mine:
            problems.append(f"{scene}: no attempt is on record at level {level}")
            continue
        written = [a for a in mine if a.get("status") == "written"]
        errors = [a for a in mine if a.get("status") == "error"]
        unfinished = [a for a in mine if a.get("status") == LEDGER_UNFINISHED]
        resumed = [a for a in mine if a.get("status") == "exists"]
        if len(written) > 1:
            problems.append(f"{scene}: {len(written)} written evaluations are on record, and "
                            "one evaluation per scene and level is allowed")
        for attempt in written:
            if attempt.get("parquet_sha256") != sha:
                problems.append(
                    f"{scene}: attempt {attempt.get('attempt')} wrote parquet "
                    f"{attempt.get('parquet_sha256')}, which is not the live parquet {sha}")
        writer = next((a for a in written if a.get("parquet_sha256") == sha), None)
        if writer is None and not written:
            named = [a for a in errors if a.get("parquet_sha256") == sha
                     and a.get("parquet_sha256_at_start") != sha]
            killed = [a for a in unfinished if when is not None
                      and _utc(a.get("started_utc")) is not None
                      and _utc(a.get("started_utc")) <= when]
            if len(named) == 1:
                writer = named[0]
            elif len(named) > 1:
                problems.append(f"{scene}: {len(named)} error attempts name the live parquet, so "
                                "which one wrote it is ambiguous")
            elif len(killed) == 1:
                writer = killed[0]
            elif len(killed) > 1:
                problems.append(f"{scene}: {len(killed)} unfinished attempts started before the "
                                "parquet was written, so which one wrote it is ambiguous")
        if writer is None:
            problems.append(f"{scene}: no attempt is on record as having written the live "
                            f"parquet {sha}")
        else:
            writers[scene] = {"attempt": writer.get("attempt"), "status": writer.get("status")}
        for attempt in errors:
            if attempt is writer:
                continue
            at_close = attempt.get("parquet_sha256")
            at_start = attempt.get("parquet_sha256_at_start")
            if at_close not in (None, at_start) and at_close == sha:
                beside = "" if writer is None else f", beside attempt {writer.get('attempt')}"
                problems.append(f"{scene}: error attempt {attempt.get('attempt')} also names the "
                                f"live parquet {sha} as its output{beside}, so the scene was "
                                "evaluated more than once")
            elif at_close not in (None, at_start):
                problems.append(f"{scene}: error attempt {attempt.get('attempt')} wrote a parquet, "
                                f"{at_close}, that is not the live one")
            elif not attempt.get("message"):
                problems.append(f"{scene}: error attempt {attempt.get('attempt')} records no "
                                "message, so nothing explains it")
            else:
                explained.append({"scene": scene, "attempt": attempt.get("attempt"),
                                  "message": attempt.get("message")})
        # A written close whose start found no parquet at the path. An attempt
        # that started before it and left a parquet there would have been found.
        clean = (_utc(writer.get("started_utc"))
                 if writer is not None and writer.get("status") == "written"
                 and writer.get("parquet_sha256_at_start") is None else None)
        # Some attempt found a file other than the live parquet at the path, so
        # a file was moved aside, and what an earlier attempt left is unknown.
        moved = any(a.get("parquet_sha256_at_start") not in (None, sha) for a in mine)
        for attempt in unfinished:
            if attempt is writer:
                continue
            if sha is not None and attempt.get("parquet_sha256_at_start") == sha:
                # An attempt that finds its output resumes it or refuses, and
                # never evaluates again, so this one wrote nothing.
                explained.append({"scene": scene, "attempt": attempt.get("attempt"),
                                  "message": "killed outright after finding the live parquet "
                                             "at its start, so it could only resume"})
                continue
            began = _utc(attempt.get("started_utc"))
            if (clean is not None and began is not None and began < clean
                    and attempt.get("parquet_sha256_at_start") is None and not moved):
                explained.append({"scene": scene, "attempt": attempt.get("attempt"),
                                  "message": "killed outright before the written attempt "
                                             "started, which found no parquet at its start, "
                                             "so it left nothing"})
                continue
            problems.append(
                f"{scene}: attempt {attempt.get('attempt')} is unfinished. It was killed "
                "outright, and nothing on record says that it wrote nothing")
        for attempt in resumed:
            if attempt.get("parquet_sha256") != sha:
                problems.append(f"{scene}: attempt {attempt.get('attempt')} resumed parquet "
                                f"{attempt.get('parquet_sha256')}, not the live one")
    return problems, {"attempts_at_level": len(at_level), "writers": writers,
                      "explained_errors": explained}


# ---------------------------------------------------------------------------
# What acceptance reads, read once and tolerantly
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _Scan:
    """The evaluation directory of the level: what is there, and its run records."""

    directory: Path
    files: dict[str, Path]
    records: dict[str, dict[str, Any]]
    sha: dict[str, str]
    problems: list[str]


@dataclasses.dataclass
class _Receipt:
    stem: str
    path: Path
    sha256: str
    payload: dict[str, Any]


@dataclasses.dataclass
class _File:
    """A JSON file the lock binds: where it is, its sha256, and its content."""

    path: Path
    sha256: str | None
    payload: Any
    error: str | None = None


@dataclasses.dataclass
class _Checkpoint:
    path: Path
    sha256: str | None
    facts: dict[str, Any] | None
    error: str | None = None


@dataclasses.dataclass
class _RowFacts:
    """What the evaluation rows say, gathered in one pass over every scene."""

    scenes: list[str] = dataclasses.field(default_factory=list)
    collapsed: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    rotation_pairs: dict[str, int] = dataclasses.field(default_factory=dict)
    n_primary: dict[str, dict[tuple[str, str], Any]] = dataclasses.field(default_factory=dict)
    explicit_drift: list[str] = dataclasses.field(default_factory=list)
    support_drift: list[str] = dataclasses.field(default_factory=list)
    partition: list[str] = dataclasses.field(default_factory=list)
    failures: list[str] = dataclasses.field(default_factory=list)
    predictor_nonfinite: list[str] = dataclasses.field(default_factory=list)
    malformed: list[str] = dataclasses.field(default_factory=list)
    failures_per_seed: dict[int, int] = dataclasses.field(default_factory=dict)
    rows: int = 0


class AcceptanceContext:
    """Every artifact the conditions read, each read once, each failure kept.

    A piece that cannot be read or verified raises Unavailable whenever a
    condition asks for it, so the conditions that rest on it fail and the
    others still run.
    """

    def __init__(
        self,
        cfg: Any,
        analysis: Any,
        config_path: Path,
        level: str,
        *,
        expected_scenes: Sequence[str] | None = None,
        folds: Sequence[Any] | None = None,
        repo_root: Path | None = None,
        tables_root: Path | None = None,
        figures_root: Path | None = None,
        suite_runner: Callable[[Path], Mapping[str, Any]] | None = None,
        phase4_check: Callable[[Any, Path], Mapping[str, Any]] | None = None,
        checkpoint_loader: Callable[[Path], Any] | None = None,
        recount: Callable[..., Mapping[tuple[str, str], int]] | None = None,
        source_reader: Callable[[str, str], bytes | None] | None = None,
    ):
        from .train import training_config_from

        self.cfg = cfg
        self.analysis = analysis
        self.config_path = Path(config_path)
        self.level = level
        self.folds = list(folds) if folds is not None else frozen_folds()
        self.expected_scenes = (list(expected_scenes) if expected_scenes is not None
                                else evaluation_scenes(self.folds))
        self.settings = training_config_from(cfg.training)
        self.seeds = [int(seed) for seed in self.settings.seeds]
        self.primary_level = cfg.primary_alignment_level
        self.run_dir = Path(cfg.run_dir)
        self.evidence_dir = Path(cfg.evidence_dir)
        self.tables_dir = (Path(tables_root) if tables_root is not None
                           else self.run_dir / TABLES_DIR) / level
        self.figures_dir = (Path(figures_root) if figures_root is not None
                            else self.run_dir / FIGURES_DIR) / level
        self.repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
        self.repo = _Repository(self.repo_root, source_reader)
        self.suite_runner = suite_runner or run_suite
        self.phase4_check = phase4_check or run_phase4_acceptance_check
        self.checkpoint_loader = checkpoint_loader or load_checkpoint_state
        self.recount = recount or recount_primary_support
        self._memo: dict[tuple, tuple[bool, Any]] = {}

    # -- memo ---------------------------------------------------------------

    def _once(self, key: tuple, compute: Callable[[], Any]) -> Any:
        if key not in self._memo:
            try:
                self._memo[key] = (True, compute())
            except Exception as error:  # noqa: BLE001
                # Kept and raised again to every condition that asks, so one
                # unreadable artifact fails each condition resting on it.
                self._memo[key] = (False, error)
        ok, value = self._memo[key]
        if ok:
            return value
        raise value

    def prefetch(self) -> None:
        """Read every file the conditions read at E and at R, and their histories, at once.

        One git process per commit, instead of one per file. What cannot be
        read here is read again, and reported, by the condition that needs it.
        """
        suite = tuple(dict.fromkeys(
            name for names in SUITE_FILES.values() for name in names)) + SUITE_SUPPORT
        with contextlib.suppress(Unavailable):
            commit = self.commit_e()
            self.repo.prefetch(commit, SOURCES_AT_E + suite + tuple(
                path for path in FROZEN_AT_E if not any(ch in path for ch in "*?[")))
        with contextlib.suppress(Unavailable):
            self.repo.prefetch(self.repo.head(),
                               (REPORT_SOURCE, POST_EVALUATION_CHANGES) + suite)
        with contextlib.suppress(Unavailable):
            self.repo.prefetch_touching(PREREGISTRATION_DOCUMENTS + TRAINING_SOURCES)

    @property
    def keys(self) -> list[str]:
        return [fold_seed_key(fold.index, seed) for fold in self.folds for seed in self.seeds]

    def fold_of(self, key: str) -> Any:
        for fold in self.folds:
            for seed in self.seeds:
                if fold_seed_key(fold.index, seed) == key:
                    return fold, seed
        raise KeyError(key)

    # -- the evaluation records ---------------------------------------------

    def scan(self) -> _Scan:
        return self._once(("scan",), self._scan)

    def _scan(self) -> _Scan:
        import pyarrow.parquet as pq

        from .evaluate import RUN_METADATA_KEY

        directory = self.run_dir / "eval" / self.level
        problems: list[str] = []
        files: dict[str, Path] = {}
        if not directory.is_dir():
            problems.append(f"no evaluation directory at {directory}")
        else:
            names = sorted(entry.name for entry in directory.iterdir())
            partial = [name for name in names if ".partial" in name]
            if partial:
                problems.append(f"{directory} holds unfinished writes {partial}")
            files = {path.stem: path for path in sorted(directory.glob("*.parquet"))
                     if path.is_file()}
        missing = [scene for scene in self.expected_scenes if scene not in files]
        extra = sorted(set(files) - set(self.expected_scenes))
        if missing:
            problems.append(f"no parquet for the evaluation scene(s) {missing}")
        if extra:
            problems.append(f"unexpected scene(s) {extra} in {directory}. Only the evaluation "
                            "scenes are evaluated, each by the fold that held it out")
        records: dict[str, dict[str, Any]] = {}
        sha: dict[str, str] = {}
        for scene in self.expected_scenes:
            path = files.get(scene)
            if path is None:
                continue
            sha[scene] = sha256_file(path)
            try:
                raw = (pq.read_schema(path).metadata or {}).get(RUN_METADATA_KEY)
                record = None if raw is None else json.loads(raw.decode("utf-8"))
            except Exception as error:  # noqa: BLE001
                problems.append(f"{path.name}: its run record cannot be read: "
                                f"{type(error).__name__}: {error}")
                continue
            if not isinstance(record, dict):
                problems.append(f"{path.name} carries no run record inside it")
                continue
            records[scene] = record
        return _Scan(directory, files, records, sha, problems)

    def records(self) -> dict[str, dict[str, Any]]:
        records = self.scan().records
        if not records:
            raise Unavailable("no evaluation run record could be read")
        return records

    def common(self, field: str) -> Any:
        """The one value every run record names for field, or Unavailable."""
        records = self.records()
        absent = [scene for scene, record in records.items() if record.get(field) is None]
        if absent:
            raise Unavailable(f"{field} is absent from the run records of {absent}")
        values = {_canonical(record[field]) for record in records.values()}
        if len(values) != 1:
            raise Unavailable(f"the run records name {len(values)} values of {field}")
        return next(iter(records.values()))[field]

    def commit_e(self) -> str:
        commit = self.common("commit")
        if not (isinstance(commit, str) and FULL_COMMIT.fullmatch(commit)):
            raise Unavailable(f"the run records name commit {commit!r}, which is not a full "
                              "commit hash")
        return commit

    def licence(self) -> dict[str, str]:
        licence = self.common("licence")
        if not isinstance(licence, dict):
            raise Unavailable(f"the run records name licence {_short(licence)}")
        return licence

    def run_values(self) -> dict[str, Any]:
        """The run fields of an output's run record, re-derived from the run records."""
        records = self.records()
        return {
            "level": self.level,
            "evaluation_commit": self.commit_e(),
            "training_commit": self.commit_e(),
            **{field: self.common(field) for field in (
                "config_digest", "fold_digest", "measurement_digest", "mean_vector_digest",
                "analysis_reporting_digest", "eval_version", "phase4_commit")},
            "seeds": list(self.common("seeds")),
            "folds": sorted({record.get("fold") for record in records.values()}),
            "scenes": [scene for scene in self.expected_scenes if scene in records],
            "bootstrap": self.bootstrap(),
            "licence": dict(self.licence()),
        }

    def bootstrap(self) -> dict[str, Any]:
        return {"primary_unit": "scene", "resamples": self.analysis.bootstrap_resamples,
                "seed": self.analysis.bootstrap_seed,
                "confidence": self.analysis.bootstrap_confidence}

    # -- the receipts -------------------------------------------------------

    def receipt_stems(self) -> dict[str, str]:
        return {KIND_INTEGRATION: "integration_gate", KIND_OVERFIT: "tiny_overfit",
                KIND_LOCK: checkpoint_lock_path(Path("."), self.level).stem}

    def receipt(self, kind: str) -> _Receipt:
        return self._once(("receipt", kind), lambda: self._receipt(kind))

    def _receipt(self, kind: str) -> _Receipt:
        stem = self.receipt_stems()[kind]
        sha = self.licence().get(stem)
        if not isinstance(sha, str):
            raise Unavailable(f"the run's licence names no {stem} receipt")
        try:
            path = locate_receipt(self.evidence_dir, stem, sha)
        except ProvenanceError as error:
            raise Unavailable("; ".join(message for _, message in error.problems)) from None
        data = path.read_bytes()
        if _sha256(data) != sha:
            raise Unavailable(f"{path} changed while it was read")
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, dict):
            raise Unavailable(f"{path} does not hold one receipt")
        return _Receipt(stem, path, sha, payload)

    def expected_identity(self) -> dict[str, Any]:
        """The identity every receipt of the run must carry: E and the frozen digests."""
        return {"commit": self.commit_e(), "config_digest": self.cfg.digest(),
                "fold_digest": fold_digest(self.folds),
                "measurement_digest": self.analysis.measurement_digest()}

    def verification(self, kind: str) -> list[str]:
        """lot.phase5_receipt.verify_against's problems with one receipt, at E."""
        def verify() -> list[str]:
            receipt = self.receipt(kind)
            label = f"the {receipt.stem} receipt"
            bindings: dict[str, Any] = {}
            if kind in (KIND_OVERFIT, KIND_LOCK):
                bindings["gate_receipt"] = self.receipt(KIND_INTEGRATION).path
            if kind == KIND_LOCK:
                bindings.update(overfit_receipt=self.receipt(KIND_OVERFIT).path,
                                level=self.level)
            return verify_against(receipt.path, self.config_path, label,
                                  self.expected_identity(), kind=kind,
                                  reference="evaluated run", **bindings)
        return self._once(("verification", kind), verify)

    def integration(self) -> dict[str, Any]:
        """The integration receipt, verified at E, or Unavailable with its problems."""
        problems = self.verification(KIND_INTEGRATION)
        if problems:
            raise Unavailable("the integration receipt does not verify at E: "
                              + "; ".join(problems))
        return self.receipt(KIND_INTEGRATION).payload

    def gate_order(self) -> list[str]:
        return [str(step.get("step")) for step in gate_steps(self.integration())]

    def gate_step(self, number: str) -> dict[str, Any]:
        """One step of the verified integration receipt, which must have passed."""
        for step in gate_steps(self.integration()):
            if str(step.get("step")) == number:
                if step.get("passed") is not True:
                    raise Unavailable(f"integration gate step {number} did not pass")
                evidence = step.get("evidence")
                if not isinstance(evidence, dict):
                    raise Unavailable(f"integration gate step {number} carries no evidence")
                return evidence
        raise Unavailable(f"the integration receipt holds no step {number}")

    # -- training, checkpoints, controls ------------------------------------

    def training(self) -> dict[str, _File]:
        return self._once(("training",), self._training)

    def _training(self) -> dict[str, _File]:
        out: dict[str, _File] = {}
        for fold in self.folds:
            for seed in self.seeds:
                path = training_record_path(self.run_dir, self.level, fold.index, seed)
                out[fold_seed_key(fold.index, seed)] = self._json_file(path)
        return out

    @staticmethod
    def _json_file(path: Path) -> _File:
        try:
            data = Path(path).read_bytes()
            return _File(Path(path), _sha256(data), json.loads(data.decode("utf-8")))
        except (OSError, ValueError) as error:
            return _File(Path(path), None, None, f"{type(error).__name__}: {error}")

    def controls(self) -> _File:
        return self._once(("controls",), lambda: self._json_file(
            controls_path(self.evidence_dir, self.level)))

    def checkpoints(self) -> dict[str, _Checkpoint]:
        return self._once(("checkpoints",), self._checkpoints)

    def _checkpoints(self) -> dict[str, _Checkpoint]:
        out: dict[str, _Checkpoint] = {}
        for fold in self.folds:
            for seed in self.seeds:
                path = checkpoint_path(self.run_dir, self.level, fold.index, seed)
                key = fold_seed_key(fold.index, seed)
                if not path.is_file():
                    out[key] = _Checkpoint(path, None, None, f"no checkpoint at {path}")
                    continue
                sha = sha256_file(path)
                try:
                    facts = _checkpoint_facts(self.checkpoint_loader(path))
                except Exception as error:  # noqa: BLE001
                    # A checkpoint that does not load is reported with the others.
                    out[key] = _Checkpoint(path, sha, None, f"{type(error).__name__}: {error}")
                    continue
                out[key] = _Checkpoint(path, sha, facts)
        return out

    def architecture(self) -> dict[str, Any]:
        """The frozen architecture's state shapes, built from the config and the gate's grid."""
        def build() -> dict[str, Any]:
            from .encoders import PATCH_SIZE
            from .phase5 import predictor_config_from
            from .predictors import build_predictor

            shape = (self.gate_step("3").get("dino_features") or {}).get("shape")
            if not (isinstance(shape, list) and len(shape) == 3):
                raise Unavailable("gate step 3 records no DINOv2 feature shape")
            grid = (int(shape[1]), int(shape[2]))
            model_cfg = predictor_config_from(self.cfg, (grid[0] * PATCH_SIZE,
                                                         grid[1] * PATCH_SIZE))
            model = build_predictor(model_cfg)
            shapes = {name: [int(size) for size in value.shape]
                      for name, value in model.state_dict().items()}
            return {"grid": list(grid), "shapes": shapes,
                    "parameter_count": int(model.parameter_count())}
        return self._once(("architecture",), build)

    # -- the evaluation rows ------------------------------------------------

    def rows(self) -> _RowFacts:
        return self._once(("rows",), self._rows)

    def _rows(self) -> _RowFacts:
        from .evaluate import read_rows

        scan = self.scan()
        explicit = [field for field in PRIMARY_FIELDS if not field.startswith("predict_")]
        predictor = [field for field in PRIMARY_FIELDS if field.startswith("predict_")]
        splits = tuple(REGION_CONTRASTS.values())
        facts = _RowFacts(failures_per_seed={seed: 0 for seed in self.seeds})
        for scene in self.expected_scenes:
            if scene not in scan.records:
                continue
            path, want = scan.files[scene], scan.sha[scene]
            before = sha256_file(path)
            rows = read_rows(path) if before == want else None
            after = sha256_file(path) if rows is not None else before
            if before != want or after != want:
                raise Unavailable(f"{path} has sha256 {after}, not the {want} it had when its "
                                  "run record was read")
            facts.rows += len(rows)
            groups: dict[tuple, dict[Any, dict[str, Any]]] = {}
            for row in rows:
                key = (row.get("context_frame_id"), row.get("target_frame_id"), row.get("region"))
                slot = groups.setdefault(key, {})
                if row.get("seed") in slot:
                    facts.malformed.append(f"{scene} {key}: two rows for seed {row.get('seed')}")
                slot[row.get("seed")] = row
            pairs: dict[tuple[str, str], dict[str, Any]] = {}
            for (context, target, region), by_seed in sorted(groups.items(), key=_group_order):
                where = f"{scene} {context} -> {target} region {region}"
                if sorted(by_seed) != sorted(self.seeds):
                    facts.malformed.append(f"{where}: rows for seeds {sorted(by_seed)}, not "
                                           f"{self.seeds}")
                    continue
                members = [by_seed[seed] for seed in self.seeds]
                counts = [member.get("n_primary") for member in members]
                if len({_canonical(count) for count in counts}) != 1:
                    facts.support_drift.append(f"{where}: n_primary is {counts} across the seeds' "
                                               "models")
                for field in explicit:
                    values = [member.get(field) for member in members]
                    if not all(_same(values[0], value) for value in values[1:]):
                        facts.explicit_drift.append(f"{where}: {field} is {values} across seeds")
                for seed, member in zip(self.seeds, members):
                    count, failures = member.get("n_primary"), member.get("n_predict_nonfinite")
                    if not (_count(count) and _count(failures) and failures <= count):
                        facts.failures.append(f"{where} seed {seed}: {failures} failures on a "
                                              f"support of {count}")
                    elif count > 0:
                        bad = [field for field in predictor if not _finite(member.get(field))]
                        if bad:
                            facts.predictor_nonfinite.append(
                                f"{where} seed {seed}: {bad} are not finite on {count} samples")
                    if region == "all" and _count(failures):
                        facts.failures_per_seed[seed] = facts.failures_per_seed.get(seed, 0) \
                            + failures
                slot = pairs.setdefault((context, target),
                                        {"regime": members[0].get("regime"), "regions": {}})
                slot["regions"][region] = members
            for (context, target), entry in sorted(pairs.items()):
                regions = entry["regions"]
                where = f"{scene} {context} -> {target}"
                missing = [region for region in REGIONS if region not in regions]
                if missing:
                    facts.partition.append(f"{where}: no rows for region(s) {missing}")
                    continue
                whole = regions["all"][0].get("n_primary")
                for left, right in splits:
                    a, b = regions[left][0].get("n_primary"), regions[right][0].get("n_primary")
                    if not (_count(a) and _count(b) and _count(whole) and a + b == whole):
                        facts.partition.append(f"{where}: n_primary of {left} and {right}, {a} "
                                               f"and {b}, does not sum to the whole {whole}")
                members = regions["all"]
                facts.n_primary.setdefault(scene, {})[(context, target)] = whole
                record: dict[str, Any] = {"scene": scene,
                                          "camera_pair": f"{scene}|{context}|{target}",
                                          "regime": entry["regime"], "n_primary": whole}
                for field in explicit:
                    record[field] = members[0].get(field)
                for field in predictor:
                    values = [member.get(field) for member in members]
                    record[field] = (float(sum(float(v) for v in values) / len(values))
                                     if all(_finite(v) for v in values) else float("nan"))
                facts.collapsed.append(record)
                if entry["regime"] == "rotation":
                    facts.rotation_pairs[scene] = facts.rotation_pairs.get(scene, 0) + 1
            facts.scenes.append(scene)
            del rows
        return facts

    # -- the published tables and figures -----------------------------------

    def published(self) -> Any:
        """The tables as lot.phase5_figures.read_published_tables verifies them."""
        def read() -> Any:
            try:
                return read_published_tables(self.tables_dir, level=self.level)
            except ProvenanceError as error:
                raise Unavailable("; ".join(f"{field}: {message}"
                                            for field, message in error.problems)) from None
        return self._once(("published",), read)

    def tables(self) -> Any:
        """The published tables, if they are the evaluated run's own."""
        def bind() -> Any:
            published = self.published()
            problems = self._binding_problems(published.run_record)
            if problems:
                raise Unavailable("the published tables are not the evaluated run's: "
                                  + "; ".join(problems[:10]))
            return published
        return self._once(("tables",), bind)

    def _binding_problems(self, record: Mapping[str, Any]) -> list[str]:
        problems = [
            f"the tables name {field} {_short(record.get(field))}, and the run {_short(want)}"
            for field, want in self.run_values().items() if record.get(field) != want
        ]
        inputs = record.get("inputs") if isinstance(record.get("inputs"), dict) else {}
        named = set(inputs.values())
        scan = self.scan()
        for scene, sha in scan.sha.items():
            relative = f"eval/{self.level}/{scene}.parquet"
            if inputs.get(relative) != sha:
                problems.append(f"the tables read {relative} with sha256 "
                                f"{inputs.get(relative)}, and it is {sha} now")
        lock = self.receipt(KIND_LOCK).payload
        bound = [(f"{group} {key}", (entry or {}).get("sha256"))
                 for group in ("checkpoints", "training_records")
                 for key, entry in (lock.get(group) or {}).items()]
        bound.append(("controls", (lock.get("controls") or {}).get("sha256")))
        bound += [(f"receipt {stem}", sha) for stem, sha in self.licence().items()]
        for name, sha in bound:
            if sha not in named:
                problems.append(f"the tables were not read from the locked {name}, {sha}")
        return problems

    def figures_record(self) -> dict[str, Any]:
        """The figures manifest's run record, read as it is."""
        def read() -> dict[str, Any]:
            try:
                manifest = json.loads((self.figures_dir / MANIFEST_FILE).read_text(
                    encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise Unavailable(f"the figures manifest cannot be read: {error}") from None
            record = manifest.get("run_record") if isinstance(manifest, dict) else None
            if not isinstance(record, dict):
                raise Unavailable("the figures manifest carries no run record")
            return record
        return self._once(("figures record",), read)

    def checked_figures(self) -> dict[str, Any]:
        def check() -> dict[str, Any]:
            try:
                return check_published_figures(self.figures_dir, self.tables_dir)
            except ProvenanceError as error:
                raise Unavailable("; ".join(f"{field}: {message}"
                                            for field, message in error.problems)) from None
        return self._once(("figures",), check)

    # -- the ledger, the code, the suite, Phase 4 ---------------------------

    def attempts(self) -> list[dict[str, Any]]:
        def read() -> list[dict[str, Any]]:
            try:
                return evaluation_attempts(self.evidence_dir / LEDGER_DIRECTORY)
            except ValueError as error:
                raise Unavailable(f"the evaluation ledger cannot be read: {error}") from None
        return self._once(("attempts",), read)

    def ancestry(self) -> dict[str, Any]:
        """verify_code_ancestry at E, with every problem kept by its field."""
        def run() -> dict[str, Any]:
            commit = self.commit_e()
            try:
                head, changes = verify_code_ancestry(commit, self.repo_root)
            except ProvenanceError as error:
                return {"head": None, "changes": {}, "problems": list(error.problems)}
            return {"head": head, "changes": dict(changes), "problems": []}
        return self._once(("ancestry",), run)

    def ancestry_problems(self, fields: Sequence[str]) -> list[str]:
        return [f"{field}: {message}" for field, message in self.ancestry()["problems"]
                if field in fields]

    def suite(self) -> dict[str, Any]:
        """The full suite at R, run once, with HEAD and the worktree before and after."""
        def run() -> dict[str, Any]:
            head = self.repo.current_head()
            status = self.repo.status()
            result = dict(self.suite_runner(self.repo_root))
            result.update(head_before=head, head_after=self.repo.current_head(),
                          worktree_before=status, worktree_after=self.repo.status())
            return result
        return self._once(("suite",), run)

    def phase4(self) -> dict[str, Any]:
        return self._once(("phase4",), lambda: dict(self.phase4_check(self.cfg, self.repo_root)))


def _group_order(item: tuple) -> tuple:
    (context, target, region), _ = item
    return (str(context), str(target), REGIONS.index(region) if region in REGIONS else 99,
            str(region))


# ---------------------------------------------------------------------------
# One condition's accumulated result
# ---------------------------------------------------------------------------

class _Check:
    """Problems, notes, and evidence, gathered part by part.

    A part that raises fails the condition with what it was checking, and the
    next part still runs.
    """

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.notes: list[str] = []
        self.evidence: dict[str, Any] = {}

    def fail(self, message: str) -> None:
        self.problems.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def mark(self) -> int:
        """How many problems are on record now, to tell later whether a part added one."""
        return len(self.problems)

    def note_if_clean(self, mark: int, message: str) -> None:
        """A note that something held, written only when no problem followed mark."""
        if len(self.problems) == mark:
            self.notes.append(message)

    @contextlib.contextmanager
    def part(self, what: str) -> Iterator[None]:
        try:
            yield
        except Unavailable as error:
            self.fail(f"{what}: {error}")
        except Exception as error:  # noqa: BLE001
            # An unexpected failure fails the condition. It never passes it.
            self.fail(f"{what}: the check raised {type(error).__name__}: {error}")

    def result(self) -> tuple[bool, list[str], dict[str, Any]]:
        shown = [f"FAIL {problem}" for problem in self.problems[:SHOWN_PROBLEMS]]
        if len(self.problems) > SHOWN_PROBLEMS:
            shown.append(f"FAIL and {len(self.problems) - SHOWN_PROBLEMS} more problems, "
                         "listed in the evidence")
        self.evidence["problems"] = list(self.problems)
        return not self.problems, shown + self.notes, self.evidence


def _suite_files(ctx: AcceptanceContext, check: _Check, number: int) -> None:
    """The suite's results at R for the test files that cover one condition.

    The results count only for the tests registered at E. Tests are neutral to
    code ancestry, so a file hollowed after E would still pass at R. Each named
    file and each module of SUITE_SUPPORT must therefore exist at E and be
    byte-identical at R, or be named with its reason in
    post_evaluation_changes.md. Their sha256 at E and at R, and any reason,
    go into the evidence.
    """
    names = SUITE_FILES.get(number, ())
    if not names:
        return
    with check.part(f"the suite's results for {', '.join(names)}"):
        files = ctx.suite().get("files") or {}
        results = {}
        mark = check.mark()
        for name in names:
            counts = files.get(name)
            results[name] = counts
            if not isinstance(counts, Mapping):
                check.fail(f"the suite at R reports no result for {name}")
            elif counts.get("failed") or counts.get("errors"):
                check.fail(f"{name} has {counts.get('failed')} failed and "
                           f"{counts.get('errors')} errored tests at R")
            elif not counts.get("passed"):
                check.fail(f"no test of {name} passed at R")
        check.evidence["suite_files"] = results
        check.note_if_clean(mark, f"the suite at R passes {', '.join(names)}")
    with check.part(f"{', '.join(names)} against E"):
        mark = check.mark()
        commit, head = ctx.commit_e(), ctx.repo.head()
        listed = post_evaluation_listing(ctx.repo.text(head, POST_EVALUATION_CHANGES) or "")
        hashes: dict[str, dict[str, Any]] = {}
        for path in dict.fromkeys(names + SUITE_SUPPORT):
            at_e = _sha256(ctx.repo.blob(commit, path))
            at_r = _sha256(ctx.repo.blob(head, path))
            hashes[path] = {"at_e": at_e, "at_r": at_r, "reason": listed.get(path)}
            if at_e is None:
                check.fail(f"{path} does not exist at E {commit}, so its results at R cannot "
                           "stand for the tests registered at E")
            elif at_e != at_r and not listed.get(path):
                check.fail(f"{path} changed since E {commit} and is not named with its reason "
                           f"in {POST_EVALUATION_CHANGES}. Its results at R are evidence for "
                           "this condition only as the tests registered at E")
        check.evidence["suite_file_sha256"] = hashes
        changed = sorted(path for path, entry in hashes.items()
                         if entry["at_e"] is not None and entry["at_e"] != entry["at_r"])
        check.note_if_clean(mark, "the suite's evidence files are E's own" + (
            f", except {changed}, each named with its reason" if changed else ""))


def _evaluate_scene(ctx: AcceptanceContext) -> ast.FunctionDef:
    commit = ctx.commit_e()
    tree = _parse(ctx.repo.text(commit, MODES_SOURCE), f"{MODES_SOURCE} at E")
    return _function(tree, "evaluate_scene")


def _model_calls(function: ast.FunctionDef) -> list[ast.Call]:
    calls = _calls(function, "model", plain=True)
    if not calls:
        raise Unavailable("evaluate_scene calls no model")
    return calls


def _headline_lift(function: ast.FunctionDef) -> ast.Call:
    lifts = _bindings(function, "lift")
    if len(lifts) != 1:
        raise Unavailable(f"evaluate_scene binds lift {len(lifts)} times, not once")
    value = lifts[0][1]
    if _call_name(value) != "context_lift_map":
        raise Unavailable(f"evaluate_scene's lift is {ast.unparse(value)}, not a "
                          "context_lift_map call")
    return value


# ---------------------------------------------------------------------------
# The conditions
# ---------------------------------------------------------------------------

Outcome = tuple[bool, list[str], dict[str, Any]]


def check_01_phase4_accepted(ctx: AcceptanceContext) -> Outcome:
    """The Phase 4 acceptance check passes on the Phase 4 run the evaluation read."""
    from .evaluate import read_run_metadata

    check = _Check()
    with check.part("the Phase 4 acceptance check"):
        result = ctx.phase4()
        check.evidence["phase4_acceptance_check"] = result
        code = result.get("returncode")
        if code == 0:
            check.note(f"{PHASE4_CHECK_SCRIPT} passed on the Phase 4 run")
        else:
            check.fail(f"{PHASE4_CHECK_SCRIPT} exited {code} on the Phase 4 run")
        for line in [line for line in result.get("stdout") or [] if line.strip()][-2:]:
            check.note(f"  {line.strip()}")
    phase4_eval = Path(ctx.cfg.phase4_dir) / "eval"
    with check.part("the Phase 4 parquets"):
        mark = check.mark()
        records = ctx.records()
        identities = receipt_scene_identities(ctx.integration())
        table = {}
        for scene, record in records.items():
            path = phase4_eval / f"{scene}.parquet"
            live = sha256_file(path) if path.is_file() else None
            gate = (identities.get(scene) or {}).get("phase4_parquet_sha256")
            named = record.get("phase4_parquet_sha256")
            table[scene] = {"evaluation": named, "gate": gate, "live": live}
            if not (named is not None and named == gate == live):
                check.fail(f"{scene}: the evaluation reconciled against Phase 4 parquet {named}, "
                           f"the gate verified {gate}, and the live parquet is {live}")
        check.evidence["phase4_parquets"] = table
        check.note_if_clean(mark, f"{len(table)} evaluation records name the Phase 4 parquet "
                                  "the gate verified, and it is unchanged")
    with check.part("the Phase 4 commit"):
        mark = check.mark()
        records = ctx.records()
        identities = receipt_scene_identities(ctx.integration())
        commits = {}
        for scene, record in records.items():
            path = phase4_eval / f"{scene}.parquet"
            live = (read_run_metadata(path) or {}).get("git_commit") if path.is_file() else None
            gate = (identities.get(scene) or {}).get("phase4_commit")
            commits[scene] = {"evaluation": record.get("phase4_commit"), "phase4_run": live,
                              "gate": gate}
            if not (record.get("phase4_commit") is not None
                    and record.get("phase4_commit") == live == gate):
                check.fail(f"{scene}: the evaluation names Phase 4 commit "
                           f"{record.get('phase4_commit')}, the Phase 4 run record names {live}, "
                           f"and the gate recorded {gate}")
        check.evidence["phase4_commit"] = commits
        check.note_if_clean(mark, "every evaluation names the Phase 4 commit its Phase 4 run "
                                  "records and the gate recorded")
    with check.part("the documentary record"):
        text = ctx.repo.text(ctx.commit_e(), FINDINGS_DOCUMENT) or ""
        if "Phase 4 accepted" in text:
            check.note(f"{FINDINGS_DOCUMENT} at E records that Phase 4 was accepted")
        else:
            check.note(f"{FINDINGS_DOCUMENT} at E does not state that Phase 4 was accepted; the "
                       "rerun check is the evidence")
    return check.result()


def check_02_preregistration(ctx: AcceptanceContext) -> Outcome:
    """The records made before outcomes exist at E, unchanged, and the reporting
    code that made the outputs is the pre-registered code."""
    check = _Check()
    with check.part("the pre-registration records"):
        mark = check.mark()
        commit = ctx.commit_e()
        reachable = ctx.repo.ancestors(commit)
        documents = {}
        for path in PREREGISTRATION_DOCUMENTS:
            blob = ctx.repo.blob(commit, path)
            touching = ctx.repo.commits_touching(path)
            late = [c for c in touching if c not in reachable]
            documents[path] = {"sha256_at_e": _sha256(blob), "commits": touching}
            if blob is None:
                check.fail(f"{path} does not exist at E {commit}, so it was not recorded before "
                           "the test outcomes")
            if late:
                check.fail(f"{path} was changed after E by {late}. A record made before the "
                           "test outcomes is not edited after them")
        check.evidence["documents"] = documents
        check.note_if_clean(mark, f"{len(documents)} pre-registration records exist at E, and "
                                  "no commit after E touched them")
    with check.part("the frozen class and the reporting changes since E"):
        mark = check.mark()
        for problem in ctx.ancestry_problems(("frozen_at_e", "classification", "reporting_paths",
                                              "commit", "worktree")):
            check.fail(problem)
        commit = ctx.commit_e()
        check.evidence["frozen_at_e"] = {path: _sha256(ctx.repo.blob(commit, path))
                                         for path in FROZEN_AT_E}
        changes = ctx.ancestry()["changes"]
        check.evidence["reporting_changes"] = changes
        check.note_if_clean(mark, f"the frozen class is unchanged since E, and {len(changes)} "
                                  "other reporting change(s) are named with their reasons")
    with check.part("the code that built the published outputs"):
        mark = check.mark()
        head = ctx.repo.head()
        built: dict[str, Any] = {}
        for name, read in (("tables", lambda: ctx.published().run_record),
                           ("figures", ctx.figures_record)):
            record = read()
            commit = record.get("report_commit")
            built[name] = commit
            if record.get("code_checked") is not True:
                check.fail(f"the {name} were built without checking the code that built them")
            elif not (isinstance(commit, str) and FULL_COMMIT.fullmatch(commit)):
                check.fail(f"the {name} name no reporting commit: {commit!r}")
            elif not ctx.repo.is_ancestor(commit, head):
                check.fail(f"the {name} were built at {commit}, which is not an ancestor of R "
                           f"{head}")
            else:
                loose = [path for path in ctx.repo.changed(commit, head)
                         if classify_path(path) != NEUTRAL]
                if loose:
                    check.fail(f"the {name} were built at {commit}, and {loose} changed since. "
                               "Rebuild them at R")
        check.evidence["outputs_built_at"] = built
        check.note_if_clean(mark, f"the tables and figures were built by the reporting code at "
                                  f"R {head}")
    return check.result()


def check_03_context_lift_no_target_depth(ctx: AcceptanceContext) -> Outcome:
    """The comparator cannot be handed target depth, and the headline lift is not."""
    check = _Check()
    _suite_files(ctx, check, 3)
    with check.part(f"{CONTEXT_LIFT_SOURCE} at E"):
        mark = check.mark()
        tree = _parse(ctx.repo.text(ctx.commit_e(), CONTEXT_LIFT_SOURCE),
                      f"{CONTEXT_LIFT_SOURCE} at E")
        function = _function(tree, "context_lift_map")
        parameters = _parameters(function)
        check.evidence["context_lift_parameters"] = parameters
        if tuple(parameters) != CONTEXT_LIFT_PARAMETERS:
            check.fail(f"context_lift_map takes {parameters}, not the frozen "
                       f"{list(CONTEXT_LIFT_PARAMETERS)}")
        names = _identifiers(function)
        offenders = sorted({name for name in names for bad in FORBIDDEN_IN_TRANSPORT
                            if bad in name})
        if offenders:
            check.fail(f"context_lift_map references target content {offenders}")
        support = sorted(names & {"visibility_masks", "covisible"})
        if support:
            check.fail(f"context_lift_map reaches the ground-truth support machinery {support}")
        depth = sorted(name for name in names if "depth" in name
                       and name not in TRANSPORT_DEPTH_NAMES)
        if depth:
            check.fail(f"context_lift_map names depth {depth}; it may read the aligned context "
                       "map only")
        check.note_if_clean(mark, "context_lift_map at E takes no target-depth parameter and "
                                  "references no target content")
    with check.part("the headline lift in evaluate_scene at E"):
        mark = check.mark()
        function = _evaluate_scene(ctx)
        lift = _headline_lift(function)
        depth = _first_argument(lift, "depth_context_aligned")
        names = _names(depth) if depth is not None else set()
        check.evidence["headline_lift_depth"] = None if depth is None else ast.unparse(depth)
        if "context_depth" not in names or names - {"torch", "context_depth", "dtype"}:
            check.fail(f"the headline Context-Lift reads depth from {sorted(names)}, not from "
                       "context_depth alone")
        bound = _bindings(function, "context_depth")
        if len(bound) != 1:
            check.fail(f"context_depth is bound {len(bound)} times in evaluate_scene")
        else:
            value = bound[0][1]
            frame = _argument(value, 1, "context_frame_id") if isinstance(value, ast.Call) \
                else None
            check.evidence["context_depth"] = ast.unparse(value)
            if _call_name(value) != "aligned_context_depth" or not (
                    isinstance(frame, ast.Name) and frame.id == "ctx"):
                check.fail(f"context_depth is {ast.unparse(value)}, not the aligned depth of the "
                           "context frame")
        frames = _bindings(function, "ctx")
        if not (len(frames) == 1 and isinstance(frames[0][1], ast.Attribute)
                and frames[0][1].attr == "context_frame_id"):
            check.fail("ctx is not bound once, to the pair's context frame")
        check.note_if_clean(mark, "the headline lift reads the aligned depth of the context "
                                  "frame only")
    with check.part("integration gate steps 6 and 10"):
        mark = check.mark()
        step6 = ctx.gate_step("6")
        difference = step6.get("max_abs_coordinate_difference")
        if difference != 0.0:
            check.fail(f"gate step 6 found Context-Lift and the predictor read {difference} px "
                       "apart")
        step10 = ctx.gate_step("10")
        residual = (step10.get("translation") or {}).get(
            "max_independent_reprojection_residual_px")
        tolerance = ctx.analysis.rotation_gate_coord_tol_px
        if not (_finite(residual) and residual <= tolerance):
            check.fail(f"gate step 10's independent reprojection residual is {residual}, not "
                       f"within {tolerance} px")
        check.evidence["gate"] = {"step6_difference": difference,
                                  "step10_translation_residual": residual}
        check.note_if_clean(mark, f"gate steps 6 and 10 passed: one landing for both methods, "
                                  f"and an independent reprojection within {residual} px")
    return check.result()


def check_04_predictor_no_target(ctx: AcceptanceContext) -> Outcome:
    """The predictor can be handed only context and camera content, and is."""
    check = _Check()
    _suite_files(ctx, check, 4)
    start = check.mark()
    with check.part(f"{PREDICTORS_SOURCE} at E"):
        tree = _parse(ctx.repo.text(ctx.commit_e(), PREDICTORS_SOURCE),
                      f"{PREDICTORS_SOURCE} at E")
        forward = _function(tree, "forward", owner="PredictWithDepth")
        parameters = [name for name in _parameters(forward) if name != "self"]
        check.evidence["forward_parameters"] = parameters
        if tuple(parameters) != FORWARD_PARAMETERS:
            check.fail(f"PredictWithDepth.forward takes {parameters}, not the frozen "
                       f"{list(FORWARD_PARAMETERS)}")
        names = _identifiers(tree)
        offenders = sorted({name for name in names for bad in FORBIDDEN_PREDICTOR_INPUTS
                            if bad in name})
        if offenders:
            check.fail(f"{PREDICTORS_SOURCE} references target content {offenders}")
        operations = sorted(names & set(FORBIDDEN_PREDICTOR_OPERATIONS))
        if operations:
            check.fail(f"{PREDICTORS_SOURCE} performs explicit geometry {operations}")
        imports = _local_imports(tree)
        if imports != PREDICTOR_LOCAL_IMPORTS:
            check.fail(f"{PREDICTORS_SOURCE} imports {imports} from lot, not only the patch size")
    with check.part(f"model_inputs in {PHASE5_SOURCE} at E"):
        tree = _parse(ctx.repo.text(ctx.commit_e(), PHASE5_SOURCE), f"{PHASE5_SOURCE} at E")
        function = _function(tree, "model_inputs")
        returned = [node.value for node in ast.walk(function)
                    if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)]
        if len(returned) != 1:
            raise Unavailable("model_inputs does not return one dict literal")
        keys = [key.value for key in returned[0].keys if isinstance(key, ast.Constant)]
        check.evidence["model_inputs_keys"] = keys
        if sorted(keys) != sorted(FORWARD_PARAMETERS):
            check.fail(f"model_inputs builds {keys}, not the forward inputs "
                       f"{list(FORWARD_PARAMETERS)}")
        names = _identifiers(function)
        offenders = sorted({name for name in names for bad in FORBIDDEN_PREDICTOR_INPUTS
                            if bad in name} | (names & {"target_frame_id", "depth_path"}))
        if offenders:
            check.fail(f"model_inputs reads target content {offenders}")
    with check.part("what evaluate_scene hands the model at E"):
        function = _evaluate_scene(ctx)
        visible = _bindings(function, "visible")
        if not (len(visible) == 1 and _call_name(visible[0][1]) == "model_inputs"):
            check.fail("visible is not bound once, to model_inputs")
        used: set[str] = set()
        for call in _model_calls(function):
            for argument in list(call.args) + [item.value for item in call.keywords]:
                stray = _names(argument) - {"visible", "device"}
                keys = [node.slice.value for node in ast.walk(argument)
                        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                        and node.value.id == "visible" and isinstance(node.slice, ast.Constant)]
                if stray or len(keys) != 1:
                    check.fail(f"the model receives {ast.unparse(argument)}, not one entry of "
                               "visible")
                used.update(keys)
        check.evidence["model_receives"] = sorted(used)
        if used != set(FORWARD_PARAMETERS):
            check.fail(f"the model receives {sorted(used)}, not {list(FORWARD_PARAMETERS)}")
    with check.part("integration gate steps 5 and 12"):
        mark = check.mark()
        step5 = ctx.gate_step("5")
        fields = step5.get("model_visible_fields")
        if list(fields or []) != list(FORWARD_PARAMETERS):
            check.fail(f"gate step 5 found the model sees {fields}")
        for regime, entry in sorted((step5.get("per_regime") or {}).items()):
            leaked = sorted({name for name in (entry or {}).get("fields") or []
                             for bad in FORBIDDEN_BATCH_FIELDS if bad in name})
            if leaked:
                check.fail(f"gate step 5's {regime} example carries {leaked}")
        ctx.gate_step("12")
        check.note_if_clean(mark, "gate steps 5 and 12 passed: the real batch carries no "
                                  "forbidden field")
    check.note_if_clean(start, "the predictor at E takes context features, context depth, the "
                               "camera vector, and context validity, and evaluate_scene hands "
                               "it nothing else")
    return check.result()


def check_05_rotation_and_geometry(ctx: AcceptanceContext) -> Outcome:
    """Gate step 17 checked the whole pure-rotation regime, and step 10 the probe pairs."""
    check = _Check()
    _suite_files(ctx, check, 5)
    tolerance = ctx.analysis.rotation_gate_coord_tol_px
    bound = ctx.analysis.rotation_position_bound_m
    with check.part("integration gate step 17"):
        evidence = ctx.gate_step("17")
        scenes = evidence.get("scenes") if isinstance(evidence.get("scenes"), dict) else {}
        missing = [scene for scene in REPLICA_SCENES if scene not in scenes]
        extra = sorted(set(scenes) - set(REPLICA_SCENES))
        if missing:
            check.fail(f"step 17 holds no rotation evidence for {missing}")
        if extra:
            check.fail(f"step 17 holds rotation evidence for unknown scenes {extra}")
        if evidence.get("n_scenes") != len(scenes):
            check.fail(f"step 17 counts {evidence.get('n_scenes')} scenes and holds {len(scenes)}")
        if evidence.get("tolerance_px") != tolerance:
            check.fail(f"step 17 ran at tolerance {evidence.get('tolerance_px')}, not {tolerance}")
        if evidence.get("rotation_position_bound_m") != bound:
            check.fail(f"step 17 ran at position bound {evidence.get('rotation_position_bound_m')}"
                       f", not {bound}")
        if evidence.get("reading") != ROTATION_READING:
            check.fail("step 17 records another reading of the comparison")
        total = sum(int((entry or {}).get("n_rotation_pairs") or 0) for entry in scenes.values())
        if evidence.get("n_rotation_pairs") != total or total == 0:
            check.fail(f"step 17 counts {evidence.get('n_rotation_pairs')} rotation pairs, and its "
                       f"scenes hold {total}")
        norm = evidence.get("max_translation_norm_m")
        if norm is not None and not (_finite(norm) and norm <= bound):
            check.fail(f"step 17's largest rotation-pair translation {norm} m exceeds {bound} m")
        parts = ("n_pairs_checked", "n_pairs_context_lift_only", "n_pairs_tl_reference_only",
                 "n_pairs_nothing_landed", "n_pairs_no_arm")
        for scene, entry in sorted(scenes.items()):
            entry = entry if isinstance(entry, dict) else {}
            n = entry.get("n_rotation_pairs")
            if not _count(n) or n != sum(int(entry.get(name) or 0) for name in parts):
                check.fail(f"{scene}: step 17's rotation pairs do not partition into checked, "
                           "one estimator, nothing landed, and no arm")
            elif n and not entry.get("n_pairs_checked"):
                check.fail(f"{scene}: step 17 checked none of its {n} rotation pairs")
            worst = entry.get("worst") if isinstance(entry.get("worst"), dict) else {}
            for name in ROTATION_COMPARISONS:
                item = worst.get(name) if isinstance(worst.get(name), dict) else {}
                if item.get("max_residual_px") is not None and not (
                        _finite(item.get("min_headroom_px")) and item["min_headroom_px"] >= 0):
                    check.fail(f"{scene}: step 17's {name} residual is beyond its allowance")
        summary = {name: (evidence.get("worst") or {}).get(name) for name in ROTATION_COMPARISONS}
        check.evidence["step17"] = {
            "n_scenes": len(scenes), "n_rotation_pairs": total,
            **{name: evidence.get(name) for name in parts}, "worst": summary,
            "max_translation_norm_m": norm, "tolerance_px": evidence.get("tolerance_px")}
        check.note(f"gate step 17: {len(scenes)} scenes, {total} rotation pairs, "
                   f"{evidence.get('n_pairs_checked')} checked on both estimators; pairs with "
                   "nothing landed are counted and have no sample to compare")
        for name in ROTATION_COMPARISONS:
            worst = summary.get(name) or {}
            check.note(f"  worst {name} residual {worst.get('max_residual_px')} px, smallest "
                       f"headroom {worst.get('min_headroom_px')} px")
    with check.part("the rotation pairs the evaluation read"):
        mark = check.mark()
        evidence = ctx.gate_step("17")
        scenes = evidence.get("scenes") if isinstance(evidence.get("scenes"), dict) else {}
        facts = ctx.rows()
        records = ctx.records()
        table = {}
        for scene in facts.scenes:
            gate = (scenes.get(scene) or {}).get("n_rotation_pairs")
            evaluated = facts.rotation_pairs.get(scene, 0)
            no_arm = sum(1 for item in (records[scene].get("audit") or {}).get("no_arm_pairs")
                         or [] if len(item) > 2 and item[2] == "rotation")
            table[scene] = {"gate": gate, "evaluated": evaluated, "no_arm": no_arm}
            if gate != evaluated + no_arm:
                check.fail(f"{scene}: step 17 checked {gate} rotation pairs, and the evaluation "
                           f"read {evaluated} with {no_arm} more without an arm")
        check.evidence["rotation_pairs"] = table
        evaluated = sum(entry["evaluated"] for entry in table.values())
        check.note_if_clean(mark, f"every evaluated scene's rotation pairs, {evaluated} in all, "
                                  "are the pairs step 17 checked")
    with check.part("integration gate step 10"):
        mark = check.mark()
        step = ctx.gate_step("10")
        rotation = step.get("rotation") or {}
        translation = step.get("translation") or {}
        for name, value in (
            ("homography residual", rotation.get("max_homography_residual_px")),
            ("depth substitution shift", rotation.get("max_depth_substitution_shift_px")),
            ("translation reprojection residual",
             translation.get("max_independent_reprojection_residual_px")),
        ):
            if not (_finite(value) and value <= tolerance):
                check.fail(f"gate step 10's {name} is {value}, not within {tolerance} px")
        check.evidence["step10"] = step
        check.note_if_clean(mark, f"gate step 10's probe pairs agree with the homography and "
                                  f"an independent reprojection within {tolerance} px")
    with check.part("the gate's step order"):
        mark = check.mark()
        order = ctx.gate_order()
        check.evidence["step_order"] = order
        if not ("14" in order and "17" in order and "15" in order
                and order.index("14") < order.index("17") < order.index("15")):
            check.fail(f"step 17 did not run after step 14 and before the pin of step 15: {order}")
        if not order or order[-1] != "16":
            check.fail(f"the verdict step 16 is not last: {order}")
        check.note_if_clean(mark, "step 17 ran after step 14 and before the pin of step 15")
    return check.result()


def check_06_identical_inputs(ctx: AcceptanceContext) -> Outcome:
    """One context depth and one set of cameras feed both headline methods."""
    check = _Check()
    _suite_files(ctx, check, 6)
    with check.part("evaluate_scene at E"):
        mark = check.mark()
        function = _evaluate_scene(ctx)
        lift = _headline_lift(function)
        depth = _first_argument(lift, "depth_context_aligned")
        depth_names = (_names(depth) - {"torch", "dtype"}) if depth is not None else set()
        cameras = {node.value.id for argument in lift.args + [k.value for k in lift.keywords]
                   for node in ast.walk(argument)
                   if isinstance(node, ast.Attribute) and node.attr in CAMERA_FIELDS
                   and isinstance(node.value, ast.Name)}
        fields = {node.attr for argument in lift.args + [k.value for k in lift.keywords]
                  for node in ast.walk(argument)
                  if isinstance(node, ast.Attribute) and node.attr in CAMERA_FIELDS}
        visible = _bindings(function, "visible")
        if not (len(visible) == 1 and isinstance(visible[0][1], ast.Call)):
            raise Unavailable("visible is not bound once, to a call")
        call = visible[0][1]
        given_cameras = _argument(call, 3, "cams")
        given_depth = _argument(call, 4, "context_depth")
        check.evidence["lift"] = {"depth": sorted(depth_names), "cameras": sorted(cameras),
                                  "camera_fields": sorted(fields)}
        check.evidence["model_inputs"] = ast.unparse(call)
        if len(depth_names) != 1 or not (isinstance(given_depth, ast.Name)
                                         and {given_depth.id} == depth_names):
            given = ast.unparse(given_depth) if given_depth is not None else None
            check.fail(f"model_inputs receives depth {given}, and the headline lift reads "
                       f"{sorted(depth_names)}")
        if len(cameras) != 1 or not (isinstance(given_cameras, ast.Name)
                                     and {given_cameras.id} == cameras):
            check.fail(f"model_inputs receives cameras "
                       f"{ast.unparse(given_cameras) if given_cameras else None}, and the headline "
                       f"lift reads {sorted(cameras)}")
        if fields != set(CAMERA_FIELDS):
            check.fail(f"the headline lift reads camera fields {sorted(fields)}, not "
                       f"{list(CAMERA_FIELDS)}")
        for name in cameras:
            bound = _bindings(function, name)
            if not (len(bound) == 1 and _call_name(bound[0][1]) == "pair_cameras"):
                check.fail(f"{name} is not bound once, to pair_cameras")
        check.note_if_clean(mark, "evaluate_scene at E hands model_inputs the depth and the "
                                  "cameras the headline lift reads")
    with check.part(f"model_inputs in {PHASE5_SOURCE} at E"):
        tree = _parse(ctx.repo.text(ctx.commit_e(), PHASE5_SOURCE), f"{PHASE5_SOURCE} at E")
        function = _function(tree, "model_inputs")
        parameters = _parameters(function)
        if len(parameters) < 5:
            raise Unavailable(f"model_inputs takes {parameters}")
        cams, depth = parameters[3], parameters[4]
        returned = [node.value for node in ast.walk(function)
                    if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)]
        entries = {key.value: value for key, value in zip(returned[0].keys, returned[0].values)
                   if isinstance(key, ast.Constant)} if returned else {}
        aligned = entries.get("depth_context_aligned")
        camera = entries.get("camera")
        if aligned is None or depth not in _names(aligned):
            check.fail(f"model_inputs does not build depth_context_aligned from {depth}")
        used = {node.attr for node in ast.walk(camera) if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name) and node.value.id == cams} if camera else set()
        if not set(CAMERA_FIELDS) <= used:
            check.fail(f"model_inputs builds the camera vector from {sorted(used)}, not every "
                       f"camera quantity {list(CAMERA_FIELDS)}")
    with check.part("each scene's aligned context depth against gate step 4"):
        mark = check.mark()
        step = ctx.gate_step("4")
        scenes = step.get("scenes") if isinstance(step.get("scenes"), dict) else {}
        table = {}
        for scene, record in ctx.records().items():
            named = record.get("aligned_depth_digest")
            gate = (scenes.get(scene) or {}).get("aligned_depth_digest")
            table[scene] = {"evaluation": named, "gate": gate}
            if not (isinstance(named, str) and named == gate):
                check.fail(f"{scene}: the evaluation used aligned context depth {named}, and gate "
                           f"step 4 computed {gate}")
        check.evidence["aligned_depth"] = table
        check.note_if_clean(mark, f"{len(table)} scenes were evaluated on the aligned context "
                                  "depth gate step 4 computed")
    return check.result()


def check_07_scene_separation(ctx: AcceptanceContext) -> Outcome:
    """Every scene is test once, evaluated by the fold that held it out, and trained
    on by no model that scores it."""
    check = _Check()
    _suite_files(ctx, check, 7)
    with check.part("the folds"):
        assert_scene_separation(ctx.folds)
        digest = fold_digest(ctx.folds)
        frozen = [(f.index, f.train, f.val, f.test) for f in frozen_folds()]
        if digest != FROZEN_FOLD_DIGEST:
            check.fail(f"the folds have digest {digest}, not the frozen {FROZEN_FOLD_DIGEST}")
        if [(f.index, f.train, f.val, f.test) for f in ctx.folds] != frozen:
            check.fail("the folds are not the frozen folds")
        check.evidence["fold_digest"] = digest
    with check.part("the evaluated scenes"):
        mark = check.mark()
        scan = ctx.scan()
        for problem in scan.problems:
            check.fail(problem)
        for scene, record in scan.records.items():
            where = scan.files[scene].name
            try:
                held_out_by = fold_of_test_scene(scene, ctx.folds).index
            except ValueError:
                held_out_by = None
            for field, want in (("phase", 5), ("scene", scene), ("level", ctx.level),
                                ("fold", held_out_by), ("eval_version", PHASE5_EVAL_VERSION),
                                ("fold_digest", FROZEN_FOLD_DIGEST)):
                if record.get(field) != want:
                    check.fail(f"{where}: its run record names {field} {record.get(field)!r}, "
                               f"not {want!r}")
            seeds = record.get("seeds") if isinstance(record.get("seeds"), list) else []
            if sorted((record.get("checkpoints") or {}).keys()) != sorted(str(s) for s in seeds):
                check.fail(f"{where}: its checkpoints are not keyed by its seeds")
        check.evidence["evaluated"] = {scene: record.get("fold")
                                       for scene, record in scan.records.items()}
        check.note_if_clean(mark, f"all {len(ctx.expected_scenes)} evaluation scenes are "
                                  "evaluated, each by the fold that held it out")
    with check.part("the receipts' and the controls' fold assignment"):
        for kind in (KIND_INTEGRATION, KIND_OVERFIT, KIND_LOCK):
            named = receipt_identity(ctx.receipt(kind).payload).get("fold_digest")
            if named != FROZEN_FOLD_DIGEST:
                check.fail(f"the {ctx.receipt_stems()[kind]} receipt names fold digest {named}")
        controls = ctx.controls()
        if controls.payload is None or controls.payload.get("fold_digest") != FROZEN_FOLD_DIGEST:
            check.fail(f"the controls file names fold digest "
                       f"{(controls.payload or {}).get('fold_digest')}")
    with check.part("integration gate step 11"):
        step = ctx.gate_step("11")
        listed = [(item.get("index"), tuple(item.get("train") or ()), tuple(item.get("val") or ()),
                   tuple(item.get("test") or ())) for item in step.get("folds") or []]
        if listed != [(f.index, f.train, f.val, f.test) for f in frozen_folds()]:
            check.fail("gate step 11 checked folds other than the frozen folds")
        if step.get("fold_digest") != FROZEN_FOLD_DIGEST:
            check.fail(f"gate step 11 names fold digest {step.get('fold_digest')}")
        if step.get("n_scenes") != len(REPLICA_SCENES):
            check.fail(f"gate step 11 found {step.get('n_scenes')} scenes on disk, not "
                       f"{len(REPLICA_SCENES)}")
    with check.part("each training record's roles"):
        mark = check.mark()
        for key, run in ctx.training().items():
            fold, seed = ctx.fold_of(key)
            record = run.payload if isinstance(run.payload, dict) else None
            if record is None:
                check.fail(f"{key}: the training record cannot be read: {run.error}")
                continue
            for field, want in (("fold", fold.index), ("seed", seed), ("level", ctx.level),
                                ("train_scenes", list(fold.train)), ("val_scenes", list(fold.val)),
                                ("test_scenes", list(fold.test))):
                if record.get(field) != want:
                    check.fail(f"{key}: the training record names {field} "
                               f"{_short(record.get(field))}, not {_short(want)}")
        check.note_if_clean(mark, f"{len(ctx.keys)} training records name their fold's roles")
    with check.part("the scenes each role actually planned from"):
        mark = check.mark()
        for key, run in ctx.training().items():
            fold, _ = ctx.fold_of(key)
            record = run.payload if isinstance(run.payload, dict) else {}
            for role in ("train", "val"):
                planned = record.get(f"{role}_scenes_planned")
                allowed = set(getattr(fold, role))
                if not (isinstance(planned, list) and planned):
                    check.fail(f"{key}: the training record names no {role} scenes it planned "
                               "from")
                elif not set(planned) <= allowed:
                    check.fail(f"{key}: {role} planned from {sorted(set(planned) - allowed)}, "
                               f"outside its {role} role"
                               + (", test scenes of its fold"
                                  if set(planned) & set(fold.test) else ""))
        controls = ctx.controls().payload or {}
        for key in ctx.keys:
            fold, _ = ctx.fold_of(key)
            result = (controls.get("results") or {}).get(key) or {}
            planned = result.get("val_scenes_planned")
            if not (isinstance(planned, list) and planned and set(planned) <= set(fold.val)):
                outside = sorted(set(planned or []) - set(fold.val))
                check.fail(f"{key}: the controls planned from {outside or planned}, not from "
                           "its validation scenes")
        check.note_if_clean(mark, "training, checkpoint selection, and the controls planned "
                                  "from their own roles only, so no model saw a scene it scores")
    return check.result()


def check_08_test_seal(ctx: AcceptanceContext) -> Outcome:
    """Nothing about a test scene reached training or selection, the models were
    locked before evaluation, and each scene was evaluated once."""
    check = _Check()
    _suite_files(ctx, check, 8)
    with check.part("integration gate step 14"):
        mark = check.mark()
        step = ctx.gate_step("14")
        for field in ("test_rejected", "train_accepts_train", "selection_accepts_val"):
            if step.get(field) is not True:
                check.fail(f"gate step 14 records {field} {step.get(field)!r}")
        check.note_if_clean(mark, "gate step 14 proved a test scene reaching training raises")
    with check.part("the lock, verified against the live files"):
        mark = check.mark()
        lock = ctx.receipt(KIND_LOCK).payload
        for problem in ctx.verification(KIND_LOCK):
            check.fail(problem)
        for group in ("checkpoints", "training_records"):
            if sorted((lock.get(group) or {}).keys()) != sorted(ctx.keys):
                check.fail(f"the lock binds the {group} {sorted((lock.get(group) or {}).keys())}, "
                           f"not {sorted(ctx.keys)}")
        check.note_if_clean(mark, f"the lock binds {len(ctx.keys)} checkpoints, their training "
                                  "records, and the controls file, and every one still has the "
                                  "bound sha256")
    with check.part("the checkpoint chain"):
        mark = check.mark()
        _checkpoint_chain(ctx, check)
        check.note_if_clean(mark, "every evaluated model is the locked, trained, selected "
                                  "checkpoint, and every run record embeds the locked records")
    with check.part("the licences"):
        mark = check.mark()
        licence = ctx.licence()
        gates = {stem: licence.get(stem) for stem in ("integration_gate", "tiny_overfit")}
        for key, run in ctx.training().items():
            ran_under = (run.payload or {}).get("licence")
            if ran_under != gates:
                check.fail(f"{key}: training ran under {_short(ran_under)}, not under the "
                           "receipts the run names")
        if (ctx.controls().payload or {}).get("licence") != gates:
            check.fail("the controls ran under other receipts than the run names")
        lock = ctx.receipt(KIND_LOCK).payload
        if lock.get(GATE_RECEIPT_DIGEST) != gates["integration_gate"] or \
                lock.get(OVERFIT_RECEIPT_DIGEST) != gates["tiny_overfit"]:
            check.fail("the lock binds other gate receipts than the run names")
        check.evidence["licence"] = licence
        check.note_if_clean(mark, "training, the controls, the lock, and every evaluation ran "
                                  "under the receipts the run names")
    with check.part("the order the artifacts were written in"):
        _timeline(ctx, check)
    with check.part("the evaluation ledger"):
        attempts = ctx.attempts()
        scan = ctx.scan()
        problems, evidence = ledger_problems(
            attempts, level=ctx.level, scenes=ctx.expected_scenes, live=scan.sha,
            written_utc={scene: record.get("written_utc")
                         for scene, record in scan.records.items()},
            commit=ctx.commit_e(), licence=ctx.licence())
        for problem in problems:
            check.fail(problem)
        check.evidence["ledger"] = evidence
        if not problems:
            check.note(f"the ledger holds {evidence['attempts_at_level']} attempt(s) at level "
                       f"{ctx.level}: one evaluation per scene"
                       + (f", and {len(evidence['explained_errors'])} error attempt(s) that wrote "
                          "nothing, each with its message" if evidence["explained_errors"]
                          else ""))
    with check.part("other evaluation records under the run directory"):
        others, unreadable = _other_evaluations(ctx)
        for path in others:
            check.fail(f"{path} holds another Phase 5 evaluation at level {ctx.level}")
        for path in unreadable:
            check.fail(f"{path} cannot be read, so it cannot be shown not to hold another "
                       "evaluation")
        check.evidence["other_evaluations"] = others
        check.evidence["unreadable_parquets"] = unreadable
        if not (others or unreadable):
            check.note(f"no other Phase 5 evaluation of level {ctx.level} is kept under the run "
                       "directory")
    with check.part("superseded training artifacts"):
        _superseded(ctx, check)
    return check.result()


def _checkpoint_chain(ctx: AcceptanceContext, check: _Check) -> None:
    """Each scene's models are the locked, trained, selected files, as embedded."""
    lock = ctx.receipt(KIND_LOCK).payload
    locked = {group: {key: (entry or {}).get("sha256")
                      for key, entry in (lock.get(group) or {}).items()}
              for group in ("checkpoints", "training_records")}
    training, checkpoints = ctx.training(), ctx.checkpoints()
    controls = ctx.controls()
    payload = controls.payload if isinstance(controls.payload, dict) else {}
    if (lock.get("controls") or {}).get("sha256") != controls.sha256:
        check.fail(f"the controls file has sha256 {controls.sha256}, and the lock binds "
                   f"{(lock.get('controls') or {}).get('sha256')}")
    for key in ctx.keys:
        record = training[key].payload if isinstance(training[key].payload, dict) else {}
        chain = {"lock": locked["checkpoints"].get(key), "live": checkpoints[key].sha256,
                 "training_record": record.get("checkpoint_sha256"),
                 "controls": (payload.get("checkpoints") or {}).get(key)}
        if len(set(chain.values())) != 1 or chain["live"] is None:
            check.fail(f"{key}: the checkpoint is {chain['live']}, the lock binds "
                       f"{chain['lock']}, its training record names {chain['training_record']}, "
                       f"and the controls ran on {chain['controls']}")
        if locked["training_records"].get(key) != training[key].sha256:
            check.fail(f"{key}: the training record is {training[key].sha256}, and the lock "
                       f"binds {locked['training_records'].get(key)}")
    commit = ctx.commit_e()
    for scene, record in ctx.records().items():
        fold = record.get("fold")
        keys = {seed: fold_seed_key(fold, seed) for seed in ctx.seeds}
        named = record.get("training_records") if isinstance(record.get("training_records"),
                                                              dict) else {}
        contents = record.get("training_record_contents")
        contents = contents if isinstance(contents, dict) else {}
        for seed, key in keys.items():
            evaluated_with = (record.get("checkpoints") or {}).get(str(seed))
            if evaluated_with != locked["checkpoints"].get(key):
                check.fail(f"{scene}: seed {seed} was evaluated with checkpoint {evaluated_with}, "
                           f"and the lock binds {locked['checkpoints'].get(key)} for {key}")
            entry = named.get(str(seed)) if isinstance(named.get(str(seed)), dict) else {}
            if entry.get("sha256") != locked["training_records"].get(key):
                check.fail(f"{scene}: seed {seed} names training record {entry.get('sha256')}, and "
                           f"the lock binds {locked['training_records'].get(key)}")
            if entry.get("commit") != commit or entry.get("config_digest") != ctx.cfg.digest():
                check.fail(f"{scene}: seed {seed}'s training record names commit "
                           f"{entry.get('commit')} and config {entry.get('config_digest')}")
            live = training.get(key)
            if live is not None and _canonical(contents.get(str(seed))) != _canonical(live.payload):
                check.fail(f"{scene}: seed {seed}'s embedded training record is not the locked "
                           f"record {key}")
        block = record.get("controls")
        expected = {"sha256": controls.sha256, "written_utc": payload.get("written_utc"),
                    "results": {key: (payload.get("results") or {}).get(key)
                                for key in keys.values()},
                    "checkpoints": {key: (payload.get("checkpoints") or {}).get(key)
                                    for key in keys.values()}}
        if _canonical(block) != _canonical(expected):
            check.fail(f"{scene}: its embedded controls are not the locked controls file's "
                       f"entries for fold {fold}")


def _timeline(ctx: AcceptanceContext, check: _Check) -> None:
    """Gate, overfit gate, training, controls, lock, evaluation, in that order."""
    stamps: dict[str, Any] = {
        "integration receipt": ctx.receipt(KIND_INTEGRATION).payload.get("stamped_utc"),
        "overfit receipt": ctx.receipt(KIND_OVERFIT).payload.get("stamped_utc"),
        "controls file": (ctx.controls().payload or {}).get("written_utc"),
        "lock receipt": ctx.receipt(KIND_LOCK).payload.get("stamped_utc"),
    }
    training = {key: (run.payload or {}).get("written_utc") for key, run in ctx.training().items()}
    evaluated = {scene: record.get("written_utc") for scene, record in ctx.records().items()}
    absent = [name for name, value in {**stamps, **training, **evaluated}.items()
              if _utc(value) is None]
    if absent:
        check.fail(f"no time of writing is recorded for {absent}")
        return
    order = [
        ("integration receipt", [stamps["integration receipt"]]),
        ("overfit receipt", [stamps["overfit receipt"]]),
        ("training records", list(training.values())),
        ("controls file", [stamps["controls file"]]),
        ("lock receipt", [stamps["lock receipt"]]),
        ("evaluation records", list(evaluated.values())),
    ]
    mark = check.mark()
    for (earlier, first), (later, second) in zip(order, order[1:]):
        if max(_utc(value) for value in first) > min(_utc(value) for value in second):
            check.fail(f"the {earlier} were written after the {later} began")
    check.evidence["timeline"] = {name: [min(values), max(values)] for name, values in order}
    check.note_if_clean(mark, "gate, overfit gate, training, controls, lock, and evaluation were "
                              "written in that order")


def _other_evaluations(ctx: AcceptanceContext) -> tuple[list[str], list[str]]:
    """Phase 5 evaluation records of this level outside its evaluation directory.

    Returns those found, and the parquets that cannot be read, which cannot be
    shown not to be one. The staging directory of an unfinished tables or
    figures build holds reporting outputs only, so it is not searched. Only
    such a directory is skipped: a direct child of the tables root or the
    figures root named {level}.partial.<32 hex digits>, as
    lot.phase5_provenance.new_staging_directory names it. Any other path,
    whatever .partial its name holds, is searched.
    """
    import pyarrow.parquet as pq

    from .evaluate import RUN_METADATA_KEY

    live = (ctx.run_dir / "eval" / ctx.level).resolve()
    roots = {ctx.tables_dir.parent.resolve(), ctx.figures_dir.parent.resolve()}

    def staging(resolved: Path) -> bool:
        return any(parent.parent in roots and STAGING_NAME.match(parent.name) is not None
                   for parent in resolved.parents)

    found, unreadable = [], []
    for path in sorted(ctx.run_dir.rglob("*.parquet")):
        relative = path.relative_to(ctx.run_dir)
        resolved = path.resolve()
        if resolved.parent == live or staging(resolved):
            continue
        try:
            raw = (pq.read_schema(path).metadata or {}).get(RUN_METADATA_KEY)
            record = None if raw is None else json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            # Reported, not skipped: an unreadable file may be a hidden evaluation.
            unreadable.append(relative.as_posix())
            continue
        if (isinstance(record, dict) and record.get("phase") == 5 and "audit" in record
                and "scene" in record and record.get("level") == ctx.level):
            found.append(relative.as_posix())
    return found, unreadable


def _superseded(ctx: AcceptanceContext, check: _Check) -> None:
    """Training artifacts a rerun moved aside, none written after evaluation began."""
    directory = ctx.run_dir / "checkpoints" / ctx.level
    first = min((_utc(record.get("written_utc")) for record in ctx.records().values()
                 if _utc(record.get("written_utc")) is not None), default=None)
    listed = []
    for path in sorted(directory.glob("*.superseded.*")) if directory.is_dir() else []:
        written = None
        if path.suffix == ".json":
            try:
                written = json.loads(path.read_text(encoding="utf-8")).get("written_utc")
            except (OSError, ValueError, AttributeError):
                written = None
        listed.append({"path": path.name, "written_utc": written})
        if first is not None and _utc(written) is not None and _utc(written) >= first:
            check.fail(f"{path.name} was written at {written}, after evaluation began")
    check.evidence["superseded_training_artifacts"] = listed
    if listed:
        check.note(f"{len(listed)} superseded training artifact(s) are kept and listed in the "
                   "evidence")


def check_09_tiny_overfit(ctx: AcceptanceContext) -> Outcome:
    """The licensed overfit receipt passed the frozen gate on fold 0's training scenes."""
    check = _Check()
    tiny = dict(ctx.cfg.tiny_overfit)
    with check.part("the overfit receipt"):
        mark = check.mark()
        receipt = ctx.receipt(KIND_OVERFIT)
        for problem in ctx.verification(KIND_OVERFIT):
            check.fail(problem)
        payload = receipt.payload
        reached, threshold = payload.get("reached_centered_cosine"), payload.get("threshold")
        frozen_threshold = float(tiny.get("threshold_centered_cosine"))
        if payload.get("passed") is not True:
            check.fail("the overfit receipt records a failed gate")
        if threshold != frozen_threshold:
            check.fail(f"the gate ran at threshold {threshold}, not the frozen {frozen_threshold}")
        if not (_finite(reached) and _finite(threshold) and reached >= threshold):
            check.fail(f"the gate reached {reached}, below its threshold {threshold}")
        if payload.get("n_pairs") != int(tiny.get("n_pairs")):
            check.fail(f"the gate fitted {payload.get('n_pairs')} pairs, not {tiny.get('n_pairs')}")
        subset = payload.get("subset") if isinstance(payload.get("subset"), list) else []
        if len(subset) != int(tiny.get("n_pairs")):
            check.fail(f"the gate names {len(subset)} subset pairs, not {tiny.get('n_pairs')}")
        regimes = sorted(str(r) for r in payload.get("regimes") or [])
        if regimes != sorted(str(r) for r in tiny.get("regimes") or []):
            check.fail(f"the gate covered regimes {regimes}, not {tiny.get('regimes')}")
        if sorted({str(item.get("regime")) for item in subset}) != regimes:
            check.fail("the subset's pairs do not span the gate's regimes")
        train = set(ctx.folds[0].train)
        outside = sorted({str(item.get("scene")) for item in subset} - train)
        if outside:
            check.fail(f"the subset reaches {outside}, outside fold 0's training scenes")
        for field, want in (("fold", ctx.folds[0].index), ("level", ctx.primary_level),
                            ("seed", int(tiny.get("seed")))):
            if payload.get(field) != want:
                check.fail(f"the gate ran at {field} {payload.get(field)!r}, not {want!r}")
        steps = payload.get("steps")
        if not (_count(steps) and steps <= int(tiny.get("max_steps"))):
            check.fail(f"the gate took {steps} steps, beyond its frozen {tiny.get('max_steps')}")
        check.evidence["overfit"] = {field: payload.get(field) for field in OVERFIT_VERDICT_FIELDS
                                     if field != "subset"}
        check.evidence["overfit"]["sha256"] = receipt.sha256
        check.note_if_clean(mark, f"the gate reached centered cosine {reached} against "
                                  f"{threshold} on {len(subset)} training pairs of fold 0 "
                                  f"spanning {regimes}")
    with check.part("the verdict each run record embeds"):
        receipt = ctx.receipt(KIND_OVERFIT)
        expected = {"sha256": receipt.sha256,
                    **{field: receipt.payload.get(field) for field in OVERFIT_VERDICT_FIELDS}}
        for scene, record in ctx.records().items():
            if _canonical(record.get("overfit_verdict")) != _canonical(expected):
                check.fail(f"{scene}: its embedded overfit verdict is not the licensed receipt's")
    return check.result()


def check_10_frozen_before_test(ctx: AcceptanceContext) -> Outcome:
    """One commit and one frozen configuration made every artifact, and the
    evaluated models have the frozen architecture."""
    check = _Check()
    expected_training = ctx.settings.digest()
    frozen_parameters = int(ctx.cfg.model.get("parameter_count"))
    with check.part("one commit and one configuration"):
        mark = check.mark()
        commit = ctx.commit_e()
        digest = ctx.cfg.digest()
        for scene, record in ctx.records().items():
            if record.get("config_digest") != digest:
                check.fail(f"{scene}: evaluated under config {record.get('config_digest')}")
        if list(ctx.common("seeds")) != ctx.seeds:
            check.fail(f"the run evaluated seeds {ctx.common('seeds')}, not {ctx.seeds}")
        for kind in (KIND_INTEGRATION, KIND_OVERFIT, KIND_LOCK):
            identity = receipt_identity(ctx.receipt(kind).payload)
            for field, want in (("commit", commit), ("config_digest", digest)):
                if identity.get(field) != want:
                    check.fail(f"the {ctx.receipt_stems()[kind]} receipt names {field} "
                               f"{identity.get(field)}, not {want}")
        for key, run in ctx.training().items():
            record = run.payload if isinstance(run.payload, dict) else {}
            for field, want in (("commit", commit), ("config_digest", digest),
                                ("training_config_digest", expected_training),
                                ("parameter_count", frozen_parameters)):
                if record.get(field) != want:
                    check.fail(f"{key}: the training record names {field} "
                               f"{record.get(field)!r}, not {want!r}")
        controls = ctx.controls().payload or {}
        for field, want in (("commit", commit), ("config_digest", digest)):
            if controls.get(field) != want:
                check.fail(f"the controls file names {field} {controls.get(field)}, not {want}")
        lock = ctx.receipt(KIND_LOCK).payload
        if lock.get("seeds") != ctx.seeds:
            check.fail(f"the lock binds seeds {lock.get('seeds')}, not {ctx.seeds}")
        if list(ctx.gate_step("1").get("seeds") or []) != ctx.seeds:
            check.fail(f"gate step 1 recorded seeds {ctx.gate_step('1').get('seeds')}")
        check.evidence.update(commit=commit, config_digest=digest,
                              training_config_digest=expected_training, seeds=ctx.seeds)
        check.note_if_clean(mark, f"the receipts, {len(ctx.keys)} training records, the "
                                  f"controls, and every run record name commit {commit} and "
                                  f"config {digest}")
    with check.part("the evaluated checkpoints"):
        mark = check.mark()
        architecture = ctx.architecture()
        if architecture["parameter_count"] != frozen_parameters:
            check.fail(f"the frozen architecture has {architecture['parameter_count']} "
                       f"parameters, and the config records {frozen_parameters}")
        gate = ctx.gate_step("12").get("parameter_count")
        if gate != frozen_parameters:
            check.fail(f"gate step 12 counted {gate} parameters, not {frozen_parameters}")
        for key, checkpoint in ctx.checkpoints().items():
            facts = checkpoint.facts
            if facts is None:
                check.fail(f"{key}: the checkpoint cannot be read: {checkpoint.error}")
                continue
            if facts["training_config_digest"] != expected_training:
                check.fail(f"{key}: the checkpoint names training_config_digest "
                           f"{facts['training_config_digest']}, not {expected_training}")
            if facts["shapes"] != architecture["shapes"]:
                check.fail(f"{key}: the checkpoint's state is not the frozen architecture's")
            if facts["numel"] != frozen_parameters:
                check.fail(f"{key}: the checkpoint holds {facts['numel']} parameters, not "
                           f"{frozen_parameters}")
        check.evidence["architecture"] = {"grid": architecture["grid"],
                                          "parameter_count": architecture["parameter_count"]}
        check.note_if_clean(mark, f"every checkpoint holds the frozen architecture's "
                                  f"{frozen_parameters} parameters, trained under training "
                                  f"config {expected_training}")
    with check.part("the code since E"):
        mark = check.mark()
        for problem in ctx.ancestry_problems(("measurement_paths", "other_paths", "commit",
                                              "worktree")):
            check.fail(problem)
        commit = ctx.commit_e()
        last = {}
        for path in TRAINING_SOURCES:
            touching = ctx.repo.commits_touching(path)
            last[path] = touching[0] if touching else None
        check.evidence["last_commits"] = last
        check.note_if_clean(mark, "no file that decides what is measured changed between E "
                                  "and R")
        reachable = ctx.repo.ancestors(commit)
        late = sorted(path for path, value in last.items() if value and value not in reachable)
        if late:
            check.note(f"{late} were touched after E; the diff since E shows no change to them")
    return check.result()


def check_11_validation_only_selection(ctx: AcceptanceContext) -> Outcome:
    """Each evaluated checkpoint is the one validation selected, by its record's history."""
    check = _Check()
    _suite_files(ctx, check, 11)
    with check.part("each checkpoint against its training record"):
        mark = check.mark()
        every = ctx.settings.validation_every_steps
        patience = ctx.settings.early_stopping_patience
        selection = {}
        for key, run in ctx.training().items():
            fold, seed = ctx.fold_of(key)
            record = run.payload if isinstance(run.payload, dict) else None
            checkpoint = ctx.checkpoints()[key]
            if record is None or checkpoint.facts is None:
                check.fail(f"{key}: the training record or the checkpoint cannot be read: "
                           f"{run.error or checkpoint.error}")
                continue
            facts = checkpoint.facts
            history = record.get("history") if isinstance(record.get("history"), list) else []
            steps = [entry[0] for entry in history]
            best_step = record.get("best_step")
            best = record.get("best_validation_centered_cosine")
            selection[key] = {"best_step": best_step, "checkpoint_step": facts["step"],
                              "best_validation": best,
                              "checkpoint_validation": facts["validation_centered_cosine"],
                              "n_validations": len(history),
                              "stopped_early": record.get("stopped_early")}
            if (facts["fold"], facts["seed"]) != (fold.index, seed):
                check.fail(f"{key}: the checkpoint was trained for fold {facts['fold']} seed "
                           f"{facts['seed']}")
            if facts["step"] != best_step:
                check.fail(f"{key}: the checkpoint was saved at step {facts['step']}, and its "
                           f"training record selected step {best_step}")
            if not _same(facts["validation_centered_cosine"], best):
                check.fail(f"{key}: the checkpoint's validation score is "
                           f"{facts['validation_centered_cosine']}, and its record's best is "
                           f"{best}")
            finite = [(step, score) for step, score in history if _finite(score)]
            if not finite:
                check.fail(f"{key}: no finite validation score is on record")
                continue
            top = max(score for _, score in finite)
            first = next(step for step, score in finite if score == top)
            if not (_same(best, top) and best_step == first):
                check.fail(f"{key}: the best validation on record is {top} at step {first}, and "
                           f"the record selected {best} at step {best_step}")
            steps_run = record.get("steps_run")
            if steps != sorted(set(steps)) or any(
                    not _count(step) or (step % every and step != steps_run) for step in steps):
                check.fail(f"{key}: validations at {steps[:5]}... do not follow the frozen "
                           f"cadence of {every} steps")
            if not (_count(steps_run) and steps and steps[-1] == steps_run
                    and steps_run <= ctx.settings.max_steps):
                check.fail(f"{key}: the run's last validation, step {steps[-1] if steps else None},"
                           f" is not its {steps_run} steps within {ctx.settings.max_steps}")
            later = len([step for step in steps if step > first])
            if record.get("stopped_early") is True and later != patience:
                check.fail(f"{key}: it stopped early {later} validations after its best, and the "
                           f"frozen patience is {patience}")
            if record.get("stopped_early") is not True and steps_run != ctx.settings.max_steps:
                check.fail(f"{key}: it neither stopped early nor ran its {ctx.settings.max_steps} "
                           f"steps; it ran {steps_run}")
        check.evidence["selection"] = selection
        check.evidence["checkpoint_selection"] = ctx.settings.checkpoint_selection
        check.note_if_clean(mark, f"{len(selection)} checkpoints are the step and the score "
                                  f"validation selected, under the frozen cadence of {every} "
                                  f"steps and patience of {patience}")
    return check.result()


def _sample_pairs(ctx: AcceptanceContext) -> dict[str, list[tuple[str, str]]]:
    """A deterministic sample: per fold, its first evaluated scene, and per regime
    its first pair in sorted order."""
    facts = ctx.rows()
    records = ctx.records()
    sample: dict[str, list[tuple[str, str]]] = {}
    for fold in ctx.folds:
        scene = next((s for s in ctx.expected_scenes if s in facts.n_primary
                      and records[s].get("fold") == fold.index), None)
        if scene is None:
            continue
        by_regime: dict[str, tuple[str, str]] = {}
        for record in facts.collapsed:
            if record["scene"] != scene:
                continue
            context, target = record["camera_pair"].split("|")[1:]
            pair = (context, target)
            if record["regime"] not in by_regime or pair < by_regime[record["regime"]]:
                by_regime[record["regime"]] = pair
        sample[scene] = [by_regime[regime] for regime in sorted(by_regime)]
    return sample


def check_12_fixed_support(ctx: AcceptanceContext) -> Outcome:
    """The headline compares on Context-Lift's own support, fixed before any model."""
    check = _Check()
    _suite_files(ctx, check, 12)
    with check.part("the Context-Lift arm across seeds and regions"):
        mark = check.mark()
        facts = ctx.rows()
        for problem in facts.malformed + facts.explicit_drift + facts.partition:
            check.fail(problem)
        check.evidence["rows"] = facts.rows
        check.note_if_clean(mark, f"over {facts.rows} rows, the Context-Lift arm and its floors "
                                  "are identical under every seed's model, and the regions "
                                  "partition each support")
    with check.part("the support's definition in evaluate_scene at E"):
        mark = check.mark()
        function = _evaluate_scene(ctx)
        bound = _bindings(function, "support")
        if not bound:
            raise Unavailable("evaluate_scene binds no support")
        value = bound[0][1]
        inner = _argument(value, 1, "evaluable") if isinstance(value, ast.Call) else None
        if not (_call_name(value) == "primary_support"
                and isinstance(_first_argument(value, "lift"), ast.Name)
                and _first_argument(value, "lift").id == "lift"
                and _call_name(inner) == "context_lift_support"
                and isinstance(_first_argument(inner, "lift"), ast.Name)
                and _first_argument(inner, "lift").id == "lift"):
            check.fail(f"the support is {ast.unparse(value)}, not Context-Lift's support refereed "
                       "by ground truth")
        regions = _bindings(function, "point_regions")
        whole = None
        if len(regions) == 1 and isinstance(regions[0][1], ast.Dict):
            for key, item in zip(regions[0][1].keys, regions[0][1].values):
                if isinstance(key, ast.Constant) and key.value == "all":
                    whole = item
        if not (isinstance(whole, ast.Name) and whole.id == "support"):
            check.fail("the whole region's points are not the support")
        check.evidence["support"] = ast.unparse(value)
        check.note_if_clean(mark, "the support at E is Context-Lift's landed samples, refereed "
                                  "by ground truth, and the headline's points are that support")
    with check.part("integration gate step 7"):
        mark = check.mark()
        counts = ctx.gate_step("7").get("counts") or {}
        support = counts.get("final_support")
        if not (_count(support) and support > 0
                and support <= min(counts.get("landed") or 0, counts.get("gt_evaluable") or 0)):
            check.fail(f"gate step 7 counts {counts}")
        check.evidence["step7_counts"] = counts
        check.note_if_clean(mark, f"gate step 7 built the support on a real pair: {support} of "
                                  f"{counts.get('landed')} landed samples")
    with check.part("an independent recount of the support"):
        mark = check.mark()
        facts = ctx.rows()
        sample = _sample_pairs(ctx)
        recounted, missing = {}, []
        for scene, pairs in sample.items():
            try:
                counts = ctx.recount(ctx.cfg, ctx.analysis, ctx.level, scene, pairs)
            except SceneInputsUnavailable as error:
                missing.append(str(error))
                continue
            for pair in pairs:
                stored = facts.n_primary[scene].get(pair)
                fresh = counts.get(pair)
                recounted[f"{scene}|{pair[0]}|{pair[1]}"] = {"stored": stored, "recount": fresh}
                if stored != fresh:
                    check.fail(f"{scene} {pair[0]} -> {pair[1]}: the record holds n_primary "
                               f"{stored}, and the recount from the scene inputs gives {fresh}")
        check.evidence["recount"] = recounted
        if recounted:
            check.note_if_clean(mark, f"recounted the support of {len(recounted)} sampled pairs "
                                      "from the scene inputs, and each matches its record")
        if missing:
            check.evidence["recount_not_run"] = missing
            check.note(f"the recount did not run for {len(missing)} of {len(sample)} sampled "
                       "scene(s), because their scene inputs are not available here; their "
                       "supports were checked across seeds, regions, and the code at E only")
    return check.result()


def check_13_support_not_shrunk(ctx: AcceptanceContext) -> Outcome:
    """A failing prediction is a counted failure on the support, never a smaller support."""
    check = _Check()
    with check.part("the support under every seed's model"):
        mark = check.mark()
        facts = ctx.rows()
        for problem in facts.support_drift + facts.failures + facts.predictor_nonfinite:
            check.fail(problem)
        check.evidence["failures_per_seed"] = {str(seed): count
                                               for seed, count in facts.failures_per_seed.items()}
        check.note_if_clean(mark, f"the support count is identical under every seed's model, "
                                  f"and failures are counted on it: "
                                  f"{check.evidence['failures_per_seed']} per seed")
    with check.part("evaluate_scene at E"):
        mark = check.mark()
        function = _evaluate_scene(ctx)
        first_model = _model_calls(function)[0].lineno
        for name in ("support", "point_regions"):
            bound = _bindings(function, name)
            if len(bound) != 1:
                check.fail(f"{name} is bound {len(bound)} times in evaluate_scene, not once")
            late = [line for line, _ in bound if line > first_model]
            if late:
                check.fail(f"{name} is bound at line(s) {late}, after the model runs at line "
                           f"{first_model}")
        for call in _model_calls(function):
            seen = _names(call) & {"support", "point_regions", "points"}
            if seen:
                check.fail(f"the model at line {call.lineno} receives {sorted(seen)}")
        for line, value in _bindings(function, "points"):
            if not (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name)
                    and value.value.id == "point_regions"):
                check.fail(f"points is bound at line {line} to {ast.unparse(value)}")
        check.evidence["first_model_call_line"] = first_model
        check.note_if_clean(mark, "evaluate_scene at E fixes the support before the model runs "
                                  "and never binds it again")
    with check.part("integration gate step 7"):
        mark = check.mark()
        step = ctx.gate_step("7")
        support = (step.get("counts") or {}).get("final_support")
        if step.get("failures_counted") != support:
            check.fail(f"gate step 7 counted {step.get('failures_counted')} failures for an "
                       f"all-nonfinite predictor on a support of {support}")
        check.note_if_clean(mark, "gate step 7: an all-nonfinite predictor left the support "
                                  "unchanged and was counted as failing on all of it")
    return check.result()


def _pooled_estimate(records: Sequence[Mapping[str, Any]], quantity: str, metric: str) -> float:
    """A headline quantity as the unweighted mean over camera pairs, by hand."""
    contributing = [r for r in records if _count(r.get("n_primary")) and r["n_primary"] > 0]

    def mean(field: str) -> float:
        values = [float(r[field]) for r in contributing if _finite(r.get(field))]
        return sum(values) / len(values) if values else float("nan")

    cl, predict, nowarp = mean(f"cl_{metric}"), mean(f"predict_{metric}"), mean(f"nowarp_{metric}")
    return {"delta_learn_pp": cl - predict, "cl_margin": cl - nowarp,
            "predict_margin": predict - nowarp}[quantity]


def check_14_paired_bootstrap(ctx: AcceptanceContext) -> Outcome:
    """Every interval is the paired scene bootstrap under the frozen settings."""
    check = _Check()
    _suite_files(ctx, check, 14)
    with check.part("the bootstrap settings"):
        mark = check.mark()
        tables = ctx.tables()
        want = ctx.bootstrap()
        # read_published_tables has held every table's and document's run
        # record to the manifest's, bootstrap included, so one comparison
        # covers them all.
        if tables.run_record.get("bootstrap") != want:
            check.fail(f"the tables name bootstrap {tables.run_record.get('bootstrap')}, not "
                       f"{want}")
        if ctx.common("analysis_reporting_digest") != ctx.analysis.reporting_digest():
            check.fail("the run was evaluated under another reporting configuration than the one "
                       "reading it")
        check.evidence["bootstrap"] = want
        check.note_if_clean(mark, f"every table's paired bootstrap resamples "
                                  f"{want['primary_unit']}s, {want['resamples']} times, seed "
                                  f"{want['seed']}, confidence {want['confidence']}")
    with check.part("a replicate count beside every interval"):
        mark = check.mark()
        tables = ctx.tables()
        for name, rows in tables.tables.items():
            columns = set().union(*(row.keys() for row in rows)) if rows else set()
            for column in sorted(columns):
                if column == "ci_low" or column.endswith("_ci_low"):
                    count = column[: -len("ci_low")] + "ci_replicates"
                    if count not in columns:
                        check.fail(f"{name}: {column} has no {count} beside it")
        for row in tables.rows(PRIMARY_TABLE):
            if row.get("supported") is not True:
                continue
            for column, value in row.items():
                if column.endswith("_ci_replicates") and _finite(row.get(column[: -len(
                        "_ci_replicates")])) and not (_count(value) and value > 0):
                    check.fail(f"{PRIMARY_TABLE} {row.get('metric')} {row.get('analysis')} "
                               f"{row.get('bin')}: {column} is {value} on a supported cell")
        check.note_if_clean(mark, "every interval in every table carries its replicate count, "
                                  "and no supported headline interval rests on none")
    with check.part("headline cells recomputed from the parquets"):
        mark = check.mark()
        tables = ctx.tables()
        facts = ctx.rows()
        by_key = {(row["metric"], row["analysis"]): row for row in tables.rows(PRIMARY_TABLE)
                  if row.get("axis") == SCOPE_AXIS}
        recomputed = []
        for metric in METRICS:
            for scope in SCOPES:
                records = [r for r in facts.collapsed
                           if scope == POOLED_SCOPE or r["regime"] == scope]
                row = by_key.get((metric, scope))
                if row is None:
                    check.fail(f"{PRIMARY_TABLE} holds no {scope} row under {metric}")
                    continue
                for quantity in RECOMPUTED_QUANTITIES:
                    by_hand = _pooled_estimate(records, quantity, metric)
                    cell = evaluate_quantity(records, quantity, metric, ctx.analysis)
                    entry = {"metric": metric, "scope": scope, "quantity": quantity,
                             "table": row.get(quantity), "by_hand": by_hand,
                             "estimate": cell.estimate, "lo": cell.lo, "hi": cell.hi,
                             "n_replicates": cell.n_replicates}
                    recomputed.append(entry)
                    where = f"{quantity} under {metric} in {scope}"
                    if not (_close(by_hand, row.get(quantity))
                            and _close(cell.estimate, row.get(quantity))):
                        check.fail(f"{where}: the table holds {row.get(quantity)}, and the "
                                   f"parquets give {by_hand} by hand and {cell.estimate} by the "
                                   "estimand layer")
                    if not (_close(cell.lo, row.get(f"{quantity}_ci_low"))
                            and _close(cell.hi, row.get(f"{quantity}_ci_high"))):
                        check.fail(f"{where}: the table's interval [{row.get(f'{quantity}_ci_low')}"
                                   f", {row.get(f'{quantity}_ci_high')}] is not the paired scene "
                                   f"bootstrap's [{cell.lo}, {cell.hi}]")
                    if cell.n_replicates != row.get(f"{quantity}_ci_replicates"):
                        check.fail(f"{where}: {row.get(f'{quantity}_ci_replicates')} replicates "
                                   f"in the table, {cell.n_replicates} recomputed")
                    for field in ("n_scenes", "n_camera_pairs", "n_feature_comparisons"):
                        if getattr(cell, field) != row.get(field):
                            check.fail(f"{where}: {field} is {row.get(field)} in the table and "
                                       f"{getattr(cell, field)} recomputed")
        check.evidence["recomputed"] = recomputed
        check.note_if_clean(mark, f"{len(recomputed)} headline cells recomputed from the "
                                  f"parquets match the table within {RECOMPUTE_TOLERANCE}")
        check.note("the intervals are recomputed through the frozen estimand layer, so for them "
                   "this is an integrity check; each estimate is also recomputed by hand")
    return check.result()


def _pin_hashes(text: str) -> dict[str, list[str]]:
    """Every sha256 pin.md lists beside a path, in the order listed."""
    listed: dict[str, list[str]] = {}
    for match in re.finditer(r"^\s+([0-9a-f]{64})\s+(\S+)", text, re.MULTILINE):
        listed.setdefault(match.group(2), []).append(match.group(1))
    return listed


def check_15_definitions_frozen(ctx: AcceptanceContext) -> Outcome:
    """The metric definitions, the protocol, and the centering are the frozen ones."""
    check = _Check()
    with check.part("the frozen documents at E"):
        mark = check.mark()
        commit = ctx.commit_e()
        freeze = _pin_hashes(ctx.repo.text(commit, FREEZE_DOCUMENT) or "")
        pin = _pin_hashes(ctx.repo.text(commit, PIN_DOCUMENT) or "")
        blobs = {}
        for path in FROZEN_BY_FREEZE + AMENDED_SINCE_FREEZE:
            blob = _sha256(ctx.repo.blob(commit, path))
            blobs[path] = blob
            if path in FROZEN_BY_FREEZE:
                want = (freeze.get(path) or [None])[0]
                if blob is None or blob != want:
                    check.fail(f"{path} at E has sha256 {blob}, and {FREEZE_DOCUMENT} freezes "
                               f"{want}")
            else:
                want = (pin.get(path) or [None])[-1]
                if blob is None or blob != want:
                    check.fail(f"{path} at E has sha256 {blob}, and {PIN_DOCUMENT} records "
                               f"{want}")
        check.evidence["documents_at_e"] = blobs
        check.note_if_clean(mark, f"{list(FROZEN_BY_FREEZE)} at E are the frozen blobs, and "
                                  f"{list(AMENDED_SINCE_FREEZE)} carry exactly the pinned "
                                  "amendments")
    with check.part("the measurement digest"):
        mark = check.mark()
        commit = ctx.commit_e()
        digest = ctx.analysis.measurement_digest()
        pinned = re.search(r"Phase 3 measurement digest: `([0-9a-f]+)`",
                           ctx.repo.text(commit, PIN_DOCUMENT) or "")
        if pinned is None or pinned.group(1) != digest:
            check.fail(f"the analysis has measurement digest {digest}, and {PIN_DOCUMENT} pins "
                       f"{pinned.group(1) if pinned else None}")
        for scene, record in ctx.records().items():
            if record.get("measurement_digest") != digest:
                check.fail(f"{scene}: measured under {record.get('measurement_digest')}")
        for kind in (KIND_INTEGRATION, KIND_OVERFIT, KIND_LOCK):
            named = receipt_identity(ctx.receipt(kind).payload).get("measurement_digest")
            if named != digest:
                check.fail(f"the {ctx.receipt_stems()[kind]} receipt names measurement digest "
                           f"{named}")
        check.evidence["measurement_digest"] = digest
        check.note_if_clean(mark, f"the run, its receipts, and the pin share measurement digest "
                                  f"{digest}")
    with check.part("the centering vector"):
        mark = check.mark()
        identities = receipt_scene_identities(ctx.integration())
        ctx.gate_step("5")
        named = {}
        for scene, record in ctx.records().items():
            accepted = (identities.get(scene) or {}).get("accepted_mean_vector_digest")
            named[scene] = {"evaluation": record.get("mean_vector_digest"), "phase4": accepted}
            if record.get("mean_vector_digest") is None or \
                    record.get("mean_vector_digest") != accepted:
                check.fail(f"{scene}: centered with {record.get('mean_vector_digest')}, and the "
                           f"accepted Phase 4 rows with {accepted}")
        check.evidence["mean_vector"] = named
        check.note_if_clean(mark, "every scene was centered with the vector the accepted Phase 4 "
                                  "rows were centered with, which gate step 5 matched to the "
                                  "live vector")
    with check.part("both metrics over the same cells"):
        mark = check.mark()
        tables = ctx.tables()
        for name, rows in sorted(tables.tables.items()):
            metrics = {row.get("metric") for row in rows}
            for left, right in METRIC_PAIRS:
                if not ({left, right} & metrics):
                    continue
                cells = {metric: sorted(_canonical([row.get(c) for c in CELL_KEY_COLUMNS])
                                        for row in rows if row.get("metric") == metric)
                         for metric in (left, right)}
                if cells[left] != cells[right]:
                    check.fail(f"{name}: {len(cells[left])} {left} cells and {len(cells[right])} "
                               f"{right} cells, not the same cells")
        check.note_if_clean(mark, "every table covers the same cells under both metrics")
    with check.part("the Mean-Feature floor"):
        mark = check.mark()
        tables = ctx.tables()
        for name, rows in sorted(tables.tables.items()):
            for row in rows:
                for column in row:
                    base = column[: -len("_status")] if column.endswith("_status") else None
                    if base not in MEAN_FEATURE_NAMES:
                        continue
                    raw = row.get("metric") in ("raw", "l2_raw")
                    status = row.get(column)
                    if status is None and row.get(base) is None:
                        # A column of another row's population, filled in so the table is
                        # rectangular. It is not part of this row.
                        continue
                    if raw and status != MEAN_FEATURE_REPORTED:
                        check.fail(f"{name}: {base} under {row.get('metric')} has status {status}")
                    if not raw and (status != MEAN_FEATURE_NOT_APPLICABLE
                                    or _finite(row.get(base))):
                        check.fail(f"{name}: {base} under {row.get('metric')} holds "
                                   f"{row.get(base)} with status {status}")
        check.note_if_clean(mark, "Mean-Feature is reported under raw cosine only, as PROTOCOL "
                                  "3.7 defines it")
    return check.result()


def _expected_disclosures(tables: Any) -> dict[tuple, Mapping[str, Any]]:
    """Every interpreted-effect cell the tables show, keyed as its disclosure entry."""
    expected: dict[tuple, Mapping[str, Any]] = {}

    def add(table: str, row: Mapping[str, Any], region: str, path: str, quantity: str) -> None:
        expected[(table, row.get("metric"), region, path, row.get("analysis"), row.get("axis"),
                  row.get("bin"), quantity)] = row

    for row in tables.rows(PRIMARY_TABLE):
        for effect in PRIMARY_EFFECTS:
            add(PRIMARY_TABLE, row, "all", PER_POINT, effect)
    for row in tables.rows(FORMULATION_TABLE):
        add(FORMULATION_TABLE, row, "all", PER_POINT, "delta_formulation")
    for row in tables.rows(SPLAT_TABLE):
        for effect in SPLAT_EFFECTS:
            add(SPLAT_TABLE, row, "all", SPLAT_POOL, effect)
    for row in tables.rows(REGION_TABLE):
        if row.get("row_kind") == "region":
            add(REGION_TABLE, row, row.get("region"), row.get("path"), REGION_GAP[row.get("path")])
    return expected


def _term(term: Any) -> PathEstimate:
    if not isinstance(term, Mapping):
        raise ValueError(f"a disclosure term is {_short(term)}")
    return PathEstimate(float(term["estimate"]), float(term["lo"]), float(term["hi"]))


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def check_16_near_zero(ctx: AcceptanceContext) -> Outcome:
    """PROTOCOL 3.9 as decision 2 reads it, on every interpreted-effect cell."""
    check = _Check()
    _suite_files(ctx, check, 16)
    band = ctx.analysis.path_agreement_tolerance
    with check.part("the disclosure entries"):
        mark = check.mark()
        tables = ctx.tables()
        document = tables.documents.get(NEAR_ZERO_FILE)
        if not isinstance(document, Mapping):
            raise Unavailable(f"the tables carry no {NEAR_ZERO_FILE}")
        expected = _expected_disclosures(tables)
        seen: dict[tuple, Mapping[str, Any]] = {}
        decomposition: dict[str, int] = {}
        for entry in document.get("entries") or []:
            key = (entry.get("table"), entry.get("metric"), entry.get("region"), entry.get("path"),
                   entry.get("analysis"), entry.get("axis"), entry.get("bin"),
                   entry.get("quantity"))
            if key in seen:
                check.fail(f"the disclosure lists {key} twice")
            seen[key] = entry
            if key not in expected:
                check.fail(f"the disclosure lists {key}, which no table shows")
        for key in expected:
            if key not in seen:
                check.fail(f"no disclosure entry for {key}")
        checked = 0
        # Each supported cell's wording by regime, quantity, and metric.
        grouped: dict[tuple[str, str, str], dict[str, int]] = {}
        for key, entry in seen.items():
            if key not in expected:
                continue
            checked += 1
            quantity = entry.get("quantity")
            where = " ".join(str(part) for part in key)
            if quantity not in INTERPRETED_EFFECTS:
                check.fail(f"{where}: {quantity} is not an interpreted effect")
                continue
            terms = [entry.get(name) for name in ("reported", "per_point", "splat_pool",
                                                  "path_difference")]
            if any(term is not None and not _count(term.get("n_replicates")) for term in terms
                   if isinstance(term, Mapping)) or not isinstance(entry.get("reported"), Mapping):
                check.fail(f"{where}: a disclosure term carries no replicate count")
                continue
            single = entry.get("splat_pool") is None
            if single != (quantity in SINGLE_PATH_BY_CONSTRUCTION):
                check.fail(f"{where}: disclosed on {'one path' if single else 'two paths'}")
                continue
            rerun = near_zero_disclosure(
                quantity, _term(entry["per_point"]),
                None if single else _term(entry["splat_pool"]), ctx.analysis,
                None if single else _term(entry["path_difference"]),
                reported=_term(entry["reported"]))
            # A cell below support makes no claim, PROTOCOL 3.4, and carries
            # no qualifier, section 9. Its terms are reproduced, and its flag
            # and wording must be the marker that claims nothing.
            supported = entry.get("supported") is True
            reproduced = (("near_zero", "wording") if supported else ()) + (
                "paths_agree_in_sign", "both_intervals_exclude_zero")
            for field in reproduced:
                if rerun.get(field) != entry.get(field):
                    check.fail(f"{where}: the disclosure records {field} "
                               f"{entry.get(field)!r}, and its terms give {rerun.get(field)!r}")
            if entry.get("band") != band:
                check.fail(f"{where}: disclosed under band {entry.get('band')}, not {band}")
            wording = entry.get("wording")
            if not supported:
                if entry.get("near_zero") is not None or wording != WORDING_BELOW_SUPPORT:
                    check.fail(f"{where}: a cell below support carries near-zero wording, "
                               f"flag {entry.get('near_zero')!r} and {wording!r}. It makes no "
                               f"claim, so it shows {WORDING_BELOW_SUPPORT!r}")
            else:
                licensed = (ENGAGED_SINGLE_PATH if single else ENGAGED_TWO_PATH) \
                    if entry.get("near_zero") else NOT_ENGAGED
                if wording not in licensed:
                    check.fail(f"{where}: the wording {wording!r} is not licensed here")
            reported = entry["reported"].get("estimate")
            if entry.get("reported_in_band") != bool(_finite(reported) and abs(reported) <= band):
                check.fail(f"{where}: reported_in_band is {entry.get('reported_in_band')}")
            row = expected[key]
            for suffix, field in (("near_zero", "near_zero"), ("near_zero_wording", "wording")):
                if row.get(f"{quantity}_{suffix}") != entry.get(field):
                    check.fail(f"{where}: the table shows {suffix} "
                               f"{row.get(f'{quantity}_{suffix}')!r}, and the disclosure "
                               f"{entry.get(field)!r}")
            if supported:
                case = f"{wording} | {entry.get('analysis')} | {quantity} | {entry.get('metric')}"
                decomposition[case] = decomposition.get(case, 0) + 1
                cell = grouped.setdefault(
                    (str(entry.get("analysis")), str(quantity), str(entry.get("metric"))), {})
                cell[str(wording)] = cell.get(str(wording), 0) + 1
        check.evidence["entries_checked"] = checked
        check.evidence["decomposition"] = dict(sorted(decomposition.items()))
        engaged = sum(1 for entry in seen.values() if entry.get("near_zero"))
        check.note_if_clean(mark, f"{checked} interpreted-effect cells are disclosed, each "
                                  f"reproduced from its persisted terms; {engaged} engage the "
                                  "near-zero wording")
        check.note("the wording is reproduced through the frozen near_zero_disclosure, so this "
                   "is an integrity check of what was written, not a second derivation")
        by_wording: dict[str, int] = {}
        for case, count in decomposition.items():
            wording = case.split(" | ")[0]
            by_wording[wording] = by_wording.get(wording, 0) + count
        for wording, count in sorted(by_wording.items()):
            check.note(f"  supported cells worded {wording!r}: {count}")
        # Never a bare total: the decomposition by regime, quantity, and metric,
        # as scripts/phase4_acceptance_check.py printed Phase 4's.
        check.note("  supported cells by regime | quantity | metric, with each wording's count:")
        for (regime, name, metric), cases in sorted(grouped.items()):
            detail = ", ".join(f"{text!r} {count}" for text, count in sorted(cases.items()))
            check.note(f"    {regime} | {name} | {metric}: {detail}")
    with check.part("the claim word"):
        tables = ctx.tables()
        found = []
        for name, rows in tables.tables.items():
            for row in rows:
                for column, value in row.items():
                    for text in (column, value):
                        if isinstance(text, str) and CLAIM_WORD.search(text):
                            found.append(f"{name} column {column}: {_short(text, 80)}")
        for name, document in tables.documents.items():
            for text in _strings(document):
                if CLAIM_WORD.search(text):
                    found.append(f"{name}: {_short(text, 80)}")
        for text in _strings(tables.manifest):
            if CLAIM_WORD.search(text):
                found.append(f"{MANIFEST_FILE}: {_short(text, 80)}")
        for item in sorted(set(found))[:10]:
            check.fail(f"an output claims what no frozen rule licenses: {item}")
        check.evidence["claim_word_found"] = len(found)
        if not found:
            check.note("no table, document, or manifest claims that the methods are the same")
    with check.part("wording only on interpreted effects"):
        mark = check.mark()
        tables = ctx.tables()
        for name, rows in tables.tables.items():
            columns = set().union(*(row.keys() for row in rows)) if rows else set()
            for column in sorted(columns):
                if column.endswith("_near_zero_wording"):
                    base = column[: -len("_near_zero_wording")]
                    if base not in INTERPRETED_EFFECTS or name in (L2_TABLE,):
                        check.fail(f"{name}: {column} words a quantity that is not an "
                                   "interpreted effect")
            for row in rows:
                if row.get("row_kind") == "contrast" and any(
                        key.endswith("_near_zero_wording") and row.get(key) is not None
                        for key in row):
                    check.fail(f"{name}: a region contrast carries near-zero wording")
        check.note_if_clean(mark, "near-zero wording appears on interpreted effects only, never "
                                  "on an L2 companion or a region contrast")
    with check.part("decision 2's outcomes"):
        mark = check.mark()
        count = _outcomes(ctx.tables(), check)
        check.note_if_clean(mark, f"{count} headline cells and the measured outcome are called "
                                  "as decision 2 calls them")
    return check.result()


def _outcomes(tables: Any, check: _Check) -> int:
    """Each headline cell's outcome, called again from its own interval.

    Returns how many headline cells were called.
    """
    primary = tables.rows(PRIMARY_TABLE)
    splat = {(row["metric"], row["analysis"], row["axis"], row["bin"]): row
             for row in tables.rows(SPLAT_TABLE)}
    by_cell = {(row["metric"], row["analysis"], row["axis"], row["bin"]): row for row in primary}
    for key, row in by_cell.items():
        where = " ".join(str(part) for part in key)
        try:
            outcome = call_outcome(row["delta_learn_pp"], row["delta_learn_pp_ci_low"],
                                   row["delta_learn_pp_ci_high"], row["supported"] is True)
            other = splat.get(key)
            sp = None if other is None else call_outcome(
                other["delta_learn_sp"], other["delta_learn_sp_ci_low"],
                other["delta_learn_sp_ci_high"], other["supported"] is True)
        except OutcomeAnomaly as error:
            check.fail(f"{where}: the cell cannot be called: {error}")
            continue
        engaged = row.get("delta_learn_pp_near_zero") is True
        expected = {
            "delta_learn_pp_outcome": outcome,
            "delta_learn_pp_outcome_wording": OUTCOME_WORDING.get(outcome) if outcome else None,
            "delta_learn_pp_qualifier": (row.get("delta_learn_pp_near_zero_wording")
                                         if outcome is not None and engaged else None),
            "delta_learn_sp_outcome": sp,
            "outcome_49": outcome_49(outcome, sp),
        }
        for field, want in expected.items():
            if row.get(field) != want:
                check.fail(f"{where}: {field} is {row.get(field)!r}, and decision 2 calls "
                           f"{want!r}")
        if other is not None and other.get("delta_learn_sp_outcome") != sp:
            check.fail(f"{where}: the splat-pool table calls "
                       f"{other.get('delta_learn_sp_outcome')!r}, and decision 2 calls {sp!r}")
        partner = by_cell.get(("raw" if key[0] == "centered" else "centered",) + key[1:])
        if partner is not None:
            both = row.get("supported") is True and partner.get("supported") is True
            sensitive = (_sign(row["delta_learn_pp"]) != _sign(partner["delta_learn_pp"])
                         if both else None)
            if row.get("metric_sensitive") != sensitive:
                check.fail(f"{where}: metric_sensitive is {row.get('metric_sensitive')!r}, and "
                           f"the two metrics' signs give {sensitive!r}")
    measured = tables.tables.get(MEASURED_OUTCOME_TABLE) or []
    if not measured:
        check.fail(f"the tables carry no {MEASURED_OUTCOME_TABLE}")
    for row in measured:
        source = by_cell.get((row.get("metric"), row.get("scope"), SCOPE_AXIS, "all"))
        if source is None:
            check.fail(f"the measured outcome's {row.get('scope')} row under {row.get('metric')} "
                       "has no headline row")
            continue
        for field, column in (("outcome", "delta_learn_pp_outcome"),
                              ("outcome_wording", "delta_learn_pp_outcome_wording"),
                              ("qualifier", "delta_learn_pp_qualifier"),
                              ("near_zero", "delta_learn_pp_near_zero"),
                              ("delta_learn_sp_outcome", "delta_learn_sp_outcome"),
                              ("outcome_49", "outcome_49"),
                              ("metric_sensitive", "metric_sensitive"),
                              ("supported", "supported"),
                              ("visibility_bucket", "visibility_bucket")):
            if row.get(field) != source.get(column):
                check.fail(f"the measured outcome's {row.get('scope')} {row.get('metric')} row "
                           f"names {field} {row.get(field)!r}, and the headline row "
                           f"{source.get(column)!r}")
    return len(by_cell)


def check_17_splat_symmetry(ctx: AcceptanceContext) -> Outcome:
    """The operational splat arm reads context depth only, verified before evaluation."""
    from .phase4 import PHASE3_SCORE_RECON_TOL

    check = _Check()
    with check.part("integration gate step 9"):
        mark = check.mark()
        step = ctx.gate_step("9")
        calls = step.get("transport_calls") if isinstance(step.get("transport_calls"), list) else []
        if step.get("verdict") != SPLAT_SYMMETRY_VERDICT:
            check.fail(f"gate step 9's verdict is {step.get('verdict')!r}")
        if not calls:
            check.fail("gate step 9 found no transport call to audit")
        tainted = [call for call in calls if any(
            name in str(call.get("first_arg")) for name in ("est_target", "target_aligned"))]
        if tainted:
            check.fail(f"gate step 9 lists transport calls on target depth {tainted}")
        check.evidence["step9_calls"] = len(calls)
        check.note_if_clean(mark, f"gate step 9 audited {len(calls)} transport calls: context "
                                  "side only")
    with check.part("verified before the operational comparison was used"):
        mark = check.mark()
        stamped = _utc(ctx.receipt(KIND_INTEGRATION).payload.get("stamped_utc"))
        first = min((_utc(r.get("written_utc")) for r in ctx.records().values()
                     if _utc(r.get("written_utc")) is not None), default=None)
        if stamped is None or first is None or not stamped < first:
            check.fail("the integration receipt was not stamped before the first evaluation")
        ctx.integration()
        check.note_if_clean(mark, "the gate that verified the symmetry is bound to E and was "
                                  "stamped before the first evaluation")
    with check.part(f"the splat transport in {REFERENCE_SOURCE} at E"):
        mark = check.mark()
        tree = _parse(ctx.repo.text(ctx.commit_e(), REFERENCE_SOURCE), f"{REFERENCE_SOURCE} at E")
        function = _function(tree, "recompute_reference_arms")
        plans = _calls(function, "transport_plan")
        if not plans:
            check.fail("recompute_reference_arms makes no transport_plan call")
        for call in plans:
            depth = _first_argument(call, "depth")
            names = _names(depth) if depth is not None else set()
            if "context_aligned" not in names or names - {"torch", "context_aligned",
                                                          "torch_dtype"}:
                check.fail(f"transport_plan at line {call.lineno} reads depth "
                           f"{ast.unparse(depth) if depth is not None else None}, not the aligned "
                           "context map")
        for call in _calls(function, "apply_transport_plan") + plans:
            if "target_aligned" in _names(call) or "est_target" in _names(call):
                check.fail(f"line {call.lineno} hands target depth to the splat transport")
        bound = _bindings(function, "context_aligned")
        if not (len(bound) == 1 and _call_name(bound[0][1]) == "aligned_depth"
                and isinstance(_argument(bound[0][1], 1, "est"), ast.Name)
                and _argument(bound[0][1], 1, "est").id == "est_context"):
            check.fail("context_aligned is not bound once, to the context frame's aligned depth")
        check.evidence["transport_plan_calls"] = [ast.unparse(call) for call in plans]
        check.note_if_clean(mark, "the splat transport at E reads the context frame's aligned "
                                  "depth only")
    with check.part("every scene's splat reconciliation"):
        mark = check.mark()
        residuals = {}
        for scene, record in ctx.records().items():
            residual = (record.get("audit") or {}).get("worst_splat_residual")
            residuals[scene] = residual
            if not (_finite(residual) and residual <= PHASE3_SCORE_RECON_TOL):
                check.fail(f"{scene}: the splat arm reconciled with Phase 4 to {residual}, not "
                           f"within {PHASE3_SCORE_RECON_TOL}")
        check.evidence["worst_splat_residual"] = residuals
        check.note_if_clean(mark, f"every scene's splat cells matched Phase 4's bit for bit and "
                                  f"its scores within {PHASE3_SCORE_RECON_TOL}")
    return check.result()


def check_18_no_target_lift(ctx: AcceptanceContext) -> Outcome:
    """The headline gap is Context-Lift minus Predict-with-Depth, and nothing else."""
    check = _Check()
    with check.part("the headline table"):
        mark = check.mark()
        tables = ctx.tables()
        primary = tables.rows(PRIMARY_TABLE)
        for row in primary:
            where = f"{row.get('metric')} {row.get('analysis')} {row.get('bin')}"
            for column, value in row.items():
                if column.startswith("tl_") or "tl_reference" in column or \
                        "target_lift" in column:
                    check.fail(f"{PRIMARY_TABLE} {where}: column {column} carries the target-lift "
                               "reference")
                if isinstance(value, str) and any(name in value for name in (
                        TL_REFERENCE, "Target-Lift", "TL-Reference")):
                    check.fail(f"{PRIMARY_TABLE} {where}: {column} names the target-lift reference")
            gap, cl, predict = (row.get(name) for name in ("delta_learn_pp", "cl_transport",
                                                           "predict_with_depth"))
            if _finite(gap) or _finite(cl) or _finite(predict):
                if not (_finite(gap) and _finite(cl) and _finite(predict)
                        and abs(gap - (cl - predict)) <= RECOMPUTE_TOLERANCE):
                    check.fail(f"{PRIMARY_TABLE} {where}: delta_learn_pp {gap} is not "
                               f"Context-Lift {cl} minus Predict-with-Depth {predict}")
            if row.get("headline_definition") != HEADLINE_DEFINITION:
                check.fail(f"{PRIMARY_TABLE} {where}: the headline is defined as "
                           f"{row.get('headline_definition')!r}")
        check.note_if_clean(mark, f"{len(primary)} headline rows: delta_learn_pp is Context-Lift "
                                  "minus Predict-with-Depth, and no column or string names the "
                                  "target-lift reference")
    with check.part("the formulation table's label"):
        mark = check.mark()
        tables = ctx.tables()
        for row in tables.rows(FORMULATION_TABLE):
            if row.get("table_label") != TABLE_LABELS[FORMULATION_TABLE]:
                check.fail(f"{FORMULATION_TABLE}: a row is labelled {row.get('table_label')!r}")
        check.note_if_clean(mark, "the target-lift reference appears only in the formulation "
                                  "table, labelled a diagnostic, not the learned-versus-explicit "
                                  "estimand")
    with check.part("the reference table's rung"):
        mark = check.mark()
        tables = ctx.tables()
        by_cell = {(row["metric"], row["analysis"], row["axis"], row["bin"]): row
                   for row in tables.rows(PRIMARY_TABLE)}
        for row in tables.tables.get(REFERENCE_TABLE) or []:
            if row.get("phase4_matched") is not True:
                continue
            key = (row["metric"], row["analysis"], row["axis"], row["bin"])
            if row.get("learned_vs_explicit_limitation_estimator") != "delta_learn_pp":
                check.fail(f"{REFERENCE_TABLE} {key}: the learned-versus-explicit rung is "
                           f"{row.get('learned_vs_explicit_limitation_estimator')!r}")
            source = by_cell.get(key)
            if source is None or not _close(row.get("learned_vs_explicit_limitation_estimate"),
                                            source.get("delta_learn_pp")):
                check.fail(f"{REFERENCE_TABLE} {key}: the rung is not the headline's "
                           "delta_learn_pp")
        for row in tables.tables.get(MEASURED_OUTCOME_TABLE) or []:
            source = by_cell.get((row.get("metric"), row.get("scope"), SCOPE_AXIS, "all"))
            if source is None or not _close(row.get("delta_learn_pp"),
                                            source.get("delta_learn_pp")):
                check.fail(f"{MEASURED_OUTCOME_TABLE}: the {row.get('scope')} "
                           f"{row.get('metric')} gap is not the headline's")
        check.note_if_clean(mark, "the Phase 5 rung and the measured outcome are the headline's "
                                  "delta_learn_pp")
    with check.part(f"where {REPORT_SOURCE} at R names the target-lift reference"):
        mark = check.mark()
        tree = _parse(ctx.repo.text(ctx.repo.head(), REPORT_SOURCE), f"{REPORT_SOURCE} at R")
        quantities = _literal_tuple(tree, "PRIMARY_QUANTITIES")
        if quantities is None:
            raise Unavailable("PRIMARY_QUANTITIES is not a literal tuple")
        leaked = [q for q in quantities if "tl_" in q or "formulation" in q]
        if leaked:
            check.fail(f"the headline table is built from {leaked}")
        places = sorted({inside for _, inside in _enclosing_functions(tree, "TL_REFERENCE")},
                        key=str)
        allowed = {"formulation_table", "_assert_no_target_lift"}
        if set(places) - allowed:
            check.fail(f"{REPORT_SOURCE} names TL_REFERENCE in {places}, outside "
                       f"{sorted(allowed)}")
        check.evidence["tl_reference_named_in"] = places
        check.note_if_clean(mark, f"the reporting code at R builds the headline from "
                                  f"{list(quantities)} and names the target-lift reference only "
                                  f"in {places}")
    return check.result()


def check_19_stable_curve(ctx: AcceptanceContext) -> Outcome:
    """Decision 5, from the training records each run record embeds."""
    check = _Check()
    tolerance = ctx.analysis.path_agreement_tolerance
    with check.part("the embedded training records"):
        mark = check.mark()
        records = ctx.records()
        by_fold: dict[Any, list[tuple[str, Any]]] = {}
        for scene, record in records.items():
            by_fold.setdefault(record.get("fold"), []).append(
                (scene, record.get("training_record_contents")))
        runs: dict[str, Any] = {}
        for fold, embedded in sorted(by_fold.items(), key=lambda item: str(item[0])):
            first_scene, contents = embedded[0]
            for scene, other in embedded[1:]:
                if _canonical(other) != _canonical(contents):
                    check.fail(f"{scene} and {first_scene} of fold {fold} embed different "
                               "training records")
            contents = contents if isinstance(contents, dict) else {}
            for seed in ctx.seeds:
                key = fold_seed_key(fold, seed)
                record = contents.get(str(seed))
                if not isinstance(record, dict):
                    check.fail(f"{key}: no embedded training record")
                    continue
                history = record.get("history") if isinstance(record.get("history"), list) else []
                stopped = record.get("stopped_early") is True
                stable = stable_validation_curve(history, stopped, tolerance)
                scores = [entry[1] for entry in history]
                finite = [score for score in scores if _finite(score)]
                runs[key] = {"n_validations": len(history), "stopped_early": stopped,
                             "best": max(finite) if finite else None,
                             "last": scores[-1] if scores else None, "stable": stable}
                if not stable:
                    check.fail(f"{key}: the validation curve is not stable: {len(history)} "
                               f"validations, stopped early {stopped}, last "
                               f"{runs[key]['last']} against best {runs[key]['best']}")
        missing = sorted({fold_seed_key(f, s) for f in by_fold for s in ctx.seeds} - set(runs))
        if missing:
            check.fail(f"no embedded training record for {missing}")
        folds_missing = sorted({f.index for f in ctx.folds} - set(by_fold), key=str)
        if folds_missing and set(ctx.expected_scenes) == set(evaluation_scenes(ctx.folds)):
            check.fail(f"no evaluated scene embeds the training records of fold(s) "
                       f"{folds_missing}")
        check.evidence["runs"] = runs
        stable = sum(1 for run in runs.values() if run["stable"])
        check.note(f"{stable} of {len(runs)} training runs have a stable validation curve: every "
                   f"score finite, at least two validations, and early stopping or a last score "
                   f"within {tolerance} of the best")
        check.note_if_clean(mark, "every scene of a fold embeds the same training records")
    return check.result()


def check_20_headline_figures(ctx: AcceptanceContext) -> Outcome:
    """The figures as drawn, from the tables as they are, with the headline figures."""
    check = _Check()
    with check.part("the published figures"):
        mark = check.mark()
        manifest = ctx.checked_figures()
        figures = manifest.get("figures") if isinstance(manifest.get("figures"), dict) else {}
        files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
        tables = ctx.published()
        for name in HEADLINE_FIGURES:
            entry = figures.get(name) if isinstance(figures.get(name), dict) else None
            path = ctx.figures_dir / name
            if entry is None or name not in files:
                check.fail(f"the headline figure {name} is not among the published figures")
                continue
            if entry.get("headline") is not True:
                check.fail(f"{name} is not marked a headline figure")
            if not path.is_file() or path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                check.fail(f"{name} is not a PNG image")
            drawn = entry.get("tables") if isinstance(entry.get("tables"), dict) else {}
            if set(drawn) != set(FIGURE_TABLES[name]):
                check.fail(f"{name} was drawn from {sorted(drawn)}, not {FIGURE_TABLES[name]}")
            for table, sha in drawn.items():
                if tables.files.get(table) != sha:
                    check.fail(f"{name} was drawn from {table} {sha}, and the current {table} is "
                               f"{tables.files.get(table)}")
        record = manifest.get("run_record") if isinstance(manifest.get("run_record"), dict) else {}
        run = ctx.run_values()
        for field in RUN_FIELDS:
            if record.get(field) != run.get(field):
                check.fail(f"the figures name {field} {_short(record.get(field))}, and the run "
                           f"{_short(run.get(field))}")
        check.evidence["figures"] = dict(files)
        check.note_if_clean(mark, f"{len(files)} published figures are as drawn, from the "
                                  f"current tables, the headline figures "
                                  f"{list(HEADLINE_FIGURES)} among them")
    with check.part("the tables the figures were drawn from"):
        ctx.tables()
        check.note("the tables they were drawn from are the evaluated run's own")
    return check.result()


def check_21_full_suite(ctx: AcceptanceContext) -> Outcome:
    """python -m pytest at R, run in a subprocess, every test green."""
    check = _Check()
    with check.part("the full suite at R"):
        suite = ctx.suite()
        counts = suite.get("counts") if isinstance(suite.get("counts"), Mapping) else {}
        summary = suite.get("summary")
        check.evidence.update(
            summary=summary, counts=dict(counts), returncode=suite.get("returncode"),
            command=suite.get("command"), head=suite.get("head_before"),
            output_tail=suite.get("output_tail"))
        if suite.get("returncode") != 0:
            check.fail(f"the suite exited {suite.get('returncode')}: {summary}")
        if counts.get("failed") or counts.get("errors"):
            check.fail(f"the suite reports {summary}")
        if not counts.get("passed"):
            check.fail("no test passed")
        if suite.get("head_before") != suite.get("head_after"):
            check.fail(f"HEAD moved from {suite.get('head_before')} to {suite.get('head_after')} "
                       "while the suite ran")
        if suite.get("head_before") != ctx.repo.head():
            check.fail(f"the suite ran at {suite.get('head_before')}, and the verdict names R "
                       f"{ctx.repo.head()}")
        for when in ("worktree_before", "worktree_after"):
            if suite.get(when):
                check.fail(f"the worktree is not clean {when.split('_')[1]} the suite: "
                           f"{suite.get(when)[:5]}")
        check.note(f"the suite at {suite.get('head_before')}: {summary}")
    with check.part("the reporting commit"):
        mark = check.mark()
        head = ctx.repo.head()
        for problem in ctx.ancestry_problems(("worktree",)):
            check.fail(problem)
        verified = ctx.ancestry().get("head")
        if verified is not None and verified != head:
            check.fail(f"the code ancestry was verified at {verified}, not at R {head}")
        now = ctx.repo.current_head()
        if now != head:
            check.fail(f"HEAD moved from R {head} to {now} while acceptance ran")
        check.note_if_clean(mark, f"HEAD stayed at R {head} while acceptance ran")
    return check.result()


CONDITIONS: tuple[tuple[int, Callable[[AcceptanceContext], Outcome]], ...] = (
    (1, check_01_phase4_accepted),
    (2, check_02_preregistration),
    (3, check_03_context_lift_no_target_depth),
    (4, check_04_predictor_no_target),
    (5, check_05_rotation_and_geometry),
    (6, check_06_identical_inputs),
    (7, check_07_scene_separation),
    (8, check_08_test_seal),
    (9, check_09_tiny_overfit),
    (10, check_10_frozen_before_test),
    (11, check_11_validation_only_selection),
    (12, check_12_fixed_support),
    (13, check_13_support_not_shrunk),
    (14, check_14_paired_bootstrap),
    (15, check_15_definitions_frozen),
    (16, check_16_near_zero),
    (17, check_17_splat_symmetry),
    (18, check_18_no_target_lift),
    (19, check_19_stable_curve),
    (20, check_20_headline_figures),
    (21, check_21_full_suite),
)


def _run_condition(number: int, function: Callable[[AcceptanceContext], Outcome],
                   ctx: AcceptanceContext) -> ConditionResult:
    title = CONDITION_TITLES[number]
    try:
        ok, notes, evidence = function(ctx)
    except Exception as error:  # noqa: BLE001
        # A condition that cannot be re-derived fails. It never passes.
        ok, notes, evidence = False, [f"FAIL the condition raised {type(error).__name__}: "
                                      f"{error}"], {"error": repr(error)}
    return ConditionResult(number, title, bool(ok), tuple(notes), _plain(evidence))


def _plain(value: Any) -> Any:
    """A JSON-ready copy: tuples become lists, keys become strings."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [_plain(item) for item in items]
    if isinstance(value, Path):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def _safe(compute: Callable[[], Any]) -> Any:
    try:
        return _plain(compute())
    except Exception as error:  # noqa: BLE001
        # A field the verdict cannot fill is recorded as unavailable.
        return {"unavailable": f"{type(error).__name__}: {error}"}


def _findings(tables: Any) -> dict[str, list[dict[str, Any]]]:
    """Cells the headline table flags, reported beside the verdict and never failed."""
    flags = {
        "metric_sensitive": lambda row: row.get("metric_sensitive") is True,
        "outcome_49": lambda row: row.get("outcome_49") is True,
        "lead_within_read": lambda row: row.get("lead_within_read") is True,
        "read_deficit_anomaly": lambda row: row.get("read_deficit_anomaly") is True,
        "near_zero_engaged": lambda row: row.get("delta_learn_pp_near_zero") is True,
        "seeds_cross_zero": lambda row: row.get("delta_learn_pp_seed_crosses_zero") is True,
    }
    found: dict[str, list[dict[str, Any]]] = {name: [] for name in flags}
    for row in tables.rows(PRIMARY_TABLE):
        cell = {field: row.get(field) for field in ("metric", "analysis", "axis", "bin",
                                                    "supported")}
        for name, flagged in flags.items():
            if flagged(row):
                found[name].append(cell)
    return found


def _measured_outcome(tables: Any) -> list[dict[str, Any]]:
    """The published measured-outcome rows, every column kept.

    Decision 2 shows the cross-path terms beside outcome 49 and never collapses
    its two statements, so each row is copied whole: both gaps with their
    intervals and support, the terms, the read deficit, the counts, and the
    visibility bucket.
    """
    return [dict(row) for row in tables.tables.get(MEASURED_OUTCOME_TABLE) or []]


def _withheld(conditions: Sequence[ConditionResult]) -> dict[str, Any]:
    """What a failed verdict records in place of the measured outcome.

    Specification step 45 makes any failure a stop, and step 52 records the
    measured outcome only when every condition holds.
    """
    return {"withheld": "acceptance did not pass. Any failure is a stop, so the "
                        "learned-versus-explicit difference is not reported",
            "failed_conditions": [c.number for c in conditions if not c.ok]}


def _verdict_record(ctx: AcceptanceContext, conditions: list[ConditionResult]) -> dict[str, Any]:
    passed = ([c.number for c in conditions] == sorted(CONDITION_TITLES)
              and all(c.ok for c in conditions))

    def context_lift() -> dict[str, Any]:
        commit = ctx.commit_e()
        return {"path": CONTEXT_LIFT_SOURCE,
                "sha256_at_e": _sha256(ctx.repo.blob(commit, CONTEXT_LIFT_SOURCE)),
                "git_blob_at_e": ctx.repo.blob_id(commit, CONTEXT_LIFT_SOURCE)}

    def tables_block() -> dict[str, Any]:
        tables = ctx.published()
        return {"directory": str(ctx.tables_dir), "manifest_sha256": tables.manifest_sha256,
                "files": dict(tables.files)}

    def figures_block() -> dict[str, Any]:
        path = ctx.figures_dir / MANIFEST_FILE
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return {"directory": str(ctx.figures_dir), "manifest_sha256": sha256_file(path),
                "files": dict(manifest.get("files") or {})}

    def ledger_block() -> dict[str, Any]:
        attempts = ctx.attempts()
        return {"directory": str(ctx.evidence_dir / LEDGER_DIRECTORY),
                "attempts": len(attempts), "rendered": None, "sha256": None}

    def suite_block() -> dict[str, Any]:
        suite = ctx.suite()
        return {field: suite.get(field) for field in ("summary", "counts", "returncode",
                                                      "head_before")}

    record: dict[str, Any] = {
        "kind": VERDICT_KIND,
        "acceptance_version": ACCEPTANCE_VERSION,
        "level": ctx.level,
        "primary_level": ctx.primary_level,
        # Decision 4: only the primary level's verdict is the Rung 2 verdict.
        # Every other level's is reported beside it.
        "level_role": level_role(ctx.cfg, ctx.level),
        "rung2_verdict": ctx.level == ctx.primary_level,
        "passed": passed,
        "failed_conditions": [c.number for c in conditions if not c.ok],
        "evaluation_commit": _safe(ctx.commit_e),
        "training_commit": _safe(ctx.commit_e),
        "report_commit": _safe(ctx.repo.head),
        "design_correction": _safe(lambda: {
            "document": DESIGN_CORRECTION,
            "commits": ctx.repo.commits_touching(DESIGN_CORRECTION)}),
        "amendments": _safe(lambda: {
            "path": AMENDMENTS_DOCUMENT,
            "sha256_at_e": _sha256(ctx.repo.blob(ctx.commit_e(), AMENDMENTS_DOCUMENT))}),
        "scene_split_hash": fold_digest(ctx.folds),
        "folds": [{"index": fold.index, "train": list(fold.train), "val": list(fold.val),
                   "test": list(fold.test)} for fold in ctx.folds],
        "seeds": list(ctx.seeds),
        "config_digest": ctx.cfg.digest(),
        "training_config_digest": ctx.settings.digest(),
        "measurement_digest": ctx.analysis.measurement_digest(),
        "analysis_reporting_digest": ctx.analysis.reporting_digest(),
        "mean_vector_digest": _safe(lambda: ctx.common("mean_vector_digest")),
        "context_lift": _safe(context_lift),
        "checkpoints": _safe(lambda: {key: c.sha256 for key, c in ctx.checkpoints().items()}),
        "training_records": _safe(lambda: {key: r.sha256 for key, r in ctx.training().items()}),
        "controls_sha256": _safe(lambda: ctx.controls().sha256),
        "evaluation_artifacts": _safe(lambda: dict(ctx.scan().sha)),
        "tables": _safe(tables_block),
        "figures": _safe(figures_block),
        "receipts": _safe(lambda: {
            stem: {"path": str(ctx.receipt(kind).path), "sha256": ctx.receipt(kind).sha256}
            for kind, stem in ctx.receipt_stems().items()}),
        "evaluation_ledger": _safe(ledger_block),
        # Read only through the tables bound to the evaluated run, and only when
        # every condition holds. A failed verdict withholds both.
        "measured_outcome": (_safe(lambda: _measured_outcome(ctx.tables())) if passed
                             else _withheld(conditions)),
        "findings": (_safe(lambda: _findings(ctx.tables())) if passed
                     else _withheld(conditions)),
        "suite": _safe(suite_block),
        "conditions": [condition.as_dict() for condition in conditions],
        "created_utc": utc_timestamp(),
        "environment": environment_identity(),
    }
    return record


def format_report(verdict: AcceptanceVerdict) -> str:
    """The verdict as the acceptance mode prints it."""
    record = verdict.record
    rule = "=" * 74
    lines = [rule, f"PHASE 5 ACCEPTANCE, level {verdict.level}", rule,
             f"evaluation commit E  {record.get('evaluation_commit')}",
             f"reporting commit R   {record.get('report_commit')}"]
    for condition in verdict.conditions:
        lines.append("")
        lines.append(f"{condition.number}. {condition.title}: "
                     f"{'PASS' if condition.ok else 'FAIL'}")
        lines.extend(f"    {note}" for note in condition.notes)
    total = len(verdict.conditions)
    held = sum(1 for condition in verdict.conditions if condition.ok)
    primary = record.get("primary_level")
    role = record.get("level_role") or "non-primary"
    # Decision 4: Rung 2 is decided at the primary level alone.
    rung2 = record.get("rung2_verdict") is True and verdict.level == primary
    lines += ["", rule]
    if verdict.passed and rung2:
        lines.append(f"ACCEPTED: {held} of {total} conditions satisfied. Rung 2 is accepted with "
                     "the measured outcome below, whichever method it favours.")
    elif verdict.passed:
        lines.append(f"SATISFIED: {held} of {total} conditions hold for the {role} level "
                     f"{verdict.level}. This is not the Rung 2 verdict. Rung 2 is decided at the "
                     f"primary level {primary} alone, and this level is reported beside it, "
                     "reporting_rules.md decision 4.")
    else:
        lines.append(f"NOT SATISFIED: {total - held} of {total} conditions failed: "
                     f"{verdict.failed()}")
        lines.append("The measured outcome is withheld: a failed condition is a stop, and the "
                     "learned-versus-explicit difference is not interpreted.")
    measured = record.get("measured_outcome")
    if verdict.passed and isinstance(measured, list) and measured:
        lines.append("")
        if rung2:
            lines.append("Measured outcome, decision 2, per regime and metric:")
        else:
            lines.append(f"Measured outcome at the {role} level {verdict.level}, decision 2, per "
                         "regime and metric. It sits beside the primary result and never "
                         "replaces it:")
        for row in measured:
            estimate = row.get("delta_learn_pp")
            low, high = row.get("delta_learn_pp_ci_low"), row.get("delta_learn_pp_ci_high")
            interval = (f"{estimate:+.4f} [{low:+.4f}, {high:+.4f}]"
                        if all(_finite(v) for v in (estimate, low, high)) else str(estimate))
            lines.append(f"  {str(row.get('metric')):<8} {str(row.get('row_label')):<30} "
                         f"delta_learn_pp {interval}  outcome {row.get('outcome')}"
                         + (f", {row.get('qualifier')}" if row.get("qualifier") else "")
                         + ("  outcome 49" if row.get("outcome_49") else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The acceptance mode
# ---------------------------------------------------------------------------

def evaluate_acceptance(
    cfg: Any,
    analysis: Any,
    config_path: Path,
    level: str,
    **injection: Any,
) -> AcceptanceVerdict:
    """Every condition re-derived for one evaluated level. Nothing is written.

    cfg must be the configuration at config_path, and level must be declared
    in it. injection passes to AcceptanceContext, for tests. The acceptance
    mode passes none of them:

    - expected_scenes and folds, as lot.phase5_provenance takes them;
    - repo_root, the repository holding E, by default this one;
    - tables_root and figures_root, where the published outputs are;
    - suite_runner(repo_root), by default run_suite;
    - phase4_check(cfg, repo_root), by default run_phase4_acceptance_check;
    - checkpoint_loader(path), by default load_checkpoint_state;
    - recount(cfg, analysis, level, scene, pairs), by default
      recount_primary_support;
    - source_reader(commit, path), a file's bytes at a commit or None, by
      default git_blob on repo_root.
    """
    from .phase5 import load_phase5_config

    declared = (cfg.primary_alignment_level, *cfg.sensitivity_alignment_levels,
                *cfg.diagnostic_alignment_levels)
    if level not in declared:
        raise ValueError(f"level {level!r} is not declared in the configuration: {declared}")
    if load_phase5_config(Path(config_path)).digest() != cfg.digest():
        raise ValueError(f"the configuration object is not the configuration at {config_path}; "
                         "the receipts are verified against that file")
    ctx = AcceptanceContext(cfg, analysis, config_path, level, **injection)
    ctx.prefetch()
    conditions = [_run_condition(number, function, ctx) for number, function in CONDITIONS]
    return AcceptanceVerdict(level, conditions, _verdict_record(ctx, conditions))


def verdict_path(evidence_dir: Path, level: str) -> Path:
    """Where the verdict of one level is written."""
    return Path(evidence_dir) / f"acceptance_{level}.json"


def run_acceptance(
    cfg: Any,
    analysis: Any,
    config_path: Path,
    level: str,
    *,
    stream: Any = None,
    **injection: Any,
) -> int:
    """The acceptance mode: render the ledger, re-derive, write the verdict once.

    The evaluation ledger's attempt files are rendered to evaluation_ledger.jsonl
    first, through lot.phase5_modes.build_evaluation_ledger, which keeps any
    earlier rendering. Every condition is then re-derived, the verdict is
    written to evidence/acceptance_{level}.json through write_once, which keeps
    any earlier verdict, and the report is printed. Returns 0 when every
    condition holds and 1 otherwise.
    """
    stream = stream if stream is not None else sys.stdout
    evidence = Path(cfg.evidence_dir)
    rendering: dict[str, Any]
    try:
        outcome = build_evaluation_ledger(evidence / LEDGER_DIRECTORY, evidence / LEDGER_FILE)
        written = Path(outcome["written"])
        rendering = {"rendered": str(written), "sha256": sha256_file(written),
                     "archived_previous": outcome.get("archived_previous")}
    except (OSError, ValueError) as error:
        rendering = {"rendered": None, "sha256": None,
                     "error": f"{type(error).__name__}: {error}"}
    verdict = evaluate_acceptance(cfg, analysis, config_path, level, **injection)
    block = verdict.record.get("evaluation_ledger")
    verdict.record["evaluation_ledger"] = {**(block if isinstance(block, dict) else {}),
                                           **rendering}
    outcome = write_once(verdict_path(evidence, level),
                         json.dumps(verdict.record, indent=2, sort_keys=True, default=str))
    print(format_report(verdict), file=stream)
    print(f"\nverdict written to {outcome['written']}", file=stream)
    if outcome["archived_previous"]:
        print(f"previous verdict kept at {outcome['archived_previous']}", file=stream)
    return 0 if verdict.passed else 1
