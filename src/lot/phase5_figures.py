"""Phase 5 figures: Stream AC, specification steps 39 to 43, from the tables alone.

The figures mode draws one evaluated level's published tables and nothing
else. It never reads a row of an evaluation parquet. Like the tables mode, it
is licensed by the evaluated run's own provenance, never by receipts at the
reporting commit. Before a figure is drawn:

1. lot.phase5_provenance.require_evaluated_run checks the run.
2. read_published_tables checks the tables before it returns a row.
   MANIFEST.json must be the tables mode's, of this report version and level.
   The directory must hold exactly the files it names, each with the sha256
   it names. Each file is read once, and its rows come from the bytes that
   were hashed. Every table carries its run record inside it, names itself
   and its label, and names the run MANIFEST.json names, with no input that
   MANIFEST.json lacks. Each JSON output names the same run.
3. bind_tables_to_run checks that the tables name that run: its commit, its
   digests, seeds, folds, scenes, bootstrap, and licence. Every file the run
   is read from must be among the tables' inputs, by sha256, and each
   evaluation parquet by its name too.

A non-primary level is drawn only beside the primary level's verified
tables, and only when its run shares the primary run's commit E, digests, and
gate receipts, reporting_rules.md decision 4. Its Figures 1 and 2 name the
level and its role, and say that the headline is the primary level's.

The figures. Each is drawn under centered cosine, the primary metric, with
raw cosine beside it.

- Figure 1, the headline: delta_learn_pp against parallax, translation pairs
  only. The read deficit sits beside each bin, labelled with its direction,
  and is never subtracted. Below, the absolute scores with the No-Warp-Copy
  floor, and Mean-Feature under raw cosine only.
- Figure 2: the same against rotation angle, rotation pairs only, with
  Context-Lift Transport-Only as the known-transform reference.
- Figure 3: delta_formulation by regime and difficulty, labelled a
  diagnostic, not the learned-versus-explicit estimand.
- Figure 4: delta_learn_pp and delta_learn_sp by regime and bin, each on its
  own population, with their path difference on the cells both paths share.
  It makes no claim that their difference is an implementation error.
- Figure 5: boundary and interior, low and high texture, with the paired
  contrasts.
- Supplementary figure S1: every training run's validation curve.
- Supplementary figure S2: the landing-offset curves, at the primary level.

The conventions are those of the Phase 3 and Phase 4 figures. Every bin of
the frozen order has a place on its axis, and a bin with no pairs is blank. A
cell below the frozen support thresholds stays plotted and is greyed. A grey
band marks a position whose headline cell is below support. A hollow grey
marker marks any cell below support. Every populated position shows its
count of camera pairs. Joint cells use a symmetric diverging scale, are
hatched below support, and print their value and count. A blank joint cell is
a combination the camera program cannot produce. Orbit is drawn in joint
cells only. Figures 1 and 2, and every bin panel, refuse an orbit row on a
marginal axis, PROTOCOL 3.3. The pooled row stays in the tables as a summary.
The figures draw regimes only, because claims are made per regime.

Artists carry gids, so a figure can be checked by reading it. A panel is
named by its kind and metric. A grey band is unsupported:{position}, a count
count:{position}, and a series' markers series:{quantity}, with
:below-support appended for the hollow ones. A joint cell is hatched:{i}:{j}
and cell:{i}:{j}. Each figure function saves and closes its figure, and
returns it.

Every figure is drawn into figures/{level}.partial.<id>/. MANIFEST.json,
written last, names each figure's sha256 and the sha256 of every table it was
drawn from. The directory is then published by one rename. An earlier output
is moved aside and never deleted.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .analysis_config import AnalysisConfig
from .datasets import parallax_bin_order, rotation_bin_order
from .evaluate import MEAN_FEATURE
from .figures import _pyplot, _shade_unsupported, assert_single_regime
from .phase5_check import sha256_file
from .phase5_estimands import (
    CL_ORACLE,
    CL_TRANSPORT,
    NO_WARP_COPY,
    OFFSET_BINS,
    OFFSET_WHOLE,
    PER_POINT,
    PREDICT_WITH_DEPTH,
    SPLAT_TRANSPORT,
    TL_REFERENCE,
)
from .phase5_outcomes import METRICS, PRIMARY_METRIC, REGIME_SCOPES
from .phase5_provenance import (
    ProvenanceError,
    output_run_record,
    require_evaluated_run,
    require_primary_chain,
    staged_output,
)
from .phase5_report import (
    ACCOUNTING_FILE,
    ADEQUACY_TABLE,
    CROSS_PATH_TABLE,
    FORMULATION_TABLE,
    HISTORY_TABLE,
    JOINT_AXIS,
    LANDING_OFFSET_PRIMARY_ONLY,
    LANDING_OFFSET_TABLE,
    MANIFEST_FILE,
    NEAR_ZERO_FILE,
    OFFSET_READING_NOTE,
    PARALLAX_AXIS,
    PRIMARY_ROLE,
    PRIMARY_TABLE,
    READ_DEPTH_NOTE,
    REGION_TABLE,
    REPORT_VERSION,
    ROTATION_AXIS,
    SCOPE_AXIS,
    SPLAT_TABLE,
    TABLE_KIND,
    TABLE_LABELS,
    TABLES_DIR,
    _write_json,
    level_role,
    require_primary_tables,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# The output
# ---------------------------------------------------------------------------

# The layout of the figures directory and its MANIFEST.json.
FIGURES_VERSION = 1
FIGURES_DIR = "figures"
FIGURES_MANIFEST_KIND = "phase5_figures_manifest"
TABLES_MANIFEST_KIND = "phase5_tables_manifest"
# The JSON outputs of the tables mode, by the kind of run record each carries.
DOCUMENT_KINDS = {NEAR_ZERO_FILE: "phase5_near_zero", ACCOUNTING_FILE: "phase5_accounting"}

FIGURE1 = "phase5_figure1_gap_vs_parallax.png"
FIGURE2 = "phase5_figure2_gap_vs_rotation.png"
FIGURE3 = "phase5_figure3_formulation_gap.png"
FIGURE4 = "phase5_figure4_primary_vs_operational.png"
FIGURE5 = "phase5_figure5_error_regions.png"
FIGURE_S1 = "phase5_figureS1_validation_curves.png"
FIGURE_S2 = "phase5_figureS2_landing_offset.png"
# Every figure, in drawing order.
FIGURE_FILES = (FIGURE1, FIGURE2, FIGURE3, FIGURE4, FIGURE5, FIGURE_S1, FIGURE_S2)
# The tables each figure is drawn from, and from nothing else.
FIGURE_TABLES = {
    FIGURE1: (PRIMARY_TABLE,),
    FIGURE2: (PRIMARY_TABLE,),
    FIGURE3: (FORMULATION_TABLE,),
    FIGURE4: (PRIMARY_TABLE, SPLAT_TABLE, CROSS_PATH_TABLE),
    FIGURE5: (REGION_TABLE,),
    FIGURE_S1: (HISTORY_TABLE, ADEQUACY_TABLE),
    FIGURE_S2: (LANDING_OFFSET_TABLE,),
}
# The headline figures: specification steps 39 and 40.
HEADLINE_FIGURES = (FIGURE1, FIGURE2)
# Figures drawn at the primary level only, each with the reason.
PRIMARY_LEVEL_ONLY = {FIGURE_S2: LANDING_OFFSET_PRIMARY_ONLY}
FIGURE_TITLES = {
    FIGURE1: "Figure 1: delta_learn_pp against parallax, translation pairs, the headline figure",
    FIGURE2: "Figure 2: delta_learn_pp against rotation angle, rotation pairs",
    FIGURE3: "Figure 3: delta_formulation by regime and difficulty, a diagnostic",
    FIGURE4: "Figure 4: delta_learn_pp and delta_learn_sp by regime and bin, with their "
             "path difference",
    FIGURE5: "Figure 5: error regions, boundary and interior, low and high texture, with "
             "the paired contrasts",
    FIGURE_S1: "Supplementary figure S1: the validation curve of every training run",
    FIGURE_S2: "Supplementary figure S2: the landing-offset curves, primary level only",
}

# The fields of an output's run record that name the evaluated run and the
# code that reported it. Every output of one tables build shares them, the
# level's role among them. created_utc and environment are each output's own,
# and inputs are compared on their own.
RECORD_IDENTITY = (
    "record_version", "phase", "level", "evaluation_commit", "training_commit",
    "report_commit", "code_checked", "config_digest", "fold_digest", "measurement_digest",
    "mean_vector_digest", "analysis_reporting_digest", "eval_version", "phase4_commit",
    "seeds", "folds", "scenes", "bootstrap", "licence", "receipts", "reporting_changes",
    "level_role",
)
# The fields that name the evaluated run alone. Outputs built at different
# reporting commits share these and no others.
RUN_FIELDS = (
    "level", "evaluation_commit", "training_commit", "config_digest", "fold_digest",
    "measurement_digest", "mean_vector_digest", "analysis_reporting_digest", "eval_version",
    "phase4_commit", "seeds", "folds", "scenes", "bootstrap", "licence",
)

# ---------------------------------------------------------------------------
# What the figures say
# ---------------------------------------------------------------------------

GAP_LABEL = f"delta_learn_pp, {CL_TRANSPORT} minus {PREDICT_WITH_DEPTH}"
SPLAT_GAP_LABEL = f"delta_learn_sp, {SPLAT_TRANSPORT} minus {PREDICT_WITH_DEPTH}, splat-pool"
PATH_DIFFERENCE_LABEL = "path_difference_learn, per-point minus splat-pool, common cells"
FORMULATION_GAP_LABEL = f"delta_formulation, {TL_REFERENCE} minus {CL_TRANSPORT}"
READ_DEFICIT_LABEL = (
    f"read deficit of {CL_ORACLE}, ground-truth context depth; a positive deficit works "
    "against Context-Lift"
)
READ_DEFICIT_PRIMARY_ONLY = (
    "The read deficit is reported once, from the primary level's records, so it is not "
    "drawn at this level."
)
KNOWN_TRANSFORM_REFERENCE = f"{CL_TRANSPORT}, the known-transform reference"
NOT_AN_IMPLEMENTATION_ERROR = (
    "The two gaps are read on different populations, and their difference is not "
    "interpreted as an implementation error."
)
BELOW_SUPPORT_MARKER = "hollow grey marker: cell below support"
SUPPORT_NOTE = (
    "Grey band: the headline cell at that position is below support. Hollow grey marker: "
    "that cell is below support. n: camera pairs, shown at every populated position. "
    "Blank position: no pairs."
)
JOINT_NOTE = (
    "Hatched joint cell: below support. Blank joint cell: a combination the camera "
    "program cannot produce, or no pairs."
)

BELOW_SUPPORT = "below-support"
GREY = "0.6"
# One colour per meaning in every figure: Context-Lift blue, Predict-with-Depth
# red, a gap between them black, a floor dark grey or brown.
COLOURS = {
    "delta_learn_pp": "black",
    "read_deficit": "tab:orange",
    "cl_transport": "tab:blue",
    "predict_with_depth": "tab:red",
    "no_warp_copy": "0.3",
    "mean_feature": "tab:brown",
    "delta_learn_sp": "tab:green",
    "path_difference_learn": "tab:purple",
    "delta_formulation": "tab:olive",
    "cl_oracle_offset": "tab:blue",
    "nowarp_offset": "0.3",
    "meanfeat_offset": "tab:brown",
    "contrast_delta_learn_pp": "black",
    "contrast_cl_transport": "tab:blue",
    "contrast_predict_with_depth": "tab:red",
}
# The first and second region of a split in Figure 5.
REGION_COLOURS = ("tab:purple", "tab:green")

# The regions of Figure 5: (panel kind, left region, right region, contrast,
# what the split is).
REGION_SPLITS = (
    ("boundary", "boundary", "interior", "boundary_minus_interior",
     "ground-truth depth boundary and interior"),
    ("texture", "low_texture", "high_texture", "low_minus_high_texture",
     "low and high texture"),
)
REGION_NAMES = {"boundary": "boundary", "interior": "interior",
                "low_texture": "low texture", "high_texture": "high texture"}
CONTRAST_SERIES = (
    ("contrast_delta_learn_pp", "the gap's contrast, delta_learn_pp", "o"),
    ("contrast_cl_transport", f"{CL_TRANSPORT}'s own contrast", "s"),
    ("contrast_predict_with_depth", f"{PREDICT_WITH_DEPTH}'s own contrast", "^"),
)


def _intervals(*names: str) -> tuple[str, ...]:
    return tuple(column for name in names for column in (name, f"{name}_ci_low",
                                                          f"{name}_ci_high"))


# The columns each figure reads, so a table of another layout is refused by
# name rather than drawn half empty.
CELL_COLUMNS = ("metric", "analysis", "axis", "bin", "supported", "n_camera_pairs")
JOINT_COLUMNS = ("rotation_bin", "parallax_bin")
CURVE_COLUMNS = CELL_COLUMNS + _intervals(
    "delta_learn_pp", "cl_transport", "predict_with_depth", "no_warp_copy", "mean_feature")
DEFICIT_COLUMNS = _intervals("read_deficit") + ("read_deficit_supported",
                                                "read_deficit_n_camera_pairs")
FORMULATION_COLUMNS = CELL_COLUMNS + JOINT_COLUMNS + _intervals("delta_formulation")
GAP_COLUMNS = {
    "delta_learn_pp": CELL_COLUMNS + JOINT_COLUMNS + _intervals("delta_learn_pp"),
    "delta_learn_sp": CELL_COLUMNS + JOINT_COLUMNS + _intervals("delta_learn_sp"),
    "path_difference_learn": CELL_COLUMNS + JOINT_COLUMNS + _intervals("path_difference_learn"),
}
REGION_COLUMNS = CELL_COLUMNS + ("region", "row_kind", "path") + _intervals(
    "delta_learn_pp", "contrast_delta_learn_pp", "contrast_cl_transport",
    "contrast_predict_with_depth")
HISTORY_COLUMNS = ("fold", "seed", "validation_index", "step", "validation_centered_cosine",
                   "is_best")
ADEQUACY_COLUMNS = ("fold", "seed", "best_step", "best_validation_centered_cosine",
                    "stable_validation_curve", "stopped_early")
OFFSET_COLUMNS = CELL_COLUMNS + ("offset_label", "offset_lo", "offset_hi") + _intervals(
    "cl_oracle_offset", "nowarp_offset", "meanfeat_offset", "read_deficit")

# The regimes with a primary curve, and the axis each is binned on.
CURVE_AXES = {"rotation": ROTATION_AXIS, "translation": PARALLAX_AXIS}
ORBIT = "orbit"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FiguresStop(ValueError):
    """The published tables cannot be drawn as they are. A stop, not a result."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _float(value: Any) -> float:
    return float(value) if _number(value) else math.nan


def _short(value: Any, limit: int = 120) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _relative(path: Path, root: Path) -> str:
    """A path under root as a POSIX name, or the whole path if outside it."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _require_columns(rows: Sequence[Mapping[str, Any]], columns: Sequence[str],
                     table: str) -> None:
    """Every column a figure reads is in the table, or a stop naming the missing."""
    if not rows:
        raise FiguresStop(f"{table} has no rows to draw")
    present: set[str] = set()
    for row in rows:
        present.update(row)
    missing = [column for column in columns if column not in present]
    if missing:
        raise FiguresStop(
            f"{table} lacks the columns {missing}, which the figures read. The figures "
            f"read the tables of report version {REPORT_VERSION}"
        )


def _order(regime: str, analysis: AnalysisConfig) -> list[str]:
    """The frozen bin order of the axis a regime is binned on."""
    return rotation_bin_order(analysis) if regime == "rotation" else parallax_bin_order(analysis)


# ---------------------------------------------------------------------------
# The published tables, verified before a row is returned
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PublishedTables:
    """One level's published tables, after every check passed.

    files maps each file MANIFEST.json names to its sha256, which each file
    had when it was read. run_record is MANIFEST.json's run record. tables
    maps each parquet to its rows, read from the hashed bytes, and documents
    each JSON output to its content.
    """

    directory: Path
    level: str
    manifest: dict[str, Any]
    manifest_sha256: str
    files: dict[str, str]
    run_record: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]
    documents: dict[str, Any]

    def rows(self, name: str) -> list[dict[str, Any]]:
        """One table's rows, or a stop when the tables hold no such table."""
        if name not in self.tables:
            raise FiguresStop(f"the published tables at {self.directory} hold no {name}")
        return self.tables[name]


def _identity_problems(record: Mapping[str, Any], reference: Mapping[str, Any],
                       where: str) -> list[tuple[str, str]]:
    """An output's run record names the run its MANIFEST.json names, or problems."""
    problems = []
    for field in RECORD_IDENTITY:
        if field not in record:
            problems.append(("run_record", f"{where}: its run record lacks {field}"))
        elif record[field] != reference.get(field):
            problems.append(("run_record", (
                f"{where}: its run record names {field} {_short(record[field])}, and "
                f"{MANIFEST_FILE} names {_short(reference.get(field))}"
            )))
    inputs = record.get("inputs")
    named = reference.get("inputs") if isinstance(reference.get("inputs"), dict) else {}
    if not isinstance(inputs, dict):
        problems.append(("run_record", f"{where}: its run record names no inputs"))
    else:
        for name, sha in sorted(inputs.items()):
            if named.get(name) != sha:
                problems.append(("run_record", (
                    f"{where}: it was read from {name} with sha256 {sha}, and "
                    f"{MANIFEST_FILE} names {named.get(name)}"
                )))
    return problems


def read_published_tables(tables_dir: Path, *, level: str | None = None) -> PublishedTables:
    """One level's published tables, after their hashes and run records are checked.

    MANIFEST.json must be the tables mode's, of REPORT_VERSION, and of level
    when it is given. The directory must hold exactly the files it names,
    each with the sha256 it names. Each file is read once and hashed. A
    table's rows are parsed from those same bytes, so the rows returned are
    the rows that were hashed.

    Every table must carry its run record inside it. The record names kind
    phase5_table, the table's own name, and its own label. It names the run
    MANIFEST.json names, and no input that MANIFEST.json lacks. Each JSON
    output carries a run record of its own kind, naming the same run. Every
    row's level is the record's.

    Tables are parsed with pyarrow, never pandas, and never through
    lot.evaluate.read_rows, which reads evaluation parquets. Every refusal
    raises ProvenanceError naming each problem's field.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .evaluate import RUN_METADATA_KEY

    directory = Path(tables_dir)
    context = f"the published tables at {directory} cannot be drawn from"
    if not directory.is_dir():
        raise ProvenanceError([("tables", f"no tables directory at {directory}")], context)
    try:
        raw = (directory / MANIFEST_FILE).read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as error:
        raise ProvenanceError(
            [("manifest", f"{MANIFEST_FILE} cannot be read: {error}")], context
        ) from None
    if not isinstance(manifest, dict):
        raise ProvenanceError([("manifest", f"{MANIFEST_FILE} is not a JSON object")], context)

    problems: list[tuple[str, str]] = []
    if manifest.get("kind") != TABLES_MANIFEST_KIND:
        problems.append(("manifest", (
            f"names kind {manifest.get('kind')!r}, not {TABLES_MANIFEST_KIND!r}")))
    if manifest.get("report_version") != REPORT_VERSION:
        problems.append(("manifest", (
            f"names report version {manifest.get('report_version')!r}, and these figures "
            f"read version {REPORT_VERSION}")))
    files = manifest.get("files")
    if not (isinstance(files, dict) and files
            and all(isinstance(name, str) and isinstance(sha, str) and _SHA256.fullmatch(sha)
                    for name, sha in files.items())):
        problems.append(("manifest", "names no files, or a file without a sha256"))
        files = {}
    record = manifest.get("run_record")
    if not (isinstance(record, dict) and record.get("kind") == TABLES_MANIFEST_KIND):
        problems.append(("manifest", "carries no run record of the tables mode"))
        record = {}
    if level is not None and record and record.get("level") != level:
        problems.append(("level", (
            f"the tables were built for level {record.get('level')!r}, not {level!r}")))
    if problems:
        raise ProvenanceError(problems, context)

    present = sorted(entry.name for entry in directory.iterdir())
    extra = [name for name in present if name != MANIFEST_FILE and name not in files]
    missing = sorted(name for name in files if name not in present)
    if extra:
        problems.append(("files", f"{extra} are in {directory} and not in {MANIFEST_FILE}"))
    if missing:
        problems.append(("files", f"{missing} are named in {MANIFEST_FILE} and absent"))
    parquets = sorted(name for name in files if name.endswith(".parquet"))
    listed = record.get("tables")
    if not (isinstance(listed, list) and sorted(listed) == parquets):
        problems.append(("manifest", (
            f"its run record lists the tables {_short(listed)}, and it names the "
            f"parquets {parquets}")))

    tables: dict[str, list[dict[str, Any]]] = {}
    documents: dict[str, Any] = {}
    for name in sorted(files):
        if name in missing:
            continue
        data = (directory / name).read_bytes()
        have = hashlib.sha256(data).hexdigest()
        if have != files[name]:
            problems.append(("sha256", (
                f"{name} has sha256 {have}, and {MANIFEST_FILE} names {files[name]}")))
            continue
        if name.endswith(".parquet"):
            try:
                table = pq.read_table(pa.BufferReader(data))
            except Exception as error:  # noqa: BLE001
                # Any file that cannot be parsed is refused with the others.
                problems.append(("files", f"{name} cannot be read as a table: {error}"))
                continue
            metadata = (table.schema.metadata or {}).get(RUN_METADATA_KEY)
            table_record = json.loads(metadata.decode("utf-8")) if metadata else None
            if not isinstance(table_record, dict):
                problems.append(("run_record", f"{name} carries no run record inside it"))
                continue
            for field, want in (("kind", TABLE_KIND), ("table", name),
                                ("report_version", REPORT_VERSION),
                                ("table_label", TABLE_LABELS.get(name))):
                if table_record.get(field) != want:
                    problems.append(("run_record", (
                        f"{name}: its run record names {field} "
                        f"{_short(table_record.get(field))}, not {_short(want)}")))
            problems += _identity_problems(table_record, record, name)
            rows = table.to_pylist()
            levels = sorted({str(row["level"]) for row in rows if "level" in row})
            if levels and levels != [str(record.get("level"))]:
                problems.append(("rows", (
                    f"{name} holds rows of level {levels}, and its run record names "
                    f"{record.get('level')!r}")))
            tables[name] = rows
        elif name.endswith(".json"):
            try:
                payload = json.loads(data.decode("utf-8"))
            except ValueError as error:
                problems.append(("files", f"{name} is not JSON: {error}"))
                continue
            document_record = payload.get("run_record") if isinstance(payload, dict) else None
            kind = DOCUMENT_KINDS.get(name)
            if not (isinstance(document_record, dict) and kind is not None
                    and document_record.get("kind") == kind):
                problems.append(("run_record", (
                    f"{name} carries no run record of the kind {_short(kind)}")))
                continue
            problems += _identity_problems(document_record, record, name)
            documents[name] = payload
        else:
            problems.append(("files", (
                f"{name} is neither a table nor a JSON output of the tables mode")))
    if problems:
        raise ProvenanceError(problems, context)
    return PublishedTables(
        directory=directory,
        level=str(record.get("level")),
        manifest=manifest,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        files=dict(files),
        run_record=record,
        tables=tables,
        documents=documents,
    )


def _run_values(identity: Any) -> dict[str, Any]:
    """The RUN_FIELDS of an evaluated run, as an output's run record names them."""
    return {
        "level": identity.level,
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
        "licence": dict(identity.licence),
    }


def bind_tables_to_run(published: PublishedTables, identity: Any) -> None:
    """The published tables are the evaluated run's own, or a refusal.

    identity is lot.phase5_provenance.RunIdentity. The tables' run record must
    name the run's level, commits, digests, seeds, folds, scenes, bootstrap
    settings, and licence. Every file the run is read from must be among the
    tables' inputs with the sha256 it has now. Each evaluation parquet must
    also keep its name. Another input is bound by its content alone, so a
    receipt that a later rerun moved aside still binds. A run whose code was
    checked is never drawn from tables built without that check. Raises
    ProvenanceError naming each field.
    """
    record = published.run_record
    problems = [
        (field, f"the tables name {_short(record.get(field))}, and the evaluated run "
                f"{_short(want)}")
        for field, want in _run_values(identity).items() if record.get(field) != want
    ]
    named = record.get("inputs") if isinstance(record.get("inputs"), dict) else {}
    held = set(named.values())
    run_dir = Path(identity.run_dir)
    evaluated = {_relative(Path(identity.eval_paths[scene]), run_dir): sha
                 for scene, sha in identity.eval_sha256.items()}
    differ = sorted({name for name, sha in identity.inputs.items() if sha not in held}
                    | {name for name, sha in evaluated.items() if named.get(name) != sha})
    if differ:
        problems.append(("inputs", (
            f"the tables were not built from the run's files as they are now: {differ[:5]}")))
    if identity.code_checked and record.get("code_checked") is not True:
        problems.append(("code_checked", (
            "the tables were built without checking the code that reported them, so they "
            "cannot be drawn for a run whose code was checked")))
    if problems:
        raise ProvenanceError(problems, "the published tables are not the evaluated run's")


# ---------------------------------------------------------------------------
# Which rows a panel draws
# ---------------------------------------------------------------------------

def curve_rows(rows: Sequence[Mapping[str, Any]], regime: str, metric: str
               ) -> tuple[Mapping[str, Any] | None, dict[str, Mapping[str, Any]]]:
    """One regime's curve under one metric: its regime row, and its rows by bin.

    PROTOCOL 3.3. Rotation pairs are the sole source of the rotation curve,
    and translation pairs of the parallax curve. Orbit has no curve and is
    drawn in joint cells only. A row of another regime on the curve's axis
    refuses, through lot.figures.assert_single_regime, and so does a row of
    the regime on any axis but its own and the regime row's. A repeated cell
    stops.
    """
    if regime not in CURVE_AXES:
        raise FiguresStop(
            f"{regime} has no primary curve. PROTOCOL 3.3 analyses orbit only in joint "
            "rotation by parallax cells, never on a marginal axis"
        )
    axis = CURVE_AXES[regime]
    chosen = [row for row in rows if row.get("metric") == metric]
    assert_single_regime(
        [{"regime": row.get("analysis")} for row in chosen if row.get("axis") == axis], regime
    )
    stray = [row for row in chosen
             if row.get("analysis") == regime and row.get("axis") not in (SCOPE_AXIS, axis)]
    if stray:
        raise FiguresStop(
            f"a {regime} row lies on the axis {stray[0].get('axis')!r}. A {regime} pair is "
            f"binned on {axis} only, and the joint axis holds orbit alone"
        )
    scope: Mapping[str, Any] | None = None
    bins: dict[str, Mapping[str, Any]] = {}
    for row in chosen:
        if row.get("analysis") != regime:
            continue
        if row.get("axis") == SCOPE_AXIS:
            if scope is not None:
                raise FiguresStop(f"the {regime} row under {metric} appears twice")
            scope = row
        else:
            label = row.get("bin")
            if label in bins:
                raise FiguresStop(f"the {regime} bin {label!r} under {metric} appears twice")
            bins[label] = row
    return scope, bins


def _curve_cells(scope: Mapping[str, Any] | None, bins: Mapping[str, Mapping[str, Any]],
                 order: Sequence[str], *, with_scope: bool) -> dict[int, Mapping[str, Any]]:
    """Rows at their positions: each bin at its frozen index, the regime row after."""
    cells: dict[int, Mapping[str, Any]] = {}
    for label, row in bins.items():
        if label not in order:
            raise FiguresStop(f"the bin {label!r} is not in the frozen order {list(order)}")
        cells[order.index(label)] = row
    if with_scope and scope is not None:
        cells[len(order)] = scope
    return cells


def joint_rows(rows: Sequence[Mapping[str, Any]], metric: str, analysis: AnalysisConfig
               ) -> dict[tuple[int, int], Mapping[str, Any]]:
    """Orbit's joint cells under one metric, by rotation and parallax bin index.

    The joint axis holds orbit alone. A row of another regime on it, a label
    outside the frozen orders, or a repeated cell stops.
    """
    rotation_order, parallax_order = rotation_bin_order(analysis), parallax_bin_order(analysis)
    out: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("metric") != metric or row.get("axis") != JOINT_AXIS:
            continue
        if row.get("analysis") != ORBIT:
            raise FiguresStop(
                f"a {row.get('analysis')} row lies on the joint axis, which holds orbit alone"
            )
        rotation, parallax = row.get("rotation_bin"), row.get("parallax_bin")
        if rotation not in rotation_order or parallax not in parallax_order:
            raise FiguresStop(f"the joint cell {row.get('bin')!r} is not in the frozen orders")
        key = (rotation_order.index(rotation), parallax_order.index(parallax))
        if key in out:
            raise FiguresStop(f"the joint cell {row.get('bin')!r} under {metric} appears twice")
        out[key] = row
    return out


def regime_rows(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[int, Mapping[str, Any]]:
    """Each regime's row under one metric, at its index in REGIME_SCOPES.

    The pooled row is a summary in the tables and is not drawn.
    """
    out: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if (row.get("metric") != metric or row.get("axis") != SCOPE_AXIS
                or row.get("analysis") not in REGIME_SCOPES):
            continue
        position = REGIME_SCOPES.index(row["analysis"])
        if position in out:
            raise FiguresStop(f"the {row['analysis']} row under {metric} appears twice")
        out[position] = row
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Point:
    """One cell of a series: its position, estimate, interval, and support."""

    position: int
    estimate: float
    lo: float
    hi: float
    supported: bool


def _point(row: Mapping[str, Any], quantity: str, position: int,
           support: str = "supported") -> _Point:
    return _Point(position, _float(row.get(quantity)), _float(row.get(f"{quantity}_ci_low")),
                  _float(row.get(f"{quantity}_ci_high")), row.get(support) is True)


def _yerr(points: Sequence[_Point]) -> list[list[float]]:
    """Error bar lengths from each interval. An undefined end draws no bar."""
    low = [max(0.0, p.estimate - p.lo) if math.isfinite(p.lo) else 0.0 for p in points]
    high = [max(0.0, p.hi - p.estimate) if math.isfinite(p.hi) else 0.0 for p in points]
    return [low, high]


def _spread(index: int, count: int, step: float = 0.14) -> float:
    """The offset of series index of count, so markers sit side by side."""
    return (index - (count - 1) / 2.0) * step


def _plot_series(panel: Any, points: Sequence[_Point], quantity: str, label: str,
                 colour: str, marker: str, *, offset: float = 0.0,
                 curve: Sequence[int] = ()) -> None:
    """One series: supported cells in colour, cells below support as hollow grey.

    Each marker carries its interval as an error bar. The supported markers
    are series:{quantity}, the hollow ones series:{quantity}:below-support.
    curve, when given, names the bin positions joined by a thin line, broken
    at a blank bin. The legend shows the series once, in its colour.
    """
    finite = [p for p in points if math.isfinite(p.estimate)]
    for supported in (True, False):
        chosen = [p for p in finite if p.supported is supported]
        if not chosen:
            continue
        container = panel.errorbar(
            [p.position + offset for p in chosen], [p.estimate for p in chosen],
            yerr=_yerr(chosen), fmt=marker, capsize=2, markersize=4, elinewidth=0.9,
            color=colour if supported else GREY,
            markerfacecolor=colour if supported else "none",
            label="_nolegend_",
        )
        container.lines[0].set_gid(
            f"series:{quantity}" if supported else f"series:{quantity}:{BELOW_SUPPORT}"
        )
    if curve:
        by_position = {p.position: p.estimate for p in finite}
        (line,) = panel.plot([x + offset for x in curve],
                             [by_position.get(x, math.nan) for x in curve],
                             "-", color=colour, linewidth=0.8, alpha=0.5, label="_nolegend_")
        line.set_gid(f"curve:{quantity}")
    panel.plot([], [], marker, color=colour, label=label)


def _shade(panel: Any, positions: Sequence[int]) -> None:
    """A grey band behind each position below support, the Phase 3 convention.

    lot.figures._shade_unsupported draws the bands, and each gets the gid
    unsupported:{position}.
    """
    positions = sorted({int(position) for position in positions})
    before = len(panel.patches)
    _shade_unsupported(panel, positions)
    added = list(panel.patches)[before:]
    if len(added) != len(positions):
        raise RuntimeError(f"{len(added)} bands were drawn for {len(positions)} positions")
    for patch, position in zip(added, positions):
        patch.set_gid(f"unsupported:{position}")


def _counts(panel: Any, labels: Mapping[int, str]) -> None:
    """Each position's count, PROTOCOL 3.4, in the foot _finish_panel reserves."""
    for position, text in sorted(labels.items()):
        artist = panel.annotate(
            text, (position, 0.01), xycoords=("data", "axes fraction"), fontsize=6,
            ha="center", va="bottom", color="0.3", linespacing=1.0,
        )
        artist.set_gid(f"count:{position}")


def _finish_panel(panel: Any, foot_lines: int = 1) -> None:
    """Close a panel: name its hollow markers, and keep its foot for the counts.

    The lower y limit moves down until no marker or interval reaches the foot
    band where the counts sit, which holds foot_lines lines of text. Called
    after everything else in the panel is drawn.
    """
    if any((line.get_gid() or "").endswith(f":{BELOW_SUPPORT}") for line in panel.get_lines()):
        panel.plot([], [], "o", color=GREY, markerfacecolor="none", linestyle="none",
                   label=BELOW_SUPPORT_MARKER)
    low, high = panel.get_ylim()
    share = min(0.4, 0.04 + 0.045 * foot_lines)
    panel.set_ylim(low - (high - low) * share / (1.0 - share), high)


def _category_axis(panel: Any, labels: Sequence[str], separator: int | None = None) -> None:
    """One position per label. separator draws a rule before that position."""
    panel.set_xticks(list(range(len(labels))))
    panel.set_xticklabels(list(labels), rotation=45, ha="right", fontsize=7)
    panel.set_xlim(-0.6, len(labels) - 0.4)
    if separator is not None:
        panel.axvline(separator - 0.5, color="0.3", linewidth=0.8, linestyle=":")


def _figure_legend(figure: Any, panels: Sequence[Any]) -> Any:
    """One legend below the panels, naming each series once. None without series.

    Three columns where they fit the figure's width, else two, else one.
    """
    handles: list[Any] = []
    labels: list[str] = []
    for panel in panels:
        for handle, label in zip(*panel.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if not handles:
        return None
    renderer = figure.canvas.get_renderer()
    for columns in (3, 2, 1):
        legend = figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
                               ncol=min(columns, len(handles)), fontsize=7, frameon=False)
        if columns == 1 or legend.get_window_extent(renderer).width <= 0.98 * figure.bbox.width:
            break
        legend.remove()
    legend.set_gid("legend")
    return legend


def _clear_title(figure: Any, title: Any, pad_points: float = 6.0) -> None:
    """Move the panels below the caption, where the layout left them under it.

    Some matplotlib versions lay out the suptitle and some do not, so the
    clearance is measured rather than assumed.
    """
    renderer = figure.canvas.get_renderer()
    bottom = title.get_window_extent(renderer).y0
    top = max(axes.get_tightbbox(renderer).y1 for axes in figure.axes)
    pad = pad_points * figure.dpi / 72.0
    if top > bottom - pad:
        figure.subplots_adjust(
            top=figure.subplotpars.top - (top - bottom + pad) / figure.bbox.height)


def _signed(panel: Any) -> None:
    panel.axhline(0.0, color="black", linewidth=1)


def _counts_of(cells: Mapping[int, Mapping[str, Any]]) -> dict[int, str]:
    return {position: f"n={row.get('n_camera_pairs')}" for position, row in cells.items()}


def _greyed(cells: Mapping[int, Mapping[str, Any]]) -> list[int]:
    return [position for position, row in cells.items() if row.get("supported") is not True]


def _extent(values: Sequence[float]) -> float:
    """The half range of a symmetric scale: the largest magnitude, or one."""
    top = max((abs(value) for value in values if math.isfinite(value)), default=0.0)
    return top if top > 0.0 else 1.0


def _heatmap(figure: Any, panel: Any, cells: Mapping[tuple[int, int], Mapping[str, Any]],
             quantity: str, analysis: AnalysisConfig, extent: float, title: str) -> None:
    """Joint cells on the full rotation by parallax grid, PROTOCOL 3.3.

    A symmetric diverging scale. A cell below support is hatched, and every
    populated cell prints its value and count. A cell with no row is blank.
    """
    from matplotlib.patches import Rectangle

    rotation_order, parallax_order = rotation_bin_order(analysis), parallax_bin_order(analysis)
    grid = np.full((len(rotation_order), len(parallax_order)), np.nan)
    for (i, j), row in cells.items():
        grid[i, j] = _float(row.get(quantity))
    image = panel.imshow(grid, cmap="RdBu_r", vmin=-extent, vmax=extent, aspect="auto")
    for (i, j), row in sorted(cells.items()):
        value = _float(row.get(quantity))
        # White on a dark cell and black on a light one, for the hatch and the
        # text alike, so neither vanishes into the cell's own colour.
        ink = "white" if math.isfinite(value) and abs(value) > 0.6 * extent else "black"
        if row.get("supported") is not True:
            patch = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                              edgecolor=ink, linewidth=0.0)
            panel.add_patch(patch)
            patch.set_gid(f"hatched:{i}:{j}")
        text = (f"{value:+.3f}\n" if math.isfinite(value) else "") + (
            f"n={row.get('n_camera_pairs')}")
        artist = panel.text(j, i, text, ha="center", va="center", fontsize=6, color=ink)
        artist.set_gid(f"cell:{i}:{j}")
    panel.set_xticks(list(range(len(parallax_order))))
    panel.set_xticklabels(parallax_order, rotation=45, ha="right", fontsize=7)
    panel.set_yticks(list(range(len(rotation_order))))
    panel.set_yticklabels(rotation_order, fontsize=7)
    panel.set_xlabel("parallax bin", fontsize=8)
    panel.set_ylabel("rotation bin, degrees", fontsize=8)
    panel.set_title(title, fontsize=8)
    figure.colorbar(image, ax=panel, fraction=0.046)


def _publish(figure: Any, path_out: Path, caption: str, size: tuple[float, float],
             legend_panels: Sequence[Any] = ()) -> Any:
    """Caption, legend, lay out, save at dpi 150, and close one figure. Returns it.

    The caption is wrapped at spaces only, so no name is broken at a hyphen.
    The series of legend_panels share one legend below the panels, and the
    layout keeps the panels clear of it and of the caption.
    """
    plt = _pyplot()
    width, _ = size
    wrapped = textwrap.fill(" ".join(caption.split()), width=max(70, int(width * 13)),
                            break_on_hyphens=False, break_long_words=False)
    title = figure.suptitle(wrapped, fontsize=9)
    title.set_gid("suptitle")
    bottom = 0.0
    legend = _figure_legend(figure, legend_panels)
    if legend is not None:
        extent = legend.get_window_extent(figure.canvas.get_renderer())
        bottom = extent.y1 / figure.bbox.height + 0.005
    figure.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
    _clear_title(figure, title)
    path_out = Path(path_out)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_out, dpi=150)
    plt.close(figure)
    return figure


def _confidence(analysis: AnalysisConfig) -> str:
    return f"{analysis.bootstrap_confidence:.0%}"


# ---------------------------------------------------------------------------
# Figures 1 and 2: the headline gap against parallax and rotation
# ---------------------------------------------------------------------------

def _level_note(primary_rows: Sequence[Mapping[str, Any]], role: str) -> str:
    """What a headline figure says about its level, reporting_rules.md decision 4.

    The primary level's Figures 1 and 2 are the headline. At any other level
    they draw the same comparison at that level, beside the primary result.
    """
    if role == PRIMARY_ROLE:
        return ""
    level = ", ".join(sorted({str(row.get("level")) for row in primary_rows}))
    return (f"The headline figure's comparison at the {role} level {level}. The headline "
            "itself is the primary level's.")


def _gap_curve_figure(primary_rows: Sequence[Mapping[str, Any]], path_out: Path,
                      analysis: AnalysisConfig, regime: str, role: str = PRIMARY_ROLE) -> Any:
    _require_columns(primary_rows, CURVE_COLUMNS, PRIMARY_TABLE)
    note = _level_note(primary_rows, role)
    deficit = any("read_deficit" in row for row in primary_rows)
    if deficit:
        _require_columns(primary_rows, DEFICIT_COLUMNS, PRIMARY_TABLE)
    order = _order(regime, analysis)
    # Every metric's rows are selected and guarded before anything is drawn.
    selected = {
        metric: _curve_cells(*curve_rows(primary_rows, regime, metric), order, with_scope=True)
        for metric in METRICS
    }
    plt = _pyplot()
    size = (13.0, 9.5)
    figure, axes = plt.subplots(2, len(METRICS), figsize=size, squeeze=False)
    labels = list(order) + ["all pairs"]
    bins = list(range(len(order)))
    axis_label = "parallax bin" if regime == "translation" else "rotation bin, degrees"
    for column, metric in enumerate(METRICS):
        cells = selected[metric]
        gap, absolute = axes[0][column], axes[1][column]
        gap.set_gid(f"gap:{metric}")
        absolute.set_gid(f"absolute:{metric}")
        for panel in (gap, absolute):
            _shade(panel, _greyed(cells))
            _counts(panel, _counts_of(cells))
            _category_axis(panel, labels, separator=len(order))
            panel.grid(alpha=0.3)
        _plot_series(gap, [_point(row, "delta_learn_pp", p) for p, row in cells.items()],
                     "delta_learn_pp", GAP_LABEL, COLOURS["delta_learn_pp"], "o",
                     offset=-0.12 if deficit else 0.0, curve=bins)
        if deficit:
            # Beside each cell, in its own marker, never combined with the gap.
            # A read deficit below its own support is greyed, with its count.
            deficits = [_point(row, "read_deficit", p, "read_deficit_supported")
                        for p, row in cells.items()]
            _plot_series(gap, deficits, "read_deficit", READ_DEFICIT_LABEL,
                         COLOURS["read_deficit"], "D", offset=0.18)
            for point in deficits:
                if point.supported or not math.isfinite(point.estimate):
                    continue
                artist = gap.annotate(
                    f"n={cells[point.position].get('read_deficit_n_camera_pairs')}",
                    (point.position + 0.18, point.estimate), xytext=(4, 0),
                    textcoords="offset points", fontsize=6, ha="left", va="center", color=GREY,
                )
                artist.set_gid(f"deficit-count:{point.position}")
        _signed(gap)
        standing = "the primary metric" if metric == PRIMARY_METRIC else "beside the primary"
        gap.set_title(f"delta_learn_pp, {metric} cosine, {standing}", fontsize=9)
        gap.set_ylabel(f"{CL_TRANSPORT}\nminus {PREDICT_WITH_DEPTH}", fontsize=7)

        reference = KNOWN_TRANSFORM_REFERENCE if regime == "rotation" else CL_TRANSPORT
        series = [("cl_transport", reference, "o"), ("predict_with_depth", PREDICT_WITH_DEPTH, "s"),
                  ("no_warp_copy", f"{NO_WARP_COPY} floor", "^")]
        if metric == "raw":
            series.append(("mean_feature", f"{MEAN_FEATURE} floor, raw cosine only", "v"))
        for index, (quantity, label, marker) in enumerate(series):
            _plot_series(absolute, [_point(row, quantity, p) for p, row in cells.items()],
                         quantity, label, COLOURS[quantity], marker,
                         offset=_spread(index, len(series)), curve=bins)
        floor = "" if metric == "raw" else f"; {MEAN_FEATURE} not applicable under centering"
        absolute.set_title(f"absolute scores with the {NO_WARP_COPY} floor, {metric} cosine{floor}",
                           fontsize=9)
        absolute.set_ylabel(f"{metric} cosine", fontsize=8)
        absolute.set_xlabel(axis_label, fontsize=8)
        for panel in (gap, absolute):
            _finish_panel(panel)

    if regime == "translation":
        head = (
            f"Figure 1: delta_learn_pp, {CL_TRANSPORT} minus {PREDICT_WITH_DEPTH}, against "
            "parallax, translation pairs only, on the primary per-point support V_P5_pp. "
            + (note or "The headline figure.")
        )
    else:
        head = (
            f"Figure 2: delta_learn_pp, {CL_TRANSPORT} minus {PREDICT_WITH_DEPTH}, against "
            "rotation angle, rotation pairs only, on the primary per-point support V_P5_pp. "
            "Under pure rotation the mapping is exact and does not depend on depth, so "
            f"{CL_TRANSPORT} is drawn as the known-transform reference."
            + (f" {note}" if note else "")
        )
    caption = (
        f"{head} Positive: explicit computation wins. Negative: the learned transformation "
        "wins. Centered cosine, the primary metric, on the left, and raw cosine beside it. "
        f"Top row: the gap with its {_confidence(analysis)} paired scene-bootstrap interval "
        "at each bin, and the regime's own row after the bins. "
    )
    if deficit:
        caption += (
            "The read deficit is drawn beside each cell, and is never subtracted from the "
            f"gap. Depth note: {READ_DEPTH_NOTE}. "
        )
    else:
        caption += READ_DEFICIT_PRIMARY_ONLY + " "
    caption += (
        f"Bottom row: the absolute scores with the {NO_WARP_COPY} floor, and {MEAN_FEATURE} "
        f"under raw cosine only. {SUPPORT_NOTE}"
    )
    return _publish(figure, path_out, caption, size, legend_panels=list(axes.flat))


def figure1_gap_vs_parallax(primary_rows: Sequence[Mapping[str, Any]], path_out: Path,
                            analysis: AnalysisConfig, *, role: str = PRIMARY_ROLE) -> Figure:
    """Figure 1, specification step 39: delta_learn_pp against parallax.

    From the headline table's translation rows alone. Each parallax bin of the
    frozen order has a place, with the translation row after the bins. The
    read deficit sits beside each cell where the table carries it, at the
    primary level, labelled with its direction. Below, the absolute scores:
    Context-Lift Transport-Only, Predict-with-Depth, the No-Warp-Copy floor,
    and Mean-Feature under raw cosine only. Refuses an orbit row on the
    parallax axis, PROTOCOL 3.3. role is the level's, lot.phase5_report.
    level_role's. At the primary level the caption calls the figure the
    headline. At any other level it names the level and its role, and says
    the headline is the primary level's.
    """
    return _gap_curve_figure(primary_rows, path_out, analysis, "translation", role)


def figure2_gap_vs_rotation(primary_rows: Sequence[Mapping[str, Any]], path_out: Path,
                            analysis: AnalysisConfig, *, role: str = PRIMARY_ROLE) -> Figure:
    """Figure 2, specification step 40: delta_learn_pp against rotation angle.

    From the headline table's rotation rows alone, laid out as Figure 1.
    Context-Lift Transport-Only is labelled the known-transform reference:
    under pure rotation its mapping is exact and depth independent. Refuses an
    orbit row on the rotation axis, PROTOCOL 3.3. At a level other than the
    primary one, the caption names the level and its role, as Figure 1's does.
    """
    return _gap_curve_figure(primary_rows, path_out, analysis, "rotation", role)


# ---------------------------------------------------------------------------
# Figure 3: the formulation gap, a diagnostic
# ---------------------------------------------------------------------------

def _gap_panel(panel: Any, cells: Mapping[int, Mapping[str, Any]], labels: Sequence[str],
               quantity: str, label: str, *, curve: bool) -> None:
    """One gap series on a category axis, with its bands and counts."""
    _shade(panel, _greyed(cells))
    _counts(panel, _counts_of(cells))
    _category_axis(panel, labels)
    _plot_series(panel, [_point(row, quantity, p) for p, row in cells.items()], quantity,
                 label, COLOURS[quantity], "o", curve=range(len(labels)) if curve else ())
    _signed(panel)
    panel.grid(alpha=0.3)
    _finish_panel(panel)


def figure3_formulation_gap(formulation_rows: Sequence[Mapping[str, Any]], path_out: Path,
                            analysis: AnalysisConfig) -> Figure:
    """Figure 3, specification step 41: delta_formulation by regime and difficulty.

    From the formulation table. Each regime's own row, then the rotation bins
    of rotation pairs, the parallax bins of translation pairs, and orbit in
    joint cells. The figure is labelled an information/formulation
    diagnostic, not the learned-versus-explicit estimand, and not a
    learned-model effect.
    """
    _require_columns(formulation_rows, FORMULATION_COLUMNS, FORMULATION_TABLE)
    selected = {
        metric: {
            "regimes": regime_rows(formulation_rows, metric),
            "rotation": _curve_cells(*curve_rows(formulation_rows, "rotation", metric),
                                     _order("rotation", analysis), with_scope=False),
            "translation": _curve_cells(*curve_rows(formulation_rows, "translation", metric),
                                        _order("translation", analysis), with_scope=False),
            ORBIT: joint_rows(formulation_rows, metric, analysis),
        }
        for metric in METRICS
    }
    plt = _pyplot()
    size = (22.0, 10.0)
    figure, axes = plt.subplots(len(METRICS), 4, figsize=size, squeeze=False)
    for row_index, metric in enumerate(METRICS):
        cells = selected[metric]
        regimes, rotation, translation, orbit = axes[row_index]
        for panel, kind, labels, curve in (
            (regimes, "regimes", list(REGIME_SCOPES), False),
            (rotation, "rotation", _order("rotation", analysis), True),
            (translation, "translation", _order("translation", analysis), True),
        ):
            panel.set_gid(f"{kind}:{metric}")
            _gap_panel(panel, cells[kind], labels, "delta_formulation", FORMULATION_GAP_LABEL,
                       curve=curve)
            panel.set_ylabel(f"delta_formulation, {metric} cosine", fontsize=8)
        regimes.set_title(f"each regime's own row, {metric} cosine", fontsize=9)
        rotation.set_title(f"rotation pairs by rotation bin, {metric} cosine", fontsize=9)
        rotation.set_xlabel("rotation bin, degrees", fontsize=8)
        translation.set_title(f"translation pairs by parallax bin, {metric} cosine", fontsize=9)
        translation.set_xlabel("parallax bin", fontsize=8)
        orbit.set_gid(f"{ORBIT}:{metric}")
        extent = _extent([_float(row.get("delta_formulation")) for row in cells[ORBIT].values()])
        _heatmap(figure, orbit, cells[ORBIT], "delta_formulation", analysis, extent,
                 f"orbit, joint rotation by parallax cells, {metric} cosine")
    caption = (
        f"Figure 3: delta_formulation, {TL_REFERENCE} minus {CL_TRANSPORT}, on the "
        "formulation support V_form, by regime and difficulty, centered cosine above and raw "
        f"cosine below. It is an {TABLE_LABELS[FORMULATION_TABLE]}, and not a learned-model "
        "effect. It sizes how far the inherited Phase 4 target-lift formulation is from the "
        "information-symmetric Phase 5 comparator. Each regime's own row, then the rotation "
        "bins of rotation pairs, the parallax bins of translation pairs, and orbit in joint "
        f"cells only. {_confidence(analysis)} paired scene-bootstrap intervals. "
        f"{SUPPORT_NOTE} {JOINT_NOTE}"
    )
    return _publish(figure, path_out, caption, size, legend_panels=list(axes.flat))


# ---------------------------------------------------------------------------
# Figure 4: the primary and the operational gap
# ---------------------------------------------------------------------------

def figure4_primary_vs_operational(
    primary_rows: Sequence[Mapping[str, Any]],
    splat_rows: Sequence[Mapping[str, Any]],
    cross_rows: Sequence[Mapping[str, Any]],
    path_out: Path,
    analysis: AnalysisConfig,
) -> Figure:
    """Figure 4, specification step 42: delta_learn_pp and delta_learn_sp by regime and bin.

    delta_learn_pp comes from the headline table, on V_P5_pp, and
    delta_learn_sp from the splat-pool table, on V_sp. Each is drawn on its
    own population and greyed on its own support. path_difference_learn, from
    the cross-path table, is their difference on the cells both paths share,
    with its paired interval. A grey band follows the headline cell. Orbit is
    drawn as joint cells only. The figure makes no claim that the two gaps'
    difference is an implementation error.
    """
    sources = (
        ("delta_learn_pp", primary_rows, PRIMARY_TABLE, GAP_LABEL, "o"),
        ("delta_learn_sp", splat_rows, SPLAT_TABLE, SPLAT_GAP_LABEL, "s"),
        ("path_difference_learn", cross_rows, CROSS_PATH_TABLE, PATH_DIFFERENCE_LABEL, "^"),
    )
    for quantity, rows, table, _, _ in sources:
        _require_columns(rows, GAP_COLUMNS[quantity], table)
    selected = {
        metric: {
            quantity: {
                "regimes": regime_rows(rows, metric),
                "rotation": _curve_cells(*curve_rows(rows, "rotation", metric),
                                         _order("rotation", analysis), with_scope=False),
                "translation": _curve_cells(*curve_rows(rows, "translation", metric),
                                            _order("translation", analysis), with_scope=False),
                ORBIT: joint_rows(rows, metric, analysis),
            }
            for quantity, rows, _, _, _ in sources
        }
        for metric in METRICS
    }
    plt = _pyplot()
    size = (18.0, 19.0)
    figure, axes = plt.subplots(2 * len(METRICS), 3, figsize=size, squeeze=False)
    for metric_index, metric in enumerate(METRICS):
        cells = selected[metric]
        lines, heats = axes[2 * metric_index], axes[2 * metric_index + 1]
        for panel, kind, labels in (
            (lines[0], "regimes", list(REGIME_SCOPES)),
            (lines[1], "rotation", _order("rotation", analysis)),
            (lines[2], "translation", _order("translation", analysis)),
        ):
            panel.set_gid(f"{kind}:{metric}")
            headline = cells["delta_learn_pp"][kind]
            _shade(panel, _greyed(headline))
            positions = sorted(set().union(*(cells[q][kind] for q, *_ in sources)))

            def count(position: int, kind: str = kind) -> str:
                # One line per population, in the order of sources, so four-digit
                # counts never run into the next position's.
                return "\n".join(
                    str(cells[q][kind][position].get("n_camera_pairs"))
                    if position in cells[q][kind] else "-" for q, *_ in sources
                )

            _counts(panel, {position: f"n={count(position)}" for position in positions})
            _category_axis(panel, labels)
            for index, (quantity, _, _, label, marker) in enumerate(sources):
                _plot_series(panel, [_point(row, quantity, p)
                                     for p, row in cells[quantity][kind].items()],
                             quantity, label, COLOURS[quantity], marker,
                             offset=_spread(index, len(sources), 0.18),
                             curve=() if kind == "regimes" else range(len(labels)))
            _signed(panel)
            panel.grid(alpha=0.3)
            panel.set_ylabel(f"gap, {metric} cosine", fontsize=8)
            _finish_panel(panel, foot_lines=len(sources))
        lines[0].set_title(f"each regime's own row, {metric} cosine", fontsize=9)
        lines[1].set_title(f"rotation pairs by rotation bin, {metric} cosine", fontsize=9)
        lines[1].set_xlabel("rotation bin, degrees", fontsize=8)
        lines[2].set_title(f"translation pairs by parallax bin, {metric} cosine", fontsize=9)
        lines[2].set_xlabel("parallax bin", fontsize=8)
        extent = _extent([_float(row.get(quantity)) for quantity, *_ in sources
                          for row in cells[quantity][ORBIT].values()])
        populations = {"delta_learn_pp": "on V_P5_pp", "delta_learn_sp": "on V_sp",
                       "path_difference_learn": "on the common cells"}
        for panel, (quantity, *_) in zip(heats, sources):
            panel.set_gid(f"{ORBIT}:{quantity}:{metric}")
            _heatmap(figure, panel, cells[quantity][ORBIT], quantity, analysis, extent,
                     f"orbit joint cells: {quantity} {populations[quantity]}, {metric} cosine")
    caption = (
        "Figure 4: the primary per-point gap and the secondary operational gap, by regime "
        "and bin, centered cosine in the upper two rows and raw cosine in the lower two. "
        f"delta_learn_pp, {CL_TRANSPORT} minus {PREDICT_WITH_DEPTH}, is read on V_P5_pp. "
        f"delta_learn_sp, {SPLAT_TRANSPORT} minus {PREDICT_WITH_DEPTH} on the splat-pool "
        "path, is read on V_sp. path_difference_learn is the per-point gap minus the "
        "splat-pool gap on the cells both paths share, with its paired interval. "
        f"{NOT_AN_IMPLEMENTATION_ERROR} The operational comparison is secondary and does not "
        "replace the primary per-point result. Orbit in joint cells only. "
        f"{_confidence(analysis)} paired scene-bootstrap intervals. Grey band: the "
        "delta_learn_pp cell is below support. Hollow grey marker: that series' own cell is "
        "below support. n: camera pairs on V_P5_pp, V_sp, and the common cells, one line "
        f"each, in that order. {JOINT_NOTE}"
    )
    return _publish(figure, path_out, caption, size, legend_panels=list(axes.flat))


# ---------------------------------------------------------------------------
# Figure 5: the error regions and their paired contrasts
# ---------------------------------------------------------------------------

def figure5_error_regions(region_rows: Sequence[Mapping[str, Any]], path_out: Path,
                          analysis: AnalysisConfig) -> Figure:
    """Figure 5, specification step 43: the error regions on the per-point path.

    From the region table, per regime. For each split, delta_learn_pp on its
    two regions, then the paired contrast over the camera pairs present in
    both regions: the gap's contrast and each method's own, whose difference
    is the gap's. A grey band marks a regime where a cell of the panel is
    below support. No near-zero wording applies to a contrast.
    """
    _require_columns(region_rows, REGION_COLUMNS, REGION_TABLE)
    index: dict[tuple, Mapping[str, Any]] = {}
    for row in region_rows:
        if row.get("axis") != SCOPE_AXIS or row.get("path") != PER_POINT:
            continue
        key = (row.get("metric"), row.get("analysis"), row.get("region"))
        if key in index:
            raise FiguresStop(f"{REGION_TABLE} holds the cell {key} twice")
        index[key] = row
    plt = _pyplot()
    size = (21.0, 9.5)
    figure, axes = plt.subplots(len(METRICS), 2 * len(REGION_SPLITS), figsize=size,
                                squeeze=False)
    positions = range(len(REGIME_SCOPES))
    for row_index, metric in enumerate(METRICS):
        for split_index, (kind, left, right, contrast, described) in enumerate(REGION_SPLITS):
            regions = axes[row_index][2 * split_index]
            contrasts = axes[row_index][2 * split_index + 1]
            regions.set_gid(f"{kind}:{metric}")
            contrasts.set_gid(f"{kind}_contrast:{metric}")

            sides = {region: {p: index[(metric, scope, region)]
                              for p, scope in enumerate(REGIME_SCOPES)
                              if (metric, scope, region) in index}
                     for region in (left, right)}
            greyed = [p for p in positions
                      if any(p in sides[region] and sides[region][p].get("supported") is not True
                             for region in (left, right))
                      or (p in sides[left]) != (p in sides[right])]
            _shade(regions, greyed)
            _counts(regions, {
                p: "n=" + "/".join(str(sides[region][p].get("n_camera_pairs"))
                                   if p in sides[region] else "-" for region in (left, right))
                for p in positions if p in sides[left] or p in sides[right]
            })
            _category_axis(regions, list(REGIME_SCOPES))
            for side, (region, marker) in enumerate(((left, "o"), (right, "s"))):
                _plot_series(regions, [_point(row, "delta_learn_pp", p)
                                       for p, row in sides[region].items()],
                             f"delta_learn_pp@{region}",
                             f"delta_learn_pp, {REGION_NAMES[region]}",
                             REGION_COLOURS[side], marker, offset=_spread(side, 2, 0.24))
            _signed(regions)
            regions.grid(alpha=0.3)
            regions.set_title(f"{described}, {metric} cosine", fontsize=9)
            regions.set_ylabel(f"delta_learn_pp, {metric} cosine", fontsize=8)
            _finish_panel(regions)

            paired = {p: index[(metric, scope, contrast)] for p, scope in enumerate(REGIME_SCOPES)
                      if (metric, scope, contrast) in index}
            _shade(contrasts, _greyed(paired))
            _counts(contrasts, _counts_of(paired))
            _category_axis(contrasts, list(REGIME_SCOPES))
            for series_index, (quantity, label, marker) in enumerate(CONTRAST_SERIES):
                _plot_series(contrasts, [_point(row, quantity, p) for p, row in paired.items()],
                             quantity, label, COLOURS[quantity], marker,
                             offset=_spread(series_index, len(CONTRAST_SERIES), 0.18))
            _signed(contrasts)
            contrasts.grid(alpha=0.3)
            contrasts.set_title(
                f"{REGION_NAMES[left]} minus {REGION_NAMES[right]}, paired, {metric} cosine",
                fontsize=9)
            contrasts.set_ylabel(f"contrast, {metric} cosine", fontsize=8)
            _finish_panel(contrasts)
    caption = (
        "Figure 5: error regions on the per-point path, centered cosine above and raw cosine "
        f"below. delta_learn_pp, {CL_TRANSPORT} minus {PREDICT_WITH_DEPTH}, on the "
        "ground-truth depth boundary and interior, and on low and high texture, per regime. "
        "Beside each split, its paired contrast, boundary minus interior or low minus high "
        "texture, over the camera pairs present in both regions: the gap's contrast and each "
        "method's own, whose difference is the gap's. No near-zero wording applies to a "
        f"contrast. {_confidence(analysis)} paired scene-bootstrap intervals. Grey band: a "
        "cell of that regime in the panel is below support. Hollow grey marker: that cell is "
        "below support. n: camera pairs in each region, or present in both for a contrast."
    )
    return _publish(figure, path_out, caption, size, legend_panels=list(axes.flat))


# ---------------------------------------------------------------------------
# Supplementary figures
# ---------------------------------------------------------------------------

def figureS1_validation_curves(history_rows: Sequence[Mapping[str, Any]],
                               adequacy_rows: Sequence[Mapping[str, Any]],
                               path_out: Path) -> Figure:
    """Supplementary figure S1: every training run's validation curve.

    From the validation history and adequacy tables. One line per fold and
    seed, with a star on the selected checkpoint, each line labelled with
    decision 5's stability. The history must mark exactly one best
    validation per run, at the adequacy table's best step and score, and
    both tables must describe the same runs, or the figure stops.
    """
    _require_columns(history_rows, HISTORY_COLUMNS, HISTORY_TABLE)
    _require_columns(adequacy_rows, ADEQUACY_COLUMNS, ADEQUACY_TABLE)
    runs: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in adequacy_rows:
        key = (int(row["fold"]), int(row["seed"]))
        if key in runs:
            raise FiguresStop(f"{ADEQUACY_TABLE} holds fold {key[0]}, seed {key[1]} twice")
        runs[key] = row
    histories: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in history_rows:
        histories.setdefault((int(row["fold"]), int(row["seed"])), []).append(row)
    if set(histories) != set(runs):
        raise FiguresStop(
            f"the validation history describes the runs {sorted(histories)}, and the "
            f"adequacy table {sorted(runs)}"
        )
    best: dict[tuple[int, int], Mapping[str, Any]] = {}
    for key, rows in histories.items():
        rows.sort(key=lambda row: row["validation_index"])
        marked = [row for row in rows if row.get("is_best") is True]
        if len(marked) != 1:
            raise FiguresStop(f"fold {key[0]}, seed {key[1]}: the history marks "
                              f"{len(marked)} best validations, not one")
        adequacy = runs[key]
        if (marked[0]["step"] != adequacy.get("best_step")
                or marked[0]["validation_centered_cosine"]
                != adequacy.get("best_validation_centered_cosine")):
            raise FiguresStop(
                f"fold {key[0]}, seed {key[1]}: the history's best validation, step "
                f"{marked[0]['step']} at {marked[0]['validation_centered_cosine']!r}, is not "
                f"the adequacy table's best step {adequacy.get('best_step')} at "
                f"{adequacy.get('best_validation_centered_cosine')!r}"
            )
        best[key] = marked[0]
    plt = _pyplot()
    folds = sorted({fold for fold, _ in runs})
    size = (5.8 * len(folds), 5.0)
    figure, axes = plt.subplots(1, len(folds), figsize=size, squeeze=False)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for column, fold in enumerate(folds):
        panel = axes[0][column]
        panel.set_gid(f"fold:{fold}")
        for index, seed in enumerate(sorted(seed for f, seed in runs if f == fold)):
            rows, adequacy = histories[(fold, seed)], runs[(fold, seed)]
            colour = colours[index % len(colours)]
            stable = "stable" if adequacy.get("stable_validation_curve") is True else "not stable"
            stopped = ("stopped early" if adequacy.get("stopped_early") is True
                       else "ran to its step limit")
            (curve,) = panel.plot([row["step"] for row in rows],
                                  [row["validation_centered_cosine"] for row in rows],
                                  "-", marker=".", markersize=4, color=colour,
                                  label=f"seed {seed}: {stable}, {stopped}")
            curve.set_gid(f"history:{fold}:{seed}")
            chosen = best[(fold, seed)]
            (star,) = panel.plot([chosen["step"]], [chosen["validation_centered_cosine"]], "*",
                                 markersize=12, color=colour, markeredgecolor="black",
                                 label="_nolegend_")
            star.set_gid(f"best:{fold}:{seed}")
        panel.set_title(f"fold {fold}", fontsize=9)
        panel.set_xlabel("training step", fontsize=8)
        panel.set_ylabel("validation centered cosine", fontsize=8)
        panel.grid(alpha=0.3)
        # Each fold's runs are labelled with their own stability, so each
        # panel keeps its own legend.
        panel.legend(fontsize=6, loc="best")
    caption = (
        "Supplementary figure S1: the validation centered cosine of every training run, one "
        "line per fold and seed. The star marks the selected checkpoint, chosen on validation "
        "scenes only. The validation score is a model-selection statistic, not an estimand. "
        "Stable is decision 5 of reporting_rules.md: every score finite, at least two "
        "validations, and early stopping fired or the last score within the frozen tolerance "
        "of the best."
    )
    return _publish(figure, path_out, caption, size)


def _offset_tick(row: Mapping[str, Any] | None, label: str) -> str:
    if label == OFFSET_WHOLE:
        return "all offsets"
    lo, hi = _float((row or {}).get("offset_lo")), _float((row or {}).get("offset_hi"))
    return f"{lo:g}-{hi:.3g}" if math.isfinite(lo) and math.isfinite(hi) else label


def _deficit_title(regime: str, metric: str, row: Mapping[str, Any] | None) -> str:
    title = f"{regime}, {metric} cosine"
    if row is None:
        return title
    estimate = _float(row.get("read_deficit"))
    lo, hi = _float(row.get("read_deficit_ci_low")), _float(row.get("read_deficit_ci_high"))
    title += (f": read deficit {estimate:+.4f} [{lo:+.4f}, {hi:+.4f}], "
              f"n={row.get('n_camera_pairs')}")
    if row.get("supported") is not True:
        title += ", below support"
    return title


def figureS2_landing_offset(offset_rows: Sequence[Mapping[str, Any]], path_out: Path,
                            analysis: AnalysisConfig) -> Figure:
    """Supplementary figure S2: the landing-offset curves, per regime.

    From the landing-offset table's regime rows, at the primary level, where
    alone it exists. Context-Lift Oracle-Transport and the No-Warp-Copy floor
    against the landing offset bins, the whole support after them, and
    Mean-Feature under raw cosine only. Each panel's title carries the read
    deficit. The caption carries the frozen reading note and depth note: the
    curve is read for its shape only, and its bins are not randomized.
    """
    _require_columns(offset_rows, OFFSET_COLUMNS, LANDING_OFFSET_TABLE)
    index: dict[tuple, Mapping[str, Any]] = {}
    for row in offset_rows:
        if row.get("axis") != SCOPE_AXIS or row.get("analysis") not in REGIME_SCOPES:
            continue
        key = (row.get("metric"), row.get("analysis"), row.get("offset_label"))
        if key in index:
            raise FiguresStop(f"{LANDING_OFFSET_TABLE} holds the cell {key} twice")
        index[key] = row
    labels = OFFSET_BINS + (OFFSET_WHOLE,)
    bins = range(len(OFFSET_BINS))
    plt = _pyplot()
    size = (17.0, 9.5)
    figure, axes = plt.subplots(len(METRICS), len(REGIME_SCOPES), figsize=size, squeeze=False)
    for row_index, metric in enumerate(METRICS):
        for column, regime in enumerate(REGIME_SCOPES):
            panel = axes[row_index][column]
            panel.set_gid(f"{regime}:{metric}")
            cells = {k: index[(metric, regime, label)] for k, label in enumerate(labels)
                     if (metric, regime, label) in index}
            _shade(panel, _greyed(cells))
            _counts(panel, _counts_of(cells))
            _category_axis(panel, [_offset_tick(cells.get(k), label)
                                   for k, label in enumerate(labels)],
                           separator=len(OFFSET_BINS))
            series = [("cl_oracle_offset", CL_ORACLE, "o"),
                      ("nowarp_offset", f"{NO_WARP_COPY} floor", "^")]
            if metric == "raw":
                series.append(("meanfeat_offset", f"{MEAN_FEATURE} floor, raw cosine only", "v"))
            for series_index, (quantity, label, marker) in enumerate(series):
                _plot_series(panel, [_point(row, quantity, p) for p, row in cells.items()],
                             quantity, label, COLOURS[quantity], marker,
                             offset=_spread(series_index, len(series)), curve=bins)
            panel.set_title(_deficit_title(regime, metric, index.get((metric, regime, "deficit"))),
                            fontsize=8)
            panel.set_xlabel("landing offset from the patch grid, patch units", fontsize=8)
            panel.set_ylabel(f"{metric} cosine", fontsize=8)
            panel.grid(alpha=0.3)
            _finish_panel(panel)
    caption = (
        f"Supplementary figure S2: {CL_ORACLE} and the {NO_WARP_COPY} floor against the "
        "landing offset from the patch grid, per regime, at the primary level, centered "
        f"cosine above and raw cosine below, with {MEAN_FEATURE} under raw cosine only. "
        f"{TABLE_LABELS[LANDING_OFFSET_TABLE]}, never an estimand of the phase. Each "
        "panel's title carries the read deficit, near-grid landings minus the whole support. "
        f"Reading note: {OFFSET_READING_NOTE}. Depth note: {READ_DEPTH_NOTE}. "
        f"{_confidence(analysis)} paired scene-bootstrap intervals. Grey band: the bin is "
        "below support. n: camera pairs, on every bin."
    )
    return _publish(figure, path_out, caption, size, legend_panels=list(axes.flat))


# ---------------------------------------------------------------------------
# The figures mode
# ---------------------------------------------------------------------------

def figures_for_level(level: str, primary_level: str) -> tuple[list[str], dict[str, str]]:
    """The figures drawn at a level, and those that are not, each with its reason."""
    if level == primary_level:
        return list(FIGURE_FILES), {}
    drawn = [name for name in FIGURE_FILES if name not in PRIMARY_LEVEL_ONLY]
    return drawn, dict(PRIMARY_LEVEL_ONLY)


def draw_figures(published: PublishedTables, directory: Path, analysis: AnalysisConfig,
                 drawn: Sequence[str]) -> None:
    """Draw each named figure into directory, from the published tables alone.

    The level's role is read from the tables' own run record, so a figure is
    regenerable from the tables alone.
    """
    directory = Path(directory)
    role = published.run_record.get("level_role", PRIMARY_ROLE)
    for name in drawn:
        path = directory / name
        if name == FIGURE1:
            figure1_gap_vs_parallax(published.rows(PRIMARY_TABLE), path, analysis, role=role)
        elif name == FIGURE2:
            figure2_gap_vs_rotation(published.rows(PRIMARY_TABLE), path, analysis, role=role)
        elif name == FIGURE3:
            figure3_formulation_gap(published.rows(FORMULATION_TABLE), path, analysis)
        elif name == FIGURE4:
            figure4_primary_vs_operational(published.rows(PRIMARY_TABLE),
                                           published.rows(SPLAT_TABLE),
                                           published.rows(CROSS_PATH_TABLE), path, analysis)
        elif name == FIGURE5:
            figure5_error_regions(published.rows(REGION_TABLE), path, analysis)
        elif name == FIGURE_S1:
            figureS1_validation_curves(published.rows(HISTORY_TABLE),
                                       published.rows(ADEQUACY_TABLE), path)
        elif name == FIGURE_S2:
            figureS2_landing_offset(published.rows(LANDING_OFFSET_TABLE), path, analysis)
        else:
            raise ValueError(f"no figure is named {name!r}")


def run_figures(
    cfg: Any,
    analysis: AnalysisConfig,
    config_path: Path,
    level: str,
    supersede: bool = False,
    *,
    expected_scenes: Sequence[str] | None = None,
    folds: Sequence[Any] | None = None,
    repo_root: Path | None = None,
    check_code: bool = True,
    tables_root: Path | None = None,
    figures_root: Path | None = None,
) -> dict[str, Any]:
    """The figures mode: one evaluated level's figures, from its tables, written once.

    An existing output is refused before any work, unless supersede is given.
    A non-primary level requires the primary level's tables, verified through
    lot.phase5_report.require_primary_tables. The run is licensed by
    require_evaluated_run, and a non-primary level's run must be the primary
    run's in all but its level, require_primary_chain, reporting_rules.md
    decision 4. The tables at tables/{level} are read through
    read_published_tables and bound to the run through bind_tables_to_run.
    They must name the level's role the configuration gives it. Every table
    the level's figures are drawn from must be there, and at the primary
    level the headline table must carry the read deficit. Each figure is then
    drawn into figures/{level}.partial.<id>/, and MANIFEST.json is written
    last. It names the level's role, each figure's sha256, and the sha256 of
    each table it was drawn from, beside the run record. The directory is
    published by one rename. An earlier output is moved aside, and nothing is
    deleted.

    expected_scenes, folds, repo_root, and check_code pass to
    require_evaluated_run, and tables_root and figures_root move the two
    directories, for tests. A reporting mode passes none of them. Returns
    where the figures were published, where an earlier output went, the files
    written in order, and the figures not drawn at this level.
    """
    from .phase5_folds import frozen_folds

    run_dir = Path(cfg.run_dir)
    tables_root = Path(tables_root) if tables_root is not None else run_dir / TABLES_DIR
    figures_root = Path(figures_root) if figures_root is not None else run_dir / FIGURES_DIR
    final = figures_root / level
    if final.exists() and not supersede:
        raise FileExistsError(
            f"{final} exists; outputs are written once. Supersede it to rebuild, which "
            "moves it aside and deletes nothing"
        )
    primary = cfg.primary_alignment_level
    role = level_role(cfg, level)
    primary_tables = (require_primary_tables(tables_root / primary, primary_level=primary)
                      if level != primary else None)
    folds = list(folds) if folds is not None else frozen_folds()
    identity = require_evaluated_run(cfg, analysis, config_path, level,
                                     expected_scenes=expected_scenes, folds=folds,
                                     repo_root=repo_root, check_code=check_code)
    if primary_tables is not None:
        require_primary_chain(primary_tables.run_record, identity)
    published = read_published_tables(tables_root / level, level=level)
    bind_tables_to_run(published, identity)
    if published.run_record.get("level_role") != role:
        raise FiguresStop(
            f"the tables at {published.directory} name level role "
            f"{published.run_record.get('level_role')!r}, and the configuration makes level "
            f"{level!r} the {role} level"
        )
    drawn, not_drawn = figures_for_level(level, primary)
    needed = sorted({table for name in drawn for table in FIGURE_TABLES[name]})
    absent = [table for table in needed if table not in published.tables]
    if absent:
        raise FiguresStop(
            f"the tables at {published.directory} lack {absent}, which the figures of level "
            f"{level!r} are drawn from"
        )
    if level == primary and not any("read_deficit" in row
                                    for row in published.rows(PRIMARY_TABLE)):
        raise FiguresStop(
            f"the headline table of the primary level {level!r} carries no read_deficit. "
            "landing_offset_diagnostic.md places the read deficit beside the headline there"
        )
    inputs = {_relative(published.directory / table, run_dir): published.files[table]
              for table in needed}
    inputs[_relative(published.directory / MANIFEST_FILE, run_dir)] = published.manifest_sha256

    with staged_output(final, supersede=supersede) as staged:
        draw_figures(published, staged.path, analysis, drawn)
        files: dict[str, str] = {}
        for name in drawn:
            path = staged.path / name
            if not path.is_file():
                raise FiguresStop(f"{name} was not drawn")
            files[name] = sha256_file(path)
        _write_json(staged.path / MANIFEST_FILE, {
            "kind": FIGURES_MANIFEST_KIND,
            "figures_version": FIGURES_VERSION,
            "report_version": REPORT_VERSION,
            "level": level,
            # The headline markers below are read at this level. Only the
            # primary level's headline figures are the headline result.
            "level_role": role,
            "files": dict(sorted(files.items())),
            "figures": {
                name: {
                    "title": FIGURE_TITLES[name],
                    "headline": name in HEADLINE_FIGURES,
                    "tables": {table: published.files[table] for table in FIGURE_TABLES[name]},
                }
                for name in drawn
            },
            "not_drawn": dict(not_drawn),
            "tables_manifest": {
                "path": _relative(published.directory / MANIFEST_FILE, run_dir),
                "sha256": published.manifest_sha256,
            },
            "run_record": output_run_record(
                identity, FIGURES_MANIFEST_KIND, inputs=inputs,
                extra={"figures_version": FIGURES_VERSION, "level_role": role,
                       "tables_report_commit": published.run_record.get("report_commit"),
                       "tables_created_utc": published.run_record.get("created_utc")},
            ),
        })
    return {**(staged.outcome or {}), "written": list(drawn) + [MANIFEST_FILE],
            "not_drawn": dict(not_drawn)}


def check_published_figures(figures_dir: Path, tables_dir: Path) -> dict[str, Any]:
    """Published figures still as drawn, from the tables as they are now, or a refusal.

    MANIFEST.json must be the figures mode's. The directory must hold exactly
    the figures it names and itself, each figure with its sha256. Every
    figure must name the sha256 that each of its tables has now, in the
    tables read through read_published_tables. The figures and the tables
    must name one evaluated run. Raises ProvenanceError naming each problem.
    Returns MANIFEST.json's content.
    """
    directory = Path(figures_dir)
    context = f"the published figures at {directory} are not the current tables' figures"
    try:
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProvenanceError(
            [("manifest", f"{MANIFEST_FILE} cannot be read: {error}")], context) from None
    if not isinstance(manifest, dict):
        raise ProvenanceError([("manifest", f"{MANIFEST_FILE} is not a JSON object")], context)
    problems: list[tuple[str, str]] = []
    if manifest.get("kind") != FIGURES_MANIFEST_KIND:
        problems.append(("manifest", (
            f"names kind {manifest.get('kind')!r}, not {FIGURES_MANIFEST_KIND!r}")))
    if manifest.get("figures_version") != FIGURES_VERSION:
        problems.append(("manifest", (
            f"names figures version {manifest.get('figures_version')!r}, not "
            f"{FIGURES_VERSION}")))
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    described = manifest.get("figures") if isinstance(manifest.get("figures"), dict) else {}
    if set(described) != set(files):
        problems.append(("manifest", (
            f"describes the figures {sorted(described)} and lists the files {sorted(files)}")))
    present = sorted(entry.name for entry in directory.iterdir())
    extra = [name for name in present if name != MANIFEST_FILE and name not in files]
    missing = sorted(name for name in files if name not in present)
    if extra:
        problems.append(("files", f"{extra} are in {directory} and not in {MANIFEST_FILE}"))
    if missing:
        problems.append(("files", f"{missing} are named in {MANIFEST_FILE} and absent"))
    for name, sha in sorted(files.items()):
        if name in missing:
            continue
        have = sha256_file(directory / name)
        if have != sha:
            problems.append(("sha256", (
                f"{name} has sha256 {have}, and {MANIFEST_FILE} names {sha}")))
    tables = read_published_tables(tables_dir, level=manifest.get("level"))
    for name, entry in sorted(described.items()):
        drawn_from = entry.get("tables") if isinstance(entry, dict) else None
        if not isinstance(drawn_from, dict) or not drawn_from:
            problems.append(("tables", f"{name} names no table it was drawn from"))
            continue
        for table, sha in sorted(drawn_from.items()):
            if tables.files.get(table) != sha:
                problems.append(("tables", (
                    f"{name} was drawn from {table} with sha256 {sha}, and the current "
                    f"{table} has {tables.files.get(table)}")))
    record = manifest.get("run_record") if isinstance(manifest.get("run_record"), dict) else {}
    for field in RUN_FIELDS:
        if record.get(field) != tables.run_record.get(field):
            problems.append(("run_record", (
                f"the figures name {field} {_short(record.get(field))}, and the tables "
                f"{_short(tables.run_record.get(field))}")))
    if problems:
        raise ProvenanceError(problems, context)
    return manifest
