"""Phase 5 figures: Stream AC, specification steps 39 to 43, from the tables alone.

The figures are drawn from synthetic tables: the tables lot.phase5_report
builds from the synthetic records of tests/test_phase5_report.py. In those
records every regime row is supported and every bin is not, so a test that
checks the grey shading first changes the support of chosen cells, and both
sides of the rule run. The figures mode then runs end to end on tables the
tables mode published, over a synthetic evaluated run on disk.

The drawing code names its panels and artists by gid. A test finds a band, a
hatched cell, a count, or a series by its gid. Each band is also checked by
its colour and its extent on the axis, so a gid that lies is caught.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pyarrow.parquet as pq  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import lot.evaluate as evaluate_module  # noqa: E402
import lot.phase5_figures as figures  # noqa: E402
import lot.phase5_provenance as provenance  # noqa: E402
import lot.phase5_report as report  # noqa: E402
import test_phase5_report as report_tests  # noqa: E402
from lot.datasets import parallax_bin_order, rotation_bin_order  # noqa: E402
from lot.evaluate import read_rows, read_run_metadata  # noqa: E402
from lot.phase5_check import sha256_file  # noqa: E402
from lot.phase5_estimands import CL_TRANSPORT, OFFSET_BINS, OFFSET_WHOLE  # noqa: E402
from lot.phase5_figures import (  # noqa: E402
    FIGURE1,
    FIGURE2,
    FIGURE3,
    FIGURE4,
    FIGURE5,
    FIGURE_FILES,
    FIGURE_S1,
    FIGURE_S2,
    FIGURE_TABLES,
    FiguresStop,
    bind_tables_to_run,
    check_published_figures,
    curve_rows,
    figures_for_level,
    read_published_tables,
)
from lot.phase5_outcomes import METRICS, REGIME_SCOPES  # noqa: E402
from lot.phase5_provenance import (  # noqa: E402
    ProvenanceError,
    require_evaluated_run,
    write_parquet_with_record,
)
from test_phase5_provenance import build_run  # noqa: E402

FAST = report_tests.FAST
LEVEL = report_tests.LEVEL
SCENES = list(report_tests.SCENES)
FOLDS = report_tests.FOLDS
PARALLAX_ORDER = parallax_bin_order(FAST)
ROTATION_ORDER = rotation_bin_order(FAST)
# lot.figures.UNSUPPORTED_BAND: colour 0.55 at alpha 0.22.
GREY_BAND = (0.55, 0.55, 0.55, 0.22)
PNG = b"\x89PNG\r\n\x1a\n"
LETTER_LABEL = re.compile(r"(?<![A-Za-z0-9_])[ABC](?![A-Za-z0-9_])")

PRIMARY = report.PRIMARY_TABLE
FORMULATION = report.FORMULATION_TABLE
SPLAT = report.SPLAT_TABLE
CROSS = report.CROSS_PATH_TABLE
OFFSET = report.LANDING_OFFSET_TABLE
REGIONS = report.REGION_TABLE
ADEQUACY = report.ADEQUACY_TABLE
HISTORY = report.HISTORY_TABLE

CURVES = [(FIGURE1, "translation", PARALLAX_ORDER), (FIGURE2, "rotation", ROTATION_ORDER)]

EXPECTED_PANELS = {
    FIGURE1: {f"{kind}:{m}" for kind in ("gap", "absolute") for m in METRICS},
    FIGURE2: {f"{kind}:{m}" for kind in ("gap", "absolute") for m in METRICS},
    FIGURE3: {f"{kind}:{m}" for kind in ("regimes", "rotation", "translation", "orbit")
              for m in METRICS},
    FIGURE4: {f"{kind}:{m}" for kind in (
        "regimes", "rotation", "translation", "orbit:delta_learn_pp", "orbit:delta_learn_sp",
        "orbit:path_difference_learn") for m in METRICS},
    FIGURE5: {f"{kind}:{m}" for kind in ("boundary", "boundary_contrast", "texture",
                                         "texture_contrast") for m in METRICS},
    FIGURE_S1: {f"fold:{fold.index}" for fold in FOLDS},
    FIGURE_S2: {f"{regime}:{m}" for regime in REGIME_SCOPES for m in METRICS},
}
TITLES = {
    FIGURE1: "Figure 1:", FIGURE2: "Figure 2:", FIGURE3: "Figure 3:", FIGURE4: "Figure 4:",
    FIGURE5: "Figure 5:", FIGURE_S1: "Supplementary figure S1:",
    FIGURE_S2: "Supplementary figure S2:",
}


# ---------------------------------------------------------------------------
# Drawing and reading the drawn figures
# ---------------------------------------------------------------------------

def draw(name: str, tables: dict, path: Path, **replaced) -> Figure:
    """Draw one figure from tables, with the tables named in replaced swapped."""
    t = {**tables, **replaced}
    if name == FIGURE1:
        return figures.figure1_gap_vs_parallax(t[PRIMARY], path, FAST)
    if name == FIGURE2:
        return figures.figure2_gap_vs_rotation(t[PRIMARY], path, FAST)
    if name == FIGURE3:
        return figures.figure3_formulation_gap(t[FORMULATION], path, FAST)
    if name == FIGURE4:
        return figures.figure4_primary_vs_operational(t[PRIMARY], t[SPLAT], t[CROSS], path, FAST)
    if name == FIGURE5:
        return figures.figure5_error_regions(t[REGIONS], path, FAST)
    if name == FIGURE_S1:
        return figures.figureS1_validation_curves(t[HISTORY], t[ADEQUACY], path)
    if name == FIGURE_S2:
        return figures.figureS2_landing_offset(t[OFFSET], path, FAST)
    raise AssertionError(name)


def panels(figure: Figure) -> dict:
    out = {}
    for axes in figure.axes:
        gid = axes.get_gid()
        if gid:
            assert gid not in out, gid
            out[gid] = axes
    return out


def bands(panel) -> list[int]:
    """The positions a panel greys as below support, checked by colour and extent."""
    found = []
    for patch in panel.patches:
        gid = patch.get_gid() or ""
        grey = tuple(round(float(c), 3) for c in patch.get_facecolor()) == GREY_BAND
        if not gid.startswith("unsupported:"):
            assert not grey, f"a grey band without its gid: {gid!r}"
            continue
        assert grey, gid
        position = int(gid.split(":")[1])
        vertices = patch.get_patch_transform().transform(patch.get_path().vertices)
        assert vertices[:, 0].min() == pytest.approx(position - 0.5), gid
        assert vertices[:, 0].max() == pytest.approx(position + 0.5), gid
        found.append(position)
    return sorted(found)


def hatched(panel) -> set[tuple[int, int]]:
    out = set()
    for patch in panel.patches:
        gid = patch.get_gid() or ""
        if gid.startswith("hatched:"):
            assert patch.get_hatch() == "///", gid
            _, i, j = gid.split(":")
            out.add((int(i), int(j)))
        else:
            assert not patch.get_hatch(), f"a hatched patch without its gid: {gid!r}"
    return out


def texts_by(panel, prefix: str) -> dict:
    out = {}
    for text in panel.texts:
        gid = text.get_gid() or ""
        if gid.startswith(prefix + ":"):
            key = tuple(int(part) for part in gid.split(":")[1:])
            out[key[0] if len(key) == 1 else key] = text.get_text()
    return out


def counts(panel) -> dict:
    return texts_by(panel, "count")


def points(panel, quantity: str, kind: str = "all") -> dict[int, float]:
    """One series' plotted values by position: supported, below support, or all."""
    gids = {"supported": [f"series:{quantity}"],
            "below": [f"series:{quantity}:below-support"]}
    wanted = gids.get(kind, gids["supported"] + gids["below"])
    out: dict[int, float] = {}
    for line in panel.get_lines():
        if (line.get_gid() or "") not in wanted:
            continue
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            position = int(round(float(x)))
            assert position not in out, (quantity, position)
            out[position] = float(y)
    return out


def xs(panel, quantity: str) -> list[float]:
    found = []
    for line in panel.get_lines():
        if (line.get_gid() or "").startswith(f"series:{quantity}"):
            found += [float(x) for x in line.get_xdata()]
    return found


def legend_texts(panel) -> list[str]:
    legend = panel.get_legend()
    return [] if legend is None else [text.get_text() for text in legend.get_texts()]


def figure_legend(figure: Figure) -> list[str]:
    """The texts of the one legend below a figure's panels."""
    assert len(figure.legends) == 1
    return [text.get_text() for text in figure.legends[0].get_texts()]


def suptitle(figure: Figure) -> str:
    titles = [text.get_text() for text in figure.texts if text.get_gid() == "suptitle"]
    assert len(titles) == 1
    return " ".join(titles[0].split())


def all_texts(figure: Figure) -> list[str]:
    found = [text.get_text() for text in figure.texts]
    for axes in figure.axes:
        found += [axes.get_title(), axes.get_xlabel(), axes.get_ylabel()]
        found += [label.get_text() for label in axes.get_xticklabels() + axes.get_yticklabels()]
        found += [text.get_text() for text in axes.texts]
        found += legend_texts(axes)
    for legend in figure.legends:
        found += [text.get_text() for text in legend.get_texts()]
    return [text for text in found if text]


def supported_as(rows: list[dict], changes: dict) -> list[dict]:
    """A copy of rows with the support of chosen cells changed.

    changes maps (analysis, bin) to the support both metrics' rows take."""
    out = copy.deepcopy(rows)
    for row in out:
        key = (row["analysis"], row["bin"])
        if key in changes:
            row["supported"] = changes[key]
    return out


def positioned(rows, regime: str, metric: str, order: list[str]) -> dict[int, dict]:
    """A regime's rows under one metric, at their positions on its curve.

    Each bin sits at its index in the frozen order, and the regime row after
    the last bin."""
    axis = "parallax_bin" if regime == "translation" else "rotation_bin"
    out = {}
    for row in rows:
        if row["metric"] != metric or row["analysis"] != regime:
            continue
        if row["axis"] == "all":
            out[len(order)] = row
        elif row["axis"] == axis:
            out[order.index(row["bin"])] = row
    return out


def joint_cells(rows, metric: str) -> dict[tuple[int, int], dict]:
    return {(ROTATION_ORDER.index(row["rotation_bin"]), PARALLAX_ORDER.index(row["parallax_bin"])):
            row for row in rows
            if row["metric"] == metric and row["analysis"] == "orbit" and row["axis"] != "all"}


def regime_rows(rows, metric: str) -> dict[int, dict]:
    return {REGIME_SCOPES.index(row["analysis"]): row for row in rows
            if row["metric"] == metric and row["analysis"] in REGIME_SCOPES
            and row["axis"] == "all"}


@pytest.fixture(scope="module")
def tables():
    """Every table, built in memory from the tables tests' synthetic records."""
    return report_tests.build().tables


@pytest.fixture(scope="module")
def drawn(tables, tmp_path_factory):
    directory = tmp_path_factory.mktemp("drawn")
    return {name: (draw(name, tables, directory / name), directory / name)
            for name in FIGURE_FILES}


# ---------------------------------------------------------------------------
# Every figure, from the tables
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", FIGURE_FILES)
def test_each_figure_is_drawn_from_the_tables(drawn, name):
    figure, path = drawn[name]
    assert isinstance(figure, Figure)
    assert path.read_bytes()[:8] == PNG
    assert set(panels(figure)) == EXPECTED_PANELS[name]
    assert suptitle(figure).startswith(TITLES[name])
    assert matplotlib.get_backend().lower() == "agg"


@pytest.mark.parametrize("name", FIGURE_FILES)
def test_the_caption_and_the_legend_clear_every_panel(drawn, name):
    """No panel sits under the caption, and no legend covers a panel.

    Supplementary figure S1 keeps a legend in each fold's panel, because its
    labels differ by fold. Every other figure has one legend, below its panels.
    """
    figure, _ = drawn[name]
    renderer = figure.canvas.get_renderer()
    boxes = [axes.get_tightbbox(renderer) for axes in figure.axes]
    title = next(text for text in figure.texts if text.get_gid() == "suptitle")
    assert title.get_window_extent(renderer).y0 >= max(box.y1 for box in boxes) - 1.0
    if name == FIGURE_S1:
        assert not figure.legends
        assert all(panel.get_legend() is not None for panel in panels(figure).values())
        return
    assert len(figure.legends) == 1
    assert figure.legends[0].get_window_extent(renderer).y1 <= min(box.y0 for box in boxes) + 1.0
    assert all(panel.get_legend() is None for panel in panels(figure).values())


@pytest.mark.parametrize("name", [name for name in FIGURE_FILES if name != FIGURE_S1])
def test_the_counts_sit_below_every_marker_and_interval(drawn, name):
    """The counts have the foot of each panel to themselves, in display space."""
    figure, _ = drawn[name]
    renderer = figure.canvas.get_renderer()
    checked = 0
    for gid, panel in panels(figure).items():
        labels = [text for text in panel.texts if (text.get_gid() or "").startswith("count:")]
        if not labels:
            continue
        top = max(text.get_window_extent(renderer).y1 for text in labels)
        lowest = math.inf
        for line in panel.get_lines():
            if (line.get_gid() or "").startswith("series:"):
                ys = panel.transData.transform(
                    list(zip(line.get_xdata(), line.get_ydata())))[:, 1]
                radius = line.get_markersize() * figure.dpi / 72.0 / 2.0
                lowest = min(lowest, float(ys.min()) - radius)
        for collection in panel.collections:
            for segment in collection.get_segments():
                lowest = min(lowest, float(panel.transData.transform(segment)[:, 1].min()))
        assert lowest > top, (name, gid)
        checked += 1
    assert checked


def test_the_panels_are_moved_below_a_caption_the_layout_left_over_them():
    """Some matplotlib versions lay out no suptitle. The clearance is measured."""
    plt = matplotlib.pyplot
    figure, _ = plt.subplots(2, 2, figsize=(8.0, 6.0))
    title = figure.suptitle("first line\nsecond line\nthird line", fontsize=9)
    figure.subplots_adjust(top=0.98)
    renderer = figure.canvas.get_renderer()

    def highest() -> float:
        return max(axes.get_tightbbox(renderer).y1 for axes in figure.axes)

    assert highest() > title.get_window_extent(renderer).y0
    figures._clear_title(figure, title)
    assert highest() <= title.get_window_extent(renderer).y0
    plt.close(figure)


def test_no_figure_text_holds_an_em_dash_or_a_letter_method_label(drawn):
    for name, (figure, _) in drawn.items():
        for text in all_texts(figure):
            assert "\u2014" not in text and "\u2013" not in text, (name, text)
            assert not LETTER_LABEL.search(text), (name, text)
            assert not re.search(r"\bmethod [ABC]\b", text, re.IGNORECASE), (name, text)
    source = Path(figures.__file__).read_text(encoding="utf-8")
    assert "\u2014" not in source and "\u2013" not in source
    assert not re.search(r"\bmethod [ABC]\b", source)


def test_a_figure_refuses_a_table_without_a_column_it_reads(tables, tmp_path):
    rows = [{k: v for k, v in row.items() if k != "delta_learn_pp_ci_low"}
            for row in tables[PRIMARY]]
    with pytest.raises(FiguresStop, match="delta_learn_pp_ci_low"):
        figures.figure1_gap_vs_parallax(rows, tmp_path / "figure1.png", FAST)
    assert not (tmp_path / "figure1.png").exists()


# ---------------------------------------------------------------------------
# Figures 1 and 2: the headline gap against parallax and rotation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, regime, order", CURVES)
def test_the_gap_curve_plots_each_cell_at_its_frozen_bin(drawn, tables, name, regime, order):
    figure, _ = drawn[name]
    for metric in METRICS:
        cells = positioned(tables[PRIMARY], regime, metric, order)
        gap = panels(figure)[f"gap:{metric}"]
        assert points(gap, "delta_learn_pp") == pytest.approx(
            {p: row["delta_learn_pp"] for p, row in cells.items()})
        assert [label.get_text() for label in gap.get_xticklabels()] == order + ["all pairs"]
        assert gap.get_title().startswith("delta_learn_pp")
        absolute = panels(figure)[f"absolute:{metric}"]
        for quantity in ("cl_transport", "predict_with_depth", "no_warp_copy"):
            assert points(absolute, quantity) == pytest.approx(
                {p: row[quantity] for p, row in cells.items()}), quantity
        if metric == "raw":
            assert points(absolute, "mean_feature") == pytest.approx(
                {p: row["mean_feature"] for p, row in cells.items()})
        else:
            assert points(absolute, "mean_feature") == {}
            assert "Mean-Feature not applicable" in absolute.get_title()
    assert "centered" in panels(figure)["gap:centered"].get_title()
    assert "primary" in panels(figure)["gap:centered"].get_title()


@pytest.mark.parametrize("name, regime, order", CURVES)
def test_a_bin_below_support_is_greyed_and_every_bin_shows_its_count(tables, tmp_path, name,
                                                                      regime, order):
    bins = sorted({row["bin"] for row in tables[PRIMARY]
                   if row["analysis"] == regime and row["axis"] != "all"}, key=order.index)
    assert len(bins) >= 2
    # The records leave every bin unsupported and the regime row supported.
    rows = supported_as(tables[PRIMARY], {(regime, bins[0]): True, (regime, "all"): False})
    figure = draw(name, tables, tmp_path / name, **{PRIMARY: rows})
    greyed = sorted([order.index(label) for label in bins[1:]] + [len(order)])
    for metric in METRICS:
        cells = positioned(rows, regime, metric, order)
        for kind in ("gap", "absolute"):
            panel = panels(figure)[f"{kind}:{metric}"]
            assert bands(panel) == greyed, (kind, metric)
            # Every populated bin shows its n, and a blank bin shows nothing.
            assert counts(panel) == {p: f"n={row['n_camera_pairs']}" for p, row in cells.items()}
        gap = panels(figure)[f"gap:{metric}"]
        assert sorted(points(gap, "delta_learn_pp", "below")) == greyed
        assert sorted(points(gap, "delta_learn_pp", "supported")) == [order.index(bins[0])]
    assert "below support" in figure_legend(figure)
    assert figures.BELOW_SUPPORT_MARKER in figure_legend(figure)


@pytest.mark.parametrize("name, axis", [(FIGURE1, "parallax_bin"), (FIGURE2, "rotation_bin")])
def test_figures_1_and_2_refuse_an_orbit_row_on_their_axis(tables, tmp_path, name, axis):
    rows = copy.deepcopy(tables[PRIMARY])
    orbit = next(row for row in rows if row["analysis"] == "orbit" and row["axis"] != "all")
    rows.append({**orbit, "axis": axis, "bin": orbit[axis]})
    with pytest.raises(ValueError, match="orbit"):
        draw(name, tables, tmp_path / name, **{PRIMARY: rows})
    assert not (tmp_path / name).exists()


@pytest.mark.parametrize("name, regime", [(FIGURE1, "translation"), (FIGURE2, "rotation")])
def test_figures_1_and_2_refuse_their_regime_on_the_joint_axis(tables, tmp_path, name, regime):
    rows = copy.deepcopy(tables[PRIMARY])
    orbit = next(row for row in rows if row["analysis"] == "orbit" and row["axis"] != "all")
    rows.append({**orbit, "analysis": regime})
    with pytest.raises(ValueError, match="joint"):
        draw(name, tables, tmp_path / name, **{PRIMARY: rows})


def test_orbit_has_no_primary_curve(tables):
    with pytest.raises(ValueError, match="orbit"):
        curve_rows(tables[PRIMARY], "orbit", "centered")
    scope, bins = curve_rows(tables[PRIMARY], "translation", "centered")
    assert scope["analysis"] == "translation" and scope["axis"] == "all"
    assert {row["analysis"] for row in bins.values()} == {"translation"}


@pytest.mark.parametrize("name, regime, order", CURVES)
def test_the_read_deficit_sits_beside_the_gap_labelled_with_its_direction(tables, tmp_path, name,
                                                                         regime, order):
    rows = copy.deepcopy(tables[PRIMARY])
    first = min(positioned(rows, regime, "centered", order))
    hidden = order[first]
    # The read deficit has its own support. Every cell of the regime holds it
    # here except the first bin's.
    for row in rows:
        if row["analysis"] == regime:
            row["read_deficit_supported"] = row["bin"] != hidden
    figure = draw(name, tables, tmp_path / name, **{PRIMARY: rows})
    assert "a positive deficit works against Context-Lift" in figures.READ_DEFICIT_LABEL
    for metric in METRICS:
        cells = positioned(rows, regime, metric, order)
        gap = panels(figure)[f"gap:{metric}"]
        assert points(gap, "read_deficit") == pytest.approx(
            {p: row["read_deficit"] for p, row in cells.items()})
        # Beside each bin, never on the gap's own marker, and never combined.
        assert not set(xs(gap, "read_deficit")) & set(xs(gap, "delta_learn_pp"))
        assert not [line for line in gap.get_lines() if "corrected" in (line.get_gid() or "")]
        assert figures.READ_DEFICIT_LABEL in figure_legend(figure)
        # A read deficit below its own support is greyed and shows its count.
        assert sorted(points(gap, "read_deficit", "below")) == [first]
        assert texts_by(gap, "deficit-count") == {
            first: f"n={cells[first]['read_deficit_n_camera_pairs']}"}


def test_without_the_landing_offset_columns_the_overlay_is_left_out_and_said_so(tables, tmp_path):
    dropped = ("lead_within_read", "cl_lead_understated", "read_deficit_anomaly",
               "read_depth_note")
    rows = [{k: v for k, v in row.items()
             if not k.startswith("read_deficit") and k not in dropped}
            for row in tables[PRIMARY]]
    figure = figures.figure1_gap_vs_parallax(rows, tmp_path / "figure1.png", FAST)
    for metric in METRICS:
        assert points(panels(figure)[f"gap:{metric}"], "read_deficit") == {}
    assert " ".join(figures.READ_DEFICIT_PRIMARY_ONLY.split()) in suptitle(figure)


def test_figure2_shows_context_lift_as_the_known_transform_reference(drawn):
    assert CL_TRANSPORT in figures.KNOWN_TRANSFORM_REFERENCE
    assert figures.KNOWN_TRANSFORM_REFERENCE in figure_legend(drawn[FIGURE2][0])
    legend = figure_legend(drawn[FIGURE1][0])
    assert CL_TRANSPORT in legend and figures.KNOWN_TRANSFORM_REFERENCE not in legend
    for metric in METRICS:
        # The reference is the Context-Lift curve of the absolute panels.
        assert points(panels(drawn[FIGURE2][0])[f"absolute:{metric}"], "cl_transport")
    assert "known-transform reference" in suptitle(drawn[FIGURE2][0])


# ---------------------------------------------------------------------------
# Figure 3: the formulation gap, a diagnostic
# ---------------------------------------------------------------------------

def test_figure3_is_labelled_a_diagnostic_and_hatches_joint_cells_below_support(tables, tmp_path):
    cells = sorted({row["bin"] for row in tables[FORMULATION]
                    if row["analysis"] == "orbit" and row["axis"] != "all"})
    rows = supported_as(tables[FORMULATION], {("orbit", cells[0]): True})
    figure = draw(FIGURE3, tables, tmp_path / "figure3.png", **{FORMULATION: rows})
    text = suptitle(figure)
    assert report.TABLE_LABELS[FORMULATION] in text
    assert "not a learned-model effect" in text
    for metric in METRICS:
        joint = joint_cells(rows, metric)
        heat = panels(figure)[f"orbit:{metric}"]
        assert hatched(heat) == {cell for cell, row in joint.items() if not row["supported"]}
        assert len(hatched(heat)) == len(joint) - 1
        # Every populated cell prints its value and n. A blank cell prints nothing.
        assert texts_by(heat, "cell") == {
            cell: f"{row['delta_formulation']:+.3f}\nn={row['n_camera_pairs']}"
            for cell, row in joint.items()}
        for regime, order in (("rotation", ROTATION_ORDER), ("translation", PARALLAX_ORDER)):
            panel = panels(figure)[f"{regime}:{metric}"]
            binned = {p: row for p, row in positioned(rows, regime, metric, order).items()
                      if p < len(order)}
            assert points(panel, "delta_formulation") == pytest.approx(
                {p: row["delta_formulation"] for p, row in binned.items()})
            assert bands(panel) == sorted(p for p, row in binned.items() if not row["supported"])
            assert counts(panel) == {p: f"n={row['n_camera_pairs']}" for p, row in binned.items()}
        regimes = panels(figure)[f"regimes:{metric}"]
        overall = regime_rows(rows, metric)
        assert points(regimes, "delta_formulation") == pytest.approx(
            {p: row["delta_formulation"] for p, row in overall.items()})
        assert [label.get_text() for label in regimes.get_xticklabels()] == list(REGIME_SCOPES)
        assert bands(regimes) == []


# ---------------------------------------------------------------------------
# Figure 4: the primary and the operational gap
# ---------------------------------------------------------------------------

def test_figure4_greys_each_series_on_its_own_support_and_claims_no_implementation_error(
    tables, tmp_path
):
    bins = sorted({row["bin"] for row in tables[PRIMARY]
                   if row["analysis"] == "translation" and row["axis"] != "all"},
                  key=PARALLAX_ORDER.index)
    # Every bin is below support in every table. One per-point bin is given
    # support, and the splat-pool regime row loses it.
    primary = supported_as(tables[PRIMARY], {("translation", bins[0]): True})
    splat = supported_as(tables[SPLAT], {("translation", "all"): False})
    figure = draw(FIGURE4, tables, tmp_path / "figure4.png", **{PRIMARY: primary, SPLAT: splat})
    end = len(PARALLAX_ORDER)
    lifted = PARALLAX_ORDER.index(bins[0])
    translation = REGIME_SCOPES.index("translation")
    for metric in METRICS:
        panel = panels(figure)[f"translation:{metric}"]

        def binned(rows):
            return {p: row for p, row in positioned(rows, "translation", metric,
                                                     PARALLAX_ORDER).items() if p < end}

        pp, sp, x = binned(primary), binned(splat), binned(tables[CROSS])
        assert points(panel, "delta_learn_pp") == pytest.approx(
            {p: row["delta_learn_pp"] for p, row in pp.items()})
        assert points(panel, "delta_learn_sp") == pytest.approx(
            {p: row["delta_learn_sp"] for p, row in sp.items()})
        assert points(panel, "path_difference_learn") == pytest.approx(
            {p: row["path_difference_learn"] for p, row in x.items()})
        # The band follows the headline cell. Each series greys on its own support.
        assert bands(panel) == sorted(p for p, row in pp.items() if not row["supported"])
        assert lifted not in bands(panel)
        assert lifted in points(panel, "delta_learn_pp", "supported")
        assert lifted in points(panel, "delta_learn_sp", "below")
        # One line per population: V_P5_pp, V_sp, and the common cells.
        assert counts(panel) == {
            p: "n=" + "\n".join(str(rows[p]["n_camera_pairs"]) for rows in (pp, sp, x))
            for p in pp}
        regimes = panels(figure)[f"regimes:{metric}"]
        assert translation not in bands(regimes)
        assert translation in points(regimes, "delta_learn_pp", "supported")
        assert points(regimes, "delta_learn_sp", "below") == pytest.approx(
            {translation: regime_rows(splat, metric)[translation]["delta_learn_sp"]})
        assert [label.get_text() for label in regimes.get_xticklabels()] == list(REGIME_SCOPES)
        for quantity, rows in (("delta_learn_pp", primary), ("delta_learn_sp", splat),
                               ("path_difference_learn", tables[CROSS])):
            heat = panels(figure)[f"orbit:{quantity}:{metric}"]
            joint = joint_cells(rows, metric)
            assert hatched(heat) == {cell for cell, row in joint.items() if not row["supported"]}
            assert set(texts_by(heat, "cell")) == set(joint)
    text = suptitle(figure)
    assert " ".join(figures.NOT_AN_IMPLEMENTATION_ERROR.split()) in text
    for line in all_texts(figure):
        if "implementation error" in line:
            assert "not interpreted as an implementation error" in " ".join(line.split())


# ---------------------------------------------------------------------------
# Figure 5: the error regions and their paired contrasts
# ---------------------------------------------------------------------------

SPLITS = (("boundary", "boundary", "interior", "boundary_minus_interior"),
          ("texture", "low_texture", "high_texture", "low_minus_high_texture"))


def region_cells(rows, metric: str, region: str, kind: str) -> dict[int, dict]:
    return {REGIME_SCOPES.index(row["analysis"]): row for row in rows
            if row["metric"] == metric and row["region"] == region and row["path"] == "per_point"
            and row["row_kind"] == kind and row["analysis"] in REGIME_SCOPES}


def test_figure5_draws_each_region_and_the_paired_contrasts(tables, tmp_path):
    rows = copy.deepcopy(tables[REGIONS])
    for row in rows:
        if (row["analysis"], row["region"], row["path"]) == ("rotation", "boundary", "per_point"):
            row["supported"] = False
    figure = draw(FIGURE5, tables, tmp_path / "figure5.png", **{REGIONS: rows})
    rotation = REGIME_SCOPES.index("rotation")
    for metric in METRICS:
        for split, left, right, contrast in SPLITS:
            panel = panels(figure)[f"{split}:{metric}"]
            for region in (left, right):
                cells = region_cells(rows, metric, region, "region")
                assert points(panel, f"delta_learn_pp@{region}") == pytest.approx(
                    {p: row["delta_learn_pp"] for p, row in cells.items()}), region
            lefts = region_cells(rows, metric, left, "region")
            rights = region_cells(rows, metric, right, "region")
            assert counts(panel) == {
                p: f"n={lefts[p]['n_camera_pairs']}/{rights[p]['n_camera_pairs']}" for p in lefts}
            contrasts = panels(figure)[f"{split}_contrast:{metric}"]
            cells = region_cells(rows, metric, contrast, "contrast")
            for quantity in ("contrast_delta_learn_pp", "contrast_cl_transport",
                             "contrast_predict_with_depth"):
                assert points(contrasts, quantity) == pytest.approx(
                    {p: row[quantity] for p, row in cells.items()}), quantity
            assert counts(contrasts) == {p: f"n={row['n_camera_pairs']}" for p, row in cells.items()}
            assert bands(contrasts) == sorted(p for p, row in cells.items() if not row["supported"])
        boundary = panels(figure)[f"boundary:{metric}"]
        assert bands(boundary) == [rotation]
        assert sorted(points(boundary, "delta_learn_pp@boundary", "below")) == [rotation]
        assert bands(panels(figure)[f"texture:{metric}"]) == []
    assert "No near-zero wording applies to a contrast" in suptitle(figure)


# ---------------------------------------------------------------------------
# Supplementary figures
# ---------------------------------------------------------------------------

def test_figure_s1_draws_every_run_and_marks_its_selected_checkpoint(drawn, tables):
    figure, _ = drawn[FIGURE_S1]
    assert len(tables[ADEQUACY]) == len(FOLDS) * len(report_tests.SEEDS)
    for row in tables[ADEQUACY]:
        fold, seed = row["fold"], row["seed"]
        lines = {line.get_gid(): line for line in panels(figure)[f"fold:{fold}"].get_lines()}
        best = lines[f"best:{fold}:{seed}"]
        assert list(best.get_xdata()) == [row["best_step"]]
        assert list(best.get_ydata()) == [row["best_validation_centered_cosine"]]
        history = sorted((h for h in tables[HISTORY] if (h["fold"], h["seed"]) == (fold, seed)),
                         key=lambda h: h["validation_index"])
        curve = lines[f"history:{fold}:{seed}"]
        assert list(curve.get_xdata()) == [h["step"] for h in history]
        assert list(curve.get_ydata()) == [h["validation_centered_cosine"] for h in history]
        assert ("not stable" in curve.get_label()) == (not row["stable_validation_curve"])
    # The synthetic run of fold 2, seed 2 ran out of steps below its best.
    assert any(not row["stable_validation_curve"] for row in tables[ADEQUACY])
    assert "not an estimand" in suptitle(figure)


def test_figure_s1_refuses_a_history_that_disagrees_with_its_adequacy_row(tables, tmp_path):
    history = copy.deepcopy(tables[HISTORY])
    for row in history:
        if (row["fold"], row["seed"]) == (0, 0):
            row["is_best"] = row["validation_index"] == 0
    with pytest.raises(FiguresStop, match="best"):
        figures.figureS1_validation_curves(history, tables[ADEQUACY], tmp_path / "s1.png")


def test_figure_s2_draws_the_offset_curves_with_their_reading_notes(drawn, tables):
    figure, _ = drawn[FIGURE_S2]
    text = suptitle(figure)
    assert report.OFFSET_READING_NOTE in text
    assert report.READ_DEPTH_NOTE in text
    labels = OFFSET_BINS + (OFFSET_WHOLE,)
    for regime in REGIME_SCOPES:
        for metric in METRICS:
            panel = panels(figure)[f"{regime}:{metric}"]
            rows = {row["offset_label"]: row for row in tables[OFFSET]
                    if (row["analysis"], row["axis"], row["metric"]) == (regime, "all", metric)}
            for quantity in ("cl_oracle_offset", "nowarp_offset"):
                assert points(panel, quantity) == pytest.approx({
                    k: rows[label][quantity] for k, label in enumerate(labels)
                    if math.isfinite(rows[label][quantity])}), quantity
            assert bands(panel) == [k for k, label in enumerate(labels)
                                    if not rows[label]["supported"]]
            assert 5 in bands(panel)  # the synthetic b5 holds no landing
            assert counts(panel) == {k: f"n={rows[label]['n_camera_pairs']}"
                                     for k, label in enumerate(labels)}
            mean_feature = points(panel, "meanfeat_offset")
            assert bool(mean_feature) == (metric == "raw")
            assert f"{rows['deficit']['read_deficit']:+.4f}" in panel.get_title()


@pytest.mark.parametrize("name", [FIGURE1, FIGURE2])
def test_a_headline_figure_at_another_level_says_the_headline_is_the_primary_levels(
    tables, tmp_path, name
):
    """reporting_rules.md decision 4: a sensitivity level is reported beside the
    primary result and never replaces it. So its Figures 1 and 2 name their
    level and its role, and Figure 1 is not called the headline figure."""
    affine = report_tests.build(level="affine", primary=LEVEL, phase4=False).tables
    draw_one = {FIGURE1: figures.figure1_gap_vs_parallax,
                FIGURE2: figures.figure2_gap_vs_rotation}[name]
    text = " ".join(suptitle(draw_one(affine[PRIMARY], tmp_path / "affine.png", FAST,
                                      role="sensitivity")).split())
    assert "The headline figure." not in text
    assert "at the sensitivity level affine" in text, text
    assert "The headline itself is the primary level's." in text
    primary = " ".join(suptitle(draw_one(tables[PRIMARY], tmp_path / "primary.png",
                                         FAST)).split())
    assert ("The headline figure." in primary) == (name == FIGURE1)
    assert "sensitivity" not in primary and "primary level's" not in primary


def test_the_landing_offset_figure_is_drawn_at_the_primary_level_only():
    drawn_now, skipped = figures_for_level("image", "image")
    assert drawn_now == list(FIGURE_FILES) and skipped == {}
    drawn_now, skipped = figures_for_level("affine", "image")
    assert FIGURE_S2 not in drawn_now and set(skipped) == {FIGURE_S2}
    assert set(drawn_now) | set(skipped) == set(FIGURE_FILES)
    assert FIGURE1 in drawn_now and FIGURE2 in drawn_now


# ---------------------------------------------------------------------------
# The figures mode on disk: tables verified, figures written once
# ---------------------------------------------------------------------------

def run_figures(run, **kwargs):
    arguments = {"expected_scenes": SCENES, "folds": FOLDS, "check_code": False, **kwargs}
    return figures.run_figures(run.cfg, FAST, run.config, LEVEL, **arguments)


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    """A synthetic evaluated run with its tables published, then its figures."""
    patcher = pytest.MonkeyPatch()
    try:
        run = build_run(tmp_path_factory.mktemp("figures_run"), patcher)
        report_tests._stage_synthetic_rows(run, FAST)
        report_tests.run_tables(run)
        yield run, run_figures(run)
    finally:
        patcher.undo()


def tables_dir(run) -> Path:
    return Path(run.run_dir) / "tables" / LEVEL


def copy_tables(run, tmp_path: Path) -> Path:
    target = tmp_path / "tables" / LEVEL
    shutil.copytree(tables_dir(run), target)
    return target


def rewrite_table(directory: Path, name: str, record_change=None, row_change=None) -> None:
    """Rewrite one table in place, and name its new sha256 in MANIFEST.json."""
    path = directory / name
    rows, record = read_rows(path), read_run_metadata(path)
    if record_change is not None:
        record_change(record)
    if row_change is not None:
        rows = [row_change(dict(row)) for row in rows]
    path.unlink()
    write_parquet_with_record(path, rows, record)
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][name] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def refused_tables(directory: Path, field: str, match: str | None = None, **kwargs):
    with pytest.raises(ProvenanceError) as error:
        read_published_tables(directory, **kwargs)
    assert field in error.value.fields, str(error.value)
    if match is not None:
        assert re.search(match, str(error.value)), str(error.value)
    return error.value


def test_the_published_tables_are_read_after_their_hashes_and_records_are_checked(published):
    run, _ = published
    directory = tables_dir(run)
    tables_read = read_published_tables(directory, level=LEVEL)
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    assert tables_read.files == manifest["files"]
    assert tables_read.run_record == manifest["run_record"]
    assert tables_read.manifest_sha256 == sha256_file(directory / "MANIFEST.json")
    assert set(tables_read.tables) == set(report.TABLE_FILES)
    assert set(tables_read.documents) == {report.NEAR_ZERO_FILE, report.ACCOUNTING_FILE}
    for name, rows in tables_read.tables.items():
        assert report_tests.same(rows, read_rows(directory / name)), name
    assert tables_read.rows(PRIMARY) is tables_read.tables[PRIMARY]


def test_a_table_whose_bytes_changed_is_refused(published, tmp_path):
    run, _ = published
    directory = copy_tables(run, tmp_path)
    path = directory / PRIMARY
    path.write_bytes(path.read_bytes() + b"\0")
    refused_tables(directory, "sha256", re.escape(PRIMARY))


def test_a_table_rewritten_under_another_run_record_is_refused(published, tmp_path):
    run, _ = published
    directory = copy_tables(run, tmp_path)
    rewrite_table(directory, FORMULATION,
                  record_change=lambda record: record.update(evaluation_commit="0" * 40))
    refused_tables(directory, "run_record", "evaluation_commit")


def test_a_table_that_names_another_table_is_refused(published, tmp_path):
    run, _ = published
    directory = copy_tables(run, tmp_path)
    rewrite_table(directory, SPLAT, record_change=lambda record: record.update(table=PRIMARY))
    refused_tables(directory, "run_record", re.escape(SPLAT))


def test_a_file_the_manifest_does_not_name_is_refused(published, tmp_path):
    run, _ = published
    directory = copy_tables(run, tmp_path)
    (directory / "stray.parquet").write_bytes(b"not a table")
    refused_tables(directory, "files", "stray.parquet")


def test_tables_without_their_manifest_are_refused(published, tmp_path):
    run, _ = published
    directory = copy_tables(run, tmp_path)
    (directory / "MANIFEST.json").unlink()
    refused_tables(directory, "manifest")


def test_tables_of_another_level_are_refused(published, tmp_path):
    run, _ = published
    refused_tables(copy_tables(run, tmp_path), "level", level="affine")


def test_the_tables_must_be_the_evaluated_runs_own(published):
    run, _ = published
    identity = require_evaluated_run(run.cfg, FAST, run.config, LEVEL, expected_scenes=SCENES,
                                     folds=FOLDS, check_code=False)
    tables_read = read_published_tables(tables_dir(run), level=LEVEL)
    bind_tables_to_run(tables_read, identity)
    with pytest.raises(ProvenanceError) as error:
        bind_tables_to_run(tables_read, dataclasses.replace(identity, commit="0" * 40))
    assert "evaluation_commit" in error.value.fields
    inputs = dict(identity.inputs)
    scene = next(name for name in inputs if name.startswith("eval/"))
    inputs[scene] = "0" * 64
    with pytest.raises(ProvenanceError) as error:
        bind_tables_to_run(tables_read, dataclasses.replace(identity, inputs=inputs))
    assert "inputs" in error.value.fields
    # A receipt a later rerun moved aside keeps its content and binds by it.
    receipt = next(name for name in identity.inputs if name.endswith("integration_gate.json"))
    moved = {(name.replace("integration_gate.json", "integration_gate.superseded.1.json")
              if name == receipt else name): sha for name, sha in identity.inputs.items()}
    bind_tables_to_run(tables_read, dataclasses.replace(identity, inputs=moved))
    # An evaluation parquet binds by its scene's name as well as its content.
    eval_paths = dict(identity.eval_paths)
    first = next(iter(eval_paths))
    renamed = {**eval_paths, first: str(Path(eval_paths[first]).with_name("renamed.parquet"))}
    with pytest.raises(ProvenanceError) as error:
        bind_tables_to_run(tables_read, dataclasses.replace(identity, eval_paths=renamed))
    assert "inputs" in error.value.fields
    with pytest.raises(ProvenanceError) as error:
        bind_tables_to_run(tables_read, dataclasses.replace(identity, code_checked=True))
    assert "code_checked" in error.value.fields


def test_the_figures_mode_publishes_every_figure_with_a_manifest_of_its_tables(published):
    run, outcome = published
    final = Path(run.run_dir) / "figures" / LEVEL
    assert outcome["published"] == str(final) and outcome["superseded"] is None
    assert sorted(p.name for p in final.iterdir()) == sorted(FIGURE_FILES + ("MANIFEST.json",))
    assert outcome["written"][-1] == "MANIFEST.json"
    manifest = json.loads((final / "MANIFEST.json").read_text(encoding="utf-8"))
    tables_manifest = json.loads((tables_dir(run) / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "phase5_figures_manifest" and manifest["level"] == LEVEL
    # What the level is, so the headline marker below is read at its level.
    assert manifest["level_role"] == "primary"
    assert manifest["run_record"]["level_role"] == "primary"
    assert set(manifest["files"]) == set(FIGURE_FILES) == set(manifest["figures"])
    for name in FIGURE_FILES:
        assert (final / name).read_bytes()[:8] == PNG
        assert manifest["files"][name] == sha256_file(final / name)
        assert manifest["figures"][name]["tables"] == {
            table: tables_manifest["files"][table] for table in FIGURE_TABLES[name]}
    assert {name for name, entry in manifest["figures"].items() if entry["headline"]} == {
        FIGURE1, FIGURE2}
    assert manifest["tables_manifest"]["sha256"] == sha256_file(tables_dir(run) / "MANIFEST.json")
    record = manifest["run_record"]
    assert record["kind"] == "phase5_figures_manifest"
    assert record["evaluation_commit"] == run.commit
    for table in set().union(*FIGURE_TABLES.values()):
        assert record["inputs"][f"tables/{LEVEL}/{table}"] == tables_manifest["files"][table]
    assert record["inputs"][f"tables/{LEVEL}/MANIFEST.json"] == sha256_file(
        tables_dir(run) / "MANIFEST.json")
    assert any(name.startswith("eval/") for name in record["inputs"])
    assert not [p for p in final.parent.iterdir() if ".partial." in p.name]
    checked = check_published_figures(final, tables_dir(run))
    assert set(checked["figures"]) == set(FIGURE_FILES)


def test_a_non_primary_level_is_drawn_only_under_the_primary_levels_chain(published, tmp_path,
                                                                          monkeypatch):
    """reporting_rules.md decision 4. A non-primary level is drawn only beside the
    primary level's verified tables, and only when its run shares their commit
    E, their digests, and their gate receipts."""
    from lot.phase5_modes import checkpoint_lock_path

    run, _ = published
    identity = require_evaluated_run(run.cfg, FAST, run.config, LEVEL, expected_scenes=SCENES,
                                     folds=FOLDS, check_code=False)
    licence = {stem: sha for stem, sha in identity.licence.items()
               if stem in ("integration_gate", "tiny_overfit")}
    licence[checkpoint_lock_path(Path("."), "affine").stem] = "1" * 64
    affine = dataclasses.replace(identity, level="affine", licence=licence)

    def draw_affine(level_identity, tables_root=None):
        monkeypatch.setattr(figures, "require_evaluated_run", lambda *a, **k: level_identity)
        return figures.run_figures(
            run.cfg, FAST, run.config, "affine", expected_scenes=SCENES, folds=FOLDS,
            check_code=False, tables_root=tables_root, figures_root=tmp_path / "figures")

    with pytest.raises(ProvenanceError) as error:
        draw_affine(dataclasses.replace(affine, commit="0" * 40, training_commit="0" * 40))
    assert "evaluation_commit" in error.value.fields, str(error.value)
    with pytest.raises(ProvenanceError) as error:
        draw_affine(dataclasses.replace(affine, licence={**licence, "tiny_overfit": "0" * 64}))
    assert "licence.tiny_overfit" in error.value.fields, str(error.value)
    # Under the primary chain it gets past the binding, and stops only because
    # no affine tables are published.
    with pytest.raises(ProvenanceError) as error:
        draw_affine(affine)
    assert error.value.fields == ("tables",), str(error.value)
    # Without the primary tables, nothing is read.
    with pytest.raises(FileNotFoundError, match="primary"):
        draw_affine(affine, tables_root=tmp_path / "no_tables")
    root = tmp_path / "figures"
    assert not root.exists() or list(root.iterdir()) == []


def _with_role(directory: Path, role: str, names=None) -> None:
    """Rewrite the level role in the run records of the tables named, every table
    by default, and in MANIFEST.json's when every table is rewritten."""
    every = names is None
    names = list(report.TABLE_FILES) if every else list(names)
    for name in names:
        if name.endswith(".parquet"):
            rewrite_table(directory, name, record_change=lambda record: record.update(
                level_role=role))
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if every:
        for name in (report.NEAR_ZERO_FILE, report.ACCOUNTING_FILE):
            path = directory / name
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["run_record"]["level_role"] = role
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest["files"][name] = sha256_file(path)
        manifest["run_record"]["level_role"] = role
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_a_table_naming_another_level_role_than_its_manifest_is_refused(published, tmp_path):
    run, _ = published
    directory = copy_tables(run, tmp_path)
    _with_role(directory, "sensitivity", names=[FORMULATION])
    refused_tables(directory, "run_record", "level_role")


def test_tables_naming_another_role_than_the_configuration_gives_are_not_drawn(published,
                                                                              tmp_path):
    """The captions of Figures 1 and 2 read the role from the tables, so the
    tables must name the role the frozen configuration gives their level."""
    run, _ = published
    directory = copy_tables(run, tmp_path)
    _with_role(directory, "sensitivity")
    assert read_published_tables(directory, level=LEVEL).run_record["level_role"] == "sensitivity"
    with pytest.raises(FiguresStop, match="level role"):
        run_figures(run, tables_root=tmp_path / "tables", figures_root=tmp_path / "figures")
    root = tmp_path / "figures"
    assert not root.exists() or list(root.iterdir()) == []


def test_an_existing_figures_output_is_refused_before_any_work(published, monkeypatch):
    run, _ = published
    monkeypatch.setattr(figures, "require_evaluated_run",
                        lambda *a, **k: pytest.fail("the run was read"))
    with pytest.raises(FileExistsError, match="written once"):
        run_figures(run)


def _placeholder_figures(published_tables, directory, analysis, drawn):
    for name in drawn:
        (Path(directory) / name).write_bytes(f"the figure {name}".encode("utf-8"))


def test_supersede_moves_the_earlier_figures_aside_and_deletes_nothing(published, tmp_path,
                                                                       monkeypatch):
    run, _ = published
    monkeypatch.setattr(figures, "draw_figures", _placeholder_figures)
    root = tmp_path / "figures"
    first = run_figures(run, figures_root=root)
    final = Path(first["published"])
    before = {p.name: p.read_bytes() for p in final.iterdir()}
    second = run_figures(run, figures_root=root, supersede=True)
    archived = Path(second["superseded"])
    assert archived.name == f"{LEVEL}.superseded.1"
    assert {p.name: p.read_bytes() for p in archived.iterdir()} == before
    assert sorted(p.name for p in final.iterdir()) == sorted(before)


def test_a_table_hash_mismatch_stops_the_figures_and_publishes_nothing(published, tmp_path):
    run, _ = published
    directory = copy_tables(run, tmp_path)
    path = directory / PRIMARY
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ProvenanceError) as error:
        run_figures(run, tables_root=tmp_path / "tables", figures_root=tmp_path / "figures")
    assert "sha256" in error.value.fields
    root = tmp_path / "figures"
    assert not root.exists() or list(root.iterdir()) == []


def test_the_figures_stop_when_a_table_they_draw_is_absent(published, tmp_path):
    run, _ = published
    directory = copy_tables(run, tmp_path)
    (directory / OFFSET).unlink()
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["files"][OFFSET]
    manifest["run_record"]["tables"].remove(OFFSET)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    read_published_tables(directory, level=LEVEL)
    with pytest.raises(FiguresStop, match=re.escape(OFFSET)):
        run_figures(run, tables_root=tmp_path / "tables", figures_root=tmp_path / "figures")


def test_the_primary_level_figures_stop_without_the_read_deficit(published, tmp_path):
    run, _ = published
    directory = copy_tables(run, tmp_path)
    rewrite_table(directory, PRIMARY, row_change=lambda row: {
        k: v for k, v in row.items() if not k.startswith("read_deficit")})
    with pytest.raises(FiguresStop, match="read_deficit"):
        run_figures(run, tables_root=tmp_path / "tables", figures_root=tmp_path / "figures")


def test_a_failed_figure_publishes_nothing_and_leaves_no_staging(published, tmp_path,
                                                                 monkeypatch):
    run, _ = published

    def fail(*args, **kwargs):
        raise RuntimeError("the drawing failed")

    monkeypatch.setattr(figures, "figure3_formulation_gap", fail)
    with pytest.raises(RuntimeError, match="the drawing failed"):
        run_figures(run, figures_root=tmp_path / "figures")
    root = tmp_path / "figures"
    assert not root.exists() or list(root.iterdir()) == []


def test_the_figures_never_read_an_evaluation_parquet(published, tmp_path, monkeypatch):
    run, _ = published

    def refuse(*args, **kwargs):
        raise AssertionError("an evaluation parquet's rows were read")

    monkeypatch.setattr(evaluate_module, "read_rows", refuse)
    monkeypatch.setattr(provenance, "read_scene_rows", refuse)
    eval_dir = (Path(run.run_dir) / "eval").resolve()
    original = pq.read_table

    def guarded(source, *args, **kwargs):
        if isinstance(source, (str, os.PathLike)) and Path(source).resolve().is_relative_to(
                eval_dir):
            raise AssertionError(f"{source} was read as a table")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", guarded)
    outcome = run_figures(run, figures_root=tmp_path / "figures")
    final = Path(outcome["published"])
    assert sorted(p.name for p in final.iterdir()) == sorted(FIGURE_FILES + ("MANIFEST.json",))


def test_a_figure_or_table_changed_after_publishing_is_caught(published, tmp_path):
    run, _ = published
    figures_copy = tmp_path / "figures" / LEVEL
    shutil.copytree(Path(run.run_dir) / "figures" / LEVEL, figures_copy)
    tables_copy = copy_tables(run, tmp_path)
    check_published_figures(figures_copy, tables_copy)

    # The tables rebuilt with a different headline value, consistently
    # published: the headline figures no longer show the current tables.
    rewrite_table(tables_copy, PRIMARY, row_change=lambda row: {
        **row, "delta_learn_pp": row["delta_learn_pp"] + 0.1})
    read_published_tables(tables_copy, level=LEVEL)
    with pytest.raises(ProvenanceError) as error:
        check_published_figures(figures_copy, tables_copy)
    assert "tables" in error.value.fields
    assert FIGURE1 in str(error.value) and FIGURE2 in str(error.value)

    shutil.rmtree(tables_copy)
    tables_copy = copy_tables(run, tmp_path)
    (figures_copy / FIGURE2).write_bytes(b"changed")
    with pytest.raises(ProvenanceError) as error:
        check_published_figures(figures_copy, tables_copy)
    assert "sha256" in error.value.fields and FIGURE2 in str(error.value)
