"""The normative analysis constants, read from configs/analysis.yaml.

PROTOCOL.md's preamble makes that file part of the protocol: "All numeric
constants this protocol references ... live in configs/analysis.yaml at the
pre-registration commit. That file is part of this protocol, and changing any
value in it is an amendment."

The audit's first blocker was that every one of those constants lived instead as
a module constant or a function default, so an audit could be run against a
target that drifts without trace. This module is the single reader. Nothing in
src/ may carry one of these values as a literal; a test enforces that.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

CONFIG_VERSION = 1

# The one place the default location is written. Tests and entrypoints resolve
# it from here so a run cannot silently pick up a different file.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "analysis.yaml"


@dataclasses.dataclass(frozen=True)
class AnalysisConfig:
    """Every constant PROTOCOL names as config-resident, loaded as one object."""

    rotation_bin_edges_deg: tuple[float, ...]
    parallax_bin_edges: tuple[float, ...]
    translation_parallax_design_floor: float
    stratum_parallax_edges: tuple[float, ...]
    stratum_rotation_edges_deg: tuple[float, ...]
    bin_right_closed: bool
    zero_parallax_tol: float
    min_expected_median_depth_m: float
    zero_rotation_tol_deg: float
    assert_translation_parallax_floor: bool

    support_min_scenes: int
    support_min_camera_pairs: int
    bootstrap_resamples: int
    bootstrap_confidence: float
    bootstrap_seed: int

    covisible_relative_depth_tol: float
    min_covisible_fraction: float
    points_per_pair: int | None  # None means exhaustive; see the config comment
    max_pairs_per_stratum: int
    path_agreement_tolerance: float
    ledger_recon_tol: float
    ledger_closure_tol: float

    # Phase 4 execution constants, Amendment A3. The confidence rule decides
    # Phase 4 validity and enters the Phase 4 measurement identity computed in
    # lot.phase4; the gate tolerances are reporting-side per PROTOCOL 3.12.
    # None of them join MEASUREMENT_FIELDS, so the Phase 3 measurement digest
    # and the corrected Phase 3 parquet's readability are untouched.
    vggt_confidence_threshold: float | None
    rotation_gate_score_tol: float
    rotation_gate_forced_tol: float
    rotation_gate_coord_tol_px: float

    rotation_position_bound_m: float
    translation_rotation_bound_deg: float

    epsilon_margin: float
    depth_boundary_dilation_px: int
    depth_boundary_gradient_threshold: float
    texture_gradient_threshold: float
    depth_convention_slope_threshold: float

    depth_convention_flat_tol: float
    depth_convention_margin: float
    depth_convention_center_crop: float
    frame_min_valid_fraction: float
    frame_min_clearance_m: float
    frame_near_depth_floor_m: float

    def as_dict(self) -> dict[str, Any]:
        """Every normative value, as plain data for a run record."""
        out: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            out[field.name] = list(value) if isinstance(value, tuple) else value
        return out

    # Which values decide what was measured, and which decide only how it is
    # reported. The split exists because the two need opposite treatment. A
    # report must be refused if it is built under a different measurement
    # config, since the rows would then describe a different experiment. It must
    # be permitted under a different reporting config, because PROTOCOL 3.4 has
    # the support thresholds set from realized counts after the run, which is a
    # reporting edit by construction. Collapsing the two would either forbid the
    # documented workflow or bind nothing.
    # Values PROTOCOL names that no implemented phase consumes. A field outside
    # this tuple that nothing reads fails the enforcement test, and a field
    # inside it that something does read fails it too, so the exemption cannot
    # park a live constant. Phase 4 consumed the last five occupants
    # (epsilon_margin and the boundary, texture, and depth-convention
    # thresholds), so the tuple is empty until a later phase declares ahead.
    RESERVED_FOR_LATER_PHASES = ()

    MEASUREMENT_FIELDS = (
        "assert_translation_parallax_floor",
        # The translation program's own floor, which the evaluation-time
        # assertion reads. It used to read the first reporting edge, which is
        # excluded below precisely because it may be widened from counts, so a
        # permitted reporting edit silently moved an evaluation gate.
        "translation_parallax_design_floor",
        # The sampling design. These decide which pairs were drawn, so a run is
        # not comparable across a change to them. The reporting bin edges are
        # deliberately not here: PROTOCOL 3.4 permits those to be widened once
        # from counts after the run, and widening a label cannot retroactively
        # change a sample.
        "stratum_parallax_edges",
        "stratum_rotation_edges_deg",
        # Which side of an edge a value belongs to. This governs the stratum
        # labels above as well as the reporting bins, so flipping it moves
        # which pairs a capped stratum draws; and unlike the edges, PROTOCOL
        # 3.4 froze it outright ("closed on the right ... frozen here"), so no
        # post-run edit to it is permitted anyway.
        "bin_right_closed",
        "covisible_relative_depth_tol",
        "min_covisible_fraction",
        "points_per_pair",
        "max_pairs_per_stratum",
        "rotation_position_bound_m",
        "translation_rotation_bound_deg",
        "min_expected_median_depth_m",
        "frame_min_valid_fraction",
        "frame_min_clearance_m",
        "frame_near_depth_floor_m",
        "depth_boundary_dilation_px",
        "depth_boundary_gradient_threshold",
        "texture_gradient_threshold",
        "depth_convention_slope_threshold",
        "depth_convention_flat_tol",
        "depth_convention_margin",
        "depth_convention_center_crop",
        # Binning tolerances sit here, not in reporting: the translation
        # parallax-floor assertion runs during evaluation and reads them, so a
        # run was accepted or rejected under these values.
        "zero_parallax_tol",
        "zero_rotation_tol_deg",
    )

    def measurement_digest(self) -> str:
        """Content hash of the values that decide what the rows contain."""
        return self._digest_of(self.MEASUREMENT_FIELDS)

    def reporting_digest(self) -> str:
        """Content hash of the values that decide only how the rows are read."""
        rest = tuple(
            field.name
            for field in dataclasses.fields(self)
            if field.name not in self.MEASUREMENT_FIELDS
        )
        return self._digest_of(rest)

    def _digest_of(self, names: tuple[str, ...]) -> str:
        import hashlib
        import json

        values = self.as_dict()
        payload = json.dumps(
            {name: values[name] for name in sorted(names)}, sort_keys=True, separators=(",", ":")
        )
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()

    def digest(self) -> str:
        """Content hash of the whole config.

        A run records this rather than the config's path. A path binds nothing:
        two runs at one commit, with different uncommitted edits to the same
        file, would agree on it. The digest changes when any normative value
        changes, which is what an amendment is.
        """
        import hashlib
        import json

        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()

    def __post_init__(self) -> None:
        # A rotation frame's position is a simulator read-back, so it carries
        # noise up to rotation_position_bound_m. That noise becomes a parallax
        # of at most bound / depth, and if that exceeds zero_parallax_tol the
        # pairs of the one regime defined by having no baseline are binned as
        # though they had one. The two constants are only meaningful together,
        # so they are checked together rather than left to agree by luck.
        worst = self.rotation_position_bound_m / self.min_expected_median_depth_m
        if worst > self.zero_parallax_tol:
            raise ValueError(
                f"rotation_position_bound_m {self.rotation_position_bound_m:g} over "
                f"min_expected_median_depth_m {self.min_expected_median_depth_m:g} "
                f"gives a parallax of {worst:g}, above zero_parallax_tol "
                f"{self.zero_parallax_tol:g}: an in-place rotation pair at the "
                "bound would be binned as if it had a baseline"
            )
        # The mirror of the above, on the other regime's other axis. A
        # translation pair may differ in orientation by up to the manifest
        # bound, and if that exceeds the zero-rotation tolerance the pair
        # leaves the zero-rotation bin that PROTOCOL 3.3 puts the whole
        # translation regime in. The two constants describe the same physical
        # slack seen from two sides, so neither can be set alone.
        if self.translation_rotation_bound_deg > self.zero_rotation_tol_deg:
            raise ValueError(
                f"translation_rotation_bound_deg {self.translation_rotation_bound_deg:g} "
                f"exceeds zero_rotation_tol_deg {self.zero_rotation_tol_deg:g}: a "
                "translation pair at the manifest bound would pass validation and "
                "then be binned outside the zero-rotation bin its regime defines"
            )

    def rotation_edges(self) -> tuple[float, ...]:
        """Rotation bin edges with the mandatory open-ended overflow appended."""
        return tuple(self.rotation_bin_edges_deg) + (float("inf"),)

    def parallax_edges(self) -> tuple[float, ...]:
        """Parallax bin edges with the mandatory open-ended overflow appended."""
        return tuple(self.parallax_bin_edges) + (float("inf"),)

    def stratum_parallax_edges_full(self) -> tuple[float, ...]:
        """Sampling-design parallax edges, overflow appended."""
        return tuple(self.stratum_parallax_edges) + (float("inf"),)

    def stratum_rotation_edges_full(self) -> tuple[float, ...]:
        """Sampling-design rotation edges, overflow appended."""
        return tuple(self.stratum_rotation_edges_deg) + (float("inf"),)


def load_analysis_config(path: Path | None = None) -> AnalysisConfig:
    """Load the normative config. Unknown or missing keys are an error."""
    import yaml

    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    version = raw.pop("analysis_config_version", None)
    if version != CONFIG_VERSION:
        raise ValueError(f"analysis config version {version}, expected {CONFIG_VERSION}")
    fields = {f.name for f in dataclasses.fields(AnalysisConfig)}
    unknown = sorted(set(raw) - fields)
    if unknown:
        raise ValueError(f"unknown analysis config keys: {unknown}")
    missing = sorted(fields - set(raw))
    if missing:
        raise ValueError(f"analysis config missing keys: {missing}")
    # vggt_confidence_threshold: null is the frozen no-gating rule and loads
    # as None; a number is a real threshold. Nothing to normalize, but the key
    # must exist, which the missing-keys check above already enforces.
    # "all" and null both mean exhaustive. Stored as None so a caller cannot
    # mistake a sentinel integer for a real cap.
    if isinstance(raw["points_per_pair"], str):
        if raw["points_per_pair"].strip().lower() != "all":
            raise ValueError(
                f"points_per_pair must be an integer or \"all\", got "
                f"{raw['points_per_pair']!r}"
            )
        raw["points_per_pair"] = None
    for key in (
        "rotation_bin_edges_deg",
        "parallax_bin_edges",
        "stratum_parallax_edges",
        "stratum_rotation_edges_deg",
    ):
        raw[key] = tuple(float(v) for v in raw[key])
    return AnalysisConfig(**raw)
