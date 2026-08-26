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
    bin_right_closed: bool
    zero_parallax_tol: float
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

    rotation_position_bound_m: float

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

    def rotation_edges(self) -> tuple[float, ...]:
        """Rotation bin edges with the mandatory open-ended overflow appended."""
        return tuple(self.rotation_bin_edges_deg) + (float("inf"),)

    def parallax_edges(self) -> tuple[float, ...]:
        """Parallax bin edges with the mandatory open-ended overflow appended."""
        return tuple(self.parallax_bin_edges) + (float("inf"),)


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
    # "all" and null both mean exhaustive. Stored as None so a caller cannot
    # mistake a sentinel integer for a real cap.
    if isinstance(raw["points_per_pair"], str):
        if raw["points_per_pair"].strip().lower() != "all":
            raise ValueError(
                f"points_per_pair must be an integer or \"all\", got "
                f"{raw['points_per_pair']!r}"
            )
        raw["points_per_pair"] = None
    raw["rotation_bin_edges_deg"] = tuple(float(v) for v in raw["rotation_bin_edges_deg"])
    raw["parallax_bin_edges"] = tuple(float(v) for v in raw["parallax_bin_edges"])
    return AnalysisConfig(**raw)
