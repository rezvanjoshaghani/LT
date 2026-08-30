"""Phase 4, rung 1: the estimated-geometry tax. PROTOCOL 4.1 through 4.9.

Replace Oracle-Transport's ground-truth depth with VGGT estimated depth under
the alignment ladder of PROTOCOL 4.3, keep everything else frozen from Phase 3,
and measure the additional transport loss. No training, no learned predictor.

What stays frozen from Phase 3, byte for byte: the pair sample and its seed,
the correspondence universe and every sample_id, the ground-truth visibility
buckets, the transport and rasterization defaults, the metrics, the floors,
the mean vector, and the bin config. Estimated depth changes only where a
correspondence lands and which samples remain transportable. PROTOCOL 4.9.

Three validity notions, kept distinct and never sharing a mask name:

- transport pre-validity (5a) and post-alignment transport validity (5c):
  finite, positive estimated depth, plus the frozen confidence rule. The rule
  is frozen as vggt_confidence_threshold null, meaning no confidence gating
  (Amendment A3). Positive scaling cannot change this set, so it is identical
  across the no-alignment, scene-scale, and image-scale levels; that identity
  is asserted per pair and a difference stops the run.
- calibration-population validity (5b): transport pre-validity plus finite,
  positive ground-truth depth. Used only to estimate alignment parameters and
  never narrows the transport population.
- scoring validity (5d): the prediction must also land. On the per-point path
  the estimated warp needs positive depth in the context camera and must fall
  inside the context sampling box; on the splat path the cell needs estimated
  coverage plus the frozen ground-truth co-visibility rule. Landing depends on
  the level when translation is nonzero, so the scored set is level-dependent
  and is reported as coverage, never asserted equal across levels.

Alignment application, Amendment A4: each level yields one transform per
pair. Level 1 is the leave-target-out scene scalar; Level 2 and the affine
sensitivity are estimated from the pair's context image alone. The transform
applies to whichever VGGT map serves as a transport input on a path: the
context frame's map on the splat path, the target frame's map on the
per-point path. The target frame's ground truth never enters any estimator,
and a per-record assertion enforces it (PROTOCOL 4.3).

The pure-rotation gates of PROTOCOL 4.5 run inline for every rotation pair at
every level, against the tolerances Amendment A3 added to the analysis
config. A breach raises Phase4GateError with the evidence the execution plan
demands and stops the run. The forced-collision machinery is an isolated copy
of the transport plan that cannot change ordinary transport; its
forcing-disabled form is asserted equal to lot.transport.transport_plan once
per scene and permanently in the test suite.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .analysis_config import DEFAULT_CONFIG_PATH, AnalysisConfig, load_analysis_config
from .correspondence import _in_box, _sampling_box
from .datasets import (
    assert_translation_parallax_floor,
    load_scene_pairs,
    scene_split,
    subsample_by_stratum,
)
from .encoders import (
    DEPTH_NAME,
    PATCH_SIZE,
    archive_digest,
    cache_dir,
    load_cache_meta,
    sample_features_bilinear,
    sample_map_bilinear,
)
from .evaluate import (
    MEAN_FEATURE,
    NO_WARP_COPY,
    ORACLE_TRANSPORT,
    PER_POINT,
    SPLAT_POOL,
    _SceneCache,
    agreement_metrics,
    git_commit,
    load_or_build_mean_vector,
    pack_mask,
    pair_geometry,
    read_run_metadata,
    vector_digest,
    write_rows,
)
from .geometry import invert_se3, pixel_grid, project, relative_pose, transform_points, unproject
from .render_replica import MANIFEST_NAME, load_manifest, validate_manifest
from .transport import TIE_RELATIVE_EPS, apply_transport_plan, transport_plan
from .visibility import fraction_per_patch

PHASE4_VERSION = 1

# Alignment levels in ladder order. Names appear verbatim in rows, tables,
# and figures, per the method-name rule in CLAUDE.md.
VGGT_NO_ALIGN = "VGGT-NoAlign"
VGGT_SCENE_SCALE = "VGGT-SceneScale"
VGGT_IMAGE_SCALE = "VGGT-ImageScale"
VGGT_IMAGE_AFFINE = "VGGT-ImageAffine"

LEVELS = (
    ("none", VGGT_NO_ALIGN),
    ("scene", VGGT_SCENE_SCALE),
    ("image", VGGT_IMAGE_SCALE),
    ("affine", VGGT_IMAGE_AFFINE),
)
MULTIPLICATIVE_LEVELS = ("none", "scene", "image")

GT_LEVEL = "gt"
POPULATION_FULL = "full"
POPULATION_MATCHED = "matched"


class Phase4GateError(RuntimeError):
    """A Phase 4 correctness gate failed. The run stops here."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Phase4Config:
    """One Phase 4 run. Numeric constants come from the analysis config."""

    experiment_name: str
    renders_root: Path
    cache_root: Path
    output_root: Path
    scenes: list[str]
    feature_encoder: str = "dinov2_vitb14"
    depth_encoder: str = "vggt_1b"
    # The Phase 3 run directory whose provenance-validated mean vector this
    # run reuses. PROTOCOL 4.1 freezes the floors and the centering statistic.
    mean_vector_dir: Path = Path("outputs/experiment_zero")
    # The corrected Phase 3 evaluation whose pair population Phase 4 inherits
    # under PROTOCOL 4.1 and must be able to prove it inherited.
    phase3_eval_dir: Path = Path("outputs/experiment_zero/eval")
    seed: int = 0
    analysis_config: Path = DEFAULT_CONFIG_PATH
    geometry_dtype: str = "float32"

    def __post_init__(self) -> None:
        self.renders_root = Path(self.renders_root)
        self.cache_root = Path(self.cache_root)
        self.output_root = Path(self.output_root)
        self.mean_vector_dir = Path(self.mean_vector_dir)
        self.phase3_eval_dir = Path(self.phase3_eval_dir)
        self.analysis_config = Path(self.analysis_config)
        if not self.scenes:
            raise ValueError("config lists no scenes")
        if self.geometry_dtype not in ("float32", "float64"):
            raise ValueError("geometry_dtype must be float32 or float64")

    @property
    def eval_dir(self) -> Path:
        return self.output_root / self.experiment_name / "eval"

    @property
    def evidence_dir(self) -> Path:
        return self.output_root / self.experiment_name / "evidence"

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float32 if self.geometry_dtype == "float32" else torch.float64


def load_phase4_config(path: Path) -> Phase4Config:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    allowed = {f.name for f in dataclasses.fields(Phase4Config)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return Phase4Config(**raw)


def phase4_measurement_digest(analysis: AnalysisConfig) -> str:
    """Identity of the values that decide what Phase 4 rows contain.

    The Phase 3 measurement digest is left untouched so the corrected Phase 3
    parquet stays readable. Phase 4 extends that identity with the one new
    value that decides Phase 4 validity, the frozen confidence rule.
    """
    payload = json.dumps(
        {
            "phase3_measurement": analysis.measurement_digest(),
            "vggt_confidence_threshold": analysis.vggt_confidence_threshold,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Estimated depth: cache, resampling, convention, validity
# ---------------------------------------------------------------------------

def load_depth_archive(cache_root: Path, encoder: str, scene: str) -> dict[str, Any]:
    """Open one scene's estimated-depth cache, verified against its digest.

    Returns {"depth": {frame_id: fp32 [H, W]}, "conf": {frame_id: fp32 | None},
    "meta": cache meta}. The digest is checked over the bytes actually read,
    the same discipline the feature cache gets in evaluation.
    """
    meta = load_cache_meta(cache_root, encoder, scene)
    if not meta.get("has_depth"):
        raise ValueError(f"{encoder} / {scene}: cache exports no depth")
    path = cache_dir(cache_root, encoder, scene) / DEPTH_NAME
    recorded = meta.get("depth_digest")
    found = archive_digest(path)
    if recorded != found:
        raise ValueError(
            f"{encoder} / {scene}: depth digest {found} does not match the "
            f"{recorded} recorded when the cache was written"
        )
    depth: dict[str, np.ndarray] = {}
    conf: dict[str, np.ndarray | None] = {}
    with np.load(path) as archive:
        for name in archive.files:
            if name.endswith("__conf"):
                continue
            depth[name] = archive[name].astype(np.float32)
            conf_name = f"{name}__conf"
            conf[name] = (
                archive[conf_name].astype(np.float32) if conf_name in archive.files else None
            )
    return {"depth": depth, "conf": conf, "meta": meta}


# ---------------------------------------------------------------------------
# Depth-convention authority, PROTOCOL 4.1
# ---------------------------------------------------------------------------

# PROTOCOL 4.1 gives the model's own source primary authority over the depth
# head's convention and demotes the secant regression to a consistency check
# when the source is definitive. A depth convention is a property of the
# network's output semantics, so exactly one decision applies to every frame a
# pinned checkpoint produces; Amendment A6 makes that global application
# explicit after a per-scene reading was found in this implementation.
DECISIVE_FUNCTION = "depth_to_cam_coords_points"
DECISIVE_MODULE = "utils/geometry.py"
# Anything that would rescale depth along the ray before assigning z. Their
# absence inside the decisive function is part of what makes planar z
# unambiguous rather than merely plausible.
RAY_MARKERS = ("sec(", "np.sqrt", "torch.sqrt", "norm(", "normalize", "/ cos", "cos(")


def vggt_package_roots() -> list[Path]:
    """Directories of the installed vggt package.

    VGGT installs as a namespace package, so __file__ is None and __path__ is
    the only way to the source. Reading __file__ alone raised on the cluster.
    """
    import vggt

    roots = [Path(p) for p in getattr(vggt, "__path__", [])]
    if getattr(vggt, "__file__", None):
        roots.append(Path(vggt.__file__).parent)
    return sorted({root for root in roots if root.is_dir()})


def vggt_source_revision() -> dict[str, Any]:
    """The immutable revision of the installed vggt source, if it has one.

    PROTOCOL 3.12's pin rule applies to the authority as much as to the
    weights: a floating branch name is not a pin. The commit comes from the
    installer's own record of what it built, which is the same value the
    feature cache recorded as code_revision when it produced the depth.
    """
    from importlib import metadata

    out: dict[str, Any] = {"distribution": None, "version": None, "commit": None,
                           "url": None, "pinned": False}
    try:
        distribution = metadata.distribution("vggt")
    except metadata.PackageNotFoundError:
        return out
    out["distribution"] = "vggt"
    try:
        out["version"] = metadata.version("vggt")
    except metadata.PackageNotFoundError:
        pass
    try:
        raw = distribution.read_text("direct_url.json")
    except OSError:
        raw = None
    if raw:
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {}
        out["url"] = payload.get("url")
        info = payload.get("vcs_info") or {}
        commit = info.get("commit_id")
        if isinstance(commit, str) and len(commit) == 40:
            out["commit"] = commit
            out["pinned"] = True
    return out


def extract_function(source: str, name: str) -> tuple[int, list[str]] | None:
    """The source lines of one top-level function, with its starting line."""
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"def {name}"):
            end = index + 1
            while end < len(lines) and not lines[end].startswith(("def ", "class ")):
                end += 1
            while end > index + 1 and not lines[end - 1].strip():
                end -= 1
            return index + 1, lines[index:end]
    return None


def source_authority() -> dict[str, Any]:
    """Establish the depth convention from the installed VGGT source.

    The decisive evidence is how VGGT itself turns its predicted depth into
    camera-frame points. depth_to_cam_coords_points is the function that
    unproject_depth_map_to_point_map calls to do exactly that. A planar
    camera-z head assigns z from the depth map directly and scales x and y by
    depth over focal length; a ray-distance head would have to rescale along
    the ray first. This checks the structure rather than trusting a claim, and
    returns unambiguous False when it cannot read the signature it expects.
    """
    evidence: dict[str, Any] = {
        "verdict": "ambiguous",
        "unambiguous": False,
        "function": DECISIVE_FUNCTION,
        "module": None,
        "first_line": None,
        "lines": [],
        "checks": {},
        "revision": vggt_source_revision(),
        "roots": [],
        "reason": None,
    }
    try:
        roots = vggt_package_roots()
    except ImportError:
        evidence["reason"] = "vggt is not installed in this environment"
        return evidence
    evidence["roots"] = [str(root) for root in roots]
    for root in roots:
        path = root / DECISIVE_MODULE
        if not path.is_file():
            continue
        extracted = extract_function(path.read_text(encoding="utf-8", errors="replace"),
                                     DECISIVE_FUNCTION)
        if extracted is None:
            continue
        first_line, body = extracted
        evidence["module"] = str(path)
        evidence["first_line"] = first_line
        evidence["lines"] = body
        code = [
            line.split("#", 1)[0].strip()
            for line in body
            if line.split("#", 1)[0].strip()
        ]
        joined = " ".join(code)
        assigns_z_directly = any(
            line.replace(" ", "").startswith("z_cam=") and "depth_map" in line
            and not any(marker.replace(" ", "") in line.replace(" ", "")
                        for marker in RAY_MARKERS)
            for line in code
        )
        scales_xy_by_depth = sum(
            1 for line in code
            if line.replace(" ", "").startswith(("x_cam=", "y_cam="))
            and "*depth_map/" in line.replace(" ", "")
        ) == 2
        no_ray_rescale = not any(marker in joined for marker in RAY_MARKERS)
        evidence["checks"] = {
            "assigns_z_from_depth_directly": bool(assigns_z_directly),
            "scales_x_and_y_by_depth_over_focal": bool(scales_xy_by_depth),
            "no_ray_rescaling_in_function": bool(no_ray_rescale),
        }
        if assigns_z_directly and scales_xy_by_depth and no_ray_rescale:
            evidence["verdict"] = "planar_z"
            evidence["unambiguous"] = True
        else:
            evidence["reason"] = (
                "the decisive function does not carry the planar-z signature; "
                "read the recorded lines before deciding"
            )
        return evidence
    evidence["reason"] = (
        f"{DECISIVE_MODULE}:{DECISIVE_FUNCTION} was not found in the installed vggt"
    )
    return evidence


def build_convention_record(
    diagnostic: dict[str, Any], authority: dict[str, Any], depth_meta: dict[str, Any]
) -> dict[str, Any]:
    """One convention decision for the whole run, with its authority.

    Amendment A6: where authoritative model source establishes the convention
    under PROTOCOL 4.1, that decision applies globally to every frame the
    pinned checkpoint produced, and no per-scene diagnostic verdict may
    control conversion. The secant results travel inside the record as
    evidence of the check having run, and of what it found.
    """
    verdict = authority.get("verdict")
    if not (authority.get("unambiguous") and verdict in ("planar_z", "ray_distance")):
        raise Phase4GateError(
            "the pinned VGGT source does not establish the depth convention "
            f"unambiguously: {authority.get('reason')}. PROTOCOL 4.1 makes the "
            "source the primary authority; without it the convention is "
            "unresolved and Phase 4 does not run."
        )
    # The source cited as authority must be the source that produced the
    # cached depth. The caching job pinned its inference revision as a full
    # commit, so a missing or differing revision here means the semantics
    # being cited were read from something other than what ran.
    cache_revision = depth_meta.get("code_revision")
    if isinstance(cache_revision, str) and len(cache_revision) == 40:
        source_revision = authority["revision"].get("commit")
        if source_revision != cache_revision:
            raise Phase4GateError(
                "the VGGT source cited as authority is not the source that "
                f"produced the depth cache: installed {source_revision!r} "
                f"against the cache's recorded inference revision "
                f"{cache_revision!r}. A convention read from other code is "
                "not evidence about these predictions."
            )
    scenes = diagnostic.get("scenes", {})
    verdicts = {scene: value.get("verdict") for scene, value in scenes.items()}
    return {
        "depth_convention": verdict,
        "depth_convention_authority": "source",
        "depth_convention_source_commit": authority["revision"].get("commit"),
        "depth_convention_source_pinned": bool(authority["revision"].get("pinned")),
        "depth_convention_source_module": authority.get("module"),
        "depth_convention_source_function": authority.get("function"),
        "depth_convention_source_first_line": authority.get("first_line"),
        "depth_convention_conversion_applied": verdict == "ray_distance",
        "secant_regression_role": "diagnostic_only",
        "checkpoint_weights_fingerprint": depth_meta.get("weights_fingerprint"),
        "checkpoint_weights_revision": depth_meta.get("weights_revision"),
        "checkpoint_code_revision": depth_meta.get("code_revision"),
        "authority_evidence": authority,
        "secant_diagnostic": diagnostic,
        "secant_diagnostic_verdicts": verdicts,
        "secant_diagnostic_disagrees_with_authority": sorted(
            scene for scene, value in verdicts.items() if value != verdict
        ),
    }


def run_convention(record: dict[str, Any], depth_meta: dict[str, Any]) -> str:
    """The single convention every scene of a run reads.

    Never keyed by scene, frame, camera program, angle, or diagnostic verdict.
    A record without a global decision is refused rather than defaulted, and
    the record is re-bound to the cache it is about to govern every time it is
    consumed: a record is a file on disk, so a stale or hand-edited one could
    otherwise dictate the conversion for depth it was never written about.
    """
    verdict = record.get("depth_convention")
    if verdict not in ("planar_z", "ray_distance"):
        raise Phase4GateError(
            "the convention record carries no global depth_convention; a run "
            "must establish exactly one convention for the pinned checkpoint"
        )
    if record.get("depth_convention_authority") != "source":
        raise Phase4GateError(
            "the convention was not established from source authority; "
            "PROTOCOL 4.1 and Amendment A6 admit no other basis for the "
            "global decision"
        )
    for field, cache_field in (
        ("checkpoint_code_revision", "code_revision"),
        ("checkpoint_weights_fingerprint", "weights_fingerprint"),
        ("checkpoint_weights_revision", "weights_revision"),
    ):
        recorded, current = record.get(field), depth_meta.get(cache_field)
        if recorded != current:
            raise Phase4GateError(
                f"the convention record was written for {field}={recorded!r} "
                f"but the depth cache being evaluated carries {current!r}. A "
                "convention established about other predictions does not "
                "govern these."
            )
    source_commit = record.get("depth_convention_source_commit")
    if source_commit != depth_meta.get("code_revision"):
        raise Phase4GateError(
            f"the convention's source authority {source_commit!r} is not the "
            f"inference revision {depth_meta.get('code_revision')!r} that "
            "produced this cache"
        )
    return verdict


def manifest_digest(scene_root: Path) -> str:
    """Content hash of a scene's manifest.

    The manifests are untracked, so a commit hash says nothing about them. A
    run that regenerated its pair sample from edited manifests would be
    self-consistent and measuring a different population; binding the digest
    into every run record is what makes that detectable afterwards.
    """
    return hashlib.sha256(
        (Path(scene_root) / MANIFEST_NAME).read_bytes()
    ).hexdigest()


# How far a Phase 4 recomputation of a Phase 3 score may sit from the value
# Phase 3 recorded before the inheritance is refused. The two are the same
# code over the same bytes in the same environment, so the honest expectation
# is bitwise equality; the bound absorbs dtype accumulation drift across
# library builds and nothing else. The path-agreement ledger reconstructed the
# recorded scores from the caches to 2e-7, an order of magnitude inside this.
PHASE3_SCORE_RECON_TOL = 1e-5


def phase3_scene_reference(
    eval_dir: Path, scene: str, feature_encoder: str
) -> dict[str, Any]:
    """What the corrected Phase 3 run recorded for a scene, as the referee.

    PROTOCOL 4.1 keeps the pairs, masks, sample identities, and ceilings
    identical to Phase 3. Phase 4 regenerates all of them from the manifests
    with the frozen code, which is the same computation but not the same
    evidence: a changed pose, depth map, or frame filter that preserves the
    pair names would still pass a name-level check while measuring a
    different population. So the reference carries, per (pair, path), the
    persisted validity mask, its count, and the recorded Oracle and No-Warp
    scores, and evaluation reconciles its own recomputation against them row
    by row. It also carries the run record's provenance so the caches and
    measurement identity can be compared, not assumed.
    """
    import pyarrow.parquet as pq

    from .evaluate import read_run_metadata

    path = Path(eval_dir) / f"{scene}.parquet"
    if not path.is_file():
        raise Phase4GateError(
            f"{path} does not exist; Phase 4 inherits the Phase 3 pair "
            "population and cannot show it is using it without the run"
        )
    meta = read_run_metadata(path)
    if meta is None:
        raise Phase4GateError(f"{path} carries no run record")
    provenance = (meta.get("cache_provenance") or {}).get(feature_encoder) or {}
    rows = pq.read_table(
        path,
        columns=["context_frame_id", "target_frame_id", "encoder", "path",
                 "variant", "n", "sample_mask", "cosine_mean"],
    ).to_pylist()
    reference: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["encoder"] != feature_encoder:
            continue
        pair = (row["context_frame_id"], row["target_frame_id"])
        slot = reference.setdefault(pair, {}).setdefault(
            row["path"], {"mask": bytes(row["sample_mask"]), "n": row["n"]}
        )
        if row["variant"] in (ORACLE_TRANSPORT, NO_WARP_COPY):
            slot[row["variant"]] = row["cosine_mean"]
    return {
        "pairs": reference,
        "git_commit": meta.get("git_commit"),
        "features_digest": provenance.get("features_digest"),
        "measurement_digest": meta.get("analysis_measurement_digest"),
        "universe_size": meta.get("universe_size"),
        "eval_version": meta.get("eval_version"),
    }


def reconcile_pair_against_phase3(
    scene: str,
    pair: Any,
    geometry: Any,
    gt_rows: Sequence[dict[str, Any]],
    reference: dict[str, dict[str, Any]],
) -> float:
    """One pair's recomputed masks and ceilings against Phase 3's record.

    Masks are discrete predicate outcomes of the frozen float32 chain, so
    they must be bit-identical; the scores must reproduce within
    PHASE3_SCORE_RECON_TOL. Returns the largest score residual so the scene
    metadata can carry the worst case rather than a boolean.
    """
    where = f"{scene} {pair.context_frame_id} -> {pair.target_frame_id}"
    worst = 0.0
    for row in gt_rows:
        recorded = reference.get(row["path"])
        if recorded is None:
            raise Phase4GateError(
                f"{where}: Phase 3 recorded no {row['path']} rows for this "
                "pair, but Phase 4 scored it there"
            )
        if row["variant"] not in (ORACLE_TRANSPORT, NO_WARP_COPY):
            continue
        if bytes(row["sample_mask"]) != recorded["mask"] or row["n"] != recorded["n"]:
            raise Phase4GateError(
                f"{where} ({row['path']}): the recomputed validity mask is not "
                "the one Phase 3 persisted. The pair names match but the "
                "geometry, depth, or eligibility filters have moved, so this "
                "is a different population wearing the same identity."
            )
        residual = abs(row["cosine_mean"] - recorded[row["variant"]])
        worst = max(worst, residual)
        if residual > PHASE3_SCORE_RECON_TOL:
            raise Phase4GateError(
                f"{where} ({row['path']}, {row['variant']}): recomputed score "
                f"{row['cosine_mean']:.9f} against Phase 3's recorded "
                f"{recorded[row['variant']]:.9f}, residual {residual:.2e} over "
                f"{PHASE3_SCORE_RECON_TOL:g}. The ceilings Phase 4 inherits "
                "are not the ones Phase 3 measured."
            )
    return worst


def resample_depth_nearest(
    depth: np.ndarray, dst_hw: tuple[int, int]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bring estimated depth to render resolution by nearest neighbor.

    PROTOCOL 4.1 freezes nearest-neighbor resampling and forbids bilinear:
    interpolation across a depth discontinuity manufactures a depth that lies
    on no surface. The current caches store depth at render resolution
    already, so this normally records the asserted identity; the general rule
    is implemented so the assertion is a fact rather than an assumption.
    Aspect-ratio changes and crops are refused, because the intrinsics at
    transport resolution would no longer describe the pixels.
    """
    src_h, src_w = depth.shape
    dst_h, dst_w = dst_hw
    record = {
        "src_hw": [int(src_h), int(src_w)],
        "dst_hw": [int(dst_h), int(dst_w)],
        "crop": None,
        "method": "identity" if (src_h, src_w) == (dst_h, dst_w) else "nearest",
    }
    if (src_h, src_w) == (dst_h, dst_w):
        return depth, record
    if src_h * dst_w != src_w * dst_h:
        raise ValueError(
            f"resampling {depth.shape} to {dst_hw} changes the aspect ratio; "
            "the intrinsics at transport resolution would not describe the pixels"
        )
    rows = np.floor((np.arange(dst_h) + 0.5) * src_h / dst_h).astype(np.int64)
    cols = np.floor((np.arange(dst_w) + 0.5) * src_w / dst_w).astype(np.int64)
    return depth[rows][:, cols], record


def secant_map(K: Tensor, height: int, width: int) -> np.ndarray:
    """Per-pixel secant of the angle from the optical axis. [H, W] float64."""
    K64 = K.to(torch.float64)
    ray_x = (torch.arange(width, dtype=torch.float64) - K64[0, 2]) / K64[0, 0]
    ray_y = (torch.arange(height, dtype=torch.float64) - K64[1, 2]) / K64[1, 1]
    yy, xx = torch.meshgrid(ray_y, ray_x, indexing="ij")
    return torch.sqrt(1.0 + xx * xx + yy * yy).numpy()


def secant_regression(
    vggt_depth: np.ndarray, gt_depth: np.ndarray, K: Tensor, threshold: float
) -> dict[str, Any]:
    """PROTOCOL 4.1's deterministic depth-convention test for one frame.

    Regress the per-pixel ratio of estimated to ground-truth depth against
    the secant of the pixel's angle from the optical axis, over pixels where
    both are finite and positive. A planar map gives a flat ratio; a ray map
    gives ratio proportional to the secant. The classification compares the
    fitted slope against the frozen threshold times the fitted ratio at the
    optical axis. No frame or region is chosen by inspection.
    """
    height, width = gt_depth.shape
    sec = secant_map(K, height, width)
    valid = (
        np.isfinite(vggt_depth) & (vggt_depth > 0) & np.isfinite(gt_depth) & (gt_depth > 0)
    )
    if int(valid.sum()) < 16:
        return {"verdict": "unresolved", "n": int(valid.sum())}
    ratio = (vggt_depth[valid] / gt_depth[valid]).astype(np.float64)
    x = sec[valid]
    design = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(design, ratio, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    center_scale = intercept + slope
    if not center_scale > 0:
        return {
            "verdict": "unresolved",
            "n": int(valid.sum()),
            "intercept": intercept,
            "slope": slope,
        }
    verdict = "ray_distance" if slope > threshold * center_scale else "planar_z"
    return {
        "verdict": verdict,
        "n": int(valid.sum()),
        "intercept": intercept,
        "slope": slope,
        "slope_share_of_center_scale": slope / center_scale,
        "threshold": threshold,
    }


def transport_prevalid(
    depth: np.ndarray, conf: np.ndarray | None, analysis: AnalysisConfig
) -> np.ndarray:
    """Validity 5a and 5c: finite, positive, and the frozen confidence rule."""
    valid = np.isfinite(depth) & (depth > 0)
    threshold = analysis.vggt_confidence_threshold
    if threshold is not None:
        if conf is None:
            raise ValueError(
                "a confidence threshold is frozen but the cache carries no confidence"
            )
        valid &= conf >= threshold
    return valid


def calibration_valid(prevalid: np.ndarray, gt_depth: np.ndarray) -> np.ndarray:
    """Validity 5b: transport pre-validity plus valid ground truth."""
    return prevalid & np.isfinite(gt_depth) & (gt_depth > 0)


# ---------------------------------------------------------------------------
# Alignment ladder, PROTOCOL 4.3
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FrameCalibration:
    """Per-frame calibration facts, computed once per scene."""

    ratios: np.ndarray          # float64, gt over vggt on calibration-valid pixels
    n_calibration: int
    image_scale: float          # median of ratios, nan when empty
    affine_s: float
    affine_b: float
    affine_n: int
    affine_failed: bool


def frame_calibration(
    vggt_depth: np.ndarray, gt_depth: np.ndarray, prevalid: np.ndarray
) -> FrameCalibration:
    """Level 2 and affine estimators for one frame.

    The affine fit is ordinary unweighted least squares of ground truth on
    estimated depth: no confidence weighting, no robust loss, no clipping,
    s unconstrained during fitting. A nonpositive fitted s marks the frame's
    affine row failed; it is reported, never repaired.
    """
    calib = calibration_valid(prevalid, gt_depth)
    n = int(calib.sum())
    nan = float("nan")
    if n == 0:
        return FrameCalibration(np.zeros(0), 0, nan, nan, nan, 0, True)
    est = vggt_depth[calib].astype(np.float64)
    gt = gt_depth[calib].astype(np.float64)
    ratios = gt / est
    scale = float(np.median(ratios))
    if n < 2:
        return FrameCalibration(ratios, n, scale, nan, nan, n, True)
    sum_e, sum_g = float(est.sum()), float(gt.sum())
    sum_ee, sum_eg = float(est @ est), float(est @ gt)
    denominator = n * sum_ee - sum_e * sum_e
    if denominator <= 0:
        return FrameCalibration(ratios, n, scale, nan, nan, n, True)
    s = (n * sum_eg - sum_e * sum_g) / denominator
    b = (sum_g - s * sum_e) / n
    failed = not (math.isfinite(s) and s > 0 and math.isfinite(b))
    return FrameCalibration(ratios, n, scale, float(s), float(b), n, failed)


def scene_scale_leave_target_out(
    calibrations: dict[str, FrameCalibration], target_frame_id: str
) -> tuple[float, dict[str, int]]:
    """Level 1: the leave-target-out scene oracle scale.

    One multiplicative scalar per pair: the median of ground truth over
    estimated depth pooled over the calibration-valid pixels of every scene
    frame except the pair's target. The audit maps each contributing frame to
    its pixel count; the target is absent by construction and the caller
    asserts that per evaluated record.
    """
    parts = [
        c.ratios
        for fid, c in calibrations.items()
        if fid != target_frame_id and c.n_calibration
    ]
    audit = {
        fid: c.n_calibration
        for fid, c in calibrations.items()
        if fid != target_frame_id and c.n_calibration
    }
    if not parts:
        raise Phase4GateError(
            f"Level 1 calibration population is empty with {target_frame_id} excluded"
        )
    scale = float(np.median(np.concatenate(parts)))
    if not (math.isfinite(scale) and scale > 0):
        raise Phase4GateError(
            f"Level 1 scale {scale} is not finite and positive for target {target_frame_id}"
        )
    return scale, audit


def aligned_depth(
    level: str,
    vggt_depth: np.ndarray,
    scene_scale: float,
    frame_calib: FrameCalibration,
) -> np.ndarray | None:
    """Apply one level's transform to one estimated map, or None when failed."""
    if level == "none":
        return vggt_depth
    if level == "scene":
        return vggt_depth * np.float32(scene_scale)
    if level == "image":
        if not (math.isfinite(frame_calib.image_scale) and frame_calib.image_scale > 0):
            raise Phase4GateError(
                f"Level 2 image scale {frame_calib.image_scale} is not finite and positive"
            )
        return vggt_depth * np.float32(frame_calib.image_scale)
    if level == "affine":
        if frame_calib.affine_failed:
            return None
        return vggt_depth * np.float32(frame_calib.affine_s) + np.float32(frame_calib.affine_b)
    raise ValueError(f"unknown alignment level {level!r}")


# ---------------------------------------------------------------------------
# Forced-collision transport: an isolated diagnostic copy, PROTOCOL 4.5
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SplatStructure:
    """Oracle's complete discrete rasterization structure, frozen for A7.

    Amendment A7: under pure rotation the forced gate freezes all three
    discrete quantities from the Oracle-Transport arm on the common-valid
    source set: each source pixel's target-cell assignment, per-cell
    candidate membership, and the collision winner ordering. winner_keys
    encodes the winning (source pixel, target pixel) pairs as
    source * n_target_pixels + target, sorted; candidate_landing holds the
    Oracle landing pixel of every kept source, -1 elsewhere.
    """

    n_cells: int
    winner_keys: np.ndarray        # sorted int64
    candidate_landing: np.ndarray  # [H*W] int64, -1 where the source is not kept


@dataclasses.dataclass
class SplatPlanDetail:
    """A transport plan plus the internals the pure-rotation gate needs.

    winner_keys encodes each winning splat as source_pixel * n_target_pixels
    plus target_pixel, sorted, so a forced run can test membership with one
    searchsorted pass instead of a Python set. landing_pixel, landing_uv,
    and landing_margin_px expose this arm's own rasterization per source
    pixel, so landing-cell instability between two arms is measurable as a
    diagnostic (A7): the flip count, the continuous coordinate residual,
    and each flipped pixel's distance to the floor(u + 0.5) boundary.
    """

    weights: Tensor             # [n_target_patches, n_source_patches] float32
    coverage: Tensor            # [Hp_out, Wp_out] float32
    keep: Tensor                # [H*W] bool over source pixels
    winner_keys: np.ndarray     # sorted int64
    landing_pixel: np.ndarray   # [H*W] int64 own landing, -1 where not kept
    landing_uv: np.ndarray      # [H*W, 2] float32 continuous landing
    landing_margin_px: np.ndarray  # [H*W] float32 distance to the cell boundary


def splat_structure(detail: SplatPlanDetail) -> SplatStructure:
    """The A7 frozen structure of one arm, ready to impose on another."""
    return SplatStructure(
        n_cells=int(detail.coverage.numel()) * PATCH_SIZE * PATCH_SIZE,
        winner_keys=detail.winner_keys,
        candidate_landing=detail.landing_pixel,
    )


def landing_flip_diagnostics(
    reference: SplatPlanDetail,
    other: SplatPlanDetail,
    out_hw_px: tuple[int, int],
    patch_size: int = PATCH_SIZE,
) -> dict[str, Any]:
    """Landing-cell instability between two arms of one geometry (A7).

    Both arms must have been built on the same source restriction. Reports
    the discrete flips (same source pixel, different landing cell), the
    landing-flip fraction of the shared kept set, the maximum continuous
    coordinate residual, each flipped pixel's distance to the floor(u + 0.5)
    rasterization boundary, and how many target patch cells the flips touch.
    Reported, never gated: a real rasterization bug, a convention mismatch,
    or a resize error moves coordinates by half pixels and floods this
    count, while one-ulp float instability shows as isolated flips at
    vanishing boundary margins. The gate's own score comparison runs under
    the frozen structure and cannot see either; this is where they appear.
    """
    both = (reference.landing_pixel >= 0) & (other.landing_pixel >= 0)
    flipped = both & (reference.landing_pixel != other.landing_pixel)
    n_both = int(both.sum())
    coord = np.abs(reference.landing_uv[both] - other.landing_uv[both])
    out_width = out_hw_px[1]
    out_patches_w = out_width // patch_size

    def cells(lin_arr: np.ndarray) -> np.ndarray:
        return ((lin_arr // out_width) // patch_size) * out_patches_w + (
            (lin_arr % out_width) // patch_size
        )

    touched = np.union1d(
        cells(reference.landing_pixel[flipped]), cells(other.landing_pixel[flipped])
    )
    margins = np.minimum(
        reference.landing_margin_px[flipped], other.landing_margin_px[flipped]
    )
    return {
        "landing_flip_count": int(flipped.sum()),
        "landing_flip_fraction": float(flipped.sum() / n_both) if n_both else 0.0,
        "landing_flip_cells": int(touched.size),
        "landing_coord_residual_max_px": float(coord.max()) if coord.size else 0.0,
        "landing_flip_margin_min_px": (
            float(margins.min()) if margins.size else float("nan")
        ),
        "flipped_cells": touched,
    }


def splat_plan_detail(
    depth_ctx_px: Tensor,
    K_ctx: Tensor,
    K_tgt: Tensor,
    T_tgt_from_ctx: Tensor,
    out_hw_px: tuple[int, int],
    patch_size: int = PATCH_SIZE,
    source_keep: Tensor | None = None,
    forced_winner_keys: np.ndarray | None = None,
    forced_structure: SplatStructure | None = None,
) -> SplatPlanDetail:
    """The transport plan with optional source restriction and forced winners.

    A documented copy of lot.transport.transport_plan, existing so the frozen
    default path is never modified. With every forcing argument None it
    reproduces the default plan exactly;
    assert_forcing_disabled_matches_default checks that at run time and the
    suite checks it permanently.

    Two forcing modes exist and are mutually exclusive:

    - forced_winner_keys replaces the z-buffer winner rule with membership in
      an externally supplied (source pixel, target pixel) key set, evaluated
      at this arm's own landings. A one-ulp landing flip makes a winner's key
      miss, so this mode conflates landing-cell instability with collision
      ordering; it is kept as the midpoint of the A7 tax decomposition, not
      as the gate.
    - forced_structure (Amendment A7) imposes the donor arm's complete
      discrete rasterization structure: winners land where the donor landed
      them, so cell assignment, candidate membership, and winner ordering
      are all frozen and the gated score comparison is a true invariant.
      This arm's own landings are still computed and returned, which is what
      makes landing flips measurable as a diagnostic.
    """
    if forced_winner_keys is not None and forced_structure is not None:
        raise ValueError("forced_winner_keys and forced_structure are exclusive")
    height, width = depth_ctx_px.shape
    patches_h, patches_w = height // patch_size, width // patch_size
    out_height, out_width = out_hw_px
    out_patches_h, out_patches_w = out_height // patch_size, out_width // patch_size
    dtype = depth_ctx_px.dtype if depth_ctx_px.dtype.is_floating_point else torch.float32

    uv = pixel_grid(height, width, dtype=dtype)
    z = depth_ctx_px.to(dtype)
    points_ctx = unproject(uv, z, K_ctx.to(dtype))
    points_tgt = transform_points(T_tgt_from_ctx.to(dtype), points_ctx)
    uv_tgt, z_tgt = project(points_tgt, K_tgt.to(dtype))

    keep = (
        (z > 0)
        & torch.isfinite(z)
        & (z_tgt > 0)
        & torch.isfinite(z_tgt)
        & torch.isfinite(uv_tgt).all(dim=-1)
    )
    safe_uv = torch.where(keep[..., None], uv_tgt, torch.zeros_like(uv_tgt))
    iu = torch.floor(safe_uv[..., 0] + 0.5).long()
    iv = torch.floor(safe_uv[..., 1] + 0.5).long()
    keep &= (iu >= 0) & (iu < out_width) & (iv >= 0) & (iv < out_height)
    if source_keep is not None:
        keep &= source_keep.reshape(height, width)

    keep_flat = keep.reshape(-1)
    source_index = torch.arange(height * width)[keep_flat]
    lin = (iv * out_width + iu).reshape(-1)[keep_flat]
    z_keep = z_tgt.reshape(-1)[keep_flat]
    n_cells = out_height * out_width

    # This arm's own rasterization, exposed for the A7 landing diagnostics.
    landing_pixel = np.full(height * width, -1, dtype=np.int64)
    landing_pixel[keep_flat.numpy()] = lin.numpy()
    fu = (safe_uv[..., 0] + 0.5) - torch.floor(safe_uv[..., 0] + 0.5)
    fv = (safe_uv[..., 1] + 0.5) - torch.floor(safe_uv[..., 1] + 0.5)
    landing_margin = torch.minimum(
        torch.minimum(fu, 1 - fu), torch.minimum(fv, 1 - fv)
    ).reshape(-1).numpy().astype(np.float32)
    landing_margin[~keep_flat.numpy()] = np.float32(np.nan)

    if forced_structure is not None:
        if forced_structure.n_cells != n_cells:
            raise Phase4GateError(
                "A7 structure was built for a different target grid: "
                f"{forced_structure.n_cells} cells versus {n_cells}"
            )
        winner_src = (forced_structure.winner_keys // n_cells).astype(np.int64)
        if not np.all(keep_flat.numpy()[winner_src]):
            raise Phase4GateError(
                "A7 structure names a winner source this arm does not keep; "
                "the structure must be built on the common-valid set"
            )
        # The donor's assignment, membership, and ordering, verbatim: winners
        # land where the donor landed them. Nothing depth-dependent survives.
        lin_w = torch.from_numpy(
            (forced_structure.winner_keys % n_cells).astype(np.int64)
        )
        src_w = torch.from_numpy(winner_src)
    else:
        if forced_winner_keys is None:
            zbuffer = torch.full((n_cells,), torch.inf, dtype=dtype)
            zbuffer.scatter_reduce_(0, lin, z_keep, reduce="amin", include_self=True)
            winners = z_keep <= zbuffer[lin] * (1 + TIE_RELATIVE_EPS)
        else:
            keys = source_index.numpy().astype(np.int64) * np.int64(n_cells) + lin.numpy()
            position = np.searchsorted(forced_winner_keys, keys)
            position = np.clip(position, 0, len(forced_winner_keys) - 1)
            winners = torch.from_numpy(
                (len(forced_winner_keys) > 0) & (forced_winner_keys[position] == keys)
            )
        lin_w = lin[winners]
        src_w = source_index[winners]

    source_patch = (
        (torch.arange(height) // patch_size)[:, None] * patches_w
        + (torch.arange(width) // patch_size)[None, :]
    )
    # Indexed by the winners' global source pixels, which is the same values
    # the earlier [keep_flat][winners] chain produced and is defined for the
    # A7 structure path, whose winners come from the donor rather than from a
    # boolean over this arm's kept pixels.
    source_of_winner = source_patch.reshape(-1)[src_w]

    count = torch.zeros((n_cells,), dtype=torch.float32)
    count.index_add_(0, lin_w, torch.ones_like(lin_w, dtype=torch.float32))
    hit = count > 0
    hits_per_patch = hit.to(torch.float32).reshape(
        out_patches_h, patch_size, out_patches_w, patch_size
    ).sum(dim=(1, 3))

    target_patch = (lin_w // out_width // patch_size) * out_patches_w + (
        lin_w % out_width
    ) // patch_size
    num_source = patches_h * patches_w
    num_target = out_patches_h * out_patches_w
    weights = torch.zeros((num_target * num_source,), dtype=torch.float32)
    weights.index_add_(0, target_patch * num_source + source_of_winner, 1.0 / count[lin_w])
    weights = weights.reshape(num_target, num_source)
    weights = weights / hits_per_patch.reshape(num_target, 1).clamp(min=1)

    winner_keys = np.sort(
        src_w.numpy().astype(np.int64) * np.int64(n_cells) + lin_w.numpy()
    )
    return SplatPlanDetail(
        weights=weights,
        coverage=hits_per_patch / float(patch_size * patch_size),
        keep=keep_flat,
        winner_keys=winner_keys,
        landing_pixel=landing_pixel,
        landing_uv=uv_tgt.reshape(-1, 2).to(torch.float32).numpy(),
        landing_margin_px=landing_margin,
    )


def assert_forcing_disabled_matches_default(
    depth: Tensor, K_ctx: Tensor, K_tgt: Tensor, T: Tensor, out_hw: tuple[int, int]
) -> None:
    """The diagnostic copy with forcing disabled must be the frozen default."""
    default = transport_plan(depth, K_ctx, K_tgt, T, out_hw)
    detail = splat_plan_detail(depth, K_ctx, K_tgt, T, out_hw)
    if not torch.equal(default.weights, detail.weights) or not torch.equal(
        default.coverage, detail.coverage
    ):
        raise Phase4GateError(
            "the forced-collision machinery with forcing disabled does not "
            "reproduce the frozen transport plan; the diagnostic copy has "
            "drifted from lot.transport.transport_plan"
        )


# ---------------------------------------------------------------------------
# Error-localization masks, PROTOCOL 4.8, frozen before outcomes are viewed
# ---------------------------------------------------------------------------

def depth_boundary_mask(gt_depth: Tensor, analysis: AnalysisConfig) -> np.ndarray:
    """Pixels near a ground-truth depth discontinuity. [H, W] bool.

    Central-difference gradient magnitude of ground-truth depth, compared
    against the frozen threshold as a fraction of local depth, then dilated by
    the frozen radius. Amendment A5 records the operationalization. Pixels
    bordering invalid depth count as boundary; a hole edge is a discontinuity.
    """
    depth = gt_depth.to(torch.float64).numpy()
    valid = np.isfinite(depth) & (depth > 0)
    safe = np.where(valid, depth, 0.0)
    gy, gx = np.gradient(safe)
    magnitude = np.sqrt(gx * gx + gy * gy)
    local = np.where(valid, np.abs(depth), np.inf)
    edge = magnitude > analysis.depth_boundary_gradient_threshold * local
    invalid_adjacent = np.zeros_like(valid)
    invalid = ~valid
    invalid_adjacent[1:] |= invalid[:-1]
    invalid_adjacent[:-1] |= invalid[1:]
    invalid_adjacent[:, 1:] |= invalid[:, :-1]
    invalid_adjacent[:, :-1] |= invalid[:, 1:]
    mask = (edge | invalid_adjacent) & valid
    radius = int(analysis.depth_boundary_dilation_px)
    if radius > 0:
        pooled = torch.nn.functional.max_pool2d(
            torch.from_numpy(mask[None, None].astype(np.float32)),
            2 * radius + 1,
            stride=1,
            padding=radius,
        )
        mask = pooled[0, 0].numpy() > 0
    return mask


def boundary_cells_from_mask(boundary_px: np.ndarray) -> np.ndarray:
    """Patch-level boundary flag: any dilated boundary pixel in the patch.

    The dilation radius already controls the band width, so the patch rule is
    presence, not majority. Amendment A5 records this. [Hp * Wp] bool.
    """
    per_patch = fraction_per_patch(torch.from_numpy(boundary_px), PATCH_SIZE)
    return (per_patch.reshape(-1).numpy() > 0)


def low_texture_cells(rgb: np.ndarray, analysis: AnalysisConfig) -> np.ndarray:
    """Patch-level low-texture flag from the RGB gradient. [Hp * Wp] bool.

    Grayscale is the channel mean scaled to [0, 1]; the statistic is the mean
    central-difference gradient magnitude over each patch; low texture means
    the statistic falls below the frozen threshold. Amendment A5.
    """
    gray = rgb[..., :3].astype(np.float64).mean(axis=-1) / 255.0
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    height, width = magnitude.shape
    per_patch = magnitude.reshape(
        height // PATCH_SIZE, PATCH_SIZE, width // PATCH_SIZE, PATCH_SIZE
    ).mean(axis=(1, 3))
    return (per_patch < analysis.texture_gradient_threshold).reshape(-1)


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------

def _per_sample_cosine(a: Tensor, b: Tensor, center: Tensor | None) -> Tensor:
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    if center is not None:
        a = a - center
        b = b - center
    a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return (a * b).sum(dim=-1)


def _metrics_with_intersection(
    prediction: Tensor,
    target: Tensor,
    center: Tensor,
    centered_defined: bool,
    in_shared: Tensor,
) -> dict[str, float]:
    out = agreement_metrics(prediction, target, center, centered_defined)
    shared = agreement_metrics(
        prediction[in_shared], target[in_shared], center, centered_defined
    )
    out["cosine_intersect_mean"] = shared["cosine_mean"]
    out["cosine_centered_intersect_mean"] = shared["cosine_centered_mean"]
    return out


@dataclasses.dataclass
class PairDepthInputs:
    """Everything estimated-depth about one pair, before any level applies."""

    est_context: np.ndarray            # converted, unaligned, render resolution
    est_target: np.ndarray
    scene_scale: float                 # Level 1 scalar for this pair
    scene_audit: dict[str, int]        # contributing frame to calibration pixels
    context_calib: FrameCalibration    # Level 2 and affine, context image only


@dataclasses.dataclass
class _LevelData:
    """One alignment level's per-pair intermediates, before row emission."""

    variant_name: str
    landed: np.ndarray                 # [N] bool over per-point samples (5d)
    n_transport_valid_pp: int          # 5c count over per-point samples
    n_transport_valid_ctx: int         # 5c pixel count on the context map
    pp_transport_valid: np.ndarray     # [N] bool, the 5c set itself
    ctx_transport_valid: np.ndarray    # [H*W] bool, the 5c set on the map
    uv_warp_est: Tensor
    est_reads: Tensor
    pp_scored: np.ndarray              # universe mask, per-point 5d
    sp_scored: np.ndarray              # universe mask, splat 5d inside Phase 3
    n_est_outside: int                 # splat cells outside the Phase 3 set
    transported_est: Tensor            # [C, cells] pooled estimated features
    context_aligned: np.ndarray
    scale: float


def evaluate_pair_phase4(
    geometry: Any,
    pair: Any,
    depth_context_gt: Tensor,
    depth_inputs: PairDepthInputs,
    features_context: Tensor,
    features_target: Tensor,
    mean_vector: Tensor,
    analysis: AnalysisConfig,
    boundary_cells: np.ndarray,
    lowtex_cells: np.ndarray,
    torch_dtype: torch.dtype,
    K_context: Tensor,
    K_target: Tensor,
    T_target_from_context: Tensor,
    gate_evidence: list[dict[str, Any]],
    feature_encoder: str = "dinov2_vitb14",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every Phase 4 row for one pair. Returns (rows, per-pair audit).

    The Phase 3 population is the referee: the per-point sample set and the
    splat cell set come from pair_geometry unchanged, and every matched
    quantity is computed on the estimated-valid subset of them, per level and
    per path, with the mask persisted on the row as proof.
    """
    rows: list[dict[str, Any]] = []
    center = mean_vector.to(torch.float32)
    channels = features_context.shape[0]
    size = geometry.size
    universe_boundary = pack_mask(boundary_cells)
    universe_lowtex = pack_mask(lowtex_cells)

    samples = geometry.samples
    n_pp = int(samples.uv_target.shape[0])
    per_point_cells = geometry.per_point_cells
    target_hw = depth_inputs.est_target.shape
    box_context = _sampling_box(depth_inputs.est_context.shape, PATCH_SIZE)
    T_context_from_target = invert_se3(T_target_from_context)

    reads_target = sample_features_bilinear(features_target, samples.uv_target)
    reads_warp_gt = sample_features_bilinear(features_context, samples.uv_context_warp)
    reads_no_warp = sample_features_bilinear(features_context, samples.uv_target)

    flat_context = features_context.to(torch.float32).reshape(channels, -1)
    flat_target = features_target.to(torch.float32).reshape(channels, -1)
    splat_cells = np.flatnonzero(geometry.splat_mask)
    transported_gt = apply_transport_plan(geometry.plan, features_context).reshape(channels, -1)

    base = pair.as_row()
    base["covisible_fraction"] = geometry.covisible_fraction
    base["parallax"] = geometry.parallax
    base["encoder"] = feature_encoder

    nan = float("nan")
    empty_gate = {
        "gate_coord_max_px": nan,
        "gate_score_max_abs": nan,
        # The forced-order gate is checked on both metrics. Centering removes a
        # shared direction, so a pooled disagreement invisible in raw cosine can
        # surface once it is gone; gating raw alone would certify a centered
        # table nothing looked at.
        "gate_forced_max_abs": nan,
        "gate_forced_max_abs_centered": nan,
        # Figure 2's series: the forced-order identity check needs the forced
        # scores themselves, not the ordinary matched scores.
        "forced_oracle_raw": nan,
        "forced_estimated_raw": nan,
        "forced_oracle_centered": nan,
        "forced_estimated_centered": nan,
        # A7: the unforced difference is the umbrella rasterization tax, and
        # it decomposes into a landing-assignment component and a
        # collision-ordering component, which telescope back to the umbrella.
        "unforced_rasterization_tax_raw": nan,
        "unforced_rasterization_tax_centered": nan,
        "landing_assignment_tax_raw": nan,
        "landing_assignment_tax_centered": nan,
        "collision_ordering_tax_raw": nan,
        "collision_ordering_tax_centered": nan,
        "collision_gate_cells": 0,
        # A7 landing-cell instability diagnostics, reported and never gated.
        "landing_flip_count": nan,
        "landing_flip_fraction": nan,
        "landing_flip_cells": nan,
        "landing_coord_residual_max_px": nan,
        "landing_flip_margin_min_px": nan,
    }

    def emit(path: str, level: str, population: str, variant: str, mask: np.ndarray,
             prediction: Tensor, target: Tensor, in_shared: Tensor,
             n_shared: int, extra: dict[str, Any]) -> None:
        rows.append(
            {
                **base,
                "path": path,
                "level": level,
                "population": population,
                "variant": variant,
                "n": int(mask.sum()),
                "n_intersect": n_shared,
                "sample_mask": pack_mask(mask),
                "boundary_mask": universe_boundary,
                "lowtex_mask": universe_lowtex,
                **_metrics_with_intersection(
                    prediction, target, center, variant != MEAN_FEATURE, in_shared
                ),
                **extra,
            }
        )

    # PROTOCOL 4.3's inviolable rule, asserted for every evaluated record.
    if pair.target_frame_id in depth_inputs.scene_audit:
        raise Phase4GateError(
            f"{pair.context_frame_id} -> {pair.target_frame_id}: the target "
            "frame contributes calibration pixels to its own Level 1 estimator"
        )
    audit: dict[str, Any] = {
        "scene_scale": depth_inputs.scene_scale,
        "scene_audit_frames": len(depth_inputs.scene_audit),
        "scene_audit_pixels": int(sum(depth_inputs.scene_audit.values())),
        "affine_failed": bool(depth_inputs.context_calib.affine_failed),
        "levels": {},
    }

    def calib_columns(level: str) -> dict[str, Any]:
        if level == "scene":
            return {"calib_pixels": audit["scene_audit_pixels"],
                    "calib_frames": audit["scene_audit_frames"]}
        if level in ("image", "affine"):
            return {"calib_pixels": depth_inputs.context_calib.n_calibration,
                    "calib_frames": 1}
        return {"calib_pixels": 0, "calib_frames": 0}

    # ---- ground-truth rows on the full Phase 3 population ------------------
    shared_full = geometry.cross_path_mask
    n_shared_full = int(shared_full.sum())
    gt_extra = {
        "n_transport_valid": n_pp,
        "n_gt": n_pp,
        "n_est_outside": 0,
        "scale": nan,
        "affine_s": nan,
        "affine_b": nan,
        "calib_pixels": 0,
        "calib_frames": 0,
        **empty_gate,
    }
    if n_pp:
        in_shared_pp = torch.from_numpy(shared_full[per_point_cells])
        mean_feature_pp = center[None, :].expand(n_pp, -1)
        for variant, prediction in (
            (ORACLE_TRANSPORT, reads_warp_gt),
            (NO_WARP_COPY, reads_no_warp),
            (MEAN_FEATURE, mean_feature_pp),
        ):
            emit(PER_POINT, GT_LEVEL, POPULATION_FULL, variant,
                 geometry.per_point_mask, prediction, reads_target, in_shared_pp,
                 n_shared_full, gt_extra)
    if splat_cells.size:
        in_shared_sp = torch.from_numpy(shared_full[splat_cells])
        targets_sp = flat_target[:, splat_cells].T
        sp_extra = {**gt_extra, "n_transport_valid": int(splat_cells.size),
                    "n_gt": int(splat_cells.size)}
        mean_feature_sp = center[None, :].expand(int(splat_cells.size), -1)
        for variant, prediction in (
            (ORACLE_TRANSPORT, transported_gt[:, splat_cells].T),
            (NO_WARP_COPY, flat_context[:, splat_cells].T),
            (MEAN_FEATURE, mean_feature_sp),
        ):
            emit(SPLAT_POOL, GT_LEVEL, POPULATION_FULL, variant,
                 geometry.splat_mask, prediction, targets_sp, in_shared_sp,
                 n_shared_full, sp_extra)

    # ---- alignment levels: intermediates first -----------------------------
    is_rotation = pair.regime == "rotation"
    levels: dict[str, _LevelData] = {}
    for level, variant_name in LEVELS:
        target_aligned = aligned_depth(
            level, depth_inputs.est_target, depth_inputs.scene_scale,
            depth_inputs.context_calib,
        )
        context_aligned = aligned_depth(
            level, depth_inputs.est_context, depth_inputs.scene_scale,
            depth_inputs.context_calib,
        )
        if target_aligned is None or context_aligned is None:
            audit["levels"][level] = {"affine_failed": True}
            continue

        # Per-point path. 5c is a depth-valid read at the frozen samples; 5d
        # adds landing: positive depth in the context camera and a warp inside
        # the context sampling box.
        est_read = sample_map_bilinear(
            torch.from_numpy(target_aligned).to(torch_dtype), samples.uv_target
        )
        read_valid = torch.isfinite(est_read) & (est_read > 0)
        safe_read = torch.where(read_valid, est_read, torch.ones_like(est_read))
        points_target = unproject(samples.uv_target, safe_read, K_target)
        points_context = transform_points(T_context_from_target, points_target)
        uv_warp_est, z_est = project(points_context, K_context)
        landed = (
            read_valid & (z_est > 0) & _in_box(uv_warp_est, box_context)
        ).numpy()
        pp_scored = np.zeros(size, dtype=bool)
        pp_scored[per_point_cells[landed]] = True

        # Splat path through the frozen operator with the aligned context map.
        est_plan = transport_plan(
            torch.from_numpy(context_aligned).to(torch_dtype),
            K_context, K_target, T_target_from_context, target_hw,
        )
        est_cov = (est_plan.coverage.reshape(-1) > 0).numpy()
        sp_scored_own = geometry.splat_covisible_ok & est_cov
        sp_scored = sp_scored_own & geometry.splat_mask
        levels[level] = _LevelData(
            variant_name=variant_name,
            landed=landed,
            n_transport_valid_pp=int(read_valid.sum()),
            n_transport_valid_ctx=int(
                (np.isfinite(context_aligned) & (context_aligned > 0)).sum()
            ),
            pp_transport_valid=read_valid.numpy().copy(),
            ctx_transport_valid=(
                np.isfinite(context_aligned) & (context_aligned > 0)
            ).reshape(-1),
            uv_warp_est=uv_warp_est,
            est_reads=sample_features_bilinear(features_context, uv_warp_est),
            pp_scored=pp_scored,
            sp_scored=sp_scored,
            n_est_outside=int((sp_scored_own & ~geometry.splat_mask).sum()),
            transported_est=apply_transport_plan(est_plan, features_context).reshape(
                channels, -1
            ),
            context_aligned=context_aligned,
            scale=(
                depth_inputs.scene_scale if level == "scene"
                else depth_inputs.context_calib.image_scale if level == "image"
                else nan
            ),
        )
        audit["levels"][level] = {
            "pp_transport_valid": levels[level].n_transport_valid_pp,
            "pp_scored": int(pp_scored.sum()),
            "sp_scored": int(sp_scored.sum()),
            "n_est_outside": levels[level].n_est_outside,
        }

    # Step 10: the multiplicative levels share one transport-valid set, on
    # both the per-point samples and the context map. The scored (5d) sets may
    # legitimately differ and are reported, never asserted.
    present = [l for l in MULTIPLICATIVE_LEVELS if l in levels]
    for left, right in zip(present, present[1:]):
        # The sets themselves, not their sizes: two levels exchanging which
        # samples are valid while keeping the count would pass a count check
        # and still violate the 4.4 identity the protocol states in set terms.
        if not np.array_equal(
            levels[left].pp_transport_valid, levels[right].pp_transport_valid
        ) or not np.array_equal(
            levels[left].ctx_transport_valid, levels[right].ctx_transport_valid
        ):
            raise Phase4GateError(
                f"{pair.context_frame_id} -> {pair.target_frame_id}: "
                f"transport-valid sets differ between levels {left} and "
                f"{right}; positive scaling cannot change finite-and-positive, "
                "so this is an implementation error"
            )

    # Ground-truth splat internals for the forced gate, once per pair.
    gt_detail = None
    if is_rotation and levels:
        gt_detail = splat_plan_detail(
            depth_context_gt, K_context, K_target, T_target_from_context, target_hw
        )

    # ---- row emission per level -------------------------------------------
    for level, data in levels.items():
        shared_level = data.pp_scored & data.sp_scored
        n_shared = int(shared_level.sum())
        common_extra = {
            "n_gt": n_pp,
            "n_est_outside": 0,
            "scale": data.scale,
            "affine_s": depth_inputs.context_calib.affine_s if level == "affine" else nan,
            "affine_b": depth_inputs.context_calib.affine_b if level == "affine" else nan,
            **calib_columns(level),
        }

        # Per-point rows on V = this level's scored subset of Phase 3.
        selected = torch.from_numpy(data.landed)
        chosen = np.flatnonzero(data.landed)
        gate_cols = dict(empty_gate)
        if chosen.size:
            targets_v = reads_target[selected]
            if is_rotation:
                residuals = (
                    data.uv_warp_est[selected] - samples.uv_context_warp[selected]
                ).abs()
                coord_max = float(residuals.max())
                gt_raw = _per_sample_cosine(reads_warp_gt[selected], targets_v, None)
                est_raw = _per_sample_cosine(data.est_reads[selected], targets_v, None)
                gt_cen = _per_sample_cosine(reads_warp_gt[selected], targets_v, center)
                est_cen = _per_sample_cosine(data.est_reads[selected], targets_v, center)
                score_max = max(
                    float((gt_raw - est_raw).abs().max()),
                    float((gt_cen - est_cen).abs().max()),
                )
                gate_cols["gate_coord_max_px"] = coord_max
                gate_cols["gate_score_max_abs"] = score_max
                gate_evidence.append(
                    {
                        "pair": f"{pair.context_frame_id} -> {pair.target_frame_id}",
                        "level": level,
                        "path": PER_POINT,
                        "coord_max_px": coord_max,
                        "score_max_abs": score_max,
                        "n": int(chosen.size),
                    }
                )
                if coord_max > analysis.rotation_gate_coord_tol_px:
                    worst = int(residuals.max(dim=-1).values.argmax())
                    raise Phase4GateError(
                        "PROTOCOL 4.5 per-point coordinate gate failed: scene "
                        f"{pair.scene}, context {pair.context_frame_id}, target "
                        f"{pair.target_frame_id}, level {level}, sample_id "
                        f"{int(samples.sample_id[chosen[worst]])}, oracle uv "
                        f"{samples.uv_context_warp[selected][worst].tolist()}, "
                        f"estimated uv {data.uv_warp_est[selected][worst].tolist()}, "
                        f"residual {coord_max:.3e} px over tolerance "
                        f"{analysis.rotation_gate_coord_tol_px:g}"
                    )
                if score_max > analysis.rotation_gate_score_tol:
                    raise Phase4GateError(
                        "PROTOCOL 4.5 per-point score gate failed: scene "
                        f"{pair.scene}, {pair.context_frame_id} -> "
                        f"{pair.target_frame_id}, level {level}, max score "
                        f"residual {score_max:.3e} over tolerance "
                        f"{analysis.rotation_gate_score_tol:g}"
                    )
            in_shared = torch.from_numpy(shared_level[per_point_cells[data.landed]])
            extra = {
                **common_extra,
                "n_transport_valid": data.n_transport_valid_pp,
                **gate_cols,
            }
            mean_feature_v = center[None, :].expand(int(chosen.size), -1)
            for variant, prediction in (
                (data.variant_name, data.est_reads[selected]),
                (ORACLE_TRANSPORT, reads_warp_gt[selected]),
                (NO_WARP_COPY, reads_no_warp[selected]),
                (MEAN_FEATURE, mean_feature_v),
            ):
                emit(PER_POINT, level, POPULATION_MATCHED, variant, data.pp_scored,
                     prediction, targets_v, in_shared, n_shared, extra)

            # PROTOCOL 4.8 splits of the same matched set, so the boundary and
            # texture contrasts are computable from aggregated rows. Masks and
            # thresholds are frozen ground-truth quantities; empty splits are
            # simply absent and the report layer treats absence as absence.
            for population, cell_flag in (
                ("boundary", boundary_cells),
                ("interior", ~boundary_cells),
                ("lowtex", lowtex_cells),
                ("hightex", ~lowtex_cells),
            ):
                keep = data.landed & cell_flag[per_point_cells]
                if not keep.any():
                    continue
                split_mask = np.zeros(size, dtype=bool)
                split_mask[per_point_cells[keep]] = True
                split_selected = torch.from_numpy(keep)
                split_targets = reads_target[split_selected]
                split_shared = torch.from_numpy(
                    (shared_level & split_mask)[per_point_cells[keep]]
                )
                split_n_shared = int((shared_level & split_mask).sum())
                for variant, prediction in (
                    (data.variant_name, data.est_reads[split_selected]),
                    (ORACLE_TRANSPORT, reads_warp_gt[split_selected]),
                    (NO_WARP_COPY, reads_no_warp[split_selected]),
                ):
                    emit(PER_POINT, level, population, variant, split_mask,
                         prediction, split_targets, split_shared, split_n_shared,
                         extra)

        # Splat rows on V = this level's scored subset of Phase 3.
        cells_v = np.flatnonzero(data.sp_scored)
        sp_gate_cols = dict(empty_gate)
        if cells_v.size:
            targets_v = flat_target[:, cells_v].T
            if is_rotation and gt_detail is not None:
                # PROTOCOL 4.5 forced-collision-order gate, under Amendment
                # A7. The common source set is the intersection of both
                # methods' kept splats. Oracle's complete discrete
                # rasterization structure is frozen on it: every winner's
                # cell assignment, per-cell candidate membership, and the
                # collision winner ordering. Both arms run under that frozen
                # structure, so the gated score comparison is the true
                # invariant 4.5 promises; a one-ulp landing flip can no
                # longer masquerade as an ordering difference. What the
                # estimated arm's own geometry does to landings is measured
                # separately below as the landing-flip diagnostics, reported
                # and never gated.
                est_detail = splat_plan_detail(
                    torch.from_numpy(data.context_aligned).to(torch_dtype),
                    K_context, K_target, T_target_from_context, target_hw,
                )
                common = gt_detail.keep & est_detail.keep
                gt_common = splat_plan_detail(
                    depth_context_gt, K_context, K_target, T_target_from_context,
                    target_hw, source_keep=common,
                )
                structure = splat_structure(gt_common)
                forced_est = splat_plan_detail(
                    torch.from_numpy(data.context_aligned).to(torch_dtype),
                    K_context, K_target, T_target_from_context, target_hw,
                    source_keep=common, forced_structure=structure,
                )
                if not torch.equal(forced_est.weights, gt_common.weights):
                    raise Phase4GateError(
                        "A7 frozen-structure transport produced weights that "
                        "differ from the structure donor's; the forced "
                        "machinery is defective"
                    )
                # The decomposition midpoint: Oracle winner membership tested
                # at this arm's own landings, which is the pre-A7 forced
                # construction. Its gap to the frozen-structure arm is the
                # landing-assignment component of the rasterization tax; its
                # gap to plain z-buffering is the collision-ordering
                # component. The two telescope to the umbrella.
                ordering_forced = splat_plan_detail(
                    torch.from_numpy(data.context_aligned).to(torch_dtype),
                    K_context, K_target, T_target_from_context, target_hw,
                    source_keep=common, forced_winner_keys=gt_common.winner_keys,
                )
                # The unforced arm runs on the same common source population as
                # the forced ones. Comparing forced-on-common against the
                # unrestricted estimated plan would let estimated-only sources
                # enter or win cells in one arm and not the other, so the
                # difference would carry missingness as well as rasterization.
                unforced_common = splat_plan_detail(
                    torch.from_numpy(data.context_aligned).to(torch_dtype),
                    K_context, K_target, T_target_from_context, target_hw,
                    source_keep=common,
                )
                flips = landing_flip_diagnostics(
                    gt_common, unforced_common, target_hw
                )
                flipped_cells = flips.pop("flipped_cells")
                sp_gate_cols.update(flips)
                pooled_oracle = flat_context @ gt_common.weights.mT
                pooled_forced = flat_context @ forced_est.weights.mT
                pooled_ordering = flat_context @ ordering_forced.weights.mT
                pooled_unforced = flat_context @ unforced_common.weights.mT
                both_covered = (
                    (gt_common.coverage.reshape(-1) > 0)
                    & (forced_est.coverage.reshape(-1) > 0)
                    & (ordering_forced.coverage.reshape(-1) > 0)
                    & (unforced_common.coverage.reshape(-1) > 0)
                ).numpy()
                gate_cells = np.flatnonzero(data.sp_scored & both_covered)
                if gate_cells.size:
                    gate_targets = flat_target[:, gate_cells].T
                    scores = {}
                    for label, pooled in (("oracle", pooled_oracle),
                                          ("forced", pooled_forced),
                                          ("ordering", pooled_ordering),
                                          ("unforced", pooled_unforced)):
                        for metric, centre in (("raw", None), ("centered", center)):
                            scores[(label, metric)] = _per_sample_cosine(
                                pooled[:, gate_cells].T, gate_targets, centre
                            )
                    forced_max = float(
                        (scores[("oracle", "raw")] - scores[("forced", "raw")]).abs().max()
                    )
                    forced_max_cen = float(
                        (scores[("oracle", "centered")]
                         - scores[("forced", "centered")]).abs().max()
                    )
                    sp_gate_cols["gate_forced_max_abs"] = forced_max
                    sp_gate_cols["gate_forced_max_abs_centered"] = forced_max_cen
                    sp_gate_cols["forced_oracle_raw"] = float(scores[("oracle", "raw")].mean())
                    sp_gate_cols["forced_estimated_raw"] = float(scores[("forced", "raw")].mean())
                    sp_gate_cols["forced_oracle_centered"] = float(
                        scores[("oracle", "centered")].mean()
                    )
                    sp_gate_cols["forced_estimated_centered"] = float(
                        scores[("forced", "centered")].mean()
                    )
                    for metric in ("raw", "centered"):
                        umbrella = float(
                            (scores[("forced", metric)]
                             - scores[("unforced", metric)]).mean()
                        )
                        assignment = float(
                            (scores[("forced", metric)]
                             - scores[("ordering", metric)]).mean()
                        )
                        ordering_part = float(
                            (scores[("ordering", metric)]
                             - scores[("unforced", metric)]).mean()
                        )
                        sp_gate_cols[f"unforced_rasterization_tax_{metric}"] = umbrella
                        sp_gate_cols[f"landing_assignment_tax_{metric}"] = assignment
                        sp_gate_cols[f"collision_ordering_tax_{metric}"] = ordering_part
                    sp_gate_cols["collision_gate_cells"] = int(gate_cells.size)
                    gate_evidence.append(
                        {
                            "pair": f"{pair.context_frame_id} -> {pair.target_frame_id}",
                            "level": level,
                            "path": SPLAT_POOL,
                            "forced_max_abs": forced_max,
                            "forced_max_abs_centered": forced_max_cen,
                            "unforced_rasterization_tax_raw":
                                sp_gate_cols["unforced_rasterization_tax_raw"],
                            "unforced_rasterization_tax_centered":
                                sp_gate_cols["unforced_rasterization_tax_centered"],
                            "landing_assignment_tax_raw":
                                sp_gate_cols["landing_assignment_tax_raw"],
                            "landing_assignment_tax_centered":
                                sp_gate_cols["landing_assignment_tax_centered"],
                            "collision_ordering_tax_raw":
                                sp_gate_cols["collision_ordering_tax_raw"],
                            "collision_ordering_tax_centered":
                                sp_gate_cols["collision_ordering_tax_centered"],
                            **flips,
                            "flipped_cells": flipped_cells.tolist()[:32],
                            "n_cells": int(gate_cells.size),
                        }
                    )
                    worst = max(forced_max, forced_max_cen)
                    if worst > analysis.rotation_gate_forced_tol:
                        metric_name = "raw" if forced_max >= forced_max_cen else "centered"
                        raise Phase4GateError(
                            "PROTOCOL 4.5 forced-collision-order gate failed: "
                            f"scene {pair.scene}, {pair.context_frame_id} -> "
                            f"{pair.target_frame_id}, level {level}, max forced "
                            f"score residual {worst:.3e} on {metric_name} cosine "
                            f"over tolerance {analysis.rotation_gate_forced_tol:g}"
                        )
            in_shared = torch.from_numpy(shared_level[cells_v])
            extra = {
                **common_extra,
                "n_transport_valid": data.n_transport_valid_ctx,
                "n_gt": int(splat_cells.size),
                "n_est_outside": data.n_est_outside,
                **sp_gate_cols,
            }
            mean_feature_v = center[None, :].expand(int(cells_v.size), -1)
            for variant, prediction in (
                (data.variant_name, data.transported_est[:, cells_v].T),
                (ORACLE_TRANSPORT, transported_gt[:, cells_v].T),
                (NO_WARP_COPY, flat_context[:, cells_v].T),
                (MEAN_FEATURE, mean_feature_v),
            ):
                emit(SPLAT_POOL, level, POPULATION_MATCHED, variant, data.sp_scored,
                     prediction, targets_v, in_shared, n_shared, extra)

            for population, cell_flag in (
                ("boundary", boundary_cells),
                ("interior", ~boundary_cells),
                ("lowtex", lowtex_cells),
                ("hightex", ~lowtex_cells),
            ):
                split_mask = data.sp_scored & cell_flag
                split_cells_v = np.flatnonzero(split_mask)
                if not split_cells_v.size:
                    continue
                split_targets = flat_target[:, split_cells_v].T
                split_shared = torch.from_numpy((shared_level & split_mask)[split_cells_v])
                split_n_shared = int((shared_level & split_mask).sum())
                for variant, prediction in (
                    (data.variant_name, data.transported_est[:, split_cells_v].T),
                    (ORACLE_TRANSPORT, transported_gt[:, split_cells_v].T),
                    (NO_WARP_COPY, flat_context[:, split_cells_v].T),
                ):
                    emit(SPLAT_POOL, level, population, variant, split_mask,
                         prediction, split_targets, split_shared, split_n_shared,
                         extra)

    return rows, audit


# ---------------------------------------------------------------------------
# Scene driver
# ---------------------------------------------------------------------------

def evaluate_scene_phase4(
    cfg: Phase4Config,
    scene: str,
    mean_vector: Tensor,
    analysis: AnalysisConfig,
    convention: dict[str, Any],
    regimes: tuple[str, ...] | None = None,
    collect_rows: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Phase 4 over one scene's frozen Phase 3 pair sample."""
    from PIL import Image

    scene_root = cfg.renders_root / scene
    manifest = load_manifest(scene_root / MANIFEST_NAME)
    validate_manifest(
        manifest,
        scene_root,
        check_files=False,
        rotation_position_bound_m=analysis.rotation_position_bound_m,
        translation_rotation_bound_deg=analysis.translation_rotation_bound_deg,
    )
    frames = {f.frame_id: f for f in manifest.frames}
    pairs = subsample_by_stratum(
        load_scene_pairs(cfg.renders_root, scene, config=analysis),
        analysis.max_pairs_per_stratum,
        seed=cfg.seed,
        config=analysis,
    )
    if regimes is not None:
        pairs = [p for p in pairs if p.regime in regimes]

    # PROTOCOL 4.1 inherits Phase 3's pairs. Regenerating them is the same
    # computation, not the same evidence, so the regenerated set is reconciled
    # against what Phase 3 recorded. A regime filter makes this run a subset,
    # which is checked as a subset; an unfiltered run must match exactly.
    # Two stages, checked separately. The Phase 3 parquet holds the pairs that
    # run scored, which is its sample minus the ones it found unscorable, so
    # the sampled set is a superset and only the containment is checkable here.
    # The equality that matters, that Phase 4 scored exactly what Phase 3
    # scored, is asserted after the evaluation loop against what was scored.
    phase3 = phase3_scene_reference(cfg.phase3_eval_dir, scene, cfg.feature_encoder)
    phase3_pairs = set(phase3["pairs"])
    regenerated = {(p.context_frame_id, p.target_frame_id) for p in pairs}
    missing = phase3_pairs - regenerated
    if regimes is None and missing:
        raise Phase4GateError(
            f"{scene}: {len(missing)} pairs Phase 3 scored are absent from the "
            f"regenerated sample, first {sorted(missing)[:3]}. Manifests, "
            "poses, the frame filter, or the sampler have moved since Phase 3 "
            "ran, so this run would measure a different population."
        )
    expected_scored = phase3_pairs & regenerated

    # The caches and measurement identity Phase 3 recorded must be the ones
    # this run is about to read. Name-level pair agreement proves nothing if
    # the features or the measurement config moved underneath them.
    feature_meta = load_cache_meta(cfg.cache_root, cfg.feature_encoder, scene)
    if phase3["features_digest"] != feature_meta.get("features_digest"):
        raise Phase4GateError(
            f"{scene}: Phase 3 was evaluated from feature cache "
            f"{phase3['features_digest']} and this run would read "
            f"{feature_meta.get('features_digest')}. The inherited ceilings "
            "would come from different features."
        )
    if phase3["measurement_digest"] != analysis.measurement_digest():
        raise Phase4GateError(
            f"{scene}: Phase 3 ran under measurement identity "
            f"{phase3['measurement_digest']}, this analysis carries "
            f"{analysis.measurement_digest()}."
        )

    # One checkpoint, one convention. The verdict comes from the run-level
    # record and never from this scene's diagnostic entry: reading a per-scene
    # verdict here let the same checkpoint be given different depth semantics
    # in different scenes, which would mix two incompatible interpretations
    # into one table. Amendment A6; the invariant is tested permanently.
    depth_cache = load_depth_archive(cfg.cache_root, cfg.depth_encoder, scene)
    verdict = run_convention(convention, depth_cache["meta"])

    cache = _SceneCache(scene_root, cfg.cache_root, [cfg.feature_encoder], scene, manifest)
    est_maps: dict[str, np.ndarray] = {}
    calibrations: dict[str, FrameCalibration] = {}
    resample_records: dict[str, dict[str, Any]] = {}
    # Every frame of the scene must have been treated under one convention.
    # Collected rather than assumed, and asserted after the loop.
    conversions: set[str] = set()
    boundary_by_frame: dict[str, np.ndarray] = {}
    lowtex_by_frame: dict[str, np.ndarray] = {}

    for frame in manifest.frames:
        raw = depth_cache["depth"][frame.frame_id]
        resampled, record = resample_depth_nearest(raw, (frame.height, frame.width))
        resample_records[frame.frame_id] = record
        if verdict == "ray_distance":
            resampled = (
                resampled / secant_map(frame.K, frame.height, frame.width)
            ).astype(np.float32)
        conversions.add(verdict)
        conf_raw = depth_cache["conf"][frame.frame_id]
        conf = (
            resample_depth_nearest(conf_raw, (frame.height, frame.width))[0]
            if conf_raw is not None
            else None
        )
        prevalid = transport_prevalid(resampled, conf, analysis)
        # The 5a rule is applied to the map itself: pixels failing it become
        # NaN, so every downstream consumer, the per-point bilinear read and
        # the splat's keep mask alike, excludes them by construction. Applying
        # the rule only to calibration would let a nonnull confidence
        # threshold change the calibration population while the reported
        # surviving set silently kept every finite positive depth. The mask is
        # scale-independent, so the step 10 invariant is untouched.
        est_maps[frame.frame_id] = np.where(
            prevalid, resampled, np.float32(np.nan)
        ).astype(np.float32)
        calibrations[frame.frame_id] = frame_calibration(
            est_maps[frame.frame_id], cache.depth(frame.depth_path).numpy(), prevalid
        )
    if len(conversions) > 1:
        raise Phase4GateError(
            f"{scene}: frames were treated under {sorted(conversions)}; one "
            "checkpoint carries one depth convention"
        )

    rows: list[dict[str, Any]] = []
    gate_evidence: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    forcing_checked = False
    affine_failed_pairs = 0
    universe_size = 0
    scored_pairs: set[tuple[str, str]] = set()
    phase3_score_max_diff = 0.0

    for pair in pairs:
        context = frames[pair.context_frame_id]
        target = frames[pair.target_frame_id]
        T_target_from_context = relative_pose(
            target.T_world_from_camera, context.T_world_from_camera
        ).to(cfg.torch_dtype)
        depth_context = cache.depth(context.depth_path).to(cfg.torch_dtype)
        depth_target = cache.depth(target.depth_path).to(cfg.torch_dtype)
        geometry = pair_geometry(
            depth_context,
            depth_target,
            context.K.to(cfg.torch_dtype),
            target.K.to(cfg.torch_dtype),
            T_target_from_context,
            scene,
            pair.context_frame_id,
            pair.target_frame_id,
            analysis,
        )
        universe_size = geometry.size
        if not geometry.scorable:
            continue
        assert_translation_parallax_floor(
            pair.regime, geometry.parallax, analysis,
            f"{scene} {pair.context_frame_id} -> {pair.target_frame_id}",
        )
        if not forcing_checked:
            assert_forcing_disabled_matches_default(
                depth_context, context.K.to(cfg.torch_dtype),
                target.K.to(cfg.torch_dtype), T_target_from_context,
                tuple(depth_target.shape),
            )
            forcing_checked = True

        if pair.target_frame_id not in boundary_by_frame:
            boundary_by_frame[pair.target_frame_id] = boundary_cells_from_mask(
                depth_boundary_mask(depth_target, analysis)
            )
            rgb = np.asarray(Image.open(scene_root / target.rgb_path))
            lowtex_by_frame[pair.target_frame_id] = low_texture_cells(rgb, analysis)

        scene_scale, scene_audit = scene_scale_leave_target_out(
            calibrations, pair.target_frame_id
        )
        inputs = PairDepthInputs(
            est_context=est_maps[pair.context_frame_id],
            est_target=est_maps[pair.target_frame_id],
            scene_scale=scene_scale,
            scene_audit=scene_audit,
            context_calib=calibrations[pair.context_frame_id],
        )
        if inputs.context_calib.affine_failed:
            affine_failed_pairs += 1
        pair_rows, audit = evaluate_pair_phase4(
            geometry,
            pair,
            depth_context,
            inputs,
            cache.features(cfg.feature_encoder, pair.context_frame_id),
            cache.features(cfg.feature_encoder, pair.target_frame_id),
            mean_vector,
            analysis,
            boundary_by_frame[pair.target_frame_id],
            lowtex_by_frame[pair.target_frame_id],
            cfg.torch_dtype,
            context.K.to(cfg.torch_dtype),
            target.K.to(cfg.torch_dtype),
            T_target_from_context,
            gate_evidence,
            feature_encoder=cfg.feature_encoder,
        )
        # The inheritance is row-level, not name-level: the recomputed masks
        # must be the ones Phase 3 persisted and the recomputed ceilings the
        # ones it recorded, or the pair identity is the only thing the two
        # runs share.
        pair_key = (pair.context_frame_id, pair.target_frame_id)
        if pair_key in phase3["pairs"]:
            gt_rows = [
                r for r in pair_rows
                if r["level"] == GT_LEVEL and r["population"] == POPULATION_FULL
            ]
            phase3_score_max_diff = max(
                phase3_score_max_diff,
                reconcile_pair_against_phase3(
                    scene, pair, geometry, gt_rows, phase3["pairs"][pair_key]
                ),
            )
        if collect_rows:
            rows.extend(pair_rows)
        audits[f"{pair.context_frame_id} -> {pair.target_frame_id}"] = audit
        scored_pairs.add(pair_key)
    cache.close()

    # The inheritance, stated as equality on what was actually scored. A pair
    # Phase 3 scored and Phase 4 dropped, or the reverse, means the two runs
    # are not measuring the same population however similar their inputs look.
    if scored_pairs != expected_scored:
        dropped = sorted(expected_scored - scored_pairs)[:3]
        added = sorted(scored_pairs - expected_scored)[:3]
        raise Phase4GateError(
            f"{scene}: Phase 4 scored {len(scored_pairs)} pairs where Phase 3 "
            f"scored {len(expected_scored)} of the same sample. Dropped here: "
            f"{dropped}. Added here: {added}. PROTOCOL 4.1 inherits the pair "
            "population unchanged."
        )

    metadata = {
        "phase4_version": PHASE4_VERSION,
        "scene": scene,
        "pairs_evaluated": len(audits),
        "affine_failed_pairs": affine_failed_pairs,
        "gate_checks": len(gate_evidence),
        "gate_coord_max_px": max(
            (g.get("coord_max_px", 0.0) for g in gate_evidence), default=0.0
        ),
        "gate_score_max_abs": max(
            (g.get("score_max_abs", 0.0) for g in gate_evidence), default=0.0
        ),
        "gate_forced_max_abs": max(
            (g.get("forced_max_abs", 0.0) for g in gate_evidence), default=0.0
        ),
        # The run-level convention and its authority, per Amendment A6. The
        # scene's own secant entry rides along as diagnostic evidence and
        # decided nothing.
        "depth_convention": verdict,
        "depth_convention_authority": convention.get("depth_convention_authority"),
        "depth_convention_source_commit": convention.get("depth_convention_source_commit"),
        "depth_convention_conversion_applied": verdict == "ray_distance",
        "secant_regression_role": "diagnostic_only",
        "secant_diagnostic_this_scene": (
            convention.get("secant_diagnostic", {}).get("scenes", {}).get(scene)
        ),
        "resample": next(iter(resample_records.values())) if resample_records else None,
        "git_commit": git_commit(),
        "seed": cfg.seed,
        "feature_encoder": cfg.feature_encoder,
        "depth_encoder": cfg.depth_encoder,
        "analysis_config_digest": analysis.digest(),
        "analysis_measurement_digest": analysis.measurement_digest(),
        "phase4_measurement_digest": phase4_measurement_digest(analysis),
        "analysis_reporting_digest": analysis.reporting_digest(),
        "features_digest": feature_meta["features_digest"],
        "feature_weights_fingerprint": feature_meta["weights_fingerprint"],
        "feature_weights_revision": feature_meta.get("weights_revision", "unpinned"),
        "feature_code_revision": feature_meta.get("code_revision", "unknown"),
        "mean_vector_digest": vector_digest(mean_vector.numpy()),
        "depth_digest": depth_cache["meta"]["depth_digest"],
        "depth_weights_fingerprint": depth_cache["meta"]["weights_fingerprint"],
        "depth_weights_revision": depth_cache["meta"].get("weights_revision", "unpinned"),
        "depth_code_revision": depth_cache["meta"].get("code_revision", "unknown"),
        "universe_size": universe_size,
        "run_scenes": sorted(cfg.scenes),
        "target_exclusion_asserted_per_record": True,
        # The untracked inputs, bound by content so a later audit can tell
        # whether this run and Phase 3 saw the same scene.
        "manifest_digest": manifest_digest(scene_root),
        "phase3_pairs_reconciled": True,
        "phase3_pair_count": len(phase3_pairs),
        "phase3_git_commit": phase3["git_commit"],
        "phase3_features_digest": phase3["features_digest"],
        "phase3_measurement_digest": phase3["measurement_digest"],
        "phase3_score_max_abs_diff": phase3_score_max_diff,
        "phase3_score_recon_tol": PHASE3_SCORE_RECON_TOL,
    }
    return rows, {"metadata": metadata, "gate_evidence": gate_evidence, "audits": audits}


# ---------------------------------------------------------------------------
# Convention report over all scenes
# ---------------------------------------------------------------------------

def convention_report(cfg: Phase4Config, analysis: AnalysisConfig) -> dict[str, Any]:
    """PROTOCOL 4.1's deterministic convention test for the whole run.

    For the first rotation-program frame of every scene, regress resampled
    estimated depth over ground truth against the secant of the pixel angle.
    The report is written before any alignment level runs; evaluation refuses
    to start without it.
    """
    report: dict[str, Any] = {
        "threshold": analysis.depth_convention_slope_threshold,
        "scenes": {},
    }
    for scene in cfg.scenes:
        scene_root = cfg.renders_root / scene
        manifest = load_manifest(scene_root / MANIFEST_NAME)
        first = next(f for f in manifest.frames if f.regime == "rotation")
        depth_cache = load_depth_archive(cfg.cache_root, cfg.depth_encoder, scene)
        resampled, _ = resample_depth_nearest(
            depth_cache["depth"][first.frame_id], (first.height, first.width)
        )
        gt = np.load(scene_root / first.depth_path)
        report["scenes"][scene] = {
            "frame": first.frame_id,
            **secant_regression(
                resampled, gt, first.K, analysis.depth_convention_slope_threshold
            ),
        }
    verdicts = {s: v["verdict"] for s, v in report["scenes"].items()}
    report["verdicts"] = verdicts
    report["unanimous"] = len(set(verdicts.values())) == 1
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4: the estimated-geometry tax.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", type=str)
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--convention", action="store_true",
        help="write the depth-convention report and exit; runs before "
        "anything else per PROTOCOL 4.1",
    )
    parser.add_argument(
        "--doc-verdict", type=str, default=None,
        choices=["planar_z", "ray_distance"],
        help="the depth convention established from VGGT source or "
        "documentation, recorded as the primary authority; a material "
        "disagreement with the regression stops the run",
    )
    parser.add_argument(
        "--gates-only", action="store_true",
        help="run the pure-rotation correctness gates for every level and "
        "write gate evidence, no parquet; PROTOCOL 4.5 runs before any other "
        "result is interpreted",
    )
    args = parser.parse_args(argv)
    cfg = load_phase4_config(args.config)
    analysis = load_analysis_config(cfg.analysis_config)
    if args.list_scenes:
        for index, scene in enumerate(cfg.scenes):
            print(index, scene)
        return

    cfg.evidence_dir.mkdir(parents=True, exist_ok=True)
    # The historical stop evidence keeps its own name and is never rewritten;
    # the record evaluation reads is a separate file carrying the global
    # decision. Preserving the failed consistency check is part of the audit
    # trail, not clutter.
    convention_path = cfg.evidence_dir / "convention_record.json"

    if args.convention:
        diagnostic = convention_report(cfg, analysis)
        authority = source_authority()
        (cfg.evidence_dir / "secant_diagnostic.json").write_text(
            json.dumps(diagnostic, indent=1)
        )
        (cfg.evidence_dir / "source_authority.json").write_text(
            json.dumps(authority, indent=1)
        )
        if not authority.get("unambiguous"):
            raise SystemExit(
                "the pinned VGGT source does not establish the depth "
                f"convention unambiguously: {authority.get('reason')}. "
                f"Evidence in {cfg.evidence_dir / 'source_authority.json'}. STOP."
            )
        if args.doc_verdict is not None and args.doc_verdict != authority["verdict"]:
            raise SystemExit(
                f"the source establishes {authority['verdict']} but "
                f"--doc-verdict says {args.doc_verdict}. Resolve the "
                "disagreement rather than overriding the source. STOP."
            )
        depth_meta = load_cache_meta(cfg.cache_root, cfg.depth_encoder, cfg.scenes[0])
        record = build_convention_record(diagnostic, authority, depth_meta)
        convention_path.write_text(json.dumps(record, indent=1))
        disagree = record["secant_diagnostic_disagrees_with_authority"]
        print(f"depth_convention: {record['depth_convention']} "
              f"(authority: source, commit "
              f"{record['depth_convention_source_commit']})")
        print(f"  {record['depth_convention_source_module']}:"
              f"{record['depth_convention_source_first_line']} "
              f"{record['depth_convention_source_function']}")
        print(f"  checks: {json.dumps(authority['checks'])}")
        print(f"  conversion applied: {record['depth_convention_conversion_applied']}")
        print(f"  secant regression: diagnostic only, disagrees on "
              f"{len(disagree)} of {len(record['secant_diagnostic_verdicts'])} "
              f"scenes {disagree}")
        print(f"-> {convention_path}")
        return

    if not convention_path.exists():
        raise SystemExit(
            f"{convention_path} does not exist. Run --convention first; "
            "PROTOCOL 4.1 decides the convention before any alignment level runs."
        )
    convention = json.loads(convention_path.read_text(encoding="utf-8"))

    if args.scene is not None:
        scenes = [args.scene]
    elif args.scene_index is not None:
        scenes = [cfg.scenes[args.scene_index]]
    else:
        scenes = list(cfg.scenes)

    train = [s for s in cfg.scenes if scene_split(s) == "train"]
    mean_vector = load_or_build_mean_vector(
        cfg.cache_root, cfg.feature_encoder, train, cfg.mean_vector_dir
    )

    for scene in scenes:
        started = time.perf_counter()
        if args.gates_only:
            _, evidence = evaluate_scene_phase4(
                cfg, scene, mean_vector, analysis, convention,
                regimes=("rotation",), collect_rows=False,
            )
            out = cfg.evidence_dir / f"gates_{scene}.json"
            out.write_text(json.dumps(
                {
                    "metadata": evidence["metadata"],
                    "gate_evidence": evidence["gate_evidence"],
                },
                indent=1,
            ))
            meta = evidence["metadata"]
            print(
                f"[{scene}] gates PASS: coord {meta['gate_coord_max_px']:.2e} px, "
                f"score {meta['gate_score_max_abs']:.2e}, forced "
                f"{meta['gate_forced_max_abs']:.2e} over {meta['gate_checks']} "
                f"checks -> {out}"
            )
            continue

        path = cfg.eval_dir / f"{scene}.parquet"
        if path.exists():
            if not args.resume:
                raise SystemExit(f"{path} exists; pass --resume to skip finished scenes")
            stored = read_run_metadata(path)
            differing = stored is None or any(
                stored.get(field) != value
                for field, value in (
                    ("phase4_version", PHASE4_VERSION),
                    ("seed", cfg.seed),
                    ("phase4_measurement_digest", phase4_measurement_digest(analysis)),
                    ("run_scenes", sorted(cfg.scenes)),
                    ("feature_encoder", cfg.feature_encoder),
                    ("depth_encoder", cfg.depth_encoder),
                    ("depth_convention", convention.get("depth_convention")),
                    ("depth_convention_conversion_applied",
                     convention.get("depth_convention_conversion_applied")),
                    ("mean_vector_digest", vector_digest(mean_vector.numpy())),
                )
            )
            if differing:
                raise SystemExit(
                    f"{path} was written by a different run; move the "
                    "directory aside rather than mixing populations"
                )
            print(f"[{scene}] results exist, skipping")
            continue
        rows, evidence = evaluate_scene_phase4(cfg, scene, mean_vector, analysis, convention)
        write_rows(path, rows, evidence["metadata"])
        (cfg.evidence_dir / f"audit_{scene}.json").write_text(
            json.dumps(
                {"gate_evidence": evidence["gate_evidence"], "audits": evidence["audits"]},
                indent=1,
            )
        )
        elapsed = time.perf_counter() - started
        print(f"[{scene}] {len(rows)} rows in {elapsed:.1f} s -> {path}")


if __name__ == "__main__":
    main()
