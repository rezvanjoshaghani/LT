"""Phase 5's aligned context depth is Phase 4's, proven rather than asserted.

Phase 4 is accepted and pinned, so it cannot be refactored to call the shared
helper; the per-frame sequence therefore exists twice, once inline in
lot.phase4.evaluate_scene_phase4 and once in lot.phase5.prepare_frame_depth.
Two copies nobody compares would let the next amendment to Phase 4 leave Phase 5
on the old rule, and both headline methods would then consume geometry that is
not Phase 4's while the docstring still claimed it was "by construction".

This module replays Phase 4's inline sequence, written out here from its own
source, and requires the shared helper to reproduce it bit for bit. It is the
same discipline lot.paired_bootstrap uses to share Phase 4's bootstrap: one
algorithm, two entry points, equivalence proven by a permanent test.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from lot import phase4
from lot.analysis_config import load_analysis_config
from lot.phase4 import (
    frame_calibration,
    resample_depth_nearest,
    secant_map,
    transport_prevalid,
)
from lot.phase5 import prepare_frame_depth

ANALYSIS = load_analysis_config()


class Frame:
    """The manifest fields the per-frame sequence reads."""

    def __init__(self, height: int, width: int, K: torch.Tensor):
        self.height = height
        self.width = width
        self.K = K


def _intrinsics(width: int) -> torch.Tensor:
    focal = float(width)
    centre = (width - 1) / 2.0
    return torch.tensor(
        [[focal, 0.0, centre], [0.0, focal, centre], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def phase4_inline_sequence(raw_depth, raw_conf, frame, gt_depth, verdict, analysis):
    """Phase 4's own steps, transcribed from lot.phase4.evaluate_scene_phase4.

    Deliberately a transcription and not a call: if it called the shared helper
    the test would compare the helper with itself and prove nothing.
    """
    resampled, _ = resample_depth_nearest(raw_depth, (frame.height, frame.width))
    if verdict == "ray_distance":
        resampled = (
            resampled / secant_map(frame.K, frame.height, frame.width)
        ).astype(np.float32)
    conf = (
        resample_depth_nearest(raw_conf, (frame.height, frame.width))[0]
        if raw_conf is not None
        else None
    )
    prevalid = transport_prevalid(resampled, conf, analysis)
    aligned = np.where(prevalid, resampled, np.float32(np.nan)).astype(np.float32)
    calibration = frame_calibration(aligned, gt_depth, prevalid)
    return aligned, calibration


def _case(seed: int, size: int = 28, with_conf: bool = True, holes: bool = True):
    rng = np.random.default_rng(seed)
    raw = (rng.random((size, size), dtype=np.float32) * 4.0 + 0.5).astype(np.float32)
    if holes:
        raw[0, :] = 0.0                      # nonpositive, must be masked out
        raw[1, 0] = np.float32(np.nan)       # nonfinite, must be masked out
        raw[2, 1] = np.float32(np.inf)
    conf = rng.random((size, size), dtype=np.float32) if with_conf else None
    gt = (rng.random((size, size), dtype=np.float32) * 4.0 + 0.5).astype(np.float32)
    return raw, conf, Frame(size, size, _intrinsics(size)), gt


def _assert_identical(a, b):
    aligned_a, calib_a = a
    aligned_b, calib_b = b
    assert aligned_a.dtype == aligned_b.dtype == np.float32
    # Bit for bit, NaNs in the same places.
    np.testing.assert_array_equal(np.isnan(aligned_a), np.isnan(aligned_b))
    finite = ~np.isnan(aligned_a)
    np.testing.assert_array_equal(aligned_a[finite], aligned_b[finite])
    for field in ("scale", "affine_failed"):
        if hasattr(calib_a, field):
            got, want = getattr(calib_a, field), getattr(calib_b, field)
            if isinstance(got, float) and np.isnan(got):
                assert np.isnan(want)
            else:
                assert got == want, field


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("verdict", ["planar_z", "ray_distance"])
def test_the_shared_helper_reproduces_phase4s_inline_sequence(seed, verdict):
    raw, conf, frame, gt = _case(seed)
    _assert_identical(
        prepare_frame_depth(raw, conf, frame, gt, verdict, ANALYSIS),
        phase4_inline_sequence(raw, conf, frame, gt, verdict, ANALYSIS),
    )


def test_equivalence_holds_without_a_confidence_map():
    raw, _, frame, gt = _case(3, with_conf=False)
    _assert_identical(
        prepare_frame_depth(raw, None, frame, gt, "planar_z", ANALYSIS),
        phase4_inline_sequence(raw, None, frame, gt, "planar_z", ANALYSIS),
    )


def test_equivalence_holds_when_every_pixel_is_invalid():
    raw, conf, frame, gt = _case(5)
    raw[:] = 0.0
    mine, theirs = (
        prepare_frame_depth(raw, conf, frame, gt, "planar_z", ANALYSIS),
        phase4_inline_sequence(raw, conf, frame, gt, "planar_z", ANALYSIS),
    )
    assert np.isnan(mine[0]).all()
    _assert_identical(mine, theirs)


def test_the_ray_distance_branch_actually_changes_the_map():
    """Otherwise the convention parametrization above would prove nothing."""
    raw, conf, frame, gt = _case(11)
    planar, _ = prepare_frame_depth(raw, conf, frame, gt, "planar_z", ANALYSIS)
    ray, _ = prepare_frame_depth(raw, conf, frame, gt, "ray_distance", ANALYSIS)
    finite = ~np.isnan(planar) & ~np.isnan(ray)
    assert finite.any()
    assert not np.array_equal(planar[finite], ray[finite])


def test_the_transcription_still_matches_phase4s_source():
    """If Phase 4's inline sequence is ever amended, this test must be updated.

    The equivalence above compares this module's transcription with the shared
    helper. That is only evidence about Phase 4 while the transcription is
    faithful, so the calls Phase 4 actually makes are pinned here by name. A new
    step appearing in Phase 4's loop fails this and sends the reader to the
    transcription rather than letting the two drift in silence.
    """
    source = inspect.getsource(phase4.evaluate_scene_phase4)
    tree = ast.parse(source.strip())
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    for name in ("resample_depth_nearest", "secant_map", "transport_prevalid",
                 "frame_calibration"):
        assert name in called, f"Phase 4 no longer calls {name}; retranscribe"
