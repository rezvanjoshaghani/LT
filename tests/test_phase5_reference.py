"""TL-Reference is a transcription of Phase 4's per-pair arm. Pin that it stays one.

lot.phase5_reference recomputes Phase 4's per-point and splat arms, because
Phase 4 persisted masks and scores but not predictions. On real data every pair
is reconciled against Phase 4's persisted rows during evaluate, so a drifted
transcription stops the run there. tests/test_phase5_modes.py drives that
reconciliation end to end against a genuine Phase 4 parquet.

These tests catch drift earlier and without data. Phase 4 is pinned and cannot
be refactored to share the code, so the two are held together by the
primitives they call: the same names, and the same objects. A new primitive in
Phase 4's per-pair function, or a substituted one in the transcription, fails
here and sends the reader to the transcription.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from lot import phase4, phase5_reference

PRIMITIVES = (
    "aligned_depth",
    "sample_map_bilinear",
    "unproject",
    "transform_points",
    "project",
    "invert_se3",
    "_sampling_box",
    "_in_box",
    "transport_plan",
    "apply_transport_plan",
    "sample_features_bilinear",
)


def _called(function) -> set[str]:
    tree = ast.parse(inspect.getsource(function).strip())
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_phase4_still_calls_every_primitive_the_transcription_uses():
    called = _called(phase4.evaluate_pair_phase4)
    for name in PRIMITIVES:
        assert name in called, (
            f"Phase 4's per-pair arm no longer calls {name}; retranscribe "
            "lot.phase5_reference.recompute_reference_arms"
        )


def test_the_transcription_calls_the_same_primitives():
    called = _called(phase5_reference.recompute_reference_arms)
    for name in PRIMITIVES:
        assert name in called, f"the transcription no longer calls {name}"


@pytest.mark.parametrize("name", PRIMITIVES)
def test_both_resolve_each_primitive_to_one_object(name):
    """Same names are not enough. A local reimplementation could shadow one."""
    assert getattr(phase5_reference, name) is getattr(phase4, name)


def test_levels_map_onto_phase4s_variants():
    assert phase5_reference.VARIANT_OF_LEVEL == dict(phase4.LEVELS)
    for level in phase5_reference.PHASE5_LEVELS:
        assert level in phase5_reference.VARIANT_OF_LEVEL
    with pytest.raises(ValueError, match="not a Phase 5 condition"):
        phase5_reference._variant_for("scene")
