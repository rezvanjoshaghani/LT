"""Validator-defined test for VALIDATION 3.5's overshoot trigger.

A near-identity rotation scaled by (1 + 1e-13) pushes the trace term's cosine
argument just above 1. The frozen implementation must return a finite angle
near zero degrees; the unclamped-acos mutant must raise or return NaN.
"""

import math

import torch

from lot.geometry import rotation_angle_deg


def _near_identity() -> torch.Tensor:
    angle = 1e-8
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    return R * (1.0 + 1e-13)


def test_overshot_trace_argument_yields_finite_angle():
    scaled = _near_identity()
    cosine = (float(torch.diagonal(scaled).sum()) - 1.0) / 2.0
    assert cosine > 1.0, "the trigger must actually overshoot the acos domain"
    angle = rotation_angle_deg(scaled)
    assert math.isfinite(angle)
    assert abs(angle) < 1e-3


def test_exact_identity_is_zero_degrees():
    assert rotation_angle_deg(torch.eye(3, dtype=torch.float64)) == 0.0
