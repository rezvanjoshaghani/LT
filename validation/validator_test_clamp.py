"""Validator-defined test for VALIDATION 3.5, the arccos overshoot condition.

VALIDATION 3.5 states the trigger explicitly: an exact identity is useless
because (trace(R) - 1) / 2 is exactly 1 and arccos is defined there. The
condition needs floating-point overshoot, so the trigger is constructed: take a
near-identity rotation and scale the matrix by (1 + 1e-13), which makes the
arccos argument compute slightly above 1. A clamped arccos returns a finite
value near zero degrees; an unclamped one raises or returns NaN.

The implementation no longer uses arccos. It is atan2 of the skew magnitude
against the trace term, which has no bounded domain to leave, so these tests now
assert that the overshoot condition simply cannot arise rather than that it is
caught. That is a stronger property and the same requirement: VALIDATION 3.5
asks that a matrix whose trace overshoots still yield a finite, correct angle.

The condition is worth keeping as a validator-defined test precisely because it
is independent of how the angle is computed. If the implementation ever returns
to a bounded-domain formulation, this fails again without being rewritten.
"""

import math

import torch

from lot.geometry import rotation_angle_deg


def test_arccos_argument_overshoot_is_clamped():
    R = torch.eye(3, dtype=torch.float64) * (1.0 + 1e-13)
    argument = (float(torch.diagonal(R).sum()) - 1.0) / 2.0
    assert argument > 1.0, "the trigger must actually overshoot"
    angle = rotation_angle_deg(R)
    assert math.isfinite(angle), f"rotation_angle_deg returned {angle!r} on an overshoot"
    assert angle < 1e-3, f"a near-identity rotation must be near zero degrees, got {angle}"


def test_arccos_argument_undershoot_is_clamped():
    R = -torch.eye(3, dtype=torch.float64)
    R[0, 0] = 1.0
    R = R * (1.0 + 1e-13)
    argument = (float(torch.diagonal(R).sum()) - 1.0) / 2.0
    assert argument < -1.0, "the trigger must actually undershoot"
    angle = rotation_angle_deg(R)
    assert math.isfinite(angle), f"rotation_angle_deg returned {angle!r} on an undershoot"
    assert abs(angle - 180.0) < 1e-3, f"expected 180 degrees, got {angle}"
