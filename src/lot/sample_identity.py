"""Deterministic identity for a physical correspondence.

PROTOCOL 3.2: "every physical correspondence carries a deterministic sample_id,
derived from scene, context frame, target frame, and the target-side sample
coordinates. All intersections this protocol references, paired differences,
surviving sets, matched ceilings, and common-valid gates, operate on sample_id,
never on camera-pair membership or equal counts."

The id is a 64-bit integer produced by a fixed mix, so it depends on nothing but
the four inputs: not on batching, not on execution order, not on how many
samples a pair happened to draw, and not on the seed. Two runs that sample the
same correspondence agree on its id, which is what makes the nulls of 3.6
reproducible per record and what lets Phase 4 intersect surviving sets against
Phase 3 without re-deriving anything.

Salts give each derived draw its own independent stream from the same id, so
Random-Patch's patch choice and Neighbor-Patch's direction cannot correlate.
"""

from __future__ import annotations

import zlib

import numpy as np
import torch
from torch import Tensor

# Fixed salts. Changing one changes which null a record draws, so they are
# constants of the protocol rather than tunables.
RANDOM_PATCH_SALT = np.uint64(0x9E3779B97F4A7C15)
NEIGHBOR_PATCH_SALT = np.uint64(0xC2B2AE3D27D4EB4F)

# The target-side coordinate is quantized to half-pixels before hashing. Patch
# centers sit at half-integers and pixel centers at integers, so this is exact
# for every location this project samples, and it makes the id independent of
# floating point noise in the coordinate arithmetic.
COORD_QUANTUM = 2.0
_COORD_STRIDE = np.uint64(1 << 21)


def _mix64(value: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer. Vectorized, wraps modulo 2^64 like the C original."""
    x = value.astype(np.uint64, copy=True)
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def pair_seed(scene: str, context_frame_id: str, target_frame_id: str) -> np.uint64:
    """The part of a sample_id that does not depend on where in the frame it sits."""
    key = f"{scene}|{context_frame_id}|{target_frame_id}".encode("utf-8")
    return _mix64(np.array([zlib.crc32(key)], dtype=np.uint64))[0]


def sample_ids(
    scene: str, context_frame_id: str, target_frame_id: str, uv_target: Tensor
) -> np.ndarray:
    """Deterministic 64-bit ids for target-side sample coordinates.

    uv_target: [N, 2] pixel coordinates in the target image.
    Returns [N] uint64. Equal coordinates in equal frames give equal ids, in any
    order and any batching.
    """
    if uv_target.dim() != 2 or uv_target.shape[-1] != 2:
        raise ValueError(f"uv_target must be [N, 2], got {tuple(uv_target.shape)}")
    quantized = torch.round(uv_target.detach().to(torch.float64) * COORD_QUANTUM)
    if not torch.equal(quantized, uv_target.detach().to(torch.float64) * COORD_QUANTUM):
        raise ValueError(
            "target-side coordinates must be whole multiples of half a pixel; "
            "sample_id is defined on that grid so it cannot drift with float noise"
        )
    coords = quantized.cpu().numpy()
    if coords.min() < 0:
        raise ValueError("target-side coordinates must be non-negative")
    code = coords[:, 0].astype(np.uint64) * _COORD_STRIDE + coords[:, 1].astype(np.uint64)
    seed = pair_seed(scene, context_frame_id, target_frame_id)
    with np.errstate(over="ignore"):
        return _mix64(np.uint64(seed) ^ _mix64(code))


def derived_draw(ids: np.ndarray, salt: np.uint64, modulus: np.ndarray | int) -> np.ndarray:
    """A per-record draw in [0, modulus) from a sample_id and a salt.

    modulus may be a scalar or a per-record array, which is what Neighbor-Patch
    needs: the number of in-bounds offsets differs at the image border.
    """
    with np.errstate(over="ignore"):
        mixed = _mix64(ids.astype(np.uint64) ^ np.uint64(salt))
    bound = np.asarray(modulus, dtype=np.uint64)
    if np.any(bound == 0):
        raise ValueError("modulus must be positive for every record")
    return (mixed % bound).astype(np.int64)
