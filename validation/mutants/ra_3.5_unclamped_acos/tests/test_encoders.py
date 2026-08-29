"""Phase 0 subset of encoders.py: the pixel-to-patch mapping and bilinear sampling.

Phase 2 re-tests sampling against manual interpolation as part of its acceptance.
These tests pin down the mapping and the interpolation arithmetic now because
correspondence sampling depends on them.
"""

import torch

from lot.encoders import (
    patch_to_pixel_coords,
    pixel_to_patch_coords,
    sample_features_bilinear,
    sample_map_bilinear,
)
from lot.geometry import pixel_grid


def test_pixel_to_patch_mapping_formula():
    u = torch.tensor([6.5, 13.5, 20.5], dtype=torch.float64)
    expected = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    assert torch.equal(pixel_to_patch_coords(u, 14), expected)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(100, generator=g, dtype=torch.float64) * 224
    assert torch.allclose(patch_to_pixel_coords(pixel_to_patch_coords(x, 14), 14), x, atol=1e-12)


def test_bilinear_matches_manual_arithmetic():
    grid = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    xy = torch.tensor(
        [[0.5, 0.5], [2.0, 1.0], [-3.0, -3.0], [10.0, 10.0], [1.25, 0.75]],
        dtype=torch.float64,
    )
    values = sample_map_bilinear(grid, xy)
    manual_center = (1 + 2 + 4 + 5) / 4
    manual_frac = 2 * 0.75 * 0.25 + 3 * 0.25 * 0.25 + 5 * 0.75 * 0.75 + 6 * 0.25 * 0.75
    expected = torch.tensor([manual_center, 6.0, 1.0, 6.0, manual_frac], dtype=torch.float64)
    assert torch.allclose(values, expected, atol=1e-12)


def test_bilinear_multichannel_matches_per_channel():
    g = torch.Generator().manual_seed(1)
    grid = torch.rand((3, 5, 7), generator=g, dtype=torch.float64)
    xy = torch.rand((20, 2), generator=g, dtype=torch.float64) * torch.tensor([6.0, 4.0])
    joint = sample_map_bilinear(grid, xy)
    per_channel = torch.stack([sample_map_bilinear(grid[c], xy) for c in range(3)], dim=-1)
    assert torch.allclose(joint, per_channel, atol=1e-12)


def test_feature_sampling_at_patch_centers_is_exact():
    g = torch.Generator().manual_seed(2)
    features = torch.rand((3, 4, 5), generator=g, dtype=torch.float32)
    centers_patch = pixel_grid(4, 5, dtype=torch.float64)
    centers_px = patch_to_pixel_coords(centers_patch, 14)
    sampled = sample_features_bilinear(features, centers_px, 14)
    assert torch.allclose(sampled, features.movedim(0, -1).to(sampled.dtype), atol=1e-7)
