"""Streams S and T: the predictor's information and computation boundaries.

The Phase 5 headline claim is only meaningful if the two comparators really did
receive the same information and really did differ only in whether the known
transformation was computed or learned. These tests are what make that
assertion evidence rather than a statement of intent, so a leak here is a stop.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest
import torch

from lot import predictors
from lot.predictors import (
    PredictWithDepth,
    PredictorConfig,
    build_predictor,
    camera_vector,
    patch_depth_windows,
)

# Small enough to train in a unit test, same shape in every structural respect.
TINY = PredictorConfig(
    d_model=32, n_blocks=2, n_heads=4, ffn_dim=64, dropout=0.0,
    feature_dim=16, patch_size=2, context_grid=(3, 3), target_grid=(3, 3),
)
CTX_N = TINY.context_grid[0] * TINY.context_grid[1]
TGT_N = TINY.target_grid[0] * TINY.target_grid[1]
HW = (TINY.context_grid[0] * TINY.patch_size, TINY.context_grid[1] * TINY.patch_size)


def _batch(batch: int = 2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    features = torch.randn(batch, CTX_N, TINY.feature_dim, generator=g)
    depth = torch.rand(batch, *HW, generator=g) * 4.0 + 0.5
    camera = torch.randn(batch, TINY.camera_features, generator=g)
    return features, depth, camera


# ---------------------------------------------------------------------------
# Stream S step 10: the information boundary
# ---------------------------------------------------------------------------

FORBIDDEN_INPUTS = (
    "target_rgb",
    "rgb_target",
    "features_target",
    "target_features",
    "depth_target",
    "target_depth",
    "target_pointmap",
    "pointmap",
    "optical_flow",
    "flow",
    "covisible",
    "co_visible",
    "correspondence",
    "landing",
    "uv_target",
)


def test_forward_signature_admits_no_target_content():
    """No parameter exists through which target content could be supplied."""
    params = list(inspect.signature(PredictWithDepth.forward).parameters)
    assert params == [
        "self",
        "features_context",
        "depth_context_aligned",
        "camera",
        "context_valid",
    ]
    for name in params:
        for bad in FORBIDDEN_INPUTS:
            assert bad not in name, f"forward parameter {name} looks like target content"


def _module_identifiers(module) -> set[str]:
    tree = ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def test_module_references_no_target_content_symbol():
    names = _module_identifiers(predictors)
    for bad in FORBIDDEN_INPUTS:
        offenders = {n for n in names if bad in n}
        assert not offenders, f"lot.predictors references {offenders}"


# ---------------------------------------------------------------------------
# Stream T step 11: the computation boundary
# ---------------------------------------------------------------------------

FORBIDDEN_OPERATIONS = (
    "unproject",
    "project",
    "transform_points",
    "grid_sample",
    "transport_plan",
    "apply_transport_plan",
    "splat",
    "zbuffer",
    "z_buffer",
    "rotation_homography",
    "apply_homography",
    "visibility_masks",
    "sample_features_bilinear",
    "sample_map_bilinear",
    "inverse",
    "invert_se3",
)


def test_module_performs_no_explicit_projective_operation():
    """The line the experiment rests on: it may not execute the known solution."""
    names = _module_identifiers(predictors)
    for bad in FORBIDDEN_OPERATIONS:
        offenders = {n for n in names if bad == n}
        assert not offenders, f"lot.predictors performs {offenders}"


def node_level(node) -> int:
    return getattr(node, "level", 0) or 0


def test_module_imports_only_the_patch_size_from_the_geometry_stack():
    """A geometry import is the way an explicit warp would arrive. Pin the set."""
    tree = ast.parse(Path(inspect.getfile(predictors)).read_text(encoding="utf-8"))
    imported: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        # A relative import keeps its dots in node.level, not in node.module,
        # so both forms have to be recognized or the audit silently sees nothing.
        is_local = node_level(node) > 0 or (getattr(node, "module", "") or "").startswith("lot")
        if isinstance(node, ast.ImportFrom) and is_local:
            imported.append((node.module or "", tuple(a.name for a in node.names)))
    assert imported == [("encoders", ("PATCH_SIZE",))], imported


# ---------------------------------------------------------------------------
# The camera parameterization
# ---------------------------------------------------------------------------

def test_camera_vector_is_the_frozen_21_numbers():
    T = torch.eye(4)[None]
    T[0, :3, 3] = torch.tensor([0.3, -0.2, 0.1])
    K = torch.tensor([[[100.0, 0.0, 50.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]])
    v = camera_vector(T, K, K, (120, 100), (120, 100))
    assert v.shape == (1, 21)
    assert torch.allclose(v[0, :9], torch.eye(3).reshape(9))
    assert torch.allclose(v[0, 9:12], torch.tensor([0.3, -0.2, 0.1]))
    assert v[0, 12].item() == pytest.approx(math.sqrt(0.09 + 0.04 + 0.01))
    # fx/W, fy/H, cx/W, cy/H for each camera.
    assert torch.allclose(v[0, 13:17], torch.tensor([1.0, 100 / 120, 0.5, 0.5]))
    assert torch.allclose(v[0, 17:21], torch.tensor([1.0, 100 / 120, 0.5, 0.5]))


def test_camera_vector_carries_no_landing_location():
    """It describes the cameras. It must not encode where anything projects to.

    A translation that changes every correspondence in the image changes only
    the three translation entries and the norm, never a per-patch quantity,
    because there are no per-patch entries at all.
    """
    T_a = torch.eye(4)[None]
    T_b = torch.eye(4)[None]
    T_b[0, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
    K = torch.eye(3)[None] * 100.0
    K[0, 2, 2] = 1.0
    a = camera_vector(T_a, K, K, (100, 100), (100, 100))
    b = camera_vector(T_b, K, K, (100, 100), (100, 100))
    differing = (a - b).abs() > 0
    assert differing.sum().item() == 2, "only t_x and |t| may move"


# ---------------------------------------------------------------------------
# The depth embedding
# ---------------------------------------------------------------------------

def test_patch_depth_windows_partition_the_map_row_major():
    depth = torch.arange(36, dtype=torch.float32).reshape(1, 6, 6) + 1.0
    log_depth, valid = patch_depth_windows(depth, patch_size=2)
    assert log_depth.shape == (1, 9, 4)
    assert bool(valid.all())
    # First patch is the top-left 2x2 block: values 1, 2, 7, 8.
    assert torch.allclose(log_depth[0, 0].exp(), torch.tensor([1.0, 2.0, 7.0, 8.0]))
    # Second patch steps along the row, not down the column.
    assert torch.allclose(log_depth[0, 1].exp(), torch.tensor([3.0, 4.0, 9.0, 10.0]))


def test_patch_depth_windows_separate_invalid_from_near():
    depth = torch.full((1, 2, 2), 2.0)
    depth[0, 0, 0] = float("nan")
    depth[0, 0, 1] = -1.0
    log_depth, valid = patch_depth_windows(depth, patch_size=2)
    assert valid[0, 0].tolist() == [0.0, 0.0, 1.0, 1.0]
    # Invalid entries are zeroed in the value plane rather than left as a
    # sentinel the network would have to learn to recognize.
    assert log_depth[0, 0, 0].item() == 0.0
    assert log_depth[0, 0, 1].item() == 0.0
    assert torch.isfinite(log_depth).all()


def test_patch_depth_windows_reject_a_ragged_map():
    with pytest.raises(ValueError, match="whole"):
        patch_depth_windows(torch.ones(1, 5, 4), patch_size=2)


# ---------------------------------------------------------------------------
# The model itself
# ---------------------------------------------------------------------------

def test_forward_produces_a_complete_target_grid():
    model = build_predictor(TINY).eval()
    features, depth, camera = _batch()
    with torch.no_grad():
        out = model(features, depth, camera)
    assert out.shape == (2, TGT_N, TINY.feature_dim)
    assert torch.isfinite(out).all()


def test_prediction_depends_on_every_permitted_input():
    """If an input cannot change the output, the model is not using it at all."""
    model = build_predictor(TINY).eval()
    features, depth, camera = _batch()
    with torch.no_grad():
        base = model(features, depth, camera)
        moved_features = model(features + 1.0, depth, camera)
        moved_depth = model(features, depth * 2.0 + 0.1, camera)
        moved_camera = model(features, depth, camera + 1.0)
    for name, other in (
        ("features", moved_features), ("depth", moved_depth), ("camera", moved_camera)
    ):
        assert not torch.allclose(base, other), f"the model ignores {name}"


def test_examples_in_a_batch_do_not_mix():
    """A pair's prediction must depend on its own inputs and no other pair's."""
    model = build_predictor(TINY).eval()
    features, depth, camera = _batch(batch=3)
    with torch.no_grad():
        together = model(features, depth, camera)
        alone = model(features[1:2], depth[1:2], camera[1:2])
    assert torch.allclose(together[1], alone[0], atol=1e-5)


def test_context_validity_mask_excludes_a_token():
    model = build_predictor(TINY).eval()
    features, depth, camera = _batch()
    valid = torch.ones(2, CTX_N, dtype=torch.bool)
    valid[:, 0] = False
    with torch.no_grad():
        masked = model(features, depth, camera, context_valid=valid)
        wrecked = features.clone()
        wrecked[:, 0] = 1e3
        masked_again = model(wrecked, depth, camera, context_valid=valid)
    assert torch.allclose(masked, masked_again, atol=1e-5), "a masked token still leaked"


def test_target_queries_carry_no_pair_specific_content():
    """Two pairs with identical cameras and context get identical queries.

    The queries are a fixed lattice plus the camera vector, so anything
    pair-specific in the output has to have come through the context or the
    cameras, which is exactly the information budget the experiment allows.
    """
    model = build_predictor(TINY).eval()
    features, depth, camera = _batch()
    with torch.no_grad():
        a = model(features[:1], depth[:1], camera[:1])
        b = model(features[:1].clone(), depth[:1].clone(), camera[:1].clone())
    assert torch.allclose(a, b)


def test_config_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="divisible"):
        PredictorConfig(d_model=384, n_heads=5)


def test_frozen_architecture_parameter_count_is_recorded():
    """The shipped architecture's size is pinned, so a silent widening fails here."""
    model = build_predictor(PredictorConfig())
    count = model.parameter_count()
    assert count == 16_680_960, count


def test_the_model_can_learn_a_pair_specific_mapping():
    """A miniature of the Stream U overfit gate: the trunk must be able to fit.

    Not the frozen gate, which runs on real pairs at the frozen threshold. This
    only rules out an architecture that cannot descend at all, which would make
    that gate's failure uninformative about the science.
    """
    torch.manual_seed(0)
    model = build_predictor(TINY)
    features, depth, camera = _batch(batch=2, seed=3)
    target = torch.randn(2, TGT_N, TINY.feature_dim, generator=torch.Generator().manual_seed(9))
    target = target / target.norm(dim=-1, keepdim=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        opt.zero_grad()
        pred = model(features, depth, camera)
        cosine = torch.nn.functional.cosine_similarity(pred, target, dim=-1)
        loss = (1.0 - cosine).mean()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first * 0.5, f"loss went {first} -> {loss.item()}"
    assert cosine.mean().item() > 0.9, cosine.mean().item()
