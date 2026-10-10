"""Predict-with-Depth: the Phase 5 learned comparator. Stream T.

The experimental contrast Phase 5 measures is *same information, different
computation*. Context-Lift Transport-Only receives

    I = { F_c, D_c, K_c, K_t, T_target_from_context }

and executes the known projective transformation. This module receives exactly
the same I and must learn that transformation from it.

Two boundaries are load bearing, and both are enforced structurally rather than
by convention.

The information boundary. The forward signature accepts no target features, no
target depth, no target RGB, no co-visible mask, and no precomputed landing
coordinates. Target-side information enters only as query *coordinates*, which
are the fixed patch lattice of the target grid and carry no target content: the
same lattice for every pair in the dataset. A support mask never enters the
model; it decides only where a loss or a score is taken, outside this module.

The computation boundary. Nothing here unprojects, lifts to 3D, transforms a
point by the pose, projects into the target, splats, z-buffers, or builds a
sampling grid from geometry. The camera variables arrive as a flat vector of
numbers and reach the network only through learned layers. The permanent tests
assert both boundaries by parsing what this module references, so a future edit
that quietly imports `unproject` fails the suite rather than the review.

Predicting centered features is a deliberate choice inherited from PROTOCOL
3.7: the centered metric is the primary one because the raw cosine between two
DINOv2 features is dominated by a shared direction that no method has to earn.
Training on the centered target removes that offset from the optimization as
well as from the reporting. The frozen Phase 3 global mean is added back when a
raw-cosine number is wanted.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor, nn

# The width of one DINO patch in pixels, and therefore the side of the depth
# window a context token summarizes. Imported rather than restated so this
# module cannot drift from the cache it consumes.
from .encoders import PATCH_SIZE


@dataclasses.dataclass(frozen=True)
class PredictorConfig:
    """The frozen architecture. Every field is read from configs/phase5.yaml."""

    d_model: int = 384
    n_blocks: int = 6
    n_heads: int = 6
    ffn_dim: int = 1536
    dropout: float = 0.1
    feature_dim: int = 768
    camera_features: int = 21
    patch_size: int = PATCH_SIZE
    context_grid: tuple[int, int] = (37, 37)
    target_grid: tuple[int, int] = (37, 37)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}"
            )


def camera_vector(
    T_target_from_context: Tensor,
    K_context: Tensor,
    K_target: Tensor,
    context_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> Tensor:
    """The frozen numerical camera parameterization, 21 numbers.

    Nine entries of the relative rotation, three of the relative translation in
    meters, its norm, and four normalized intrinsics per camera. Intrinsics are
    normalized by image size so the network sees a scale-free description of the
    optics rather than a pixel count it would have to memorize.

    This is a description of the cameras, not a solution: it contains no
    correspondence, no ray, and no landing location. Which is the point. The
    same numbers are available to the explicit comparator, which turns them into
    a projective map; the network has to discover that map.
    """
    if T_target_from_context.shape[-2:] != (4, 4):
        raise ValueError("T_target_from_context must be [..., 4, 4]")
    R = T_target_from_context[..., :3, :3].reshape(*T_target_from_context.shape[:-2], 9)
    t = T_target_from_context[..., :3, 3]
    t_norm = t.norm(dim=-1, keepdim=True)

    def normalized(K: Tensor, hw: tuple[int, int]) -> Tensor:
        height, width = hw
        fx = K[..., 0, 0] / width
        fy = K[..., 1, 1] / height
        cx = K[..., 0, 2] / width
        cy = K[..., 1, 2] / height
        return torch.stack([fx, fy, cx, cy], dim=-1)

    return torch.cat(
        [R, t, t_norm, normalized(K_context, context_hw), normalized(K_target, target_hw)],
        dim=-1,
    )


def patch_depth_windows(
    depth_context_aligned: Tensor, patch_size: int = PATCH_SIZE
) -> tuple[Tensor, Tensor]:
    """Split an aligned context depth map into per-patch windows.

    depth_context_aligned: [B, H, W] planar z-depth in meters, after alignment.
    Returns (log_depth, valid), each [B, N, patch_size * patch_size], where N is
    the number of patches in row-major order.

    Log depth rather than metric depth because scene depth spans an order of
    magnitude and a linear input would make the near field dominate the
    embedding's dynamic range. Invalid pixels are zeroed in the value plane and
    marked in the validity plane, so the network can tell "close to the camera"
    from "no measurement" instead of having to infer it from a sentinel.
    """
    if depth_context_aligned.dim() != 3:
        raise ValueError("depth_context_aligned must be [B, H, W]")
    batch, height, width = depth_context_aligned.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"depth map {(height, width)} is not whole {patch_size} px patches")
    grid_h, grid_w = height // patch_size, width // patch_size

    windows = depth_context_aligned.reshape(batch, grid_h, patch_size, grid_w, patch_size)
    windows = windows.permute(0, 1, 3, 2, 4).reshape(
        batch, grid_h * grid_w, patch_size * patch_size
    )
    valid = torch.isfinite(windows) & (windows > 0)
    safe = torch.where(valid, windows, torch.ones_like(windows))
    log_depth = torch.where(valid, safe.log(), torch.zeros_like(safe))
    return log_depth, valid.to(log_depth.dtype)


class DecoderBlock(nn.Module):
    """Pre-norm block: self-attention over queries, cross-attention to context, FFN."""

    def __init__(self, cfg: PredictorConfig) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(cfg.d_model)
        self.self_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm_cross_q = nn.LayerNorm(cfg.d_model)
        self.norm_cross_kv = nn.LayerNorm(cfg.d_model)
        self.cross_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, queries: Tensor, context: Tensor, context_pad: Tensor | None) -> Tensor:
        h = self.norm_self(queries)
        queries = queries + self.drop(self.self_attn(h, h, h, need_weights=False)[0])
        q = self.norm_cross_q(queries)
        kv = self.norm_cross_kv(context)
        queries = queries + self.drop(
            self.cross_attn(q, kv, kv, key_padding_mask=context_pad, need_weights=False)[0]
        )
        return queries + self.drop(self.ffn(self.norm_ffn(queries)))


class PredictWithDepth(nn.Module):
    """The learned comparator. Consumes I and predicts a centered target grid.

    forward returns [B, N_target, feature_dim] centered feature predictions, one
    per target patch coordinate, in the same row-major order the caches use. A
    complete grid is produced by construction, which is why Stream V forbids the
    predictor from narrowing the evaluation support: it always has an answer
    everywhere, so a missing answer would be a model failure and is counted as
    one rather than dropped.
    """

    def __init__(self, cfg: PredictorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n_context = cfg.context_grid[0] * cfg.context_grid[1]
        n_target = cfg.target_grid[0] * cfg.target_grid[1]
        window = cfg.patch_size * cfg.patch_size

        self.feature_in = nn.Linear(cfg.feature_dim, cfg.d_model)
        # Two planes per depth window: the log values and their validity.
        self.depth_in = nn.Linear(2 * window, cfg.d_model)
        self.context_pos = nn.Parameter(torch.zeros(n_context, cfg.d_model))
        self.target_pos = nn.Parameter(torch.zeros(n_target, cfg.d_model))
        self.query_token = nn.Parameter(torch.zeros(n_target, cfg.d_model))
        self.camera_in = nn.Sequential(
            nn.Linear(cfg.camera_features, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.norm_context = nn.LayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.n_blocks))
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.feature_out = nn.Linear(cfg.d_model, cfg.feature_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (self.context_pos, self.target_pos, self.query_token):
            nn.init.normal_(p, std=0.02)
        nn.init.zeros_(self.feature_out.bias)
        nn.init.normal_(self.feature_out.weight, std=0.02 / math.sqrt(self.cfg.n_blocks))

    def forward(
        self,
        features_context: Tensor,
        depth_context_aligned: Tensor,
        camera: Tensor,
        context_valid: Tensor | None = None,
    ) -> Tensor:
        """
        features_context: [B, N_context, feature_dim] frozen DINOv2 patch features.
        depth_context_aligned: [B, H, W] aligned context depth in meters.
        camera: [B, camera_features] from camera_vector.
        context_valid: optional [B, N_context] bool, False marks a context token
            the attention must ignore entirely.

        Every argument is context side or camera side. There is deliberately no
        parameter through which target content could arrive.
        """
        if features_context.dim() != 3:
            raise ValueError("features_context must be [B, N_context, C]")
        batch = features_context.shape[0]

        log_depth, depth_valid = patch_depth_windows(depth_context_aligned, self.cfg.patch_size)
        if log_depth.shape[1] != features_context.shape[1]:
            raise ValueError(
                f"depth gives {log_depth.shape[1]} patches, features give "
                f"{features_context.shape[1]}"
            )
        camera_token = self.camera_in(camera)

        context = (
            self.feature_in(features_context)
            + self.depth_in(torch.cat([log_depth, depth_valid], dim=-1))
            + self.context_pos[None]
            + camera_token[:, None, :]
        )
        context = self.norm_context(context)

        # nn.MultiheadAttention takes True to mean "ignore this key".
        context_pad = None if context_valid is None else ~context_valid

        queries = (
            self.query_token[None].expand(batch, -1, -1)
            + self.target_pos[None]
            + camera_token[:, None, :]
        )
        for block in self.blocks:
            queries = block(queries, context, context_pad)
        return self.feature_out(self.norm_out(queries))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_predictor(cfg: PredictorConfig) -> PredictWithDepth:
    return PredictWithDepth(cfg)
