"""Frozen encoders, the feature cache, and feature-grid sampling.

The pixel-to-patch coordinate mapping is defined here once for the whole
repository: pixel (u, v) maps to patch coordinates ((u + 0.5) / patch_size - 0.5,
same for v). Integer patch coordinates are patch centers. The center of patch p
sits at pixel coordinate patch_size * p + (patch_size - 1) / 2.

Phase 2 adds the encoder wrappers and the cache. Both encoders are frozen and
are only ever run under inference mode. Rendered frames are already a whole
number of patches on each side, so nothing here resizes an image: a resize would
change the pixel-to-patch mapping above and silently invalidate every warp
computed against the manifest intrinsics.

The heavy dependencies are imported lazily inside the loaders, so this module
imports on any machine and everything except the two model wrappers is testable
without weights, network, or GPU. That is the same split render_replica.py uses
for Habitat.

Cache layout, one directory per encoder and scene:
    cache_root/{encoder}/{scene}/features.npz   one fp16 [C, Hp, Wp] per frame id
    cache_root/{encoder}/{scene}/depth.npz      estimated depth, encoders that have it
    cache_root/{encoder}/{scene}/meta.json      spec, grid, frame ids, throughput
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Sequence

import numpy as np
import torch
from torch import Tensor

from .geometry import common_dtype

PATCH_SIZE = 14


def pixel_to_patch_coords(uv_px: Tensor, patch_size: int = PATCH_SIZE) -> Tensor:
    """Map continuous pixel coordinates to continuous patch-grid coordinates.

    Pixel (u, v) maps to ((u + 0.5) / patch_size - 0.5, (v + 0.5) / patch_size - 0.5).
    """
    return (uv_px + 0.5) / patch_size - 0.5


def patch_cell_index(
    uv_px: Tensor, hw_px: tuple[int, int], patch_size: int = PATCH_SIZE
) -> np.ndarray:
    """Row-major patch-grid cell of each pixel coordinate. [N] int64.

    Rounds the patch coordinate this module defines above, so which cell a
    location belongs to and where that location samples the patch grid are two
    readings of one mapping. CLAUDE.md keeps that mapping in this module alone;
    it had been rewritten inline in correspondence.py and again in evaluate.py,
    which left sample identity and cross-path cell assignment free to drift onto
    different conventions the next time the mapping is touched.
    """
    if uv_px.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    patches_w = hw_px[1] // patch_size
    patch = pixel_to_patch_coords(uv_px, patch_size)
    cols = torch.round(patch[:, 0]).long()
    rows = torch.round(patch[:, 1]).long()
    return (rows * patches_w + cols).cpu().numpy()


def patch_to_pixel_coords(uv_patch: Tensor, patch_size: int = PATCH_SIZE) -> Tensor:
    """Inverse of pixel_to_patch_coords."""
    return (uv_patch + 0.5) * patch_size - 0.5


def sample_map_bilinear(grid: Tensor, xy: Tensor) -> Tensor:
    """Bilinear sampling on a regular grid whose cell centers sit at integer coordinates.

    grid: [H, W] or [C, H, W].
    xy: [..., 2] continuous coordinates (x along width, y along height), in grid units.
    Coordinates outside the grid are clamped to the border.
    Returns [...] for a 2D grid and [..., C] for a 3D grid.
    """
    squeeze = grid.dim() == 2
    g = grid[None] if squeeze else grid
    if g.dim() != 3:
        raise ValueError(f"grid must be [H, W] or [C, H, W], got {tuple(grid.shape)}")
    channels, height, width = g.shape
    dtype = common_dtype(g, xy)
    g = g.to(dtype)
    x = xy[..., 0].to(dtype).clamp(0, width - 1)
    y = xy[..., 1].to(dtype).clamp(0, height - 1)
    x0 = x.floor().clamp(max=max(width - 2, 0)).long()
    y0 = y.floor().clamp(max=max(height - 2, 0)).long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    wx = x - x0
    wy = y - y0
    v00 = g[:, y0, x0]
    v01 = g[:, y0, x1]
    v10 = g[:, y1, x0]
    v11 = g[:, y1, x1]
    out = (
        v00 * (1 - wx) * (1 - wy)
        + v01 * wx * (1 - wy)
        + v10 * (1 - wx) * wy
        + v11 * wx * wy
    )
    out = torch.movedim(out, 0, -1)
    return out[..., 0] if squeeze else out


def sample_map_nearest(grid: Tensor, xy: Tensor) -> Tensor:
    """Nearest-cell sampling on a regular grid whose cell centers sit at integer coordinates.

    grid: [H, W] or [C, H, W].
    xy: [..., 2] continuous coordinates (x along width, y along height), in grid units.
    Must be finite. A coordinate is assigned to the cell whose extent contains it,
    by the same floor(x + 0.5) rule the transport splat uses, so the two agree on
    which cell a continuous location belongs to. Coordinates outside the grid are
    clamped to the border.
    Returns [...] for a 2D grid and [..., C] for a 3D grid.

    Nearest sampling reads a value the grid actually holds. Bilinear sampling of a
    depth map does not: across a depth edge it returns a depth that lies on neither
    surface. See visibility.py for why that distinction is load bearing there.
    """
    squeeze = grid.dim() == 2
    g = grid[None] if squeeze else grid
    if g.dim() != 3:
        raise ValueError(f"grid must be [H, W] or [C, H, W], got {tuple(grid.shape)}")
    _, height, width = g.shape
    x = torch.floor(xy[..., 0] + 0.5).long().clamp(0, width - 1)
    y = torch.floor(xy[..., 1] + 0.5).long().clamp(0, height - 1)
    out = torch.movedim(g[:, y, x], 0, -1)
    return out[..., 0] if squeeze else out


def sample_features_bilinear(features: Tensor, uv_px: Tensor, patch_size: int = PATCH_SIZE) -> Tensor:
    """Sample a patch-grid feature map at continuous pixel coordinates.

    features: [C, Hp, Wp] patch-grid feature map.
    uv_px: [..., 2] pixel coordinates in the image the features were computed from.
    Uses the pixel-to-patch mapping defined above, then bilinear interpolation on
    the patch grid with border clamping.
    Returns [..., C].
    """
    return sample_map_bilinear(features, pixel_to_patch_coords(uv_px, patch_size))


def patch_grid_shape(hw_px: tuple[int, int], patch_size: int = PATCH_SIZE) -> tuple[int, int]:
    """Patch-grid shape (Hp, Wp) for an image size (H, W) in pixels.

    Both sides must be whole patches. Nothing in this project resizes or crops a
    rendered frame, so a size that does not divide is a configuration error.
    """
    height, width = hw_px
    if height <= 0 or width <= 0:
        raise ValueError(f"image size must be positive, got {hw_px}")
    if height % patch_size or width % patch_size:
        raise ValueError(f"image size {hw_px} is not a whole number of {patch_size} px patches")
    return height // patch_size, width // patch_size


# ---------------------------------------------------------------------------
# Frozen encoders
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclasses.dataclass(frozen=True)
class EncoderSpec:
    """Static description of a frozen encoder.

    channels is None when the width of the feature vector is only known once the
    model is loaded. The cache records the value that was actually produced, so a
    silent change of model version shows up as a cache validation failure rather
    than as drifting numbers.
    """

    name: str
    patch_size: int
    normalization: str  # "imagenet" or "unit"
    provides_depth: bool
    source: str  # torch.hub entry point, or pretrained identifier
    channels: int | None = None


ENCODERS: dict[str, EncoderSpec] = {
    "dinov2_vitb14": EncoderSpec(
        name="dinov2_vitb14",
        patch_size=14,
        normalization="imagenet",
        provides_depth=False,
        source="dinov2_vitb14",
        channels=768,
    ),
    # The same backbone trained with register tokens, which the DINOv2 authors
    # added to absorb high-norm outlier tokens. Registered here so Phase 3 can
    # compare it directly if outlier tokens turn out to matter for the value
    # agreement metrics. Both variants cache identically.
    "dinov2_vitb14_reg": EncoderSpec(
        name="dinov2_vitb14_reg",
        patch_size=14,
        normalization="imagenet",
        provides_depth=False,
        source="dinov2_vitb14_reg",
        channels=768,
    ),
    "vggt_1b": EncoderSpec(
        name="vggt_1b",
        patch_size=14,
        normalization="unit",
        provides_depth=True,
        source="facebook/VGGT-1B",
        channels=None,
    ),
}


class EncodedBatch(NamedTuple):
    features: Tensor        # [B, C, Hp, Wp] float32
    depth: Tensor | None    # [B, H, W] float32 estimated planar z-depth, encoder units
    depth_conf: Tensor | None  # [B, H, W] float32 confidence, or None


def preprocess_images(
    images_uint8: np.ndarray, spec: EncoderSpec, device: torch.device | str | None = None
) -> Tensor:
    """Turn rendered RGB frames into an encoder input tensor.

    images_uint8: [B, H, W, 3] uint8 as written by the renderer.
    device: where to build the result. The frames are moved while still uint8 and
        converted there, which transfers a quarter of the bytes and keeps the
        scaling off the CPU. That is the difference between being bound by the
        host and being bound by the model.
    Returns [B, 3, H, W] float32 normalized as the encoder expects. The image is
    never resized or cropped, so the patch grid matches the manifest intrinsics.
    """
    arr = np.asarray(images_uint8)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"images must be [B, H, W, 3], got {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"images must be uint8, got {arr.dtype}")
    patch_grid_shape(arr.shape[1:3], spec.patch_size)
    if spec.normalization not in ("imagenet", "unit"):
        raise ValueError(f"unknown normalization {spec.normalization!r}")
    x = torch.from_numpy(np.ascontiguousarray(arr))
    if device is not None:
        x = x.to(device)
    x = x.permute(0, 3, 1, 2).contiguous().to(torch.float32).div_(255.0)
    if spec.normalization == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
    return x


def model_fingerprint(model: Any) -> str:
    """A hash of every frozen parameter, recorded beside the cache it produced.

    Torch Hub resolves a mutable branch head and the VGGT loader takes no
    revision, so a later rebuild can pull different weights while every version
    string in the metadata stays the same. Pinning the revisions is the real
    fix and needs the exact identifiers; this makes the change detectable
    meanwhile, which is the difference between a cache that is wrong and a cache
    that is wrong and silent.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(
            tensor.detach().to(torch.float32).cpu().numpy().tobytes(order="C")
        )
    return digest.hexdigest()


class FrozenEncoder:
    """Common behaviour of the frozen encoder wrappers.

    Subclasses load their model lazily in _load and implement _forward. The model
    is put in eval mode with gradients disabled, and every call runs under
    inference mode, so nothing here can train or be trained.
    """

    def __init__(self, spec: EncoderSpec, device: str = "cpu") -> None:
        self.spec = spec
        self.device = torch.device(device)
        self._model: Any = None
        self._channels: int | None = spec.channels
        self._fingerprint: str | None = None

    @property
    def channels(self) -> int | None:
        """Feature width. None until the first batch, for models that declare none."""
        return self._channels

    @property
    def fingerprint(self) -> str:
        """Hash of the loaded weights. Loads the model if it is not loaded yet."""
        if self._fingerprint is None:
            self._fingerprint = model_fingerprint(self.model)
        return self._fingerprint

    @property
    def model(self) -> Any:
        if self._model is None:
            model = self._load()
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._model = model.to(self.device)
        return self._model

    def _load(self) -> Any:
        raise NotImplementedError

    def _forward(self, images_uint8: np.ndarray) -> EncodedBatch:
        raise NotImplementedError

    def encode(self, images_uint8: np.ndarray) -> EncodedBatch:
        """Encode a batch of [B, H, W, 3] uint8 frames."""
        with torch.inference_mode():
            batch = self._forward(images_uint8)
        self._check(batch, images_uint8.shape)
        return batch

    def _check(self, batch: EncodedBatch, images_shape: tuple[int, ...]) -> None:
        count, height, width, _ = images_shape
        patches_h, patches_w = patch_grid_shape((height, width), self.spec.patch_size)
        expected = (count, patches_h, patches_w)
        got = (batch.features.shape[0], batch.features.shape[2], batch.features.shape[3])
        if batch.features.dim() != 4 or got != expected:
            raise RuntimeError(
                f"{self.spec.name} returned features of shape "
                f"{tuple(batch.features.shape)}, expected [B, C, Hp, Wp] with "
                f"B, Hp, Wp = {expected}"
            )
        channels = int(batch.features.shape[1])
        if self._channels is None:
            self._channels = channels
        elif channels != self._channels:
            raise RuntimeError(
                f"{self.spec.name} returned {channels} channels, expected "
                f"{self._channels}. The model version changed under the cache."
            )


class DinoV2Encoder(FrozenEncoder):
    """DINOv2 ViT-B/14 from torch.hub, patch tokens only.

    forward_features returns the final normalized patch tokens as
    x_norm_patchtokens, shaped [B, Hp * Wp, C] in row-major patch order. That is
    the only output this project uses; the class token is not a scene property.
    """

    HUB_REPO = "facebookresearch/dinov2"

    def _load(self) -> Any:
        # LOT_DINOV2_REVISION pins the Torch Hub ref, which is a code revision.
        # It fixes the hubconf that chooses the checkpoint URL; it does not fix
        # the bytes served at that URL, because dl.fbaipublicfiles.com serves an
        # unversioned file. So the ref is recorded as code_revision, and
        # weights_revision says plainly that there is nothing to pin. Recording
        # the ref under weights_revision, which an earlier version did, told a
        # reader the checkpoint was retrievable when only the loader was.
        revision = os.environ.get("LOT_DINOV2_REVISION")
        repo = f"{self.HUB_REPO}:{revision}" if revision else self.HUB_REPO
        self.code_revision = revision or "unpinned"
        self.revision = "unpinnable: torch.hub checkpoint URL is unversioned"
        return torch.hub.load(repo, self.spec.source, trust_repo=True)

    def _forward(self, images_uint8: np.ndarray) -> EncodedBatch:
        x = preprocess_images(images_uint8, self.spec, self.device)
        out = self.model.forward_features(x)
        if "x_norm_patchtokens" not in out:
            raise RuntimeError(
                "DINOv2 forward_features did not return x_norm_patchtokens; got "
                f"{sorted(out)}"
            )
        tokens = out["x_norm_patchtokens"]
        count, num_tokens, channels = tokens.shape
        patches_h, patches_w = patch_grid_shape(images_uint8.shape[1:3], self.spec.patch_size)
        if num_tokens != patches_h * patches_w:
            raise RuntimeError(
                f"DINOv2 returned {num_tokens} patch tokens for a "
                f"{patches_h} by {patches_w} grid"
            )
        features = tokens.reshape(count, patches_h, patches_w, channels)
        features = features.permute(0, 3, 1, 2).to(torch.float32).contiguous()
        return EncodedBatch(features=features, depth=None, depth_conf=None)


class VggtEncoder(FrozenEncoder):
    """VGGT, patch tokens from the aggregator and its estimated depth.

    VGGT is a multi-view model. This project feeds it one view at a time, which
    is what Phase 4 needs: depth estimated from the context view alone, with no
    look at the target. The aggregator is called with a sequence of length one.

    The estimated depth is in VGGT's own scale, not meters. Phase 4 is
    responsible for aligning it to the ground-truth scale before it is used as a
    transport input, and for reporting how that alignment was done.
    """

    def _load(self) -> Any:
        try:
            from vggt.models.vggt import VGGT
        except ImportError as e:
            raise RuntimeError(
                "vggt is not installed. Install it per scripts/README.md and run "
                "on a machine with the weights cached. Every other part of this "
                "module works without it."
            ) from e
        # LOT_VGGT_REVISION pins the Hugging Face revision, for the same reason.
        revision = os.environ.get("LOT_VGGT_REVISION")
        self.revision = revision or "unpinned"
        # And the implementation that will run those weights, which the runbook
        # installs from a git branch and which decides what the features are.
        self.code_revision = package_revision("vggt")
        if revision:
            return VGGT.from_pretrained(self.spec.source, revision=revision)
        return VGGT.from_pretrained(self.spec.source)

    def _forward(self, images_uint8: np.ndarray) -> EncodedBatch:
        x = preprocess_images(images_uint8, self.spec, self.device)
        patches_h, patches_w = patch_grid_shape(images_uint8.shape[1:3], self.spec.patch_size)
        # VGGT takes [B, S, 3, H, W] with S the number of views. One view each.
        views = x[:, None]
        # The model's own forward runs the aggregator and then the heads. Take
        # the aggregator output as it passes rather than calling the aggregator
        # separately, which would run the billion parameter trunk twice per
        # batch. Capturing it in place also guarantees the tokens are exactly
        # the ones the depth head consumed, whatever context the model wraps
        # its trunk in.
        captured: dict[str, Any] = {}

        def capture(_module: Any, _inputs: Any, outputs: Any) -> None:
            captured["outputs"] = outputs

        handle = self.model.aggregator.register_forward_hook(capture)
        try:
            predictions = self.model(views)
        finally:
            handle.remove()
        if "outputs" not in captured:
            raise RuntimeError(
                "VGGT forward did not call its aggregator, so the patch tokens "
                "could not be captured. Check the installed version."
            )
        tokens_per_layer, patch_start = captured["outputs"]
        tokens = tokens_per_layer[-1][:, 0, patch_start:, :]
        count, num_tokens, channels = tokens.shape
        if num_tokens != patches_h * patches_w:
            raise RuntimeError(
                f"VGGT aggregator returned {num_tokens} patch tokens for a "
                f"{patches_h} by {patches_w} grid. Check the version and the "
                "patch start index."
            )
        features = tokens.reshape(count, patches_h, patches_w, channels)
        features = features.permute(0, 3, 1, 2).to(torch.float32).contiguous()

        depth = _squeeze_view_and_channel(predictions["depth"], "depth")
        conf = (
            _squeeze_view_and_channel(predictions["depth_conf"], "depth_conf")
            if "depth_conf" in predictions
            else None
        )
        return EncodedBatch(
            features=features,
            depth=depth.to(torch.float32),
            depth_conf=None if conf is None else conf.to(torch.float32),
        )


def _squeeze_view_and_channel(tensor: Tensor, name: str) -> Tensor:
    """Reduce a VGGT prediction to [B, H, W].

    VGGT returns [B, S, H, W] or [B, S, H, W, 1] depending on the head. The view
    axis is length one here because this project encodes one view at a time.
    """
    out = tensor
    if out.dim() == 5 and out.shape[-1] == 1:
        out = out[..., 0]
    if out.dim() != 4 or out.shape[1] != 1:
        raise RuntimeError(f"VGGT {name} had shape {tuple(tensor.shape)}, expected [B, 1, H, W]")
    return out[:, 0]


def load_encoder(name: str, device: str = "cpu") -> FrozenEncoder:
    """Build a frozen encoder wrapper by name. Weights load on first use."""
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; known: {sorted(ENCODERS)}")
    spec = ENCODERS[name]
    if name.startswith("dinov2_"):
        return DinoV2Encoder(spec, device)
    if name.startswith("vggt_"):
        return VggtEncoder(spec, device)
    raise ValueError(f"no wrapper registered for {name!r}")


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------

CACHE_VERSION = 2
# 2: meta records features_digest, a content hash of the stored feature arrays,
#    and weights_fingerprint is required rather than merely recorded. A cache
#    written under 1 carries neither, so nothing downstream can tell whether it
#    came from the weights this run believes it did.
FEATURES_NAME = "features.npz"
DEPTH_NAME = "depth.npz"
CACHE_META_NAME = "meta.json"


def cache_dir(cache_root: Path, encoder_name: str, scene: str) -> Path:
    """Directory holding one scene's cache for one encoder."""
    return Path(cache_root) / encoder_name / scene


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an npz through a temporary file, then rename it into place.

    A half-written cache that still carries the final name would pass a later
    existence check and be read as complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def cache_scene_features(
    manifest: Any,
    scene_root: Path,
    encoder: FrozenEncoder,
    cache_root: Path,
    batch_size: int = 8,
    export_depth: bool = False,
) -> dict[str, Any]:
    """Encode every frame of one scene and write the cache. Returns the metadata.

    manifest: a render_replica.Manifest. Frames are encoded in manifest order.
    scene_root: the directory the manifest's relative rgb paths resolve against.
    Refuses to run when the cache metadata already exists, so a rerun never
    overwrites a finished cache.
    """
    from PIL import Image

    if not manifest.frames:
        raise ValueError(f"manifest for {manifest.scene} has no frames")
    out_dir = cache_dir(cache_root, encoder.spec.name, manifest.scene)
    meta_path = out_dir / CACHE_META_NAME
    if meta_path.exists():
        raise FileExistsError(
            f"{meta_path} exists; caches are never overwritten. Delete the "
            "directory to re-encode."
        )
    if export_depth and not encoder.spec.provides_depth:
        raise ValueError(f"{encoder.spec.name} does not produce depth")
    scene_root = Path(scene_root)

    features: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    encode_seconds = 0.0
    started = time.perf_counter()
    for batch in _batched(manifest.frames, batch_size):
        images = np.stack(
            [np.asarray(Image.open(scene_root / f.rgb_path))[..., :3] for f in batch]
        )
        mark = time.perf_counter()
        out = encoder.encode(images)
        if encoder.device.type == "cuda":
            torch.cuda.synchronize()
        encode_seconds += time.perf_counter() - mark
        arrays = out.features.to(torch.float16).cpu().numpy()
        for frame, array in zip(batch, arrays):
            features[frame.frame_id] = array
        if export_depth:
            if out.depth is None:
                raise RuntimeError(f"{encoder.spec.name} returned no depth")
            depth_arrays = out.depth.to(torch.float16).cpu().numpy()
            conf_arrays = (
                out.depth_conf.to(torch.float16).cpu().numpy()
                if out.depth_conf is not None
                else None
            )
            for i, frame in enumerate(batch):
                depths[frame.frame_id] = depth_arrays[i]
                if conf_arrays is not None:
                    depths[f"{frame.frame_id}__conf"] = conf_arrays[i]
    total_seconds = time.perf_counter() - started

    first = manifest.frames[0]
    patches_h, patches_w = patch_grid_shape(
        (first.height, first.width), encoder.spec.patch_size
    )
    meta = {
        "cache_version": CACHE_VERSION,
        "encoder": encoder.spec.name,
        "scene": manifest.scene,
        "channels": int(encoder.channels),
        "patch_size": encoder.spec.patch_size,
        "patch_grid": [patches_h, patches_w],
        "image_hw": [first.height, first.width],
        "dtype": "float16",
        "frame_count": len(manifest.frames),
        "frame_ids": [f.frame_id for f in manifest.frames],
        "has_depth": bool(export_depth),
        "batch_size": batch_size,
        "device": str(encoder.device),
        "source": encoder.spec.source,
        "weights_fingerprint": encoder.fingerprint,
        "weights_revision": getattr(encoder, "revision", "unpinned"),
        "code_revision": getattr(encoder, "code_revision", "n/a"),
        "features_digest": features_digest(features),
        "depth_digest": features_digest(depths) if export_depth else None,
        "encode_seconds": round(encode_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "frames_per_second": round(len(manifest.frames) / max(total_seconds, 1e-9), 3),
        "torch_version": torch.__version__,
    }
    _write_npz(out_dir / FEATURES_NAME, features)
    if export_depth:
        _write_npz(out_dir / DEPTH_NAME, depths)
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def package_revision(name: str) -> str:
    """The commit an installed package was built from, or its version.

    VGGT's inference implementation is a third artifact beside its weights and
    the analysis code, and the runbook installs it from a git branch. The same
    state dict run through different inference code produces different features,
    so a weights fingerprint alone does not identify what made a cache.
    """
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib here
        return "unknown"
    try:
        direct = metadata.distribution(name).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return "absent"
    except OSError:
        direct = None
    if direct:
        try:
            info = json.loads(direct).get("vcs_info") or {}
        except ValueError:
            info = {}
        commit = info.get("commit_id")
        if commit:
            return commit
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "absent"


def features_digest(features: dict[str, np.ndarray]) -> str:
    """Content hash of a scene's feature arrays, over frame ids and bytes.

    Computed once when the cache is written and stored in its metadata, so
    everything downstream can check what it is reading without re-reading it.
    The alternative, hashing at every use, would make the mean-vector cache
    pointless: it exists so an 18-task array does not read the whole feature
    cache eighteen times.

    Frame ids are hashed alongside the bytes because a cache with the right
    arrays under the wrong names is not the same cache.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(features):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(features[name]).tobytes())
    return digest.hexdigest()


def _batched(items: list, size: int) -> Iterator[list]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_cache_meta(cache_root: Path, encoder_name: str, scene: str) -> dict[str, Any]:
    """Read one scene's cache metadata. Raises ValueError on a version mismatch."""
    path = cache_dir(cache_root, encoder_name, scene) / CACHE_META_NAME
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    if meta.get("cache_version") != CACHE_VERSION:
        raise ValueError(f"cache version {meta.get('cache_version')}, expected {CACHE_VERSION}")
    return meta


def load_cached_features(cache_root: Path, encoder_name: str, scene: str, frame_id: str) -> Tensor:
    """Read one frame's cached features as a [C, Hp, Wp] float16 tensor.

    Opens the scene archive per call. Callers that read many frames of one scene
    should hold the archive open with numpy directly instead.
    """
    path = cache_dir(cache_root, encoder_name, scene) / FEATURES_NAME
    with np.load(path) as archive:
        if frame_id not in archive:
            raise KeyError(f"{frame_id} not in {path}")
        return torch.from_numpy(archive[frame_id])


def load_cached_depth(cache_root: Path, encoder_name: str, scene: str, frame_id: str) -> Tensor:
    """Read one frame's estimated depth as an [H, W] float16 tensor, encoder units."""
    path = cache_dir(cache_root, encoder_name, scene) / DEPTH_NAME
    with np.load(path) as archive:
        if frame_id not in archive:
            raise KeyError(f"{frame_id} not in {path}")
        return torch.from_numpy(archive[frame_id])


def validate_feature_cache(
    cache_root: Path,
    encoder_name: str,
    manifest: Any,
    check_digest: bool = False,
) -> dict[str, Any]:
    """Check a scene cache against the manifest it was built from.

    Raises ValueError with the first problem found. Returns the metadata.

    The provenance fields are required, not merely read when present. Recording
    a weights fingerprint that nothing compares against documents a cache
    without validating it, and a cache from a silently updated checkpoint would
    still be accepted.

    check_digest re-reads every array, features and depth alike, and recomputes
    both content hashes. It is off by default because it costs a full pass over
    the cache, and on for the validation entrypoint, whose job is exactly that
    pass.
    """
    meta = load_cache_meta(cache_root, encoder_name, manifest.scene)
    if meta["encoder"] != encoder_name or meta["scene"] != manifest.scene:
        raise ValueError(f"cache is for {meta['encoder']} / {meta['scene']}")
    for field in (
        "weights_fingerprint", "features_digest", "weights_revision", "code_revision"
    ):
        if not meta.get(field):
            raise ValueError(
                f"{encoder_name} / {manifest.scene}: cache metadata carries no "
                f"{field}. Re-cache this scene; a cache whose provenance is "
                "unknown cannot be told apart from one built with other weights"
            )
    manifest_ids = [f.frame_id for f in manifest.frames]
    if meta["frame_ids"] != manifest_ids:
        missing = sorted(set(manifest_ids) - set(meta["frame_ids"]))
        raise ValueError(
            f"cache covers {len(meta['frame_ids'])} frames, manifest has "
            f"{len(manifest_ids)}; first missing: {missing[:3]}"
        )
    # The expected shape comes from the manifest and the registered encoder
    # spec, never from the cache's own metadata. A cache that declares its own
    # dimensions and is checked against them agrees with itself whatever it
    # holds: a 1 by 1 grid claimed for a 28 by 28 frame passed, because nothing
    # asked what shape the frame and the encoder imply.
    spec = ENCODERS.get(encoder_name)
    first = manifest.frames[0]
    # The grid the manifest and the declared patch size imply. This holds even
    # for an encoder that is not in the registry, because it ties the declared
    # grid to the frame size rather than to another field of the same file.
    declared_grid = patch_grid_shape((first.height, first.width), int(meta["patch_size"]))
    if tuple(int(v) for v in meta["patch_grid"]) != tuple(declared_grid):
        raise ValueError(
            f"{encoder_name} / {manifest.scene}: cache declares patch grid "
            f"{list(meta['patch_grid'])}, but {first.height}x{first.width} at "
            f"patch size {meta['patch_size']} is {list(declared_grid)}"
        )
    if tuple(int(v) for v in meta["image_hw"]) != (first.height, first.width):
        raise ValueError(
            f"{encoder_name} / {manifest.scene}: cache declares image "
            f"{list(meta['image_hw'])}, manifest says {[first.height, first.width]}"
        )
    for frame in manifest.frames:
        if (frame.height, frame.width) != (first.height, first.width):
            raise ValueError(
                f"{frame.frame_id}: frame is {frame.height}x{frame.width}, "
                f"{first.frame_id} is {first.height}x{first.width}; one cache "
                "holds one image size"
            )
    if spec is not None:
        # And for a registered encoder, the patch size and channel count are
        # facts about the encoder rather than claims of the file.
        if int(meta["patch_size"]) != int(spec.patch_size):
            raise ValueError(
                f"{encoder_name} / {manifest.scene}: cache declares patch size "
                f"{meta['patch_size']}, {encoder_name} has {spec.patch_size}"
            )
        # VGGT's width is not known before the model runs, so its spec declares
        # None and there is nothing to compare against. Guarding on that rather
        # than coercing it: int(None) would have raised on the real cache.
        if spec.channels is not None and int(meta["channels"]) != int(spec.channels):
            raise ValueError(
                f"{encoder_name} / {manifest.scene}: cache declares "
                f"{meta['channels']} channels, {encoder_name} produces {spec.channels}"
            )
    expected = (int(meta["channels"]), *declared_grid)
    path = cache_dir(cache_root, encoder_name, manifest.scene) / FEATURES_NAME
    with np.load(path) as archive:
        stored = set(archive.files)
        # Extra keys are refused, not ignored. dataset_mean_vector averages
        # every array in the archive, so one stray key moves the Mean-Feature
        # floor and the centering statistic together, and a check that only
        # looks for what it expects cannot see it.
        extra = sorted(stored - set(manifest_ids))
        if extra:
            raise ValueError(
                f"{encoder_name} / {manifest.scene}: {len(extra)} feature arrays "
                f"the manifest does not name, first {extra[:3]}. Everything in "
                "this archive is averaged into the global mean vector"
            )
        for frame_id in manifest_ids:
            if frame_id not in stored:
                raise ValueError(f"{frame_id}: missing from {path}")
            array = archive[frame_id]
            if array.shape != expected:
                raise ValueError(f"{frame_id}: features {array.shape}, expected {expected}")
            if array.dtype != np.float16:
                raise ValueError(f"{frame_id}: features dtype {array.dtype}, expected float16")
    if check_digest:
        with np.load(path) as archive:
            stored_arrays = {name: archive[name] for name in archive.files}
        found = features_digest(stored_arrays)
        if found != meta["features_digest"]:
            raise ValueError(
                f"{encoder_name} / {manifest.scene}: features digest {found} does "
                f"not match the {meta['features_digest']} recorded when the cache "
                "was written. The archive has been rebuilt or modified in place"
            )
    if meta["has_depth"]:
        # Depth and confidence are hashed too. Frame presence and shape do not
        # protect content: every value in a depth archive can change while the
        # keys and shapes stay right. That is inert for Experiment Zero, whose
        # geometry is ground truth, and it is a Phase 4 blocker, because VGGT
        # depth and its confidence become scientific inputs there.
        if not meta.get("depth_digest"):
            raise ValueError(
                f"{encoder_name} / {manifest.scene}: cache declares depth and "
                "records no depth_digest. Re-cache this scene"
            )
        depth_path = cache_dir(cache_root, encoder_name, manifest.scene) / DEPTH_NAME
        image_hw = tuple(int(v) for v in meta["image_hw"])
        with np.load(depth_path) as archive:
            stored_depth = {name: archive[name] for name in archive.files}
        allowed = set(manifest_ids) | {f"{frame_id}__conf" for frame_id in manifest_ids}
        extra_depth = sorted(set(stored_depth) - allowed)
        if extra_depth:
            raise ValueError(
                f"{encoder_name} / {manifest.scene}: {len(extra_depth)} depth "
                f"arrays the manifest does not name, first {extra_depth[:3]}"
            )
        for frame_id in manifest_ids:
            if frame_id not in stored_depth:
                raise ValueError(f"{frame_id}: missing from {depth_path}")
            if stored_depth[frame_id].shape != image_hw:
                raise ValueError(
                    f"{frame_id}: depth {stored_depth[frame_id].shape}, expected {image_hw}"
                )
            if stored_depth[frame_id].dtype != np.float16:
                raise ValueError(
                    f"{frame_id}: depth dtype {stored_depth[frame_id].dtype}, expected float16"
                )
            # Confidence rides in the same archive under a suffixed key. It gates
            # which depths Phase 4 trusts, so an unchecked confidence map is an
            # unchecked depth map by another route.
            conf_key = f"{frame_id}__conf"
            if conf_key in stored_depth:
                conf = stored_depth[conf_key]
                if conf.shape != image_hw:
                    raise ValueError(f"{conf_key}: confidence {conf.shape}, expected {image_hw}")
                if conf.dtype != np.float16:
                    raise ValueError(f"{conf_key}: confidence dtype {conf.dtype}, expected float16")
        if check_digest:
            found = features_digest(stored_depth)
            if found != meta["depth_digest"]:
                raise ValueError(
                    f"{encoder_name} / {manifest.scene}: depth digest {found} does "
                    f"not match the {meta['depth_digest']} recorded when the cache "
                    "was written. The archive has been rebuilt or modified in place"
                )
    return meta


def feature_statistics(
    cache_root: Path,
    encoder_name: str,
    scenes: Sequence[str],
    max_frames: int = 64,
    samples: int = 20000,
    seed: int = 0,
) -> dict[str, Any]:
    """How much of a cached feature is shared by every patch.

    Cosine similarity only tells things apart when the features are spread over
    the sphere. If every patch points in nearly the same direction, every cosine
    is near one, the floors rise to meet any method, and the metric stops
    resolving anything, whatever the representation actually knows.

    Returns, for the raw features and again after subtracting the mean feature:
    shared_direction_fraction, the norm of the mean feature over the mean norm
    of a feature, which is near one when a single direction dominates;
    cosine_within_frame, between two random patches of one frame; and
    cosine_across_frames, between patches of different frames. The centered
    numbers say whether centering would restore the metric's range, which is a
    decision to take deliberately rather than a knob to turn until a number
    improves.
    """
    generator = torch.Generator().manual_seed(seed)
    maps: list[Tensor] = []
    for scene in scenes:
        path = cache_dir(cache_root, encoder_name, scene) / FEATURES_NAME
        with np.load(path) as archive:
            names = sorted(archive.files)[: max(1, max_frames // max(1, len(scenes)))]
            for name in names:
                maps.append(torch.from_numpy(archive[name]).to(torch.float32))
        if len(maps) >= max_frames:
            break
    if not maps:
        raise ValueError(f"no cached features for {encoder_name}")

    channels = maps[0].shape[0]
    per_frame = [m.reshape(channels, -1).T for m in maps]  # [P, C] each
    flat = torch.cat(per_frame, dim=0)
    mean = flat.mean(dim=0)

    def report(vectors: Tensor, frames: list[Tensor]) -> dict[str, float]:
        norms = vectors.norm(dim=1)
        unit = vectors / norms.clamp(min=1e-12)[:, None]
        within = []
        for frame in frames:
            count = frame.shape[0]
            a = torch.randint(count, (samples // len(frames) + 1,), generator=generator)
            b = torch.randint(count, (samples // len(frames) + 1,), generator=generator)
            f = frame / frame.norm(dim=1).clamp(min=1e-12)[:, None]
            within.append((f[a] * f[b]).sum(dim=1))
        total = vectors.shape[0]
        i = torch.randint(total, (samples,), generator=generator)
        j = torch.randint(total, (samples,), generator=generator)
        return {
            "shared_direction_fraction": float(
                vectors.mean(dim=0).norm() / norms.mean().clamp(min=1e-12)
            ),
            "cosine_within_frame": float(torch.cat(within).mean()),
            "cosine_across_frames": float((unit[i] * unit[j]).sum(dim=1).mean()),
        }

    raw = report(flat, per_frame)
    centered = report(flat - mean, [frame - mean for frame in per_frame])
    return {
        "encoder": encoder_name,
        "frames": len(maps),
        "channels": channels,
        "raw": raw,
        "centered": centered,
    }


# ---------------------------------------------------------------------------
# Config and entrypoint
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CacheConfig:
    """One feature-caching experiment. Loaded from a yaml file, one file per run."""

    renders_root: Path
    cache_root: Path
    scenes: list[str]
    encoder: str = "dinov2_vitb14"
    batch_size: int = 8
    device: str = "cuda"
    export_depth: bool = False

    def __post_init__(self) -> None:
        self.renders_root = Path(self.renders_root)
        self.cache_root = Path(self.cache_root)
        if not self.scenes:
            raise ValueError("config lists no scenes")
        if self.encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {self.encoder!r}; known: {sorted(ENCODERS)}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.export_depth and not ENCODERS[self.encoder].provides_depth:
            raise ValueError(f"{self.encoder} does not produce depth")


def load_cache_config(path: Path) -> CacheConfig:
    """Load a CacheConfig from yaml. Unknown keys are an error, not a warning."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    allowed = {f.name for f in dataclasses.fields(CacheConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    missing = [k for k in ("renders_root", "cache_root", "scenes") if k not in raw]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    return CacheConfig(**raw)


def cache_scene(cfg: CacheConfig, scene: str, encoder: FrozenEncoder) -> dict[str, Any]:
    """Cache one scene end to end and validate the result against its manifest."""
    from .render_replica import MANIFEST_NAME, load_manifest

    scene_root = cfg.renders_root / scene
    manifest = load_manifest(scene_root / MANIFEST_NAME)
    meta = cache_scene_features(
        manifest,
        scene_root,
        encoder,
        cfg.cache_root,
        batch_size=cfg.batch_size,
        export_depth=cfg.export_depth,
    )
    validate_feature_cache(cfg.cache_root, encoder.spec.name, manifest)
    print(
        f"[{scene}] {meta['frame_count']} frames, {meta['channels']} channels, "
        f"{meta['patch_grid'][0]}x{meta['patch_grid'][1]} grid, "
        f"{meta['frames_per_second']:.2f} frames/s"
    )
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cache frozen features for LT.")
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scene", type=str, help="cache a single scene by name")
    group.add_argument(
        "--scene-index",
        type=int,
        help="cache a single scene by index into the config scene list "
        "(for SLURM array jobs)",
    )
    parser.add_argument(
        "--list-scenes", action="store_true", help="print the scene list and exit"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip scenes that already have a cache instead of failing",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate existing caches instead of encoding",
    )
    parser.add_argument(
        "--feature-stats",
        action="store_true",
        help="report how far the cached features spread on the sphere",
    )
    args = parser.parse_args(argv)
    cfg = load_cache_config(args.config)
    if args.list_scenes:
        for i, scene in enumerate(cfg.scenes):
            print(i, scene)
        return
    if args.scene is not None:
        if args.scene not in cfg.scenes:
            raise SystemExit(f"scene {args.scene!r} not in config scene list")
        scenes = [args.scene]
    elif args.scene_index is not None:
        if not 0 <= args.scene_index < len(cfg.scenes):
            raise SystemExit(
                f"--scene-index {args.scene_index} outside 0..{len(cfg.scenes) - 1}"
            )
        scenes = [cfg.scenes[args.scene_index]]
    else:
        scenes = list(cfg.scenes)

    from .render_replica import MANIFEST_NAME, load_manifest

    if args.feature_stats:
        stats = feature_statistics(cfg.cache_root, cfg.encoder, scenes)
        print(f"{stats['encoder']}: {stats['frames']} frames, {stats['channels']} channels")
        print(f"{'':22s} {'shared dir':>11s} {'within frame':>13s} {'across frames':>14s}")
        for label in ("raw", "centered"):
            entry = stats[label]
            print(
                f"  {label:20s} {entry['shared_direction_fraction']:11.4f} "
                f"{entry['cosine_within_frame']:13.4f} {entry['cosine_across_frames']:14.4f}"
            )
        return
    if args.validate_only:
        failures = []
        fingerprints: dict[str, list[str]] = {}
        for scene in scenes:
            try:
                manifest = load_manifest(cfg.renders_root / scene / MANIFEST_NAME)
                meta = validate_feature_cache(
                    cfg.cache_root, cfg.encoder, manifest, check_digest=True
                )
            except (FileNotFoundError, KeyError) as e:
                print(f"[{scene}] MISSING: {e}")
                failures.append(scene)
            except ValueError as e:
                print(f"[{scene}] INVALID: {e}")
                failures.append(scene)
            else:
                fingerprints.setdefault(meta["weights_fingerprint"], []).append(scene)
                print(
                    f"[{scene}] cache valid, {meta['frame_count']} frames, "
                    f"weights {meta['weights_fingerprint'][:12]}, "
                    f"revision {meta.get('weights_revision', 'unpinned')}"
                )
        if len(fingerprints) > 1:
            # One encoder, one set of weights. Scenes cached from different
            # checkpoints are not one frozen representation, and every
            # cross-scene aggregate over them is a mixture.
            print("weights differ across scenes:")
            for fingerprint, scene_list in sorted(fingerprints.items()):
                print(f"  {fingerprint[:12]}  {len(scene_list)} scenes: {scene_list[:4]}")
            raise SystemExit(
                f"{len(fingerprints)} distinct weight fingerprints for {cfg.encoder}; "
                "re-cache the minority scenes"
            )
        if failures:
            raise SystemExit(
                f"{len(failures)} of {len(scenes)} scenes failed validation: "
                + ", ".join(failures)
            )
        print(f"all {len(scenes)} scenes valid")
        return

    encoder = load_encoder(cfg.encoder, cfg.device)
    total_frames = 0
    total_seconds = 0.0
    for scene in scenes:
        if args.resume and (
            cache_dir(cfg.cache_root, cfg.encoder, scene) / CACHE_META_NAME
        ).exists():
            print(f"[{scene}] cache exists, skipping")
            continue
        meta = cache_scene(cfg, scene, encoder)
        total_frames += meta["frame_count"]
        total_seconds += meta["total_seconds"]
    if total_frames:
        rate = total_frames / total_seconds
        print(
            f"{total_frames} frames in {total_seconds:.1f} s, {rate:.2f} frames/s. "
            f"At this rate the full 18-scene set is "
            f"{5136 / rate / 60:.1f} min for 5136 frames."
        )


if __name__ == "__main__":
    main()
