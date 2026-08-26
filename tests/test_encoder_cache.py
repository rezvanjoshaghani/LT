"""PLAN Phase 2: encoder wrappers and the feature cache.

The two model wrappers need weights, network, and a GPU, so they are exercised
by a smoke test gated on LOT_ENCODER_SMOKE, the same way the Habitat render is
gated. Everything else here runs anywhere against a stub encoder, so the cache
format, the metadata, and the validation rules are pinned without downloads.
"""

import dataclasses
import json
import os

from typing import Any

import numpy as np
import pytest
import torch

from lot.encoders import (
    CACHE_META_NAME,
    CACHE_VERSION,
    ENCODERS,
    FEATURES_NAME,
    CacheConfig,
    EncodedBatch,
    EncoderSpec,
    FrozenEncoder,
    cache_dir,
    cache_scene_features,
    load_cache_config,
    load_cached_depth,
    load_cached_features,
    load_encoder,
    patch_grid_shape,
    preprocess_images,
    validate_feature_cache,
)
from test_render_replica import fake_scene_dir

STUB_CHANNELS = 5


def stub_spec(provides_depth: bool = False) -> EncoderSpec:
    return EncoderSpec(
        name="stub_depth" if provides_depth else "stub",
        patch_size=14,
        normalization="unit",
        provides_depth=provides_depth,
        source="stub",
        channels=STUB_CHANNELS,
    )


class StubEncoder(FrozenEncoder):
    """Deterministic stand-in for a frozen encoder.

    Each channel of a patch holds the mean brightness of that patch's pixels, so
    a cached value can be checked against the source image by hand. It carries a
    parameterless module so the weight fingerprint the cache records is defined
    for it too, rather than being a hole only the stubs fall into.
    """

    def _load(self) -> Any:
        return torch.nn.Module()

    def _forward(self, images_uint8: np.ndarray) -> EncodedBatch:
        count, height, width, _ = images_uint8.shape
        patches_h, patches_w = patch_grid_shape((height, width), self.spec.patch_size)
        gray = torch.from_numpy(images_uint8.astype(np.float32)).mean(dim=-1)
        pooled = gray.reshape(
            count, patches_h, self.spec.patch_size, patches_w, self.spec.patch_size
        ).mean(dim=(2, 4))
        features = pooled[:, None].expand(count, STUB_CHANNELS, patches_h, patches_w)
        depth = None
        if self.spec.provides_depth:
            depth = gray / 100.0
        return EncodedBatch(features=features.contiguous(), depth=depth, depth_conf=None)


class BadShapeEncoder(FrozenEncoder):
    """Returns a transposed patch grid, which the base class must reject."""

    def _load(self) -> Any:
        return torch.nn.Module()

    def _forward(self, images_uint8: np.ndarray) -> EncodedBatch:
        count, height, width, _ = images_uint8.shape
        patches_h, patches_w = patch_grid_shape((height, width), self.spec.patch_size)
        features = torch.zeros((count, STUB_CHANNELS, patches_w + 1, patches_h))
        return EncodedBatch(features=features, depth=None, depth_conf=None)


# ---------------------------------------------------------------------------
# Grid and preprocessing
# ---------------------------------------------------------------------------

def test_patch_grid_shape():
    assert patch_grid_shape((518, 518)) == (37, 37)
    assert patch_grid_shape((28, 42)) == (2, 3)
    with pytest.raises(ValueError, match="whole number"):
        patch_grid_shape((518, 500))


def test_preprocess_normalizes_and_keeps_the_image_size():
    spec = ENCODERS["dinov2_vitb14"]
    images = np.zeros((2, 28, 28, 3), dtype=np.uint8)
    images[1] = 255
    x = preprocess_images(images, spec)
    assert x.shape == (2, 3, 28, 28)
    # Black and white map to the ImageNet-normalized extremes of each channel.
    assert torch.allclose(x[0, 0, 0, 0], torch.tensor(-0.485 / 0.229), atol=1e-6)
    assert torch.allclose(x[1, 0, 0, 0], torch.tensor((1 - 0.485) / 0.229), atol=1e-6)


def test_preprocess_rejects_sizes_that_are_not_whole_patches():
    """A resize would silently break the pixel-to-patch mapping this module owns."""
    with pytest.raises(ValueError, match="whole number"):
        preprocess_images(np.zeros((1, 30, 28, 3), dtype=np.uint8), ENCODERS["dinov2_vitb14"])
    with pytest.raises(ValueError, match="uint8"):
        preprocess_images(np.zeros((1, 28, 28, 3), dtype=np.float32), ENCODERS["dinov2_vitb14"])


def test_unit_normalization_leaves_the_zero_to_one_range():
    spec = ENCODERS["vggt_1b"]
    images = np.full((1, 28, 28, 3), 255, dtype=np.uint8)
    assert torch.allclose(preprocess_images(images, spec), torch.ones(1, 3, 28, 28))


def test_encoder_checks_the_shape_it_was_given_back():
    encoder = BadShapeEncoder(stub_spec(), "cpu")
    with pytest.raises(RuntimeError, match="expected"):
        encoder.encode(np.zeros((1, 28, 28, 3), dtype=np.uint8))


def test_load_encoder_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown encoder"):
        load_encoder("resnet50")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_cache_round_trip(tmp_path):
    root, manifest = fake_scene_dir(tmp_path)
    cache_root = tmp_path / "cache"
    encoder = StubEncoder(stub_spec(), "cpu")
    meta = cache_scene_features(manifest, root, encoder, cache_root, batch_size=2)

    assert meta["cache_version"] == CACHE_VERSION
    assert meta["channels"] == STUB_CHANNELS
    assert meta["patch_grid"] == [2, 2]
    assert meta["dtype"] == "float16"
    assert meta["frame_count"] == len(manifest.frames)
    assert meta["frames_per_second"] > 0
    assert not meta["has_depth"]

    validate_feature_cache(cache_root, "stub", manifest)
    for frame in manifest.frames:
        cached = load_cached_features(cache_root, "stub", manifest.scene, frame.frame_id)
        assert cached.shape == (STUB_CHANNELS, 2, 2)
        assert cached.dtype == torch.float16
    # The cached value is the patch mean brightness of the stored image.
    from PIL import Image

    first = manifest.frames[0]
    rgb = np.asarray(Image.open(root / first.rgb_path))[..., :3].astype(np.float32)
    expected = rgb.mean(axis=-1)[:14, :14].mean()
    cached = load_cached_features(cache_root, "stub", manifest.scene, first.frame_id)
    assert abs(float(cached[0, 0, 0]) - expected) < 0.5


def test_cache_refuses_to_overwrite(tmp_path):
    root, manifest = fake_scene_dir(tmp_path)
    cache_root = tmp_path / "cache"
    encoder = StubEncoder(stub_spec(), "cpu")
    cache_scene_features(manifest, root, encoder, cache_root)
    with pytest.raises(FileExistsError):
        cache_scene_features(manifest, root, encoder, cache_root)


def test_cache_leaves_no_partial_archive(tmp_path):
    """The archive is renamed into place, so a half-written file never has the final name."""
    root, manifest = fake_scene_dir(tmp_path)
    cache_root = tmp_path / "cache"
    cache_scene_features(manifest, root, StubEncoder(stub_spec(), "cpu"), cache_root)
    directory = cache_dir(cache_root, "stub", manifest.scene)
    assert not list(directory.glob("*.partial"))


def test_depth_export(tmp_path):
    root, manifest = fake_scene_dir(tmp_path)
    cache_root = tmp_path / "cache"
    encoder = StubEncoder(stub_spec(provides_depth=True), "cpu")
    meta = cache_scene_features(manifest, root, encoder, cache_root, export_depth=True)
    assert meta["has_depth"]
    validate_feature_cache(cache_root, "stub_depth", manifest)
    depth = load_cached_depth(cache_root, "stub_depth", manifest.scene, manifest.frames[0].frame_id)
    assert depth.shape == (28, 28)
    assert depth.dtype == torch.float16


def test_depth_export_refused_for_encoders_without_depth(tmp_path):
    root, manifest = fake_scene_dir(tmp_path)
    with pytest.raises(ValueError, match="does not produce depth"):
        cache_scene_features(
            manifest, root, StubEncoder(stub_spec(), "cpu"), tmp_path / "cache", export_depth=True
        )


def test_validation_catches_a_frame_missing_from_the_archive(tmp_path):
    root, manifest = fake_scene_dir(tmp_path)
    cache_root = tmp_path / "cache"
    cache_scene_features(manifest, root, StubEncoder(stub_spec(), "cpu"), cache_root)
    directory = cache_dir(cache_root, "stub", manifest.scene)
    with np.load(directory / FEATURES_NAME) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays.pop(manifest.frames[-1].frame_id)
    np.savez(directory / FEATURES_NAME, **arrays)
    with pytest.raises(ValueError, match="missing from"):
        validate_feature_cache(cache_root, "stub", manifest)


def test_validation_catches_a_manifest_that_grew(tmp_path):
    root, manifest = fake_scene_dir(tmp_path)
    cache_root = tmp_path / "cache"
    cache_scene_features(manifest, root, StubEncoder(stub_spec(), "cpu"), cache_root)
    grown = dataclasses.replace(manifest, frames=manifest.frames + manifest.frames[:1])
    with pytest.raises(ValueError, match="cache covers"):
        validate_feature_cache(cache_root, "stub", grown)


def test_validation_catches_a_version_bump(tmp_path):
    root, manifest = fake_scene_dir(tmp_path)
    cache_root = tmp_path / "cache"
    cache_scene_features(manifest, root, StubEncoder(stub_spec(), "cpu"), cache_root)
    path = cache_dir(cache_root, "stub", manifest.scene) / CACHE_META_NAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["cache_version"] = CACHE_VERSION + 1
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="cache version"):
        validate_feature_cache(cache_root, "stub", manifest)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_cache_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "renders_root: data/r\ncache_root: cache/f\nscenes: [room_0]\nbatch: 4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown config keys"):
        load_cache_config(path)


def test_cache_config_requires_a_known_encoder():
    with pytest.raises(ValueError, match="unknown encoder"):
        CacheConfig(renders_root="a", cache_root="b", scenes=["room_0"], encoder="clip")


def test_cache_config_refuses_depth_from_an_encoder_without_it():
    with pytest.raises(ValueError, match="does not produce depth"):
        CacheConfig(
            renders_root="a",
            cache_root="b",
            scenes=["room_0"],
            encoder="dinov2_vitb14",
            export_depth=True,
        )


def test_shipped_configs_load():
    from pathlib import Path

    for name in ("cache_features_pilot.yaml", "cache_features_all.yaml"):
        cfg = load_cache_config(Path(__file__).resolve().parents[1] / "configs" / name)
        assert cfg.encoder in ENCODERS
        assert cfg.scenes


# ---------------------------------------------------------------------------
# VGGT wrapper, against a stand-in with the same interface
# ---------------------------------------------------------------------------

VGGT_CHANNELS = 16
VGGT_PREFIX = 5  # register and camera tokens ahead of the patch tokens


class FakeAggregator(torch.nn.Module):
    """Returns per-layer tokens shaped the way VGGT's aggregator does."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.last_shape = None

    def forward(self, views):
        self.calls += 1
        self.last_shape = tuple(views.shape)
        count, seq, _, height, width = views.shape
        patches_h, patches_w = patch_grid_shape((height, width))
        total = VGGT_PREFIX + patches_h * patches_w
        tokens = torch.arange(
            count * seq * total * VGGT_CHANNELS, dtype=torch.float32
        ).reshape(count, seq, total, VGGT_CHANNELS)
        return [tokens, tokens], VGGT_PREFIX


class FakeVggtModel(torch.nn.Module):
    """Stand-in with VGGT's call shape: aggregator inside forward, dict out."""

    def __init__(self) -> None:
        super().__init__()
        self.aggregator = FakeAggregator()

    def forward(self, views):
        # VGGT runs its trunk inside forward and feeds the heads from it. The
        # wrapper takes the tokens as they pass, so the fake must do the same.
        self.aggregator(views)
        count, seq, _, height, width = views.shape
        return {
            "depth": torch.full((count, seq, height, width, 1), 2.0),
            "depth_conf": torch.full((count, seq, height, width), 0.5),
        }


def fake_vggt_encoder():
    from lot.encoders import VggtEncoder

    encoder = VggtEncoder(ENCODERS["vggt_1b"], "cpu")
    encoder._model = FakeVggtModel().eval()
    return encoder


def test_vggt_wrapper_shapes_and_patch_token_offset():
    """Pins the parts of the VGGT contract that no local weights can check.

    The prefix tokens must be dropped, the single view axis squeezed, and the
    remaining tokens laid out row-major on the patch grid.
    """
    encoder = fake_vggt_encoder()
    out = encoder.encode(np.zeros((2, 28, 42, 3), dtype=np.uint8))
    assert out.features.shape == (2, VGGT_CHANNELS, 2, 3)
    assert out.depth.shape == (2, 28, 42)
    assert out.depth_conf.shape == (2, 28, 42)
    assert encoder.channels == VGGT_CHANNELS
    # The first patch token is the one just past the prefix, not token zero.
    expected_first = float(VGGT_PREFIX * VGGT_CHANNELS)
    assert float(out.features[0, 0, 0, 0]) == expected_first


def test_vggt_runs_its_trunk_once_per_batch():
    """The aggregator is the billion parameter part; calling it twice doubles the job."""
    encoder = fake_vggt_encoder()
    encoder.encode(np.zeros((1, 28, 28, 3), dtype=np.uint8))
    assert encoder.model.aggregator.calls == 1


def test_vggt_wrapper_rejects_a_token_count_that_does_not_fit_the_grid():
    encoder = fake_vggt_encoder()
    encoder.model.aggregator.forward = lambda views: (
        [torch.zeros((1, 1, 3, VGGT_CHANNELS))],
        0,
    )
    with pytest.raises(RuntimeError, match="patch tokens"):
        encoder.encode(np.zeros((1, 28, 28, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Real weights, gated
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("LOT_ENCODER_SMOKE"),
    reason="set LOT_ENCODER_SMOKE=1 to download weights and run the encoder",
)
def test_dinov2_grid_orientation_and_shape():
    """The patch grid must carry image rows in dim 1 and image columns in dim 2.

    A transposed reshape would keep the shape and the channel count and would
    pass every other check in this file, while silently mirroring every warp.
    A bright stripe must therefore show up on the axis it was drawn on.
    """
    size = 518
    encoder = load_encoder("dinov2_vitb14", "cuda" if torch.cuda.is_available() else "cpu")
    for axis in ("row", "col"):
        image = np.zeros((1, size, size, 3), dtype=np.uint8)
        if axis == "row":
            image[0, 210:280] = 255
        else:
            image[0, :, 210:280] = 255
        features = encoder.encode(image).features
        assert features.shape == (1, 768, 37, 37)
        unit = features[0].float()
        unit = unit / unit.norm(dim=0, keepdim=True).clamp(min=1e-6)
        background = unit.reshape(768, -1).median(dim=1).values
        background = background / background.norm()
        distinct = 1 - (unit * background[:, None, None]).sum(dim=0)
        rows = distinct.mean(dim=1)
        cols = distinct.mean(dim=0)
        if axis == "row":
            assert 14 <= int(rows.argmax()) <= 21
            assert float(rows.max() - rows.min()) > float(cols.max() - cols.min())
        else:
            assert 14 <= int(cols.argmax()) <= 21
            assert float(cols.max() - cols.min()) > float(rows.max() - rows.min())


# ---------------------------------------------------------------------------
# Feature spread diagnostic
# ---------------------------------------------------------------------------

def write_fake_cache(root, encoder, scene, maps):
    directory = cache_dir(root, encoder, scene)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / "features.npz",
        **{f"f{i:03d}": m.to(torch.float16).numpy() for i, m in enumerate(maps)},
    )


def test_feature_statistics_detects_a_dominant_shared_direction(tmp_path):
    """The diagnostic that says whether cosine can resolve anything at all."""
    from lot.encoders import feature_statistics

    generator = torch.Generator().manual_seed(0)
    spread = [torch.randn((6, 3, 3), generator=generator) for _ in range(4)]
    offset = torch.zeros((6, 1, 1))
    offset[0] = 20.0
    write_fake_cache(tmp_path, "dinov2_vitb14", "room_0", spread)
    write_fake_cache(tmp_path, "vggt_1b", "room_0", [m + offset for m in spread])

    wide = feature_statistics(tmp_path, "dinov2_vitb14", ["room_0"], samples=2000)
    narrow = feature_statistics(tmp_path, "vggt_1b", ["room_0"], samples=2000)

    # A big common offset pushes every cosine towards one and hides the content.
    assert narrow["raw"]["shared_direction_fraction"] > 0.9
    assert wide["raw"]["shared_direction_fraction"] < 0.5
    assert narrow["raw"]["cosine_across_frames"] > 0.9
    assert narrow["raw"]["cosine_across_frames"] > wide["raw"]["cosine_across_frames"]
    # Centering removes the offset, so the two agree again on the same content.
    assert narrow["centered"]["cosine_across_frames"] == pytest.approx(
        wide["centered"]["cosine_across_frames"], abs=0.05
    )


def test_vggt_sees_one_frame_at_a_time():
    """Scopes the Phase 3 finding: no frame can leak into another's tokens.

    VGGT's aggregator alternates attention within a frame and across the frames
    of a sequence. If a context and its target were ever handed over as one
    sequence, the global attention would mix them and the transportability
    measurement would be reading a representation that had already seen the
    answer. The wrapper passes a sequence of length one, always.
    """
    encoder = fake_vggt_encoder()
    encoder.encode(np.zeros((3, 28, 28, 3), dtype=np.uint8))
    count, sequence = encoder.model.aggregator.last_shape[:2]
    assert count == 3 and sequence == 1


@pytest.mark.skipif(
    not os.environ.get("LOT_ENCODER_SMOKE"),
    reason="set LOT_ENCODER_SMOKE=1 to download weights and run the encoder",
)
def test_vggt_batching_does_not_mix_frames():
    """The same claim against the real model: batching must not couple images.

    A sequence of length one rules out the aggregator's cross-frame attention,
    but only measurement rules out the batch axis coupling images some other
    way. Encoding two frames together must equal encoding each alone.
    """
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (2, 518, 518, 3), dtype=np.uint8)
    encoder = load_encoder("vggt_1b", "cuda" if torch.cuda.is_available() else "cpu")
    together = encoder.encode(frames).features
    apart = torch.cat([encoder.encode(frames[i : i + 1]).features for i in (0, 1)])
    assert torch.allclose(together, apart, atol=1e-3)


def test_depth_content_is_protected_not_only_its_shape(tmp_path):
    """Shape and frame presence do not protect content.

    Every value in a depth archive can change while the keys and shapes stay
    right. That is inert for Experiment Zero, whose geometry is ground truth,
    and it is a Phase 4 blocker: VGGT depth and its confidence become scientific
    inputs there, and the validator claimed digest mode covered every array.
    """
    root, manifest = fake_scene_dir(tmp_path)
    cache_root = tmp_path / "cache"
    encoder = StubEncoder(stub_spec(provides_depth=True), "cpu")
    cache_scene_features(manifest, root, encoder, cache_root, export_depth=True)
    validate_feature_cache(cache_root, "stub_depth", manifest, check_digest=True)

    depth_path = cache_dir(cache_root, "stub_depth", manifest.scene) / "depth.npz"
    with np.load(depth_path) as archive:
        tampered = {name: np.full_like(archive[name], 9.0) for name in archive.files}
    np.savez(depth_path, **tampered)
    with pytest.raises(ValueError, match="depth digest"):
        validate_feature_cache(cache_root, "stub_depth", manifest, check_digest=True)
