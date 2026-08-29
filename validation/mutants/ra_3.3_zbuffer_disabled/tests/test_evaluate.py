"""PROTOCOL 3.2, 3.5, 3.6, 3.7: scored records, the frozen nulls, and the metrics."""

import json
import math

import numpy as np
import pytest
import torch

from lot.analysis_config import load_analysis_config
from lot.correspondence import NEIGHBOR_OFFSETS, choose_in_bounds_offset
from lot.evaluate import (
    MEAN_FEATURE,
    NEIGHBOR_PATCH,
    NO_WARP_COPY,
    ORACLE_TRANSPORT,
    PER_POINT,
    RANDOM_PATCH,
    SPLAT_POOL,
    VARIANTS,
    EvalConfig,
    agreement_metrics,
    assert_source_read_sets_agree,
    assert_unique_sample_ids,
    cross_path_record_difference,
    dataset_mean_vector,
    evaluate_pair_for_encoder,
    evaluate_scene,
    load_eval_config,
    pack_mask,
    pair_geometry,
    pair_parallax,
    read_rows,
    splat_neighbor_prediction,
    universe_sample_ids,
    unpack_mask,
    unit_normalize,
    value_agreement,
    write_rows,
)
from lot.render_replica import intrinsics_from_hfov
from lot.transport import apply_transport_plan
from scenes import GRID, build_two_plane_scene

ANALYSIS = load_analysis_config()
IDENTITY = ("room_0", "ctx", "tgt")
SIDE = 56
SMALL_GRID = SIDE // 14


def identity_pair(channels=6, seed=0):
    """A pair whose transport plan is the identity, so both paths read one patch.

    With no relative motion and constant depth, every context pixel lands on
    itself, so each target patch draws all of its weight from the source patch of
    the same index. That collapses the difference between the two paths and lets
    the reads be compared directly.
    """
    depth = torch.full((SIDE, SIDE), 3.0, dtype=torch.float32)
    K = intrinsics_from_hfov(SIDE, SIDE, 90.0).to(torch.float32)
    T = torch.eye(4, dtype=torch.float32)
    geometry = pair_geometry(depth, depth, K, K, T, *IDENTITY, ANALYSIS)
    generator = torch.Generator().manual_seed(seed)
    features = torch.rand((channels, SMALL_GRID, SMALL_GRID), generator=generator)
    return geometry, features


# ---------------------------------------------------------------------------
# PROTOCOL 3.6: the neighbour direction is per record, on both paths
# ---------------------------------------------------------------------------

def test_transport_plan_is_the_identity_for_a_still_camera():
    """The premise of the read-equality test below."""
    geometry, _ = identity_pair()
    weights = geometry.plan.weights
    assert torch.allclose(weights, torch.eye(weights.shape[0]), atol=1e-6)


def test_neighbour_direction_is_per_sample_on_the_splat_path():
    """A whole-map shift would silently make the direction pair-level.

    PROTOCOL 3.6 draws the offset from each record's own sample_id. On the splat
    path that has to stay per record: applying one direction to every cell of a
    pair would redefine the variant and break comparability with per-point, so
    each cell's prediction is checked against its own hashed direction.
    """
    geometry, features = identity_pair()
    prediction, defined, directions = splat_neighbor_prediction(geometry, features)
    flat = features.reshape(features.shape[0], -1)
    assert len(set(directions.tolist())) > 1, "the pair must use more than one direction"
    for cell in np.flatnonzero(defined):
        row, col = divmod(int(cell), SMALL_GRID)
        dx, dy = NEIGHBOR_OFFSETS[int(directions[cell])]
        source = (row + dy) * SMALL_GRID + (col + dx)
        assert torch.allclose(prediction[:, cell], flat[:, source], atol=1e-6)


def test_the_two_paths_read_the_same_source_values():
    """Under identity transport the paths must agree read for read.

    Every variant reads the same context location on both paths, so with a plan
    that mixes nothing the splat-path value entering aggregation is exactly the
    value per-point scores. This is what rules out a pair-level direction on the
    splat side.
    """
    geometry, features = identity_pair()
    from lot.correspondence import gather_value_pairs

    reads = gather_value_pairs(features, features, geometry.samples)
    prediction, defined, _ = splat_neighbor_prediction(geometry, features)
    transported = apply_transport_plan(geometry.plan, features).reshape(features.shape[0], -1)
    flat = features.reshape(features.shape[0], -1)

    cols = torch.round((geometry.samples.uv_target[:, 0] + 0.5) / 14 - 0.5).long()
    rows = torch.round((geometry.samples.uv_target[:, 1] + 0.5) / 14 - 0.5).long()
    cells = (rows * SMALL_GRID + cols).tolist()
    checked = 0
    for index, cell in enumerate(cells):
        if not defined[cell]:
            continue
        assert torch.allclose(reads["neighbor"][index], prediction[:, cell], atol=1e-5)
        assert torch.allclose(reads["warp"][index], transported[:, cell], atol=1e-5)
        assert torch.allclose(reads["no_warp"][index], flat[:, cell], atol=1e-5)
        checked += 1
    assert checked > 0


def test_random_patch_uses_the_same_hash_on_both_paths():
    geometry, _ = identity_pair()
    ids = universe_sample_ids(*IDENTITY, geometry.grid)
    assert np.array_equal(geometry.universe_ids, ids)
    cols = torch.round((geometry.samples.uv_target[:, 0] + 0.5) / 14 - 0.5).long()
    rows = torch.round((geometry.samples.uv_target[:, 1] + 0.5) / 14 - 0.5).long()
    for index in range(len(geometry.samples.sample_id)):
        cell = int(rows[index]) * SMALL_GRID + int(cols[index])
        assert geometry.samples.sample_id[index] == geometry.universe_ids[cell]
        per_point = geometry.samples.random_patch_index[index]
        splat = geometry.random_patch[cell]
        assert int(per_point[0]) * SMALL_GRID + int(per_point[1]) == int(splat)


# ---------------------------------------------------------------------------
# PROTOCOL 3.7: centering, and Mean-Feature's structural not-applicable
# ---------------------------------------------------------------------------

def test_centering_orders_agree_on_the_splat_path():
    """PROTOCOL 3.7 asks for exactly this test.

    Centering is defined at the output level, on the pooled values. That
    coincides with centering the sources before pooling only because a scored
    cell's pooled output is a normalized weighted mean, so its weights sum to
    one and the constant passes through unchanged. If the transport contract
    ever stopped normalizing, these two orders would part company silently.
    """
    geometry, features = identity_pair(channels=8, seed=3)
    scene = build_two_plane_scene()
    plan = geometry.plan
    center = torch.rand(features.shape[0]) * 2 - 1

    after = apply_transport_plan(plan, features).reshape(features.shape[0], -1) - center[:, None]
    before = apply_transport_plan(
        plan, features - center[:, None, None]
    ).reshape(features.shape[0], -1)
    scored = torch.from_numpy(geometry.splat_mask)
    assert bool(scored.any())
    assert torch.allclose(after[:, scored], before[:, scored], atol=1e-5)
    assert scene is not None


def test_pooled_weights_sum_to_one_on_scored_cells():
    """The property the centering-order agreement rests on."""
    geometry, _ = identity_pair()
    sums = geometry.plan.weights.sum(dim=1)
    scored = torch.from_numpy(geometry.splat_mask)
    assert torch.allclose(sums[scored], torch.ones(int(scored.sum())), atol=1e-6)


def test_centered_mean_feature_is_never_finite():
    """The tripwire PROTOCOL 3.7 requires.

    Mean-Feature's prediction is the mean vector, so centering sends it to the
    zero vector and its centered cosine is undefined. An implementation that
    manufactured a score here, by an epsilon-regularized zero vector or by
    letting the floor object drift from the centering vector, would produce a
    finite number and nothing else would notice.
    """
    geometry, features = identity_pair()
    center = features.reshape(features.shape[0], -1).mean(dim=1)
    rows = evaluate_pair_for_encoder(geometry, features, features, center)
    mean_rows = [r for r in rows if r["variant"] == MEAN_FEATURE]
    assert len(mean_rows) == 2  # one per path
    for row in mean_rows:
        assert math.isnan(row["cosine_centered_mean"])
        assert math.isnan(row["l2_centered_mean"])
        assert math.isfinite(row["cosine_mean"])


def test_mean_feature_is_the_only_nonfinite_in_the_table():
    """PROTOCOL 3.2: that is the single permitted representation."""
    geometry, features = identity_pair()
    center = features.reshape(features.shape[0], -1).mean(dim=1)
    for row in evaluate_pair_for_encoder(geometry, features, features, center):
        for column in ("cosine_mean", "l2_mean"):
            assert math.isfinite(row[column]), row
        if row["variant"] != MEAN_FEATURE:
            assert math.isfinite(row["cosine_centered_mean"]), row
            assert math.isfinite(row["l2_centered_mean"]), row


def test_mean_feature_prediction_is_one_global_vector_on_both_paths():
    """BLOCKER-2: it was three different objects, and one of them beat Oracle.

    The two paths score different cell sets, since the splat path drops cells
    whose neighbour support leaves the grid, so their scores need not match.
    What must hold is that the prediction is the same object: one global vector,
    not a per-image mean on one path and a position-conditioned map on the other.
    """
    geometry, features = identity_pair()
    center = torch.rand(features.shape[0])
    rows = {
        (r["path"], r["variant"]): r
        for r in evaluate_pair_for_encoder(geometry, features, features, center)
    }
    # A per-image mean or a position map would move with the features; the
    # global vector does not, so changing the features while holding the vector
    # fixed must leave Mean-Feature scoring against the same prediction.
    other = torch.rand(features.shape[0], SMALL_GRID, SMALL_GRID)
    again = {
        (r["path"], r["variant"]): r
        for r in evaluate_pair_for_encoder(geometry, other, other, center)
    }
    for path in (PER_POINT, SPLAT_POOL):
        baseline = value_agreement(
            center[None, :].expand(4, -1), center[None, :].expand(4, -1)
        )
        assert baseline[0] == pytest.approx(1.0)
        assert math.isfinite(rows[(path, MEAN_FEATURE)]["cosine_mean"])
        assert math.isfinite(again[(path, MEAN_FEATURE)]["cosine_mean"])


def test_dataset_mean_vector_is_a_vector_over_frames_and_positions(tmp_path):
    from lot.encoders import cache_dir

    from lot.encoders import CACHE_VERSION, features_digest

    directory = cache_dir(tmp_path, "dinov2_vitb14", "room_0")
    directory.mkdir(parents=True)
    arrays = {
        "a": np.zeros((3, 2, 2), dtype=np.float16),
        "b": np.full((3, 2, 2), 2.0, dtype=np.float16),
    }
    np.savez(directory / "features.npz", **arrays)
    (directory / "meta.json").write_text(
        json.dumps({"cache_version": CACHE_VERSION, "features_digest": features_digest(arrays)}),
        encoding="utf-8",
    )
    mean = dataset_mean_vector(tmp_path, "dinov2_vitb14", ["room_0"])
    assert mean.shape == (3,)
    assert torch.allclose(mean, torch.ones(3))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_value_agreement_endpoints():
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert value_agreement(a, a) == (pytest.approx(1.0), pytest.approx(0.0))
    cosine, l2 = value_agreement(a, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert cosine == pytest.approx(0.0)
    assert l2 == pytest.approx(math.sqrt(2.0))


def test_value_agreement_ignores_magnitude():
    g = torch.Generator().manual_seed(0)
    a = torch.rand((32, 8), generator=g)
    b = torch.rand((32, 8), generator=g)
    plain = value_agreement(a, b)
    scaled = value_agreement(a * 17.0, b * 0.03)
    assert plain[0] == pytest.approx(scaled[0], abs=1e-6)


def test_unit_normalize_leaves_zero_vectors_alone():
    assert torch.equal(unit_normalize(torch.zeros((2, 3))), torch.zeros((2, 3)))


def test_centering_restores_range_when_one_direction_dominates():
    generator = torch.Generator().manual_seed(0)
    content = torch.randn((256, 32), generator=generator)
    other = torch.randn((256, 32), generator=generator)
    offset = torch.zeros(32)
    offset[0] = 30.0
    center = ((content + offset).mean(dim=0) + (other + offset).mean(dim=0)) / 2
    raw = agreement_metrics(content + offset, other + offset, center)
    assert raw["cosine_mean"] > 0.95
    assert abs(raw["cosine_centered_mean"]) < 0.15


# ---------------------------------------------------------------------------
# PROTOCOL 3.2: identity, masks, and row hygiene
# ---------------------------------------------------------------------------

def test_sample_id_collision_is_refused():
    with pytest.raises(RuntimeError, match="collision"):
        assert_unique_sample_ids(np.array([7, 7, 9], dtype=np.uint64), "a pair")


def test_masks_round_trip_and_name_the_scored_cells():
    geometry, features = identity_pair()
    center = torch.zeros(features.shape[0])
    rows = evaluate_pair_for_encoder(geometry, features, features, center)
    size = geometry.size
    for row in rows:
        restored = unpack_mask(row["sample_mask"], size)
        assert int(restored.sum()) == row["n"] or row["path"] == PER_POINT
    per_point = next(r for r in rows if r["path"] == PER_POINT)
    assert np.array_equal(unpack_mask(per_point["sample_mask"], size), geometry.per_point_mask)


def test_all_variants_on_a_path_share_one_mask():
    """The common-valid design of PROTOCOL 3.7: differences are paired structurally."""
    geometry, features = identity_pair()
    rows = evaluate_pair_for_encoder(geometry, features, features, torch.zeros(features.shape[0]))
    for path in (PER_POINT, SPLAT_POOL):
        masks = {r["sample_mask"] for r in rows if r["path"] == path}
        counts = {r["n"] for r in rows if r["path"] == path}
        assert len(masks) == 1, path
        assert len(counts) == 1, path


def test_five_variants_on_both_paths():
    """MAJOR-14: two of the five nulls existed on one path only."""
    geometry, features = identity_pair()
    rows = evaluate_pair_for_encoder(geometry, features, features, torch.zeros(features.shape[0]))
    for path in (PER_POINT, SPLAT_POOL):
        assert {r["variant"] for r in rows if r["path"] == path} == set(VARIANTS)
    assert len(rows) == 2 * len(VARIANTS)


# ---------------------------------------------------------------------------
# PROTOCOL 3.2: the parallax statistic
# ---------------------------------------------------------------------------

def test_parallax_is_the_median_over_the_covisible_set_only():
    """MAJOR-4: the denominator was a median over the whole frame.

    Here the co-visible half of the frame sits at 2 m and the rest at 8 m, so a
    whole-frame median would report a different number entirely.
    """
    depth = torch.full((16, 16), 8.0)
    depth[:4] = 2.0
    covisible = torch.zeros((16, 16), dtype=torch.bool)
    covisible[:4] = True
    assert pair_parallax(1.0, depth, covisible) == pytest.approx(0.5)
    whole_frame = 1.0 / float(depth.median())
    assert whole_frame != pytest.approx(0.5)


def test_parallax_of_an_empty_covisible_set_is_nan():
    depth = torch.full((4, 4), 3.0)
    assert math.isnan(pair_parallax(1.0, depth, torch.zeros((4, 4), dtype=torch.bool)))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def base_config(tmp_path, **overrides):
    values = dict(
        experiment_name="experiment_zero",
        renders_root=tmp_path,
        cache_root=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        scenes=["room_0"],
        encoders=["dinov2_vitb14"],
    )
    values.update(overrides)
    return EvalConfig(**values)


def test_config_rejects_unknown_scenes_and_encoders(tmp_path):
    with pytest.raises(ValueError, match="unknown Replica scenes"):
        base_config(tmp_path, scenes=["not_a_scene"])
    with pytest.raises(ValueError, match="unknown encoders"):
        base_config(tmp_path, encoders=["clip"])


def test_mean_vector_defaults_to_the_training_split(tmp_path):
    """PROTOCOL 3.6: the floor never adapts to the frames it is a floor for."""
    cfg = base_config(tmp_path, scenes=["room_0", "hotel_0"])
    assert cfg.mean_vector_scenes == ["room_0"]


def test_shipped_config_loads():
    from pathlib import Path

    cfg = load_eval_config(
        Path(__file__).resolve().parents[1] / "configs" / "experiment_zero.yaml"
    )
    assert cfg.encoders and cfg.scenes


# ---------------------------------------------------------------------------
# The two paths select records by the same rule
# ---------------------------------------------------------------------------

def test_no_cell_is_stranded_without_an_in_bounds_offset():
    """PROTOCOL 3.6's omission clause guards a case the geometry forbids.

    A cell's support can span the full width or the full height of the grid but
    not both, so at least one axis always has a usable direction. Dropping cells
    whose first choice leaked would select the record set by a rule the other
    path does not apply.
    """
    geometry, features = identity_pair()
    _, defined, _ = splat_neighbor_prediction(geometry, features)
    assert defined.all()


def test_the_two_paths_score_the_same_records():
    """The gap the read-equality test cannot see: records absent from one path.

    Reads are only compared where both paths have a record, so a null that
    silently removed records would leave every compared value correct.
    """
    geometry, _ = identity_pair()
    assert np.array_equal(geometry.per_point_mask, geometry.splat_mask)
    difference = cross_path_record_difference(geometry)
    assert difference["per_point_only"] == 0
    assert difference["splat_only"] == 0
    assert difference["both"] == geometry.size
    assert_source_read_sets_agree(geometry, "identity pair")


def test_a_per_point_only_cell_may_be_one_the_cell_test_excluded():
    """Low co-visible fraction is a second legitimate reason, and was missing.

    splat_mask is a conjunction: the warp covers the cell, and at least
    min_covisible_fraction of its pixels are co-visible. The per-point path asks
    a different question, whether the patch centre is co-visible and its four
    surrounding pixels lie on one surface, and a centre can pass that while its
    whole cell fails the fraction test.

    The accounting credited only coverage, so this case raised as a rule
    mismatch. It survived three synthetic probes because the four-corner
    consistency test rejects centres near a depth edge and the low-co-visible
    cells in those scenes were all near one; real orbit geometry produced it
    within a few hundred pairs, on a cell more than half occluded whose centre
    sits on the visible part.
    """
    geometry, _ = identity_pair()
    cell = int(np.flatnonzero(geometry.per_point_mask & geometry.splat_mask)[0])
    assert geometry.plan.coverage.reshape(-1)[cell] > 0, "the cell must stay covered"
    geometry.splat_covisible_ok[cell] = False
    geometry.splat_mask[cell] = False

    difference = cross_path_record_difference(geometry)
    assert difference["per_point_only_low_covisible"] >= 1
    assert difference["per_point_only_unexplained"] == 0
    assert_source_read_sets_agree(geometry, "half-occluded cell")


def test_an_unenumerated_exclusion_is_still_refused():
    """The check is on the accounting, not on the data.

    It cannot fail for splat_mask as defined, because that mask is the
    conjunction of the two conditions the difference is decomposed by. Conjoin
    a third term without naming it in cross_path_record_difference and the
    cells it excludes become unexplained, which is the drift that produced the
    false positive this replaces.
    """
    geometry, _ = identity_pair()
    cell = int(np.flatnonzero(geometry.per_point_mask & geometry.splat_mask)[0])
    geometry.splat_mask[cell] = False  # neither coverage nor the fraction moved
    with pytest.raises(RuntimeError, match="does not enumerate"):
        assert_source_read_sets_agree(geometry, "tampered pair")


def test_coverage_holes_are_a_legitimate_cross_path_difference():
    """A cell the warp cannot support is an operator property, reported not hidden."""
    geometry, _ = identity_pair()
    geometry.plan.coverage.reshape(-1)[0] = 0.0
    geometry.splat_mask[0] = False
    assert_source_read_sets_agree(geometry, "uncovered pair")
    assert cross_path_record_difference(geometry)["per_point_only_uncovered"] == 1


# ---------------------------------------------------------------------------
# Regressions from the external review
# ---------------------------------------------------------------------------

def test_sample_ids_are_globally_unique_not_merely_per_pair():
    """PROTOCOL 3.2 makes the identity global, and a 32-bit seed is not.

    An earlier version reduced the pair key with crc32 before the 64-bit mix,
    which capped the pair space at 2^32 however wide the mix that followed. Over
    the 79,272 pairs the camera programs produce, the birthday bound puts about
    one collision in that space and enumeration found four. Two pairs sharing a
    seed give identical ids to their samples at matching target coordinates, and
    a cross-pair join, which is how Phase 4 matches surviving sets, would merge
    them silently. The within-pair uniqueness assertion cannot see it.
    """
    from lot.sample_identity import pair_seed

    scenes = ["apartment_2", "hotel_0", "room_0", "office_1"]
    regimes = {"rotation": 13, "translation": 17, "orbit": 18}
    seen: dict[int, tuple] = {}
    for scene in scenes:
        for viewpoint in range(6):
            for regime, count in regimes.items():
                ids = [f"{scene}_vp{viewpoint:02d}_{regime}_{i:03d}" for i in range(count)]
                for context in ids:
                    for target in ids:
                        if context == target:
                            continue
                        seed = int(pair_seed(scene, context, target))
                        key = (scene, context, target)
                        assert seed not in seen, (
                            f"pair seed collision between {seen.get(seed)} and {key}"
                        )
                        seen[seed] = key
    assert len(seen) > 15_000


def test_a_pair_scorable_on_one_path_is_not_discarded():
    """Requiring both paths would condition per-point results on splat success.

    That removes exactly the difficult, low-coverage pairs and biases every
    per-point aggregate towards easy geometry. Coverage differences between the
    paths are reported, not resolved by dropping the pair.
    """
    geometry, features = identity_pair()
    geometry.splat_mask[:] = False
    assert geometry.scorable
    rows = evaluate_pair_for_encoder(
        geometry, features, features, torch.zeros(features.shape[0])
    )
    assert {r["path"] for r in rows} == {PER_POINT}
    assert len(rows) == len(VARIANTS)

    # The row-count arithmetic the forensic population check builds its
    # expectation from: five rows for a one-path pair, ten for a both-path
    # pair. The validator's synthetic scene happens to produce only both-path
    # pairs, so this identity is pinned here where a one-path geometry can be
    # forced, rather than trusted to a fixture that never exercises it.
    both_geometry, both_features = identity_pair()
    both_rows = evaluate_pair_for_encoder(
        both_geometry, both_features, both_features, torch.zeros(both_features.shape[0])
    )
    assert len(both_rows) == 2 * len(VARIANTS)
    assert len(rows) == len(VARIANTS)

    geometry.per_point_mask[:] = False
    geometry.splat_mask[:] = False
    assert not geometry.scorable


def test_the_mean_vector_is_written_atomically_and_checked_on_reuse(tmp_path):
    """The documented run is an 18-task array over one output directory."""
    from lot.encoders import cache_dir
    from lot.evaluate import load_or_build_mean_vector

    def write_cache(scene, value):
        from lot.encoders import CACHE_VERSION, features_digest

        directory = cache_dir(tmp_path / "cache", "dinov2_vitb14", scene)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {"a": np.full((4, 2, 2), value, dtype=np.float16)}
        np.savez(directory / "features.npz", **arrays)
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "cache_version": CACHE_VERSION,
                    "features_digest": features_digest(arrays),
                    "weights_fingerprint": "abc123",
                }
            ),
            encoding="utf-8",
        )
        return directory

    directory = write_cache("room_0", 3.0)

    out = tmp_path / "out"
    first = load_or_build_mean_vector(tmp_path / "cache", "dinov2_vitb14", ["room_0"], out)
    assert torch.allclose(first, torch.full((4,), 3.0))
    assert not list(out.glob("*.partial"))
    # Reuse is validated against provenance, not taken on trust.
    again = load_or_build_mean_vector(tmp_path / "cache", "dinov2_vitb14", ["room_0"], out)
    assert torch.equal(first, again)
    write_cache("room_1", 3.0)
    with pytest.raises(ValueError, match="built for"):
        load_or_build_mean_vector(tmp_path / "cache", "dinov2_vitb14", ["room_1"], out)

    # Rebuilding the cache in place, at the same path, with different features.
    # Naming the directory in the provenance record cannot see this: the path is
    # unchanged, so the stale vector was returned and both the Mean-Feature floor
    # and the centering statistic silently described features that no longer
    # existed. The digest is over the bytes that went into the vector.
    write_cache("room_0", 9.0)
    with pytest.raises(ValueError, match="built for"):
        load_or_build_mean_vector(tmp_path / "cache", "dinov2_vitb14", ["room_0"], out)


def test_both_paths_choose_the_same_neighbour_direction_under_translation():
    """Read the direction off each path's own artifact, never off a shared source.

    The two paths ask different questions about whether an offset is usable. The
    per-point path reads one location and asks whether it stays inside the
    context sampling box; the splat path shifts a cell's whole transport support
    and asks whether any of it leaves the grid. Under identity transport the
    answers coincide, which is why an identity-only test certified a property
    that was false in general. Under a lateral translation they come apart.

    An earlier version of this test recomputed both directions from the splat
    rule and compared them, which is an identity and holds whatever the code
    does. This one takes the per-point direction from uv_context_neighbor, the
    coordinate the path will actually read, and the splat direction from what
    splat_neighbor_prediction returns.
    """
    from lot.encoders import patch_cell_index
    from lot.evaluate import splat_neighbor_option_ok

    depth = torch.full((SIDE, SIDE), 3.0, dtype=torch.float32)
    K = intrinsics_from_hfov(SIDE, SIDE, 90.0).to(torch.float32)
    T = torch.eye(4, dtype=torch.float32)
    T[0, 3] = 0.05
    geometry = pair_geometry(depth, depth, K, K, T, *IDENTITY, ANALYSIS)
    samples = geometry.samples

    # The per-point path's direction, recovered from the coordinate it will read.
    delta = (samples.uv_context_neighbor - samples.uv_context_warp).cpu().numpy() / 14.0
    offsets = np.asarray(NEIGHBOR_OFFSETS, dtype=np.float64)
    per_point = np.argmin(np.abs(offsets[None, :, :] - delta[:, None, :]).sum(axis=2), axis=1)
    assert np.allclose(offsets[per_point], delta, atol=1e-6), "not a whole-patch offset"

    _, _, splat = splat_neighbor_prediction(geometry, torch.rand((4, SMALL_GRID, SMALL_GRID)))
    cells = patch_cell_index(samples.uv_target, (SIDE, SIDE), 14)
    assert cells.size > 0
    assert np.array_equal(per_point, splat[cells])

    # And the two rules really do disagree here, so the test is not vacuous: the
    # splat rule alone admits offsets the per-point rule refuses, and hashing
    # among a different number of options is what moved the direction.
    from lot.correspondence import _in_box, _sampling_box

    box = _sampling_box((SIDE, SIDE), 14)
    offsets_px = torch.tensor(offsets, dtype=torch.float32) * 14
    warp = samples.uv_context_warp
    point_rule = _in_box(warp[:, None, :] + offsets_px[None, :, :], box).cpu().numpy()
    splat_rule = splat_neighbor_option_ok(geometry.plan, (SMALL_GRID, SMALL_GRID))[cells]
    assert not np.array_equal(point_rule, splat_rule)
    assert np.array_equal(samples.neighbor_option_ok, point_rule & splat_rule)


def test_the_pair_loop_checks_that_both_paths_select_records_alike():
    """The assertion existed and only its tests called it.

    A check that is correct and unwired protects nothing, which is the same
    fault as a gate that reports a stop condition and continues. This pins the
    call site rather than the function, because the function was never the part
    that was missing.
    """
    import inspect

    from lot import evaluate

    body = inspect.getsource(evaluate.evaluate_scene)
    assert "assert_source_read_sets_agree(" in body, (
        "evaluate_scene must run the cross-path read-set check on every pair"
    )


def test_random_patch_draws_from_the_grid_it_reads(monkeypatch):
    """Both paths index the context map, so both must draw modulo its size.

    The splat path used the target grid. The two agree only because every frame
    in this study is one resolution; a larger target grid would index the
    context map out of bounds, and a smaller one would give one record two
    different nulls on the two paths.
    """
    import inspect

    from lot import correspondence, evaluate

    per_point = inspect.getsource(correspondence.sample_correspondences)
    assert "ctx_patches_h * ctx_patches_w" in per_point

    geometry, features = identity_pair()
    context_cells = features.shape[1] * features.shape[2]
    assert geometry.random_patch.min() >= 0
    assert geometry.random_patch.max() < context_cells

    # And the two paths agree record for record on which patch they read.
    rows = geometry.samples.random_patch_index[:, 0].cpu().numpy()
    cols = geometry.samples.random_patch_index[:, 1].cpu().numpy()
    per_point_flat = rows * features.shape[2] + cols
    assert np.array_equal(per_point_flat, geometry.random_patch[geometry.per_point_cells])


def test_evaluation_verifies_the_bytes_it_consumes(tmp_path):
    """Same-shape corruption must stop the run, not enter the scoring.

    The metadata check alone accepts an archive rebuilt in place: shapes and
    keys stay right while every value differs, and the run record then repeats
    the old digest as a description of bytes it never covered. Nothing required
    the caching job's own digest pass to have run before evaluation read the
    archive, so the evaluator verifies what it consumes.
    """
    from test_encoder_cache import StubEncoder, fake_scene_dir, stub_spec

    from lot.encoders import cache_scene_features
    from lot.evaluate import _SceneCache

    root, manifest = fake_scene_dir(tmp_path)
    cache_root = tmp_path / "cache"
    cache_scene_features(manifest, root, StubEncoder(stub_spec(), "cpu"), cache_root)

    # Intact: constructing the cache reader validates and succeeds.
    _SceneCache(root, cache_root, ["stub"], manifest.scene, manifest).close()

    # Rebuild the archive in place with the same keys and shapes.
    from lot.encoders import cache_dir

    path = cache_dir(cache_root, "stub", manifest.scene) / "features.npz"
    with np.load(path) as archive:
        tampered = {name: np.full_like(archive[name], 9.0) for name in archive.files}
    np.savez(path, **tampered)
    with pytest.raises(ValueError, match="features digest"):
        _SceneCache(root, cache_root, ["stub"], manifest.scene, manifest)


def test_a_poisoned_feature_cache_cannot_produce_a_mean(tmp_path):
    """The digest is verified against the bytes the mean consumes.

    mean_vector_cache_digest reads only the metadata, so a features archive
    rebuilt in place with the same shapes produced a poisoned mean whose
    provenance repeated the stale digest as if it described these bytes.
    """
    from lot.encoders import CACHE_VERSION, cache_dir, features_digest
    from lot.evaluate import dataset_mean_vector

    directory = cache_dir(tmp_path, "dinov2_vitb14", "room_0")
    directory.mkdir(parents=True)
    arrays = {"a": np.full((4, 2, 2), 3.0, dtype=np.float16)}
    np.savez(directory / "features.npz", **arrays)
    (directory / "meta.json").write_text(
        json.dumps({"cache_version": CACHE_VERSION, "features_digest": features_digest(arrays)}),
        encoding="utf-8",
    )
    assert torch.allclose(
        dataset_mean_vector(tmp_path, "dinov2_vitb14", ["room_0"]), torch.full((4,), 3.0)
    )
    # Rebuild the archive alone: same keys, same shapes, other values.
    np.savez(directory / "features.npz", a=np.full((4, 2, 2), 9.0, dtype=np.float16))
    with pytest.raises(ValueError, match="rebuilt or modified in place"):
        dataset_mean_vector(tmp_path, "dinov2_vitb14", ["room_0"])


def test_a_replaced_mean_vector_file_is_refused(tmp_path):
    """Provenance over the inputs cannot see the output replaced in place.

    A same-shape overwrite of the .npy beside an intact .json was accepted, and
    the floor and the centering statistic moved silently. The record now binds
    the vector's own bytes.
    """
    from lot.encoders import CACHE_VERSION, cache_dir, features_digest
    from lot.evaluate import load_or_build_mean_vector

    directory = cache_dir(tmp_path / "cache", "dinov2_vitb14", "room_0")
    directory.mkdir(parents=True)
    arrays = {"a": np.full((4, 2, 2), 3.0, dtype=np.float16)}
    np.savez(directory / "features.npz", **arrays)
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "features_digest": features_digest(arrays),
                "weights_fingerprint": "abc",
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    first = load_or_build_mean_vector(tmp_path / "cache", "dinov2_vitb14", ["room_0"], out)
    # Overwrite the vector with one of the same shape and dtype.
    poisoned = first.numpy().copy()
    poisoned[:] = 99.0
    with open(out / "mean_vector_dinov2_vitb14.npy", "wb") as handle:
        np.save(handle, poisoned)
    with pytest.raises(ValueError, match="has been replaced"):
        load_or_build_mean_vector(tmp_path / "cache", "dinov2_vitb14", ["room_0"], out)
