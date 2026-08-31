"""Stream Q: the fold assignment is frozen, separated, and derived from names only."""

from __future__ import annotations

import pytest

from lot.phase5_folds import (
    FROZEN_FOLD_DIGEST,
    FROZEN_FOLDS,
    N_FOLDS,
    Fold,
    build_folds,
    canonical_order,
    fold_digest,
    fold_of_test_scene,
    frozen_folds,
    assert_scene_separation,
    scene_family,
)
from lot.render_replica import REPLICA_SCENES


def test_committed_literal_equals_the_rule():
    """The auditable literal and the executable rule must not drift apart.

    The literal is what a reader checks; the rule is what a run executes. If
    only one is edited the suite fails, which is the point.
    """
    computed = build_folds()
    assert len(computed) == len(FROZEN_FOLDS) == N_FOLDS
    for fold, spec in zip(computed, FROZEN_FOLDS):
        assert sorted(fold.test) == sorted(spec["test"]), f"fold {fold.index} test"
        assert sorted(fold.val) == sorted(spec["val"]), f"fold {fold.index} val"
        assert sorted(fold.train) == sorted(spec["train"]), f"fold {fold.index} train"


def test_frozen_digest_matches():
    assert fold_digest(frozen_folds()) == FROZEN_FOLD_DIGEST


def test_digest_is_order_independent_but_content_sensitive():
    folds = frozen_folds()
    shuffled = [
        Fold(index=f.index, train=tuple(reversed(f.train)), val=tuple(reversed(f.val)),
             test=tuple(reversed(f.test)))
        for f in folds
    ]
    assert fold_digest(shuffled) == fold_digest(folds)

    moved = list(folds)
    swapped = Fold(
        index=0,
        train=folds[0].train[1:] + (folds[0].val[0],),
        val=(folds[0].train[0],) + folds[0].val[1:],
        test=folds[0].test,
    )
    moved[0] = swapped
    assert fold_digest(moved) != fold_digest(folds)


def test_every_scene_is_test_exactly_once():
    folds = frozen_folds()
    assert_scene_separation(folds)
    test_union = [s for f in folds for s in f.test]
    assert sorted(test_union) == sorted(REPLICA_SCENES)
    assert len(test_union) == len(set(test_union))


def test_no_scene_appears_in_two_roles_of_one_fold():
    """The leakage condition Stream Q exists for, asserted per fold."""
    for fold in frozen_folds():
        assert not (set(fold.train) & set(fold.test))
        assert not (set(fold.val) & set(fold.test))
        assert not (set(fold.train) & set(fold.val))
        assert len(fold.train) + len(fold.val) + len(fold.test) == len(REPLICA_SCENES)


def test_fold_rejects_an_overlapping_construction():
    with pytest.raises(ValueError, match="share"):
        Fold(index=9, train=("room_0",), val=(), test=("room_0",))


def test_test_sets_are_six_scenes_each():
    for fold in frozen_folds():
        assert len(fold.test) == 6, f"fold {fold.index} test size"
        assert len(fold.val) == 3, f"fold {fold.index} val size"
        assert len(fold.train) == 9, f"fold {fold.index} train size"


def test_test_folds_are_family_balanced():
    """Balance uses names only, and no family may pile into one fold.

    frl_apartment has six members and apartment and room have three each, so a
    balanced round-robin puts two frl_apartment scenes and one each of the other
    two in every fold. office has five and hotel one, which cannot divide
    evenly; those are bounded rather than fixed.
    """
    folds = frozen_folds()
    counts: dict[str, list[int]] = {}
    for fold in folds:
        for scene in fold.test:
            counts.setdefault(scene_family(scene), [0] * len(folds))[fold.index] += 1
    assert counts["frl_apartment"] == [2, 2, 2]
    assert counts["apartment"] == [1, 1, 1]
    assert counts["room"] == [1, 1, 1]
    for family, per_fold in counts.items():
        assert max(per_fold) - min(per_fold) <= 1, f"{family} is unbalanced: {per_fold}"


def test_validation_sets_are_not_single_family():
    """A validation set drawn from one family would select checkpoints on one
    scene type, so the offset round-robin is checked to have avoided it."""
    for fold in frozen_folds():
        families = {scene_family(s) for s in fold.val}
        assert len(families) == len(fold.val), f"fold {fold.index} val repeats a family"


def test_visible_scenes_excludes_test():
    for fold in frozen_folds():
        visible = set(fold.visible_scenes())
        assert visible == set(fold.train) | set(fold.val)
        assert not (visible & set(fold.test))


def test_fold_of_test_scene_is_unique_and_total():
    for scene in REPLICA_SCENES:
        fold = fold_of_test_scene(scene)
        assert scene in fold.test
        assert fold.role_of(scene) == "test"
    with pytest.raises(ValueError):
        fold_of_test_scene("not_a_scene_0")


def test_scene_family_and_order_are_name_derived():
    assert scene_family("frl_apartment_4") == "frl_apartment"
    assert scene_family("room_2") == "room"
    assert scene_family("hotel_0") == "hotel"
    with pytest.raises(ValueError):
        scene_family("no_index")
    order = canonical_order()
    assert order == sorted(order, key=lambda s: (scene_family(s), int(s.rsplit("_", 1)[1])))


def test_assert_scene_separation_catches_a_repeat():
    folds = frozen_folds()
    broken = list(folds)
    broken[1] = Fold(
        index=1,
        train=tuple(s for s in folds[1].train if s != folds[0].test[0]),
        val=folds[1].val,
        test=folds[1].test + (folds[0].test[0],),
    )
    with pytest.raises(ValueError, match="test in fold"):
        assert_scene_separation(broken)
