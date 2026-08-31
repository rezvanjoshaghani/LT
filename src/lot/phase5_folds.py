"""Phase 5, Stream Q: frozen scene-level folds and the leakage rules around them.

Phase 5 is the first phase that trains anything, so it is the first phase whose
results can be contaminated by a scene appearing on both sides of a split. The
rule is scene-level cross-fitting: no scene may appear in both training and
test for the same model, each scene is test exactly once across three folds,
and frame-level splitting inside one scene is forbidden outright. A model that
saw frame 12 of a room has seen that room's furniture, its lighting, and its
depth statistics; scoring it on frame 47 of the same room measures memorization
of a scene, not transfer across viewpoint.

The assignment is a pure function of the scene names, computed here once and
asserted against a committed literal. It cannot depend on any Phase 3, Phase 4,
or Phase 5 outcome, because none of those values is imported by this module and
the fold identity is pinned by a digest before training starts.

Why the folds are not PLAN.md's 13/5 split. PLAN.md Phase 5 says "evaluation on
the five held-out scenes", which is a single split with thirteen training
scenes. The Phase 5 specification this module implements requires three-fold
cross-fitting in which every scene is test exactly once, so that the headline
estimand is measured on all eighteen scenes rather than on five. Eighteen
scenes over three folds is six test scenes per fold exactly. The departure is
recorded in the Phase 5 pin; it widens the test population and cannot narrow
it, and the scene remains the primary independent unit either way.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Sequence

from .render_replica import REPLICA_SCENES

# Three folds, fixed before any Phase 5 training runs. The count is not a
# tunable: it is the largest number of folds that keeps every test fold at or
# above the frozen support_min_scenes of 3, while keeping enough training
# scenes per fold for the predictor to have something to generalize from.
N_FOLDS = 3

# The family a scene name belongs to, used only for balance. Scene names and
# types are the sole admissible input to fold construction; nothing measured
# may enter. Derived from the name by taking everything before the trailing
# index, so it needs no table to maintain and cannot drift from the scene list.
def scene_family(scene: str) -> str:
    """The scene's family, from its name alone. 'frl_apartment_4' -> 'frl_apartment'."""
    head, sep, tail = scene.rpartition("_")
    if not sep or not tail.isdigit():
        raise ValueError(f"{scene!r} does not end in a family index")
    return head


def canonical_order(scenes: Sequence[str] = REPLICA_SCENES) -> list[str]:
    """Scenes ordered by (family, index), the order the fold rule walks.

    Sorting by family first and index second, rather than by the raw name, is
    what makes the round-robin below balance families: consecutive members of a
    family land in consecutive folds.
    """
    def key(scene: str) -> tuple[str, int]:
        family = scene_family(scene)
        return family, int(scene[len(family) + 1 :])

    return sorted(scenes, key=key)


@dataclasses.dataclass(frozen=True)
class Fold:
    """One fold's three disjoint scene sets.

    train and val are the only scenes a model of this fold may see. test is
    sealed: nothing about it may reach architecture, optimizer, duration,
    checkpoint selection, early stopping, loss design, or debugging.
    """

    index: int
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        sets = {"train": set(self.train), "val": set(self.val), "test": set(self.test)}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sets[a] & sets[b]
            if overlap:
                raise ValueError(
                    f"fold {self.index}: {a} and {b} share {sorted(overlap)}; "
                    "scene-level separation is the whole point of the fold"
                )
        total = len(self.train) + len(self.val) + len(self.test)
        if total != len(REPLICA_SCENES):
            raise ValueError(
                f"fold {self.index}: {total} scenes assigned, expected {len(REPLICA_SCENES)}"
            )

    def role_of(self, scene: str) -> str:
        """Which role a scene plays in this fold. Raises for an unknown scene."""
        for role in ("train", "val", "test"):
            if scene in getattr(self, role):
                return role
        raise ValueError(f"{scene!r} is not assigned in fold {self.index}")

    def visible_scenes(self) -> tuple[str, ...]:
        """Every scene this fold's models are allowed to read, in canonical order."""
        return tuple(canonical_order(self.train + self.val))


def build_folds(scenes: Sequence[str] = REPLICA_SCENES, n_folds: int = N_FOLDS) -> list[Fold]:
    """The fold assignment, derived from scene names alone.

    Test membership is a round-robin over the family-ordered scene list, so each
    fold's test set holds every family's scenes spread as evenly as the counts
    allow and each scene is test exactly once. Validation is a second, offset
    round-robin over each fold's remaining scenes, which keeps the validation
    sets disjoint in composition across folds without ever consulting a result.
    """
    order = canonical_order(scenes)
    folds: list[Fold] = []
    for index in range(n_folds):
        test = tuple(s for i, s in enumerate(order) if i % n_folds == index)
        rest = [s for s in order if s not in set(test)]
        # Every fourth scene of the remainder, offset by the fold, is validation.
        # Stride 4 over twelve scenes gives three validation scenes and nine
        # training scenes, and the offset stops the three folds from selecting
        # the same positions.
        val = tuple(s for i, s in enumerate(rest) if (i + index) % 4 == 0)
        train = tuple(s for s in rest if s not in set(val))
        folds.append(Fold(index=index, train=train, val=val, test=test))
    return folds


# The committed assignment. build_folds recomputes it and a test asserts the two
# agree, so the literal is the artifact a reader can audit without running code
# and the function is what the run uses. A change to either without the other
# fails the suite.
FROZEN_FOLDS: tuple[dict[str, tuple[str, ...]], ...] = (
    {
        "test": (
            "apartment_0",
            "frl_apartment_0",
            "frl_apartment_3",
            "hotel_0",
            "office_2",
            "room_0",
        ),
        "val": ("apartment_1", "frl_apartment_4", "office_3"),
        "train": (
            "apartment_2",
            "frl_apartment_1",
            "frl_apartment_2",
            "frl_apartment_5",
            "office_0",
            "office_1",
            "office_4",
            "room_1",
            "room_2",
        ),
    },
    {
        "test": (
            "apartment_1",
            "frl_apartment_1",
            "frl_apartment_4",
            "office_0",
            "office_3",
            "room_1",
        ),
        "val": ("frl_apartment_2", "office_1", "room_2"),
        "train": (
            "apartment_0",
            "apartment_2",
            "frl_apartment_0",
            "frl_apartment_3",
            "frl_apartment_5",
            "hotel_0",
            "office_2",
            "office_4",
            "room_0",
        ),
    },
    {
        "test": (
            "apartment_2",
            "frl_apartment_2",
            "frl_apartment_5",
            "office_1",
            "office_4",
            "room_2",
        ),
        "val": ("frl_apartment_0", "hotel_0", "room_0"),
        "train": (
            "apartment_0",
            "apartment_1",
            "frl_apartment_1",
            "frl_apartment_3",
            "frl_apartment_4",
            "office_0",
            "office_2",
            "office_3",
            "room_1",
        ),
    },
)

# The digest of the committed assignment above, recorded so the pin, the run
# records, and the evaluation artifacts all name one fold identity. A test
# asserts it against fold_digest(frozen_folds()).
FROZEN_FOLD_DIGEST = "25f0c03f72d58cc8e3ff2d8d4123241f6459d4ed50d3c303c5dc108bb0e19865"


def frozen_folds() -> list[Fold]:
    """The committed folds as Fold objects, validated by Fold's own invariants."""
    return [
        Fold(index=i, train=spec["train"], val=spec["val"], test=spec["test"])
        for i, spec in enumerate(FROZEN_FOLDS)
    ]


def fold_digest(folds: Sequence[Fold]) -> str:
    """A stable digest of the scene assignment, for the pin and every run record.

    Sorted inside each role so the digest names the assignment, not the order a
    literal happened to be typed in.
    """
    payload = [
        {
            "index": fold.index,
            "train": sorted(fold.train),
            "val": sorted(fold.val),
            "test": sorted(fold.test),
        }
        for fold in sorted(folds, key=lambda f: f.index)
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def assert_scene_separation(folds: Sequence[Fold]) -> None:
    """Every scene is test exactly once, and no fold leaks a scene across roles.

    Fold.__post_init__ already holds the within-fold disjointness. This adds the
    across-fold condition that makes the union of the test sets a partition of
    the scene list, which is what lets the headline estimand pool test scores
    over all eighteen scenes without scoring any scene twice.
    """
    seen: dict[str, int] = {}
    for fold in folds:
        for scene in fold.test:
            if scene in seen:
                raise ValueError(
                    f"{scene} is test in fold {seen[scene]} and again in fold {fold.index}"
                )
            seen[scene] = fold.index
    missing = set(REPLICA_SCENES) - set(seen)
    if missing:
        raise ValueError(f"never test in any fold: {sorted(missing)}")


def fold_of_test_scene(scene: str, folds: Sequence[Fold] | None = None) -> Fold:
    """The one fold whose test set holds this scene.

    Evaluation reads a test scene through the model trained on the fold that
    held it out. Going through this function rather than an index makes the
    wrong-model mistake impossible to make silently.
    """
    for fold in folds if folds is not None else frozen_folds():
        if scene in fold.test:
            return fold
    raise ValueError(f"{scene!r} is not a test scene of any fold")
