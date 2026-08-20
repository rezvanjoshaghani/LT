#!/bin/bash
# One-time setup for Phase 1 rendering on Borah (or any Linux box).
# Installs micromamba, creates the render environment, and downloads the
# Replica dataset with the official downloader. Safe to re-run; every step
# skips work that is already done.
#
# Run from anywhere inside the repository, on a login node. No GPU needed.
# The Replica download is tens of GB; home quotas are usually too small,
# so point REPLICA_ROOT at scratch. Everything is controlled by
# environment variables, nothing cluster-specific is hard-coded:
#
#   REPLICA_ROOT        where the dataset lands (default <repo>/data/replica)
#   LOT_ENV             environment name        (default lot-render)
#   MAMBA_ROOT_PREFIX   micromamba root         (default $HOME/micromamba)
#   HABITAT_VERSION     habitat-sim version     (default 0.3.1)
#
# Example:
#   REPLICA_ROOT=/bsuscratch/$USER/replica ./scripts/setup_borah.sh
#
# Note on Python versions: the habitat-sim conda builds pin Python 3.9,
# while this package targets 3.10+. The render environment therefore does
# not pip-install the package; the run scripts execute it via
# PYTHONPATH=src, which is all the rendering path needs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOT_ENV="${LOT_ENV:-lot-render}"
HABITAT_VERSION="${HABITAT_VERSION:-0.3.1}"
REPLICA_ROOT="${REPLICA_ROOT:-$REPO_ROOT/data/replica}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

SCENES="apartment_0 apartment_1 apartment_2 frl_apartment_0 frl_apartment_1
frl_apartment_2 frl_apartment_3 frl_apartment_4 frl_apartment_5 hotel_0
office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2"

step() { printf '\n=== %s ===\n' "$*"; }

# --- 1. micromamba ---------------------------------------------------------
step "micromamba"
if command -v micromamba >/dev/null 2>&1; then
    MM="$(command -v micromamba)"
else
    MM="$HOME/.local/bin/micromamba"
    if [ ! -x "$MM" ]; then
        echo "installing micromamba to $MM"
        mkdir -p "$HOME/.local/bin"
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
            | tar -xj -C "$HOME/.local/bin" --strip-components=1 bin/micromamba
    fi
fi
echo "micromamba: $MM (root prefix $MAMBA_ROOT_PREFIX)"

# --- 2. render environment -------------------------------------------------
step "environment $LOT_ENV"
if [ -d "$MAMBA_ROOT_PREFIX/envs/$LOT_ENV" ]; then
    echo "environment exists, skipping creation"
else
    # habitat-sim conda builds are Python 3.9. headless selects the EGL
    # build for cluster nodes without a display; withbullet is the official
    # recommended variant.
    "$MM" create -y -n "$LOT_ENV" -c conda-forge python=3.9
    "$MM" install -y -n "$LOT_ENV" -c conda-forge -c aihabitat \
        "habitat-sim=$HABITAT_VERSION" withbullet headless
    # The compiled habitat bindings predate numpy 2. Torch is CPU-only
    # here; rendering does not use CUDA through torch.
    "$MM" install -y -n "$LOT_ENV" -c conda-forge \
        "numpy<2" pytorch-cpu pyyaml pillow matplotlib pytest
fi

step "environment check"
"$MM" run -n "$LOT_ENV" python - <<'PY'
import habitat_sim, matplotlib, numpy, PIL, torch, yaml
print("habitat_sim", habitat_sim.__version__)
print("torch", torch.__version__, "| numpy", numpy.__version__)
PY
PYTHONPATH="$REPO_ROOT/src" "$MM" run -n "$LOT_ENV" python - <<'PY'
from lot.render_replica import REPLICA_SCENES
print("lot.render_replica imports, scenes:", len(REPLICA_SCENES))
PY

# --- 3. Replica dataset ----------------------------------------------------
step "Replica dataset -> $REPLICA_ROOT"
find_scene_root() {
    # The official downloader may extract scenes at the root or one level
    # down. Return the directory that contains apartment_0.
    if [ -f "$1/apartment_0/mesh.ply" ]; then
        echo "$1"
    else
        find "$1" -maxdepth 2 -type f -name mesh.ply -path '*/apartment_0/*' \
            2>/dev/null | head -1 | xargs -r dirname | xargs -r dirname
    fi
}

mkdir -p "$REPLICA_ROOT"
EFFECTIVE_ROOT="$(find_scene_root "$REPLICA_ROOT")"
if [ -n "$EFFECTIVE_ROOT" ]; then
    echo "dataset already present at $EFFECTIVE_ROOT, skipping download"
else
    echo "downloading Replica via the official download.sh (tens of GB)"
    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT
    git clone --depth 1 https://github.com/facebookresearch/Replica-Dataset.git \
        "$TMP/Replica-Dataset"
    ( cd "$TMP" && bash "$TMP/Replica-Dataset/download.sh" "$REPLICA_ROOT" )
    EFFECTIVE_ROOT="$(find_scene_root "$REPLICA_ROOT")"
    if [ -z "$EFFECTIVE_ROOT" ]; then
        echo "ERROR: download finished but apartment_0/mesh.ply not found under $REPLICA_ROOT" >&2
        exit 1
    fi
fi

step "dataset check"
missing=0
navmesh=0
for s in $SCENES; do
    if [ ! -f "$EFFECTIVE_ROOT/$s/mesh.ply" ]; then
        echo "MISSING scene mesh: $s" >&2
        missing=$((missing + 1))
    fi
    [ -f "$EFFECTIVE_ROOT/$s/habitat/mesh_semantic.navmesh" ] && navmesh=$((navmesh + 1))
done
if [ "$missing" -gt 0 ]; then
    echo "ERROR: $missing of 18 scenes missing; re-run after fixing the download" >&2
    exit 1
fi
echo "all 18 scene meshes present; $navmesh/18 ship a navmesh"
if [ "$navmesh" -lt 18 ]; then
    echo "(scenes without one get a navmesh recomputed at render time; the"
    echo " manifest metadata records which happened)"
fi

# Configs address the dataset as data/replica inside the repo. Link it
# there when the data lives elsewhere.
LINK="$REPO_ROOT/data/replica"
if [ "$EFFECTIVE_ROOT" != "$LINK" ]; then
    mkdir -p "$REPO_ROOT/data"
    if [ -L "$LINK" ]; then
        ln -sfn "$EFFECTIVE_ROOT" "$LINK"
    elif [ -e "$LINK" ]; then
        echo "ERROR: $LINK exists and is not a symlink; move it aside so it can" >&2
        echo "point at $EFFECTIVE_ROOT" >&2
        exit 1
    else
        ln -s "$EFFECTIVE_ROOT" "$LINK"
    fi
    echo "linked $LINK -> $EFFECTIVE_ROOT"
fi

step "done"
echo "next: export SLURM_ACCOUNT=... SLURM_PARTITION=... and run"
echo "  ./scripts/run_phase1_render.sh pilot"
