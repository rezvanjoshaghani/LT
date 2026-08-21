# Cluster runbook

Phase 1 renders Replica. Phase 2 caches frozen features from those renders.
The two phases use different environments, because habitat-sim pins an old
Python and the encoders want a current torch.

# Phase 1: rendering

Habitat-Sim runs on Linux only. Rendering happens on the Borah cluster (or
any Linux box with a GPU; sync the data afterwards, scenes are small). The
camera programs, manifest handling, and depth-convention logic are tested
locally without Habitat by `tests/test_render_replica.py`.

Two scripts do everything:

| script | what it does | where |
| --- | --- | --- |
| `setup_borah.sh` | micromamba, render env, Replica download | login node, once |
| `run_phase1_render.sh` | submits the rendering experiment | login node |

## 1. Setup (once)

```bash
REPLICA_ROOT=/bsuscratch/$USER/replica ./scripts/setup_borah.sh
```

The dataset is tens of GB, so point `REPLICA_ROOT` at scratch; the script
symlinks it to `data/replica` where the configs expect it. It installs
micromamba to `~/.local/bin` if missing, creates the `lot-render` env
(habitat-sim 0.3.1 headless + withbullet, CPU torch, numpy < 2), downloads
the official Replica v1.0 release parts with `wget -c` (resumable, kept
next to the target until the scene check passes, no pigz needed), and
verifies all 18 scene meshes.

Python note: habitat-sim conda builds pin Python 3.9 while this package
targets 3.10+, so the render env never pip-installs the package. The run
scripts set `PYTHONPATH=src` instead; the rendering path needs nothing
more. Scenes without a shipped navmesh get one recomputed at render time,
recorded as `metadata.navmesh: recomputed` in the manifest.

## 2. Pilot: one scene end to end (Phase 1 acceptance, first half)

```bash
export SLURM_ACCOUNT=<account> SLURM_PARTITION=<gpu partition>
./scripts/run_phase1_render.sh pilot
```

Optional quick check first: `./scripts/run_phase1_render.sh smoke` runs the
habitat pytest smoke test (a tiny render into a temp dir) as a short GPU
job.

When the pilot job finishes, check in `data/replica_renders_pilot/room_0/`:

1. `qc/qc_rotation.png`, `qc_translation.png`, `qc_orbit.png`: RGB and
   depth aligned, textures present, no empty renders, plausible depth
   ranges.
2. `manifest.json` `metadata.depth_convention.raw_verdict` is `planar_z`
   or `euclidean_ray` (ambiguity aborts the render; probe images sit in
   `probes/`).
3. `./scripts/run_phase1_render.sh validate configs/render_replica_pilot.yaml`
   passes.

If the RGB tiles render untextured or gray: the build lacks PTex support
for Replica's `mesh.ply`. Set `scene_relpath: habitat/mesh_semantic.ply`
in both configs and re-run the pilot; record the change in the commit.

## 3. Batch: all 18 scenes (Phase 1 acceptance, second half)

```bash
./scripts/run_phase1_render.sh all        # refuses to run before the pilot
./scripts/run_phase1_render.sh validate   # after the array job finishes
```

Outputs land in `data/replica_renders/<scene>/`. Existing scene manifests
are never overwritten; delete a scene directory manually to re-render it.

## Knobs (environment variables, nothing hard-coded)

- `SLURM_ACCOUNT`, `SLURM_PARTITION`: required to submit.
- `LOT_GPU_GRES`: gres string if the cluster needs a typed request
  (default `gpu:1`).
- `LOT_MODULES`: modules to load inside jobs, space separated, if EGL
  needs them on your nodes.
- `LOT_ENV`, `MAMBA_ROOT_PREFIX`, `HABITAT_VERSION`, `REPLICA_ROOT`: see
  `setup_borah.sh` header.

If headless EGL rendering fails on the cluster even with modules loaded,
render on a Linux machine with a display driver and sync
`data/replica_renders*/` back (the CLAUDE.md fallback).

# Phase 2: feature caching

Encoding needs a current torch and a GPU, but not habitat-sim, so it runs in
its own environment. Everything except the two model wrappers is tested
locally by `tests/test_encoder_cache.py` against a stub encoder.

## 1. Environment (once)

```bash
micromamba create -y -n lot-encode -c conda-forge python=3.11 pyyaml pillow numpy
micromamba run -n lot-encode pip install torch --index-url https://download.pytorch.org/whl/cu121
```

VGGT is a separate install and is only needed for the `vggt_1b` config:

```bash
micromamba run -n lot-encode pip install "vggt @ git+https://github.com/facebookresearch/vggt"
```

Weights download once. `cache_features.sbatch` points `TORCH_HOME` and
`HF_HOME` at `cache/` in the repository, so compute nodes without internet
still work after one login-node run has populated them:

```bash
TORCH_HOME=$PWD/cache/torch micromamba run -n lot-encode \
    python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitb14',trust_repo=True)"
```

## 2. Pilot, then the batch

```bash
export SLURM_ACCOUNT=<account> SLURM_PARTITION=<gpu partition>
./scripts/run_phase2_cache.sh pilot
./scripts/run_phase2_cache.sh all       # refuses to run before the pilot
./scripts/run_phase2_cache.sh vggt      # VGGT features and estimated depth
./scripts/run_phase2_cache.sh validate  # no GPU
```

To find the account and partition: inside a running job,
`echo "$SLURM_JOB_ACCOUNT $SLURM_JOB_PARTITION"`; otherwise
`sacctmgr -nP show assoc user=$USER format=account` and
`sinfo -o "%P %G" | grep -i gpu`. The wrapper says the same thing if either
variable is unset, which raw `sbatch` does not.

Caches land in `cache/features/<encoder>/<scene>/`. A finished cache is never
overwritten. The wrapper submits array indices only for scenes that have no
cache yet, so a partially finished batch resumes without queueing jobs that
would no-op, and the sbatch also passes `--resume` as a second guard.

## What to check

`meta.json` per scene records the channel count, the patch grid, the frame
ids, and the measured rate. `--validate-only` re-checks every frame's shape
and dtype against the manifest, so a truncated archive fails loudly rather
than surfacing as missing features in Phase 3.

Reference numbers from an RTX 2080 Super at 518 px, DINOv2 ViT-B/14 at batch
8: about 9 frames per second end to end, which is roughly 10 minutes for all
5136 frames, in 0.8 GB of GPU memory. The model alone runs at about 21 frames
per second, so the rest is PNG decode and host transfer; a faster GPU moves
the ceiling but not the floor. The cache is about 2.1 MB per frame, so the
full set is roughly 11 GB per encoder. VGGT is a much larger model; start it
at batch 2 and raise it if memory allows.
