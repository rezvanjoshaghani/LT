# Cluster runbook for Phase 1 rendering

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
Replica with the official `download.sh`, and verifies all 18 scene meshes.

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
