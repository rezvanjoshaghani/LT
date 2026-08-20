# Cluster runbook for Phase 1 rendering

Habitat-Sim runs on Linux only. Rendering happens on the Borah cluster (or
any Linux box with a GPU; sync the data afterwards, scenes are small). The
camera programs, manifest handling, and depth-convention logic are tested
locally without Habitat by `tests/test_render_replica.py`.

## One-time setup on the cluster

```bash
conda create -n lot python=3.10 cmake=3.14.0 -y
conda activate lot
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat -y
pip install -e ".[dev]"
```

If headless EGL rendering fails on the cluster, render on a Linux machine
with a display driver and sync `data/replica_renders/` back.

## Replica dataset

Download per the official instructions at
<https://github.com/facebookresearch/Replica-Dataset> (the `download.sh`
script) into `data/replica/`, so that each scene sits at
`data/replica/<scene>/mesh.ply` with its navmesh at
`data/replica/<scene>/habitat/mesh_semantic.navmesh`. If your download
places meshes elsewhere, point `scene_relpath` and `navmesh_relpath` in the
config at the right files instead of moving data around.

## Pilot: one scene end to end (Phase 1 acceptance, first half)

```bash
export SLURM_ACCOUNT=<account> SLURM_PARTITION=<gpu-partition> LOT_CONDA_ENV=lot
sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
       scripts/render_replica.sbatch configs/render_replica_pilot.yaml
```

Or interactively: `python -m lot.render_replica --config
configs/render_replica_pilot.yaml`.

Then check, in `data/replica_renders/room_0/`:

1. `manifest.json` exists and `python -m lot.render_replica --config
   configs/render_replica_pilot.yaml --validate-only` passes.
2. `qc/qc_rotation.png`, `qc_translation.png`, `qc_orbit.png` look right:
   RGB and depth aligned, no empty renders, plausible depth ranges.
3. The depth-convention finding is recorded under
   `metadata.depth_convention` in the manifest (`raw_verdict` is
   `planar_z` or `euclidean_ray`, never left ambiguous; ambiguity aborts
   the render, see `probes/` for the probe images).
4. Optionally run the cluster-side smoke test:
   `REPLICA_ROOT=data/replica pytest tests/test_render_replica.py -k habitat`.

## Full batch: 18 scenes (Phase 1 acceptance, second half)

```bash
sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" \
       --array 0-17 scripts/render_replica.sbatch configs/render_replica_all.yaml
```

Outputs land in `data/replica_renders/<scene>/`. Existing scene manifests
are never overwritten; delete a scene directory manually to re-render it.
