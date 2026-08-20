# CLAUDE.md

This repository implements a diagnostic study with the working title "Learn or Transport? Dissecting Cross-View Prediction in Frozen Visual Representations." The question: when a frozen visual representation must be predicted across camera ego-motion in a static scene, how much of the apparent change should be learned by a neural predictor, and how much can be computed exactly by projective geometry? The study is an error ladder, not an architecture paper. Phases and acceptance criteria live in @PLAN.md. Work one phase at a time and do not start a phase before the previous phase's acceptance criteria pass.

## Method names

Use these descriptive names everywhere: code identifiers, configs, figures, docs. Never use letter labels like A, B, or C for methods.

- Transport-Only: no training. Lift context features with depth, reproject into the target camera, z-buffered forward splat, pool to the patch grid. Disoccluded regions stay empty.
- Oracle-Transport: Transport-Only using ground-truth depth instead of estimated depth.
- Predict-with-Depth: a trained network that receives context features, context depth, and the relative camera transform, and must learn the transformation. No hard-coded projection.
- Predict-Everything: the same trained network with depth withheld.
- Floors: No-Warp-Copy (context features copied unwarped) and Mean-Feature (dataset mean feature map).

Parked and out of scope for now: any hybrid (transport plus learned completion), cycle or composition probes, inverse-pose probes, scale sweeps, RealEstate10K, multiple context views. Do not build these.

## Environment

- Python 3.10 or newer, PyTorch, single GPU for training and encoding. Small predictors over cached frozen features; nothing here needs multi-GPU.
- Habitat-Sim with the Replica dataset is needed only for Phase 1 rendering. Install per the official Habitat-Sim instructions (conda build recommended). If headless EGL rendering fails on the cluster, render locally and sync the data; scenes are small.
- Heavy batch jobs (rendering, feature caching, training) run through SLURM on the Borah cluster. Keep SLURM templates in scripts/ and never hard-code account or partition names; read them from environment variables.
- Encoders: DINOv2 ViT-B/14 via torch.hub (facebookresearch/dinov2) and VGGT (facebookresearch/vggt) for features and estimated depth. Both frozen. Weights download once; cache everything after that.

## Repository layout

```
src/lot/
  geometry.py        # unproject, transform, project, relative poses, parallax
  visibility.py      # z-buffer reprojection visibility, co-visible and disoccluded masks
  transport.py       # pixel-level forward splat with z-buffer, pool to patch grid, coverage
  correspondence.py  # Experiment Zero sampling: value pairs after GT warp, plus all nulls
  encoders.py        # DINOv2 and VGGT wrappers, feature caching, bilinear feature sampling
  render_replica.py  # Habitat camera programs and per-scene manifests
  datasets.py        # pair manifests, regime tags, parallax bins, scene splits
  predictors.py      # Predict-Everything and Predict-with-Depth (same trunk, depth toggled)
  train.py           # single-GPU training over cached features
  evaluate.py        # ladder metrics, stratified; writes outputs/eval/*.parquet
  figures.py         # ladder figure, parallax curves, disocclusion contrast
tests/               # analytic synthetic-scene tests; must pass before any phase advances
scripts/             # slurm templates and entrypoints
configs/             # one yaml per experiment
data/, cache/, outputs/   # gitignored
```

## Conventions that must not drift

- Camera model: OpenCV pinhole. x right, y down, z forward. Intrinsics K are 3x3.
- Poses are stored as T_world_from_camera (4x4). The relative transform is T_target_from_context = inv(T_world_from_target) @ T_world_from_context. Write this formula once in geometry.py and import it everywhere.
- Depth is planar z-depth in meters, float32, aligned to RGB. Phase 1 includes a test that determines whether the renderer outputs planar z-depth or euclidean ray distance, and converts if needed. Do not assume.
- Feature maps are [C, Hp, Wp] at patch stride 14, cached as fp16 npz. Continuous sampling is bilinear on the patch grid; pixel (u, v) maps to patch coordinates ((u + 0.5) / 14 - 0.5, same for v). Define this mapping once in encoders.py.
- Transport contract: transport(features_ctx, depth_ctx_px, K_ctx, K_tgt, T_tgt_from_ctx, out_hw_px) returns (features_tgt [C, Hp, Wp], coverage [Hp, Wp] in [0, 1], zbuffer). Splat at pixel resolution carrying each pixel's patch feature, resolve occlusion with the z-buffer at pixel level, then pool into target patches. Coverage is the fraction of a target patch's pixels that received support.
- Visibility buckets always come from ground-truth geometry of both views, never from estimated depth. Co-visible means the context camera sees the target surface point within a relative depth tolerance of 1.5 percent; disoccluded means it does not. Estimated depth may be an input to a method; it is never the referee.
- Every reported metric (cosine and L2 on unit-normalized features) is accompanied by the No-Warp-Copy and Mean-Feature floors, and the headline quantity is the margin over No-Warp-Copy. Stratify all results by regime, parallax bin, and visibility bucket.
- Reproducibility: fixed seeds, one yaml config plus one entrypoint per experiment, outputs under outputs/{experiment_name}/, never overwrite existing outputs, and every figure must be regenerable from outputs/eval/*.parquet alone.

## Working style

- Tests first for all geometry, visibility, transport, and correspondence code. The analytic synthetic scenes in tests/ are the referee, not visual inspection. Run the full test suite before declaring any phase done.
- Small pure functions for geometry with type hints. No hidden state. No implicit coordinate conventions; if a function consumes or produces coordinates, its docstring names the convention.
- Prose in docs, comments, and commit messages: short sentences, one idea per sentence, no em dashes.
- When a result looks anomalous relative to the expectations written in @PLAN.md, stop and report rather than tuning until it looks right. Anomalies are findings in this project, not bugs to hide.
- Commit after each green test suite with a message naming the phase and what changed.
