# Second real run of the integration gate: a pass with step 13 on the CPU

Recorded 2026-10-11. No Phase 5 model has trained at scale and no Phase 5
result exists.

## The run

The gate ran on Borah at commit `3a0f92cf7c2ee75466bdf6bbc40be34ddaf48780`,
Borah's merge of GitHub's `0b0e800`, inside
`srun --partition gpu-l40 --gres=gpu:1`. Config digest
`8d731c5e1dabe8764c40f3599037de530ae39de9f2537c1675e068f3c909b1f8`, fold digest
`25f0c03f72d58cc8e3ff2d8d4123241f6459d4ed50d3c303c5dc108bb0e19865`, measurement
digest `27244e6481d521159e513f2ea8799482`, torch 2.5.1 built for CUDA 12.1. All
17 steps passed, in the order 1 to 14, 17, 15, 16.

## Step 17, the pure-rotation gate across the regime

It checked every rotation-regime pair of every scene at the image level.

| quantity | value |
|---|---|
| scenes | 18 |
| rotation pairs | 4108 |
| pairs checked | 4108 |
| pairs skipped, for any reason | 0 |
| largest camera translation in a rotation pair | 0.0 m |
| worst Context-Lift residual against the homography | 7.00e-5 px |
| worst landing shift under depth substitution | 9.16e-5 px |
| worst TL-Reference residual against the inverse homography | 7.78e-5 px |
| tolerance | 1e-3 px |

Every rotation pair shares its camera position exactly, so the translation
allowance of reporting_rules.md section 7 was zero throughout. The smallest
headroom under the tolerance was 9.08e-4 px.

## Step 13 on the CPU

Step 12's evidence records `device: cpu`. SLURM granted an L40, but torch saw
no CUDA device, so step 12 dry-ran the training batch on the CPU, about 5.3 s
forward and 4.4 s backward, and step 13 had no GPU memory to measure. The gate
passed with the memory-fit test skipped. The runbook already said the gate
must run on a GPU, and the gate did not enforce it.

The training templates had the same weakness. They did not pass `--device`,
so a job whose torch saw no GPU would have trained on the CPU, at about 10 s a
step, for days.

## The fix

- Gate step 12 takes its device from `lot.phase5_gate.dry_run_device`, which
  stops as a missing resource when torch sees no CUDA device.
- The training and job templates pass `--device cuda`, so such a job stops at
  once.
- The overfit verdict and every training record name the device and the
  environment they ran on.

The cause on the node, a driver or a visibility problem, is diagnosed
separately. The gate is rerun on a node where torch sees the GPU. That
supersedes this run's receipt, so the overfit gate is rerun after it. Any
overfit result bound to this run's receipt is superseded with it.

## The cause, diagnosed on 2026-10-11

Short jobs pinned to each node showed a node-level permission, not a code,
driver, or environment problem.

- On `gpu118`, an L40 node, the GPU device files `/dev/nvidia0` to
  `/dev/nvidia3` and `/dev/nvidiactl` are owned by `root:vglusers`, mode 0660.
  The account is not in `vglusers`. SLURM allocates the card, but every
  process the account runs is refused it: `nvidia-smi` reports "Failed to
  initialize NVML: Insufficient Permissions", and torch sees zero devices.
  Those device files were changed on September 22.
- On `gpu114` and `gpu117`, also L40 nodes of `gpu-l40`, the device files are
  `root:root`, mode 0666, and torch sees the L40. Five other `gpu-l40` nodes
  were busy and could not be tested.
- The A30, V100, RTX 8000, and P100 nodes tested are all open, and torch sees
  each card.
- The same environment, torch 2.5.1 for CUDA 12.1, works on every open node.

The run stays on L40s. `LOT_EXCLUDE_NODES` keeps every submitted job off the
restricted node, and the `srun` that runs `check` excludes it too. A job that
lands on another restricted node stops at once under `--device cuda`, and the
node is then added to the list.
