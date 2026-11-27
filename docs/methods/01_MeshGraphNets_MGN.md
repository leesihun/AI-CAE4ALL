# 01 — MeshGraphNets (MGN), the flat baseline

| | |
| --- | --- |
| Route | `model meshgraphnets` |
| Directory | [methods/MeshGraphNets/](../../methods/MeshGraphNets/) |
| Entrypoint | [MeshGraphNets_main.py](../../methods/MeshGraphNets/MeshGraphNets_main.py) |
| Spec | [ai_cae4all/specs/meshgraphnets.py](../../ai_cae4all/specs/meshgraphnets.py) |
| Paper | Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*, ICLR 2021 |

MGN is a deterministic, node-resolved surrogate for unstructured meshes. It
runs message passing on the mesh graph, plus optional world edges for contact.
Temporal data is rolled out autoregressively, and static data (`T = 1`) is
predicted in one shot.

This doc covers the **flat** processor (`use_multiscale false`). The same route
and directory also run the hierarchical variants [HI-MGN](02_HI-MGN.md) and
[BSMS-GNN](03_BSMS-GNN.md). For spread or one-to-many targets, use
[MGN-V](04_MeshGraphNets_Variational.md).

Read [00_shared_foundations.md](00_shared_foundations.md) first. It covers
graph encoding, normalization, the blocks, training and resume.

## Architecture

[`MeshGraphNets`](../../methods/MeshGraphNets/model/MeshGraphNets.py) wraps an
`EncoderProcessorDecoder`
([encoder_decoder.py](../../methods/MeshGraphNets/model/encoder_decoder.py)),
which runs these stages in order:

1. **Encode:** a node MLP and a mesh-edge MLP (`8 → latent_dim`), plus a
   world-edge MLP when `use_world_edges` is on. The MLPs are built by
   [mlp.py](../../methods/MeshGraphNets/model/mlp.py).
2. **Process:** `message_passing_num` `GnBlock`s
   ([blocks.py](../../methods/MeshGraphNets/model/blocks.py)). Each one does an
   edge update, then a sum-aggregated node update with residuals. When world
   edges are on, the node update is a `HybridNodeBlock`. The blocks run on raw
   tensors, which keeps them compatible with `torch.compile` and activation
   checkpointing.
3. **Decode:** an MLP without LayerNorm produces `[N, output_var]` in
   normalized units. This is the delta when `T > 1` and the absolute field
   when `T = 1`.

Training-time input noise and the target correction are described in
[00 §5](00_shared_foundations.md#5-training-and-rollout).

## Data contract

The dataset is the shared mesh HDF5
([DATASET_FORMAT.md](../reference/DATASET_FORMAT.md)). `cond_var` rows,
`use_node_types` (last row) and `use_world_edges` behave as described in
[00](00_shared_foundations.md).

## Modes and keys

The modes are `train` and `inference`. To see the required fields for each
mode, run `python AI_CAE4ALL_main.py --describe meshgraphnets`. The spec is
the authority on keys and defaults. [CONFIGURATION.md](../CONFIGURATION.md)
explains the grammar.

| Group | Keys |
| --- | --- |
| Routing | `model`, `mode`, `gpu_ids` (`-1` = CPU), `parallel_mode` (`ddp` default, or `model_split` with `pipeline_microbatches`) |
| Data | `dataset_dir`, `infer_dataset`, `infer_timesteps`, `split_seed`, `split_group_attr`, `write_preprocessing` |
| Shape | `input_var`, `output_var`, `cond_var`, `edge_var` (= 8), `positional_features`, `use_node_types`, `latent_dim`, `message_passing_num` (required when flat) |
| Edges | `use_deformed_edge`, `displacement_indices`, `use_world_edges`, `world_radius_multiplier`, `world_max_num_neighbors`, `world_edge_backend` |
| Training | `training_epochs`, `batch_size`, `learningr`, `weight_decay`, `warmup_epochs`, `std_noise`, `noise_gamma`, `feature_loss_weights`, `time_integration`, `augment_geometry`, `grad_accum_steps`, `use_amp`, `use_checkpointing`, `use_compile`, `use_ema`, `ema_decay`, `init_modelpath` (initial weights), `resume_training`, `resume_interval_minutes` |
| Loader | `num_workers`, `prefetch_factor`, `pin_memory` |
| Monitoring | `val_interval`, `test_interval`, `test_batch_idx`, `test_max_batches`, `train_eval_subset_size`, `plot_feature_idx`, `display_testset`, `display_trainset`, `log_file_dir` |

Spec checks that apply to the flat model:

- `MGN-EDGE-001`: `edge_var` must be 8.
- `MGN-MP-REQ`: `message_passing_num` is required when multiscale is off.
- `MGN-PARALLEL`: `model_split` needs at least two GPUs.

Shipped configs follow the pattern
`configs/MeshGraphNets/deterministic/<slot>/baseline/config_{train,infer}_meshgraphnets.txt`,
for example
[ex1 train](../../configs/MeshGraphNets/deterministic/ex1/baseline/config_train_meshgraphnets.txt)
and [ex1 infer](../../configs/MeshGraphNets/deterministic/ex1/baseline/config_infer_meshgraphnets.txt).
The slots are ex1, ex2, ex3_full, ex3_mid and ex4–ex10.

## Outputs

All paths are written by the config and resolved from `methods/MeshGraphNets/`.

- **Checkpoint:** written to `modelpath` once training ends. It holds the
  model and EMA weights, the normalization statistics, the node-type mapping,
  the world-edge radius and `model_config`. With `resume_training`, a
  `<modelpath>.resume` file sits beside it while the run is in progress.
- **Epoch log and periodic PNG/HDF5 snapshots:** written to `log_file_dir`.
- **Rollouts:** `rollout_sample{id}_steps{N}.h5` in `inference_output_dir`
  (default `../../output/meshgraphnets/rollout`). Each file stores
  `num_features = 3 + output_var + 1` and `output_var` as attributes. The
  layout is coordinates, then predicted channels, then node type.

## Tests and validation

```bash
python AI_CAE4ALL_main.py --config configs/MeshGraphNets/deterministic/ex1/baseline/config_train_meshgraphnets.txt --check
cd methods/MeshGraphNets && python -m pytest -q tests/
```

The tests cover these areas:

- AR rollout and input noise
- `cond_var` conditioning and grouped splits
- edge geometry (`test_state_geometry`)
- multiscale statistics, the hierarchy cache, attention transfer, learned
  interpolation and multi-partition
- surface reconstruction and the offscreen-render guard

## Known limitations

- **Receptive field:** information moves one hop per block, so global coupling
  on large meshes needs many blocks or [HI-MGN](02_HI-MGN.md).
- **No intermediate checkpoints:** the checkpoint is written only at the end.
  An interrupted run keeps nothing unless `resume_training` is on.
- **Loader bound:** large graphs at small batch sizes tend to be limited by the
  data loader.
- **Deterministic:** the model gives one output per input and cannot represent
  spread.
- **Error accumulation:** errors build up over long rollouts. `std_noise` or
  `ar_rt` reduce this but do not remove it.
