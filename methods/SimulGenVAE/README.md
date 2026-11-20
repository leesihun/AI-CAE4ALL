# SimulGenVAE

SimulGenVAE is a hierarchical variational autoencoder for fixed-geometry
simulation fields plus a latent conditioner (LC). It learns a compact field
representation, maps physical conditions to that latent space, and decodes a
full field without rerunning the full-order solver.

This runtime is vendored under `methods/SimulGenVAE/` and uses the common
AI-CAE4ALL launcher and shared HDF5 data contract. The former
`SimulGen-VAE.py`, `preset.txt`, `input_data/condition.txt`, `launch_ddp.py`,
and pickle-dataset workflow have been retired.

## Quick start

Run these commands from the `AI-CAE4ALL` root:

```bash
# Inspect the registered modes and config contract.
python AI_CAE4ALL_main.py --describe simulgenvae

# Validate or print the native command without launching.
python AI_CAE4ALL_main.py --config configs/SimulGenVAE/deterministic/ex6/baseline/config_train_lsh_vae.txt --check
python AI_CAE4ALL_main.py --config configs/SimulGenVAE/deterministic/ex6/baseline/config_train_lsh_vae.txt --dry-run

# Train the VAE and then the latent conditioner.
python AI_CAE4ALL_main.py --config configs/SimulGenVAE/deterministic/ex6/baseline/config_train_lsh_vae.txt

# Decode conditions through LC -> VAE into a field HDF5 artifact.
python AI_CAE4ALL_main.py --config configs/SimulGenVAE/deterministic/ex6/baseline/config_infer_lsh_vae.txt
```

Direct backend execution is also supported. Run it from the method directory so
the config's `../../dataset/...` and `../../output/...` paths retain their native
meaning:

```bash
cd methods/SimulGenVAE
python SimulGenVAE_main.py --config ../../configs/SimulGenVAE/deterministic/ex6/baseline/config_train_lsh_vae.txt
```

## Modes

| `mode` | Purpose |
| --- | --- |
| `train` | Sequential VAE -> LC pipeline with compatible-checkpoint reuse |
| `train_vae` | Train only the hierarchical VAE |
| `train_lc` | Train only the latent conditioner against a frozen VAE |
| `reconstruct` | Map conditions through the LC and VAE decoder and write field predictions |

The merged `train` config uses `vae_*` and `lc_*` prefixes for stage-specific
training settings. Standalone `train_vae` and `train_lc` configs use unprefixed
settings such as `training_epochs`, `batch_size`, and `learningr`.

## Data contract

SimulGenVAE reads the shared mesh HDF5 layout:

```text
data/{sample_id}/nodal_data[num_features, num_timesteps, num_nodes]
```

Rows `0:3` contain reference coordinates. Physical-field rows begin at
`field_start_row` and `num_var` selects how many are used. The model flattens
each field into a dense `[samples, channels, time]` tensor, so every sample must
have the same node count and timestep count. `mesh_edge` may be present as part
of the shared contract but is not consumed by this dense VAE.

The latent conditioner accepts:

- `lc_data_type csv`: a headerless condition matrix ordered by sorted sample ID;
- `lc_data_type image`: ordered PNG/JPG condition images;
- `lc_data_type hdf5`: constant, input-only conditioning rows selected by
  `cond_var` from the same HDF5 file.

See the [shared dataset reference](../../docs/reference/DATASET_FORMAT.md) for
the complete layout.

## Artifacts and checkpoints

Checked-in configs write only to the root artifact tree. They live under
`configs/SimulGenVAE/deterministic/<slot>/baseline/` for the slots `ex3_full`,
`ex3_mid`, `ex6` and `ex9`, and each slot's train/infer pair uses
`output/dataset_matrix/deterministic/<slot>/lsh_vae/` (`lsh_vae` is
SimulGenVAE's campaign key). For `ex6`:

```text
output/dataset_matrix/deterministic/ex6/lsh_vae/pipeline.log
output/dataset_matrix/deterministic/ex6/lsh_vae/vae.log
output/dataset_matrix/deterministic/ex6/lsh_vae/lc.log
output/dataset_matrix/deterministic/ex6/lsh_vae/vae.pth
output/dataset_matrix/deterministic/ex6/lsh_vae/lc.pth
output/dataset_matrix/deterministic/ex6/lsh_vae/infer/reconstructions.h5
```

Periodic pictures (`vae_recon_epoch#####.png`, `lc_parity_epoch#####.png`) land
beside the stage logs.

Both stage checkpoints are dictionaries containing the stage, epoch, model
state, saved config, and normalization metadata. The LC checkpoint additionally
stores its latent/input normalization and shape metadata.

## Runtime layout

```text
SimulGenVAE_main.py          # config-driven mode dispatch and in-process DDP
general_modules/
  fom_dataset.py             # shared mesh HDF5 -> dense FOM tensor
  load_config.py             # native flat key/value parser
  distributed.py             # worker/device/DDP helpers
model/
  vae.py                     # hierarchical VAE
  encoder.py  decoder.py
  latent_conditioner.py      # CSV/HDF5 MLP conditioner
  latent_conditioner_cnn.py  # image conditioner
training_profiles/
  train_pipeline.py          # VAE -> LC orchestration
  train_vae.py  train_lc.py
inference_profiles/
  reconstruct.py             # conditions -> field HDF5
tests/
  conftest.py
  test_fom_dataset.py
  test_hdf5_conditions.py
  test_split_provenance.py
  test_reconstruct_normalization.py
  test_config_contract.py
```

Multi-GPU training uses in-process `torch.multiprocessing.spawn` when
`parallel_mode ddp` and multiple `gpu_ids` are configured. It does not require
the retired `launch_ddp.py` wrapper.

## Install and test

Install dependencies into the interpreter assigned to this method:

```bash
python -m pip install -r methods/SimulGenVAE/requirements.txt
```

Run the native tests from the method directory:

```bash
cd methods/SimulGenVAE
python -m pytest -q tests/
```

For architecture details, config-key migration, checkpoint internals, and known
limitations, see [CLAUDE.md](CLAUDE.md). The method overview is
[docs/methods/12_SimulGenVAE.md](../../docs/methods/12_SimulGenVAE.md), and the
suite-level config behavior is documented in
[docs/CONFIGURATION.md](../../docs/CONFIGURATION.md).
