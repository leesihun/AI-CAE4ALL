# Portable CPU inference bundle

[`inference/`](../../inference/) is stand-alone, CPU-only inference for
AI-CAE4ALL checkpoints. It does not depend on the rest of the repository: copy
the folder anywhere, install its `requirements.txt`, and run it. The Studio's
Deploy workspace runs it as well (see [GUI.md](../GUI.md#workspaces)).

| Supported | Not supported (use the native route) |
| --- | --- |
| `point_deeponet`, `deeponet`, `fno`, `transolver`, `meshgraphnets`, `meshgraphnets-v` with `latent_type global`, `sdfflow` | `meshgraphnets-v` with `latent_type local`/`hybrid`, legacy `chi-mgnflow`, MLP, SimulGen-VAE |

The bundle reads the family from the checkpoint, so there is no `--model` flag.
Unsupported checkpoints are rejected with an error that names the native route.

## Quick start

```bash
cd inference
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python run_inference.py --checkpoint model.pth --input scene.h5 --output out.h5
python run_inference.py --checkpoint sdfflow.pth --output out/ --num-samples 4   # SDFFlow: no --input
```

From Python, with one checkpoint per call:

```python
from cae_infer import infer
infer(checkpoint="model.pth", input="scene.h5", output="out.h5")
```

## Family detection

`cae_infer.detect_family()` reads the checkpoint's top-level keys, which are the
same keys each native `save_checkpoint` writes. Rows are checked in order:

| Checkpoint | Family |
| --- | --- |
| `schema_version == 'deeponet_repo_v1'` | `neural_operator` (`selected_model` picks point_deeponet / deeponet / fno) |
| `schema_version == 'sdfflow_infer_v1'`, or legacy `stage` `vae`/`fm` identified as SDFFlow | `geometry` |
| `checkpoint_version` present | `transolver` |
| `model_config` has `message_passing_num` and `chi-mgnflow`/`flow_*` keys | rejected (legacy HI-MGNflow) |
| ... and `latent_type` other than `global` | rejected (HI-MGNflow / HI-MGN-hybrid) |
| ... and `use_vae` | `meshgraphnets_v` |
| ... otherwise | `meshgraphnets` |

`stage='vae'` alone does not identify SDFFlow, because SimulGen-VAE uses the same
label.

## CLI

```text
--checkpoint PATH       required
--output PATH           required: .h5 for rollout families, a directory of STLs for sdfflow
--input PATH            input mesh HDF5 (not used by sdfflow)
--device {cpu,auto}     always resolves to CPU; kept for CLI parity
--timesteps N           rollout steps (default: the input's full trajectory)
--query-chunk-size N    Neural_Operator query decode chunking (0 = off)
--split-seed N          Neural_Operator: sensor-sampling seed (default 42; see Limitations)
--num-samples N         sdfflow: geometries to sample (1)
--ode-steps N           sdfflow: flow-matching ODE steps (50)
--cfg-scale F           sdfflow: classifier-free guidance scale (1.0)
--mc-resolution N       sdfflow: marching-cubes grid (128)
--seed N                sdfflow: sampler seed
--cond-values "a,b"     sdfflow: target conditions in the checkpoint's cond_names order
```

A family driver ignores flags it does not use.

## Layout

```text
inference/
  run_inference.py        CLI entrypoint
  pyinstaller.spec        one-folder .exe build
  requirements.txt        pinned CPU versions
  cae_infer/
    __init__.py           infer(), detect_family()
    cli.py                argparse -> infer()
    registry.py           family -> driver; one family per process
    common/               device.py, hdf5_io.py
    families/             neural_operator/ transolver/ meshgraphnets/ meshgraphnets_v/ geometry/
```

Each family folder vendors its method's `model/` and `general_modules/` under
their original module names, so the vendored files need no import changes.
Those names mean different code in each family, so **only one family can be
loaded per process**, and `registry.py` raises if a second one is requested. The
CLI and `infer()` each handle one checkpoint per call, so neither hits this
limit. There is no automatic re-vendoring: when a method's model code changes,
copy its files into the family again and keep `driver.py` and `cae_infer/*` as
they are.

## Why CPU only

The bundle never ships CUDA wheels or `torch_cluster`, and `resolve_device()`
always returns CPU. The MeshGraphNets world-edge search uses scipy's `cKDTree`.

## SDFFlow: one combined checkpoint

The FM checkpoint written by `methods/SDFFlow/training_profiles/train_fm.py`
embeds its frozen VAE (`ckpt['vae']`), so one `.pth` is a complete artifact. An
older VAE + FM pair can be merged from the repo root:

```bash
python methods/SDFFlow/scripts/merge_sdfflow_checkpoint.py \
    --vae <sdfflow_vae.pth> --fm <sdfflow_fm.pth> --output <sdfflow.pth>
```

Without an embedded `vae` block, the driver falls back to `ckpt['vae_modelpath']`.
This works only while that file is still reachable, so hand others the merged
file.

## Building the .exe

```bash
cd inference
pip install pyinstaller
pyinstaller pyinstaller.spec
```

This builds `dist/run_inference/run_inference.exe`, a one-folder build. Copy the
whole `dist/run_inference/` folder; the target machine needs no Python. The
Studio's Deploy workspace has a **Build .exe with PyInstaller** button that
writes under `studio/runtime/deploy/`.

## Limitations

- **Parity.** Each family was checked against its native inference code on CPU
  with the pinned torch. All families matched bit-for-bit except Transolver,
  which differs by about 5e-6 relative error. Treat Transolver as correct to
  about 1e-5. A different torch build, or GPU against CPU, adds normal fp32
  drift of about 1e-6. Re-run parity checks before moving off the pinned
  versions in `requirements.txt`.
- **`split_seed` is not stored** in Neural_Operator checkpoints. The driver
  assumes 42, the training default. Pass `--split-seed` for a point_deeponet
  trained with another value; otherwise the sensor subsets differ from training.
- **`coarse_world_edges` is not stored** in MeshGraphNets checkpoints, but with
  `use_multiscale` and `use_world_edges` it changes weight shapes. The driver
  defaults it to `False`. If that is wrong, `load_state_dict(strict=True)` fails
  with a shape mismatch. Pass `infer(..., coarse_world_edges=True)` to override.
- **World edges always use `cKDTree`.** For a checkpoint trained with the
  `torch_cluster` radius backend, check parity before trusting the results; the
  two backends are not guaranteed to be bit-identical.
- **MeshGraphNets-V** samples the prior latent once per trajectory and holds it
  fixed, as the native `rollout.py` does.
