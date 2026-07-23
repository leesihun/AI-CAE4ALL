# AI-CAE4ALL inference bundle

Stand-alone, CPU-only inference for AI-CAE4ALL checkpoints. This folder has
**no dependency on the rest of the AI-CAE4ALL repository** — copy it
anywhere, install `requirements.txt`, and run.

Supported checkpoints: `point_deeponet`, `deeponet`, `fno`, `gino`
(Neural_Operator), `transolver`, `meshgraphnets`, `meshgraphnets-v`, and the
geometry generator `sdfflow`. You don't need to say which one — the
checkpoint file tells the tool.

## Quick start

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python run_inference.py --checkpoint model.pth --input scene.h5 --output out/
```

Or, once built (see "Building the .exe" below), on a machine with no Python
installed at all:

```
run_inference.exe --checkpoint model.pth --input scene.h5 --output out/
```

For SDFFlow (generative, no `--input`):

```bash
python run_inference.py --checkpoint sdfflow.pth --output out/ --num-samples 4
```

## How family detection works

`cae_infer.detect_family()` inspects the checkpoint's own top-level keys —
the same keys each method's native `save_checkpoint` already writes — and
picks the right forward pass. No filename convention, no `--model` flag:

| Checkpoint shape | Family | Architectures |
| --- | --- | --- |
| `schema_version == 'deeponet_repo_v1'` | `neural_operator` | point_deeponet, deeponet, fno, gino (`checkpoint['selected_model']` picks which) |
| `schema_version == 'sdfflow_infer_v1'` or `stage in {'vae','fm'}` | `geometry` | SDFFlow (VAE + flow-matching) |
| `checkpoint_version` present | `transolver` | Transolver |
| `model_config` has `use_vae` | `meshgraphnets_v` | MeshGraphNets (variational) |
| `model_config` has `message_passing_num`, no `use_vae` | `meshgraphnets` | MeshGraphNets |

## CLI flags

```
--checkpoint PATH        Required. The .pth file.
--input PATH             Input HDF5 mesh dataset (MGN contract). Not used by sdfflow.
--output PATH            Required. Output .h5 (rollout families) or a directory (sdfflow STLs).
--device cpu             This bundle is CPU-only; kept for CLI parity.
--timesteps N            Rollout steps (default: full trajectory from the input file).
--query-chunk-size N     Neural_Operator memory control for point/query decode (0 = no chunking).
--num-samples N          sdfflow only: number of geometries to sample (default 1).
--ode-steps N            sdfflow only: flow-matching ODE steps (default 50).
--cfg-scale F            sdfflow only: classifier-free guidance scale (default 1.0).
--mc-resolution N        sdfflow only: Marching Cubes grid resolution (default 128).
--seed N                 sdfflow only: sampler seed.
--cond-values "a,b,c"    sdfflow only: comma-separated target condition values, in the
                         checkpoint's cond_names order. Omit for unconditional generation.
```

Each family driver ignores flags it doesn't use.

## Library usage

```python
from cae_infer import infer

infer(checkpoint="model.pth", input="scene.h5", output="out/")
```

## Why CPU-only

The bundle never ships CUDA wheels or `torch_cluster`. Device is always
`torch.device('cpu')`. This removes the biggest packaging risk for a
one-folder, zero-setup deliverable; GINO's neighbor search runs its scipy
`cKDTree` path instead of the (CUDA-only-relevant) `torch_cluster`
accelerator.

## Folder structure

```
inference/
  run_inference.py          stand-alone CLI entrypoint
  cae_infer/
    __init__.py              public API: infer(), detect_family()
    cli.py                   argparse -> infer()
    registry.py               family -> driver dispatch, one-family-per-process
    common/                   device.py, hdf5_io.py -- shared across rollout families
    families/
      neural_operator/        point_deeponet, deeponet, fno, gino (one shared driver)
      transolver/
      meshgraphnets/
      meshgraphnets_v/
      geometry/                sdfflow
  pyinstaller.spec
  requirements.txt
```

### One family per process

Each family folder is a self-contained mini-package that keeps its
**original internal module names** (`model.*`, `general_modules.*`) so the
vendored files needed zero import rewriting. Because `model`/`general_modules`
mean different things per family, only **one family may be imported per
Python process** — `registry.py` enforces this and raises if you try to load
a second, different family in the same process. Both entrypoints already
satisfy this naturally: the CLI and `infer()` each handle one checkpoint per
call.

## SDFFlow checkpoints: one combined `.pth`

New checkpoints written by the current `Geometry_generation/training_profiles/train_fm.py`
already embed the frozen VAE they were trained against (`ckpt['vae']`), so a
single `sdfflow_fm.pth` (or however you rename it) is a complete,
self-contained inference artifact.

If you have an **older** checkpoint pair (separate `sdfflow_vae.pth` +
`sdfflow_fm.pth`), merge them once, from the `Geometry_generation/` repo
(not from this bundle):

```bash
python Geometry_generation/merge_sdfflow_checkpoint.py \
    --vae outputs/ex1/sdfflow_vae.pth --fm outputs/ex1/sdfflow_fm.pth \
    --output outputs/ex1/sdfflow.pth
```

The driver also has a fallback that reads `ckpt['vae_modelpath']` directly if
no `'vae'` block is present, but that requires the original VAE file to still
be reachable at that path — the merged single file is what you actually want
to hand to someone else.

## Building the .exe

```bash
pip install pyinstaller
pyinstaller pyinstaller.spec
```

Produces `dist/run_inference/run_inference.exe` (a one-folder build — faster
startup than one-file, and easier to inspect if something's missing). Copy
the whole `dist/run_inference/` folder to hand off; it needs no Python
install on the target machine.

## Rebuilding this bundle from the main repo

`inference/` is not meant to be hand-edited file-by-file when the source
repos change underneath it — see `INFERENCE_BUNDLE_PLAN.md` (repo root)
section 10 for the intended `rebuild_bundle.py` automation. It has not been
written yet; today, re-vendoring is manual (re-copy the family's `model.py`/
`general_modules.py` files, keep `driver.py` and `cae_infer/*` as hand-written
glue).
