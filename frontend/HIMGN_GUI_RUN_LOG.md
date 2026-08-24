# HI-MGN profile built and run entirely through the Studio GUI — problem log

**Date:** 2026-08-23
**Scope:** kill the running AI-CAE4ALL Studio → restart it → build a HI-MGN
profile by hand on a blank canvas → run it. Every action was performed through
the browser UI (palette clicks, socket clicks, inspector fields, the config
sheet, `Run pipeline`). No config file was written behind the GUI's back.

**Environment:** Windows 10, Studio server `frontend/start_studio.py 8099`
(PID 79268, restarted for this session), Chromium via Playwright at 1600×1000.

---

## 0. Restart

| Step | Result |
| --- | --- |
| Killed PID 53424 (`start_studio.py 8099`) and PID 24588 (a leftover `studio_server` on 8100 from an earlier session) | ports 8080–8200 clear |
| Restarted `python start_studio.py 8099` | `HTTP 200`, 19 866 bytes on `/index.html` |
| Runtime health in the top bar | `11/11 ROUTES LIVE` |

No problems in the restart itself. Worth noting that **two** Studio servers were
alive before the restart (8099 and a scratch one on 8100); nothing in the UI
tells you that another instance is holding a port.

---

## 1. Problems found

### P1 — `Fit graph` silently fails on a tall graph: blocks land off-stage, and the top one becomes unclickable under the toolbar

**Severity: high (blocks graph editing during the normal build order).**

Reproduction, keyboard only (`L` = Auto layout, `F` = Fit graph), 1600×1000,
every container scroll reset before measuring so nothing but the app moved the
view:

```
blank canvas -> add 7 blocks -> L -> F
zoom pinned at 45%   stage = [top 138, bottom 1000]   stage scroll = [0, 0]
source_hdf5_1     top =    7  bottom =  134   <-- entirely above the stage
output_export_7   top = 1009  bottom = 1136   <-- entirely below the stage
```

The browser itself confirms the top block is unreachable — `elementFromPoint`
on the centre of its output socket returns the toolbar's template `<select>`,
not the socket:

```
first socket reachability: {node: source_hdf5_1, port: data, at: [787, 90],
                            hits: SELECT, ok: false}
```

That is exactly what stopped the build: the first link
(`ex9.h5 → MeshGraphNets`) could not be made at all —
`<header class="canvas-toolbar"> intercepts pointer events` — and the pipeline
could only be finished by dragging the canvas down by hand and retrying
(6 → 7 edges after one `pan(0, +160)`).

**Root cause** — [frontend/src/graph.js:762-783](src/graph.js#L762-L783):

```js
const scale = Math.min(1.1, Math.max(.45, Math.min(
  (rect.width - 90) / Math.max(1, maxX - minX),
  (rect.height - 110) / Math.max(1, maxY - minY)
)));
state.view.y = (rect.height - (maxY - minY) * scale) / 2 - minY * scale;
```

The `.45` zoom floor wins whenever the graph is taller than ≈1 900 world px, but
the centring below then uses that clamped scale *as if it had fit*, so the
overflow is split symmetrically — half above the stage, half below. Above the
stage means under the toolbar, where clicks die.

**Why the normal build order walks straight into it:** `Auto layout` places
blocks by dependency level, so pressing it *before* wiring — the natural order
when you have just dropped seven blocks — puts every block at level 0, i.e. one
2 500 px-tall column (verified: all 7 blocks at `left = 690`). The tooltip says
"Tidy the blocks into readable columns", which is not what happens for an
unwired graph.

**Scope check:** the five shipped templates (`himgn`, `physics`, `simulgen`,
`generative`, `parametric`) all fit correctly (`FITS` at 59–74 % zoom), so this
only bites graphs that need less than 45 % — which is what Auto-layout-before-
wiring produces.

**Suggested fix:** when the computed scale hits the floor, anchor the view at the
graph's top-left below the toolbar instead of centring — or lift the floor for
`fitGraphView` only, since "show me everything" is the one place where zooming
past 45 % is exactly what was asked for.

---

### P5 — The live runtime drawer covers the canvas zoom controls and eats their clicks

**Severity: high (recurrence of the occlusion-bug family).**

While a job is live, `#runtimeDrawer` is painted over the bottom-right zoom
cluster. Verified with `document.elementFromPoint` — this is the browser's own
hit-testing, not an automation artifact:

```
zoomIn    centre(1563,965) -> button#runtimeCancel        BLOCKED
zoomOut   centre(1429,965) -> button#runtimeCopyLog       BLOCKED
fitGraph  centre(1469,965) -> button#runtimeCopyLog       BLOCKED
zoomLevel centre(1516,965) -> button#runtimeCancel        BLOCKED
runtimeDrawer rect 866,626 -> 1586,986   class="runtime-drawer open"
```

**Minimising the drawer does not help** — the collapsed header still sits on the
same four controls:

```
zoomIn    -> button#runtimeDismiss        BLOCKED
zoomOut   -> header (runtimeDrawer)       BLOCKED
fitGraph  -> button#runtimeExperiments    BLOCKED
zoomLevel -> button#runtimeMinimize       BLOCKED
runtimeDrawer rect 866,933 -> 1586,986   class="runtime-drawer open minimized"
```

So for the entire duration of a training run — precisely when you want to zoom
in on the block that is running — `+`, `−`, and `⌗` are dead. The keyboard
shortcuts (`+`, `−`, `F`) still work, which is the only reason the measurements
above could be taken at all.

**Suggested fix:** shift the `.zoom` cluster up (or the drawer's reserved space
down) whenever `.runtime-drawer` is open, the same way the inspector already
reflows the stage.

---

### P2 — The shipped “HI-MGN” preset produces a configuration that cannot pass preflight

**Severity: high (the preset for the flagship model is unusable on its own).**

On a fresh MeshGraphNets block, `Apply preset → HI-MGN` changes exactly three
keys:

```
use_multiscale    -> True
coarsening_type   -> voronoi_seedmean
multiscale_levels -> 3
```

Running the authoritative preflight on that result (`Run preflight` in the
config sheet) returns **4 errors**:

```
[MGN-MULTI-REQ] voronoi_clusters: voronoi_clusters is required when use_multiscale=True.
[MGN-MULTI-REQ] mp_per_level:     mp_per_level is required when use_multiscale=True.
[CFG-REQ-001]   input_var / output_var: required for meshgraphnets train
```

`voronoi_clusters` and `mp_per_level` are *never* offered by the preset even
though the launcher spec makes both mandatory the moment `use_multiscale=True`
([cae_suite/specs/meshgraphnets.py:95-163](../cae_suite/specs/meshgraphnets.py#L95-L163)),
and `mp_per_level` must contain `2 × levels + 1` entries — so the preset's
`multiscale_levels 3` silently demands a 7-entry list the user has to invent.

The comment above the preset in [frontend/src/config.js:296](src/config.js#L296)
shows the `voronoi` → `voronoi_seedmean` alias was already fixed once; the
companion keys were not.

**Both multiscale presets have it**, checked by applying each one to a fresh
MeshGraphNets block:

| preset | use_multiscale | coarsening_type | multiscale_levels | voronoi_clusters | mp_per_level |
| --- | --- | --- | --- | --- | --- |
| Flat MGN | False | none | — | — | — (not needed) |
| **HI-MGN** | True | voronoi_seedmean | 3 | **UNSET** | **UNSET** |
| **BSMS-GNN** | True | bfs | 3 | **UNSET** | **UNSET** |

`MGN-MULTI-REQ` requires all four keys whenever `use_multiscale=True` —
`voronoi_clusters` included, even for `bfs`
([cae_suite/specs/meshgraphnets.py:93-102](../cae_suite/specs/meshgraphnets.py#L93-L102)).

**Worse, the config sheet does not surface the gap.** Its own diagnostics line
after applying HI-MGN reads only

```
Missing required for train: dataset_dir, input_var, output_var
```

The two conditional keys are not in the sheet's `required` set, get no "required"
badge, and are never mentioned — so a user has no way to know they are missing
until the authoritative preflight (≈6 s) or the run itself rejects the config.

**Suggested fix:** make the preset self-consistent, e.g.

```js
if (preset === "mgn_hi") values = {
  use_multiscale: "True", coarsening_type: "voronoi_seedmean",
  multiscale_levels: "2", voronoi_clusters: "500, 100", mp_per_level: "4, 6, 8, 6, 4"
};
```

which is what the checked-in `configs/MeshGraphNets/ex9/config_train_hi_plasticity.txt`
and the shipped `himgn` pipeline template already use.

---

### P8 — Setting your own `inference_output_dir` makes the Studio blind to the results, and the pipeline fails with a message that is factually wrong

**Severity: high — this is what killed the run.**

The run trained and inferred successfully, then died at step 3 of 4:

```
[studio] Step 1 wrote 8 result file(s) to MeshGraphNets/outputs/train/0/2.
[studio] Step 2/4: MeshGraphNets · inference
    ... Rollout inference complete. Processed 87 scene(s) = 87 output file(s).
[studio] Evaluate Predictions · score skipped: prediction_path is empty because the block it reads from produced no output.
[studio] Pipeline failed.
```

Note what is missing: **step 2 never got a `[studio] Step 2 wrote N result file(s)` line**, while step 1 did. The Studio concluded the inference block "produced no output" — but it demonstrably did:

```
$ ls output/meshgraphnets/studio_himgn/inference/ | wc -l
87
rollout_sample10_steps19.h5, rollout_sample11_steps19.h5, ...   (0.5 MB each)
```

The persisted job record makes it unambiguous — the train step got its results
pinned, the inference step got `None`:

```
frontend/runtime/jobs/9280c5c9b4ad/job.json     status: failed
  step 1: 'MeshGraphNets · train'      results='MeshGraphNets/outputs/train/0/2'  samples=8
  step 2: 'MeshGraphNets · inference'  results=None                              samples=None
  step 3: 'Evaluate Predictions · score'  results=None
  step 4: 'Export Results · package'      results=None
```

**Root cause** — [frontend/studio_backend/prediction_preview.py:554-564](studio_backend/prediction_preview.py#L554-L564):

```python
def outputs_since(repository: Path, since: float, limit: int = 12) -> list[dict[str, Any]]:
    roots = [repository / name for name in ("outputs", "output")]
    return _scan_roots([root for root in roots if root.is_dir()], limit, since=since)
```

Results are only ever looked for **inside the method repository** —
`MeshGraphNets/outputs/` and `MeshGraphNets/output/`. My profile set

```
inference_output_dir         ../output/meshgraphnets/studio_himgn/inference
```

which resolves to `SUITE_ROOT/output/...`, outside both scanned roots. Zero
results found → `prediction_path` empty → the evaluation step is skipped → the
whole pipeline is marked failed, after ~25 minutes of real training.

The function's docstring justifies the scan-based approach with *"guessing the
path from config keys cannot work, because the epoch number in
`outputs/<split>/<gpu>/<epoch>/` is only known to the training loop"*. That is
true for the **train** step. It is not true for the **inference** step, where
`inference_output_dir` is a config key the user set explicitly and the native
process even echoes back (`Saving results to: …\studio_himgn\inference\…`).

**Why the shipped template never hits this:** neither `EX9_MESH` nor the
MeshGraphNets block defaults set `inference_output_dir`, so the launcher applies
its own default (`outputs/rollout`, from `defaults_by_mode` in the spec), which
lands *inside* `MeshGraphNets/outputs/` and is therefore found. The moment a user
uses the config sheet to point results somewhere of their own — an ordinary
thing to do, and the key is right there in the sheet — the analysis chain breaks.

**Suggested fix:** the answer is already in the payload the caller holds.
[state.py:654-658](studio_backend/state.py#L654-L658) reads
`item["preflight"]["route"]["repository"]`; the same preflight payload carries
`resolved_paths` ([state.py:355](studio_backend/state.py#L355)), and the
MeshGraphNets spec declares
`PathRule("inference_output_dir", PathKind.OUTPUT_DIR, {"inference"})`
([specs/meshgraphnets.py:232](../cae_suite/specs/meshgraphnets.py#L232)) — so for
an inference step the resolved output directory is right there:

```python
resolved = (item.get("preflight") or {}).get("resolved_paths") or {}
configured = resolved.get("inference_output_dir")
found = scan(SUITE_ROOT / configured) if configured else outputs_since(SUITE_ROOT / repository, step_started)
```

Falling back to the repository scan only when the key is absent keeps the
existing behaviour for the train step, where the epoch-numbered directory really
is unknowable from the config. Failing that, the skip message must at least stop
claiming the block "produced no output" when the Studio simply did not look
where the output went.

**Verification:** the run was repeated with this profile unchanged except for
clearing `inference_output_dir` — see §4.

---

### P9 — After a failure the canvas forgets everything: every block is reset to `idle`

**Severity: medium.**

Once the job failed, all seven blocks were repainted `idle`:

```
node model_meshgraphnets_3   class="node idle"    <-- this step SUCCEEDED (25 min of training)
node run_inference_4         class="node idle"    <-- this step SUCCEEDED (87 rollouts)
node evaluate_predictions_5  class="node idle"    <-- this is the step that FAILED
```

[frontend/src/run.js:130](src/run.js#L130) collapses every non-running,
non-completed job to `idle`:

```js
node.status = job.status === "running" ? "running" : job.status === "completed" ? "complete" : "idle";
```

So the canvas — the thing the user is looking at — shows no trace of which step
failed or how far the run got. The information is available (the job carries
`step`/`total_steps` and the per-step records), it is simply discarded.

The run banner is stale for the same reason: after the terminal state the banner
still read `Real job · running | MeshGraphNets · inference` while the job was
`failed` at step 3/4. Only the toast was correct.

---

### P10 — The suite's own checkpoint cannot be inspected by the suite's own checkpoint probe

**Severity: medium (a preflight safety net that never fires for MeshGraphNets).**

Every inference step in this run reported:

```
[CHECKPOINT-PROBE-002] [modelpath] Safe metadata inspection was unavailable for modelpath:
UnpicklingError: Weights only load failed.
```

Checked directly against the checkpoint MeshGraphNets had just written:

```
torch.load('output/meshgraphnets/studio_himgn/himgn_ex9.pth', weights_only=True)
-> UnpicklingError: Unsupported global: GLOBAL numpy._core.multiarray._reconstruct
                    was not an allowed global by default
```

The checkpoint stores its normalization statistics as **numpy** arrays (the same
`delta_mean` / `delta_std` the rollout prints), and `numpy._core.multiarray._reconstruct`
is not in torch's default `weights_only` allowlist. So
[cae_suite/checkpoint_probe.py:25](../cae_suite/checkpoint_probe.py#L25) can never
read an MGN checkpoint, and the model/stage/normalization consistency check it
exists to perform is silently skipped on every MeshGraphNets inference run.

**Suggested fix:** `torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct, numpy.dtype, ...])`
in the probe before loading — it stays `weights_only=True`, so the safety
property is preserved — or save the stats as tensors on the training side.

---

### P6 — The runtime log is 96 % progress-bar frames, so the real messages are unreadable

**Severity: medium (the live log is the only window into a running job).**

Measured on this run, 10 minutes into training:

| | |
| --- | --- |
| `frontend/runtime/jobs/9280c5c9b4ad/run.log` | 429 713 bytes, 5 546 lines |
| lines that are **not** a tqdm frame | **200** (3.6 %) |
| the drawer's visible tail | 1 549 lines, of which **17** are not tqdm frames |

Every single tqdm repaint becomes its own log line. The cause is in
[frontend/studio_backend/state.py:600-619](studio_backend/state.py#L600-L619):

```python
process = subprocess.Popen(..., text=True, bufsize=1, ...)
for line in process.stdout:
    self._append_log(job, line)
```

Text-mode iteration splits on universal newlines, and that includes the bare
`\r` tqdm uses to repaint in place. A terminal collapses those into one line; the
Studio keeps all 3 420 of them per epoch, both on disk and in the drawer.

The consequence is not cosmetic — the lines a user actually needs are drowned:

```
Epoch 0/3 TrainOpt: 6.59e-01 Valid: 4.09e-01 LR: 1.00e-04
  Test loss: 4.06e-01 (9.0s)
  Train reconstruction loss: 1.26e-01
```

Three useful lines out of 1 549 on screen. At this rate a real 500-epoch run
writes roughly 70 MB of log, ~96 % of it duplicated frames.

**Suggested fix:** when a chunk ends in `\r` rather than `\n`, replace the
previous line instead of appending it — the same in-place update a terminal
does. Both the drawer and the persisted `run.log` then stay readable, and the
metric extractor in `training_metrics.py` keeps working unchanged.

---

### P3 — `POST /api/preflight` answers `422` for a config that merely has validation errors

**Severity: low (cosmetic, but pollutes the console).**

A config with preflight errors comes back as `422 Unprocessable Entity`, which
the browser logs as `Failed to load resource: the server responded with a status
of 422`. The UI handles it correctly and renders the diagnostics, so this is
only noise — but it makes a *successful* diagnostic round-trip look like a
frontend failure in devtools.

---

### P7 — `message_passing_num` is presented as a live knob although HI-MGN ignores it

**Severity: low (misleading, and it is on the HI-MGN hot path).**

The native trainer says so itself, in this very run's log:

```
  Level 0 pre:  4 blocks
  Level 1 pre:  6 blocks
  Coarsest:    8 blocks
  [message_passing_num is IGNORED when use_multiscale=True]
```

The config sheet, with `use_multiscale True` set, still shows:

```
message_passing_num   value='15'   badge='SET'
help: "Manual input is retained because the live spec does not publish a closed value set for this field."
```

No `inactive` badge, no mention that the value is dead — even though the sheet's
own schema note promises "Shared-family inactive and rejected keys remain visible
for diagnostic honesty". Both the shipped `himgn` template and the HI-MGN preset
path leave `message_passing_num 15` in the config, so anyone tuning HI-MGN will
reasonably assume that number does something. It does not; `mp_per_level` owns
the depth.

**Suggested fix:** extend `keyDisposition` so `message_passing_num` reports
`inactive` while `use_multiscale=True`, the same way variant-specific keys
already do.

---

### P4 — While preflight is in flight the diagnostics panel keeps showing the previous verdict

**Severity: low (misleading for ~6 s).**

`Run preflight` disables the button and relabels it "Running real preflight…",
which is good, but the diagnostics panel underneath still shows the *previous*
run's verdict verbatim — including stale field names and paths from before the
edit. During the ~6.4 s the authoritative preflight takes on ex9, a user reading
the panel sees a confident, wrong answer. Clearing the panel (or greying it) on
click would remove the ambiguity.

*(Noted for completeness: this initially looked like a stale-result bug. Measured
with an explicit response wait it is only the in-flight window —
`POST /api/preflight` → `200` in 6.4 s, and the panel then repaints correctly.)*

---

## 2. Things that worked correctly (worth recording)

* **Blank canvas + palette** — all 23 palette blocks list correctly; clicking one
  adds exactly one node; block IDs are unique.
* **Click-to-connect** — 6 of 7 links were made by clicking an output socket then
  its input socket, with compatible sockets highlighted; the only failure was P1's
  occluded block.
* **`parallel_mode` dropdown correctly refuses `single`.** The options are
  `ddp / model_split`, matching `MGN-PARALLEL-001` in the spec. This is the
  config-contract drift class of bug from a previous audit, and here the GUI was
  right — my own hand-typed value was wrong.
* **The config sheet is complete** — every key of the 35-key HI-MGN profile had an
  editable field; nothing had to be typed into the raw `.txt` pane.
* **Authoritative preflight on the finished profile: 0 errors, 1 warning,
  1 notice**, with the dataset probe confirming the real ex9 contract:
  `sample_count 900`, `nodal_shape [7, 20, 3131]`, `edge_shape [2, 6130]`.

---

## 3. The profile that was built

`HI-MGN hand-built (ex9)` — 7 blocks, 7 links:

```
dataset/ex9.h5 ─────────────► MeshGraphNets (HI-MGN) ──model──► Inference Run ──pred──► Evaluate ──report──► Export
                                     └──metrics──► Train Metrics                ▲
dataset/ex9_infer.h5 ──────────────────────────────────data────┘                │
                     └──────────────────────────────────truth──────────────────┘
```

Trainer configuration (35 keys set, typed through the config sheet):

```
model                        meshgraphnets
mode                         train
gpu_ids                      0
parallel_mode                ddp
dataset_dir                  ../dataset/ex9.h5
infer_dataset                ../dataset/ex9_infer.h5
modelpath                    ../output/meshgraphnets/studio_himgn/himgn_ex9.pth
log_file_dir                 ../output/meshgraphnets/studio_himgn/log
inference_output_dir         ../output/meshgraphnets/studio_himgn/inference
input_var                    2
output_var                   2
cond_var                     2
edge_var                     8
feature_loss_weights         1.0, 1.0
positional_features          4
use_node_types               False
infer_timesteps              19
split_seed                   42
use_multiscale               True
coarsening_type              voronoi_seedmean
multiscale_levels            2
voronoi_clusters             500, 100
mp_per_level                 4, 6, 8, 6, 4
message_passing_num          15
latent_dim                   128
use_world_edges              False
training_epochs              3
batch_size                   4
learningr                    0.0001
weight_decay                 0.0001
warmup_epochs                1
val_interval                 1
use_amp                      True
use_ema                      True
ema_decay                    0.99
num_workers                  2
```

---

## 4. Run

Launched with `Run pipeline` in the top bar. The confirmation dialog listed the
four steps the GUI derived from the graph, which was correct:

```
Execute the real AI-CAE4ALL launcher?
1. MeshGraphNets · train      2. MeshGraphNets · inference
3. Evaluate Predictions · score   4. Export Results · package
```

Job `9280c5c9b4ad`, PID 8832 → 68472, 18:29:19 → 18:54:44 (25 min 25 s).

### What actually happened

| Step | Result |
| --- | --- |
| 1/4 MeshGraphNets · train | **succeeded** — 3 epochs, 8 result files |
| 2/4 MeshGraphNets · inference | **succeeded** — 87 rollouts, 19 steps each |
| 3/4 Evaluate Predictions · score | **skipped, and the run was marked failed** — see P8 |
| 4/4 Export Results · package | never reached |

**Final status: `failed` at step 3/4** — not because the model or the launcher
failed, but because the Studio could not find results it had just written to the
directory the user configured (P8). Run 2 below repeats this with that one field
cleared and completes 4/4, which is what pins the cause.

### Training was genuinely healthy

```
2-level Voronoi hierarchy: [500, 100] clusters, mp_per_level 4/6/8/6/4
  Level 0 pre: 4 blocks | Level 1 pre: 6 | Coarsest: 8 | Level 1 post: 6 | Level 0 post: 4
[mscache] Building hierarchy cache for 900 samples (16 workers) → done in 15s (0.4 GB)
Dataset loaded: 17100 AR pairs → split 720 train / 90 val / 90 test

Epoch 0/3  TrainOpt 6.59e-01  Valid 4.09e-01  LR 1.00e-04   Test 4.06e-01
Epoch 1/3  TrainOpt 2.25e-01  Valid 1.58e-01  LR 5.00e-05
Epoch 2/3  TrainOpt 1.42e-01  Valid 1.23e-01  LR 1.00e-04   Test 1.22e-01
Training finished. Final model saved at epoch 2 with validation loss 1.23e-01
```

Loss fell 5.4× in three epochs and the checkpoint was written to the configured
path. Inference then rolled out all 87 held-out scenes in 0.58 s each
(0.031 s/step), writing `rollout_sample*_steps19.h5` (0.5 MB each) to
`output/meshgraphnets/studio_himgn/inference/`.

So the HI-MGN profile itself is correct and the native run is sound; the failure
is entirely in the Studio's result-plumbing.

### Run 2 — the controlled experiment that proves P8, and the completed run

The run was repeated through the GUI with **exactly one change**: the
`inference_output_dir` field was cleared in the config sheet, so the launcher's
own default (`outputs/rollout`, inside the method repo) applied. Nothing else
about the profile, the graph, or the blocks was touched.

Job `6b9eb77330b8`, 18:58:03 → 19:22:46 (24 min 43 s) — **completed, 4/4**:

```
[studio] Step 1 wrote 8 result file(s) to MeshGraphNets/outputs/train/0/2.
[studio] Step 2 wrote 87 result file(s) to MeshGraphNets/outputs/rollout.   <-- detected this time
[studio] Running Evaluate Predictions · score (evaluation).
[studio] Evaluate Predictions · score wrote frontend/runtime/evaluation/f011fe3af795/report.json.
[studio] Running Export Results · package (export).
[studio] Export Results · package wrote frontend/runtime/exports/output-studio-run_001-e13e2108.json.
[studio] Pipeline completed.
```

Same model, same 87 rollout files, same 19 steps — the only difference is which
directory they landed in. That is P8 confirmed by experiment, not inference.

Training reproduced within noise of run 1:

```
Epoch 0/3  TrainOpt 6.61e-01  Valid 4.00e-01   Test 3.77e-01
Epoch 1/3  TrainOpt 2.24e-01  Valid 1.58e-01
Epoch 2/3  TrainOpt 1.41e-01  Valid 1.22e-01   Test 1.23e-01
Training finished. Final model saved at epoch 2 with validation loss 1.22e-01
```

And the evaluation block scored the rollouts against the held-out
`dataset/ex9_infer.h5` — 87 samples, 0 skipped, fields paired correctly
(`prediction_start 3`, `truth_start 3`, `num_fields 2`):

| metric | mean | median | p95 |
| --- | --- | --- | --- |
| relative L2 | 0.953 | 0.918 | 1.357 |
| MAE | 0.327 | 0.278 | 0.616 |
| RMSE | 0.613 | 0.519 | 1.069 |
| R² | −0.055 | 0.010 | 0.611 |

**Read those numbers as a plumbing check, not a result.** The profile was
deliberately set to `training_epochs 3` so the GUI path could be watched end to
end; three epochs of HI-MGN on ex9 is nowhere near converged, and a 19-step
autoregressive rollout compounds that error — R² ≈ 0 is the expected outcome, not
a defect. Node statuses on the canvas were correct this time (`complete` on the
trainer, inference, evaluation, and export blocks).

Final artifacts:

```
output/meshgraphnets/studio_himgn/himgn_ex9.pth                      (checkpoint)
MeshGraphNets/outputs/rollout/rollout_sample*_steps19.h5             (87 rollouts)
frontend/runtime/evaluation/f011fe3af795/report.json                 (+ per_sample_metrics.csv)
frontend/runtime/exports/output-studio-run_001-e13e2108.json         (export package)
```

---

## 5. Second session: longer training, and geometric optimal design

### P11 — The Studio writes its own bookkeeping keys into the block config, and the launcher rejects them

**Severity: medium.**

After any successful run, `run.js` copies the step's results onto the model
block ([run.js:164-165](src/run.js#L164-L165)):

```js
node.config.results_path = step.results;
node.config.results_samples = String(step.results_samples ?? "");
```

`rawConfig()` emits unknown keys as well as catalog keys, so those two land in
the flat `.txt` handed to the launcher:

```
frontend/runtime/configs/20260823-193601-meshgraphnets-train-saved-8d3ae497.txt
  41: results_path                 MeshGraphNets/outputs/train/0/2
  42: results_samples              8
```

and preflight answers:

```
Unknown keys will fail preflight: results_path, results_samples
[CFG-UNKNOWN-001] results_path: Unknown config key for meshgraphnets: results_path
```

Harmless today (warnings), an error under `--strict`, and it means every config
the Studio exports after a run is not a clean launcher config. They should live
outside `node.config`, or be filtered in `rawConfig`.

---

### P12 — Throughput collapses as batch size grows: batch 32 is 6× slower per sample than batch 4

**Severity: medium (it caps what any training run can achieve).**

Measured on an otherwise idle machine, real launcher path, tqdm rate read back:

| batch | workers | it/s | **samples/s** | min/epoch |
| --- | --- | --- | --- | --- |
| 4 | 2 | 8.04 | **32.2** | 7.1 |
| 4 | 6 | 7.95 | **31.8** | 7.2 |
| 4 | 12 | 8.34 | **33.4** | 6.8 |
| 8 | 4 | 1.53 | **12.2** | 18.6 |
| 32 | 2 | ~0.17 | **~5.5** | ~48 |

Sample throughput *falls* monotonically with batch size — the opposite of the
usual expectation, and GPU memory never exceeds 0.1 GB of an 8.6 GB card, so
nothing is saturated. Worker count moves nothing (2/6/12 within 4%), which
matches the known "workers don't help" behaviour of this loader. The practical
consequence: 7 min/epoch is a hard floor, so the checked-in 500-epoch budget for
ex9 is a ~58-hour run on this hardware.

*Method note:* the first two sweeps were invalid — `CTRL_BREAK` to the launcher
never reached the grandchild trainer, so 8 orphaned trainers plus 26 DataLoader
workers (≈45 GB RSS) accumulated and competed with each subsequent combo. The
numbers above are from the third sweep, which kills the whole process tree with
psutil and asserts free RAM between combos.

---

### P13 — Graph validation demands training data regardless of mode, so the shipped generative template can never run

**Severity: high.**

Pressing Run on the CAD Generator block of the stock "Generative design
optimization" pipeline gives:

```
Cannot run: SDFFlow: missing training data
```

even though the same SDFFlow configuration passes the authoritative preflight
with **0 errors** — because `sample` mode requires no dataset at all
(`required_by_mode["sample"] = {vae_modelpath, fm_modelpath, output_dir,
num_samples, seed, ode_steps, mc_resolution}`).

The cause is [validate.js:271-278](src/validate.js#L271-L278):

```js
spec.inputs.filter(port => port.required).forEach(port => {
  const linked = state.edges.some(edge => edge.toNode === node.id && edge.toPort === port.id);
  if (!linked) errors.push(`${spec.label}: missing ${port.label}`);
});
```

`required` is a static property of the block spec; the node's `mode` is never
consulted. And the shipped `generative` template has no dataset block at all, so
**the template cannot be run as delivered** — the only fix is to add an HDF5
block and wire it in purely to satisfy a check that does not apply.

---

### P14 — Nothing connects generation to optimization: the candidate table has to be built by hand

**Severity: high (the generative pipeline has no working middle).**

`run_optimization` requires a CSV with objective columns
([analysis.py:406-411](studio_backend/analysis.py#L406-L411)). SDFFlow's sample
mode writes STLs plus `sample_<seed>_meta.json` — which already contains exactly
the right numbers per candidate (`volume`, `area`, `extents`, `watertight`,
`faces`) — but **no component converts that JSON into the CSV**, and the
`generator → optimization` edge the template draws carries nothing usable.
The autofill only picks up a CSV if an upstream path already ends in `.csv`
([autofill.js:279](src/autofill.js#L279)).

For this session the table was aggregated from the generator's own metadata with
a small script. That step is the one part of the workflow that could not be done
in the GUI.

---

### P15 — The Optimization block never executes in a pipeline run

**Severity: high (the block, its API, and its template are all dead on the canvas).**

Running the full generative graph produced only two steps:

```
[studio] Starting Generative design optimization with 2 executable step(s).
[studio] Step 1/2: SDFFlow · sample
[studio] Step 2/2: SDFFlow · sample
[studio] Pipeline completed.
```

The Optimization and Export blocks were silently omitted, and the pipeline still
reported success. `analysisStep()`
([validate.js:212-243](src/validate.js#L212-L243)) handles exactly two node
types — `evaluate.predictions` and `output.export` — and returns `null` for
everything else, `optimize.design` included. Export then drops out too, because
its upstream produced no step so `analysisInput()` returns "".

This is despite the backend implementing the whole thing (`run_optimization`,
`/api/optimization`, `/api/optimization/schema`), the block having a full
inspector form, and an entire shipped template built around it.

---

### P16 — Every artifact picker is starved by the Studio's own saved configs

**Severity: high (it made the Optimization workspace unusable).**

The Optimization workspace's CSV picker offered **zero** options.
`/api/files?kind=artifact` returns 750 items, `truncated: true`, and the mix is:

```
{'.txt': 702, '.png': 17, '.json': 15, '.log': 16}     .csv: none
```

`walk_files` ([paths.py:78-98](studio_backend/paths.py#L78-L98)) walks
`(RUNTIME_ROOT, output, outputs)` in that order and **stops at `FILE_LIMIT = 750`
before sorting**:

```python
if len(records) >= FILE_LIMIT:
    records.sort(key=lambda item: item["modified"], reverse=True)
    return {"items": records, "truncated": True, ...}
```

`frontend/runtime/configs/` currently holds **2 023 `.txt` files** — one per
config save and per preflight, written by the Studio itself — and `.txt` is in
the artifact suffix set. They fill the cap before the walk ever reaches
`output/`, so no artifact under `output/` can appear anywhere in the GUI. The
newest-first sort is applied *after* truncation, so recency does not save it.

Two independent fixes: sort before truncating (or use a heap), and stop counting
the Studio's own scratch configs as artifacts.

---

## 6. Geometric optimal design — what was run and what came out

Built on the shipped "Generative design optimization" template, using the
SDFFlow checkpoints already trained on DeepJEB. The FM checkpoint is conditional
on `(bbox_x, bbox_z, volume, area)`, so the design space is real:

```
cond_names   mean       std        design points used (±1.5σ)
bbox_x       1.0658     0.0049     fixed — this shape family has a fixed footprint
bbox_z       0.6370     0.0170     held at mean
volume       0.2570     0.0758     0.1433  /  0.3708
area         4.4530     0.6714     3.4459  /  5.4602
```

Five generation batches through the GUI (one unconditional baseline of 24, four
conditioned corners of 12 each, `cfg_scale 2.0`, `ode_steps 50`,
`mc_resolution 128`): **72 candidates, 72/72 watertight**. The conditioning
demonstrably bites — the light-lean corner produced a median volume of 0.0998
against a request of 0.1433 (30% relative error) and 2.845 area against 3.446,
versus an unconditional mean volume of 0.252.

Objectives were deliberately set to fight each other — minimise `volume` (mass),
maximise `bbox_z` (section depth, the bending-stiffness proxy for a bracket) —
because minimising volume and area together would just pick the smallest shape.
Constraints: `bbox_x <= 1.08; bbox_y <= 1.81; watertight >= 1`.

Result (`frontend/runtime/optimization/c364d49ef7d4.json`): **72 rows → 70
feasible → 9 Pareto-optimal → 6 diverse selected**:

| design | volume | bbox_z |
| --- | --- | --- |
| seed2_000 | 0.14100 | 0.6449 |
| seed1_002 | 0.10425 | 0.6396 |
| seed1_010 | 0.08592 | 0.6255 |
| **seed1_001** | **0.05013** | **0.6208** |
| seed3_002 | 0.01253 | 0.3413 |
| seed3_006 | 0.00328 | 0.1352 |

The engineering read: `seed1_001` holds 96% of the deepest section
(0.621 vs 0.645) for **36% of the material** (0.050 vs 0.141). Below it the
front falls off a cliff — `seed3_002` saves another 75% of volume but loses
45% of section depth.

**Caveat, stated plainly:** these are *geometric* objectives only. No physics
ran on the candidates — the `physics evaluators` input of the Optimization block
is optional and nothing in the Studio evaluates a generated STL. `bbox_z` is a
stiffness *proxy*, not a stiffness. This is screening, which is exactly what the
Pareto adapter implements.

---

## 7. Fixes applied (third session)

Every finding above was fixed, plus five more found while fixing them. Each fix
is listed with how it was verified.

### The minimize button now actually minimizes

`Minimize` used to leave a 720 px full-width header bar parked over the canvas.
It now collapses to a pill in the corner, and the canvas controls stay reachable
in **both** states:

```
expanded    drawer 720x360 at (866,626)   zoomIn/zoomOut/fitGraph/zoomLevel: CLICKABLE
minimized   drawer 226x47  at (1360,939)  zoomIn/zoomOut/fitGraph/zoomLevel: CLICKABLE
restored    drawer 720x360 at (866,626)   zoomIn/zoomOut/fitGraph/zoomLevel: CLICKABLE
```

Previously all four were `BLOCKED` in both states (P5). The drawer now mirrors
its state onto `<body>` (`drawer-open` / `drawer-mini`) so CSS lifts the zoom
cluster by the drawer's real height, and drops it back to a pill's clearance
when minimized. The pill keeps the status dot, a truncated job title, and the
restore/dismiss buttons — enough to know a job is alive and to get back to it.

### Fix table

| # | Fix | Verified by |
| --- | --- | --- |
| P1 | `fitGraphView` may zoom below the manual floor (`FIT_MIN_ZOOM 0.22`) and anchors instead of centring when content still overflows | 7-block column now fits at 30% zoom, 0 blocks off-stage, first socket `CLICKABLE` |
| P2 | HI-MGN and BSMS-GNN presets emit `voronoi_clusters` + `mp_per_level` with the correct `2*levels+1` count | preset applied in the GUI → both keys present, 5 entries for levels=2 |
| P3 | `/api/preflight` returns 200 with `ok:false` instead of 422 | no console error on an invalid config; clients already branch on `ok` |
| P4 | Diagnostics panel says it is running instead of showing the previous verdict | visible during the ~6 s preflight |
| P5 | Runtime drawer no longer covers the canvas controls | table above |
| P6 | tqdm repaints collapsed by reading the pipe **binary** and folding `\r` frames (text mode had already rewritten them to `\n`) | 123 raw lines → 6 emitted, every real line kept |
| P7 | `message_passing_num` reports `inactive` under multiscale, with a help line naming `mp_per_level` | badge reads `INACTIVE` in the sheet |
| P8 | Inference results resolved from `preflight.resolved_paths["inference_output_dir"]`, falling back to the repository scan only when unset | already proven by the run-2 experiment; the fix removes the need for the workaround |
| P9 | Failed runs mark the stopping block `failed`, earlier blocks `complete`; the banner updates on terminal states | new `.node.failed` styling |
| P10 | Checkpoint probe allowlists numpy's array-reconstruction globals, keeping `weights_only=True` | probe now returns `ok:true` for MGN **and** SDFFlow checkpoints; preflight shows `Checkpoint: init_modelpath {'has_normalization': True, 'has_ema': True}` |
| P11 | Studio bookkeeping keys are excluded from the emitted flat config | `results_path`/`results_samples` no longer in the .txt; no CFG-UNKNOWN-001 |
| P12 | (documented, not fixed) batch-size throughput collapse — the measurements now inform the config rather than the default | see §5 |
| P13 | Graph validation asks the launcher's per-mode required set before demanding a dataset link | shipped generative template validates untouched |
| P14 | The generator step writes `candidates.csv` from its own `sample_*_meta.json` and publishes it as the step's result | 72 rows with `requested_*` design-point columns, produced by the backend |
| P15 | `analysisStep()` emits an `optimization` step; the backend registers `"optimization": run_optimization` | the block now appears as a pipeline step |
| P16 | `walk_files` sorts before truncating and excludes the Studio's own config scratch directory | artifact catalog went from 702 scratch `.txt` to 537 `.h5` / 82 `.json` / 72 `.stl` / 11 `.csv` |

### Five more defects found while fixing, and fixed

**P17 — importing `studio_backend.state` rewrote a live server's job records.**
`STATE = StudioState()` runs at module import, and recovery marked every job it
found in a running state as `interrupted`. A one-line import check in another
shell stamped the in-flight 60-epoch training run as interrupted (`pid: None`,
`finished_at: now`) while it went on training for another four hours. Recovery
now only interrupts a job whose recorded PID is genuinely gone.

**P18 — AR-RT (rollout training) crashed instantly on any 2-D problem.**

```
ar_rollout.py:288  deformed_pos = graph.pos + state[:, :3]
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1
```

`inference_profiles/rollout.py` has always padded when `input_var < 3`; AR-RT —
whose own docstring calls itself the training-time twin of that code — sliced
three columns unconditionally. Every 2-D dataset in the suite, ex9 included,
could not use rollout training at all. Both sites now go through a padding
helper, and AR-RT runs (15.2 min/epoch at batch 4).

**P19 — two Studio servers could bind the same port, and requests split between them.**
`allow_reuse_address = 1` (socketserver's default) means SO_REUSEADDR, which on
Windows lets a second process bind a port another process is actively listening
on. Both instances "succeeded" on 8099:

```
LocalPort 8099  OwningProcess 59260   (started 03:47, registry predates a spec edit)
LocalPort 8099  OwningProcess 101136  (started 04:18, current)
```

The symptom was maddening: a config key existed on disk, in the spec, and in the
served JavaScript, yet never appeared in the UI — because `/api/models` was
answered by the *stale* instance while static files came from the new one. The
server now refuses to share a port:

```
AI-CAE4ALL Studio could not use port 8099: [WinError 10048] ...
Try another port:
```

**P20 — config-sheet search only searched the selected section.** Typing an
exact key name showed nothing whenever that key lived in another section and the
current one happened to hold some other match — searching `modelpath` from
"Data & output" showed `init_modelpath` and hid `modelpath` itself. Search now
spans every section and says so.

**P22 — `resolved_paths` is empty for every step after the first, which silently
defeated two of the fixes above.** Dependency checks are deferred for later
launcher steps (`deferred = launcher_seen > 0` → `skip_filesystem=True`), and
path resolution belongs to the filesystem layer — so `preflight.resolved_paths`
is `{}` from step 2 onwards. The inference step is *always* step 2+, so the P8
fix as first written would have fallen straight back to the repository scan and
fixed nothing; the candidate-table publishing (P14) failed the same way, which
is how it was caught:

```
[studio] Step 2/4: SDFFlow · sample        <- generated 8 watertight shapes
[studio] Optimization · select skipped: csv_path is empty because the block it reads from produced no output.
[studio] Pipeline failed.
```

Both now read the key back out of the config text the step actually ran,
resolving it against the method repository (which is the cwd the native process
uses, and what the `../output/...` form in these configs is relative to).
`resolved_paths` is kept as the preferred source for step 1, where it is
populated.

*Two lessons worth keeping: a fix that is only verified in a unit test can still
be inert in the pipeline, and "the analysis step was skipped" was again the only
symptom of a plumbing failure three layers away.*

**P23 — the optimization step was dropped whenever its producer ran in the same
pipeline.** `analysisInput()` returns `@results:<node_id>` for a block that has
not written yet, and my first version of the optimize.design step required the
upstream value to end in `.csv` — which `@results:generator` never does. So the
step existed only when the CSV already sat on disk, i.e. never for the shipped
template. The reference form is now accepted directly.

### Enhancement: the sheet now warns about keys a switch just made mandatory

`requiredFor` deliberately mirrors only the spec's per-mode required set and
leaves conditional rules to the authoritative preflight — the right call, since
mirroring the whole rule set invites drift. But P2 showed the cost: turning on
`use_multiscale` silently makes four more keys mandatory, and nothing said so
until a preflight six seconds later, or a failed run.

The sheet now hints (still a hint — preflight stays the authority — and the text
names the rule so the source is obvious):

```
use_multiscale is True, so MGN-MULTI-REQ also requires: coarsening_type, multiscale_levels, voronoi_clusters, mp_per_level
mp_per_level needs 7 entries for multiscale_levels 3; it has 3.
```

Both appear the moment the offending value is typed.

### A workspace audit found nothing further

All ten workspaces were crawled at 1600×1000 and 1440×900, and the two modal
overlays additionally at 1280×800 and 1152×720, checking for JS errors, failed
API calls, panels stuck on a loading placeholder, and controls covered by other
elements. Result: **no JS errors, no failed API calls, no stuck panels, no
covered controls.** The apparent "off screen" hits were content below the fold
in a properly scrollable container (`.config-fields-panel`, scrollHeight 3028 vs
client 492) — reachable by scrolling, so not defects.

### End-to-end proof that the generative workflow now works

The shipped "Generative design optimization" template, run untouched apart from
pointing SDFFlow at its checkpoints and naming two objectives in the inspector:

```
[studio] Starting Generative design optimization with 4 executable step(s).
[studio] Step 1/4: SDFFlow · sample
[studio] Step 2/4: SDFFlow · sample     -> 8/8 valid, 8/8 watertight
[studio] Step 2 tabulated 8 candidate(s) to output/geometry_generation/studio_chain2/candidates.csv.
[studio] Running Optimization · select (optimization).
[studio] Optimization · select wrote frontend/runtime/optimization/eb9cd77a7da2.json.
[studio] Running Export Results · package (export).
[studio] Export Results · package wrote frontend/runtime/exports/output-studio-run_001-f8f214e1.json.
[studio] Pipeline completed.
```

`8 rows → 8 feasible → 4 Pareto`, objectives `{volume: min, bbox_z: max}`, with
the candidate table written by the generator step itself — no hand-built CSV
anywhere. Before these fixes the same template could not be started at all, and
when forced through it ran two sampling steps and reported success having
silently dropped the optimization and the export.

**P17 verified in the exact failing scenario:** a second Studio instance was
started on port 8110 while the rollout fine-tune was running. The job stayed
`running` with its PID intact; previously that startup would have stamped it
`interrupted`.

**P21 (enhancement) — `init_modelpath` warm start.** Training always started
from scratch, so the standard recipe for rollout training — pretrain one-step
(cheap, stable), then fine-tune the same weights on the full rollout — could not
be expressed at all. Added to the MeshGraphNets trainer, its spec, both
`PATH_KEYS` mirrors, and the Studio catalog, mirroring SDFFlow's existing
`init_vae_modelpath`. Strict by design: an architecture mismatch raises rather
than silently training from noise.

```
Warm start: loaded weights from ...\himgn_ex9.pth (epoch 59)
Warm start: EMA shadow seeded from the same weights
Time integration: AR-RT (19-step rollout, BPTT through the full unroll, per-step gradient checkpointing)
```

---

## 8. Accuracy: what more training actually bought, and where it stopped

The target was rollout R² > 0.9. **It was not reached.** Best achieved: **mean
0.505**. Here is the whole trajectory, all scored the same way — 87 held-out
samples, 19-step autoregressive rollout, `dataset/ex9_infer.h5`.

| run | epochs | R² mean | R² median | R² p95 | R² min | rel-L2 mean |
| --- | --- | --- | --- | --- | --- | --- |
| AR-OT | 3 | −0.055 | 0.010 | 0.611 | −2.13 | 0.953 |
| AR-OT | 60 | 0.322 | **0.732** | **0.964** | −2.58 | 0.665 |
| AR-OT 60 → **AR-RT** | +16 | **0.505** | 0.588 | 0.853 | **−0.652** | 0.644 |

Training itself converged well: AR-OT drove the one-step loss from 4.78e-01 to
8.04e-03 over 60 epochs (test 4.06e-01 → 4.00e-02), and the AR-RT fine-tune
improved its own rollout objective from 5.36e-01 to 4.29e-01.

### The interesting part: AR-RT inverted the error profile

Per-timestep R², same 87 samples:

| t | AR-OT (60 ep) | AR-RT (+16 ep) |
| --- | --- | --- |
| 1 | **0.891** | **−1.125** |
| 5 | 0.452 | −0.738 |
| 8 | 0.314 | −0.240 |
| 12 | 0.425 | 0.403 |
| 15 | 0.463 | 0.536 |
| 19 | 0.520 | **0.617** |

AR-OT is accurate at the first step and decays as error compounds. AR-RT is the
mirror image: poor early, improving monotonically, and better than AR-OT from
t≈12 onward. It also cut the worst-case sample from R² −2.58 to −0.65, which is
exactly the catastrophic-drift failure rollout training exists to remove.

The cause is visible in AR-RT's own design note: its loss is normalized by the
**trajectory-scale** `state_std`, not by per-step magnitude. Early steps have
small state magnitudes, so a large *relative* error there barely registers in
the loss — the optimizer correctly spends its capacity on the late steps that
dominate the objective. For an evaluation that weights every timestep equally,
that trade lifts the mean (0.322 → 0.505) while lowering the median
(0.732 → 0.588).

### What reaching R² > 0.9 would actually take

Pooled R² is capped by one-step accuracy: at t=1 the AR-OT model is at 0.891, so
even a drift-free rollout could not average above that. Getting the pooled figure
past 0.9 needs t=1 ≳ 0.98 **and** controlled drift — i.e. both terms, not one.

* **One-step accuracy** was still improving when the budget ran out (train loss
  still falling at epoch 60). The checked-in `configs/MeshGraphNets/ex9/
  config_train_hi_plasticity.txt` asks for **500 epochs**; at the measured
  7 min/epoch ceiling on this GPU (see P12 — throughput is input-bound and gets
  *worse* with larger batches) that is a **~58-hour** run.
* **Then** an AR-RT fine-tune on top, which this session showed is the right
  tool for the drift term — ideally after fixing the loss scaling so it does not
  abandon early steps.

Both are now possible in one config thanks to the `init_modelpath` warm start
(P21) and the AR-RT crash fix (P18); neither was before this session.

---

## 9. Summary of findings

Across three sessions: six full GUI pipeline runs of a hand-built HI-MGN profile,
six SDFFlow generation batches, and two Pareto selections. **23 problems found,
23 fixed** (P12 documented rather than fixed — it is a measurement, not a defect
with an obvious owner). Seven were found only while fixing the others.

The ones that stopped real work: P8 (a configured output directory made results
invisible), P15/P16 (the optimization block was unreachable from both the canvas
and the workspace), P19 (two servers sharing a port answered from different code),
and P22 (the P8 and P14 fixes were inert in the pipeline until it was found).

| # | Finding | Severity |
| --- | --- | --- |
| P8 | A user-set `inference_output_dir` makes the Studio blind to the results; pipeline fails claiming the block "produced no output" — **proven by controlled re-run** | **high** |
| P15 | The Optimization block never executes in a pipeline; the run reports success with the step silently omitted | **high** |
| P16 | Artifact pickers are starved — 2 023 Studio-written configs fill the 750-file cap *before* sorting, so nothing under `output/` is ever offered | **high** |
| P13 | Graph validation demands training data regardless of mode, so the shipped generative template can never run | **high** |
| P14 | Nothing converts the generator's metadata into the candidate CSV the optimizer requires | **high** |
| P1 | `Fit graph` leaves blocks off-stage; the top one sits under the toolbar where clicks die | **high** |
| P5 | The live runtime drawer covers all four canvas zoom controls, minimised or not | **high** |
| P2 | HI-MGN and BSMS-GNN presets emit configs that cannot preflight, and the sheet does not flag the missing keys | **high** |
| P9 | After a failure every block resets to `idle`; the canvas shows nothing about where it failed | medium |
| P10 | MGN checkpoints are not `weights_only`-loadable, so the checkpoint probe never runs | medium |
| P6 | 96 % of the runtime log is tqdm repaint frames | medium |
| P11 | Studio bookkeeping keys (`results_path`, `results_samples`) leak into the launcher config | medium |
| P12 | Sample throughput falls 6× from batch 4 to batch 32; 7 min/epoch is a hard floor | medium |
| P7 | `message_passing_num` shown as a live knob although HI-MGN ignores it | low |
| P3 | `/api/preflight` returns 422 for a merely-invalid config, logging a console error | low |
| P4 | Diagnostics panel keeps the previous verdict on screen while preflight is in flight | low |
| P17 | Importing `studio_backend.state` marks another server's *running* jobs interrupted | **high** |
| P18 | AR-RT rollout training crashes on any 2-D state (`input_var < 3`), ex9 included | **high** |
| P19 | Two Studio servers can bind the same port; requests split between stale and current code | **high** |
| P22 | `resolved_paths` is empty for steps after the first, silently defeating the P8 and P14 fixes | **high** |
| P23 | The optimization step was dropped whenever its producer ran in the same pipeline | **high** |
| P20 | Config-sheet search only searched the selected section | medium |
| P21 | *(enhancement)* `init_modelpath` warm start, without which pretrain→rollout-finetune is unexpressible | — |

