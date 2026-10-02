# The AI-CAE4ALL Studio (GUI)

The Studio is a browser front end for the same launcher the CLI uses. You build
a pipeline out of typed blocks, and every executable block is turned into a
native flat-text config, validated by `cae_suite`'s real preflight, and run as
the same subprocess `AI_CAE4ALL_main.py` would start.

Nothing in the Studio reimplements a model. If the GUI and the launcher ever
disagree, the launcher is right and the GUI has a bug.

- **Manual (this file)** — what the GUI does and how to drive it.
- [guides/studio.md](guides/studio.md) — the front-end's own notes: local API
  surface, review URLs, integration boundary.
- [`cae_suite/specs/`](../cae_suite/specs/) — the executable source of truth for
  every config key the GUI shows.

---

## Start it

```bash
cd studio
python start_studio.py          # http://127.0.0.1:8080/index.html (default port 8080)
python start_studio.py 8090     # any other port
```

`START_STUDIO.bat` does the same on Windows and opens a browser.

The console must print `AI-CAE4ALL Studio is ready`. The badge at top right
reports live discovery (for example `11/11 entrypoints found`) and must agree
with `GET /api/models`.

**One server per port.** `StudioHTTPServer` sets `allow_reuse_address = False`
precisely so a second instance fails loudly instead of splitting requests with
the first. If an edit to a spec or to `studio_backend/` "does not take", the
server is holding the old module: restart it. The registry is read once at
startup.

**Local requests only.** The server launches processes and reads the
repository, so on a specific bind address (the default `127.0.0.1`) it answers
only a `Host` of `127.0.0.1`, `localhost`, `[::1]` or the bound host, at its
own port (anything else is 403, which defeats DNS rebinding). A POST whose
`Origin` is present and foreign -- including `null` -- is 403. JSON endpoints
require `Content-Type: application/json` (415 otherwise); `/api/upload` takes a
raw body and is exempt. A wildcard bind (`0.0.0.0`) skips the Host check.

---

## The screen

| Region | What it is |
| --- | --- |
| Top bar | Workspace nav, runtime health, **Validate**, **Run pipeline** |
| Left | Block library (search, drag or click to add) |
| Centre | Canvas: blocks, typed ports, zoom/fit/auto-layout, pipeline name |
| Right | Inspector for the selected block or connection |
| Bottom | Runtime drawer: live log, per-step diagnostics, other jobs |

Twelve workspaces open over the canvas: Models, Data, Runs, Optimization,
Benchmarks, Artifacts, Deploy, System, Docs, plus the Evaluation, Comparison and
Export surfaces reached from their blocks.

---

## Pipelines and templates

The picker in the canvas toolbar is generated from `TEMPLATES` in
`studio/src/constants.js` and grouped by what each pipeline trains. Every live
route has a default profile.

| Group | Templates |
| --- | --- |
| Mesh field surrogates | HI-MGN multiscale (default), MeshGraphNets flat, MeshGraphNets-V, cHI-MGNflow, Transolver, FNO, DeepONet, Point-DeepONet |
| Fixed-geometry and tabular | SimulGen-VAE reconstruction, Parametric response estimation (MLP) |
| Generative geometry | SDFFlow train (DeepJEB), Design optimization |
| Data preparation | Geometry to HDF5 (ingest) |
| Start from scratch | Untitled pipeline |

The mesh templates all target **ex9 plasticity**: `dataset/deterministic/ex9_plasticity.h5` to train and
`dataset/deterministic/ex9_plasticity_infer.h5` held out (900 / 87 samples, 20 steps, 3131 nodes). Each
mirrors its own checked-in training config — the two MeshGraphNets pipelines
follow `configs/MeshGraphNets/deterministic/ex9/`, cHI-MGNflow `configs/HI_MGNFlow/ex9/`,
Transolver `configs/Transolver/deterministic/ex9/`, the three operators
`configs/Neural_Operator/deterministic/ex9/`. MeshGraphNets-V has no checked-in ex9 config
(it was left out of the ex4–ex9 roster as a one-to-many method), so its template
reuses the ex9 dataset keys with MGN-V's own architecture.

**The held-out split is a separate source block on purpose.** Feeding the
training dataset into the Inference block makes the graph *say* "predict what
you trained on", and the graph wins over the trainer's own `infer_dataset`. Two
blocks keep the split visible where a reader can see it.

**Design optimization preflights red on a fresh checkout, by design.** It
consumes `output/geometry_generation/studio/sdfflow_{vae,fm}.pth`, which do not
exist until *SDFFlow train* writes them. Its name says so.

---

## Blocks

23 block types. `native` means a real launcher route or backend endpoint;
`adapter` means the Studio composes existing outputs rather than computing new
physics.

| Category | Blocks |
| --- | --- |
| Sources (4) | CAD, HDF5 Dataset, Design Parameters, Saved ML Model |
| Preparation (1) | Geometry → HDF5 Dataset |
| Models (10) | one per live route: MeshGraphNets, MeshGraphNets-V, cHI-MGNflow, Transolver, FNO, DeepONet, Point-DeepONet, SimulGen-VAE, SDFFlow, Simple MLP |
| Execution (2) | Inference Run, CAD Generator |
| Optimization (1) | Optimization |
| Evaluation (3) | Evaluate Predictions, Train Metrics, Compare Models |
| Outputs (1) | Export Results |
| Deployment (1) | API Deployment |

`geometry_ingest` is the eleventh live route; it is exposed as the
**Geometry → HDF5 Dataset** block rather than a model block, because it prepares
data instead of learning.

### Ports are typed

A link is offered only when the receiving port accepts the sending type **and**
the link would not close a loop. The `artifact` wildcard is receive-side only:
Export accepts anything, but Export's own `files` output is not a substitute for
a dataset. The same rule applies when a pipeline is imported, so a saved
document cannot reintroduce a link the canvas refuses.

A required port is marked `*` — and only in modes that actually require it.
SDFFlow's `sample` mode reads no dataset, so its `data` port is not starred
there.

**A port that changes nothing is not shipped.** The Optimization block takes one
input — the candidate/evaluation CSV it ranks — because that is all its code
reads. A pipeline saved before a port was removed still loads; the stale link is
dropped and named.

### What a block card shows

Every card carries up to four facts read from real sources, never a drawn
placeholder: an HDF5 Dataset reads its file's attributes through
`/api/hdf5/facts` (samples · nodes · timesteps, the state fields, the
input-only `cond_var` conditions — or shapes/SDF points for SDFFlow, rows and
columns for a table); a model block shows its data file and the architecture
keys that identify the variant (HI-MGN levels, Transolver slices, FNO modes…);
Inference and Evaluate name the model and the held-out file they read. The
subtitle says the block's role in *this* graph — `Training set` or `Held-out
set` for a dataset, taken from what its output feeds — and a model's variant and
mode. One button opens the block's detail view; executable blocks add Run.

### What the inspector shows

**At a glance** repeats the card's facts, **Connections** lists what each port
is linked to and flags a required input that is not, and the settings in
between are labelled in words with the raw config key beside them. Prose rows
that describe a block rather than configure it (`split`, `viewer`, `metrics`…)
are not shown, and an Inference block hides the knobs its linked family cannot
read (`infer_chunk_size` is Transolver-only, the `flow_*` keys cHI-MGNflow-only)
unless you filled one in.

For a **model block**, eight rows chosen by mode, not by the order keys happen
to be written: the mode, the dataset it reads, the checkpoint it writes or
loads, the epoch/batch/learning-rate trio (or the per-stage spellings SDFFlow
and SimulGen-VAE use), then the keys that distinguish that route. Training-only
keys are hidden in non-training modes. Everything else is under **Full config**.

For every other block, all its fields with a one-line explanation each, because
those blocks have no config sheet to carry the explanation.

Rows tagged *fixed behaviour* are facts about what the block does, not controls.
Rows tagged *auto · &lt;source&gt;* were filled from the graph; type over one to
take manual control, clear it to follow the graph again.

---

## Full config

Every key the live `MethodSpec` accepts, sectioned (Required, Data & output,
Architecture, Training, Resources & runtime, Inference & evaluation,
Optimization, Advanced, Inactive / rejected), with:

- **status** — required, set, defaulted (with the backend default shown),
  optional, inactive, or rejected;
- **help** — 182 documented keys, and an honest fallback when a key has none;
- **dropdowns** wherever the spec publishes a closed value set, so the sheet
  cannot offer a value the launcher rejects;
- **rejected/inactive marking** for keys a route knows but does not honour —
  the 34 keys MeshGraphNets' removed-feature guard raises on, and the 22 the
  deterministic runtime silently ignores, are labelled instead of looking live.

Controls: section tabs, search, *changed only*, *show inactive*, mode switch,
presets, `.txt` import/export, a raw-text tab, **Explain**, **Run preflight**,
and **Save**.

Pasting or loading a `.txt` marks a key as a manual override **only where its
value differs from what the graph supplies** — an untouched `dataset_dir` keeps
following the connected block — and the keys that genuinely were overridden are
named. The raw tab is form ↔ text: comments and the author's line order are not
preserved, and a paste says how many comment lines it dropped. A preflight error
naming an accepted key renders a **Show field** button that scrolls to and
flashes it; the diagnostics list stays put, so you can work down it.

**High performance** and **Low VRAM** read the largest and the smallest
checked-in train config for the route (`PRESET_SOURCES` in
`studio/src/constants.js`; the menu names each one, e.g. "High performance
(ex9 HI-MGN)"). Only the recipe is copied — architecture, optimizer, precision,
memory switches. Paths, `opt_*`, seeds, epoch budgets and their schedules,
`*_interval` cadence, the data contract (`input_var`/`output_var`/`cond_var`,
node types, positional features, rollout) and dataset-tuned values
(`kl_weight`, `num_vae_samples`) stay the block's own, and mesh-sized values
(`voronoi_clusters`, FNO grid/modes, DeepONet sensor grid) are filled only when
the block has none — a flat block switched to a hierarchy gets `voronoi_clusters
512, 64`, the HI-MGN preset's ex9 value, not the source mesh's count. A value
equal to the block's own (`1e-6` and `0.000001`) is not listed as a change. Low
VRAM also sets `batch_size 1` where the recipe or the block runs more, and on
MGN-V and cHI-MGNflow `vae_batch_size 1` (trajectories rolled out together —
memory and speed only); SDFFlow and SimulGen-VAE keep their stage batch sizes
as the source has them. For MeshGraphNets both presets are the HI-MGN recipe: activations are
checkpointed per GnBlock, so memory follows the full-resolution block count
(15 flat, 4 + 4 HI-MGN). MLP has no checked-in config and keeps
**Studio defaults**.

The other presets apply a fixed set of values or a checked-in config. Both SDFFlow closed-loop
presets load the one checked-in optimize config,
`configs/SDFFlow/geometry_generation/ex1/baseline/config_optimize_sdfflow.txt`;
the surrogate preset then switches `opt_analysis surrogate`, points
`opt_surrogate_config`/`opt_surrogate_checkpoint` at the ex13 (DeepJEB vertical
load) HI-MGN, and sets the values its labels fix: `opt_load_cases vertical`,
`opt_length_scale 0.102323`, `opt_material_rho 4470`, `opt_vertical_disp_max
0.36` (the ex13 median), and `opt_fea_verify False` (fea.py does not model the
bolt-bore/lug-bore boundary conditions, so optimize refuses it on this layout).
Until that model has trained, preflight names the missing checkpoint.

A CAD Generator wired to the SDFFlow block runs with **its own** `mode` and
`opt_analysis` (they are laid over the model block's), so applying either
closed-loop preset also sets those two on every connected generator, and the
confirmation dialog lists them. Before this, the generator's defaults
(`sample`, `fea`) silently undid the preset at Run.

The CAD Generator's optional **HI-MGN surrogate** port takes a MeshGraphNets
model block or a saved HI-MGN checkpoint and fills the generator's
`opt_surrogate_checkpoint` from it (re-rooted to `methods/SDFFlow`), so a canvas
can train HI-MGN and SDFFlow side by side and close the loop on the trained
surrogate. The trainer runs first because the wire orders it. The wire carries
only the checkpoint: `opt_surrogate_config`, the matching HI-MGN *inference*
config, still comes from the SDFFlow block's Full config. Validate rejects the
wire outside `mode optimize` + `opt_analysis surrogate`, and rejects a non-MGN
model on it.

`opt_fea_verify` makes a surrogate search honest inside the run: after CMA-ES,
the optimized design, the best baseline and the typical baseline are re-solved
with the tet4 FEA solver (`design_loop/verify_with_fea.py`). report.md gains an
"FEA verification" section (surrogate vs FEA, limit verdicts), and the
Optimization workspace's table gains `fea_*` columns -- rank and constrain on
those (for example `fea_vertical_displacement_mm <= 0.2`, minimize
`fea_mass_kg`), not on the surrogate's. The CAD Generator inspector's
"Surrogate accuracy gate" says which of the two states the run is in.

---

## Validate, then run

**Validate** runs the launcher's real preflight over every executable step. The
first step includes the native probe, which starts the method's own interpreter,
so expect tens of seconds.

Findings render in the runtime drawer as rows carrying the step, the diagnostic
code, the field and a hint. A row with an owning block gets **Fix now →**, which
selects that block and opens its config at the offending field. Graph-level
problems (a missing input, an unselected file, a type mismatch) render the same
way rather than as a toast that disappears.

`Run pipeline` executes the steps in dependency order, re-preflighting the exact
saved config immediately before each launch. The drawer streams the native
process's own stdout. Jobs live in the backend, so closing the tab does not stop
a run.

Reopening the page rejoins every job still in flight, but a job paints its
status only onto **the canvas that launched it**. Each canvas carries a
`canvas_id` (a new one per loaded template), sent with the run and recorded on
the job. Node ids cannot do this alone: every copy of a template shares them, so
a freshly loaded SDFFlow template used to show another pipeline's live training
run as its own. Jobs from before `canvas_id` existed are still claimed by node
ids, but only by a canvas restored from a document of the same age, and only
while such a job is **still running**: a claimed job is recorded in
`owned_jobs`, so it stays this canvas's after it finishes, while a *finished*
untagged job is never claimed -- node ids cannot tell this canvas's old runs
from every other copy's. The flag survives a reload (`legacy_canvas`) and is
dropped once no untagged pipeline job is running (an untagged command job has
no node ids to claim by, so it does not keep the flag alive).

A run that **finished while the page was closed** is painted too: on connect,
and after a pipeline is loaded, imported or undone, the canvas rebuilds every
block's status from the runs it owns -- finished ones oldest first, so the
latest run of each block is what shows, then the runs still in flight.
Status is rebuilt in full; **results are not**. Each block records the run
whose outputs it last received (`results_from` in the saved document), and a
replayed run writes its results path, report, export or Train Metrics binding
only onto blocks it has not reached -- blocks whose last delivery is an older
run. So an evaluation report, an export or a Train Metrics run the user chose
after a run survives every reload and undo, while a run that finished unseen
still lands. A document saved before `results_from` existed is migrated on
first replay: a block holding exactly what one of its runs wrote counts as
delivered by that run, anything else as the user's own later choice. A run the
poller has already seen finish is remembered for the page's lifetime, so a
`/api/jobs` listing fetched before that poll cannot bring it back as running.

**Load pipeline** (Runs) reopens a run's graph as a *copy*: it gets a new
`canvas_id` and owns exactly that run (`owned_jobs` in the saved document), so
later runs of the original canvas do not paint onto it, and its runs not onto
the original. The graph is the run's launch snapshot, so its blocks are marked
as having received nothing yet and the run's own results replace whatever the
snapshot carried. Compare Models follows the same rule: a model block resolves
to a run this canvas owns (on a legacy canvas, an untagged run only while it is
in flight); failing that, the newest run of the same model is shown, marked
*not this canvas*.

**Stop** takes effect wherever the run is: a cancel that lands during a step's
launch preflight stops the launch before the native process starts, and one
that lands while the process is starting kills it as soon as it is registered.
A pipeline whose last step had already succeeded when the cancel arrived is
recorded as *completed*. A runner error of any kind (an analysis step raising,
a bookkeeping bug) ends the job as *failed* with the traceback in its log,
never as a job stuck at *running*.

**Restarting the server does not orphan a run.** A job whose native process is
still alive is adopted: Stop reaches it by pid, and a watcher finalises it when
the process exits -- as *interrupted*, with a note, because its exit code cannot
be recovered and any later pipeline steps were not run. The drawer shows the
last 120 000 characters of a log and says so when it truncates (`log_truncated`);
the full log is still on disk.

Artifacts land under the repository's single `output/` root; the Studio's own
scratch (saved configs, exports, evaluation reports, job logs) lives under
`studio/runtime/`.

---

## Evaluation

`Evaluate Predictions` compares a real prediction HDF5 (or a directory of them)
against a real ground-truth HDF5 and writes a report plus a per-sample CSV.

The scoring contract is inspected before anything is computed: sample matching,
array shapes, and a field mapping. Two rules matter.

1. **Fields pair by declared name** when both files declare them. Positional
   pairing is offered only when the counts match, is labelled *confirm*, and
   will not score until you tick the confirmation.
2. **Reference-coordinate rows are never scored.** Rows 0:3 of `nodal_data` are
   copied into every rollout unchanged, so scoring them reports zero error and
   R² = 1 for any model whatsoever. A coordinate-only mapping is refused with an
   explanation. Rows named `x`/`y`/`z` count as coordinates, not only
   `x_coord`/`y_coord`/`z_coord`. Files that carry no coordinates at all —
   SimulGen-VAE's `reconstruct` writes `nodal_field` with the physical rows
   only — keep every channel.
3. **The trailing node-type row is not a prediction either.** Every mesh rollout
   writer appends a per-node part/`node_type` label and copies it through
   unchanged. The five writers now record `output_var` beside `num_features`, so
   the scoreable rows are read from the file rather than inferred; where a file
   predates that, the row is dropped by name.
4. **A positional mapping starts unticked.** When the rows were lined up by index
   rather than by name, no row is pre-selected — you choose the ones to score and
   then confirm. Pre-ticking them made the confirmation the only thing standing
   between a categorical row and a displacement field.

A prediction may be a **directory**: `mode inference` writes one HDF5 per sample,
and the picker lists such directories alongside single files.

Samples pair by ID, never by coincidence. The file name stands in for a
sample ID only when the file holds a single record, so two multi-sample files
that happen to share a name (`ex_infer.h5`) are not paired. A single-sample
prediction and a single-sample truth are paired without an ID match only when
at least one of them has no ID; two different declared IDs are refused as
different samples.

Metrics are fixed: relative L2, MAE, RMSE, max absolute error and R², each
aggregated as mean, median, p95, min and max. **Relative L2 and R² are the
macro-average over the scored fields**, each computed per field: pooled over
fields, a field with a large offset (a pressure near 1000) made any predictor's
R² look near-perfect. The pooled values stay available as `relative_l2_pooled`
and `r2_pooled`, the per-field values as `relative_l2[<field>]` / `r2[<field>]`
columns in the per-sample CSV and as `aggregate_by_field` in the report. MAE,
RMSE and max absolute error are pooled over the scored values. A 1-D embedded
vector (MLP `Y_pred`/`Y_true` with one output) is scored row by row.

## Comparison

`Compare Models` ranks the mean of one numeric column across the CSVs you
select. Choosing outputs from the same held-out set is yours to get right — but
the claim is no longer left unexamined: the array `contract` and the sample IDs
recorded in each per-sample CSV are cross-checked, and a mismatch is reported
above the ranking ("these runs share no sample IDs", "scored under different
array contracts"). Shape bookkeeping columns (`fields`, `timesteps`, `nodes`,
`values`) are not offered as metrics. The group column is filled from the
detected columns until you edit it; a blank you leave there (one mean per CSV)
is kept.

---

## Workspaces

| Workspace | What it gives you |
| --- | --- |
| **Models** | All 11 live routes: repository, entrypoint, modes, key count, dataset kind, install health; per-route details and checked-in examples |
| **Data** | The HDF5 catalog with a sample viewer (mesh, points, field, timestep player); "Inspect HDF5" renders the file's contract inline |
| **Runs** | Studio-launched jobs with status, step and log; Train Metrics plots every metric parsed from a real run log |
| **Optimization** | Feasible Pareto set and crowding-distance top-k from a real candidate CSV. It ranks numbers you give it; it does not invent physics. Constraints are checked against the chosen CSV as you type; an all-infeasible result names the constraint that rejected each row and the closest value observed; the selected designs are written back as a CSV in the source's own columns |
| **Benchmarks** | Every train and infer config named by `configs/campaigns/dataset_matrix/manifest.json`, each preflightable and loadable into the canvas. A passing config is not a result |
| **Artifacts** | Every output and checkpoint, searchable, with "Use in pipeline" to drop a correct source block onto the canvas |
| **Deploy** | Portable CPU inference and the PyInstaller `.exe` build. Families the bundle cannot run say so and disable the button |
| **System** | Interpreters, CUDA devices, install health, and a full `--audit-configs` over every checked-in config |
| **Docs** | The repository's Markdown, rendered, with working in-repo links, a "Start here" strip, and rows grouped by area rather than by modification time |

---

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| A change to a spec or `studio_backend/` has no effect | The server holds the old module. Restart it; check `GET /api/models` |
| Validate seems to hang | The first step's native probe starts the method's own interpreter. Tens of seconds is normal |
| A workspace looks empty | Workspaces load real repository state for 3–6 s. Wait for "Loading real AI-CAE4ALL state…" to clear |
| Evaluate reports a perfect score | Check the field pairing. Scoring coordinate rows is the classic cause and is now refused |
| Inference produced nothing to evaluate | The Inference block should name `inference_output_dir`; it is auto-filled next to the checkpoint |
| A run needs a value the Studio will not guess | `input_var` / `output_var` on mesh routes are deliberately yours to set: with `cond_var` rows present they are not the feature-row count |
| The Optimization Run button stays disabled | A constraint names a column the selected CSV does not have, or is not `column <= value`. The note under the field says which |
| A dataset card shows no facts | The path does not end in `.h5`/`.hdf5`, the file is missing, or the server is offline — the facts come from the file itself |

---

## Extending it

| To add | Do this |
| --- | --- |
| A block | Add a `BLOCK_SPECS` entry in `studio/src/constants.js` (ports, defaults, category, maturity) |
| A model route | Nothing, if the backend registers it: `registerLiveModel` installs a block from the live spec. Add a `MODEL_CATALOG` entry for richer copy and defaults |
| A template | Add a `TEMPLATES` entry; the picker and its grouping are generated from node types |
| Help for a key | Add it to `HELP` in `constants.js`; the config sheet and non-model inspectors both read it |
| A dropdown | Add the value set to `CHOICES`; mirror the spec validator exactly, and never invent a value |

Front-end modules: `constants.js` (catalog), `graph.js` (canvas), `cards.js`
(card and At-a-glance facts), `inspector.js`,
`config.js` (config sheet), `validate.js` (graph checks and step construction),
`autofill.js` (graph-derived values), `run.js` (jobs and diagnostics),
`studio.js` (workspaces), `viewer.js`/`render3d.js` (samples),
`markdown.js` (docs rendering). Backend: `studio_backend/` — `state.py` owns
jobs, `analysis.py` evaluation/comparison/optimization, `suite_bridge.py` the
launcher bridge.

Tests: `python -m pytest -q studio/studio_backend` for the backend (11 modules,
69 tests). `studio_backend/conftest.py` redirects the runtime root to a
temporary directory for the run, because importing `studio_backend.state`
recovers jobs and the tests write configs and exports. The root `tests/` suite that used to cover the launcher contracts the
GUI depends on was deleted in commit `9884fb1` and has no replacement — use
`python AI_CAE4ALL_main.py --audit-configs` for that layer. See
[guides/testing.md](guides/testing.md).
