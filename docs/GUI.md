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
python start_studio.py          # http://127.0.0.1:8123/index.html on 8123
python start_studio.py 8090     # any other port
```

`START_STUDIO.bat` does the same on Windows and opens a browser.

The console must print `AI-CAE4ALL Studio is ready`. The badge at top right
reports live discovery (for example `12/12 entrypoints found`) and must agree
with `GET /api/models`.

**One server per port.** `StudioHTTPServer` sets `allow_reuse_address = False`
precisely so a second instance fails loudly instead of splitting requests with
the first. If an edit to a spec or to `studio_backend/` "does not take", the
server is holding the old module: restart it. The registry is read once at
startup.

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
| Mesh field surrogates | HI-MGN multiscale (default), MeshGraphNets flat, MeshGraphNets-V, cHI-MGNflow, Transolver, FNO, GINO, DeepONet, Point-DeepONet |
| Fixed-geometry and tabular | SimulGen-VAE reconstruction, Parametric response estimation (MLP) |
| Generative geometry | SDFFlow train (DeepJEB), Design optimization |
| Data preparation | Geometry to HDF5 (ingest) |
| Start from scratch | Untitled pipeline |

The mesh templates all target **ex9 plasticity**: `dataset/ex9.h5` to train and
`dataset/ex9_infer.h5` held out (900 / 87 samples, 20 steps, 3131 nodes). Each
mirrors its own checked-in training config — the two MeshGraphNets pipelines
follow `configs/MeshGraphNets/ex9/`, cHI-MGNflow `configs/HI_MGNFlow/ex9/`,
Transolver `configs/Transolver/ex9/`, the four operators
`configs/Neural_Operator/ex9/`. MeshGraphNets-V has no checked-in ex9 config
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

24 block types. `native` means a real launcher route or backend endpoint;
`adapter` means the Studio composes existing outputs rather than computing new
physics.

| Category | Blocks |
| --- | --- |
| Sources (4) | CAD, HDF5 Dataset, Design Parameters, Saved ML Model |
| Preparation (1) | Geometry → HDF5 Dataset |
| Models (11) | one per live route: MeshGraphNets, MeshGraphNets-V, cHI-MGNflow, Transolver, FNO, GINO, DeepONet, Point-DeepONet, SimulGen-VAE, SDFFlow, Simple MLP |
| Execution (2) | Inference Run, CAD Generator |
| Optimization (1) | Optimization |
| Evaluation (3) | Evaluate Predictions, Train Metrics, Compare Models |
| Outputs (1) | Export Results |
| Deployment (1) | API Deployment |

`geometry_ingest` is the twelfth live route; it is exposed as the
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

### What the inspector shows

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
flashes it.

Presets apply Studio defaults or a checked-in config; the two SDFFlow
closed-loop presets load `configs/SDFFlow/config_optimize.txt` and
`config_optimize_surrogate.txt` respectively.

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

Metrics are fixed: relative L2, MAE, RMSE, max absolute error and R², each
aggregated as mean, median, p95, min and max.

## Comparison

`Compare Models` ranks the mean of one numeric column across the CSVs you
select. Choosing outputs from the same held-out set is yours to get right — but
the claim is no longer left unexamined: the array `contract` and the sample IDs
recorded in each per-sample CSV are cross-checked, and a mismatch is reported
above the ranking ("these runs share no sample IDs", "scored under different
array contracts"). Shape bookkeeping columns (`fields`, `timesteps`, `nodes`,
`values`) are not offered as metrics.

---

## Workspaces

| Workspace | What it gives you |
| --- | --- |
| **Models** | All 12 live routes: repository, entrypoint, modes, key count, dataset kind, install health; per-route details and checked-in examples |
| **Data** | The HDF5 catalog with a sample viewer (mesh, points, field, timestep player); "Inspect HDF5" renders the file's contract inline |
| **Runs** | Studio-launched jobs with status, step and log; Train Metrics plots every metric parsed from a real run log |
| **Optimization** | Feasible Pareto set and crowding-distance top-k from a real candidate CSV. It ranks numbers you give it; it does not invent physics. Constraints are checked against the chosen CSV as you type; an all-infeasible result names the constraint that rejected each row and the closest value observed; the selected designs are written back as a CSV in the source's own columns |
| **Benchmarks** | The checked-in `configs/campaigns/benchmarks_all/roster.tsv` arms, each preflightable and loadable into the canvas. A passing config is not a result |
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
| A model card shows empty axes | That block has no run behind it. The card draws its result only once there is one |

---

## Extending it

| To add | Do this |
| --- | --- |
| A block | Add a `BLOCK_SPECS` entry in `studio/src/constants.js` (ports, defaults, category, maturity) |
| A model route | Nothing, if the backend registers it: `registerLiveModel` installs a block from the live spec. Add a `MODEL_CATALOG` entry for richer copy and defaults |
| A template | Add a `TEMPLATES` entry; the picker and its grouping are generated from node types |
| Help for a key | Add it to `HELP` in `constants.js`; the config sheet and non-model inspectors both read it |
| A dropdown | Add the value set to `CHOICES`; mirror the spec validator exactly, and never invent a value |

Front-end modules: `constants.js` (catalog), `graph.js` (canvas), `inspector.js`,
`config.js` (config sheet), `validate.js` (graph checks and step construction),
`autofill.js` (graph-derived values), `run.js` (jobs and diagnostics),
`studio.js` (workspaces), `viewer.js`/`render3d.js` (samples),
`markdown.js` (docs rendering). Backend: `studio_backend/` — `state.py` owns
jobs, `analysis.py` evaluation/comparison/optimization, `suite_bridge.py` the
launcher bridge.

Tests: `python -m pytest -q studio/studio_backend` for the backend;
`python -m pytest -q tests/` for the launcher contracts the GUI depends on.
