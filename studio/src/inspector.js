import { $, $$, escapeHtml, toast, formatBytes, on } from "./dom.js";
import { state, snapshot } from "./state.js";
import { savePipelineState } from "./persistence.js";
import { BLOCK_SPECS, MODEL_CATALOG, TYPE_META, INPUT_SOURCE_META, HELP } from "./constants.js";
import { apiRequest, requireRuntime } from "./api.js";
import { previewGraphic, nodeVisualLabel } from "./graphics.js";
import { typeColor, STANDALONE_INFERENCE_MODEL_IDS } from "./validate.js";
import { duplicateNode, deleteSelected, nodeEvidenceLabel, render } from "./graph.js";
import { openConfig, choicesFor, requiredFor } from "./config.js";
import { openArtifact } from "./viewer.js";
import { runGraph } from "./run.js";
import { activateStudioWorkspace, openStudio, openModelDetailWorkspace, openTrainingMetricsWorkspace, liveShell, liveError } from "./studio.js";
import { applyGraphAutofill, autoFillCount, autoFillMeta, markManualConfigValue, selectedParameterCandidate } from "./autofill.js";

export function inspectorCard(title, status, text, stateClass = "") {
  return `<article class="inspect-card"><header class="inspect-card-head"><strong>${escapeHtml(title)}</strong><span class="state-pill ${stateClass}">${escapeHtml(status)}</span></header><p>${escapeHtml(text)}</p></article>`;
}

function parameterNames(node, key) {
  return String(node?.config?.[key] || "")
    .split(",")
    .map(name => name.trim())
    .filter(Boolean);
}

function parameterTableEditor(node) {
  let table = null;
  try { table = JSON.parse(node.config.parameter_table || "null"); } catch { /* Show the empty summary. */ }
  const inputCount = table?.columns?.filter(column => column.kind === "input").length || parameterNames(node, "condition_names").length;
  const outputCount = table?.columns?.filter(column => column.kind === "output").length || parameterNames(node, "feature_names").length;
  const rowCount = table?.rows?.length || 0;
  const datasetPath = table?.dataset_path || node.config.parameter_dataset || "Uses the HDF5 dataset connected to the target model";
  const feedsGenerator = state.edges.some(edge => edge.fromNode === node.id
    && state.nodes.find(candidate => candidate.id === edge.toNode)?.type === "run.cad_generator");
  const selected = selectedParameterCandidate(node);
  return `<section class="inspect-section parameter-editor">
    <div class="parameter-editor-head">
      <div>
        <div class="section-title">Dataset-aligned parameters</div>
        <p>Rows follow the connected HDF5 sample order. MLP uses paired Input and Output columns.</p>
      </div>
      <button class="button small primary" id="openParameterSpreadsheet" type="button">Open spreadsheet</button>
    </div>
    <div class="stat-grid">
      <div class="stat-card"><strong>${rowCount || "—"}</strong><small>matched rows</small></div>
      <div class="stat-card"><strong>${inputCount} / ${outputCount}</strong><small>input / output columns</small></div>
      ${feedsGenerator ? `<div class="stat-card"><strong>${selected.ready ? escapeHtml(selected.selectedSampleId) : "—"}</strong><small>generation row</small></div>` : ""}
    </div>
    <p class="parameter-editor-help">${escapeHtml(datasetPath)}</p>
    ${feedsGenerator ? `<p class="parameter-editor-help">${selected.ready
      ? `Native conditions: ${escapeHtml(selected.conditionNames)} → ${escapeHtml(selected.condValues)}`
      : "Open the spreadsheet, choose one generation row, and enter finite numeric values for every Input column."}</p>` : ""}
  </section>`;
}

export function embeddedInspector(node, spec) {
  if (spec.modelId === "simulgenvae") {
    return `<div class="inspect-section">
      <div class="section-title">SimulGen-VAE live contract</div>
      <div class="inspect-card-list">
        ${inspectorCard("Sequential pipeline", "Native", "train executes the hierarchical VAE stage and then the latent-conditioner stage; compatible completed stages may be reused.")}
        ${inspectorCard("VAE stage", "Native", "Compress fixed-geometry field tensors into a main latent code plus per-level hierarchical latent codes.")}
        ${inspectorCard("Latent conditioner", "Native", "Map conditions into the VAE latent representation: the dataset's own cond_var rows (lc_data_type hdf5, the default), an ordered CSV, or condition images.")}
        ${inspectorCard("Reconstruction", "Native", "Load both checkpoints, generate fields from conditions, write reconstructions.h5, and report field MSE.")}
        ${inspectorCard("Dataset gate", "Required", "The current loader requires uniform node count N and timestep count T across samples.", "adapter")}
      </div>
    </div>`;
  }
  if (spec.isModel) {
    return `<div class="inspect-section">
      <div class="section-title">Model-owned workspace</div>
      <div class="inspect-card-list">
        ${inspectorCard("Data contract", "Full config", "input_var / output_var / cond_var, edge and positional features, and normalization live in Full config; preflight checks them against the dataset's real row layout before launch.")}
        ${inspectorCard("Automatic preflight", "Before run", "Graph, config, route, paths, environment, dataset, checkpoint, native dry-run, and command checks.")}
        ${inspectorCard("Resources", "Presets + preflight", "gpu_ids, mixed precision, activation checkpointing, batch size and the Low-VRAM preset. Peak VRAM is read back from the run log in Train Metrics; it is not estimated in advance.")}
        ${inspectorCard("Training outputs", ".pth on disk", "Loss curves in Train Metrics, the checkpoint at modelpath, periodic prediction dumps under log_file_dir, and warm-start via the resume port.")}
      </div>
    </div>`;
  }
  if (node.type === "source.hdf5") {
    return `<div class="inspect-section"><div class="section-title">Dataset workspace</div><div class="inspect-card-list">
      ${inspectorCard("Samples and fields", "Click preview", "Inspect mesh, points, topology, field channels, timesteps, split, and provenance.")}
      ${inspectorCard("Safe parameter binding", "Schema aware", "Only declared input roles are editable; coordinates, targets, IDs, and outputs stay protected.")}
      ${inspectorCard("SimulGen compatibility", "Fixed geometry", "Show whether every selected sample has the same N and T required by SimulGen-VAE.", "adapter")}
    </div></div>`;
  }
  if (node.type === "run.inference") {
    return `<div class="inspect-section"><div class="section-title">Inference modes</div><div class="inspect-card-list">
      ${inspectorCard("Family-resolved run", "Automatic", "Single, batch, rollout, ensemble, and SimulGen reconstruction controls follow the linked model metadata.")}
      ${inspectorCard("SimulGen reconstruction", "Native", "Uses the VAE + LC bundle and ordered conditions to create field reconstructions and sample MSE.")}
      ${inspectorCard("Result viewer", "Embedded", "Prediction, truth, error, timestep player, distributions, and per-sample export.")}
    </div></div>`;
  }
  if (node.type === "optimize.design") {
    return `<div class="inspect-section"><div class="section-title">Optimization layers</div><div class="inspect-card-list">
      ${inspectorCard("Geometry feasibility", "CSV evidence", "Use explicit geometry-check columns from the selected evaluation CSV as hard constraints.", "adapter")}
      ${inspectorCard("Physics evaluators", "Actual outputs", "Consume completed inference or benchmark CSVs; no physics score is synthesized.", "adapter")}
      ${inspectorCard("Objectives and constraints", "CSV evidence", "Numeric columns, min/max direction, and hard inequalities read from the selected CSV.", "adapter")}
      ${inspectorCard("Pareto and diversity", "CSV evidence", "Feasible non-dominated set with crowding-distance top-k. There is no scalarization.", "adapter")}
      ${inspectorCard("Search and verification", "Roadmap", "DOE/evolutionary/Bayesian search, solver verification, OOD gates, and active learning.", "roadmap")}
    </div></div>`;
  }
  return "";
}

export function inputSourcePanel(node) {
  const meta = INPUT_SOURCE_META[node.type];
  if (!meta) return "";
  return `<section class="inspect-section input-source-panel">
    <div class="section-title">Input source</div>
    <div class="input-source-current">
      <strong>${escapeHtml(meta.label)}</strong>
      <small>${escapeHtml(node.config[meta.key] || "No input selected")}</small>
    </div>
    <div class="input-source-actions">
      <button class="button" id="browseInputSource">Browse repository…</button>
      <button class="button primary" id="uploadInputSource">Upload local file…</button>
    </div>
    <input id="inputSourceFile" type="file" accept="${escapeHtml(meta.accept)}" hidden>
    ${node.type === "source.cad" ? `<button class="button" id="createGeometrySample" style="width:100%;margin-top:7px">Create sample geometry</button>
    <p class="input-source-help">No CAD file handy? Generates a tiny real unit-cube STL under studio/runtime so the Geometry → HDF5 block is runnable end to end with no external dataset.</p>` : ""}
    <p class="input-source-help">The selected path is stored on this source block and follows its links into model preflight and execution.</p>
  </section>`;
}

export async function openInputPicker(nodeId) {
  const node = state.nodes.find(item => item.id === nodeId);
  const meta = node && INPUT_SOURCE_META[node.type];
  if (!node || !meta || !requireRuntime()) return;
  const request = activateStudioWorkspace("data", node.id);
  const container = liveShell(`Select ${meta.label}`, "Choose a real repository file. The path will be written to the selected source block.", request);
  try {
    const result = await apiRequest(`/api/files?kind=${encodeURIComponent(meta.kind)}`);
    if (!container?.isConnected) return;
    const accepted = new Set(meta.accept.split(","));
    const files = result.items
      .filter(item => accepted.has(item.extension))
      .sort((left, right) => left.path.localeCompare(right.path, undefined, { sensitivity: "base" }));
    container.innerHTML = `<div class="live-toolbar"><span><strong>${escapeHtml(meta.label)}</strong><small>${files.length}${result.truncated ? "+" : ""} selectable files</small></span><input id="inputPickerSearch" type="search" placeholder="Filter paths…"></div><div class="live-list" id="inputPickerList"></div>`;
    const renderFiles = query => {
      const normalized = query.trim().toLowerCase();
      const visible = files.filter(item => !normalized || item.path.toLowerCase().includes(normalized)).slice(0, 300);
      $("#inputPickerList").innerHTML = visible.length ? visible.map(item => `<article class="live-row">
        <span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.path)}</small></span>
        <span class="chip-row"><span class="chip">${escapeHtml(item.extension)}</span></span>
        <span><strong>${formatBytes(item.size)}</strong><small>${escapeHtml(item.modified)}</small></span>
        <span class="live-actions"><button class="button small primary" data-use-input="${escapeHtml(item.path)}">Use as input</button></span>
      </article>`).join("") : `<div class="live-empty">No matching files.</div>`;
      $$("[data-use-input]", container).forEach(button => button.addEventListener("click", () => {
        snapshot();
        node.config[meta.key] = button.dataset.useInput;
        markManualConfigValue(node, meta.key, button.dataset.useInput);
        applyGraphAutofill();
        savePipelineState();
        $("#studioOverlay").classList.remove("open");
        render();
        toast(`${meta.label} selected: ${button.dataset.useInput}`);
      }));
    };
    renderFiles("");
    on("#inputPickerSearch", "input", event => renderFiles(event.target.value));
  } catch (error) {
    liveError(container, error);
  }
}

export async function uploadInputFile(nodeId, file) {
  const node = state.nodes.find(item => item.id === nodeId);
  const meta = node && INPUT_SOURCE_META[node.type];
  if (!node || !meta || !file || !requireRuntime()) return;
  const button = $("#uploadInputSource");
  const original = button.textContent;
  button.disabled = true;
  button.textContent = `Uploading ${file.name}…`;
  try {
    const response = await fetch(`/api/upload?kind=${encodeURIComponent(meta.kind)}`, {
      method: "POST",
      headers: { "X-Filename": encodeURIComponent(file.name) },
      body: file
    });
    const result = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    if (!state.nodes.some(item => item.id === node.id)) return;
    snapshot();
    node.config[meta.key] = result.path;
    markManualConfigValue(node, meta.key, result.path);
    applyGraphAutofill();
    savePipelineState();
    render();
    toast(`Uploaded and selected ${file.name} (${formatBytes(result.size)}).`);
  } catch (error) {
    toast(`Upload failed: ${error.message}`, "error");
    button.disabled = false;
    button.textContent = original;
  }
}

export async function createGeometrySample(nodeId) {
  const node = state.nodes.find(item => item.id === nodeId);
  const meta = node && INPUT_SOURCE_META[node.type];
  if (!node || !meta || !requireRuntime()) return;
  const button = $("#createGeometrySample");
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Creating sample geometry…";
  try {
    const fixture = await apiRequest("/api/geometry/smoke-fixture", { method: "POST", body: {} });
    if (!state.nodes.some(item => item.id === node.id)) return;
    snapshot();
    node.config[meta.key] = fixture.path;
    markManualConfigValue(node, meta.key, fixture.path);
    applyGraphAutofill();
    savePipelineState();
    render();
    toast(`Created a real sample CAD file: ${fixture.path}`);
  } catch (error) {
    toast(`Could not create sample geometry: ${error.message}`, "error");
    button.disabled = false;
    button.textContent = original;
  }
}

/** Keys a run writes back onto a block: evidence, never user input. */
const RUN_EVIDENCE_KEYS = new Set([
  "results_path", "results_samples", "report_path", "export_path", "evaluated_samples", "job_id"
]);

/**
 * Keys whose value states what the block already does rather than configuring
 * it: nothing in the Studio, the launcher, or any native repo reads them. They
 * were rendered as ordinary text inputs, so "split · seeded 80/10/10" on a
 * dataset block looked like the control that picks the split (the real one is
 * split_seed on the model block) and editing it silently did nothing. Keep the
 * fact — it is worth knowing — but show it as a fact.
 */
const FIXED_BEHAVIOUR_KEYS = new Set([
  "split", "edit_mode", "range_policy", "version", "geometry_checks",
  "error_view", "qualification", "selection"
]);

/**
 * Same idea, scoped per block type, for keys that are real controls on one
 * block and pure statements on another. `mode` is the obvious case: it drives
 * model blocks, prep.geometry and run.cad_generator, but on run.inference it
 * is derived from the connected model and on optimize.design there is only one
 * mode. Everything listed here was verified to be read by no code at all:
 * editing it changed the label and nothing else.
 */
const FIXED_BEHAVIOUR_BY_TYPE = {
  "run.inference": new Set(["mode", "viewer"]),
  // selection/objectives/directions/constraints/top_k are all owned by the
  // Optimization workspace, which writes them back through assignManualConfig;
  // `selection` is a description of the fixed algorithm, not a choice.
  "optimize.design": new Set(["mode", "selection"]),
  // field_pairs and mapping_confirmed are the evaluation gate. Typing "True"
  // into mapping_confirmed here used to satisfy the "I inspected this mapping"
  // check without ever opening the mapping -- the one control whose whole point
  // is that a human looked at it.
  "evaluate.predictions": new Set(["metrics", "aggregate", "field_pairs", "mapping_confirmed"]),
  // The Comparison workspace resolves the metric from the runs' shared metric
  // keys and writes csv_metric/csv_direction; these two rows were free text that
  // nothing read back.
  "evaluate.compare": new Set(["metric", "direction", "qualification"]),
  "output.export": new Set(["format", "path"]),
  "source.cad": new Set(["units"]),
  // Filled from the checkpoint's own metadata by autofill; typing over it
  // detached the row from the file without changing what would be loaded.
  "source.checkpoint": new Set(["compatibility"])
};

/**
 * prep.geometry rows that the block's own current mode/reader makes inert.
 *
 * `inspect` is a dry run -- geometryConfigText emits no output_dataset and the
 * native pipeline writes nothing -- and the two gmsh sizing knobs reach no code
 * path when the reader is trimesh. Both were shown as ordinary editable fields,
 * so the block offered four settings that could not affect its own run.
 */
/**
 * Where a read-only row's value is actually set, for the rows a Studio workspace
 * owns. Without this the inspector labelled them "fixed behaviour", which reads
 * as "nothing can change this" for values the user genuinely can change -- just
 * not from here.
 */
const FIXED_BEHAVIOUR_SOURCE = {
  "evaluate.compare": { metric: "set in Comparison", direction: "set in Comparison", qualification: "not enforced" },
  "evaluate.predictions": { field_pairs: "set in Evaluation", mapping_confirmed: "set in Evaluation" },
  "optimize.design": { selection: "fixed algorithm" }
};

/** Mirrors pipeline.pointcloud_output_path() in methods/GeometryIngest. */
function pointCloudSidecar(outputDataset) {
  const path = String(outputDataset || "").trim();
  if (!path) return "<output_dataset>_pointcloud.h5";
  const dot = path.lastIndexOf(".");
  const slash = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return dot > slash
    ? `${path.slice(0, dot)}_pointcloud${path.slice(dot)}`
    : `${path}_pointcloud.h5`;
}

function inertGeometryKey(node, key) {
  if (node.type !== "prep.geometry") return false;
  const mode = String(node.config.mode || "").toLowerCase();
  const reader = String(node.config.reader || "").toLowerCase();
  // `emit` stays visible in inspect: it still decides whether process_one
  // computes the point cloud, which the dry run reports. num_fields reaches
  // writer.write_contract only, and output_dataset is never opened.
  if (mode === "inspect" && ["output_dataset", "num_fields"].includes(key)) return true;
  if (reader === "trimesh" && ["mesh_size_min", "mesh_size_max"].includes(key)) return true;
  if (!String(node.config.emit || "").includes("pointcloud")
    && ["num_points", "resample_method"].includes(key)) return true;
  return false;
}

function isFixedBehaviour(node, key) {
  return FIXED_BEHAVIOUR_KEYS.has(key) || Boolean(FIXED_BEHAVIOUR_BY_TYPE[node.type]?.has(key));
}

/**
 * Which config rows a model block shows in the side inspector.
 *
 * This used to be `Object.entries(node.config).slice(0, 6)` — the first six keys
 * *in the order the defaults object literal happened to be typed*. The result
 * was that every model surfaced `model` (which is the block's own identity and
 * must not be edited) and buried `training_epochs`, `batch_size` and
 * `learningr` — the three knobs anyone actually turns — behind "Full config".
 * FNO, for instance, spent two of its six rows on `coordinate_normalization`
 * (which has exactly one legal value) and `fno_grid_resolution`, and showed no
 * training control at all. Worse, the panel silently reshuffled whenever
 * someone reordered a defaults object.
 *
 * Eight rows, not six: after the mode, the primary input, the artifact and the
 * training trio, two slots remain for the keys that distinguish this route
 * (hidden_layers on MLP, slice_num on Transolver, the multiscale controls on
 * HI-MGN). Six left no room for any of them.
 *
 * So: rank by what the block is for, and stay mode-aware — a block running a
 * trained model wants the data it reads, the checkpoint it loads and where it
 * writes, not an epoch budget.
 * Keys absent from the config are skipped, and anything left over keeps its
 * original order, so a route with unusual keys still fills its rows.
 */
const MODEL_INSPECTOR_PRIORITY = [
  "mode",
  // Primary input. `inference` flips which of the two is the real one, so both
  // are listed and the irrelevant one is simply absent from that mode's config.
  "infer_dataset", "dataset_dir",
  // Primary artifact, including the staged variants. output_dir is where the
  // non-training modes write (and is required by sdfflow evaluate/sample/
  // reconstruct/interpolate/optimize), so it belongs with them.
  "modelpath", "vae_modelpath", "fm_modelpath", "lc_modelpath", "output_dir",
  // The training trio, plus the per-stage spellings SDFFlow/SimulGen-VAE use.
  "training_epochs", "vae_training_epochs", "fm_training_epochs", "lc_training_epochs",
  "batch_size", "vae_batch_size", "fm_batch_size", "lc_batch_size",
  "learningr", "vae_learningr", "fm_learningr", "lc_learningr",
  // Generative run knobs: for sdfflow sample/interpolate these *are* the job.
  "num_samples", "seed", "ode_steps", "mc_resolution",
  "gpu_ids"
];

/**
 * The one-line explanation under a field.
 *
 * Model blocks have a Full config sheet that already shows HELP for every key,
 * so repeating it in an 8-row summary panel would crowd it out. Every other
 * block has no second surface at all: prep.geometry's twelve fields -- reader,
 * emit, mesh_size_max and the rest -- had nowhere to say what they mean.
 */
function rowHelp(node, key) {
  if (BLOCK_SPECS[node.type]?.isModel) return "";
  const text = HELP[key];
  return text ? `<small class="row-help">${escapeHtml(text)}</small>` : "";
}

function modelInspectorEntries(node, modelId, limit) {
  const config = node.config || {};
  const mode = String(config.mode || "").toLowerCase();
  // Every mode that is not one of the four training modes runs a trained model.
  // Keying on that, rather than on a list of inference-ish names, is what keeps
  // sdfflow's `evaluate` / `sample` / `interpolate` / `optimize` correct: they
  // showed three epoch budgets each while the artifact they write did not fit.
  const TRAINING_MODES = new Set(["train", "train_vae", "train_fm", "train_lc"]);
  const isRunMode = Boolean(mode) && !TRAINING_MODES.has(mode);
  const readsHeldOut = mode === "inference" || mode === "reconstruct";
  // The live spec's per-mode required set decides between competing spellings,
  // so this needs no per-model table and cannot drift from the launcher. It is
  // what separates SDFFlow's merged `train` (vae_training_epochs /
  // fm_training_epochs are required, plain training_epochs does nothing) from
  // its `train_vae` / `train_fm` stages, which use the generic trio instead.
  let required;
  try { required = requiredFor(modelId, mode); } catch { required = new Set(); }
  // Training-only knobs stay *in the config* when the mode is inference or
  // reconstruct -- the block keeps its training values so switching back is
  // lossless -- but they do nothing in those modes, and the panel used to spend
  // three of its rows on them. Hide them here; the Full config sheet still
  // lists everything.
  // Deliberately NOT batch_size: SimulGen-VAE's `reconstruct` batches its
  // decode loop with it (inference_profiles/reconstruct.py reads
  // config['batch_size']), so hiding it there would hide a live control.
  // Epochs, learning rate, warm-up and weight decay are optimizer-only.
  const trainingOnly = key => isRunMode && /^(?:vae_|fm_|lc_)?(?:training_epochs|learningr|warmup_epochs|weight_decay)$/.test(key);
  const applicable = MODEL_INSPECTOR_PRIORITY.filter(key => {
    if (!(key in config)) return false;
    // Both dataset keys are usually present; show the one this mode reads.
    if (key === "dataset_dir" && readsHeldOut && "infer_dataset" in config) return false;
    if (key === "infer_dataset" && !readsHeldOut) return false;
    if (trainingOnly(key)) return false;
    return true;
  });
  const ranked = [
    ...applicable.filter(key => required.has(key)),
    ...applicable.filter(key => !required.has(key))
  ];
  // `model` is deliberately excluded: it is the block's identity, the header
  // already names it, and the summary line above prints the exact route id.
  const rest = Object.keys(config).filter(key => key !== "model" && !ranked.includes(key) && !trainingOnly(key));
  return [...ranked, ...rest].slice(0, limit).map(key => [key, config[key]]);
}

const WORKSPACE_ACTION_LABELS = {
  comparison: "Open comparison",
  deploy: "Open deployment",
  evaluation: "Open evaluation",
  export: "Open export",
  optimization: "Open optimization"
};

/** Keep the prominent actions truthful and avoid duplicate destinations. */
function inspectorActions(node, spec) {
  if (spec.isModel) return { primary: "Start / resume", secondary: "Model details" };
  if (spec.isMetricsViewer) return { primary: "Open metrics" };
  if (spec.workspace) {
    return { primary: WORKSPACE_ACTION_LABELS[spec.workspace] || "Open workspace" };
  }
  if (node.type === "source.hdf5") return { primary: "Open samples" };
  if (node.type === "source.cad") return { primary: "Browse files", secondary: "Open geometry" };
  if (node.type === "source.parameters") return { primary: "Browse files", secondary: "Open spreadsheet" };
  return { primary: "Run selected", secondary: "Open samples" };
}

export function renderInspector() {
  applyGraphAutofill();
  const node = state.nodes.find(item => item.id === state.selectedNode);
  const edge = state.edges.find(item => item.id === state.selectedEdge);
  if (!node && edge) {
    const source = state.nodes.find(item => item.id === edge.fromNode);
    const target = state.nodes.find(item => item.id === edge.toNode);
    const sourceSpec = source && BLOCK_SPECS[source.type];
    const targetSpec = target && BLOCK_SPECS[target.type];
    const output = sourceSpec?.outputs.find(port => port.id === edge.fromPort);
    const input = targetSpec?.inputs.find(port => port.id === edge.toPort);
    $("#inspectorHint").textContent = "selected connection";
    $("#inspectorContent").innerHTML = `<section class="inspect-hero">
      <div class="inspect-meta"><span class="type-chip">${escapeHtml(output?.type || "connection")}</span><span class="status"><i></i>Connected</span></div>
      <h2>${escapeHtml(sourceSpec?.label || edge.fromNode)} → ${escapeHtml(targetSpec?.label || edge.toNode)}</h2>
      <p>This exact typed connection is selected. Remove it without deleting either block.</p>
    </section>
    <section class="inspect-section"><div class="section-title">Connection contract</div><div class="port-list">
      <div class="port-row"><i style="--port:${typeColor(output?.type || "artifact")}"></i><span>→ ${escapeHtml(output?.label || edge.fromPort)}</span><small>${escapeHtml(output?.type || "unknown")}</small></div>
      <div class="port-row"><i style="--port:${typeColor(input?.type || "artifact")}"></i><span>← ${escapeHtml(input?.label || edge.toPort)}</span><small>${escapeHtml(input?.type || "unknown")}</small></div>
    </div></section>
    <section class="inspect-section"><button class="button danger" id="deleteConnection" style="width:100%">Delete connection</button><p class="input-source-help">You can also press Delete or Backspace while the connection is selected.</p></section>`;
    on("#deleteConnection", "click", deleteSelected);
    return;
  }
  if (!node) {
    $("#inspectorHint").textContent = "select a block";
    $("#inspectorContent").innerHTML = `<div class="inspect-empty"><div><span>⌁</span><strong>Select a pipeline block</strong><p>Configure it, inspect typed ports, link it to other blocks, run dependencies, or open individual samples.</p></div></div>`;
    return;
  }
  const spec = BLOCK_SPECS[node.type];
  $("#inspectorHint").textContent = spec.maturity;
  const configEntries = spec.isModel
    ? modelInspectorEntries(node, spec.modelId, 8)
    : Object.entries(node.config)
      .filter(([key]) => node.type !== "source.parameters" || !["condition_names", "feature_names", "parameter_table", "parameter_dataset"].includes(key))
      .filter(([key]) => !inertGeometryKey(node, key))
      .slice(0, 20);
  const inspectorChoices = node.type === "prep.geometry"
    ? {
        mode: ["inspect", "ingest"],
        reader: ["auto", "trimesh", "gmsh"],
        mesh_type: ["surface", "volume"],
        emit: ["graph", "pointcloud", "graph, pointcloud"],
        resample_method: ["fps", "random"]
      }
    : node.type === "run.inference"
      // Only the families whose checkpoints record enough to rebuild the model
      // without their training config; the rest still need their model block.
      ? {
          model_id: ["", ...STANDALONE_INFERENCE_MODEL_IDS],
          flow_solver: ["", "heun", "euler"],
          flow_predict: ["", "sample", "mean", "ensemble_mean"]
        }
      // mapping_confirmed is NOT offered here. It gates whether a positional
      // field mapping may score, and its entire meaning is "a human looked at
      // the mapping" -- which is exactly what the Evaluation workspace shows and
      // this panel does not. It was a dropdown; flipping it to True from here
      // satisfied the gate without ever opening the mapping. It is now a
      // read-only statement (FIXED_BEHAVIOUR_BY_TYPE) sourced from that
      // workspace, alongside field_pairs.
      : node.type === "evaluate.predictions"
        ? { mapping_mode: ["schema", "legacy"] }
        : node.type === "run.cad_generator"
          // `optimize` runs the closed generate -> analyze -> search loop
          // instead of producing a plain candidate batch; opt_analysis then
          // picks whether "analyze" is the exact FEA solve or a fast but
          // currently unproven HI-MGN forward pass.
          ? { mode: ["sample", "reconstruct", "interpolate", "optimize"],
              opt_analysis: ["fea", "surrogate"] }
          : {};
  const ports = [
    ...spec.inputs.map(port => ({ ...port, direction: "in" })),
    ...spec.outputs.map(port => ({ ...port, direction: "out" }))
  ];
  const actions = inspectorActions(node, spec);
  $("#inspectorContent").innerHTML = `
    <section class="inspect-hero">
      <div class="inspect-meta"><span class="type-chip">${escapeHtml(node.type)}</span><span class="status"><i></i>${node.status === "idle" ? "Ready" : node.status}</span></div>
      <h2>${escapeHtml(spec.label)}</h2>
      <p>${escapeHtml(spec.description)}</p>
      <div class="inspect-actions"><button class="button primary" id="inspectorRun">${escapeHtml(actions.primary)}</button>${actions.secondary ? `<button class="button" id="inspectorSamples">${escapeHtml(actions.secondary)}</button>` : ""}</div>
    </section>
    <section class="inspect-section">
      <div class="section-title">${spec.isModel ? "ML configuration" : "Configuration"}</div>
      ${spec.isModel ? `<div class="config-summary"><span><strong>${escapeHtml(spec.modelId)}</strong><small>${MODEL_CATALOG[spec.modelId].keys.length} keys · ${MODEL_CATALOG[spec.modelId].modes.length} modes · ${escapeHtml(MODEL_CATALOG[spec.modelId].dataset)}${autoFillCount(node) ? ` · ${autoFillCount(node)} graph-filled` : ""}</small></span><button class="button small primary" id="openFullConfig">Full config</button></div>` : ""}
      <div style="margin-top:${spec.isModel ? 9 : 0}px">${configEntries.map(([key, value]) => {
        // Model blocks: reuse the config sheet's own choice table rather than a
        // second hand-written one. Only `mode` used to become a <select> here,
        // so parallel_mode, activation, coordinate_normalization, lc_data_type,
        // coarsening_type, flow_solver, best_by and every boolean were free-text
        // in the inspector while the Full config sheet offered a dropdown for
        // the exact same key -- the inspector happily accepted values the
        // launcher rejects.
        const modelChoices = spec.isModel ? choicesFor(spec.modelId, key) : null;
        if (modelChoices?.length) {
          return `<div class="form-row"><label>${escapeHtml(key.replaceAll("_", " "))}</label><select class="field inspector-config" data-key="${key}">${modelChoices.map(choice => `<option value="${escapeHtml(choice)}"${String(value) === String(choice) ? " selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select></div>`;
        }
        if (inspectorChoices[key]) {
          return `<div class="form-row"><label>${escapeHtml(key.replaceAll("_", " "))}</label><select class="field inspector-config" data-key="${key}">${inspectorChoices[key].map(choice => `<option value="${escapeHtml(choice)}"${String(value) === choice ? " selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select>${rowHelp(node, key)}</div>`;
        }
        if (RUN_EVIDENCE_KEYS.has(key)) {
          // What a run produced, not something to configure. Rendering it as a
          // text input invited edits that change the label without changing
          // anything real -- typing over "results samples" would relabel the
          // canvas while the results on disk stayed exactly as they were.
          return `<div class="form-row run-evidence"><label>${escapeHtml(key.replaceAll("_", " "))}<small class="inline-auto">from the last run</small></label><output class="field readonly" title="${escapeHtml(value)}">${escapeHtml(value) || "—"}</output></div>`;
        }
        if (isFixedBehaviour(node, key)) {
          // A fixed row can still be graph-filled (source.checkpoint's
          // compatibility comes from the file's metadata); say where it came
          // from rather than the generic tag when that is the case. Rows a
          // workspace owns say so, so "fixed behaviour" is not used to describe
          // a value the user really can change -- somewhere else.
          const filled = autoFillMeta(node, key);
          const note = filled
            ? `auto · ${escapeHtml(filled.sourceLabel)}`
            : FIXED_BEHAVIOUR_SOURCE[node.type]?.[key] || "fixed behaviour";
          return `<div class="form-row run-evidence"><label>${escapeHtml(key.replaceAll("_", " "))}<small class="inline-auto">${note}</small></label><output class="field readonly" title="${escapeHtml(value)}">${escapeHtml(value) || "—"}</output></div>`;
        }
        const automatic = autoFillMeta(node, key);
        return `<div class="form-row${automatic ? " graph-autofilled" : ""}"><label>${escapeHtml(key.replaceAll("_", " "))}${automatic ? `<small class="inline-auto">auto · ${escapeHtml(automatic.sourceLabel)}</small>` : ""}</label><input class="field inspector-config" data-key="${key}" value="${escapeHtml(value)}">${rowHelp(node, key)}</div>`;
      }).join("")}</div>
    </section>
    ${node.type === "run.cad_generator"
      && String(node.config.mode || "").toLowerCase() === "optimize"
      && String(node.config.opt_analysis || "fea").toLowerCase() === "surrogate"
      ? `<section class="inspect-section"><div class="section-title">Surrogate accuracy gate</div><div class="diagnostic warning"><i></i><div><strong>Demonstration path, not verified structural evidence.</strong><br>Add <code>opt_surrogate_checkpoint</code> and <code>opt_surrogate_config</code> in the connected SDFFlow block's Full config. Preflight blocks a missing pair. Use FEA for actionable stress or displacement values until the surrogate is validated on representative held-out designs.</div></div></section>`
      : ""}
    ${node.type === "prep.geometry" && String(node.config.emit || "").includes("pointcloud") && String(node.config.mode || "").toLowerCase() === "ingest"
      // pipeline.py writes the point cloud to a sidecar next to the graph file
      // (pointcloud_output_path: "<stem>_pointcloud<ext>"). Nothing in the graph
      // named it, so a run configured for both emits produced a second dataset
      // that no downstream block and no export could see.
      ? `<section class="inspect-section"><div class="section-title">Also written</div><div class="form-row run-evidence"><label>point cloud<small class="inline-auto">sidecar file</small></label><output class="field readonly">${escapeHtml(pointCloudSidecar(node.config.output_dataset))}</output></div><p class="input-source-help">The <code>graph</code> emit writes <code>output_dataset</code>; the <code>pointcloud</code> emit writes this second file beside it. Point an Export or HDF5 Dataset block at it to use it.</p></section>`
      : ""}
    ${inputSourcePanel(node)}
    ${embeddedInspector(node, spec)}
    ${node.type === "source.parameters" ? parameterTableEditor(node) : ""}
    <section class="inspect-section"><div class="section-title">Typed ports</div><div class="port-list">
      ${ports.map(port => `<div class="port-row"><i style="--port:${typeColor(port.type)}"></i><span>${port.direction === "in" ? "←" : "→"} ${escapeHtml(port.label)}${port.required ? " *" : ""}</span><small>${escapeHtml(TYPE_META[port.type]?.label || port.type)}</small></div>`).join("")}
    </div></section>
    ${node.type === "source.parameters" ? "" : `<section class="inspect-section"><div class="section-title">Evidence preview</div><article class="artifact-strip"><div class="artifact-strip-visual">${previewGraphic(spec.visual, node.id.length + 3)}</div><footer><span><strong>${escapeHtml(nodeEvidenceLabel(node, spec))}</strong><small>${escapeHtml(spec.workspace ? `${spec.workspace} workspace status` : nodeVisualLabel(spec))}</small></span></footer></article></section>`}
    <section class="inspect-section"><div style="display:grid;grid-template-columns:1fr 1fr;gap:6px"><button class="button" id="duplicateNode">Duplicate</button><button class="button danger" id="deleteNode">Delete block</button></div></section>
  `;
  $$(".inspector-config").forEach(control => control.addEventListener("change", () => {
    snapshot();
    node.config[control.dataset.key] = control.value;
    markManualConfigValue(node, control.dataset.key, control.value);
    applyGraphAutofill();
    renderInspector();
    // The canvas card shows the configured mode (title verb and kind line) and
    // autofill may have rewritten other blocks' values, so redraw it too --
    // editing `mode` here used to leave the card claiming the old one.
    render();
    toast(`Updated ${control.dataset.key}.`);
  }));
  $("#openParameterSpreadsheet")?.addEventListener("click", () => openArtifact(node.id));
  on("#inspectorRun", "click", () => {
    if (spec.workspace) openStudio(spec.workspace, node.id);
    else if (node.type === "evaluate.predictions") openStudio("evaluation");
    else if (node.type === "evaluate.compare") openStudio("comparison");
    else if (node.type === "evaluate.training_metrics") openTrainingMetricsWorkspace(node.id);
    else if (node.type === "output.export") openStudio("export");
    else if (node.type === "source.hdf5") openArtifact(node.id);
    else if (node.type === "source.cad" || node.type === "source.parameters") openStudio("data");
    else if (node.type === "source.checkpoint" || node.type === "deploy.api") openStudio("deploy");
    else runGraph(node.id);
  });
  const openPrimaryDetails = () => spec.isModel
    ? openModelDetailWorkspace(spec.modelId)
    : spec.isMetricsViewer
      ? openTrainingMetricsWorkspace(node.id)
      : spec.workspace
        ? openStudio(spec.workspace, node.id)
      : openArtifact(node.id);
  on("#inspectorSamples", "click", openPrimaryDetails);
  on("#duplicateNode", "click", () => duplicateNode(node.id));
  on("#deleteNode", "click", deleteSelected);
  $("#openFullConfig")?.addEventListener("click", () => openConfig(node.id));
  $("#browseInputSource")?.addEventListener("click", () => openInputPicker(node.id));
  $("#uploadInputSource")?.addEventListener("click", () => $("#inputSourceFile").click());
  $("#createGeometrySample")?.addEventListener("click", () => createGeometrySample(node.id));
  $("#inputSourceFile")?.addEventListener("change", event => {
    const file = event.target.files?.[0];
    if (file) uploadInputFile(node.id, file);
    event.target.value = "";
  });
}
