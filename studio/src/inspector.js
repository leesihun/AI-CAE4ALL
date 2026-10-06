import { $, $$, escapeHtml, toast, formatBytes, on, closeOverlay } from "./dom.js";
import { state, snapshot } from "./state.js";
import { savePipelineState } from "./persistence.js";
import { BLOCK_SPECS, MODEL_CATALOG, TYPE_META, INPUT_SOURCE_META, HELP } from "./constants.js";
import { apiRequest, requireRuntime } from "./api.js";
import {
  typeColor, STANDALONE_INFERENCE_MODEL_IDS, SDFFLOW_OPTIMIZE_INERT_KEYS, CAD_GENERATOR_ROW_MODES, VER_SURROGATE_SETTINGS,
  inferenceModel, cadGeneratorMode, cadGeneratorValue, cadRowActive, cadSurrogate, cadUsesSurrogate
} from "./validate.js";
import { blockFacts, factsTable } from "./cards.js";
import { duplicateNode, deleteSelected, render } from "./graph.js";
import { openConfig, choicesFor, requiredFor } from "./config.js";
import { openArtifact } from "./viewer.js";
import { runGraph } from "./run.js";
import { activateStudioWorkspace, openStudio, openModelDetailWorkspace, openTrainingMetricsWorkspace, liveShell, liveError } from "./studio.js";
import { applyGraphAutofill, autoFillCount, autoFillMeta, markManualConfigValue, selectedParameterCandidate } from "./autofill.js";

/**
 * Closed-choice values compare case-insensitively, exactly as the config
 * sheet's dropdown does: the native parsers lowercase every non-path value, so
 * "DDP" in a loaded config is "ddp". A case-sensitive match selected nothing,
 * and the <select> then displayed its FIRST option as though it were the
 * configured value. Path keys keep their case but never carry a choice list,
 * so they cannot reach this comparison.
 */
function sameChoice(value, choice) {
  return String(value ?? "").toLowerCase() === String(choice ?? "").toLowerCase();
}

/** A choice list that still offers (and so selects) a configured value outside it. */
function withCurrent(choices, current) {
  const value = String(current ?? "").trim();
  return value && !choices.some(choice => sameChoice(value, choice)) ? [...choices, value] : choices;
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

export function inputSourcePanel(node) {
  const meta = INPUT_SOURCE_META[node.type];
  if (!meta) return "";
  return `<section class="inspect-section input-source-panel">
    <div class="section-title">${escapeHtml(meta.label)}</div>
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
        closeOverlay("studioOverlay");
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
  "results_path", "results_samples", "results_dir", "report_path", "export_path", "evaluated_samples", "job_id"
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

// `label` is markup (keyLabel output or a literal); `value` and `note` are text.
const readonlyRow = (label, value, note) => `<div class="form-row run-evidence"><label>${label}${note ? `<small class="inline-auto">${escapeHtml(note)}</small>` : ""}</label><output class="field readonly" title="${escapeHtml(value)}">${escapeHtml(value) || "—"}</output></div>`;

/**
 * What the "HI-MGN surrogate" wire does to this CAD Generator, and the
 * surrogate search's honesty statement.
 *
 * Every value here is read from what the run config is built from
 * (executableSteps): the wired block, the generator's own rows, and the
 * connected SDFFlow block under them. A wire the current mode ignores says so
 * instead of blocking the run.
 */
function surrogateSection(node) {
  const surrogate = cadSurrogate(node);
  const active = cadUsesSurrogate(node);
  if (!surrogate && !active) return "";
  const mode = cadGeneratorMode(node);
  const sourceLabel = surrogate ? (BLOCK_SPECS[surrogate.source.type]?.label || surrogate.source.id) : "";
  if (!active) {
    const next = mode === "optimize"
      ? "Set <strong>opt analysis</strong> to <code>surrogate</code> to rank designs with it."
      : `Switch <strong>mode</strong> to <code>optimize</code> to search designs with it; this ${escapeHtml(mode)} run ignores it.`;
    return `<section class="inspect-section"><div class="section-title">HI-MGN surrogate</div><div class="diagnostic"><i></i><div><strong>${escapeHtml(sourceLabel)} is wired but unused in this mode.</strong><br>It is read only by <code>mode optimize</code> with <code>opt_analysis surrogate</code>. ${next}</div></div></section>`;
  }
  const modelEdge = state.edges.find(edge => edge.toNode === node.id && edge.toPort === "model");
  const modelNode = modelEdge && state.nodes.find(candidate => candidate.id === modelEdge.fromNode);
  const merged = { ...(modelNode?.config || {}), ...node.config };
  const ver = surrogate?.layout === "ver";
  const rows = [
    readonlyRow("surrogate model", surrogate ? `${sourceLabel} (${surrogate.source.id})` : "", surrogate ? "wired" : "wire an HI-MGN model block in"),
    readonlyRow("checkpoint", String(merged.opt_surrogate_checkpoint || ""), surrogate ? "from the wire" : "SDFFlow Full config"),
    surrogate?.block
      ? readonlyRow("inference config", `written from ${surrogate.block.id} at Validate / Run`, "automatic")
      : readonlyRow("inference config", String(merged.opt_surrogate_config || ""), "SDFFlow Full config"),
    ...(ver ? Object.entries(VER_SURROGATE_SETTINGS).map(([key, value]) => readonlyRow(keyLabel(key), value, "fixed by the ex13 labels")) : [])
  ].join("");
  const verified = !ver && ["true", "1", "yes", "on"].includes(String(merged.opt_fea_verify ?? "").trim().toLowerCase());
  const layoutNote = ver
    ? "This surrogate was trained on the ex13 vertical-load labels (output_var 4, cond_var 0), so the run is set to the vertical load case in DeepJEB's frame at 4470 kg/m³. FEA re-verification stays off: <code>opt_fea_verify</code> re-solves with fea.py's pad / lug-crown boundary conditions, not the bolt-bore / lug-bore rule these labels used, so it would grade the surrogate against a different problem."
    : "";
  const gate = verified
    ? `<div class="diagnostic"><i></i><div><strong>FEA verification on (<code>opt_fea_verify</code>).</strong><br>The search ranks with HI-MGN; afterwards the optimized design, the best baseline, and the typical baseline are re-solved with the tet4 FEA solver. report.md and the <code>fea_*</code> columns of the optimization table carry the solver's numbers: judge the design on those.</div></div>`
    : `<div class="diagnostic warning"><i></i><div><strong>Demonstration path, not verified structural evidence.</strong><br>Every number this run reports is a HI-MGN prediction. ${ver ? layoutNote + " Check the winner with an independent analysis before acting on it." : "Set <code>opt_fea_verify</code> True in the connected SDFFlow block's Full config to re-solve the result with the real solver at the end, or use FEA analysis."}</div></div>`;
  return `<section class="inspect-section"><div class="section-title">HI-MGN surrogate</div>${rows}${gate}</section>`;
}

/**
 * The CAD Generator's rows for its current mode: mode first, then the rows
 * only this mode reads (interpolate's endpoints, reconstruct's mesh), then the
 * rest. A row the mode does not read is not offered (the run config blanks it
 * too), and a row a saved graph predates is shown at the value it runs with.
 */
function cadGeneratorEntries(node) {
  const mode = cadGeneratorMode(node);
  const rank = key => key === "mode" ? 0 : CAD_GENERATOR_ROW_MODES[key]?.length === 1 && CAD_GENERATOR_ROW_MODES[key][0] === mode ? 1 : 2;
  return [...new Set([...Object.keys(BLOCK_SPECS[node.type].defaults), ...Object.keys(node.config)])]
    // Pre-rename names; their values run (and show) as num_samples / cfg_scale.
    .filter(key => !["candidates", "guidance"].includes(key))
    // Shown, with what it is used for, in the HI-MGN surrogate section.
    .filter(key => key !== "opt_surrogate_checkpoint")
    .filter(key => cadRowActive(mode, key))
    .filter(key => !(mode === "optimize" && SDFFLOW_OPTIMIZE_INERT_KEYS.has(key)))
    .map((key, index) => ({ key, index }))
    .sort((a, b) => rank(a.key) - rank(b.key) || a.index - b.index)
    .map(({ key }) => [key, cadGeneratorValue(node, key)]);
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
/**
 * Readable labels for the rows the inspector shows. The raw key still renders
 * beside it in small monospace, because it is what the config file, the logs
 * and the docs all use; the label only has to say what the value *is*.
 */
const KEY_LABELS = {
  mode: "Mode", model_id: "Model family", gpu_ids: "GPUs", seed: "Random seed",
  dataset_dir: "Training data", infer_dataset: "Held-out data", path: "File",
  modelpath: "Checkpoint", vae_modelpath: "VAE checkpoint", fm_modelpath: "Flow checkpoint",
  lc_modelpath: "LC checkpoint", output_dir: "Output folder",
  training_epochs: "Epochs", vae_training_epochs: "VAE epochs", fm_training_epochs: "Flow epochs",
  lc_training_epochs: "LC epochs", batch_size: "Batch size", fm_batch_size: "Flow batch size",
  lc_batch_size: "LC batch size", learningr: "Learning rate", vae_learningr: "VAE learning rate",
  fm_learningr: "Flow learning rate", lc_learningr: "LC learning rate",
  num_samples: "Shapes to generate", ode_steps: "ODE steps", mc_resolution: "Mesh resolution",
  cfg_scale: "Guidance scale", cond_values: "Condition values", opt_analysis: "Analysis backend", opt_vertical_disp_max: "Vertical deflection limit (mm)", opt_load_cases: "Load cases",
  infer_timesteps: "Rollout steps", inference_output_dir: "Results folder", num_workers: "Loader workers",
  infer_chunk_size: "Node chunk", infer_query_chunk_size: "Query chunk",
  num_vae_samples: "Ensemble draws", flow_steps: "Flow steps", flow_solver: "Flow solver",
  flow_predict: "Flow output", reader: "Reader", mesh_type: "Mesh type", emit: "Writes",
  num_fields: "Field rows", num_points: "Points per sample", resample_method: "Resampling",
  mesh_size_min: "Min element size", mesh_size_max: "Max element size",
  output_dataset: "Output dataset", limit: "File limit", export_label: "Export name",
  csv_path: "Candidate CSV", objectives: "Objectives", directions: "Directions",
  constraints: "Constraints", top_k: "Designs kept", mapping_mode: "Field mapping",
  field_pairs: "Field pairs", mapping_confirmed: "Mapping reviewed", job_id: "Training run",
  excluded_metrics: "Hidden metrics", smoothing: "Smoothing", y_scale: "Y axis", metric: "Metric",
  direction: "Direction", compatibility: "Detected model", checkpoint_path: "Checkpoint",
  input_path: "Sample input", output_name: "Output name", timesteps: "Rollout steps",
  results_path: "Results", results_samples: "Result samples", results_dir: "Results folder",
  report_path: "Report",
  export_path: "Exported to", evaluated_samples: "Samples scored", binding: "Bound to",
  value: "Value", format: "Copy mode"
};

function keyLabel(key) {
  const label = KEY_LABELS[key] || (key.charAt(0).toUpperCase() + key.slice(1).replaceAll("_", " "));
  return `${escapeHtml(label)}<span class="raw-key">${escapeHtml(key)}</span>`;
}

/**
 * Rows that are prose about the block, not settings of it. They were already
 * read-only, but a read-only "split · seeded 80/10/10" still reads as a fact
 * about *this* dataset -- and the split is chosen by the model block's
 * split_seed, not here -- while "viewer · prediction · truth · error ..." is a
 * feature list. The block description and the At-a-glance facts say what is
 * true; these rows only took space from the settings that matter.
 */
const STATEMENT_ROWS = {
  "*": new Set(["split", "edit_mode", "range_policy", "version", "geometry_checks", "error_view", "qualification", "selection"]),
  "run.inference": new Set(["mode", "viewer"]),
  "optimize.design": new Set(["mode"]),
  "evaluate.predictions": new Set(["metrics", "aggregate"]),
  "source.cad": new Set(["units"])
};

function isStatementRow(node, key) {
  return STATEMENT_ROWS["*"].has(key) || Boolean(STATEMENT_ROWS[node.type]?.has(key));
}

/**
 * run.inference carries every family's inference knobs, because which family
 * runs is decided by the link, not the block. Mirrors the specs' known_keys:
 * transolver alone reads infer_chunk_size, the operators infer_query_chunk_size,
 * the stochastic MGN pair the ensemble keys, cHI-MGNflow the flow keys. A key
 * the linked family cannot read is hidden unless the user filled it in.
 */
const INFERENCE_KEY_FAMILIES = {
  infer_chunk_size: ["transolver"],
  infer_query_chunk_size: ["fno", "deeponet", "point_deeponet"],
  num_vae_samples: ["meshgraphnets-v", "chi-mgnflow"],
  vae_batch_size: ["meshgraphnets-v", "chi-mgnflow"],
  flow_steps: ["chi-mgnflow"], flow_solver: ["chi-mgnflow"], flow_predict: ["chi-mgnflow"]
};

function inferenceKeyApplies(node, key, resolved) {
  const filled = Boolean(String(node.config[key] || "").trim());
  // With a model block linked, the family is that block's (inferenceModel
  // prefers the trainer), so the field would be ignored and only invited edits.
  if (key === "model_id") return resolved.source !== "trainer";
  if (key === "infer_timesteps" && resolved.modelId === "mlp") return filled;
  const families = INFERENCE_KEY_FAMILIES[key];
  if (!families) return true;
  return families.includes(resolved.modelId) || filled;
}

/** What each port is linked to, so a missing required input is visible here. */
function connectionsSection(node, spec) {
  const nameOf = id => {
    const other = state.nodes.find(item => item.id === id);
    return other ? (BLOCK_SPECS[other.type]?.label || other.type) : id;
  };
  const rows = [
    ...spec.inputs.map(port => {
      const linked = state.edges.filter(edge => edge.toNode === node.id && edge.toPort === port.id).map(edge => nameOf(edge.fromNode));
      const missing = !linked.length && port.required;
      return `<div class="connection-row${missing ? " missing" : ""}"><i style="--port:${typeColor(port.type)}"></i><span>← ${escapeHtml(port.label)}</span><small>${linked.length ? escapeHtml(linked.join(", ")) : missing ? "required · not linked" : "optional"}</small></div>`;
    }),
    ...spec.outputs.map(port => {
      const linked = state.edges.filter(edge => edge.fromNode === node.id && edge.fromPort === port.id).map(edge => nameOf(edge.toNode));
      return `<div class="connection-row"><i style="--port:${typeColor(port.type)}"></i><span>→ ${escapeHtml(port.label)}</span><small>${linked.length ? escapeHtml(linked.join(", ")) : "not used"}</small></div>`;
    })
  ];
  return rows.length ? `<section class="inspect-section"><div class="section-title">Connections</div>${rows.join("")}</section>` : "";
}

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
  // Executable blocks: the secondary opens what the run wrote, so it says so.
  // It read "Open samples" on Inference / CAD Generator / Geometry blocks,
  // whose output is results, candidates and a converted dataset.
  return { primary: "Run", secondary: "Open results" };
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
  // The hint used to print spec.maturity ("native" / "adapter"), an internal
  // implementation grade that means nothing to the person configuring a block.
  $("#inspectorHint").textContent = spec.category;
  const resolved = node.type === "run.inference" ? inferenceModel(node) : null;
  const configEntries = spec.isModel
    ? modelInspectorEntries(node, spec.modelId, 8)
    : (node.type === "run.cad_generator" ? cadGeneratorEntries(node) : Object.entries(node.config))
      .filter(([key]) => node.type !== "source.parameters" || !["condition_names", "feature_names", "parameter_table", "parameter_dataset"].includes(key))
      .filter(([key]) => !inertGeometryKey(node, key))
      .filter(([key]) => !isStatementRow(node, key))
      .filter(([key]) => !resolved || inferenceKeyApplies(node, key, resolved))
      .slice(0, 20);
  const glance = blockFacts(node, () => {
    if (state.selectedNode === node.id && !document.getElementById("inspectorContent")?.contains(document.activeElement)) renderInspector();
  });
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
              opt_analysis: ["fea", "surrogate"],
              // cond_sweep (CHOICES) is not offered: its cond_values_a/b and
              // sweep_steps are Full config keys this block does not carry.
              interpolation_space: withCurrent(["slerp_noise", "lerp_latent"], node.config.interpolation_space) }
          : node.type === "evaluate.training_metrics"
            ? { y_scale: ["linear", "log"] }
            : {};
  const actions = inspectorActions(node, spec);
  $("#inspectorContent").innerHTML = `
    <section class="inspect-hero">
      <div class="inspect-meta"><span class="type-chip">${escapeHtml(node.type)}</span><span class="status"><i></i>${node.status === "idle" ? "Ready" : node.status}</span></div>
      <h2>${escapeHtml(spec.label)}</h2>
      <p>${escapeHtml(spec.description)}</p>
      <div class="inspect-actions"><button class="button primary" id="inspectorRun">${escapeHtml(actions.primary)}</button>${actions.secondary ? `<button class="button" id="inspectorSamples">${escapeHtml(actions.secondary)}</button>` : ""}</div>
    </section>
    ${glance.length ? `<section class="inspect-section"><div class="section-title">At a glance</div>${factsTable(glance, "inspect-facts")}</section>` : ""}
    <section class="inspect-section">
      <div class="section-title">${spec.isModel ? "Key settings" : "Settings"}</div>
      ${spec.isModel ? `<div class="config-summary"><span><strong>${escapeHtml(spec.modelId)}</strong><small>${Object.keys(node.config).length} settings in this block · every key in Full config${autoFillCount(node) ? ` · ${autoFillCount(node)} filled from links` : ""}</small></span><button class="button small primary" id="openFullConfig">Full config</button></div>` : ""}
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
          return `<div class="form-row"><label>${keyLabel(key)}</label><select class="field inspector-config" data-key="${escapeHtml(key)}">${modelChoices.map(choice => `<option value="${escapeHtml(choice)}"${sameChoice(value, choice) ? " selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select></div>`;
        }
        if (inspectorChoices[key]) {
          return `<div class="form-row"><label>${keyLabel(key)}</label><select class="field inspector-config" data-key="${escapeHtml(key)}">${inspectorChoices[key].map(choice => `<option value="${escapeHtml(choice)}"${sameChoice(value, choice) ? " selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select>${rowHelp(node, key)}</div>`;
        }
        if (RUN_EVIDENCE_KEYS.has(key)) {
          // What a run produced, not something to configure. Rendering it as a
          // text input invited edits that change the label without changing
          // anything real -- typing over "results samples" would relabel the
          // canvas while the results on disk stayed exactly as they were.
          return `<div class="form-row run-evidence"><label>${keyLabel(key)}<small class="inline-auto">from the last run</small></label><output class="field readonly" title="${escapeHtml(value)}">${escapeHtml(value) || "—"}</output></div>`;
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
          return `<div class="form-row run-evidence"><label>${keyLabel(key)}<small class="inline-auto">${note}</small></label><output class="field readonly" title="${escapeHtml(value)}">${escapeHtml(value) || "—"}</output></div>`;
        }
        const automatic = autoFillMeta(node, key);
        return `<div class="form-row${automatic ? " graph-autofilled" : ""}"><label>${keyLabel(key)}${automatic ? `<small class="inline-auto">auto · ${escapeHtml(automatic.sourceLabel)}</small>` : ""}</label><input class="field inspector-config" data-key="${escapeHtml(key)}" value="${escapeHtml(value)}">${rowHelp(node, key)}</div>`;
      }).join("")}</div>
    </section>
    ${node.type === "run.cad_generator" ? surrogateSection(node) : ""}
    ${node.type === "prep.geometry" && String(node.config.emit || "").includes("pointcloud") && String(node.config.mode || "").toLowerCase() === "ingest"
      // pipeline.py writes the point cloud to a sidecar next to the graph file
      // (pointcloud_output_path: "<stem>_pointcloud<ext>"). Nothing in the graph
      // named it, so a run configured for both emits produced a second dataset
      // that no downstream block and no export could see.
      ? `<section class="inspect-section"><div class="section-title">Also written</div><div class="form-row run-evidence"><label>point cloud<small class="inline-auto">sidecar file</small></label><output class="field readonly">${escapeHtml(pointCloudSidecar(node.config.output_dataset))}</output></div><p class="input-source-help">The <code>graph</code> emit writes <code>output_dataset</code>; the <code>pointcloud</code> emit writes this second file beside it. Point an Export or HDF5 Dataset block at it to use it.</p></section>`
      : ""}
    ${inputSourcePanel(node)}
    ${node.type === "source.parameters" ? parameterTableEditor(node) : ""}
    ${connectionsSection(node, spec)}
    <section class="inspect-section"><div style="display:grid;grid-template-columns:1fr 1fr;gap:6px"><button class="button" id="duplicateNode">Duplicate</button><button class="button danger" id="deleteNode">Delete block</button></div></section>
  `;
  $$(".inspector-config").forEach(control => control.addEventListener("change", () => {
    snapshot();
    node.config[control.dataset.key] = control.value;
    markManualConfigValue(node, control.dataset.key, control.value);
    // Switching a generator with an HI-MGN wired in to optimize means the
    // surrogate search: that wire has no other use.
    const ranks = node.type === "run.cad_generator" && control.dataset.key === "mode"
      && cadGeneratorMode(node) === "optimize" && cadSurrogate(node) && !cadUsesSurrogate(node);
    if (ranks) node.config.opt_analysis = "surrogate";
    applyGraphAutofill();
    renderInspector();
    // The canvas card shows the configured mode (title verb and kind line) and
    // autofill may have rewritten other blocks' values, so redraw it too --
    // editing `mode` here used to leave the card claiming the old one.
    render();
    toast(ranks ? "Updated mode. The wired HI-MGN now ranks the designs (opt analysis: surrogate)." : `Updated ${control.dataset.key}.`);
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
