import { $, $$, escapeHtml, toast, formatBytes } from "./dom.js";
import { state, snapshot } from "./state.js";
import { BLOCK_SPECS, MODEL_CATALOG, TYPE_META, INPUT_SOURCE_META } from "./constants.js";
import { apiRequest, requireRuntime } from "./api.js";
import { previewGraphic, nodeVisualLabel } from "./graphics.js";
import { typeColor } from "./validate.js";
import { duplicateNode, deleteSelected, nodeEvidenceLabel } from "./graph.js";
import { openConfig } from "./config.js";
import { openArtifact } from "./viewer.js";
import { runGraph } from "./run.js";
import { openStudio, openModelDetailWorkspace, openTrainingMetricsWorkspace, renderStudio, liveShell, liveError } from "./studio.js";
import { applyGraphAutofill, autoFillCount, autoFillMeta, markManualConfigValue } from "./autofill.js";

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
    </div>
    <p class="parameter-editor-help">${escapeHtml(datasetPath)}</p>
  </section>`;
}

export function embeddedInspector(node, spec) {
  if (spec.modelId === "simulgenvae") {
    return `<div class="inspect-section">
      <div class="section-title">SimulGen-VAE live contract</div>
      <div class="inspect-card-list">
        ${inspectorCard("Sequential pipeline", "Native", "train executes the hierarchical VAE stage and then the latent-conditioner stage; compatible completed stages may be reused.")}
        ${inspectorCard("VAE stage", "Native", "Compress fixed-geometry field tensors into a main latent code plus per-level hierarchical latent codes.")}
        ${inspectorCard("Latent conditioner", "Native", "Map ordered CSV parameter rows or condition images into the VAE latent representation.")}
        ${inspectorCard("Reconstruction", "Native", "Load both checkpoints, generate fields from conditions, write reconstructions.h5, and report field MSE.")}
        ${inspectorCard("Dataset gate", "Required", "The current loader requires uniform node count N and timestep count T across samples.", "adapter")}
      </div>
    </div>`;
  }
  if (spec.isModel) {
    return `<div class="inspect-section">
      <div class="section-title">Model-owned workspace</div>
      <div class="inspect-card-list">
        ${inspectorCard("Data mapping", "Inside model", "Choose named inputs, targets, timesteps, units, and compatible condition bindings.")}
        ${inspectorCard("Automatic preflight", "Before run", "Graph, config, route, paths, environment, dataset, checkpoint, native dry-run, and command checks.")}
        ${inspectorCard("Resources and VRAM", "Live estimate", "GPU selection, precision, batch feasibility, measured peak allocation, throughput, and low-memory controls.")}
        ${inspectorCard("Training outputs", ".pth ready", "Loss curves, validation samples, checkpoints, resume state, and model download.")}
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
      ${inspectorCard("Objectives and constraints", "Native", "Numeric columns, min/max direction, hard inequalities, and evaluator response mapping.")}
      ${inspectorCard("Pareto and diversity", "Native", "Feasible non-dominated set and crowding-distance top-k; scalarization remains optional.")}
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
    <p class="input-source-help">The selected path is stored on this source block and follows its links into model preflight and execution.</p>
  </section>`;
}

export async function openInputPicker(nodeId) {
  const node = state.nodes.find(item => item.id === nodeId);
  const meta = node && INPUT_SOURCE_META[node.type];
  if (!node || !meta || !requireRuntime()) return;
  state.studioSection = "data";
  $("#studioOverlay").classList.add("open");
  renderStudio();
  const container = liveShell(`Select ${meta.label}`, "Choose a real repository file. The path will be written to the selected source block.");
  try {
    const result = await apiRequest(`/api/files?kind=${encodeURIComponent(meta.kind)}`);
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
        $("#studioOverlay").classList.remove("open");
        renderInspector();
        toast(`${meta.label} selected: ${button.dataset.useInput}`);
      }));
    };
    renderFiles("");
    $("#inputPickerSearch").addEventListener("input", event => renderFiles(event.target.value));
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
    snapshot();
    node.config[meta.key] = result.path;
    renderInspector();
    toast(`Uploaded and selected ${file.name} (${formatBytes(result.size)}).`);
  } catch (error) {
    toast(`Upload failed: ${error.message}`, "error");
    button.disabled = false;
    button.textContent = original;
  }
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
    $("#deleteConnection").addEventListener("click", deleteSelected);
    return;
  }
  if (!node) {
    $("#inspectorHint").textContent = "select a block";
    $("#inspectorContent").innerHTML = `<div class="inspect-empty"><div><span>⌁</span><strong>Select a pipeline block</strong><p>Configure it, inspect typed ports, link it to other blocks, run dependencies, or open individual samples.</p></div></div>`;
    return;
  }
  const spec = BLOCK_SPECS[node.type];
  $("#inspectorHint").textContent = spec.maturity;
  const configEntries = Object.entries(node.config)
    .filter(([key]) => node.type !== "source.parameters" || !["condition_names", "feature_names", "parameter_table", "parameter_dataset"].includes(key))
    .slice(0, spec.isModel ? 6 : 20);
  const inspectorChoices = node.type === "prep.geometry"
    ? {
        mode: ["inspect", "ingest"],
        reader: ["auto", "trimesh", "gmsh"],
        mesh_type: ["surface", "volume"],
        emit: ["graph", "pointcloud", "graph, pointcloud"],
        resample_method: ["fps", "random"]
      }
    : {};
  const ports = [
    ...spec.inputs.map(port => ({ ...port, direction: "in" })),
    ...spec.outputs.map(port => ({ ...port, direction: "out" }))
  ];
  $("#inspectorContent").innerHTML = `
    <section class="inspect-hero">
      <div class="inspect-meta"><span class="type-chip">${escapeHtml(node.type)}</span><span class="status"><i></i>${node.status === "idle" ? "Ready" : node.status}</span></div>
      <h2>${escapeHtml(spec.label)}</h2>
      <p>${escapeHtml(spec.description)}</p>
      <div class="inspect-actions"><button class="button primary" id="inspectorRun">▶ ${spec.isModel ? "Start / resume" : spec.isMetricsViewer ? "Open metrics" : "Run selected"}</button><button class="button" id="inspectorSamples">${spec.isModel ? "Model details" : spec.isMetricsViewer ? "Metric plots" : node.type === "source.parameters" ? "▦ Spreadsheet" : "⌾ Samples"}</button></div>
    </section>
    <section class="inspect-section">
      <div class="section-title">${spec.isModel ? "ML configuration" : "Configuration"}</div>
      ${spec.isModel ? `<div class="config-summary"><span><strong>${MODEL_CATALOG[spec.modelId].keys.length} live keys</strong><small>${MODEL_CATALOG[spec.modelId].modes.length} modes · ${escapeHtml(MODEL_CATALOG[spec.modelId].dataset)}${autoFillCount(node) ? ` · ${autoFillCount(node)} graph-filled` : ""}</small></span><button class="button small primary" id="openFullConfig">Full config</button></div>` : ""}
      <div style="margin-top:${spec.isModel ? 9 : 0}px">${configEntries.map(([key, value]) => {
        if (spec.isModel && key === "mode") {
          return `<div class="form-row"><label>${escapeHtml(key.replaceAll("_", " "))}</label><select class="field inspector-config" data-key="${key}">${MODEL_CATALOG[spec.modelId].modes.map(mode => `<option value="${mode}"${String(value) === mode ? " selected" : ""}>${mode}</option>`).join("")}</select></div>`;
        }
        if (inspectorChoices[key]) {
          return `<div class="form-row"><label>${escapeHtml(key.replaceAll("_", " "))}</label><select class="field inspector-config" data-key="${key}">${inspectorChoices[key].map(choice => `<option value="${escapeHtml(choice)}"${String(value) === choice ? " selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select></div>`;
        }
        const automatic = autoFillMeta(node, key);
        return `<div class="form-row${automatic ? " graph-autofilled" : ""}"><label>${escapeHtml(key.replaceAll("_", " "))}${automatic ? `<small class="inline-auto">auto · ${escapeHtml(automatic.sourceLabel)}</small>` : ""}</label><input class="field inspector-config" data-key="${key}" value="${escapeHtml(value)}"></div>`;
      }).join("")}</div>
    </section>
    ${inputSourcePanel(node)}
    ${embeddedInspector(node, spec)}
    ${node.type === "source.parameters" ? parameterTableEditor(node) : ""}
    <section class="inspect-section"><div class="section-title">Typed ports</div><div class="port-list">
      ${ports.map(port => `<div class="port-row"><i style="--port:${typeColor(port.type)}"></i><span>${port.direction === "in" ? "←" : "→"} ${escapeHtml(port.label)}${port.required ? " *" : ""}</span><small>${escapeHtml(TYPE_META[port.type]?.label || port.type)}</small></div>`).join("")}
    </div></section>
    ${node.type === "source.parameters" ? "" : `<section class="inspect-section"><div class="section-title">${spec.workspace || spec.isMetricsViewer || spec.isModel ? "Primary workspace" : "Latest artifact"}</div><article class="artifact-strip"><div class="artifact-strip-visual" id="artifactStrip">${previewGraphic(spec.visual, node.id.length + 3)}</div><footer><span><strong>${escapeHtml(nodeEvidenceLabel(node, spec))}</strong><small>${escapeHtml(spec.workspace ? `${spec.workspace} evidence + controls` : nodeVisualLabel(spec))}</small></span><button class="button small" id="artifactMini">Open</button></footer></article></section>`}
    <section class="inspect-section"><div style="display:grid;grid-template-columns:1fr 1fr;gap:6px"><button class="button" id="duplicateNode">Duplicate</button><button class="button danger" id="deleteNode">Delete block</button></div></section>
  `;
  $$(".inspector-config").forEach(control => control.addEventListener("change", () => {
    snapshot();
    node.config[control.dataset.key] = control.value;
    markManualConfigValue(node, control.dataset.key, control.value);
    applyGraphAutofill();
    renderInspector();
    toast(`Updated ${control.dataset.key}.`);
  }));
  $("#openParameterSpreadsheet")?.addEventListener("click", () => openArtifact(node.id));
  $("#inspectorRun").addEventListener("click", () => {
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
  $("#inspectorSamples").addEventListener("click", openPrimaryDetails);
  $("#artifactStrip")?.addEventListener("click", openPrimaryDetails);
  $("#artifactMini")?.addEventListener("click", openPrimaryDetails);
  $("#duplicateNode").addEventListener("click", () => duplicateNode(node.id));
  $("#deleteNode").addEventListener("click", deleteSelected);
  $("#openFullConfig")?.addEventListener("click", () => openConfig(node.id));
  $("#browseInputSource")?.addEventListener("click", () => openInputPicker(node.id));
  $("#uploadInputSource")?.addEventListener("click", () => $("#inputSourceFile").click());
  $("#inputSourceFile")?.addEventListener("change", event => {
    const file = event.target.files?.[0];
    if (file) uploadInputFile(node.id, file);
    event.target.value = "";
  });
}
