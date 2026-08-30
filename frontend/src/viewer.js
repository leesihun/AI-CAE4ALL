import { $, $$, escapeHtml, toast } from "./dom.js";
import { state, snapshot } from "./state.js";
import { BLOCK_SPECS, ICONS } from "./constants.js";
import { apiRequest } from "./api.js";
import { previewGraphic, parametersTableGraphic } from "./graphics.js";
import { openStudio } from "./studio.js";
import { createRenderer, createFallbackRenderer, defaultCamera, fieldColor, turboColor } from "./render3d.js";
import { schedulePipelineSave } from "./persistence.js";
import { applyGraphAutofill, selectedParameterCandidate } from "./autofill.js";

export { fieldColor };

const HDF5_EXTENSIONS = [".h5", ".hdf5"];
const GEOMETRY_EXTENSIONS = [
  ".stl", ".ply", ".obj", ".off", ".step", ".stp", ".iges", ".igs", ".brep",
  ".vtk", ".vtu", ".vtp", ".msh"
];
const PREVIEW_EXTENSIONS = [...HDF5_EXTENSIONS, ...GEOMETRY_EXTENSIONS];

let viewport = null;
let renderer = null;

// Async preview work is latest-selection-wins. Fetch cannot reliably be
// cancelled once the server starts reading a large HDF5 artifact, so each
// request captures a generation and verifies it before touching shared state
// or viewer DOM.
let artifactLoadGeneration = 0;
let sampleLoadGeneration = 0;
let datasetPickerGeneration = 0;
let playbackGeneration = 0;

function beginArtifactLoad() {
  sampleLoadGeneration += 1;
  artifactLoadGeneration += 1;
  return artifactLoadGeneration;
}

function isCurrentArtifactLoad(generation) {
  return generation === artifactLoadGeneration;
}

function isCurrentSampleLoad(generation, artifact, sampleIndex) {
  return generation === sampleLoadGeneration
    && state.realArtifact === artifact
    && state.artifactSample === sampleIndex;
}

const PARAMETER_SHEET_MIN_ROWS = 14;
const PARAMETER_CATALOG_LIMIT = 10000;
const PARAMETER_TABLE_VERSION = 2;

function commaNames(node, key) {
  return String(node?.config?.[key] || "")
    .split(",")
    .map(name => name.trim())
    .filter(Boolean);
}

function parameterColumnId(kind, columns) {
  let index = 1;
  while (columns.some(column => column.id === `${kind}_${index}`)) index += 1;
  return `${kind}_${index}`;
}

function addParameterColumn(columns, kind, name = "") {
  const id = parameterColumnId(kind, columns);
  columns.push({ id, kind, name: name || `${kind === "input" ? "Input" : "Output"} ${id.split("_").pop()}` });
  return id;
}

function parameterContext(node) {
  const outgoing = state.edges
    .filter(edge => edge.fromNode === node.id)
    .map(edge => state.nodes.find(candidate => candidate.id === edge.toNode))
    .filter(Boolean);
  let modelNode = outgoing.find(candidate => BLOCK_SPECS[candidate.type]?.isModel);
  let datasetNode = outgoing.find(candidate => candidate.type === "source.hdf5");
  if (!modelNode && datasetNode) {
    modelNode = state.edges
      .filter(edge => edge.fromNode === datasetNode.id)
      .map(edge => state.nodes.find(candidate => candidate.id === edge.toNode))
      .find(candidate => candidate && BLOCK_SPECS[candidate.type]?.isModel);
  }
  if (!datasetNode && modelNode) {
    datasetNode = state.edges
      .filter(edge => edge.toNode === modelNode.id)
      .map(edge => state.nodes.find(candidate => candidate.id === edge.fromNode))
      .find(candidate => candidate?.type === "source.hdf5");
  }
  const modelSpec = modelNode && BLOCK_SPECS[modelNode.type];
  const modelId = modelSpec?.modelId || "conditions";
  return {
    modelId,
    modelLabel: modelSpec?.label || "Condition",
    paired: modelId === "mlp",
    generative: outgoing.some(candidate => candidate.type === "run.cad_generator"),
    datasetPath: String(node.config.parameter_dataset || datasetNode?.config?.path || (/\.(h5|hdf5)$/i.test(node.config.binding || "") ? node.config.binding : ""))
  };
}

function storedParameterTable(node) {
  try {
    const parsed = JSON.parse(node?.config?.parameter_table || "null");
    if (!parsed || !Array.isArray(parsed.columns) || !Array.isArray(parsed.rows)) return null;
    return {
      version: PARAMETER_TABLE_VERSION,
      dataset_path: String(Object.hasOwn(parsed, "dataset_path") ? parsed.dataset_path : node.config.parameter_dataset || ""),
      selected_sample_id: String(parsed.selected_sample_id || ""),
      total_samples: Number(parsed.total_samples) || parsed.rows.length,
      truncated: Boolean(parsed.truncated),
      columns: parsed.columns
        .filter(column => column && ["input", "output"].includes(column.kind))
        .map(column => ({ id: String(column.id), kind: column.kind, name: String(column.name || column.id) })),
      rows: parsed.rows.map((row, index) => ({
        sample_id: String(row?.sample_id ?? index),
        sample_label: String(row?.sample_label || `Dataset row ${index + 1}`),
        values: row?.values && typeof row.values === "object"
          ? Object.fromEntries(Object.entries(row.values).map(([key, value]) => [key, String(value ?? "")]))
          : {}
      }))
    };
  } catch {
    return null;
  }
}

function initialParameterColumns(node, context) {
  const columns = [];
  commaNames(node, "condition_names").forEach(name => addParameterColumn(columns, "input", name));
  if (context.paired) commaNames(node, "feature_names").forEach(name => addParameterColumn(columns, "output", name));
  if (!columns.some(column => column.kind === "input")) addParameterColumn(columns, "input");
  if (context.paired && !columns.some(column => column.kind === "output")) addParameterColumn(columns, "output");
  return columns;
}

function blankParameterRows(count = PARAMETER_SHEET_MIN_ROWS) {
  return Array.from({ length: count }, (unused, index) => ({
    sample_id: `pending:${index}`,
    sample_label: `Dataset row ${index + 1}`,
    values: {}
  }));
}

function parameterTableFor(node, catalog = null) {
  const context = parameterContext(node);
  const stored = storedParameterTable(node);
  const table = stored || {
    version: PARAMETER_TABLE_VERSION,
    profile: context.paired ? "mlp_paired" : "conditions",
    dataset_path: "",
    selected_sample_id: "",
    total_samples: PARAMETER_SHEET_MIN_ROWS,
    truncated: false,
    columns: initialParameterColumns(node, context),
    rows: blankParameterRows()
  };
  table.profile = context.paired ? "mlp_paired" : "conditions";
  if (!table.columns.some(column => column.kind === "input")) addParameterColumn(table.columns, "input");
  if (context.paired && !table.columns.some(column => column.kind === "output")) addParameterColumn(table.columns, "output");
  if (!catalog?.samples) return table;

  const placeholderOnly = !table.dataset_path
    && table.columns.every(column => /^(Input|Output) \d+$/.test(column.name))
    && table.rows.every(row => Object.values(row.values).every(value => !String(value).trim()));
  if (placeholderOnly) table.columns = [];
  const catalogInputs = catalog.condition_names || [];
  const catalogOutputs = context.paired
    ? (catalog.output_names || (catalog.feature_names || []).filter(name =>
        !new Set(catalogInputs.map(input => String(input).trim().toLowerCase())).has(String(name).trim().toLowerCase())
      ))
    : [];
  const knownColumns = new Set(table.columns.map(column => `${column.kind}:${column.name.trim().toLowerCase()}`));
  const catalogColumns = [["input", catalogInputs]];
  if (context.paired) catalogColumns.push(["output", catalogOutputs]);
  for (const [kind, names] of catalogColumns) {
    for (const name of names) {
      const key = `${kind}:${String(name).trim().toLowerCase()}`;
      if (!knownColumns.has(key)) {
        addParameterColumn(table.columns, kind, String(name));
        knownColumns.add(key);
      }
    }
  }
  if (!table.columns.some(column => column.kind === "input")) addParameterColumn(table.columns, "input");
  if (context.paired && !table.columns.some(column => column.kind === "output")) addParameterColumn(table.columns, "output");

  const sameDataset = Boolean(table.dataset_path && table.dataset_path === catalog.path);
  const previousById = new Map(table.rows.map(row => [row.sample_id, row]));
  table.rows = catalog.samples.map((sample, index) => {
    const previous = sameDataset ? previousById.get(String(sample.id)) : !table.dataset_path ? table.rows[index] : null;
    const values = previous?.values ? { ...previous.values } : {};
    const declaredColumns = [
      ...catalogInputs.map(name => ({ kind: "input", name })),
      ...catalogOutputs.map(name => ({ kind: "output", name }))
    ];
    (sample.parameter_values || []).slice(0, declaredColumns.length).forEach((value, valueIndex) => {
      const declared = declaredColumns[valueIndex];
      const column = table.columns.find(candidate =>
        candidate.kind === declared.kind
        && candidate.name.trim().toLowerCase() === String(declared.name).trim().toLowerCase()
      );
      if (column && values[column.id] === undefined) values[column.id] = value == null ? "" : String(value);
    });
    return {
      sample_id: String(sample.id),
      sample_label: String(sample.label || `Dataset sample ${index + 1}`),
      values
    };
  });
  table.dataset_path = String(catalog.path || node.config.binding || "");
  table.total_samples = Number(catalog.total_samples) || table.rows.length;
  table.truncated = Boolean(catalog.truncated);
  if (!table.rows.some(row => row.sample_id === table.selected_sample_id)) table.selected_sample_id = "";
  return table;
}

function saveParameterTable(node, table) {
  node.config.parameter_table = JSON.stringify(table);
  const inputs = table.columns.filter(column => column.kind === "input").map(column => column.name.trim()).filter(Boolean);
  const outputs = table.columns.filter(column => column.kind === "output").map(column => column.name.trim()).filter(Boolean);
  if (inputs.length) node.config.condition_names = inputs.join(", ");
  else delete node.config.condition_names;
  if (outputs.length) node.config.feature_names = outputs.join(", ");
  else delete node.config.feature_names;
  if (table.dataset_path) node.config.parameter_dataset = table.dataset_path;
  applyGraphAutofill();
  refreshParameterCardPreview(node);
  refreshParameterSheetInfo(node, table);
  schedulePipelineSave();
}

function parameterSheetRow(table, row, index, selectable) {
  const selected = table.selected_sample_id === row.sample_id;
  return `<tr data-parameter-row="${index}" data-sample-id="${escapeHtml(row.sample_id)}"${selected ? ` class="generation-selected"` : ""}>
    <th scope="row">${index + 1}</th>
    <td class="parameter-sample-cell"><strong>${escapeHtml(row.sample_label)}</strong><small>ID: ${escapeHtml(row.sample_id)}</small>${selectable ? `<label class="parameter-generation-choice"><input type="radio" name="parameter-generation-candidate" data-select-parameter-row="${index}"${selected ? " checked" : ""}> Use for generation</label>` : ""}</td>
    ${table.columns.map(column => `<td><input class="parameter-sheet-value" data-column-id="${escapeHtml(column.id)}" data-row="${index}" value="${escapeHtml(row.values[column.id] || "")}" aria-label="${escapeHtml(column.name)}, dataset row ${index + 1}" autocomplete="off" spellcheck="false"></td>`).join("")}
  </tr>`;
}

function refreshParameterCardPreview(node) {
  const preview = $(`[data-preview="${node.id}"] .parameters-table`);
  if (preview) preview.outerHTML = parametersTableGraphic(node, true);
}

function refreshParameterSheetInfo(node, table = storedParameterTable(node) || parameterTableFor(node)) {
  const context = parameterContext(node);
  const inputs = table.columns.filter(column => column.kind === "input");
  const outputs = table.columns.filter(column => column.kind === "output");
  const source = table.dataset_path || node.config.parameter_dataset || parameterContext(node).datasetPath || "No HDF5 dataset selected";
  const selected = selectedParameterCandidate(node);
  const info = $("#parameterSheetInfo");
  if (!info) return;
  info.innerHTML = `<section class="info-block">
    <h3>${context.paired ? "MLP paired dataset" : `${escapeHtml(context.modelLabel)} conditions`}</h3>
    <p>${context.paired ? "Each row is one MLP training pair: inputs and target outputs share the same HDF5 sample." : "Each row supplies conditions for the HDF5 sample at the same position; outputs remain owned by the dataset/model."}</p>
    <div class="stat-grid">
      <div class="stat-card"><strong>${table.rows.length}${table.total_samples > table.rows.length ? ` / ${table.total_samples}` : ""}</strong><small>matched rows</small></div>
      <div class="stat-card"><strong>${table.columns.length}</strong><small>editable columns</small></div>
      <div class="stat-card"><strong>${inputs.length}</strong><small>input columns</small></div>
      <div class="stat-card"><strong>${outputs.length}</strong><small>output columns</small></div>
    </div>
    ${table.truncated ? `<p class="parameter-sheet-warning">Showing the first ${table.rows.length} of ${table.total_samples} samples.</p>` : ""}
    ${context.generative ? selected.ready
      ? `<p class="parameter-sheet-selection"><strong>Generation candidate:</strong> ${escapeHtml(selected.row.sample_label)} (ID: ${escapeHtml(selected.selectedSampleId)})<br><code>${escapeHtml(selected.conditionNames)} → ${escapeHtml(selected.condValues)}</code></p>`
      : `<p class="parameter-sheet-warning"><strong>No runnable generation candidate.</strong> Choose one row and give every Input column a unique name and finite numeric value.</p>`
      : ""}
  </section>
  <section class="info-block">
    <div class="section-title">Dataset source</div>
    <p class="parameter-sheet-source">${escapeHtml(source)}</p>
    <p>Use <strong>+ Add dataset</strong> to lock rows to an HDF5 dataset.${context.paired ? " Add Input or Output columns as needed." : " Add condition Input columns as needed."}</p>
  </section>`;
}

function bindParameterSheetCells(node, table) {
  $$(".parameter-sheet-value").forEach(control => {
    control.addEventListener("focus", () => {
      if (control.dataset.historyCaptured === "true") return;
      snapshot();
      control.dataset.historyCaptured = "true";
    });
    control.addEventListener("input", () => {
      const row = table.rows[Number(control.dataset.row)];
      if (!row) return;
      row.values[control.dataset.columnId] = control.value;
      saveParameterTable(node, table);
    });
    control.addEventListener("keydown", event => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      const next = $(`.parameter-sheet-value[data-row="${Number(control.dataset.row) + 1}"][data-column-id="${control.dataset.columnId}"]`);
      if (next) next.focus();
    });
  });
  $$(".parameter-column-name").forEach(control => {
    control.addEventListener("focus", () => {
      if (control.dataset.historyCaptured === "true") return;
      snapshot();
      control.dataset.historyCaptured = "true";
    });
    control.addEventListener("input", () => {
      const column = table.columns.find(item => item.id === control.dataset.columnId);
      if (!column) return;
      column.name = control.value;
      saveParameterTable(node, table);
    });
  });
  $$("[data-remove-parameter-column]").forEach(button => button.addEventListener("click", () => {
    snapshot();
    const columnId = button.dataset.removeParameterColumn;
    table.columns = table.columns.filter(column => column.id !== columnId);
    table.rows.forEach(row => delete row.values[columnId]);
    saveParameterTable(node, table);
    renderParameterSpreadsheet(node, state.realArtifact, table);
  }));
  $$("[data-select-parameter-row]").forEach(control => control.addEventListener("change", () => {
    const row = table.rows[Number(control.dataset.selectParameterRow)];
    if (!row || !control.checked) return;
    snapshot();
    table.selected_sample_id = row.sample_id;
    saveParameterTable(node, table);
    renderParameterSpreadsheet(node, state.realArtifact, table);
    toast(`Generation candidate set to ${row.sample_label} (ID: ${row.sample_id}).`);
  }));
}

function setParameterSheetMode(enabled) {
  $(".artifact-body").classList.toggle("parameter-sheet-mode", enabled);
  $("#artifactCompare").hidden = enabled;
  $("#artifactDownload").hidden = enabled;
  $("#artifactCopyId").hidden = enabled;
  $("#artifactUseInPipeline").hidden = enabled;
  $("#artifactDatasetFile").accept = enabled ? ".h5,.hdf5" : PREVIEW_EXTENSIONS.join(",");
  const footerCopy = $(".artifact-shell > .modal-foot > span:first-child");
  if (footerCopy) {
    footerCopy.textContent = enabled
      ? "Rows stay locked to HDF5 order; the explicitly selected generation row is materialized as native cond_values in Input-column order."
      : "Geometry, topology, samples, and field values are read from the configured repository artifact through the local Studio API.";
  }
}

function renderParameterSpreadsheet(node, catalog = null, providedTable = null) {
  sampleLoadGeneration += 1;
  stopViewerPlayback();
  closeArtifactDatasetPicker();
  state.artifactNode = node.id;
  state.artifactSample = null;
  state.realArtifact = catalog?.samples ? { ...catalog, node, currentSample: null } : null;
  renderer?.dispose();
  renderer = null;
  viewport = null;
  setParameterSheetMode(true);
  $("#artifactAddDataset").hidden = false;
  $("#artifactIcon").textContent = ICONS[BLOCK_SPECS[node.type].icon];
  $("#artifactIcon").style.color = BLOCK_SPECS[node.type].accent;
  $("#artifactOverlay").classList.add("open");
  $("#sampleList").innerHTML = "";
  $("#sampleInfo").innerHTML = `<div id="parameterSheetInfo"></div>`;

  const table = providedTable || parameterTableFor(node, catalog);
  if (node.config.parameter_table !== JSON.stringify(table)) saveParameterTable(node, table);
  const context = parameterContext(node);
  $("#artifactTitle").textContent = context.paired ? "Design Parameters · MLP paired dataset" : `Design Parameters · ${context.modelLabel} conditions`;
  $("#artifactSubtitle").textContent = context.paired
    ? "Each row pairs the inputs and target outputs for the HDF5 sample at the same position"
    : "Each row maps condition values to the HDF5 sample at the same position";
  $("#viewerVisual").innerHTML = `<section class="parameter-sheet${context.generative ? " generative" : ""}" aria-label="Design Parameters spreadsheet">
    <header class="parameter-sheet-head">
      <span><strong>${context.paired ? "MLP input / output pairs" : `${escapeHtml(context.modelLabel)} per-sample conditions`}</strong><small>${table.dataset_path ? `${table.rows.length} rows locked to HDF5 order` : "Add an HDF5 dataset to lock row order"}</small></span>
      <span class="parameter-sheet-actions"><button class="button small" data-add-parameter-column="input" type="button">+ Input column</button>${context.paired ? `<button class="button small" data-add-parameter-column="output" type="button">+ Output column</button>` : ""}</span>
    </header>
    <div class="parameter-sheet-table-wrap">
      <table class="parameter-sheet-table" aria-label="Dataset-aligned Design Parameters spreadsheet">
        <thead><tr><th aria-label="Row number">#</th><th class="parameter-sample-heading">Dataset sample</th>${table.columns.map(column => `<th class="parameter-column-heading" data-column-heading="${escapeHtml(column.id)}"><div><span class="parameter-column-kind ${column.kind}">${column.kind}</span><input class="parameter-column-name" data-column-id="${escapeHtml(column.id)}" value="${escapeHtml(column.name)}" aria-label="Rename ${escapeHtml(column.kind)} column ${escapeHtml(column.name)}" autocomplete="off" spellcheck="false"><button type="button" data-remove-parameter-column="${escapeHtml(column.id)}" aria-label="Remove ${escapeHtml(column.name)} column">×</button></div></th>`).join("")}</tr></thead>
        <tbody id="parameterSheetRows">${table.rows.map((row, index) => parameterSheetRow(table, row, index, context.generative)).join("")}</tbody>
      </table>
    </div>
  </section>`;
  bindParameterSheetCells(node, table);
  $$('[data-add-parameter-column]').forEach(button => button.addEventListener("click", () => {
    snapshot();
    const columnId = addParameterColumn(table.columns, button.dataset.addParameterColumn);
    saveParameterTable(node, table);
    renderParameterSpreadsheet(node, state.realArtifact, table);
    $(`[data-column-heading="${columnId}"] .parameter-column-name`)?.focus();
  }));
  refreshParameterSheetInfo(node, table);
}

/** Build the persistent canvas + HUD once; a WebGL context must not churn. */
function ensureViewport() {
  const host = $("#viewerVisual");
  if (viewport && host.contains(viewport.canvas)) return viewport;
  host.innerHTML = `
    <canvas class="viewer-canvas" aria-label="Sample geometry viewport"></canvas>
    <div class="viewer-hud">
      <div class="viewer-hud-title"></div>
      <div class="viewer-hud-bottom">
        <div class="viewer-hud-legend" hidden>
          <span class="viewer-hud-legend-low"></span>
          <span class="viewer-hud-legend-bar"></span>
          <span class="viewer-hud-legend-high"></span>
        </div>
        <div class="viewer-hud-foot"></div>
      </div>
    </div>
    <div class="viewer-message live-empty" hidden></div>`;
  viewport = {
    canvas: host.querySelector(".viewer-canvas"),
    hud: host.querySelector(".viewer-hud"),
    title: host.querySelector(".viewer-hud-title"),
    legend: host.querySelector(".viewer-hud-legend"),
    legendLow: host.querySelector(".viewer-hud-legend-low"),
    legendBar: host.querySelector(".viewer-hud-legend-bar"),
    legendHigh: host.querySelector(".viewer-hud-legend-high"),
    foot: host.querySelector(".viewer-hud-foot"),
    message: host.querySelector(".viewer-message")
  };
  renderer?.dispose();
  renderer = createRenderer(viewport.canvas) || createFallbackRenderer(viewport.canvas);
  return viewport;
}

export function showViewerMessage(text) {
  const view = ensureViewport();
  view.message.textContent = text;
  view.message.hidden = false;
  view.canvas.hidden = true;
  view.hud.hidden = true;
}

function formatValue(value) {
  return Number.isFinite(value) ? Number(value).toPrecision(4) : "—";
}

function topologyLabel(sample) {
  const mesh = sample.mesh;
  if (!mesh) return "point cloud";
  const parts = [];
  if (mesh.returned_elements) {
    parts.push(`${mesh.returned_elements.toLocaleString()} / ${Number(mesh.total_elements || mesh.returned_elements).toLocaleString()} elements`);
  }
  if (mesh.returned_edges) {
    parts.push(`${mesh.returned_edges.toLocaleString()} / ${Number(mesh.total_edges).toLocaleString()} edges`);
  }
  return parts.join(" · ") || "point cloud";
}

function supportedMode(sample, mode) {
  return Boolean(sample?.supports?.[mode]);
}

function chooseViewerMode(sample, preferred = state.viewerMode) {
  if (supportedMode(sample, preferred)) return preferred;
  return ["mesh", "field", "points"].find(mode => supportedMode(sample, mode)) || "points";
}

export function renderViewerMode() {
  const sample = state.realArtifact?.currentSample;
  if (!sample) {
    $$("[data-view-mode]").forEach(button => {
      button.disabled = true;
      button.classList.remove("active");
    });
    $("#viewerModeMeta").textContent = "No sample selected";
    return;
  }
  state.viewerMode = chooseViewerMode(sample);
  $$("[data-view-mode]").forEach(button => {
    const supported = supportedMode(sample, button.dataset.viewMode);
    button.disabled = !supported;
    button.classList.toggle("active", supported && button.dataset.viewMode === state.viewerMode);
    button.title = supported ? "" : `${button.dataset.viewMode} view is not available for this sample`;
  });

  const view = ensureViewport();
  if (!renderer) {
    showViewerMessage("This browser cannot open a canvas viewport for the sample geometry.");
    return;
  }
  if (!sample.x?.length) {
    showViewerMessage("The selected sample has no plottable coordinates.");
    return;
  }
  view.message.hidden = true;
  view.canvas.hidden = false;
  view.hud.hidden = false;

  const drawn = renderer.draw(sample, state.viewerMode, state.viewerCamera);
  state.viewerDraw = { ...drawn, renderer: renderer.kind, mode: state.viewerMode };
  const showField = state.viewerMode === "field" && sample.supports?.field;
  view.title.textContent = [
    state.viewerMode.toUpperCase(),
    sample.dataset,
    sample.supports?.field ? sample.feature_name : null,
    sample.timestep_count > 1 ? `t = ${sample.timestep}` : null,
    showField && drawn.constant ? "uniform" : null
  ].filter(Boolean).join("  ·  ");
  view.legend.hidden = !showField;
  if (showField) {
    const [low, high] = drawn.domain;
    view.legendLow.textContent = formatValue(low);
    view.legendHigh.textContent = formatValue(high);
    view.legendBar.style.background = `linear-gradient(90deg, ${
      [0, 0.25, 0.5, 0.75, 1].map(stop => `${turboColor(stop)} ${stop * 100}%`).join(", ")
    })`;
  }
  view.foot.textContent = [
    `${drawn.planar ? "planar" : "3D"} · yaw ${Math.round(state.viewerCamera.yaw * 180 / Math.PI)}° · pitch ${Math.round(state.viewerCamera.pitch * 180 / Math.PI)}° · ${Math.round(state.viewerCamera.zoom * 100)}%`,
    topologyLabel(sample),
    `${sample.returned_points.toLocaleString()} / ${sample.total_points.toLocaleString()} nodes`
  ].join("  ·  ");

  $("#viewerModeMeta").textContent = state.viewerMode === "points"
    ? `${sample.returned_points.toLocaleString()} nodes`
    : drawn.drewFaces
      ? `${sample.mesh.returned_elements.toLocaleString()} ${sample.mesh.element_kind === "quad" ? "quad" : "triangular"} elements`
      : drawn.drewEdges
        ? `${sample.mesh.returned_edges.toLocaleString()} mesh edges`
        : `${sample.returned_points.toLocaleString()} nodes`;
}

export function resetViewerCamera(render = true) {
  state.viewerCamera = defaultCamera(state.realArtifact?.currentSample);
  if (render && state.realArtifact?.currentSample) renderViewerMode();
}

export function bindViewerInteractions() {
  const visual = $("#viewerVisual");
  if (!visual || visual.dataset.cameraBound === "true") return;
  visual.dataset.cameraBound = "true";

  visual.addEventListener("wheel", event => {
    if (!state.realArtifact?.currentSample) return;
    event.preventDefault();
    const factor = Math.exp(-event.deltaY * 0.001);
    state.viewerCamera.zoom = Math.max(0.2, Math.min(8, state.viewerCamera.zoom * factor));
    renderViewerMode();
  }, { passive: false });

  visual.addEventListener("pointerdown", event => {
    if (!state.realArtifact?.currentSample) return;
    const mode = event.button === 2 || event.button === 1 || (event.button === 0 && event.shiftKey)
      ? "pan"
      : event.button === 0
        ? "rotate"
        : null;
    if (!mode) return;
    event.preventDefault();
    state.viewerPointer = { id: event.pointerId, mode, x: event.clientX, y: event.clientY };
    visual.setPointerCapture?.(event.pointerId);
    visual.classList.add("camera-dragging");
  });

  visual.addEventListener("pointermove", event => {
    const pointer = state.viewerPointer;
    if (!pointer || pointer.id !== event.pointerId || !state.realArtifact?.currentSample) return;
    const dx = event.clientX - pointer.x;
    const dy = event.clientY - pointer.y;
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    if (pointer.mode === "rotate") {
      state.viewerCamera.yaw += dx * 0.008;
      state.viewerCamera.pitch = Math.max(-1.45, Math.min(1.45, state.viewerCamera.pitch + dy * 0.008));
    } else {
      state.viewerCamera.panX += dx;
      state.viewerCamera.panY += dy;
    }
    renderViewerMode();
  });

  const stopPointer = event => {
    if (!state.viewerPointer || state.viewerPointer.id !== event.pointerId) return;
    visual.releasePointerCapture?.(event.pointerId);
    state.viewerPointer = null;
    visual.classList.remove("camera-dragging");
  };
  visual.addEventListener("pointerup", stopPointer);
  visual.addEventListener("pointercancel", stopPointer);
  visual.addEventListener("contextmenu", event => event.preventDefault());
  visual.addEventListener("dblclick", () => resetViewerCamera());
  visual.addEventListener("keydown", event => {
    if (!state.realArtifact?.currentSample) return;
    const camera = state.viewerCamera;
    if (event.key === "ArrowLeft") camera.yaw -= 0.08;
    else if (event.key === "ArrowRight") camera.yaw += 0.08;
    else if (event.key === "ArrowUp") camera.pitch = Math.max(-1.45, camera.pitch - 0.08);
    else if (event.key === "ArrowDown") camera.pitch = Math.min(1.45, camera.pitch + 0.08);
    else if (event.key === "+" || event.key === "=") camera.zoom = Math.min(8, camera.zoom * 1.12);
    else if (event.key === "-" || event.key === "_") camera.zoom = Math.max(0.2, camera.zoom / 1.12);
    else if (event.key === "0") {
      resetViewerCamera();
      event.preventDefault();
      return;
    } else return;
    event.preventDefault();
    renderViewerMode();
  });
}

function renderEmptyViewer(message = "Choose a sample from the left to visualize its actual values.") {
  sampleLoadGeneration += 1;
  if (state.realArtifact) state.realArtifact.currentSample = null;
  $$("[data-view-mode]").forEach(button => {
    button.disabled = true;
    button.classList.remove("active");
  });
  $("#viewerModeMeta").textContent = "No sample selected";
  showViewerMessage(message);
  $("#sampleInfo").innerHTML = `<section class="info-block"><h3>No sample selected</h3><p>The dataset is ready. Select a sample explicitly to read and render it.</p></section>${catalogNamesBlock()}`;
  const timeline = $(".viewer-timeline input");
  timeline.min = "0";
  timeline.max = "0";
  timeline.value = "0";
  timeline.disabled = true;
  timeline.onchange = null;
  $(".viewer-timeline span").textContent = "t = 0 / 0";
  $("#viewerPlay").disabled = true;
  $("#viewerReset").disabled = true;
  $("#artifactDownload").disabled = true;
}

export function renderArtifactCatalog(query = "") {
  const artifact = state.realArtifact;
  if (!artifact) return;
  const normalizedQuery = query.trim().toLowerCase();
  const matches = artifact.samples
    .map((sample, index) => ({ sample, index }))
    .filter(({ sample }) => {
      if (!normalizedQuery) return true;
      const searchable = [
        sample.id,
        sample.label,
        ...(sample.datasets || []).flatMap(dataset => [dataset.name, ...(dataset.shape || [])])
      ].join(" ").toLowerCase();
      return searchable.includes(normalizedQuery);
    });
  $("#sampleList").innerHTML = matches.length
    ? matches.map(({ sample, index }) => {
      const datasets = sample.datasets || [];
      const primary = datasets.find(item => ["nodal_data", "surface_points"].includes(item.name)) || datasets[0];
      const label = sample.label || `sample ${sample.id}`;
      return `<button class="sample-item${index === state.artifactSample ? " active" : ""}" data-real-sample="${index}"><span class="sample-thumb">${previewGraphic(artifact.source_kind === "geometry" ? "geometry" : "dataset", index + 2)}</span><span><strong>${escapeHtml(label)}</strong><small>${primary ? `${escapeHtml(primary.name)}${primary.shape?.length ? ` [${primary.shape.join(" × ")}]` : ""}` : "sample"}</small></span></button>`;
    }).join("")
    : `<div class="live-empty">No samples match “${escapeHtml(query)}”.</div>`;
  $$("[data-real-sample]").forEach(button =>
    button.addEventListener("click", () => {
      stopViewerPlayback();
      resetViewerCamera(false);
      renderRealArtifactSample(Number(button.dataset.realSample), null, 0);
    })
  );
}

function formatStat(value) {
  return Number.isFinite(value) ? Number(value).toPrecision(5) : "—";
}

function sampleShape(sample) {
  return Array.isArray(sample.shape) && sample.shape.length ? ` [${sample.shape.join(" × ")}]` : "";
}

/** Names the dataset declares, shown before any sample has been read. */
function catalogNamesBlock() {
  const artifact = state.realArtifact;
  const features = artifact?.feature_names || [];
  const conditions = artifact?.condition_names || [];
  if (!features.length && !conditions.length) return "";
  const list = (title, names) => (names.length
    ? `<div class="section-title">${title}</div><ol class="name-list">${
      names.map(name => `<li><span>${escapeHtml(name)}</span></li>`).join("")
    }</ol>`
    : "");
  return `<section class="info-block">${list("Feature channels", features)}${list("Condition parameters", conditions)}</section>`;
}

/** Named channel picker plus the full channel list for the loaded sample. */
function featureBlock(sample) {
  const names = sample.feature_names?.length
    ? sample.feature_names
    : Array.from({ length: sample.feature_count }, (unused, index) => `feature ${index}`);
  const options = names.map((name, index) =>
    `<option value="${index}"${index === sample.feature ? " selected" : ""}>${index} · ${escapeHtml(name)}</option>`
  ).join("");
  return `<section class="info-block"><div class="section-title">Field selector</div>
    <label class="config-help" for="realFeature">Feature channel</label>
    <select class="config-control" id="realFeature"${names.length <= 1 ? " disabled" : ""}>${options}</select>
    <label class="config-help" for="realTimestep">Timestep</label>
    <input class="config-control" id="realTimestep" type="number" min="0" max="${Math.max(0, sample.timestep_count - 1)}" value="${sample.timestep}"${sample.timestep_count <= 1 ? " disabled" : ""}>
    <button class="button primary" id="loadRealField" style="width:100%;margin-top:8px">Load actual field</button>
    <div class="section-title" style="margin-top:14px">Feature channels</div>
    <ol class="name-list">${names.map((name, index) =>
      `<li${index === sample.feature ? ' class="active"' : ""}><button type="button" data-load-feature="${index}">${escapeHtml(name)}</button></li>`
    ).join("")}</ol>
  </section>`;
}

/** Per-sample scalar conditions - the parametric-study inputs of a sample. */
function parameterBlock(sample) {
  const parameters = sample.parameters || [];
  if (!parameters.length) return "";
  return `<section class="info-block"><div class="section-title">Sample parameters</div><table class="param-table"><tbody>${
    parameters.map(item =>
      `<tr><th>${escapeHtml(item.name)}</th><td>${formatStat(Number(item.value))}</td></tr>`
    ).join("")
  }</tbody></table></section>`;
}

export async function renderRealArtifactSample(sampleIndex, feature = null, timestep = 0) {
  const artifact = state.realArtifact;
  if (!artifact) return false;
  const selected = artifact.samples[sampleIndex];
  if (!selected) return false;
  const requestGeneration = ++sampleLoadGeneration;
  const changingSample = state.artifactSample !== sampleIndex;
  state.artifactSample = sampleIndex;
  if (changingSample) state.realArtifact.currentSample = null;
  const requestedFeature = feature == null ? Number(selected.default_feature || 0) : feature;
  renderArtifactCatalog($("#artifactSampleSearch")?.value || "");
  $("#artifactDownload").disabled = true;
  $("#viewerReset").disabled = true;
  showViewerMessage(`Reading actual sample ${selected.id}…`);
  try {
    const sample = await apiRequest(
      `/api/preview/sample?path=${encodeURIComponent(artifact.path)}&sample=${encodeURIComponent(selected.id)}&feature=${requestedFeature}&timestep=${timestep}${artifact.truth_path ? `&truth=${encodeURIComponent(artifact.truth_path)}` : ""}`
    );
    if (!isCurrentSampleLoad(requestGeneration, artifact, sampleIndex)) return false;
    state.realArtifact.currentSample = sample;
    // A new sample chooses its own orientation; reloading a feature or
    // timestep on the same sample must not throw the camera away.
    if (changingSample) state.viewerCamera = defaultCamera(sample);
    state.viewerMode = chooseViewerMode(sample, artifact.default_mode);
    renderViewerMode();
    $("#artifactDownload").disabled = false;
    $("#viewerReset").disabled = false;
    const stats = sample.stats || {};
    const spec = BLOCK_SPECS[artifact.node.type];
    $("#artifactTitle").textContent = `${spec.label} · actual samples`;
    $("#artifactSubtitle").textContent = `${artifact.path} · actual repository values`;
    const mesh = sample.mesh;
    const topology = mesh
      ? `<br>elements=${Number(mesh.returned_elements || 0).toLocaleString()} / ${Number(mesh.total_elements || 0).toLocaleString()} (${escapeHtml(mesh.element_kind || "none")})<br>edges=${Number(mesh.returned_edges || 0).toLocaleString()} / ${Number(mesh.total_edges || 0).toLocaleString()}`
      : "";
    $("#sampleInfo").innerHTML = `<section class="info-block"><h3>Sample ${escapeHtml(sample.sample)}</h3><p>${escapeHtml(sample.path)} · ${escapeHtml(sample.dataset)}${sampleShape(sample)}</p><div class="stat-grid">
      <span class="stat-card"><strong>${sample.supports?.field ? formatStat(stats.min) : "—"}</strong><small>${sample.supports?.field ? "actual minimum" : "field unavailable"}</small></span>
      <span class="stat-card"><strong>${sample.supports?.field ? formatStat(stats.max) : "—"}</strong><small>${sample.supports?.field ? "actual maximum" : "field unavailable"}</small></span>
      <span class="stat-card"><strong>${sample.supports?.field ? formatStat(stats.mean) : "—"}</strong><small>${sample.supports?.field ? "actual mean" : "field unavailable"}</small></span>
      <span class="stat-card"><strong>${sample.total_points.toLocaleString()}</strong><small>nodes / values</small></span>
    </div></section>
    ${sample.supports?.field
      ? featureBlock(sample)
      : `<section class="info-block"><div class="section-title">Surface geometry</div><p>This sample has geometry but no scalar field. Point mode preserves the real coordinates${sample.mesh ? "; Mesh mode also preserves the real topology" : ""}.</p></section>`}
    ${parameterBlock(sample)}
    <section class="info-block"><div class="section-title">Provenance</div><p>source=${escapeHtml(sample.path)}<br>sample=${escapeHtml(sample.sample)}<br>reader=${escapeHtml(sample.source_kind)}<br>nodes=${sample.returned_points.toLocaleString()} / ${sample.total_points.toLocaleString()}${sample.metadata?.node_reduction ? ` (${escapeHtml(sample.metadata.node_reduction)})` : ""}${topology}</p></section>`;
    $("#loadRealField")?.addEventListener("click", () =>
      renderRealArtifactSample(sampleIndex, Number($("#realFeature").value), Number($("#realTimestep").value))
    );
    // Picking a channel is a request to see it. Requiring a separate Load click
    // afterwards is a step with no decision in it, and silently leaves the view
    // showing the previous channel while the dropdown claims otherwise.
    $("#realFeature")?.addEventListener("change", () =>
      renderRealArtifactSample(sampleIndex, Number($("#realFeature").value), Number($("#realTimestep").value || sample.timestep))
    );
    $$("[data-load-feature]").forEach(button =>
      button.addEventListener("click", () =>
        renderRealArtifactSample(sampleIndex, Number(button.dataset.loadFeature), sample.timestep)
      )
    );
    const timeline = $(".viewer-timeline input");
    const playButton = $("#viewerPlay");
    timeline.max = String(Math.max(0, sample.timestep_count - 1));
    timeline.value = String(sample.timestep);
    timeline.disabled = sample.timestep_count <= 1;
    if (playButton) playButton.disabled = sample.timestep_count <= 1;
    $(".viewer-timeline span").textContent = `t = ${sample.timestep} / ${Math.max(0, sample.timestep_count - 1)}`;
    timeline.onchange = () => renderRealArtifactSample(sampleIndex, sample.feature, Number(timeline.value));
    return true;
  } catch (error) {
    if (!isCurrentSampleLoad(requestGeneration, artifact, sampleIndex)) return false;
    state.realArtifact.currentSample = null;
    $("#artifactDownload").disabled = true;
    $("#viewerReset").disabled = true;
    showViewerMessage(`Could not visualize this real sample: ${error.message}`);
    $("#sampleInfo").innerHTML = `<section class="info-block"><h3>Reader error</h3><p>${escapeHtml(error.message)}</p></section>`;
    return false;
  }
}

function normalizeConfiguredPath(value) {
  let text = String(value || "").trim().replaceAll("\\", "/");
  if (!text) return "";
  if (/^[A-Za-z]:\//.test(text)) return text;
  text = text.replace(/^\.\//, "");
  while (text.startsWith("../")) text = text.slice(3);
  const roots = ["dataset/", "output/", "outputs/", "frontend/", "configs/"];
  for (const root of roots) {
    const index = text.indexOf(root);
    if (index >= 0) return text.slice(index);
  }
  return text;
}

function hasPreviewExtension(path) {
  const lower = path.toLowerCase();
  return PREVIEW_EXTENSIONS.some(extension => lower.endsWith(extension));
}

function upstreamNode(nodeId, predicate) {
  return state.edges
    .filter(edge => edge.toNode === nodeId)
    .map(edge => state.nodes.find(node => node.id === edge.fromNode))
    .find(node => node && predicate(node));
}

function configuredPreviewPath(node, spec) {
  if (node.type === "source.cad") return normalizeConfiguredPath(node.config.path);
  if (node.type === "source.hdf5") return normalizeConfiguredPath(node.config.path);
  if (node.type === "prep.geometry") {
    const source = upstreamNode(node.id, candidate => candidate.type === "source.cad");
    return normalizeConfiguredPath(node.config.input_geometry || source?.config.path);
  }
  if (spec.isModel) {
    return normalizeConfiguredPath(node.config.dataset_dir || node.config.infer_dataset);
  }
  // A finished run records exactly where its predictions went. Two reasons this
  // value is returned raw: it is a directory of per-sample result files rather
  // than a single file, so it must skip the extension test below, and it is
  // already suite-relative -- normalizeConfiguredPath exists to strip a method
  // repo's "../" prefixes and would slice "MeshGraphNets/outputs/rollout" down
  // to "outputs/rollout" at the first known root, which resolves to nothing.
  if (node.config.results_path) return String(node.config.results_path).trim();
  const candidate = normalizeConfiguredPath(
    node.config.path
    || node.config.infer_dataset
    || node.config.inference_output_dir
    || node.config.output_dir
  );
  return hasPreviewExtension(candidate) ? candidate : "";
}

function catalogKind(node, spec) {
  if (node.type === "source.cad" || node.type === "prep.geometry" || node.type === "run.cad_generator") {
    return "geometry";
  }
  if (node.type === "source.hdf5" || spec.isModel) return "dataset";
  if (["run.inference", "evaluate.predictions", "evaluate.compare", "output.export"].includes(node.type)) {
    return "artifact";
  }
  return "";
}

/**
 * The dataset an Inference block predicted from, if the graph knows it.
 *
 * A rollout result contains only the prediction; naming its source dataset is
 * what lets the viewer show truth and error too. It is optional everywhere --
 * without it the prediction still renders on its own.
 */
function truthForNode(node) {
  if (!node || node.type !== "run.inference") return "";
  return String(node.config?.dataset_path || "").trim();
}

async function loadPreviewCatalog(path, limit = null, truth = "") {
  const limitQuery = limit == null ? "" : `&limit=${encodeURIComponent(limit)}`;
  const truthQuery = truth ? `&truth=${encodeURIComponent(truth)}` : "";
  const catalog = await apiRequest(`/api/preview/samples?path=${encodeURIComponent(path)}${limitQuery}${truthQuery}`);
  if (!catalog.samples?.length) throw new Error(`${path} contains no visualizable samples.`);
  return catalog;
}

async function resolvePreview(node, spec) {
  const configured = configuredPreviewPath(node, spec);
  if (configured) {
    try {
      return await loadPreviewCatalog(configured, null, truthForNode(node));
    } catch (error) {
      if (node.type === "source.cad" || node.type === "source.hdf5") throw error;
    }
  }

  if (node.type === "run.inference") {
    // Only ever show THIS block's own results. Falling back to the newest run in
    // the repository looked helpful and was actively misleading: the viewer
    // would render some other model's predictions under this block's name, with
    // nothing on screen saying they were unrelated.
    if (node.config.results_path) {
      throw new Error(`The recorded results for this block (${node.config.results_path}) could not be read — they may have been moved or deleted. Run the block again to regenerate them.`);
    }
    throw new Error("No results yet. Run this Inference block to predict with the connected dataset and checkpoint; its results then open here.");
  }

  const kind = catalogKind(node, spec);
  if (!kind) {
    throw new Error(`${spec.label} does not expose geometry or HDF5 samples. Use its dedicated workspace instead.`);
  }
  const files = await apiRequest(`/api/files?kind=${kind}`);
  const supported = files.items.filter(item =>
    kind === "geometry"
      ? GEOMETRY_EXTENSIONS.includes(item.extension)
      : PREVIEW_EXTENSIONS.includes(item.extension)
  );
  const preferred = supported.find(item => configured && item.path.endsWith(configured)) || supported[0];
  if (!preferred) throw new Error(`No visualizable ${kind} artifact is currently present in the repository.`);
  return loadPreviewCatalog(preferred.path);
}

function formatBytes(value) {
  const bytes = Number(value) || 0;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

async function loadArtifactPath(path, showToast = true) {
  const node = state.nodes.find(item => item.id === state.artifactNode) || state.realArtifact?.node;
  if (!node) throw new Error("The sample viewer is not attached to a pipeline block.");
  const requestGeneration = beginArtifactLoad();
  stopViewerPlayback();
  $("#artifactDatasetList").innerHTML = `<div class="live-empty">Reading ${escapeHtml(path)}…</div>`;
  let catalog;
  try {
    catalog = await loadPreviewCatalog(path, node.type === "source.parameters" ? PARAMETER_CATALOG_LIMIT : null);
  } catch (error) {
    if (!isCurrentArtifactLoad(requestGeneration)) return false;
    throw error;
  }
  if (!isCurrentArtifactLoad(requestGeneration)) return false;
  if (node.type === "source.parameters") {
    snapshot();
    const table = parameterTableFor(node, catalog);
    saveParameterTable(node, table);
    renderParameterSpreadsheet(node, catalog, table);
    if (showToast) {
      toast(`Matched ${table.rows.length} spreadsheet rows to ${catalog.path || path}.`);
    }
    return true;
  }
  state.realArtifact = { ...catalog, node, currentSample: null };
  state.artifactSample = null;
  state.viewerMode = catalog.default_mode || "field";
  resetViewerCamera(false);
  const spec = BLOCK_SPECS[node.type];
  $("#artifactTitle").textContent = `${spec.label} · repository samples`;
  $("#artifactSubtitle").textContent = `${catalog.path} · choose a sample to visualize`;
  $("#artifactSampleSearch").value = "";
  renderArtifactCatalog();
  renderEmptyViewer();
  closeArtifactDatasetPicker();
  if (showToast) toast(`Opened ${catalog.path}. Choose a sample to visualize.`);
  return true;
}

export function renderArtifactDatasetChoices(query = "") {
  const normalizedQuery = query.trim().toLowerCase();
  const choices = state.viewerDatasetChoices.filter(item =>
    !normalizedQuery
    || `${item.name} ${item.path} ${item.extension}`.toLowerCase().includes(normalizedQuery)
  );
  $("#artifactDatasetList").innerHTML = choices.length
    ? choices.map(item => {
      const sourceKind = HDF5_EXTENSIONS.includes(item.extension) ? "HDF5" : "geometry";
      return `<button class="artifact-dataset-item" data-preview-dataset="${escapeHtml(item.path)}"><span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.path)}</small></span><span><b>${sourceKind}</b><small>${formatBytes(item.size)}</small></span></button>`;
    }).join("")
    : `<div class="live-empty">No visualizable datasets match “${escapeHtml(query)}”.</div>`;
  $$("[data-preview-dataset]").forEach(button =>
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const loaded = await loadArtifactPath(button.dataset.previewDataset);
        if (!loaded) button.disabled = false;
      } catch (error) {
        button.disabled = false;
        toast(`Could not open dataset: ${error.message}`, "error");
      }
    })
  );
}

export async function openArtifactDatasetPicker() {
  if (!state.api.connected) {
    toast("Runtime is offline; datasets cannot be listed.", "error");
    return;
  }
  const picker = $("#artifactDatasetPicker");
  const requestGeneration = ++datasetPickerGeneration;
  const artifactGeneration = artifactLoadGeneration;
  const artifactNode = state.artifactNode;
  picker.hidden = false;
  $("#artifactDatasetSearch").value = "";
  $("#artifactDatasetList").innerHTML = `<div class="live-empty">Scanning repository datasets and geometry…</div>`;
  try {
    const node = state.nodes.find(item => item.id === state.artifactNode);
    const parametersOnly = node?.type === "source.parameters";
    const [datasets, geometry] = await Promise.all([
      apiRequest("/api/files?kind=dataset"),
      parametersOnly ? Promise.resolve({ items: [] }) : apiRequest("/api/files?kind=geometry")
    ]);
    if (requestGeneration !== datasetPickerGeneration
      || artifactGeneration !== artifactLoadGeneration
      || artifactNode !== state.artifactNode
      || picker.hidden) return;
    const unique = new Map();
    [...(datasets.items || []), ...(geometry.items || [])]
      .filter(item => PREVIEW_EXTENSIONS.includes(String(item.extension || "").toLowerCase()))
      .filter(item => !parametersOnly || HDF5_EXTENSIONS.includes(String(item.extension || "").toLowerCase()))
      .forEach(item => unique.set(item.path, item));
    state.viewerDatasetChoices = [...unique.values()].sort((left, right) =>
      left.path.localeCompare(right.path, undefined, { numeric: true })
    );
    renderArtifactDatasetChoices($("#artifactDatasetSearch").value);
    $("#artifactDatasetSearch").focus();
  } catch (error) {
    if (requestGeneration !== datasetPickerGeneration
      || artifactGeneration !== artifactLoadGeneration
      || artifactNode !== state.artifactNode
      || picker.hidden) return;
    $("#artifactDatasetList").innerHTML = `<div class="live-empty">Could not list datasets: ${escapeHtml(error.message)}</div>`;
  }
}

export function closeArtifactDatasetPicker() {
  datasetPickerGeneration += 1;
  $("#artifactDatasetPicker").hidden = true;
}

export async function uploadArtifactDataset(file) {
  if (!file || !state.api.connected) return;
  const lowerName = file.name.toLowerCase();
  const node = state.nodes.find(item => item.id === state.artifactNode);
  if (node?.type === "source.parameters" && !HDF5_EXTENSIONS.some(extension => lowerName.endsWith(extension))) {
    toast("Design Parameters can import its input/output names from an HDF5 dataset.", "warn");
    return;
  }
  if (!hasPreviewExtension(lowerName)) {
    toast("Choose an HDF5, CAD, mesh, or VTK file.", "warn");
    return;
  }
  const kind = HDF5_EXTENSIONS.some(extension => lowerName.endsWith(extension)) ? "dataset" : "geometry";
  const button = $("#artifactUploadDataset");
  const original = button.textContent;
  const uploadArtifactGeneration = artifactLoadGeneration;
  const uploadArtifactNode = state.artifactNode;
  let loadingUploadedArtifact = false;
  button.disabled = true;
  button.textContent = `Uploading ${file.name}…`;
  try {
    const response = await fetch(`/api/upload?kind=${encodeURIComponent(kind)}`, {
      method: "POST",
      headers: { "X-Filename": encodeURIComponent(file.name) },
      body: file
    });
    const result = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    if (uploadArtifactGeneration !== artifactLoadGeneration || uploadArtifactNode !== state.artifactNode) return;
    loadingUploadedArtifact = true;
    const loaded = await loadArtifactPath(result.path, false);
    if (loaded) toast(`Uploaded ${file.name} (${formatBytes(result.size)}). Choose a sample to visualize.`);
  } catch (error) {
    if (loadingUploadedArtifact
      || (uploadArtifactGeneration === artifactLoadGeneration && uploadArtifactNode === state.artifactNode)) {
      toast(`Upload failed: ${error.message}`, "error");
    }
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

export async function openArtifact(nodeId) {
  const node = state.nodes.find(item => item.id === nodeId);
  if (!node) return;
  const spec = BLOCK_SPECS[node.type];
  const requestGeneration = beginArtifactLoad();
  stopViewerPlayback();
  if (spec.isModel) {
    const { openModelDetailWorkspace } = await import("./studio.js");
    if (!isCurrentArtifactLoad(requestGeneration)) return;
    await openModelDetailWorkspace(spec.modelId);
    return;
  }
  if (node.type === "source.parameters") {
    renderParameterSpreadsheet(node);
    const datasetPath = parameterContext(node).datasetPath;
    if (state.api.connected && datasetPath) {
      try {
        const catalog = await loadPreviewCatalog(datasetPath, PARAMETER_CATALOG_LIMIT);
        if (!isCurrentArtifactLoad(requestGeneration)) return;
        const table = parameterTableFor(node, catalog);
        saveParameterTable(node, table);
        renderParameterSpreadsheet(node, catalog, table);
      } catch (error) {
        if (!isCurrentArtifactLoad(requestGeneration)) return;
        toast(`Could not match Design Parameters to ${datasetPath}: ${error.message}`, "error");
      }
    }
    return;
  }
  if (!state.api.connected) {
    toast("Runtime is offline; actual samples cannot be read. Start with START_STUDIO.bat.", "error");
    return;
  }
  closeArtifactDatasetPicker();
  setParameterSheetMode(false);
  state.artifactNode = nodeId;
  state.artifactSample = null;
  state.viewerMode = "field";
  state.realArtifact = null;
  resetViewerCamera(false);
  $("#artifactIcon").textContent = ICONS[spec.icon];
  $("#artifactIcon").style.color = spec.accent;
  // A model has exactly one configured dataset_dir/infer_dataset, not a
  // catalog to browse and add to — showing "+ Add dataset" here let a user
  // pick a file that never actually got wired to the model.
  $("#artifactAddDataset").hidden = Boolean(spec.isModel);
  $("#artifactAddDataset").title = spec.isModel
    ? "Set this model's dataset_dir/infer_dataset in its configuration instead."
    : "";
  $("#artifactTitle").textContent = `${spec.label} · repository samples`;
  $("#artifactSubtitle").textContent = "Resolving the configured artifact…";
  $("#artifactOverlay").classList.add("open");
  $("#sampleList").innerHTML = `<div class="live-empty">Scanning the configured source…</div>`;
  showViewerMessage("Waiting for an actual sample…");
  $("#sampleInfo").innerHTML = "";
  $("#artifactSampleSearch").value = "";
  try {
    const catalog = await resolvePreview(node, spec);
    if (!isCurrentArtifactLoad(requestGeneration)) return;
    state.realArtifact = { ...catalog, node, currentSample: null };
    state.viewerMode = catalog.default_mode || "field";
    $("#artifactTitle").textContent = `${spec.label} · repository samples`;
    $("#artifactSubtitle").textContent = `${catalog.path} · choose a sample to visualize`;
    renderArtifactCatalog();
    renderEmptyViewer();
  } catch (error) {
    if (!isCurrentArtifactLoad(requestGeneration)) return;
    $("#sampleList").innerHTML = "";
    showViewerMessage(error.message);
    $("#sampleInfo").innerHTML = `<section class="info-block"><h3>No actual preview available</h3><p>${escapeHtml(error.message)}</p></section>`;
    toast(`Actual sample viewer: ${error.message}`, "error");
  }
}

function scheduleViewerPlayback(generation) {
  if (!state.viewerPlaying || generation !== playbackGeneration) return;
  state.viewerPlayTimer = window.setTimeout(async () => {
    state.viewerPlayTimer = null;
    if (!state.viewerPlaying || generation !== playbackGeneration) return;
    const timeline = $(".viewer-timeline input");
    const sample = state.realArtifact?.currentSample;
    if (!timeline || timeline.disabled || !sample) {
      stopViewerPlayback();
      return;
    }
    const max = Number(timeline.max) || 0;
    const next = Number(timeline.value) >= max ? 0 : Number(timeline.value) + 1;
    await renderRealArtifactSample(state.artifactSample, sample.feature, next);
    if (!state.viewerPlaying || generation !== playbackGeneration) return;
    if (!state.realArtifact?.currentSample) {
      stopViewerPlayback();
      return;
    }
    scheduleViewerPlayback(generation);
  }, 450);
}

export function toggleViewerPlayback() {
  if (state.viewerPlaying) {
    stopViewerPlayback();
    return;
  }
  const timeline = $(".viewer-timeline input");
  const playButton = $("#viewerPlay");
  if (!timeline || timeline.disabled || !state.realArtifact?.currentSample) return;
  state.viewerPlaying = true;
  if (playButton) playButton.textContent = "❚❚";
  const generation = ++playbackGeneration;
  scheduleViewerPlayback(generation);
}

export function stopViewerPlayback() {
  playbackGeneration += 1;
  if (state.viewerPlayTimer) window.clearTimeout(state.viewerPlayTimer);
  state.viewerPlayTimer = null;
  state.viewerPlaying = false;
  const playButton = $("#viewerPlay");
  if (playButton) playButton.textContent = "▶";
}

export function compareCurrentSample() {
  if (!state.realArtifact) {
    toast("Open an artifact before comparing.", "warn");
    return;
  }
  const path = String(state.realArtifact.path || "");
  const lower = path.toLowerCase();
  $("#artifactOverlay").classList.remove("open");
  if (lower.endsWith(".csv")) {
    state.pendingComparisonPaths = [path];
    toast("Opening Compare with the current CSV retained as run 1.");
    openStudio("comparison");
    return;
  }
  if (lower.endsWith(".h5") || lower.endsWith(".hdf5")) {
    state.pendingEvaluationPrediction = path;
    toast("Opening Evaluation with the current HDF5 retained as the prediction source.");
    openStudio("evaluation");
    return;
  }
  toast("This artifact type has no numeric comparison contract. Export comparable CSV or HDF5 evidence first.", "warn");
}

export function downloadCurrentSample() {
  const sample = state.realArtifact?.currentSample;
  if (!sample) {
    toast("No sample is loaded yet.", "warn");
    return;
  }
  const payload = {
    path: sample.path,
    sample: sample.sample,
    dataset: sample.dataset,
    feature: sample.feature,
    feature_name: sample.feature_name,
    feature_names: sample.feature_names,
    parameters: sample.parameters,
    timestep: sample.timestep,
    shape: sample.shape,
    stats: sample.stats,
    metadata: sample.metadata,
    x: sample.x,
    y: sample.y,
    z: sample.z,
    values: sample.values,
    mesh: sample.mesh
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `sample-${String(sample.sample).replaceAll("/", "-")}-f${sample.feature}-t${sample.timestep}.json`;
  link.click();
  URL.revokeObjectURL(url);
  toast(`Downloaded sample ${sample.sample} as JSON.`);
}

export async function copyArtifactId() {
  const artifact = state.realArtifact;
  if (!artifact) {
    toast("No artifact is open.", "warn");
    return;
  }
  const id = `${artifact.path}#${artifact.samples[state.artifactSample]?.id ?? ""}`;
  try {
    await navigator.clipboard.writeText(id);
    toast(`Copied artifact ID: ${id}`);
  } catch {
    toast(`Artifact ID: ${id}`, "warn");
  }
}

export function useArtifactInPipeline(addBlock, selectNode) {
  const artifact = state.realArtifact;
  if (!artifact) {
    toast("No artifact is open.", "warn");
    return;
  }
  const path = artifact.path;
  const type = artifact.source_kind === "geometry" ? "source.cad" : "source.hdf5";
  let node = state.nodes.find(item => item.type === type && item.config.path === path);
  if (!node) {
    addBlock(type);
    node = state.nodes.find(item => item.id === state.selectedNode);
    node.config.path = path;
  }
  // Carry the dataset's own names into the graph so downstream blocks - a
  // parametric study above all - name their channels instead of numbering them.
  const features = artifact.feature_names || [];
  const conditions = artifact.condition_names || [];
  if (features.length) node.config.feature_names = features.join(", ");
  if (conditions.length) node.config.condition_names = conditions.join(", ");
  propagateNames(node.id, features, conditions);
  $("#artifactOverlay").classList.remove("open");
  selectNode(node.id);
  const named = [
    features.length ? `${features.length} feature names` : "",
    conditions.length ? `${conditions.length} condition names` : ""
  ].filter(Boolean).join(" and ");
  toast(`${path} is available as a pipeline source block${named ? ` with ${named}` : ""}.`);
}

/** Push the source's channel names onto every block fed by it. */
function propagateNames(sourceId, features, conditions) {
  if (!features.length && !conditions.length) return;
  const visited = new Set([sourceId]);
  const queue = [sourceId];
  while (queue.length) {
    const current = queue.shift();
    for (const edge of state.edges.filter(item => item.fromNode === current)) {
      if (visited.has(edge.toNode)) continue;
      visited.add(edge.toNode);
      queue.push(edge.toNode);
      const target = state.nodes.find(item => item.id === edge.toNode);
      if (!target) continue;
      if (features.length && !target.config.feature_names) {
        target.config.feature_names = features.join(", ");
      }
      if (conditions.length && !target.config.condition_names) {
        target.config.condition_names = conditions.join(", ");
      }
    }
  }
}
