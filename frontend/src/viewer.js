import { $, $$, escapeHtml, toast } from "./dom.js";
import { state } from "./state.js";
import { BLOCK_SPECS, ICONS } from "./constants.js";
import { apiRequest } from "./api.js";
import { previewGraphic } from "./graphics.js";
import { openStudio } from "./studio.js";
import { createRenderer, createFallbackRenderer, defaultCamera, fieldColor, turboColor } from "./render3d.js";

export { fieldColor };

const HDF5_EXTENSIONS = [".h5", ".hdf5"];
const GEOMETRY_EXTENSIONS = [
  ".stl", ".ply", ".obj", ".off", ".step", ".stp", ".iges", ".igs", ".brep",
  ".vtk", ".vtu", ".vtp", ".msh"
];
const PREVIEW_EXTENSIONS = [...HDF5_EXTENSIONS, ...GEOMETRY_EXTENSIONS];

let viewport = null;
let renderer = null;

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
  if (!artifact) return;
  const selected = artifact.samples[sampleIndex];
  if (!selected) return;
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
      `/api/preview/sample?path=${encodeURIComponent(artifact.path)}&sample=${encodeURIComponent(selected.id)}&feature=${requestedFeature}&timestep=${timestep}`
    );
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
  } catch (error) {
    state.realArtifact.currentSample = null;
    $("#artifactDownload").disabled = true;
    $("#viewerReset").disabled = true;
    showViewerMessage(`Could not visualize this real sample: ${error.message}`);
    $("#sampleInfo").innerHTML = `<section class="info-block"><h3>Reader error</h3><p>${escapeHtml(error.message)}</p></section>`;
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

async function loadPreviewCatalog(path) {
  const catalog = await apiRequest(`/api/preview/samples?path=${encodeURIComponent(path)}`);
  if (!catalog.samples?.length) throw new Error(`${path} contains no visualizable samples.`);
  return catalog;
}

async function resolvePreview(node, spec) {
  const configured = configuredPreviewPath(node, spec);
  if (configured) {
    try {
      return await loadPreviewCatalog(configured);
    } catch (error) {
      if (node.type === "source.cad" || node.type === "source.hdf5") throw error;
    }
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
  stopViewerPlayback();
  $("#artifactDatasetList").innerHTML = `<div class="live-empty">Reading ${escapeHtml(path)}…</div>`;
  const catalog = await loadPreviewCatalog(path);
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
        await loadArtifactPath(button.dataset.previewDataset);
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
  picker.hidden = false;
  $("#artifactDatasetSearch").value = "";
  $("#artifactDatasetList").innerHTML = `<div class="live-empty">Scanning repository datasets and geometry…</div>`;
  try {
    const [datasets, geometry] = await Promise.all([
      apiRequest("/api/files?kind=dataset"),
      apiRequest("/api/files?kind=geometry")
    ]);
    const unique = new Map();
    [...(datasets.items || []), ...(geometry.items || [])]
      .filter(item => PREVIEW_EXTENSIONS.includes(String(item.extension || "").toLowerCase()))
      .forEach(item => unique.set(item.path, item));
    state.viewerDatasetChoices = [...unique.values()].sort((left, right) =>
      left.path.localeCompare(right.path, undefined, { numeric: true })
    );
    renderArtifactDatasetChoices($("#artifactDatasetSearch").value);
    $("#artifactDatasetSearch").focus();
  } catch (error) {
    $("#artifactDatasetList").innerHTML = `<div class="live-empty">Could not list datasets: ${escapeHtml(error.message)}</div>`;
  }
}

export function closeArtifactDatasetPicker() {
  $("#artifactDatasetPicker").hidden = true;
}

export async function uploadArtifactDataset(file) {
  if (!file || !state.api.connected) return;
  const lowerName = file.name.toLowerCase();
  if (!hasPreviewExtension(lowerName)) {
    toast("Choose an HDF5, CAD, mesh, or VTK file.", "warn");
    return;
  }
  const kind = HDF5_EXTENSIONS.some(extension => lowerName.endsWith(extension)) ? "dataset" : "geometry";
  const button = $("#artifactUploadDataset");
  const original = button.textContent;
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
    await loadArtifactPath(result.path, false);
    toast(`Uploaded ${file.name} (${formatBytes(result.size)}). Choose a sample to visualize.`);
  } catch (error) {
    toast(`Upload failed: ${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

export async function openArtifact(nodeId) {
  const node = state.nodes.find(item => item.id === nodeId);
  if (!node) return;
  if (!state.api.connected) {
    toast("Runtime is offline; actual samples cannot be read. Start with START_STUDIO.bat.", "error");
    return;
  }
  stopViewerPlayback();
  closeArtifactDatasetPicker();
  const spec = BLOCK_SPECS[node.type];
  state.artifactNode = nodeId;
  state.artifactSample = null;
  state.viewerMode = "field";
  state.realArtifact = null;
  resetViewerCamera(false);
  $("#artifactIcon").textContent = ICONS[spec.icon];
  $("#artifactIcon").style.color = spec.accent;
  $("#artifactTitle").textContent = `${spec.label} · repository samples`;
  $("#artifactSubtitle").textContent = "Resolving the configured artifact…";
  $("#artifactOverlay").classList.add("open");
  $("#sampleList").innerHTML = `<div class="live-empty">Scanning the configured source…</div>`;
  showViewerMessage("Waiting for an actual sample…");
  $("#sampleInfo").innerHTML = "";
  $("#artifactSampleSearch").value = "";
  try {
    const catalog = await resolvePreview(node, spec);
    state.realArtifact = { ...catalog, node, currentSample: null };
    state.viewerMode = catalog.default_mode || "field";
    $("#artifactTitle").textContent = `${spec.label} · repository samples`;
    $("#artifactSubtitle").textContent = `${catalog.path} · choose a sample to visualize`;
    renderArtifactCatalog();
    renderEmptyViewer();
  } catch (error) {
    $("#sampleList").innerHTML = "";
    showViewerMessage(error.message);
    $("#sampleInfo").innerHTML = `<section class="info-block"><h3>No actual preview available</h3><p>${escapeHtml(error.message)}</p></section>`;
    toast(`Actual sample viewer: ${error.message}`, "error");
  }
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
  state.viewerPlayTimer = window.setInterval(() => {
    const max = Number(timeline.max) || 0;
    const next = Number(timeline.value) >= max ? 0 : Number(timeline.value) + 1;
    const sample = state.realArtifact?.currentSample;
    if (!sample) {
      stopViewerPlayback();
      return;
    }
    renderRealArtifactSample(state.artifactSample, sample.feature, next);
  }, 450);
}

export function stopViewerPlayback() {
  if (state.viewerPlayTimer) window.clearInterval(state.viewerPlayTimer);
  state.viewerPlayTimer = null;
  state.viewerPlaying = false;
  const playButton = $("#viewerPlay");
  if (playButton) playButton.textContent = "▶";
}

export function compareCurrentSample() {
  if (!state.realArtifact) {
    toast("Open a sample before comparing.", "warn");
    return;
  }
  $("#artifactOverlay").classList.remove("open");
  toast("Opening Compare — pick the other model/run's output CSV to rank them side by side.");
  openStudio("comparison");
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
