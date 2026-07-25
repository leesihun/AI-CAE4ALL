import { $, $$, escapeHtml, toast } from "./dom.js";
import { state } from "./state.js";
import { BLOCK_SPECS, ICONS } from "./constants.js";
import { apiRequest } from "./api.js";
import { previewGraphic } from "./graphics.js";
import { openStudio } from "./studio.js";

const HDF5_EXTENSIONS = [".h5", ".hdf5"];
const GEOMETRY_EXTENSIONS = [
  ".stl", ".ply", ".obj", ".off", ".step", ".stp", ".iges", ".igs", ".brep",
  ".vtk", ".vtu", ".vtp", ".msh"
];
const PREVIEW_EXTENSIONS = [...HDF5_EXTENSIONS, ...GEOMETRY_EXTENSIONS];

export function fieldColor(value, low, high) {
  const numeric = Number.isFinite(value) ? value : low;
  const ratio = Math.max(0, Math.min(1, (numeric - low) / (high - low || 1)));
  return `hsl(${(225 - ratio * 225).toFixed(0)} 82% ${48 + ratio * 10}%)`;
}

function finiteNumbers(values = []) {
  return values.filter(Number.isFinite);
}

function sceneProjection(sample, camera = state.viewerCamera) {
  const coordinates = {
    x: sample.x || [],
    y: sample.y || [],
    z: sample.z || []
  };
  const axes = Object.entries(coordinates).map(([name, values]) => {
    const finite = finiteNumbers(values);
    const low = finite.length ? Math.min(...finite) : 0;
    const high = finite.length ? Math.max(...finite) : 0;
    return { name, low, high, center: (low + high) / 2, range: high - low };
  });
  const maxRange = Math.max(...axes.map(axis => axis.range), 1);
  const active = [...axes].sort((left, right) => right.range - left.range);
  const planar = active[2].range <= maxRange * 1e-6;
  const center = Object.fromEntries(axes.map(axis => [axis.name, axis.center]));
  const horizontalAxis = active[0];
  const verticalAxis = active[1];
  const depthAxis = active[2];
  const rawProject = (x, y, z) => {
    const point = { x, y, z };
    const horizontal = point[horizontalAxis.name] - center[horizontalAxis.name];
    const vertical = point[verticalAxis.name] - center[verticalAxis.name];
    const depth = point[depthAxis.name] - center[depthAxis.name];
    const baseX = horizontal + (planar ? 0 : depth * 0.22);
    const baseY = vertical - (planar ? 0 : depth * 0.3);
    const yawCos = Math.cos(camera.yaw);
    const yawSin = Math.sin(camera.yaw);
    const pitchCos = Math.cos(camera.pitch);
    const pitchSin = Math.sin(camera.pitch);
    const yawX = baseX * yawCos + depth * yawSin;
    const yawDepth = -baseX * yawSin + depth * yawCos;
    return {
      x: yawX,
      y: baseY * pitchCos - yawDepth * pitchSin,
      depth: baseY * pitchSin + yawDepth * pitchCos
    };
  };
  const projected = coordinates.x.map((x, index) =>
    rawProject(x, coordinates.y[index], coordinates.z[index])
  ).filter(point => [point.x, point.y, point.depth].every(Number.isFinite));
  const xExtent = Math.max(...projected.map(point => Math.abs(point.x)), 1);
  const yExtent = Math.max(...projected.map(point => Math.abs(point.y)), 1);
  const scale = Math.min(345 / xExtent, 198 / yExtent);
  const orientation = planar
    ? `${horizontalAxis.name.toUpperCase()}–${verticalAxis.name.toUpperCase()} plane`
    : `isometric ${horizontalAxis.name.toUpperCase()}${verticalAxis.name.toUpperCase()}${depthAxis.name.toUpperCase()}`;
  return {
    label: `${orientation} · yaw ${Math.round(camera.yaw * 180 / Math.PI)}° · pitch ${Math.round(camera.pitch * 180 / Math.PI)}° · ${Math.round(camera.zoom * 100)}%`,
    project(x, y, z) {
      const point = rawProject(x, y, z);
      return {
        x: 400 + camera.panX + point.x * scale * camera.zoom,
        y: 247 + camera.panY - point.y * scale * camera.zoom,
        depth: point.depth
      };
    }
  };
}

function triangleGraphic(mesh, projection, mode, valueLow, valueHigh) {
  const triangles = mesh?.triangles;
  const count = Number(mesh?.returned_faces || 0);
  if (!triangles || !count) return "";
  const polygons = [];
  for (let index = 0; index < count; index += 1) {
    const raw = [
      [triangles.x1?.[index], triangles.y1?.[index], triangles.z1?.[index]],
      [triangles.x2?.[index], triangles.y2?.[index], triangles.z2?.[index]],
      [triangles.x3?.[index], triangles.y3?.[index], triangles.z3?.[index]]
    ];
    if (!raw.flat().every(Number.isFinite)) continue;
    const points = raw.map(point => projection.project(...point));
    const signedArea = (
      (points[1].x - points[0].x) * (points[2].y - points[0].y)
      - (points[1].y - points[0].y) * (points[2].x - points[0].x)
    );
    if (Math.abs(signedArea) < 0.03) continue;
    const shade = Math.max(0, Math.min(1, 0.44 + Math.abs(signedArea) / 900));
    const fieldValue = Number(mesh.values?.[index]);
    const fill = mode === "field" && Number.isFinite(fieldValue)
      ? fieldColor(fieldValue, valueLow, valueHigh)
      : `hsl(159 24% ${Math.round(35 + shade * 24)}%)`;
    polygons.push({
      depth: points.reduce((sum, point) => sum + point.depth, 0) / 3,
      markup: `<polygon points="${points.map(point => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ")}" fill="${fill}" stroke="${mode === "mesh" ? "#b9d1c7" : "#6f9184"}" stroke-width="${mode === "mesh" ? ".52" : ".34"}" opacity="${mode === "mesh" ? ".9" : ".82"}"/>`
    });
  }
  return polygons.sort((left, right) => left.depth - right.depth).map(item => item.markup).join("");
}

function edgeGraphic(mesh, projection, mode, valueLow, valueHigh) {
  if (!mesh?.returned_edges) return "";
  const lines = [];
  for (let index = 0; index < mesh.returned_edges; index += 1) {
    const raw = [
      [mesh.x1?.[index], mesh.y1?.[index], mesh.z1?.[index]],
      [mesh.x2?.[index], mesh.y2?.[index], mesh.z2?.[index]]
    ];
    if (!raw.flat().every(Number.isFinite)) continue;
    const [start, end] = raw.map(point => projection.project(...point));
    const color = mode === "field"
      ? fieldColor(mesh.values?.[index], valueLow, valueHigh)
      : "#a9c0b6";
    lines.push(
      `<line x1="${start.x.toFixed(1)}" y1="${start.y.toFixed(1)}" x2="${end.x.toFixed(1)}" y2="${end.y.toFixed(1)}" stroke="${color}" stroke-width="${mode === "field" ? "1.28" : ".92"}" opacity="${mode === "field" ? ".68" : ".55"}"/>`
    );
  }
  return lines.join("");
}

export function realMeshFieldGraphic(sample, mode = "field") {
  const coordinates = { x: sample.x || [], y: sample.y || [], z: sample.z || [] };
  const values = sample.values || [];
  if (!coordinates.x.length) {
    return `<div class="live-empty">The selected sample has no plottable values.</div>`;
  }
  const projection = sceneProjection(sample, state.viewerCamera);
  const finiteValues = finiteNumbers(values);
  const valueLow = Number.isFinite(sample.stats?.min)
    ? sample.stats.min
    : (finiteValues.length ? Math.min(...finiteValues) : 0);
  const valueHigh = Number.isFinite(sample.stats?.max)
    ? sample.stats.max
    : (finiteValues.length ? Math.max(...finiteValues) : 1);
  const mesh = sample.mesh;
  const faces = mode === "points" ? "" : triangleGraphic(mesh, projection, mode, valueLow, valueHigh);
  const lines = mode === "points" ? "" : edgeGraphic(mesh, projection, mode, valueLow, valueHigh);

  const pointLimit = mode === "mesh" ? 900 : 1800;
  const pointStride = Math.max(1, Math.ceil(coordinates.x.length / pointLimit));
  const circles = [];
  const seriesPoints = [];
  for (let index = 0; index < coordinates.x.length; index += pointStride) {
    const raw = [coordinates.x[index], coordinates.y[index], coordinates.z[index]];
    if (!raw.every(Number.isFinite)) continue;
    const point = projection.project(...raw);
    const color = mode === "mesh" || !sample.supports?.field
      ? "#e5f1ec"
      : fieldColor(values[index], valueLow, valueHigh);
    seriesPoints.push(`${point.x.toFixed(1)},${point.y.toFixed(1)}`);
    circles.push(
      `<circle cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="${mode === "points" ? "2.65" : mode === "field" ? "2.0" : "1.45"}" fill="${color}" stroke="${mode === "mesh" ? "#385a4e" : "none"}" stroke-width=".7" opacity="${mode === "mesh" ? ".72" : ".9"}"/>`
    );
  }
  const series = sample.preview_kind === "series" && mode !== "points"
    ? `<polyline points="${seriesPoints.join(" ")}" fill="none" stroke="#8fd5bd" stroke-width="1.65" opacity=".82"/>`
    : "";
  const meshLabel = mesh?.returned_faces
    ? `${mesh.returned_faces.toLocaleString()} / ${mesh.total_faces.toLocaleString()} faces`
    : mesh?.returned_edges
      ? `${mesh.returned_edges.toLocaleString()} / ${mesh.total_edges.toLocaleString()} edges`
      : "point cloud";
  const legend = mode === "field" && sample.supports?.field
    ? `<defs><linearGradient id="field-legend" x1="0" x2="1"><stop stop-color="${fieldColor(valueLow, valueLow, valueHigh)}"/><stop offset=".5" stop-color="${fieldColor((valueLow + valueHigh) / 2, valueLow, valueHigh)}"/><stop offset="1" stop-color="${fieldColor(valueHigh, valueLow, valueHigh)}"/></linearGradient></defs><rect x="574" y="474" width="192" height="8" rx="4" fill="url(#field-legend)"/><text x="574" y="468" fill="#bcd0c7" font-size="9" font-family="monospace">${Number(valueLow).toPrecision(4)}</text><text x="766" y="468" text-anchor="end" fill="#bcd0c7" font-size="9" font-family="monospace">${Number(valueHigh).toPrecision(4)}</text>`
    : "";
  const sourceLabel = sample.source_kind === "geometry" ? "GEOMETRY" : "HDF5";
  return `<svg viewBox="0 0 800 500" role="img" aria-label="Actual ${sourceLabel} ${escapeHtml(mode)} visualization">
    <rect width="800" height="500" rx="12" fill="#13251f"/>
    ${legend}
    <g>${faces}</g>
    <g>${lines}${series}</g>
    <g>${circles.join("")}</g>
    <text x="24" y="27" fill="#e1f0e9" font-size="12" font-family="monospace">${escapeHtml(mode.toUpperCase())} · actual ${escapeHtml(sample.dataset)}${sample.supports?.field ? ` · ${escapeHtml(sample.feature_name || `feature ${sample.feature}`)} · t=${sample.timestep}` : ""}</text>
    <text x="24" y="486" fill="#a9c0b6" font-size="10" font-family="monospace">${escapeHtml(projection.label)} · ${meshLabel} · ${sample.returned_points.toLocaleString()} / ${sample.total_points.toLocaleString()} points</text>
  </svg>`;
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
  $("#viewerModeMeta").textContent = state.viewerMode === "points"
    ? `${sample.returned_points.toLocaleString()} sampled points`
    : sample.mesh?.returned_faces
      ? `${sample.mesh.returned_faces.toLocaleString()} mesh faces`
      : sample.mesh?.returned_edges
        ? `${sample.mesh.returned_edges.toLocaleString()} mesh edges${sample.supports?.field ? ` · ${sample.feature_name}` : ""}`
        : sample.supports?.field
          ? `${sample.returned_points.toLocaleString()} field points · ${sample.feature_name}`
          : `${sample.returned_points.toLocaleString()} surface points`;
  $("#viewerVisual").innerHTML = realMeshFieldGraphic(sample, state.viewerMode);
}

export function resetViewerCamera(render = true) {
  state.viewerCamera = { yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0 };
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
      const bounds = visual.getBoundingClientRect();
      state.viewerCamera.panX += dx * 800 / Math.max(bounds.width, 1);
      state.viewerCamera.panY += dy * 500 / Math.max(bounds.height, 1);
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
  $("#viewerVisual").innerHTML = `<div class="live-empty">${escapeHtml(message)}</div>`;
  $("#sampleInfo").innerHTML = `<section class="info-block"><h3>No sample selected</h3><p>The dataset is ready. Select a sample explicitly to read and render it.</p></section>`;
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
  $("#viewerVisual").innerHTML = `<div class="live-empty">Reading actual sample ${escapeHtml(selected.id)}…</div>`;
  try {
    const sample = await apiRequest(
      `/api/preview/sample?path=${encodeURIComponent(artifact.path)}&sample=${encodeURIComponent(selected.id)}&feature=${requestedFeature}&timestep=${timestep}`
    );
    state.realArtifact.currentSample = sample;
    state.viewerMode = chooseViewerMode(sample, artifact.default_mode);
    renderViewerMode();
    $("#artifactDownload").disabled = false;
    $("#viewerReset").disabled = false;
    const stats = sample.stats || {};
    const spec = BLOCK_SPECS[artifact.node.type];
    $("#artifactTitle").textContent = `${spec.label} · actual samples`;
    $("#artifactSubtitle").textContent = `${artifact.path} · actual repository values`;
    const metadata = sample.metadata || {};
    const topology = metadata.total_faces != null
      ? `<br>faces=${Number(metadata.total_faces).toLocaleString()}`
      : sample.mesh?.total_edges != null
        ? `<br>edges=${Number(sample.mesh.total_edges).toLocaleString()}`
        : "";
    $("#sampleInfo").innerHTML = `<section class="info-block"><h3>Sample ${escapeHtml(sample.sample)}</h3><p>${escapeHtml(sample.path)} · ${escapeHtml(sample.dataset)}${sampleShape(sample)}</p><div class="stat-grid">
      <span class="stat-card"><strong>${sample.supports?.field ? formatStat(stats.min) : "—"}</strong><small>${sample.supports?.field ? "actual minimum" : "field unavailable"}</small></span>
      <span class="stat-card"><strong>${sample.supports?.field ? formatStat(stats.max) : "—"}</strong><small>${sample.supports?.field ? "actual maximum" : "field unavailable"}</small></span>
      <span class="stat-card"><strong>${sample.supports?.field ? formatStat(stats.mean) : "—"}</strong><small>${sample.supports?.field ? "actual mean" : "field unavailable"}</small></span>
      <span class="stat-card"><strong>${sample.total_points.toLocaleString()}</strong><small>points / values</small></span>
    </div></section>
    ${sample.supports?.field ? `<section class="info-block"><div class="section-title">Field selector</div>
      <label class="config-help">Feature channel</label><input class="config-control" id="realFeature" type="number" min="0" max="${Math.max(0, sample.feature_count - 1)}" value="${sample.feature}"${sample.feature_count <= 1 ? " disabled" : ""}>
      <label class="config-help">Timestep</label><input class="config-control" id="realTimestep" type="number" min="0" max="${Math.max(0, sample.timestep_count - 1)}" value="${sample.timestep}"${sample.timestep_count <= 1 ? " disabled" : ""}>
      <button class="button primary" id="loadRealField" style="width:100%;margin-top:8px">Load actual field</button>
    </section>` : `<section class="info-block"><div class="section-title">Surface geometry</div><p>This sample has geometry but no scalar field. Point mode preserves the real coordinates${sample.mesh ? "; Mesh mode also preserves the available topology" : ""}.</p></section>`}
    <section class="info-block"><div class="section-title">Provenance</div><p>source=${escapeHtml(sample.path)}<br>sample=${escapeHtml(sample.sample)}<br>reader=${escapeHtml(sample.source_kind)}<br>downsample=${sample.returned_points}/${sample.total_points}${topology}</p></section>`;
    $("#loadRealField")?.addEventListener("click", () =>
      renderRealArtifactSample(sampleIndex, Number($("#realFeature").value), Number($("#realTimestep").value))
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
    $("#viewerVisual").innerHTML = `<div class="live-empty">Could not visualize this real sample: ${escapeHtml(error.message)}</div>`;
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
  $("#viewerVisual").innerHTML = `<div class="live-empty">Waiting for an actual sample…</div>`;
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
    $("#viewerVisual").innerHTML = `<div class="live-empty">${escapeHtml(error.message)}</div>`;
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
  $("#artifactOverlay").classList.remove("open");
  selectNode(node.id);
  toast(`${path} is available as a pipeline source block.`);
}
