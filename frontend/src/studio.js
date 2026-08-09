import { $, $$, escapeHtml, toast, formatBytes } from "./dom.js";
import { state, snapshot } from "./state.js";
import { ICONS, MODEL_CATALOG, BLOCK_SPECS, STUDIO_SECTIONS } from "./constants.js";
import { apiRequest } from "./api.js";
import { addBlock, render, selectNode } from "./graph.js";
import { configureViaLlm, loadConfigExample, openConfig } from "./config.js";
import { beginCommandJob, renderRuntimeJob } from "./run.js";
import { applyGraphAutofill, autoFillCount, autoFillMeta, markManualConfigValue } from "./autofill.js";

function assignManualConfig(node, values) {
  Object.assign(node.config, values);
  Object.entries(values).forEach(([key, value]) => markManualConfigValue(node, key, value));
  applyGraphAutofill();
}

export async function openStudio(section, nodeId = null) {
  if (!STUDIO_SECTIONS[section]) return;
  state.studioSection = section;
  state.studioNode = nodeId;
  $("#studioOverlay").classList.add("open");
  renderStudio();
  await renderLiveWorkspace(section);
}

export function studioCards(section) {
  if (section.modelCards) {
    return Object.entries(MODEL_CATALOG).map(([modelId, model]) => [
      model.label, "model", "native", model.description,
      [`${model.keys.length} config keys`, ...model.modes, model.dataset], `model.${modelId}`
    ]);
  }
  return section.cards || [];
}

export function renderStudio() {
  const section = STUDIO_SECTIONS[state.studioSection];
  $("#studioIcon").textContent = ICONS[section.icon] || ICONS.docs;
  $("#studioIcon").style.color = section.color;
  $("#studioTitle").textContent = section.title;
  $("#studioSubtitle").textContent = section.description;
  $("#studioSidebar").innerHTML = `<div class="studio-nav-label">Studio workspaces</div>${Object.entries(STUDIO_SECTIONS).map(([id, item]) => `<button class="studio-nav-button${id === state.studioSection ? " active" : ""}" data-studio-id="${id}" style="--section-color:${item.color}"><span class="studio-nav-icon">${ICONS[item.icon]}</span><span class="studio-nav-copy"><strong>${item.label}</strong><small>${item.note}</small></span><small>${studioCards(item).length}</small></button>`).join("")}`;
  $$("[data-studio-id]").forEach(button => button.addEventListener("click", () => openStudio(button.dataset.studioId, null)));
  const cards = studioCards(section);
  const actionable = cards.filter(([, , , , , block]) => block && BLOCK_SPECS[block]).length;
  const roadmap = cards.filter(([, , maturity]) => maturity === "roadmap").length;
  const stats = [
    [String(cards.length), "declared capabilities"],
    [String(actionable), "pipeline actions"],
    [String(cards.length - actionable), "information only"],
    [String(roadmap), "roadmap"]
  ];
  $("#studioMain").innerHTML = `<section class="studio-hero"><span><span class="studio-kicker">AI-CAE4All Studio</span><h3>${escapeHtml(section.title)}</h3><p>${escapeHtml(section.description)}</p></span><span class="studio-stats">${stats.map(([value, label]) => `<span class="studio-stat"><strong>${escapeHtml(value)}</strong><small>${escapeHtml(label)}</small></span>`).join("")}</span></section>
    <section class="capability-grid">${cards.map(([title, iconName, maturity, description, chips, block]) => {
      const available = Boolean(block && BLOCK_SPECS[block]);
      const actionLabel = available ? "Open in pipeline →" : maturity === "roadmap" ? "Not implemented" : "No live action";
      return `<article class="capability-card${available ? "" : " unavailable"}" style="--card-color:${section.color}"><header class="capability-head"><span class="capability-title"><span class="capability-icon">${ICONS[iconName] || ICONS.docs}</span>${escapeHtml(title)}</span><span class="maturity ${maturity}">${maturity}</span></header><p>${escapeHtml(description)}</p><div class="chip-row">${chips.map(chip => `<span class="chip">${escapeHtml(chip)}</span>`).join("")}</div><button class="capability-link" data-capability-block="${available ? block : ""}"${available ? "" : " disabled"}>${actionLabel}</button></article>`;
    }).join("")}</section>`;
  $$("[data-capability-block]").forEach(button => button.addEventListener("click", () => {
    const type = button.dataset.capabilityBlock;
    if (type && BLOCK_SPECS[type]) {
      $("#studioOverlay").classList.remove("open");
      let node = state.nodes.find(item => item.type === type);
      if (!node) {
        addBlock(type);
        node = state.nodes.find(item => item.id === state.selectedNode);
      } else selectNode(node.id);
      toast(`${BLOCK_SPECS[type].label} is selected in the pipeline.`);
    }
  }));
}

export function liveShell(title, description) {
  $("#studioMain").innerHTML = `<section class="studio-hero"><span><span class="studio-kicker">Live repository data</span><h3>${escapeHtml(title)}</h3><p>${escapeHtml(description)}</p></span></section><section class="live-grid"><div class="live-empty">Loading real AI-CAE4ALL state…</div></section>`;
  return $(".live-grid", $("#studioMain"));
}

export function liveError(container, error) {
  container.innerHTML = `<div class="live-empty"><strong>Could not load live data</strong><br><br>${escapeHtml(error.message || String(error))}<br><br>Restart with frontend/START_STUDIO.bat.</div>`;
}

function ensureModelNode(modelId) {
  const type = `model.${modelId}`;
  if (!BLOCK_SPECS[type]) return null;
  let node = state.nodes.find(item => item.type === type);
  if (!node) {
    addBlock(type);
    node = state.nodes.find(item => item.id === state.selectedNode);
  } else selectNode(node.id);
  return node;
}

function editModelConfig(modelId, useLlm = false) {
  const node = ensureModelNode(modelId);
  if (!node) return;
  $("#studioOverlay").classList.remove("open");
  openConfig(node.id);
  if (useLlm) configureViaLlm();
}

export async function renderModelsWorkspace(container) {
  const models = [...(state.api.models.length ? state.api.models : (await apiRequest("/api/models")).items)]
    .sort((left, right) =>
      left.model.localeCompare(right.model, undefined, { sensitivity: "base" })
    );
  container.innerHTML = `<div class="live-summary">
    <span><strong>${models.length}</strong><small>registered routes</small></span>
    <span><strong>${models.filter(model => model.healthy).length}</strong><small>healthy installations</small></span>
    <span><strong>${models.reduce((sum, model) => sum + model.modes.length, 0)}</strong><small>actual route modes</small></span>
    <span><strong>${models.reduce((sum, model) => sum + model.known_keys.length, 0)}</strong><small>accepted-key entries</small></span>
  </div><div class="live-list">${models.map(model => `<article class="live-row live-row-clickable" data-model-row="${escapeHtml(model.model)}">
    <span><strong>${escapeHtml(model.model)} · ${escapeHtml(model.method)}</strong><small>${escapeHtml(model.repository)} → ${escapeHtml(model.entrypoint)}</small></span>
    <span class="chip-row">${model.modes.map(mode => `<span class="chip">${escapeHtml(mode)}</span>`).join("")}</span>
    <span><strong>${model.known_keys.length} keys</strong><small>${escapeHtml(model.dataset_kind || "no dataset contract")} · ${model.healthy ? "healthy" : "broken"}</small></span>
    <span class="live-actions"><button class="button small" data-live-configs="${escapeHtml(model.model)}">Examples</button>${BLOCK_SPECS[`model.${model.model}`] ? `<button class="button small primary" data-live-model="${escapeHtml(model.model)}">Open block</button>` : ""}</span>
  </article>`).join("")}</div>`;
  $$("[data-live-model]", container).forEach(button => button.addEventListener("click", event => {
    event.stopPropagation();
    editModelConfig(button.dataset.liveModel);
  }));
  $$("[data-live-configs]", container).forEach(button => button.addEventListener("click", event => {
    event.stopPropagation();
    renderModelConfigs(container, button.dataset.liveConfigs);
  }));
  $$("[data-model-row]", container).forEach(row => row.addEventListener("click", () => {
    renderModelDetail(container, models.find(item => item.model === row.dataset.modelRow));
  }));
}

async function renderModelConfigs(container, modelId) {
  const configs = await apiRequest(`/api/configs?model=${encodeURIComponent(modelId)}`);
  container.innerHTML = `<div class="live-toolbar"><strong>${escapeHtml(modelId)} checked-in configurations</strong><button class="button small" id="liveBackModels">Back to models</button></div><div class="live-list">${configs.items.map(item => `<article class="live-row">
    <span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.path)}</small></span>
    <span class="chip-row"><span class="chip">${escapeHtml(item.mode || "unknown mode")}</span></span>
    <span><strong>${formatBytes(item.size)}</strong><small>${escapeHtml(item.modified)}</small></span>
    <span class="live-actions"><button class="button small primary" data-load-config="${escapeHtml(item.path)}">Load into block</button></span>
  </article>`).join("") || `<div class="live-empty">No checked-in configuration declares model ${escapeHtml(modelId)}.</div>`}</div>`;
  $("#liveBackModels").addEventListener("click", () => renderModelsWorkspace(container));
  $$("[data-load-config]", container).forEach(load => load.addEventListener("click", () => loadConfigExample(modelId, load.dataset.loadConfig)));
  return configs;
}

function normalizedModelToken(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function modelConfigRows(keys, values, query = "", node = null) {
  const normalized = query.trim().toLowerCase();
  const visible = keys.filter(key => !normalized || key.toLowerCase().includes(normalized));
  return visible.length ? visible.map(key => {
    const configured = Object.hasOwn(values, key) && String(values[key]) !== "";
    const automatic = autoFillMeta(node, key);
    return `<div class="model-config-row${configured ? " configured" : ""}${automatic ? " graph-autofilled" : ""}">
      <code>${escapeHtml(key)}</code>
      <span>${configured ? escapeHtml(values[key]) : "not set"}${automatic ? `<small class="inline-auto">auto · ${escapeHtml(automatic.sourceLabel)}</small>` : ""}</span>
    </div>`;
  }).join("") : `<div class="live-empty">No configuration keys match this filter.</div>`;
}

/** Combined per-model view: the full accepted config contract, checked-in
 * configs, and every matching job from this server session. */
async function renderModelDetail(container, model) {
  if (!model) return;
  container.innerHTML = `<div class="live-empty">Loading ${escapeHtml(model.model)} details…</div>`;
  const [configs, jobs, metricCatalog] = await Promise.all([
    apiRequest(`/api/configs?model=${encodeURIComponent(model.model)}`),
    apiRequest("/api/jobs"),
    apiRequest(`/api/training-metrics?model_id=${encodeURIComponent(model.model)}`)
  ]);
  const blockSpec = BLOCK_SPECS[`model.${model.model}`];
  const currentNode = state.nodes.find(item => item.type === `model.${model.model}`);
  applyGraphAutofill();
  const configValues = currentNode?.config || MODEL_CATALOG[model.model]?.defaults || {};
  const configuredCount = model.known_keys.filter(key => Object.hasOwn(configValues, key) && String(configValues[key]) !== "").length;
  const needles = [model.model, model.method, blockSpec?.label].filter(Boolean).map(normalizedModelToken);
  const relatedJobs = jobs.items.filter(job => {
    const routedModels = (job.steps || []).map(step => step.route?.model).filter(Boolean);
    if (routedModels.length) return routedModels.includes(model.model);
    const haystack = normalizedModelToken(`${job.label || ""} ${job.step_label || ""} ${(job.steps || []).map(step => step.label || "").join(" ")}`);
    return needles.some(needle => haystack.includes(needle));
  });
  const metricJobIds = new Set(metricCatalog.items.map(item => item.job_id));
  container.innerHTML = `<div class="live-toolbar"><span><strong>${escapeHtml(model.model)} · ${escapeHtml(model.method)}</strong><small>${escapeHtml(model.repository)} → ${escapeHtml(model.entrypoint)}</small></span><button class="button small" id="liveBackModels">Back to models</button></div>
  <div class="live-summary">
    <span><strong>${model.known_keys.length}</strong><small>accepted keys</small></span>
    <span><strong>${model.modes.length}</strong><small>route modes</small></span>
    <span><strong>${configs.items.length}</strong><small>checked-in configs</small></span>
    <span><strong>${relatedJobs.length}</strong><small>persisted Studio jobs</small></span>
  </div>
  <div class="live-toolbar"><span><strong>Full configuration contract</strong><small>${configuredCount}/${model.known_keys.length} keys currently set from ${currentNode ? "the pipeline block" : "catalog defaults"}${currentNode && autoFillCount(currentNode) ? ` · ${autoFillCount(currentNode)} graph-filled` : ""}.</small></span><span class="live-actions"><button class="button small" id="modelConfigureLlm">Configure via LLM</button><button class="button small primary" id="modelEditConfig">Edit full config</button></span></div>
  <div class="model-config-browser">
    <label class="search"><span>⌕</span><input id="modelConfigSearch" type="search" placeholder="Filter all ${model.known_keys.length} config keys"></label>
    <div class="model-config-list" id="modelConfigList">${modelConfigRows(model.known_keys, configValues, "", currentNode)}</div>
  </div>
  <div class="live-toolbar"><span><strong>Checked-in configurations</strong></span>${BLOCK_SPECS[`model.${model.model}`] ? `<button class="button small primary" data-live-model="${escapeHtml(model.model)}">Open block</button>` : ""}</div>
  <div class="live-list">${configs.items.map(item => `<article class="live-row">
    <span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.path)}</small></span>
    <span class="chip-row"><span class="chip">${escapeHtml(item.mode || "unknown mode")}</span></span>
    <span><strong>${formatBytes(item.size)}</strong><small>${escapeHtml(item.modified)}</small></span>
    <span class="live-actions"><button class="button small primary" data-load-config="${escapeHtml(item.path)}">Load into block</button></span>
  </article>`).join("") || `<div class="live-empty">No checked-in configuration declares model ${escapeHtml(model.model)}.</div>`}</div>
  <div class="live-toolbar"><span><strong>Training status</strong><small>Jobs whose persisted route metadata identifies ${escapeHtml(model.model)}.</small></span></div>
  <div class="live-list">${relatedJobs.map(job => `<article class="live-row">
    <span><strong>${escapeHtml(job.label)}</strong><small>${escapeHtml(job.id)} · ${escapeHtml(job.created_at)}</small></span>
    <span class="chip-row"><span class="chip ${job.status === "failed" ? "warn" : ""}">${escapeHtml(job.status)}</span><span class="chip">${job.current_step}/${job.total_steps}</span></span>
    <span><strong>${job.returncode == null ? "running" : `exit ${job.returncode}`}</strong><small>${escapeHtml(job.step_label || "queued")}</small></span>
    <span class="live-actions">${metricJobIds.has(job.id) ? `<button class="button small" data-model-metrics-job="${escapeHtml(job.id)}">Metrics</button>` : ""}<button class="button small primary" data-open-job="${escapeHtml(job.id)}">Open log</button></span>
  </article>`).join("") || `<div class="live-empty">No persisted Studio job identifies ${escapeHtml(model.model)} yet. Run a configured block to see status here.</div>`}</div>`;
  $("#liveBackModels").addEventListener("click", () => renderModelsWorkspace(container));
  $("#modelConfigSearch").addEventListener("input", event => {
    $("#modelConfigList").innerHTML = modelConfigRows(model.known_keys, configValues, event.target.value, currentNode);
  });
  $("#modelEditConfig").addEventListener("click", () => editModelConfig(model.model));
  $("#modelConfigureLlm").addEventListener("click", () => editModelConfig(model.model, true));
  $$("[data-live-model]", container).forEach(button => button.addEventListener("click", () => {
    editModelConfig(button.dataset.liveModel);
  }));
  $$("[data-load-config]", container).forEach(load => load.addEventListener("click", () => loadConfigExample(model.model, load.dataset.loadConfig)));
  $$("[data-open-job]", container).forEach(button => button.addEventListener("click", async () => {
    const job = await apiRequest(`/api/jobs/${encodeURIComponent(button.dataset.openJob)}`);
    renderRuntimeJob(job);
  }));
  $$("[data-model-metrics-job]", container).forEach(button => button.addEventListener("click", () => {
    openTrainingMetricsWorkspace("", button.dataset.modelMetricsJob, { id: model.model, label: blockSpec?.label || model.method });
  }));
}

/** Opens the model-owned primary information surface used by pipeline card
 * previews and Inspect actions. Model cards never enter the dataset picker. */
export async function openModelDetailWorkspace(modelId) {
  if (!state.api.connected) {
    editModelConfig(modelId);
    toast("Runtime is offline, so training status is unavailable; opened the full config editor instead.", "warn");
    return;
  }
  state.studioSection = "models";
  $("#studioOverlay").classList.add("open");
  renderStudio();
  const section = STUDIO_SECTIONS.models;
  const container = liveShell(section.title, section.description);
  try {
    const models = state.api.models.length ? state.api.models : (await apiRequest("/api/models")).items;
    const model = models.find(item => item.model === modelId);
    if (!model) throw new Error(`Model route ${modelId} is not registered.`);
    await renderModelDetail(container, model);
  } catch (error) {
    liveError(container, error);
  }
}

const TRAINING_METRIC_COLORS = [
  "#167864", "#b56d32", "#6b5b95", "#3d7595", "#a64f59",
  "#708a38", "#8b653f", "#49746c", "#90658d", "#4f668a"
];

function formatMetricValue(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  if (numeric === 0) return "0";
  const magnitude = Math.abs(numeric);
  if (magnitude >= 10000 || magnitude < .001) return numeric.toExponential(3);
  return numeric.toLocaleString(undefined, { maximumSignificantDigits: 5 });
}

function sampledMetricPoints(points, limit = 700) {
  if (points.length <= limit) return points;
  const stride = (points.length - 1) / (limit - 1);
  return Array.from({ length: limit }, (unused, index) => points[Math.round(index * stride)]);
}

function smoothedMetric(metric, factor) {
  const smoothing = Math.max(0, Math.min(.95, Number(factor) || 0));
  if (!smoothing || metric.points.length < 2) return metric;
  let previous = Number(metric.points[0].y);
  return {
    ...metric,
    points: metric.points.map((point, index) => {
      const raw = Number(point.y);
      previous = index ? smoothing * previous + (1 - smoothing) * raw : raw;
      return { ...point, y: previous };
    })
  };
}

function downloadMetricCsv(job, metrics) {
  if (!metrics.length) {
    toast("Select at least one metric before downloading.", "warn");
    return;
  }
  const csvCell = value => {
    const text = String(value ?? "");
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  const rows = [["job_id", "run", "model", "metric", "event", "step", "value", "log_line"]];
  metrics.forEach(metric => metric.points.forEach(point => rows.push([
    job.job_id,
    job.label,
    (job.models || []).join("|"),
    metric.label,
    point.event || metric.event || job.x_label,
    point.x,
    point.y,
    point.line || ""
  ])));
  const text = `${rows.map(row => row.map(csvCell).join(",")).join("\n")}\n`;
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${String(job.label || job.job_id).replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase()}-metrics.csv`;
  link.click();
  URL.revokeObjectURL(url);
  toast(`Downloaded ${metrics.length} visible metric series as CSV.`);
}

function trainingMetricPlot(metric, color, xLabel) {
  const width = 620;
  const height = 190;
  const pad = { left: 48, right: 18, top: 18, bottom: 34 };
  const points = sampledMetricPoints(metric.points || []);
  const xs = points.map(point => Number(point.x));
  const ys = points.map(point => Number(point.y));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const rawMinY = Math.min(...ys);
  const rawMaxY = Math.max(...ys);
  const ySpan = rawMaxY - rawMinY;
  const yPad = ySpan ? ySpan * .08 : Math.max(Math.abs(rawMaxY) * .08, 1e-9);
  const minY = rawMinY - yPad;
  const maxY = rawMaxY + yPad;
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const xPosition = value => minX === maxX ? pad.left + plotWidth / 2 : pad.left + (value - minX) / (maxX - minX) * plotWidth;
  const yPosition = value => pad.top + (maxY - value) / (maxY - minY) * plotHeight;
  const coordinates = points.map(point => `${xPosition(Number(point.x)).toFixed(2)},${yPosition(Number(point.y)).toFixed(2)}`).join(" ");
  const dots = points.length <= 80
    ? points.map(point => `<circle cx="${xPosition(Number(point.x)).toFixed(2)}" cy="${yPosition(Number(point.y)).toFixed(2)}" r="3" fill="${color}"/>`).join("")
    : "";
  return `<svg class="training-metric-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(metric.label)} plot">
    <g class="training-metric-grid">${[0, 1, 2, 3, 4].map(index => {
      const y = pad.top + index * plotHeight / 4;
      return `<path d="M${pad.left} ${y}H${width - pad.right}"/><text x="${pad.left - 7}" y="${y + 3}" text-anchor="end">${escapeHtml(formatMetricValue(maxY - index * (maxY - minY) / 4))}</text>`;
    }).join("")}</g>
    ${points.length > 1 ? `<polyline points="${coordinates}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>` : ""}
    ${dots}
    <text class="training-axis-label" x="${pad.left}" y="${height - 9}">${escapeHtml(xLabel)} ${escapeHtml(minX)}</text>
    <text class="training-axis-label" x="${width - pad.right}" y="${height - 9}" text-anchor="end">${escapeHtml(maxX)}</text>
  </svg>`;
}

function connectedTrainingModel(nodeId) {
  const edge = state.edges.find(item => item.toNode === nodeId && item.toPort === "metrics");
  const source = edge && state.nodes.find(item => item.id === edge.fromNode);
  const sourceSpec = source && BLOCK_SPECS[source.type];
  return sourceSpec?.isModel ? {
    id: sourceSpec.modelId,
    label: sourceSpec.label
  } : null;
}

function metricExclusions(node) {
  return new Set(String(node?.config?.excluded_metrics || "").split(",").map(item => item.trim()).filter(Boolean));
}

function setMetricNodeConfig(node, values) {
  if (!node) return;
  if (state.nodes.includes(node)) snapshot();
  Object.assign(node.config, values);
}

async function renderTrainingMetricsWorkspace(container, nodeId = "", preferredJobId = "", linkedModelOverride = null) {
  const pipelineNode = state.nodes.find(item => item.id === nodeId);
  if (nodeId && (!pipelineNode || pipelineNode.type !== "evaluate.training_metrics")) {
    container.innerHTML = `<div class="live-empty">The Train Metrics block is no longer in this pipeline.</div>`;
    return;
  }
  const node = pipelineNode || { config: { job_id: preferredJobId, excluded_metrics: "", smoothing: "0" } };
  const result = await apiRequest("/api/training-metrics");
  const linkedModel = linkedModelOverride || connectedTrainingModel(nodeId);
  const jobs = [...result.items].sort((left, right) => {
    const leftMatch = linkedModel && left.models.includes(linkedModel.id) ? 1 : 0;
    const rightMatch = linkedModel && right.models.includes(linkedModel.id) ? 1 : 0;
    return rightMatch - leftMatch;
  });
  if (!jobs.length) {
    container.innerHTML = `<div class="live-toolbar"><span><strong>Train Metrics</strong><small>Actual Studio job logs only</small></span><button class="button small" id="metricsBackJobs">All jobs</button></div>
      <div class="live-empty"><strong>No epoch metrics were found.</strong><br><br>Run a connected model first. Any metric written on an Epoch, Iteration, or Step line will appear automatically.</div>`;
    $("#metricsBackJobs").addEventListener("click", () => renderJobsWorkspace(container));
    return;
  }

  let selected = jobs.find(job => job.job_id === (preferredJobId || node.config.job_id))
    || jobs.find(job => linkedModel && job.models.includes(linkedModel.id))
    || jobs[0];
  if (!node.config.job_id) node.config.job_id = selected.job_id;

  const renderDashboard = () => {
    const excluded = metricExclusions(node);
    const visible = selected.metrics.filter(metric => !excluded.has(metric.key));
    const smoothing = Math.max(0, Math.min(.95, Number(node.config.smoothing) || 0));
    const modelText = linkedModel ? `Connected model: ${linkedModel.label}` : "No model is connected; showing all parsed jobs.";
    container.innerHTML = `
      <div class="live-toolbar training-metrics-toolbar">
        <span><strong>Train Metrics</strong><small>${escapeHtml(modelText)} · values parsed from ${escapeHtml(selected.log_path)}</small></span>
        <span class="live-actions"><button class="button small" id="metricsDownload">Download visible CSV</button><button class="button small" id="metricsRefresh">Refresh</button><button class="button small" id="metricsBackJobs">All jobs</button></span>
      </div>
      <div class="training-job-select">
        <label for="trainingJob">Model run</label>
        <select id="trainingJob">${jobs.map(job => `<option value="${escapeHtml(job.job_id)}"${job.job_id === selected.job_id ? " selected" : ""}>${escapeHtml(job.label)} · ${job.metric_count} metrics · ${escapeHtml(job.status)}</option>`).join("")}</select>
        <button class="button small" id="metricsOpenLog">Open log</button>
      </div>
      <div class="live-summary training-metrics-summary">
        <span><strong>${selected.metric_count}</strong><small>discovered metrics</small></span>
        <span><strong>${visible.length}</strong><small>currently plotted</small></span>
        <span><strong>${selected.point_count}</strong><small>actual log points</small></span>
        <span><strong>${escapeHtml(selected.status)}</strong><small>training status</small></span>
      </div>
      <div class="training-metrics-layout">
        <aside class="training-metric-picker">
          <header><span><strong>Metrics to plot</strong><small>All discovered items start selected.</small></span><span><button class="button small" id="metricsSelectAll">Plot all</button><button class="button small" id="metricsSelectNone">Plot none</button></span></header>
          <label class="training-smoothing"><span><strong>Smoothing</strong><small>Visual only; statistics and CSV stay raw.</small></span><output>${Math.round(smoothing * 100)}%</output><input id="metricsSmoothing" type="range" min="0" max=".95" step=".05" value="${smoothing}"></label>
          <div class="training-metric-options">${selected.metrics.map((metric, index) => `<label class="training-metric-option">
            <input type="checkbox" data-metric-toggle="${escapeHtml(metric.key)}"${excluded.has(metric.key) ? "" : " checked"}>
            <i style="--metric-color:${TRAINING_METRIC_COLORS[index % TRAINING_METRIC_COLORS.length]}"></i>
            <span><strong>${escapeHtml(metric.label)}</strong><small>${metric.count} points · last ${escapeHtml(formatMetricValue(metric.last))}</small></span>
          </label>`).join("")}</div>
        </aside>
        <section class="training-metric-plots" aria-live="polite">${visible.length ? visible.map(metric => {
          const index = selected.metrics.findIndex(item => item.key === metric.key);
          const color = TRAINING_METRIC_COLORS[index % TRAINING_METRIC_COLORS.length];
          return `<article class="training-metric-card" data-metric-plot="${escapeHtml(metric.key)}">
            <header><span><i style="--metric-color:${color}"></i><strong>${escapeHtml(metric.label)}</strong></span><small>last <b>${escapeHtml(formatMetricValue(metric.last))}</b> · min ${escapeHtml(formatMetricValue(metric.min))} · max ${escapeHtml(formatMetricValue(metric.max))}</small></header>
            ${trainingMetricPlot(smoothedMetric(metric, smoothing), color, selected.x_label)}
          </article>`;
        }).join("") : `<div class="live-empty training-no-plots"><strong>No metrics selected.</strong><br><br>Check one or more metric items, or choose Plot all.</div>`}</section>
      </div>`;

    $("#trainingJob").addEventListener("change", event => {
      const next = jobs.find(job => job.job_id === event.target.value);
      if (!next) return;
      selected = next;
      setMetricNodeConfig(node, { job_id: next.job_id, excluded_metrics: "" });
      renderDashboard();
    });
    $("#metricsSelectAll").addEventListener("click", () => {
      setMetricNodeConfig(node, { excluded_metrics: "" });
      renderDashboard();
    });
    $("#metricsSelectNone").addEventListener("click", () => {
      setMetricNodeConfig(node, { excluded_metrics: selected.metrics.map(metric => metric.key).join(",") });
      renderDashboard();
    });
    $("#metricsSmoothing").addEventListener("change", event => {
      setMetricNodeConfig(node, { smoothing: event.target.value });
      renderDashboard();
    });
    $("#metricsDownload").addEventListener("click", () => downloadMetricCsv(selected, visible));
    $$("[data-metric-toggle]", container).forEach(control => control.addEventListener("change", () => {
      const nextExcluded = metricExclusions(node);
      if (control.checked) nextExcluded.delete(control.dataset.metricToggle);
      else nextExcluded.add(control.dataset.metricToggle);
      setMetricNodeConfig(node, { excluded_metrics: [...nextExcluded].join(",") });
      renderDashboard();
    }));
    $("#metricsRefresh").addEventListener("click", () => renderTrainingMetricsWorkspace(container, nodeId, selected.job_id, linkedModel));
    $("#metricsBackJobs").addEventListener("click", () => renderJobsWorkspace(container));
    $("#metricsOpenLog").addEventListener("click", async () => {
      const job = await apiRequest(`/api/jobs/${encodeURIComponent(selected.job_id)}`);
      renderRuntimeJob(job);
    });
  };

  renderDashboard();
}

/** Open the information surface owned by a Train Metrics pipeline block. */
export async function openTrainingMetricsWorkspace(nodeId = "", preferredJobId = "", linkedModel = null) {
  state.studioSection = "experiments";
  $("#studioOverlay").classList.add("open");
  renderStudio();
  const container = liveShell("Train Metrics", "Plot every metric found in an actual connected model run, then exclude only the series you do not need.");
  try {
    if (!state.api.connected) throw new Error("Start the Studio runtime to read persisted training logs.");
    await renderTrainingMetricsWorkspace(container, nodeId, preferredJobId, linkedModel);
  } catch (error) {
    liveError(container, error);
  }
}

export async function renderFilesWorkspace(container, kind) {
  const result = await apiRequest(`/api/files?kind=${encodeURIComponent(kind)}`);
  const title = kind === "dataset" ? "Repository datasets and parameter files" : "Output artifacts and checkpoints";
  container.innerHTML = `<div class="live-toolbar"><span><strong>${title}</strong><small>${result.items.length}${result.truncated ? "+" : ""} files</small></span><input id="liveFileSearch" type="search" placeholder="Filter path or extension"></div><div class="live-list" id="liveFileList"></div>`;
  const renderRows = query => {
    const text = query.trim().toLowerCase();
    const items = result.items.filter(item => !text || item.path.toLowerCase().includes(text)).slice(0, 250);
    const pipelineType = item => {
      if ([".h5", ".hdf5"].includes(item.extension)) return "source.hdf5";
      if ([".pth", ".pt", ".ckpt"].includes(item.extension)) return "source.checkpoint";
      if ([".stl", ".step", ".stp", ".iges", ".igs", ".brep", ".obj", ".ply", ".off", ".msh", ".vtk", ".vtu", ".vtp"].includes(item.extension)) return "source.cad";
      return "";
    };
    $("#liveFileList").innerHTML = items.map(item => `<article class="live-row">
      <span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.path)}</small></span>
      <span class="chip-row"><span class="chip">${escapeHtml(item.extension || "file")}</span><span class="chip">${escapeHtml(item.kind)}</span></span>
      <span><strong>${formatBytes(item.size)}</strong><small>${escapeHtml(item.modified)}</small></span>
      <span class="live-actions">${[".h5", ".hdf5"].includes(item.extension) ? `<button class="button small" data-inspect-hdf5="${escapeHtml(item.path)}">Inspect HDF5</button>` : ""}${pipelineType(item) ? `<button class="button small primary" data-use-file="${escapeHtml(item.path)}" data-use-type="${pipelineType(item)}">Use in pipeline</button>` : ""}</span>
    </article>`).join("") || `<div class="live-empty">No files match this filter.</div>`;
    $$("[data-inspect-hdf5]", container).forEach(button => button.addEventListener("click", () => inspectHdf5(container, button.dataset.inspectHdf5, kind)));
    $$("[data-use-file]", container).forEach(button => button.addEventListener("click", () => {
      addBlock(button.dataset.useType);
      const node = state.nodes.find(item => item.id === state.selectedNode);
      if (!node) return;
      node.config.path = button.dataset.useFile;
      $("#studioOverlay").classList.remove("open");
      render();
      toast(`${BLOCK_SPECS[node.type].label} now points to ${button.dataset.useFile}.`);
    }));
  };
  renderRows("");
  $("#liveFileSearch").addEventListener("input", event => renderRows(event.target.value));
}

export async function inspectHdf5(container, path, kind) {
  container.innerHTML = `<div class="live-empty">Reading HDF5 structure without loading the full dataset…</div>`;
  try {
    const data = await apiRequest(`/api/hdf5?path=${encodeURIComponent(path)}`);
    container.innerHTML = `<div class="live-toolbar"><span><strong>${escapeHtml(data.path)}</strong><small>${formatBytes(data.size)} · ${data.items.length}${data.truncated ? "+" : ""} objects</small></span><button class="button small" id="liveBackFiles">Back</button></div>
      <div class="live-list">${data.items.map(item => `<article class="live-row">
        <span><strong>${escapeHtml(item.path)}</strong><small>${escapeHtml(item.type)}</small></span>
        <span class="chip-row">${item.shape ? item.shape.map(value => `<span class="chip">${escapeHtml(value)}</span>`).join("") : ""}</span>
        <span><strong>${escapeHtml(item.dtype || "group")}</strong><small>${item.attrs ? `${Object.keys(item.attrs).length} attrs` : ""}</small></span>
        <span></span>
      </article>`).join("")}</div>`;
    $("#liveBackFiles").addEventListener("click", () => renderFilesWorkspace(container, kind));
  } catch (error) {
    liveError(container, error);
  }
}

export async function renderJobsWorkspace(container) {
  const [result, metricCatalog] = await Promise.all([
    apiRequest("/api/jobs"),
    apiRequest("/api/training-metrics")
  ]);
  const metricJobs = new Map(metricCatalog.items.map(item => [item.job_id, item]));
  container.innerHTML = `<div class="live-toolbar"><span><strong>Studio-launched processes</strong><small>${result.items.length} persisted jobs</small></span><button class="button small" id="refreshJobs">Refresh</button></div><div class="live-list">${result.items.map(job => `<article class="live-row">
    <span><strong>${escapeHtml(job.label)}</strong><small>${escapeHtml(job.id)} · ${escapeHtml(job.created_at)}</small></span>
    <span class="chip-row"><span class="chip">${escapeHtml(job.status)}</span><span class="chip">${job.current_step}/${job.total_steps}</span></span>
    <span><strong>${job.returncode == null ? "running" : `exit ${job.returncode}`}</strong><small>${escapeHtml(job.step_label || "queued")}</small></span>
    <span class="live-actions">${metricJobs.has(job.id) ? `<button class="button small" data-job-metrics="${escapeHtml(job.id)}">Metrics</button>` : ""}<button class="button small primary" data-open-job="${escapeHtml(job.id)}">Open log</button></span>
  </article>`).join("") || `<div class="live-empty">No Studio jobs have been started. Run or validate a configured block.</div>`}</div>`;
  $("#refreshJobs").addEventListener("click", () => renderJobsWorkspace(container));
  $$("[data-open-job]", container).forEach(button => button.addEventListener("click", async () => {
    const job = await apiRequest(`/api/jobs/${encodeURIComponent(button.dataset.openJob)}`);
    renderRuntimeJob(job);
  }));
  $$("[data-job-metrics]", container).forEach(button => button.addEventListener("click", () => {
    const metricJob = metricJobs.get(button.dataset.jobMetrics);
    const modelId = metricJob?.models?.[0] || "";
    openTrainingMetricsWorkspace("", button.dataset.jobMetrics, modelId ? { id: modelId, label: MODEL_CATALOG[modelId]?.label || modelId } : null);
  }));
}

export async function renderDocsWorkspace(container) {
  const result = await apiRequest("/api/docs");
  container.innerHTML = `<div class="live-toolbar"><span><strong>Repository documentation</strong><small>${result.items.length}${result.truncated ? "+" : ""} Markdown files</small></span><input id="liveDocSearch" type="search" placeholder="Filter documentation"></div><div class="live-list" id="liveDocList"></div>`;
  const renderRows = query => {
    const text = query.trim().toLowerCase();
    const items = result.items.filter(item => !text || item.path.toLowerCase().includes(text)).slice(0, 250);
    $("#liveDocList").innerHTML = items.map(item => `<article class="live-row">
      <span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.path)}</small></span>
      <span class="chip-row"><span class="chip">Markdown</span></span>
      <span><strong>${formatBytes(item.size)}</strong><small>${escapeHtml(item.modified)}</small></span>
      <span class="live-actions"><button class="button small primary" data-open-doc="${escapeHtml(item.path)}">Read</button></span>
    </article>`).join("") || `<div class="live-empty">No documents match this filter.</div>`;
    $$("[data-open-doc]", container).forEach(button => button.addEventListener("click", () => openLiveDoc(container, button.dataset.openDoc)));
  };
  renderRows("");
  $("#liveDocSearch").addEventListener("input", event => renderRows(event.target.value));
}

export async function openLiveDoc(container, path) {
  const doc = await apiRequest(`/api/doc?path=${encodeURIComponent(path)}`);
  container.innerHTML = `<div class="live-toolbar"><strong>${escapeHtml(doc.path)}</strong><button class="button small" id="liveBackDocs">Back to documents</button></div><pre class="live-document">${escapeHtml(doc.text)}</pre>`;
  $("#liveBackDocs").addEventListener("click", () => renderDocsWorkspace(container));
}

export async function renderSystemWorkspace(container) {
  const [health, llm] = await Promise.all([
    apiRequest("/api/health"),
    apiRequest("/api/llm/settings").catch(() => null)
  ]);
  container.innerHTML = `<div class="live-summary">
    <span><strong>${health.ok ? "Connected" : "Failed"}</strong><small>suite runtime</small></span>
    <span><strong>${health.healthy_models}/${health.models}</strong><small>healthy routes</small></span>
    <span><strong>${health.gpus.length}</strong><small>visible NVIDIA GPUs</small></span>
    <span><strong>${escapeHtml(health.python_version)}</strong><small>${escapeHtml(health.python)}</small></span>
  </div><div class="live-list">${health.gpus.map(gpu => `<article class="live-row">
    <span><strong>GPU ${escapeHtml(gpu.index)} · ${escapeHtml(gpu.name)}</strong><small>driver ${escapeHtml(gpu.driver)} · ${escapeHtml(gpu.temperature_c)} °C</small></span>
    <span class="chip-row"><span class="chip">${escapeHtml(gpu.utilization_percent)}% util</span></span>
    <span><strong>${escapeHtml(gpu.memory_used_mb)} / ${escapeHtml(gpu.memory_total_mb)} MiB</strong><small>current device memory</small></span>
    <span></span>
  </article>`).join("") || `<div class="live-empty">No NVIDIA devices were reported by nvidia-smi.</div>`}</div>
  <div class="live-toolbar"><span><strong>Configuration audit</strong><small>Runs the real launcher's structural preflight (parse, spec, route) over every checked-in configs/*.txt.</small></span><label class="check-row"><input id="auditStrict" type="checkbox"> Strict</label><button class="button primary" id="runConfigAudit">Run audit</button></div>
  <div id="auditResults"></div>
  <div class="live-toolbar"><span><strong>LLM configuration service</strong><small>Master node for "Configure via LLM" in the block config editor (LLM_API_fast, /v1/chat/completions).</small></span></div>
  <div class="config-card">
    ${llm ? `<p class="config-help">Currently targeting <strong>${escapeHtml(llm.base_url)}</strong> as user <strong>${escapeHtml(llm.username)}</strong>${llm.configured ? "" : " (built-in default; not yet saved)"}.</p>` : `<p class="config-help">Could not read the current LLM settings.</p>`}
    <div style="display:grid;grid-template-columns:2fr 1fr;gap:6px">
      <label class="config-help">Master IP<input class="config-control" id="llmMasterIp" value="${escapeHtml(llm?.master_ip || "")}" placeholder="10.228.69.135"></label>
      <label class="config-help">Port<input class="config-control" id="llmMasterPort" type="number" value="${escapeHtml(llm?.port || 10002)}"></label>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:7px">
      <label class="config-help">Username<input class="config-control" id="llmUsername" value="${escapeHtml(llm?.username || "")}" placeholder="admin"></label>
      <label class="config-help">Password (leave blank to keep current)<input class="config-control" id="llmPassword" type="password" placeholder="••••••••"></label>
    </div>
    <button class="button primary" id="saveLlmSettings" style="margin-top:8px">Save LLM settings</button>
  </div>`;
  $("#runConfigAudit").addEventListener("click", () => runConfigAudit(container));
  $("#saveLlmSettings").addEventListener("click", async () => {
    const button = $("#saveLlmSettings");
    button.disabled = true;
    try {
      await apiRequest("/api/llm/settings", {
        method: "POST",
        body: {
          master_ip: $("#llmMasterIp").value,
          port: $("#llmMasterPort").value,
          username: $("#llmUsername").value,
          password: $("#llmPassword").value
        }
      });
      toast("LLM configuration service settings saved.");
      await renderSystemWorkspace(container);
    } catch (error) {
      toast(`Could not save LLM settings: ${error.message}`, "error");
      button.disabled = false;
    }
  });
}

async function runConfigAudit(container) {
  const button = $("#runConfigAudit");
  const resultsEl = $("#auditResults");
  const strict = $("#auditStrict").checked;
  button.disabled = true;
  button.textContent = "Auditing…";
  resultsEl.innerHTML = `<div class="live-empty">Running the real spec/parse/route checks over every checked-in config…</div>`;
  try {
    const audit = await apiRequest(`/api/audit-configs?strict=${strict ? "1" : "0"}`);
    const failing = audit.files.filter(item => item.status === "FAIL");
    resultsEl.innerHTML = `<div class="live-summary">
      <span><strong>${audit.summary.files}</strong><small>configs scanned</small></span>
      <span><strong>${audit.summary.files - failing.length}</strong><small>passing</small></span>
      <span><strong>${failing.length}</strong><small>failing</small></span>
      <span><strong>${audit.summary.warnings}</strong><small>total warnings</small></span>
    </div><div class="live-list" id="auditFileList">${audit.files.map(item => `<article class="live-row">
      <span><strong>${item.status === "PASS" ? "✓" : "✕"} ${escapeHtml(item.path)}</strong><small>${escapeHtml(item.model || "unresolved model")} · ${escapeHtml(item.mode || "unresolved mode")}</small></span>
      <span class="chip-row"><span class="chip ${item.status === "FAIL" ? "warn" : ""}">${item.errors} errors</span><span class="chip">${item.warnings} warnings</span></span>
      <span><strong class="${item.status === "FAIL" ? "graph-warning" : ""}">${item.status}</strong><small></small></span>
      <span class="live-actions">${item.report.diagnostics.length ? `<button class="button small" data-audit-detail="${escapeHtml(item.path)}">Diagnostics</button>` : ""}</span>
    </article>`).join("")}</div><div id="auditDetail"></div>`;
    $$("[data-audit-detail]", resultsEl).forEach(detailButton => detailButton.addEventListener("click", () => {
      const item = audit.files.find(entry => entry.path === detailButton.dataset.auditDetail);
      $("#auditDetail").innerHTML = `<div class="live-toolbar"><strong>${escapeHtml(item.path)}</strong></div><div class="diagnostics">${item.report.diagnostics.map(diag => `<div class="diagnostic ${diag.severity === "error" ? "error" : diag.severity === "warning" ? "warn" : ""}"><i></i><span>[${escapeHtml(diag.code)}]${diag.field ? ` ${escapeHtml(diag.field)}:` : ""} ${escapeHtml(diag.message)}${diag.hint ? ` Hint: ${escapeHtml(diag.hint)}` : ""}</span></div>`).join("") || "<div class=\"live-empty\">No diagnostics.</div>"}</div>`;
    }));
    toast(`Audited ${audit.summary.files} configs: ${failing.length} failing, ${audit.summary.warnings} warnings.`, failing.length ? "error" : "");
  } catch (error) {
    liveError(resultsEl, error);
    toast(`Config audit failed: ${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Run audit";
  }
}

export async function renderBenchmarksWorkspace(container) {
  const configs = await apiRequest("/api/configs");
  const items = configs.items.filter(item => item.path.toLowerCase().includes("benchmarks/"));
  container.innerHTML = `<div class="live-toolbar"><span><strong>Checked-in benchmark protocols</strong><small>${items.length} real configs</small></span></div><div class="live-list">${items.map(item => `<article class="live-row">
    <span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.path)}</small></span>
    <span class="chip-row"><span class="chip">${escapeHtml(item.model || "unknown")}</span><span class="chip">${escapeHtml(item.mode || "unknown")}</span></span>
    <span><strong>${formatBytes(item.size)}</strong><small>checked in · not executed here</small><small class="benchmark-preflight-result" data-benchmark-result="${escapeHtml(item.path)}"></small></span>
    <span class="live-actions"><button class="button small" data-benchmark-preflight="${escapeHtml(item.path)}">Preflight</button>${BLOCK_SPECS[`model.${item.model}`] ? `<button class="button small primary" data-benchmark-config="${escapeHtml(item.path)}" data-benchmark-model="${escapeHtml(item.model)}">Load</button>` : ""}</span>
  </article>`).join("")}</div>`;
  $$("[data-benchmark-config]", container).forEach(button => button.addEventListener("click", () => loadConfigExample(button.dataset.benchmarkModel, button.dataset.benchmarkConfig)));
  $$("[data-benchmark-preflight]", container).forEach(button => button.addEventListener("click", async () => {
    const path = button.dataset.benchmarkPreflight;
    const resultEl = $$("[data-benchmark-result]", container).find(item => item.dataset.benchmarkResult === path);
    button.disabled = true;
    button.textContent = "Checking…";
    try {
      const config = await apiRequest(`/api/config?path=${encodeURIComponent(path)}`);
      const result = await apiRequest("/api/preflight", {
        method: "POST",
        allowError: true,
        body: { config: config.text, label: path, strict: false }
      });
      const summary = result.report?.summary || { errors: 1, warnings: 0 };
      resultEl.className = `benchmark-preflight-result ${result.ok ? "pass" : "fail"}`;
      resultEl.textContent = result.ok
        ? `PASS · ${summary.warnings} warning${summary.warnings === 1 ? "" : "s"}`
        : `FAIL · ${summary.errors} error${summary.errors === 1 ? "" : "s"}`;
    } catch (error) {
      resultEl.className = "benchmark-preflight-result fail";
      resultEl.textContent = `ERROR · ${error.message}`;
    } finally {
      button.disabled = false;
      button.textContent = "Preflight";
    }
  }));
}

function connectedNodeValue(nodeId, targetPort, keys, suffixes = []) {
  if (!nodeId) return "";
  const edge = state.edges.find(item => item.toNode === nodeId && item.toPort === targetPort);
  const source = edge && state.nodes.find(item => item.id === edge.fromNode);
  if (!source) return "";
  const values = keys.flatMap(key => {
    if (key === "optimizationReport") return [source.optimizationReport];
    if (key === "savedConfigPath") return [source.savedConfigPath];
    return [source.config?.[key]];
  });
  return values.find(value => {
    if (typeof value !== "string" || !value.trim()) return false;
    return !suffixes.length || suffixes.some(suffix => value.toLowerCase().endsWith(suffix));
  }) || "";
}

/** inference/cae_infer classifies checkpoints into 5 families covering 8
 * model IDs (point_deeponet/deeponet/fno/gino -> neural_operator,
 * meshgraphnets(-v), transolver, sdfflow -> geometry). mlp and simulgenvae
 * checkpoints raise a real "Could not classify checkpoint" error there — the
 * repository-wide output/<model>/... path convention lets the Deploy
 * workspace warn about that before submitting a job that is certain to fail. */
function unsupportedDeployFamily(path) {
  const segments = String(path || "").toLowerCase().split(/[\\/]+/);
  if (segments.includes("mlp")) return "mlp";
  if (segments.includes("simulgenvae")) return "simulgenvae";
  return null;
}

export async function renderDeployWorkspace(container, nodeId = null) {
  const [deploy, checkpoints, datasets] = await Promise.all([
    apiRequest("/api/deploy"),
    apiRequest("/api/files?kind=checkpoint"),
    apiRequest("/api/files?kind=dataset")
  ]);
  const requestedNode = state.nodes.find(item => item.id === nodeId);
  const checkpointSource = requestedNode?.type === "source.checkpoint" ? requestedNode : null;
  const node = state.nodes.find(item => item.id === nodeId && item.type === "deploy.api")
    || state.nodes.find(item => item.type === "deploy.api");
  const hdf5 = datasets.items.filter(item => [".h5", ".hdf5"].includes(item.extension)).slice(0, 250);
  const connectedCheckpoint = connectedNodeValue(
    node?.id,
    "model",
    ["checkpoint_path", "checkpoint", "path", "modelpath", "vae_modelpath"],
    [".pth", ".pt", ".ckpt"]
  );
  const selectedCheckpoint = checkpointSource?.config.path || node?.config.checkpoint_path || connectedCheckpoint || "";
  const selectedInput = node?.config.input_path || "";
  const checkpointOptions = checkpoints.items.slice(0, 300);
  if (selectedCheckpoint && !checkpointOptions.some(item => item.path === selectedCheckpoint)) {
    checkpointOptions.unshift({ path: selectedCheckpoint });
  }
  container.innerHTML = `<div class="live-summary">
    <span><strong>${deploy.existing_exe ? "Available" : "Not built"}</strong><small>portable .exe</small></span>
    <span><strong>${deploy.pyinstaller_available ? "Installed" : "Missing"}</strong><small>PyInstaller</small></span>
    <span><strong>${deploy.families.length}</strong><small>portable inference families</small></span>
    <span><strong>POST</strong><small>${escapeHtml(deploy.api_endpoint)}</small></span>
  </div>
  <div class="live-toolbar"><span><strong>Portable CPU inference</strong><small>Runs inference/run_inference.py and auto-detects the checkpoint family.</small></span></div>
  <div class="config-card">
    <label class="config-help">Checkpoint${connectedCheckpoint ? " · graph-connected" : ""}</label><select class="config-control" id="deployCheckpoint"><option value="">Select a real checkpoint…</option>${checkpointOptions.map(item => `<option value="${escapeHtml(item.path)}"${item.path === selectedCheckpoint ? " selected" : ""}>${escapeHtml(item.path)}</option>`).join("")}</select>
    <div id="deployCheckpointWarning"></div>
    <label class="config-help">Input HDF5 (not used by SDFFlow)</label><select class="config-control" id="deployInput"><option value="">No input / SDFFlow</option>${hdf5.map(item => `<option value="${escapeHtml(item.path)}"${item.path === selectedInput ? " selected" : ""}>${escapeHtml(item.path)}</option>`).join("")}</select>
    <label class="config-help">Output folder (written under frontend/runtime/inference)</label><input class="config-control" id="deployOutput" value="${escapeHtml(node?.config.output_name || "studio-inference")}">
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:7px">
      <input class="config-control" id="deployTimesteps" value="${escapeHtml(node?.config.timesteps || "")}" placeholder="timesteps">
      <input class="config-control" id="deploySamples" value="${escapeHtml(node?.config.num_samples || "")}" placeholder="num samples">
      <input class="config-control" id="deployOdeSteps" value="${escapeHtml(node?.config.ode_steps || "")}" placeholder="ODE steps">
      <input class="config-control" id="deployConditions" value="${escapeHtml(node?.config.cond_values || "")}" placeholder="condition values">
    </div>
    <button class="button primary" id="runPortableInference" style="margin-top:8px">Run real portable inference</button>
  </div>
  <div class="live-toolbar"><span><strong>Windows executable</strong><small>${deploy.existing_exe ? escapeHtml(deploy.existing_exe.path) : "Build output stays under frontend/runtime/deploy."}</small></span><button class="button" id="buildPortableExe"${deploy.pyinstaller_available ? "" : " disabled"}>Build .exe with PyInstaller</button></div>`;
  const persistDeploy = () => {
    if (!node) return;
    const next = {
      checkpoint_path: $("#deployCheckpoint").value,
      input_path: $("#deployInput").value,
      output_name: $("#deployOutput").value,
      timesteps: $("#deployTimesteps").value,
      num_samples: $("#deploySamples").value,
      ode_steps: $("#deployOdeSteps").value,
      cond_values: $("#deployConditions").value
    };
    if (Object.entries(next).every(([key, value]) => String(node.config[key] || "") === String(value))) return;
    snapshot();
    assignManualConfig(node, next);
  };
  const renderCheckpointWarning = () => {
    const family = unsupportedDeployFamily($("#deployCheckpoint").value);
    $("#deployCheckpointWarning").innerHTML = family
      ? `<p class="config-help" style="color:var(--red,#b3261e)">This looks like a ${escapeHtml(family)} checkpoint. The portable bundle only classifies the ${deploy.families.length} families above (point_deeponet/deeponet/fno/gino, transolver, meshgraphnets(-v), sdfflow) — running it here will fail with "Could not classify checkpoint". ${family === "mlp" ? "Use the model's own inference/evaluation blocks instead." : "Use the SimulGen-VAE reconstruct mode instead."}</p>`
      : "";
  };
  renderCheckpointWarning();
  ["deployCheckpoint", "deployInput", "deployOutput", "deployTimesteps", "deploySamples", "deployOdeSteps", "deployConditions"]
    .forEach(id => $("#" + id).addEventListener("change", persistDeploy));
  $("#deployCheckpoint").addEventListener("change", renderCheckpointWarning);
  $("#runPortableInference").addEventListener("click", async () => {
    const checkpoint = $("#deployCheckpoint").value;
    if (!checkpoint) {
      toast("Select a real checkpoint first.", "error");
      return;
    }
    const unsupportedFamily = unsupportedDeployFamily(checkpoint);
    if (unsupportedFamily) {
      if (!window.confirm(`This checkpoint looks like ${unsupportedFamily}, which the portable bundle cannot classify and will reject. Run it anyway to see the real error?`)) return;
    } else if (!window.confirm("Run the portable CPU inference bundle with the selected repository files?")) return;
    persistDeploy();
    try {
      const job = await apiRequest("/api/inference/run", {
        method: "POST",
        body: {
          checkpoint,
          input: $("#deployInput").value,
          output_name: $("#deployOutput").value,
          timesteps: $("#deployTimesteps").value,
          num_samples: $("#deploySamples").value,
          ode_steps: $("#deployOdeSteps").value,
          cond_values: $("#deployConditions").value
        }
      });
      if (node) {
        snapshot();
        node.config.job_id = job.id;
      }
      beginCommandJob(job);
    } catch (error) {
      toast(error.message, "error");
    }
  });
  $("#buildPortableExe").addEventListener("click", async () => {
    if (!window.confirm("Build the real PyInstaller bundle? This can take several minutes and writes only under frontend/runtime/deploy.")) return;
    try {
      const job = await apiRequest("/api/build/exe", { method: "POST", body: {} });
      beginCommandJob(job);
    } catch (error) {
      toast(error.message, "error");
    }
  });
}

function connectedOptimizationCsv(nodeId) {
  if (!nodeId) return "";
  const edge = state.edges.find(item => item.toNode === nodeId && item.toPort === "candidates");
  const source = edge && state.nodes.find(item => item.id === edge.fromNode);
  if (!source) return "";
  const candidates = [
    source.config?.candidate_csv,
    source.config?.csv_path,
    source.config?.metrics_csv,
    source.config?.output_csv,
    source.config?.path
  ];
  return candidates.find(value => typeof value === "string" && value.toLowerCase().endsWith(".csv")) || "";
}

export async function renderOptimizationWorkspace(container, nodeId = null) {
  const artifacts = await apiRequest("/api/files?kind=artifact");
  const csvFiles = artifacts.items.filter(item => item.extension === ".csv");
  const node = state.nodes.find(item => item.id === nodeId && item.type === "optimize.design")
    || state.nodes.find(item => item.type === "optimize.design");
  const selectedCsv = node?.config.csv_path || connectedOptimizationCsv(node?.id) || "";
  container.innerHTML = `<div class="live-toolbar"><span><strong>Evidence-based Pareto selection</strong><small>Reads actual numeric rows from an output CSV. No surrogate score is invented.</small></span></div>
    <div class="config-card">
      <label class="config-help">Candidate/evaluation CSV</label><select class="config-control" id="optimizationCsv"><option value="">Select an output CSV…</option>${csvFiles.map(item => `<option value="${escapeHtml(item.path)}"${item.path === selectedCsv ? " selected" : ""}>${escapeHtml(item.path)}</option>`).join("")}</select>
      <section id="optimizationSchema" class="optimization-schema" aria-live="polite"><div class="live-empty">Select a CSV to inspect its real columns.</div></section>
      <label class="config-help">Constraints (semicolon-separated)</label><input class="config-control" id="optimizationConstraints" list="optimizationConstraintExamples" value="${escapeHtml(node?.config.constraints || "")}" placeholder="displacement <= 1.0; mass < 20"><datalist id="optimizationConstraintExamples"></datalist>
      <label class="config-help">Diversity-aware Pareto top-k</label><input class="config-control" id="optimizationTopK" type="number" min="1" max="200" value="${escapeHtml(node?.config.top_k || 10)}">
      <button class="button primary" id="runOptimization" style="margin-top:8px" disabled>Evaluate actual CSV</button>
    </div><div id="optimizationResults"></div>`;
  let schemaRevision = 0;
  const objectiveDirections = () => {
    const names = String(node?.config.objectives || "").split(",").map(item => item.trim()).filter(Boolean);
    const directions = String(node?.config.directions || "").split(",").map(item => item.trim().toLowerCase());
    return new Map(names.map((name, index) => [name, directions[index] === "max" ? "max" : "min"]));
  };
  const selectedObjectives = () => $$("[data-optimization-objective]:checked", container).map(input => input.value);
  const collectDirections = () => selectedObjectives().map(name => {
    const control = $$("[data-objective-direction]", container).find(item => item.dataset.objectiveDirection === name);
    return control?.value === "max" ? "max" : "min";
  });
  const syncRunState = () => {
    $("#runOptimization").disabled = !$("#optimizationCsv").value || selectedObjectives().length === 0;
  };
  const persistControls = () => {
    if (!node) return;
    const next = {
      csv_path: $("#optimizationCsv").value,
      objectives: selectedObjectives().join(","),
      directions: collectDirections().join(","),
      constraints: $("#optimizationConstraints").value,
      top_k: $("#optimizationTopK").value
    };
    if (Object.entries(next).every(([key, value]) => String(node.config[key] || "") === String(value))) return;
    snapshot();
    assignManualConfig(node, next);
  };
  const inspectCsv = async () => {
    const csvPath = $("#optimizationCsv").value;
    const revision = ++schemaRevision;
    $("#runOptimization").disabled = true;
    if (!csvPath) {
      $("#optimizationSchema").innerHTML = '<div class="live-empty">Select a CSV to inspect its real columns.</div>';
      persistControls();
      return;
    }
    $("#optimizationSchema").innerHTML = '<div class="live-empty">Inspecting CSV columns…</div>';
    try {
      const schema = await apiRequest("/api/optimization/schema", {
        method: "POST",
        body: { csv_path: csvPath }
      });
      if (revision !== schemaRevision) return;
      const saved = objectiveDirections();
      $("#optimizationSchema").innerHTML = `<header><strong>Objectives</strong><small>${schema.rows_sampled} rows sampled · ${schema.numeric_columns.length} numeric columns. Choose explicitly; identifiers are excluded from suggestions.</small></header>
        <div class="optimization-objectives">${schema.objective_columns.length ? schema.objective_columns.map(name => `<label class="optimization-objective">
          <input type="checkbox" data-optimization-objective value="${escapeHtml(name)}"${saved.has(name) ? " checked" : ""}>
          <span><strong>${escapeHtml(name)}</strong><small>${schema.numeric_counts[name]} numeric values</small></span>
          <select class="config-control" data-objective-direction="${escapeHtml(name)}" aria-label="${escapeHtml(name)} direction"><option value="min"${saved.get(name) !== "max" ? " selected" : ""}>Minimize</option><option value="max"${saved.get(name) === "max" ? " selected" : ""}>Maximize</option></select>
        </label>`).join("") : '<div class="live-empty">No usable numeric objective columns were found.</div>'}</div>
        ${schema.identifier_columns.length ? `<small class="schema-note">Identifier columns: ${schema.identifier_columns.map(escapeHtml).join(", ")}</small>` : ""}`;
      $("#optimizationConstraintExamples").innerHTML = schema.numeric_columns
        .map(name => `<option value="${escapeHtml(name)} <= "></option>`)
        .join("");
      $$("[data-optimization-objective], [data-objective-direction]", container).forEach(control => control.addEventListener("change", () => {
        persistControls();
        syncRunState();
      }));
      persistControls();
      syncRunState();
    } catch (error) {
      if (revision !== schemaRevision) return;
      $("#optimizationSchema").innerHTML = `<div class="live-empty"><strong>Schema inspection failed</strong><br><br>${escapeHtml(error.message)}</div>`;
    }
  };
  $("#optimizationCsv").addEventListener("change", inspectCsv);
  $("#optimizationConstraints").addEventListener("change", persistControls);
  $("#optimizationTopK").addEventListener("change", persistControls);
  $("#runOptimization").addEventListener("click", async () => {
    const csvPath = $("#optimizationCsv").value;
    const objectives = selectedObjectives();
    if (!csvPath || !objectives.length) {
      toast("Select a CSV and at least one detected objective.", "error");
      return;
    }
    persistControls();
    try {
      const report = await apiRequest("/api/optimization/run", {
        method: "POST",
        body: {
          csv_path: csvPath,
          objectives: objectives.join(","),
          directions: collectDirections().join(","),
          constraints: $("#optimizationConstraints").value,
          top_k: Number($("#optimizationTopK").value)
        }
      });
      if (node) {
        snapshot();
        node.config.report_path = report.report_path;
        node.optimizationReport = report.report_path;
      }
      $("#optimizationResults").innerHTML = `<div class="live-summary">
        <span><strong>${report.rows}</strong><small>CSV rows</small></span>
        <span><strong>${report.numeric_candidates}</strong><small>numeric candidates</small></span>
        <span><strong>${report.feasible}</strong><small>feasible</small></span>
        <span><strong>${report.pareto}</strong><small>Pareto designs</small></span>
      </div><div class="live-list" style="margin-top:8px">${report.selected.map((candidate, index) => `<article class="live-row">
        <span><strong>#${index + 1} · ${escapeHtml(candidate.id)}</strong><small>row ${candidate.index} · ${escapeHtml(report.report_path)}</small></span>
        <span class="chip-row">${Object.entries(candidate.objectives).map(([name, value]) => `<span class="chip">${escapeHtml(name)}=${escapeHtml(value)}</span>`).join("")}</span>
        <span><strong>${candidate.crowding == null ? "boundary" : Number(candidate.crowding).toFixed(4)}</strong><small>Pareto crowding</small></span><span></span>
      </article>`).join("")}</div>`;
      toast(`Optimization complete: ${report.pareto} Pareto candidates, ${report.selected.length} selected.`);
    } catch (error) {
      toast(`Optimization failed: ${error.message}`, "error");
    }
  });
  if (selectedCsv) await inspectCsv();
}

export async function renderFieldEvaluationWorkspace(container, nodeId = null) {
  const [artifacts, datasets] = await Promise.all([
    apiRequest("/api/files?kind=artifact"),
    apiRequest("/api/files?kind=dataset")
  ]);
  const predictions = artifacts.items.filter(item => [".h5", ".hdf5"].includes(item.extension));
  const truthFiles = datasets.items.filter(item => [".h5", ".hdf5"].includes(item.extension));
  const evaluationNode = state.nodes.find(node => node.id === nodeId && node.type === "evaluate.predictions")
    || state.nodes.find(node => node.type === "evaluate.predictions");
  const pendingPrediction = state.pendingEvaluationPrediction;
  state.pendingEvaluationPrediction = "";
  const selectedPrediction = evaluationNode?.config.prediction_path || pendingPrediction || connectedNodeValue(
    evaluationNode?.id,
    "prediction",
    ["prediction_path", "output_path", "path"],
    [".h5", ".hdf5"]
  ) || "";
  const selectedTruth = evaluationNode?.config.truth_path || connectedNodeValue(
    evaluationNode?.id,
    "truth",
    ["path", "dataset_path"],
    [".h5", ".hdf5"]
  ) || "";
  if (selectedPrediction && !predictions.some(item => item.path === selectedPrediction)) {
    predictions.unshift({ path: selectedPrediction });
  }
  if (selectedTruth && !truthFiles.some(item => item.path === selectedTruth)) {
    truthFiles.unshift({ path: selectedTruth });
  }
  container.innerHTML = `<div class="live-toolbar"><span><strong>Actual HDF5 field comparison</strong><small>Arrays are matched by sample ID; incompatible node counts are reported, never resampled silently.</small></span></div>
    <div class="config-card">
      <label class="config-help">Prediction / reconstruction HDF5</label><select class="config-control" id="evaluationPrediction"><option value="">Select a real HDF5 output…</option>${predictions.map(item => `<option value="${escapeHtml(item.path)}"${item.path === selectedPrediction ? " selected" : ""}>${escapeHtml(item.path)}</option>`).join("")}</select>
      <label class="config-help">Ground-truth HDF5</label><select class="config-control" id="evaluationTruth"><option value="">Select a real dataset…</option>${truthFiles.map(item => `<option value="${escapeHtml(item.path)}"${item.path === selectedTruth ? " selected" : ""}>${escapeHtml(item.path)}</option>`).join("")}</select>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:7px">
        <label class="config-help">Prediction field row<input class="config-control" id="evaluationPredictionStart" type="number" min="0" value="${escapeHtml(evaluationNode?.config.prediction_start ?? 3)}"></label>
        <label class="config-help">Truth field row<input class="config-control" id="evaluationTruthStart" type="number" min="0" value="${escapeHtml(evaluationNode?.config.truth_start ?? 3)}"></label>
        <label class="config-help">Number of fields<input class="config-control" id="evaluationFields" type="number" min="1" value="${escapeHtml(evaluationNode?.config.num_fields ?? 1)}"></label>
      </div>
      <button class="button primary" id="runFieldEvaluation" style="margin-top:8px">Compute real field metrics</button>
    </div><div id="evaluationResults"></div>`;
  const persistEvaluation = () => {
    if (!evaluationNode) return;
    const next = {
      prediction_path: $("#evaluationPrediction").value,
      truth_path: $("#evaluationTruth").value,
      prediction_start: $("#evaluationPredictionStart").value,
      truth_start: $("#evaluationTruthStart").value,
      num_fields: $("#evaluationFields").value
    };
    if (Object.entries(next).every(([key, value]) => String(evaluationNode.config[key] || "") === String(value))) return;
    snapshot();
    assignManualConfig(evaluationNode, next);
  };
  ["evaluationPrediction", "evaluationTruth", "evaluationPredictionStart", "evaluationTruthStart", "evaluationFields"]
    .forEach(id => $("#" + id).addEventListener("change", persistEvaluation));
  $("#runFieldEvaluation").addEventListener("click", async () => {
    if (!$("#evaluationPrediction").value || !$("#evaluationTruth").value) {
      toast("Select both prediction and ground-truth HDF5 files.", "error");
      return;
    }
    persistEvaluation();
    try {
      const report = await apiRequest("/api/evaluation/run", {
        method: "POST",
        body: {
          prediction_path: $("#evaluationPrediction").value,
          truth_path: $("#evaluationTruth").value,
          prediction_start: Number($("#evaluationPredictionStart").value),
          truth_start: Number($("#evaluationTruthStart").value),
          num_fields: Number($("#evaluationFields").value)
        }
      });
      if (evaluationNode) {
        snapshot();
        Object.assign(evaluationNode.config, {
          prediction_path: $("#evaluationPrediction").value,
          truth_path: $("#evaluationTruth").value,
          prediction_start: $("#evaluationPredictionStart").value,
          truth_start: $("#evaluationTruthStart").value,
          num_fields: $("#evaluationFields").value,
          metrics_csv: report.per_sample_csv,
          report_path: report.report_path
        });
      }
      const relativeL2 = report.aggregate.relative_l2 || {};
      const mae = report.aggregate.mae || {};
      const rmse = report.aggregate.rmse || {};
      $("#evaluationResults").innerHTML = `<div class="live-summary">
        <span><strong>${report.evaluated_samples}</strong><small>evaluated samples</small></span>
        <span><strong>${Number(relativeL2.mean).toExponential(4)}</strong><small>mean relative L2</small></span>
        <span><strong>${Number(mae.mean).toExponential(4)}</strong><small>mean MAE</small></span>
        <span><strong>${Number(rmse.mean).toExponential(4)}</strong><small>mean RMSE</small></span>
      </div><div class="live-toolbar"><span><strong>Evidence saved</strong><small>${escapeHtml(report.per_sample_csv)} · ${escapeHtml(report.report_path)}${report.skipped.length ? ` · ${report.skipped.length} skipped` : ""}</small></span><button class="button small" id="exportEvaluation">Open Export</button></div>`;
      $("#exportEvaluation").addEventListener("click", () => openStudio("export"));
      toast(`Evaluated ${report.evaluated_samples} actual field samples.`);
    } catch (error) {
      toast(`Evaluation failed: ${error.message}`, "error");
    }
  });
}

let comparisonRunNextId = 1;

function comparisonRunRow(index, csvFiles, selected = "") {
  return `<div class="comparison-run-row" data-run-row="${index}">
    <select class="config-control" data-run-select="${index}"><option value="">Select a real CSV…</option>${csvFiles.map(item => `<option value="${escapeHtml(item.path)}"${item.path === selected ? " selected" : ""}>${escapeHtml(item.path)}</option>`).join("")}</select>
    <button class="button square" data-remove-run="${index}" aria-label="Remove this run" title="Remove this run">×</button>
  </div>`;
}

function renderComparisonRuns(csvFiles, selectedPaths = []) {
  const initial = selectedPaths.length ? selectedPaths : [""];
  comparisonRunNextId = initial.length;
  $("#comparisonRunList").innerHTML = initial.map((selected, index) => comparisonRunRow(index, csvFiles, selected)).join("");
  $$("[data-remove-run]").forEach(button => button.addEventListener("click", () => {
    if ($$("[data-run-row]").length <= 1) {
      toast("Keep at least one run selected.", "warn");
      return;
    }
    $(`[data-run-row="${button.dataset.removeRun}"]`).remove();
  }));
}

function connectedComparisonSources(nodeId, metricJobs) {
  if (!nodeId) return { runs: [], unresolved: [] };
  const sources = state.edges
    .filter(edge => edge.toNode === nodeId && edge.toPort === "metrics")
    .map(edge => state.nodes.find(node => node.id === edge.fromNode))
    .filter(Boolean);
  const runs = [];
  const unresolved = [];
  const seenJobs = new Set();

  sources.forEach(source => {
    const spec = BLOCK_SPECS[source.type];
    if (source.type === "evaluate.predictions" && source.config.metrics_csv) return;
    let job = null;
    let modelNode = source;
    if (source.type === "evaluate.training_metrics") {
      job = metricJobs.find(item => item.job_id === source.config.job_id) || null;
      const modelEdge = state.edges.find(edge => edge.toNode === source.id && edge.toPort === "metrics");
      modelNode = modelEdge && state.nodes.find(node => node.id === modelEdge.fromNode);
    }
    const modelSpec = modelNode && BLOCK_SPECS[modelNode.type];
    if (!job && modelNode) {
      job = metricJobs.find(item => item.node_ids?.includes(modelNode.id))
        || (modelSpec?.isModel ? metricJobs.find(item => item.models?.includes(modelSpec.modelId)) : null);
    }
    if (!job) {
      unresolved.push({ node: source, label: spec?.label || source.id });
      return;
    }
    if (seenJobs.has(job.job_id)) return;
    seenJobs.add(job.job_id);
    runs.push({ source, modelNode, job });
  });
  return { runs, unresolved };
}

function connectedEvaluationCsvPaths(nodeId) {
  if (!nodeId) return [];
  return state.edges
    .filter(edge => edge.toNode === nodeId && edge.toPort === "metrics")
    .map(edge => state.nodes.find(node => node.id === edge.fromNode))
    .filter(node => node?.type === "evaluate.predictions" && node.config.metrics_csv)
    .map(node => node.config.metrics_csv)
    .filter((path, index, all) => all.indexOf(path) === index);
}

function sharedRunMetricKeys(runs) {
  if (!runs.length) return [];
  const first = runs[0].job.metrics.map(metric => metric.key);
  return first.filter(key => runs.every(run => run.job.metrics.some(metric => metric.key === key)));
}

function connectedRunPlot(runs, metricKey) {
  const width = 900;
  const height = 270;
  const pad = { left: 58, right: 22, top: 22, bottom: 38 };
  const series = runs.map((run, index) => ({
    ...run,
    color: TRAINING_METRIC_COLORS[index % TRAINING_METRIC_COLORS.length],
    metric: run.job.metrics.find(metric => metric.key === metricKey)
  })).filter(item => item.metric?.points?.length);
  const points = series.flatMap(item => item.metric.points);
  const xs = points.map(point => Number(point.x));
  const ys = points.map(point => Number(point.y));
  if (!points.length) return "";
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const rawMinY = Math.min(...ys);
  const rawMaxY = Math.max(...ys);
  const yPad = rawMaxY === rawMinY ? Math.max(Math.abs(rawMaxY) * .08, 1e-9) : (rawMaxY - rawMinY) * .08;
  const minY = rawMinY - yPad;
  const maxY = rawMaxY + yPad;
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const px = value => minX === maxX ? pad.left + plotWidth / 2 : pad.left + (value - minX) / (maxX - minX) * plotWidth;
  const py = value => pad.top + (maxY - value) / (maxY - minY) * plotHeight;
  return `<div class="connected-run-chart">
    <div class="connected-run-legend">${series.map(item => `<span><i style="--metric-color:${item.color}"></i>${escapeHtml(item.job.label)}</span>`).join("")}</div>
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Connected training run comparison">
      <g class="training-metric-grid">${[0, 1, 2, 3, 4].map(index => {
        const y = pad.top + index * plotHeight / 4;
        return `<path d="M${pad.left} ${y}H${width - pad.right}"/><text x="${pad.left - 8}" y="${y + 3}" text-anchor="end">${escapeHtml(formatMetricValue(maxY - index * (maxY - minY) / 4))}</text>`;
      }).join("")}</g>
      ${series.map(item => {
        const coordinates = item.metric.points.map(point => `${px(Number(point.x)).toFixed(2)},${py(Number(point.y)).toFixed(2)}`).join(" ");
        const dots = item.metric.points.length <= 80 ? item.metric.points.map(point => `<circle cx="${px(Number(point.x)).toFixed(2)}" cy="${py(Number(point.y)).toFixed(2)}" r="3" fill="${item.color}"/>`).join("") : "";
        return `${item.metric.points.length > 1 ? `<polyline points="${coordinates}" fill="none" stroke="${item.color}" stroke-width="2.5" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>` : ""}${dots}`;
      }).join("")}
      <text class="training-axis-label" x="${pad.left}" y="${height - 10}">${escapeHtml(series[0]?.job.x_label || "epoch")} ${minX}</text>
      <text class="training-axis-label" x="${width - pad.right}" y="${height - 10}" text-anchor="end">${maxX}</text>
    </svg>
  </div>`;
}

function renderConnectedRunComparison(container, compareNode, resolved) {
  const target = $("#connectedComparison", container);
  if (!target) return;
  if (!compareNode) {
    target.innerHTML = `<div class="live-empty"><strong>No Compare Models block is open.</strong><br><br>Open the center of a Compare Models block to resolve its graph-connected runs. CSV ranking remains available below.</div>`;
    return;
  }
  if (!resolved.runs.length) {
    target.innerHTML = `<div class="live-empty"><strong>No connected run has metric evidence yet.</strong><br><br>Connect two or more Train Metrics outputs, choose a real job in each block, then return here.${resolved.unresolved.length ? `<br><br>Unresolved: ${resolved.unresolved.map(item => escapeHtml(item.label)).join(", ")}` : ""}</div>`;
    return;
  }
  const sharedKeys = sharedRunMetricKeys(resolved.runs);
  const configured = String(compareNode.config.metric || "");
  const metricKey = sharedKeys.includes(configured)
    ? configured
    : sharedKeys.find(key => /val|validation/.test(key))
      || sharedKeys.find(key => /loss|error|recon/.test(key))
      || sharedKeys[0]
      || "";
  const direction = compareNode.config.direction === "max" ? "max" : "min";
  const ranked = metricKey ? resolved.runs.map(run => {
    const metric = run.job.metrics.find(item => item.key === metricKey);
    return { ...run, metric, value: Number(metric.last) };
  }).sort((left, right) => direction === "max" ? right.value - left.value : left.value - right.value) : [];

  target.innerHTML = `<div class="connected-comparison-head">
    <span><strong>Graph-connected training runs</strong><small>${resolved.runs.length} resolved · ${resolved.unresolved.length} unresolved · last raw value used for ranking</small></span>
    ${sharedKeys.length ? `<span><label>Metric<select id="connectedMetric">${sharedKeys.map(key => {
      const label = resolved.runs[0].job.metrics.find(metric => metric.key === key)?.label || key;
      return `<option value="${escapeHtml(key)}"${key === metricKey ? " selected" : ""}>${escapeHtml(label)}</option>`;
    }).join("")}</select></label><label>Direction<select id="connectedDirection"><option value="min"${direction === "min" ? " selected" : ""}>Lower is better</option><option value="max"${direction === "max" ? " selected" : ""}>Higher is better</option></select></label></span>` : ""}
  </div>
  <div class="connected-run-cards">${resolved.runs.map(item => `<article><span><strong>${escapeHtml(item.job.label)}</strong><small>${escapeHtml(item.job.job_id)} · ${escapeHtml((item.job.models || []).join(", ") || "unknown model")}</small></span><span class="chip ${item.job.status === "failed" ? "warn" : ""}">${escapeHtml(item.job.status)}</span></article>`).join("")}</div>
  ${sharedKeys.length ? `${connectedRunPlot(resolved.runs, metricKey)}<div class="connected-ranking">${ranked.map((item, index) => `<article><strong>#${index + 1} · ${escapeHtml(item.job.label)}</strong><span>${escapeHtml(item.metric.label)} = <b>${escapeHtml(formatMetricValue(item.value))}</b></span></article>`).join("")}</div>` : `<div class="live-empty"><strong>The connected runs have no common metric key.</strong><br><br>Training losses from different model families are not automatically comparable. Connect evaluation results produced on the same held-out dataset, or use the qualified CSV ranking below.</div>`}`;

  $("#connectedMetric", target)?.addEventListener("change", event => {
    snapshot();
    compareNode.config.metric = event.target.value;
    renderConnectedRunComparison(container, compareNode, resolved);
  });
  $("#connectedDirection", target)?.addEventListener("change", event => {
    snapshot();
    compareNode.config.direction = event.target.value;
    renderConnectedRunComparison(container, compareNode, resolved);
  });
}

/** Connects multiple model runs' output CSVs into one ranked comparison, not just one CSV's rows. */
export async function renderComparisonWorkspace(container, nodeId = null) {
  const [artifacts, metricCatalog] = await Promise.all([
    apiRequest("/api/files?kind=artifact"),
    apiRequest("/api/training-metrics")
  ]);
  const csvFiles = artifacts.items.filter(item => item.extension === ".csv");
  const compareNode = state.nodes.find(node => node.id === nodeId && node.type === "evaluate.compare") || null;
  const connectedRuns = connectedComparisonSources(nodeId, metricCatalog.items);
  const evaluationCsvPaths = connectedEvaluationCsvPaths(nodeId);
  let savedCsvPaths = [];
  try {
    const parsed = JSON.parse(compareNode?.config.csv_paths || "[]");
    if (Array.isArray(parsed)) savedCsvPaths = parsed.filter(path => typeof path === "string");
  } catch { /* Ignore legacy or manually edited values. */ }
  const pendingCsvPaths = state.pendingComparisonPaths;
  state.pendingComparisonPaths = [];
  const initialCsvPaths = evaluationCsvPaths.length
    ? evaluationCsvPaths
    : savedCsvPaths.length
      ? savedCsvPaths
      : pendingCsvPaths;
  container.innerHTML = `<div class="live-toolbar"><span><strong>Connected run comparison</strong><small>Graph links resolve persisted run IDs first; only actual logged values are plotted.</small></span></div>
    <section class="connected-comparison config-card" id="connectedComparison"></section>
    <div class="live-toolbar"><span><strong>Qualified evaluation CSV ranking</strong><small>Use comparable held-out evaluation outputs when training metrics differ across model families.</small></span></div>
    <div class="config-card">
      <label class="config-help">Runs to compare${evaluationCsvPaths.length ? ` · ${evaluationCsvPaths.length} graph-connected evaluation output${evaluationCsvPaths.length === 1 ? "" : "s"} preselected` : ""}</label>
      <div id="comparisonRunList"></div>
      <button class="button small" id="comparisonAddRun" style="margin-top:6px">+ Add run</button>
      <label class="config-help" style="margin-top:10px">Model / group column (used when a CSV has one)</label><input class="config-control" id="comparisonGroup" value="${escapeHtml(compareNode?.config.group_column || "model")}">
      <label class="config-help">Numeric metric column</label><select class="config-control" id="comparisonMetric"><option value="">Select CSV runs first</option></select>
      <div class="config-help" id="comparisonSchemaStatus" role="status" aria-live="polite">The common CSV schema will be detected automatically.</div>
      <label class="config-help">Direction</label><select class="config-control" id="comparisonDirection"><option value="min"${compareNode?.config.csv_direction === "max" ? "" : " selected"}>Lower is better</option><option value="max"${compareNode?.config.csv_direction === "max" ? " selected" : ""}>Higher is better</option></select>
      <button class="button primary" id="runComparison" style="margin-top:8px" disabled>Rank actual rows</button>
    </div><div id="comparisonResults"></div>`;
  renderConnectedRunComparison(container, compareNode, connectedRuns);
  renderComparisonRuns(csvFiles, initialCsvPaths);
  let schemaRevision = 0;
  const selectedCsvPaths = () => $$("[data-run-select]").map(select => select.value).filter(Boolean);
  const persistCsvComparison = () => {
    if (!compareNode) return;
    const next = {
      csv_paths: JSON.stringify(selectedCsvPaths()),
      group_column: $("#comparisonGroup").value,
      csv_metric: $("#comparisonMetric").value,
      csv_direction: $("#comparisonDirection").value
    };
    if (Object.entries(next).every(([key, value]) => String(compareNode.config[key] || "") === String(value))) return;
    snapshot();
    assignManualConfig(compareNode, next);
  };
  const refreshComparisonSchema = async () => {
    const revision = ++schemaRevision;
    const csvPaths = selectedCsvPaths();
    const metricSelect = $("#comparisonMetric");
    const runButton = $("#runComparison");
    if (!csvPaths.length) {
      metricSelect.innerHTML = '<option value="">Select CSV runs first</option>';
      runButton.disabled = true;
      $("#comparisonSchemaStatus").textContent = "The common CSV schema will be detected automatically.";
      persistCsvComparison();
      return;
    }
    runButton.disabled = true;
    $("#comparisonSchemaStatus").textContent = `Inspecting ${csvPaths.length} selected CSV run${csvPaths.length === 1 ? "" : "s"}…`;
    try {
      const schema = await apiRequest("/api/comparison/schema", { method: "POST", body: { csv_paths: csvPaths } });
      if (revision !== schemaRevision) return;
      const previousMetric = metricSelect.value || compareNode?.config.csv_metric || "";
      const preferredMetric = schema.numeric_columns.find(column => column === previousMetric)
        || schema.numeric_columns.find(column => /^(mean_)?relative_l2$/i.test(column))
        || schema.numeric_columns.find(column => /rmse|mae|error|accuracy|score|loss/i.test(column))
        || schema.numeric_columns[0]
        || "";
      metricSelect.innerHTML = schema.numeric_columns.length
        ? schema.numeric_columns.map(column => `<option value="${escapeHtml(column)}"${column === preferredMetric ? " selected" : ""}>${escapeHtml(column)}</option>`).join("")
        : '<option value="">No common numeric column</option>';
      const groupInput = $("#comparisonGroup");
      const preferredGroup = schema.group_columns.find(column => column === groupInput.value)
        || schema.group_columns.find(column => /model|run|name|file|case|sample|id/i.test(column))
        || "";
      groupInput.value = preferredGroup;
      groupInput.setAttribute("list", "comparisonGroupColumns");
      let groupList = $("#comparisonGroupColumns");
      if (!groupList) {
        groupInput.insertAdjacentHTML("afterend", '<datalist id="comparisonGroupColumns"></datalist>');
        groupList = $("#comparisonGroupColumns");
      }
      groupList.innerHTML = schema.group_columns.map(column => `<option value="${escapeHtml(column)}"></option>`).join("");
      runButton.disabled = !preferredMetric;
      $("#comparisonSchemaStatus").textContent = preferredMetric
        ? `${schema.common_columns.length} common columns · ${schema.numeric_columns.length} numeric metric${schema.numeric_columns.length === 1 ? "" : "s"} · ${schema.sources.reduce((sum, source) => sum + source.rows_sampled, 0)} rows sampled`
        : "These CSV runs have no common numeric metric column. Choose comparable evaluation outputs.";
      persistCsvComparison();
    } catch (error) {
      if (revision !== schemaRevision) return;
      metricSelect.innerHTML = '<option value="">Schema unavailable</option>';
      runButton.disabled = true;
      $("#comparisonSchemaStatus").textContent = `Schema inspection failed: ${error.message}`;
    }
  };
  $$("[data-run-select]").forEach(select => select.addEventListener("change", refreshComparisonSchema));
  $$("[data-remove-run]").forEach(button => button.addEventListener("click", refreshComparisonSchema));
  $("#comparisonGroup").addEventListener("change", persistCsvComparison);
  $("#comparisonMetric").addEventListener("change", persistCsvComparison);
  $("#comparisonDirection").addEventListener("change", persistCsvComparison);
  $("#comparisonAddRun").addEventListener("click", () => {
    if ($$("[data-run-row]").length >= 12) {
      toast("Compare at most 12 runs at once.", "warn");
      return;
    }
    const index = comparisonRunNextId++;
    $("#comparisonRunList").insertAdjacentHTML("beforeend", comparisonRunRow(index, csvFiles));
    const button = $(`[data-remove-run="${index}"]`);
    $(`[data-run-select="${index}"]`).addEventListener("change", refreshComparisonSchema);
    button.addEventListener("click", () => {
      if ($$("[data-run-row]").length <= 1) {
        toast("Keep at least one run selected.", "warn");
        return;
      }
      button.closest("[data-run-row]").remove();
      refreshComparisonSchema();
    });
  });
  await refreshComparisonSchema();
  $("#runComparison").addEventListener("click", async () => {
    const csvPaths = selectedCsvPaths();
    if (!csvPaths.length) {
      toast("Select at least one real comparison CSV.", "error");
      return;
    }
    try {
      const report = await apiRequest("/api/comparison/run", {
        method: "POST",
        body: {
          csv_paths: csvPaths,
          group_column: $("#comparisonGroup").value,
          metric: $("#comparisonMetric").value,
          direction: $("#comparisonDirection").value
        }
      });
      if (compareNode) {
        snapshot();
        Object.assign(compareNode.config, {
          csv_paths: JSON.stringify(csvPaths),
          group_column: $("#comparisonGroup").value,
          csv_metric: $("#comparisonMetric").value,
          csv_direction: $("#comparisonDirection").value,
          report_path: report.report_path
        });
      }
      $("#comparisonResults").innerHTML = `<div class="live-summary">
        <span><strong>${report.numeric_rows}</strong><small>numeric rows</small></span>
        <span><strong>${report.runs}</strong><small>connected run${report.runs === 1 ? "" : "s"}</small></span>
        <span><strong>${escapeHtml(report.best.name)}</strong><small>best model / group</small></span>
        <span><strong>${Number(report.best.value).toExponential(5)}</strong><small>${escapeHtml(report.metric)} · ${escapeHtml(report.direction)}</small></span>
      </div>
      ${report.runs > 1 ? `<div class="chip-row" style="margin-top:8px">${report.sources.map(source => `<span class="chip">${escapeHtml(source.run)}: ${source.numeric_rows}/${source.rows} numeric</span>`).join("")}</div>` : ""}
      <div class="live-list" style="margin-top:8px">${report.ranked.slice(0, 25).map(item => `<article class="live-row">
        <span><strong>#${item.rank} · ${escapeHtml(item.name)}</strong><small>${report.runs > 1 ? `${escapeHtml(item.source)} · ` : ""}source row ${item.index}</small></span>
        <span class="chip-row"><span class="chip">${escapeHtml(report.metric)}=${escapeHtml(item.value)}</span></span>
        <span></span><span></span>
      </article>`).join("")}</div>`;
      toast(`Ranked ${report.numeric_rows} actual model-result rows across ${report.runs} run${report.runs === 1 ? "" : "s"}.`);
    } catch (error) {
      toast(`Comparison failed: ${error.message}`, "error");
    }
  });
}

export async function renderExportWorkspace(container, nodeId = null) {
  const [artifacts, datasets, checkpoints] = await Promise.all([
    apiRequest("/api/files?kind=artifact"),
    apiRequest("/api/files?kind=dataset"),
    apiRequest("/api/files?kind=checkpoint")
  ]);
  const items = [...artifacts.items, ...datasets.items, ...checkpoints.items]
    .filter((item, index, all) => all.findIndex(candidate => candidate.path === item.path) === index)
    .slice(0, 1200);
  const node = state.nodes.find(item => item.id === nodeId && item.type === "output.export")
    || state.nodes.find(item => item.type === "output.export");
  const connectedPath = connectedNodeValue(
    node?.id,
    "input",
    ["report_path", "metrics_csv", "output_path", "csv_path", "path", "optimizationReport"]
  );
  const selectedPath = node?.config.source_path || connectedPath || "";
  container.innerHTML = `<div class="live-toolbar"><span><strong>Isolated artifact handoff</strong><small>Exports are written only under frontend/runtime/exports; source files are never rewritten.</small></span></div>
    <div class="config-card">
      <label class="config-help">File or directory path${connectedPath ? " · graph-connected" : ""}</label><input class="config-control" id="exportPath" list="exportPaths" value="${escapeHtml(selectedPath)}" placeholder="output/... or frontend/runtime/..."><datalist id="exportPaths">${items.map(item => `<option value="${escapeHtml(item.path)}"></option>`).join("")}</datalist>
      <label class="config-help">Export label</label><input class="config-control" id="exportLabel" value="${escapeHtml(node?.config.export_label || "ai-cae4all-artifact")}">
      <button class="button primary" id="runExport" style="margin-top:8px">Create downloadable export</button>
    </div><div id="exportResults"></div>`;
  const persistExport = () => {
    if (!node) return;
    const next = { source_path: $("#exportPath").value, export_label: $("#exportLabel").value };
    if (Object.entries(next).every(([key, value]) => String(node.config[key] || "") === String(value))) return;
    snapshot();
    assignManualConfig(node, next);
  };
  $("#exportPath").addEventListener("change", persistExport);
  $("#exportLabel").addEventListener("change", persistExport);
  $("#runExport").addEventListener("click", async () => {
    if (!$("#exportPath").value) {
      toast("Select or enter an existing repository artifact path.", "error");
      return;
    }
    persistExport();
    try {
      const result = await apiRequest("/api/export", {
        method: "POST",
        body: { path: $("#exportPath").value, label: $("#exportLabel").value }
      });
      if (node) {
        snapshot();
        Object.assign(node.config, {
          source_path: $("#exportPath").value,
          export_label: $("#exportLabel").value,
          export_path: result.path
        });
      }
      $("#exportResults").innerHTML = `<div class="live-toolbar"><span><strong>${escapeHtml(result.path)}</strong><small>${formatBytes(result.size)} · source ${escapeHtml(result.source)}</small></span><a class="button primary" href="${escapeHtml(result.browser_path)}" download>Download</a></div>`;
      toast("Export created from the real artifact.");
    } catch (error) {
      toast(`Export failed: ${error.message}`, "error");
    }
  });
}

export async function renderLiveWorkspace(sectionId) {
  if (!state.api.connected || !["models", "data", "experiments", "artifacts", "docs", "system", "benchmarks", "deploy", "optimization", "evaluation", "comparison", "export"].includes(sectionId)) return;
  const section = STUDIO_SECTIONS[sectionId];
  const container = liveShell(section.title, section.description);
  try {
    if (sectionId === "models") await renderModelsWorkspace(container);
    if (sectionId === "data") await renderFilesWorkspace(container, "dataset");
    if (sectionId === "experiments") await renderJobsWorkspace(container);
    if (sectionId === "artifacts") await renderFilesWorkspace(container, "artifact");
    if (sectionId === "docs") await renderDocsWorkspace(container);
    if (sectionId === "system") await renderSystemWorkspace(container);
    if (sectionId === "benchmarks") await renderBenchmarksWorkspace(container);
    if (sectionId === "deploy") await renderDeployWorkspace(container, state.studioNode);
    if (sectionId === "optimization") await renderOptimizationWorkspace(container, state.studioNode);
    if (sectionId === "evaluation") await renderFieldEvaluationWorkspace(container, state.studioNode);
    if (sectionId === "comparison") await renderComparisonWorkspace(container, state.studioNode);
    if (sectionId === "export") await renderExportWorkspace(container, state.studioNode);
  } catch (error) {
    liveError(container, error);
  }
}
