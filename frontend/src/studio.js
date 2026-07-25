import { $, $$, escapeHtml, toast, formatBytes } from "./dom.js";
import { state } from "./state.js";
import { ICONS, MODEL_CATALOG, BLOCK_SPECS, STUDIO_SECTIONS } from "./constants.js";
import { apiRequest } from "./api.js";
import { addBlock, selectNode } from "./graph.js";
import { loadConfigExample, openConfig } from "./config.js";
import { beginCommandJob, renderRuntimeJob } from "./run.js";

export async function openStudio(section) {
  if (!STUDIO_SECTIONS[section]) return;
  state.studioSection = section;
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
  $$("[data-studio-id]").forEach(button => button.addEventListener("click", () => openStudio(button.dataset.studioId)));
  const cards = studioCards(section);
  $("#studioMain").innerHTML = `<section class="studio-hero"><span><span class="studio-kicker">AI-CAE4All Studio</span><h3>${escapeHtml(section.title)}</h3><p>${escapeHtml(section.description)}</p></span><span class="studio-stats">${section.stats.map(([value, label]) => `<span class="studio-stat"><strong>${escapeHtml(value)}</strong><small>${escapeHtml(label)}</small></span>`).join("")}</span></section>
    <section class="capability-grid">${cards.map(([title, iconName, maturity, description, chips, block]) => `<article class="capability-card" style="--card-color:${section.color}"><header class="capability-head"><span class="capability-title"><span class="capability-icon">${ICONS[iconName] || ICONS.docs}</span>${escapeHtml(title)}</span><span class="maturity ${maturity}">${maturity}</span></header><p>${escapeHtml(description)}</p><div class="chip-row">${chips.map(chip => `<span class="chip">${escapeHtml(chip)}</span>`).join("")}</div><button class="capability-link" data-capability-block="${block || ""}">${block ? "Open in pipeline →" : "View details →"}</button></article>`).join("")}</section>`;
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
    } else toast("This capability is documented and demonstrated directly in this workspace.", "warn");
  }));
}

export function liveShell(title, description) {
  $("#studioMain").innerHTML = `<section class="studio-hero"><span><span class="studio-kicker">Live repository data</span><h3>${escapeHtml(title)}</h3><p>${escapeHtml(description)}</p></span></section><section class="live-grid"><div class="live-empty">Loading real AI-CAE4ALL state…</div></section>`;
  return $(".live-grid", $("#studioMain"));
}

export function liveError(container, error) {
  container.innerHTML = `<div class="live-empty"><strong>Could not load live data</strong><br><br>${escapeHtml(error.message || String(error))}<br><br>Restart with frontend/START_STUDIO.bat.</div>`;
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
  </div><div class="live-list">${models.map(model => `<article class="live-row">
    <span><strong>${escapeHtml(model.model)} · ${escapeHtml(model.method)}</strong><small>${escapeHtml(model.repository)} → ${escapeHtml(model.entrypoint)}</small></span>
    <span class="chip-row">${model.modes.map(mode => `<span class="chip">${escapeHtml(mode)}</span>`).join("")}</span>
    <span><strong>${model.known_keys.length} keys</strong><small>${escapeHtml(model.dataset_kind || "no dataset contract")} · ${model.healthy ? "healthy" : "broken"}</small></span>
    <span class="live-actions"><button class="button small" data-live-configs="${escapeHtml(model.model)}">Examples</button>${BLOCK_SPECS[`model.${model.model}`] ? `<button class="button small primary" data-live-model="${escapeHtml(model.model)}">Open block</button>` : ""}</span>
  </article>`).join("")}</div>`;
  $$("[data-live-model]", container).forEach(button => button.addEventListener("click", () => {
    const type = `model.${button.dataset.liveModel}`;
    $("#studioOverlay").classList.remove("open");
    let node = state.nodes.find(item => item.type === type);
    if (!node) {
      addBlock(type);
      node = state.nodes.find(item => item.id === state.selectedNode);
    } else selectNode(node.id);
    openConfig(node.id);
  }));
  $$("[data-live-configs]", container).forEach(button => button.addEventListener("click", async () => {
    const modelId = button.dataset.liveConfigs;
    const configs = await apiRequest(`/api/configs?model=${encodeURIComponent(modelId)}`);
    container.innerHTML = `<div class="live-toolbar"><strong>${escapeHtml(modelId)} checked-in configurations</strong><button class="button small" id="liveBackModels">Back to models</button></div><div class="live-list">${configs.items.map(item => `<article class="live-row">
      <span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.path)}</small></span>
      <span class="chip-row"><span class="chip">${escapeHtml(item.mode || "unknown mode")}</span></span>
      <span><strong>${formatBytes(item.size)}</strong><small>${escapeHtml(item.modified)}</small></span>
      <span class="live-actions"><button class="button small primary" data-load-config="${escapeHtml(item.path)}">Load into block</button></span>
    </article>`).join("") || `<div class="live-empty">No checked-in configuration declares model ${escapeHtml(modelId)}.</div>`}</div>`;
    $("#liveBackModels").addEventListener("click", () => renderModelsWorkspace(container));
    $$("[data-load-config]", container).forEach(load => load.addEventListener("click", () => loadConfigExample(modelId, load.dataset.loadConfig)));
  }));
}

export async function renderFilesWorkspace(container, kind) {
  const result = await apiRequest(`/api/files?kind=${encodeURIComponent(kind)}`);
  const title = kind === "dataset" ? "Repository datasets and parameter files" : "Output artifacts and checkpoints";
  container.innerHTML = `<div class="live-toolbar"><span><strong>${title}</strong><small>${result.items.length}${result.truncated ? "+" : ""} files</small></span><input id="liveFileSearch" type="search" placeholder="Filter path or extension"></div><div class="live-list" id="liveFileList"></div>`;
  const renderRows = query => {
    const text = query.trim().toLowerCase();
    const items = result.items.filter(item => !text || item.path.toLowerCase().includes(text)).slice(0, 250);
    $("#liveFileList").innerHTML = items.map(item => `<article class="live-row">
      <span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.path)}</small></span>
      <span class="chip-row"><span class="chip">${escapeHtml(item.extension || "file")}</span><span class="chip">${escapeHtml(item.kind)}</span></span>
      <span><strong>${formatBytes(item.size)}</strong><small>${escapeHtml(item.modified)}</small></span>
      <span class="live-actions">${[".h5", ".hdf5"].includes(item.extension) ? `<button class="button small primary" data-inspect-hdf5="${escapeHtml(item.path)}">Inspect HDF5</button>` : ""}</span>
    </article>`).join("") || `<div class="live-empty">No files match this filter.</div>`;
    $$("[data-inspect-hdf5]", container).forEach(button => button.addEventListener("click", () => inspectHdf5(container, button.dataset.inspectHdf5, kind)));
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
  const result = await apiRequest("/api/jobs");
  container.innerHTML = `<div class="live-toolbar"><span><strong>Studio-launched processes</strong><small>${result.items.length} jobs in this server session</small></span><button class="button small" id="refreshJobs">Refresh</button></div><div class="live-list">${result.items.map(job => `<article class="live-row">
    <span><strong>${escapeHtml(job.label)}</strong><small>${escapeHtml(job.id)} · ${escapeHtml(job.created_at)}</small></span>
    <span class="chip-row"><span class="chip">${escapeHtml(job.status)}</span><span class="chip">${job.current_step}/${job.total_steps}</span></span>
    <span><strong>${job.returncode == null ? "running" : `exit ${job.returncode}`}</strong><small>${escapeHtml(job.step_label || "queued")}</small></span>
    <span class="live-actions"><button class="button small primary" data-open-job="${escapeHtml(job.id)}">Open log</button></span>
  </article>`).join("") || `<div class="live-empty">No Studio jobs have been started. Run or validate a configured block.</div>`}</div>`;
  $("#refreshJobs").addEventListener("click", () => renderJobsWorkspace(container));
  $$("[data-open-job]", container).forEach(button => button.addEventListener("click", async () => {
    const job = await apiRequest(`/api/jobs/${encodeURIComponent(button.dataset.openJob)}`);
    renderRuntimeJob(job);
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
  const health = await apiRequest("/api/health");
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
  <div id="auditResults"></div>`;
  $("#runConfigAudit").addEventListener("click", () => runConfigAudit(container));
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
    <span><strong>${formatBytes(item.size)}</strong><small>qualified by repository workflow</small></span>
    <span class="live-actions">${BLOCK_SPECS[`model.${item.model}`] ? `<button class="button small primary" data-benchmark-config="${escapeHtml(item.path)}" data-benchmark-model="${escapeHtml(item.model)}">Load</button>` : ""}</span>
  </article>`).join("")}</div>`;
  $$("[data-benchmark-config]", container).forEach(button => button.addEventListener("click", () => loadConfigExample(button.dataset.benchmarkModel, button.dataset.benchmarkConfig)));
}

export async function renderDeployWorkspace(container) {
  const [deploy, checkpoints, datasets] = await Promise.all([
    apiRequest("/api/deploy"),
    apiRequest("/api/files?kind=checkpoint"),
    apiRequest("/api/files?kind=dataset")
  ]);
  const hdf5 = datasets.items.filter(item => [".h5", ".hdf5"].includes(item.extension)).slice(0, 250);
  container.innerHTML = `<div class="live-summary">
    <span><strong>${deploy.existing_exe ? "Available" : "Not built"}</strong><small>portable .exe</small></span>
    <span><strong>${deploy.pyinstaller_available ? "Installed" : "Missing"}</strong><small>PyInstaller</small></span>
    <span><strong>${deploy.families.length}</strong><small>portable inference families</small></span>
    <span><strong>POST</strong><small>${escapeHtml(deploy.api_endpoint)}</small></span>
  </div>
  <div class="live-toolbar"><span><strong>Portable CPU inference</strong><small>Runs inference/run_inference.py and auto-detects the checkpoint family.</small></span></div>
  <div class="config-card">
    <label class="config-help">Checkpoint</label><select class="config-control" id="deployCheckpoint"><option value="">Select a real checkpoint…</option>${checkpoints.items.slice(0, 300).map(item => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.path)}</option>`).join("")}</select>
    <label class="config-help">Input HDF5 (not used by SDFFlow)</label><select class="config-control" id="deployInput"><option value="">No input / SDFFlow</option>${hdf5.map(item => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.path)}</option>`).join("")}</select>
    <label class="config-help">Output folder (written under frontend/runtime/inference)</label><input class="config-control" id="deployOutput" value="studio-inference">
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:7px">
      <input class="config-control" id="deployTimesteps" placeholder="timesteps">
      <input class="config-control" id="deploySamples" placeholder="num samples">
      <input class="config-control" id="deployOdeSteps" placeholder="ODE steps">
      <input class="config-control" id="deployConditions" placeholder="condition values">
    </div>
    <button class="button primary" id="runPortableInference" style="margin-top:8px">Run real portable inference</button>
  </div>
  <div class="live-toolbar"><span><strong>Windows executable</strong><small>${deploy.existing_exe ? escapeHtml(deploy.existing_exe.path) : "Build output stays under frontend/runtime/deploy."}</small></span><button class="button" id="buildPortableExe"${deploy.pyinstaller_available ? "" : " disabled"}>Build .exe with PyInstaller</button></div>`;
  $("#runPortableInference").addEventListener("click", async () => {
    const checkpoint = $("#deployCheckpoint").value;
    if (!checkpoint) {
      toast("Select a real checkpoint first.", "error");
      return;
    }
    if (!window.confirm("Run the portable CPU inference bundle with the selected repository files?")) return;
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

export async function renderOptimizationWorkspace(container) {
  const artifacts = await apiRequest("/api/files?kind=artifact");
  const csvFiles = artifacts.items.filter(item => item.extension === ".csv");
  const node = state.nodes.find(item => item.type === "optimize.design");
  container.innerHTML = `<div class="live-toolbar"><span><strong>Evidence-based Pareto selection</strong><small>Reads actual numeric rows from an output CSV. No surrogate score is invented.</small></span></div>
    <div class="config-card">
      <label class="config-help">Candidate/evaluation CSV</label><select class="config-control" id="optimizationCsv"><option value="">Select an output CSV…</option>${csvFiles.map(item => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.path)}</option>`).join("")}</select>
      <label class="config-help">Objective columns (comma-separated)</label><input class="config-control" id="optimizationObjectives" value="${escapeHtml(node?.config.objectives || "")}" placeholder="peak_stress,mass">
      <label class="config-help">Directions, one per objective</label><input class="config-control" id="optimizationDirections" value="min,min" placeholder="min,min">
      <label class="config-help">Constraints (semicolon-separated)</label><input class="config-control" id="optimizationConstraints" value="${escapeHtml(node?.config.constraints || "")}" placeholder="displacement <= 1.0; mass < 20">
      <label class="config-help">Diversity-aware Pareto top-k</label><input class="config-control" id="optimizationTopK" type="number" min="1" max="200" value="${escapeHtml(node?.config.top_k || 10)}">
      <button class="button primary" id="runOptimization" style="margin-top:8px">Evaluate actual CSV</button>
    </div><div id="optimizationResults"></div>`;
  $("#runOptimization").addEventListener("click", async () => {
    const csvPath = $("#optimizationCsv").value;
    if (!csvPath) {
      toast("Select a real output CSV.", "error");
      return;
    }
    try {
      const report = await apiRequest("/api/optimization/run", {
        method: "POST",
        body: {
          csv_path: csvPath,
          objectives: $("#optimizationObjectives").value,
          directions: $("#optimizationDirections").value,
          constraints: $("#optimizationConstraints").value,
          top_k: Number($("#optimizationTopK").value)
        }
      });
      if (node) {
        node.config.objectives = $("#optimizationObjectives").value;
        node.config.constraints = $("#optimizationConstraints").value;
        node.config.top_k = $("#optimizationTopK").value;
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
}

export async function renderFieldEvaluationWorkspace(container) {
  const [artifacts, datasets] = await Promise.all([
    apiRequest("/api/files?kind=artifact"),
    apiRequest("/api/files?kind=dataset")
  ]);
  const predictions = artifacts.items.filter(item => [".h5", ".hdf5"].includes(item.extension));
  const truthFiles = datasets.items.filter(item => [".h5", ".hdf5"].includes(item.extension));
  container.innerHTML = `<div class="live-toolbar"><span><strong>Actual HDF5 field comparison</strong><small>Arrays are matched by sample ID; incompatible node counts are reported, never resampled silently.</small></span></div>
    <div class="config-card">
      <label class="config-help">Prediction / reconstruction HDF5</label><select class="config-control" id="evaluationPrediction"><option value="">Select a real HDF5 output…</option>${predictions.map(item => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.path)}</option>`).join("")}</select>
      <label class="config-help">Ground-truth HDF5</label><select class="config-control" id="evaluationTruth"><option value="">Select a real dataset…</option>${truthFiles.map(item => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.path)}</option>`).join("")}</select>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:7px">
        <label class="config-help">Prediction field row<input class="config-control" id="evaluationPredictionStart" type="number" min="0" value="3"></label>
        <label class="config-help">Truth field row<input class="config-control" id="evaluationTruthStart" type="number" min="0" value="3"></label>
        <label class="config-help">Number of fields<input class="config-control" id="evaluationFields" type="number" min="1" value="1"></label>
      </div>
      <button class="button primary" id="runFieldEvaluation" style="margin-top:8px">Compute real field metrics</button>
    </div><div id="evaluationResults"></div>`;
  $("#runFieldEvaluation").addEventListener("click", async () => {
    if (!$("#evaluationPrediction").value || !$("#evaluationTruth").value) {
      toast("Select both prediction and ground-truth HDF5 files.", "error");
      return;
    }
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

export async function renderComparisonWorkspace(container) {
  const artifacts = await apiRequest("/api/files?kind=artifact");
  const csvFiles = artifacts.items.filter(item => item.extension === ".csv");
  container.innerHTML = `<div class="live-toolbar"><span><strong>Evidence-backed model ranking</strong><small>Reads the selected numeric metric directly from a real CSV.</small></span></div>
    <div class="config-card">
      <label class="config-help">Comparison or evaluation CSV</label><select class="config-control" id="comparisonCsv"><option value="">Select a real CSV…</option>${csvFiles.map(item => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.path)}</option>`).join("")}</select>
      <label class="config-help">Model / group column</label><input class="config-control" id="comparisonGroup" value="model">
      <label class="config-help">Numeric metric column</label><input class="config-control" id="comparisonMetric" value="mean_relative_l2">
      <label class="config-help">Direction</label><select class="config-control" id="comparisonDirection"><option value="min">Lower is better</option><option value="max">Higher is better</option></select>
      <button class="button primary" id="runComparison" style="margin-top:8px">Rank actual rows</button>
    </div><div id="comparisonResults"></div>`;
  $("#runComparison").addEventListener("click", async () => {
    if (!$("#comparisonCsv").value) {
      toast("Select a real comparison CSV.", "error");
      return;
    }
    try {
      const report = await apiRequest("/api/comparison/run", {
        method: "POST",
        body: {
          csv_path: $("#comparisonCsv").value,
          group_column: $("#comparisonGroup").value,
          metric: $("#comparisonMetric").value,
          direction: $("#comparisonDirection").value
        }
      });
      $("#comparisonResults").innerHTML = `<div class="live-summary">
        <span><strong>${report.numeric_rows}</strong><small>numeric rows</small></span>
        <span><strong>${escapeHtml(report.best.name)}</strong><small>best model / group</small></span>
        <span><strong>${Number(report.best.value).toExponential(5)}</strong><small>${escapeHtml(report.metric)}</small></span>
        <span><strong>${escapeHtml(report.direction)}</strong><small>ranking direction</small></span>
      </div><div class="live-list" style="margin-top:8px">${report.ranked.slice(0, 25).map(item => `<article class="live-row">
        <span><strong>#${item.rank} · ${escapeHtml(item.name)}</strong><small>source row ${item.index} · ${escapeHtml(report.report_path)}</small></span>
        <span class="chip-row"><span class="chip">${escapeHtml(report.metric)}=${escapeHtml(item.value)}</span></span>
        <span></span><span></span>
      </article>`).join("")}</div>`;
      toast(`Ranked ${report.numeric_rows} actual model-result rows.`);
    } catch (error) {
      toast(`Comparison failed: ${error.message}`, "error");
    }
  });
}

export async function renderExportWorkspace(container) {
  const [artifacts, datasets, checkpoints] = await Promise.all([
    apiRequest("/api/files?kind=artifact"),
    apiRequest("/api/files?kind=dataset"),
    apiRequest("/api/files?kind=checkpoint")
  ]);
  const items = [...artifacts.items, ...datasets.items, ...checkpoints.items]
    .filter((item, index, all) => all.findIndex(candidate => candidate.path === item.path) === index)
    .slice(0, 1200);
  container.innerHTML = `<div class="live-toolbar"><span><strong>Isolated artifact handoff</strong><small>Exports are written only under frontend/runtime/exports; source files are never rewritten.</small></span></div>
    <div class="config-card">
      <label class="config-help">File or directory path</label><input class="config-control" id="exportPath" list="exportPaths" placeholder="output/... or frontend/runtime/..."><datalist id="exportPaths">${items.map(item => `<option value="${escapeHtml(item.path)}"></option>`).join("")}</datalist>
      <label class="config-help">Export label</label><input class="config-control" id="exportLabel" value="ai-cae4all-artifact">
      <button class="button primary" id="runExport" style="margin-top:8px">Create downloadable export</button>
    </div><div id="exportResults"></div>`;
  $("#runExport").addEventListener("click", async () => {
    if (!$("#exportPath").value) {
      toast("Select or enter an existing repository artifact path.", "error");
      return;
    }
    try {
      const result = await apiRequest("/api/export", {
        method: "POST",
        body: { path: $("#exportPath").value, label: $("#exportLabel").value }
      });
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
    if (sectionId === "deploy") await renderDeployWorkspace(container);
    if (sectionId === "optimization") await renderOptimizationWorkspace(container);
    if (sectionId === "evaluation") await renderFieldEvaluationWorkspace(container);
    if (sectionId === "comparison") await renderComparisonWorkspace(container);
    if (sectionId === "export") await renderExportWorkspace(container);
  } catch (error) {
    liveError(container, error);
  }
}
