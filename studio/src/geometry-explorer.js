import { apiRequest } from "./api.js";
import { escapeHtml } from "./dom.js";
import { createRenderer, createFallbackRenderer, turboColor } from "./render3d.js";

const API = "/api/geometry-explorer";
const instances = new WeakMap();
const titles = { car: "Car / DrivAerML", deepjeb: "DeepJEB", mcb: "MCB" };
const descriptions = {
  car: "Vehicle geometry", deepjeb: "Structural bracket", mcb: "Mechanical component"
};
// The slider snaps to a few stops so every move is a visibly different shape.
// Each stop is generated once at full quality (~0.9 s on one GPU) and reused;
// idle time generates the remaining stops of the current endpoint pair.
const STOPS = [0, .25, .5, .75, 1];
const CACHE_LIMIT = 60;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const number = value => Number.isFinite(Number(value)) ? Number(value) : null;
const compact = value => Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : "—";
const alphaLabel = value => Number(value).toFixed(5).replace(/\.?0+$/, "") || "0";
const emptySample = { x: [], y: [], z: [], mesh: { faces: [], edges: [] }, supports: {} };

function flatten(values) {
  return Array.isArray(values?.[0]) ? values.flat() : values || [];
}

function coordinates(mesh) {
  if (Array.isArray(mesh?.vertices?.[0])) return mesh.vertices;
  if (mesh?.vertices?.length) {
    const points = [];
    for (let i = 0; i < mesh.vertices.length; i += 3) points.push(mesh.vertices.slice(i, i + 3));
    return points;
  }
  if (mesh?.x?.length) return mesh.x.map((x, i) => [x, mesh.y[i], mesh.z[i]]);
  return [];
}

function frameFor(mesh) {
  const points = coordinates(mesh);
  const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
  for (const point of points) {
    for (let axis = 0; axis < 3; axis += 1) {
      if (!Number.isFinite(point[axis])) continue;
      low[axis] = Math.min(low[axis], point[axis]);
      high[axis] = Math.max(high[axis], point[axis]);
    }
  }
  const center = low.map((value, i) => Number.isFinite(value) ? (value + high[i]) / 2 : 0);
  let radius = 0;
  for (const point of points) radius = Math.max(radius, Math.hypot(...point.map((value, i) => value - center[i])));
  // Source geometries are z-up. Keep this frame across alpha updates instead
  // of changing axis order or fitting away the generator's size changes.
  return { order: ["x", "z", "y"], signs: [1, -1, 1], center, radius: (radius || 1) * 1.08, planar: false };
}

function asSample(mesh, viewFrame, field = false) {
  const points = coordinates(mesh);
  const values = field ? mesh.values || mesh.u_z || [] : [];
  let stats = mesh.stats;
  if (field && !stats) {
    let min = Infinity, max = -Infinity;
    for (const value of values) {
      if (!Number.isFinite(value)) continue;
      min = Math.min(min, value); max = Math.max(max, value);
    }
    stats = { min: Number.isFinite(min) ? min : 0, max: Number.isFinite(max) ? max : 0 };
  }
  return {
    x: points.map(point => point[0]), y: points.map(point => point[1]), z: points.map(point => point[2]),
    values, stats, viewFrame,
    mesh: { faces: flatten(mesh.faces), edges: flatten(mesh.edges) },
    supports: { mesh: true, field: field && values.length > 0, points: true }
  };
}

function defaultCamera(profile) {
  return { yaw: -.55, pitch: profile === "mcb" ? .6 : .28, zoom: profile === "car" ? 1.35 : 1.1, panX: 0, panY: 0 };
}

/** Mount the real generator workspace into the ordinary Studio DOM. */
export async function mountGeometryExplorer(container, { autoGenerate = true } = {}) {
  if (!container) throw new Error("Geometry Explorer needs a workspace container.");
  closeGeometryExplorer(container);
  const model = {
    disposed: false, profiles: [], profile: "car", clientId: `studio-${globalThis.crypto?.randomUUID?.() || Date.now()}`,
    revision: 0, requestedKey: "", releaseRevision: -1, manualAnalysis: -1, epoch: 0,
    wanted: null, inflight: null, shown: { revision: -1 }, cache: new Map(), serverLatest: null,
    generation: null, mesh: null, graph: null, field: null, geometryFrame: null, analysisFrame: null,
    sample: null, display: "geometry", camera: defaultCamera("car"),
    generationJob: null, analysisJob: null, analysisRevision: -1, pointer: null,
    controller: new AbortController(), renderer: null, resize: null, observer: null, controlsDirty: false, reframeNext: false
  };
  instances.set(container, model);
  container.classList.add("geometry-explorer-host");
  container.innerHTML = `
    <section class="geometry-explorer" aria-label="Geometry Explorer">
      <div class="ge-profiles" role="group" aria-label="Geometry model profile"></div>
      <div class="ge-workbench">
        <div class="ge-scene-column">
          <div class="ge-view-toolbar">
            <div class="ge-segmented" role="group" aria-label="Geometry display">
              <button type="button" data-ge-display="geometry" aria-pressed="true">Geometry</button>
              <button type="button" data-ge-display="graph" aria-pressed="false" disabled>Surface graph</button>
              <button type="button" data-ge-display="field" aria-pressed="false" disabled>Field</button>
            </div>
            <span class="ge-toolbar-end"><span class="ge-runtime" data-ge="runtime">Connecting models</span><button type="button" class="button small" data-ge="fit" title="Fit the current geometry to the viewport">Fit view</button></span>
          </div>
          <div class="ge-viewport" tabindex="0" role="img" aria-label="Generated geometry. Drag to rotate, scroll to zoom, Shift-drag to pan.">
            <canvas class="ge-canvas" aria-label="Generated mesh viewport"></canvas>
            <div class="ge-view-title"><strong data-ge="scene-title">Choose a model</strong><span data-ge="shown-alpha"></span></div>
            <div class="ge-empty" data-ge="empty"><span aria-hidden="true">◇</span><strong>Ready to explore</strong><p>Choose a model and generate a geometry.</p></div>
            <div class="ge-pending" data-ge="pending" hidden><i></i><span>Generating geometry</span></div>
            <div class="ge-legend" data-ge="legend" hidden><span data-ge="legend-title">Z displacement · mm</span><div><span data-ge="legend-low"></span><i data-ge="legend-bar"></i><span data-ge="legend-high"></span></div></div>
            <div class="ge-view-footer"><span data-ge="topology">No geometry generated</span><span>Drag to rotate · Scroll to zoom</span></div>
          </div>
          <div class="ge-interpolation">
            <div class="ge-slider-heading"><label for="ge-alpha-${model.clientId}">Shape interpolation</label>
              <div><span data-ge="alpha-value">α 0</span></div></div>
            <input class="ge-slider" data-ge="alpha" id="ge-alpha-${model.clientId}" type="range" min="0" max="1" step="${1 / (STOPS.length - 1)}" value="0" aria-label="Shape interpolation alpha" disabled>
            <div class="ge-stops" role="group" aria-label="Interpolation stops">${STOPS.map((stop, i) => `<button type="button" data-ge-stop="${stop}" disabled><i></i>${i === 0 ? '<span data-ge="endpoint-a">A</span>' : i === STOPS.length - 1 ? '<span data-ge="endpoint-b">B</span>' : alphaLabel(stop)}</button>`).join("")}</div>
            <p data-ge="slider-note">Every stop is a shape generated by the selected model.</p>
          </div>
          <div class="ge-status" role="status" aria-live="polite"><i data-ge="status-dot"></i><span data-ge="status">Loading available profiles…</span></div>
        </div>
        <aside class="ge-controls" aria-label="Generation and analysis settings">
          <section class="ge-control-section">
            <h4>Generation</h4><p data-ge="profile-description">SDFFlow geometry model</p>
            <label class="ge-form-row">Random seed<input data-ge="seed" type="number" min="0" max="2147483647" step="1" value="42"></label>
            <div class="ge-input-pair"><label class="ge-form-row">Endpoint A<input data-ge="index-a" type="number" min="0" max="15" step="1" value="0"></label><label class="ge-form-row">Endpoint B<input data-ge="index-b" type="number" min="0" max="15" step="1" value="1"></label></div>
            <button type="button" class="button primary ge-generate" data-ge="generate" disabled>Generate geometry</button>
          </section>
          <section class="ge-control-section ge-analysis" data-ge="analysis-section" hidden>
            <div class="ge-section-heading"><h4>Structural response</h4><span>HI-MGN</span></div>
            <label class="ge-check"><input data-ge="auto-analysis" type="checkbox"> Analyze after slider release</label>
            <div class="ge-stages" aria-label="Analysis progress">
              <span data-ge-stage="mesh"><i></i>Surface graph</span><span data-ge-stage="inference"><i></i>HI-MGN prediction</span><span data-ge-stage="result"><i></i>Displacement field</span>
            </div>
            <label class="ge-form-row">Result field<select data-ge="field-select" disabled><option value="u_z">Z displacement (u_z)</option></select></label>
            <button type="button" class="button ge-analyze" data-ge="analyze" disabled>Predict response</button>
            <div class="ge-result-summary" data-ge="result-summary" hidden></div>
            <p class="ge-note">Surrogate prediction on a newly built surface graph.</p>
          </section>
          <details class="ge-details"><summary>Model &amp; generation details</summary><dl data-ge="details"><dt>Model</dt><dd>Loading…</dd></dl></details>
        </aside>
      </div>
    </section>`;
  const q = name => container.querySelector(`[data-ge="${name}"]`);
  const on = (element, event, handler, options = {}) => element.addEventListener(event, handler, { ...options, signal: model.controller.signal });
  const current = revision => !model.disposed && instances.get(container) === model && revision === model.revision && container.isConnected;
  // Generation results are ordered by revision, not discarded when a newer
  // request exists: the epoch only changes when the profile or settings do.
  const alive = epoch => !model.disposed && instances.get(container) === model && epoch === model.epoch && container.isConnected;

  function status(message, kind = "ready") {
    q("status").textContent = message;
    q("status-dot").dataset.kind = kind;
  }
  function pending(message = "") {
    q("pending").hidden = !message;
    if (message) q("pending").querySelector("span").textContent = message;
  }
  function profile() { return model.profiles.find(item => item.id === model.profile); }
  function params() {
    const alpha = number(q("alpha").value);
    const seed = number(q("seed").value), indexA = number(q("index-a").value), indexB = number(q("index-b").value);
    if (alpha === null || alpha < 0 || alpha > 1) throw new Error("Choose an interpolation alpha between 0 and 1.");
    if (![seed, indexA, indexB].every(value => Number.isInteger(value) && value >= 0)) throw new Error("Seed and endpoints must be non-negative integers.");
    const limit = profile()?.default?.source_num_samples || 16;
    if (indexA >= limit || indexB >= limit) throw new Error(`Endpoint indices must be between 0 and ${limit - 1}.`);
    return { profile: model.profile, alpha, seed, index_a: indexA, index_b: indexB, client_id: model.clientId };
  }
  const cacheKey = request => JSON.stringify([request.profile, request.seed, request.index_a, request.index_b, request.alpha]);
  function updateAlpha(value) {
    const alpha = Math.min(1, Math.max(0, Number(value) || 0));
    q("alpha").value = String(alpha); q("alpha-value").textContent = `α ${alphaLabel(q("alpha").value)}`;
    q("alpha").style.setProperty("--ge-alpha", `${alpha * 100}%`);
    renderStops();
  }
  /** Stop chips show which shapes of the current endpoint pair exist already. */
  function renderStops() {
    let base = null;
    try { base = params(); } catch { /* invalid settings: no stop is ready */ }
    const ready = profile()?.available && !model.controlsDirty;
    let generated = 0;
    container.querySelectorAll("[data-ge-stop]").forEach(button => {
      const stop = Number(button.dataset.geStop);
      const key = base && cacheKey({ ...base, alpha: stop });
      const entry = key && model.cache.get(key);
      if (entry?.mesh) generated += 1;
      button.disabled = !ready;
      button.dataset.state = entry?.mesh ? "ready" : key && model.inflight?.key === key ? "busy" : "waiting";
      button.setAttribute("aria-pressed", String(Number(q("alpha").value) === stop));
      button.title = entry?.mesh ? `α ${alphaLabel(stop)} · generated` : `α ${alphaLabel(stop)} · not generated yet`;
    });
    return generated;
  }
  function remember(key, entry) {
    model.cache.delete(key); model.cache.set(key, entry);
    while (model.cache.size > CACHE_LIMIT) model.cache.delete(model.cache.keys().next().value);
  }
  function resetAnalysis() {
    model.graph = null; model.field = null; model.analysisFrame = null;
    model.analysisJob = null; model.analysisRevision = -1;
    q("result-summary").hidden = true;
    container.querySelectorAll("[data-ge-stage]").forEach(item => { item.dataset.state = "waiting"; });
    if (model.display !== "geometry") model.display = "geometry";
    refreshDisplay();
  }
  function refreshDisplay() {
    container.querySelectorAll("[data-ge-display]").forEach(button => {
      const mode = button.dataset.geDisplay;
      button.disabled = mode === "geometry" ? !model.mesh : mode === "graph" ? !model.graph : !model.field;
      button.setAttribute("aria-pressed", String(mode === model.display));
    });
    q("field-select").disabled = !model.field;
    q("analyze").disabled = !model.generation || model.shown.revision !== model.revision || Boolean(model.analysisJob) || Boolean(model.field) || model.controlsDirty;
    draw();
  }
  function draw() {
    if (model.disposed || !model.renderer) return;
    const data = model.display === "graph" ? model.graph : model.display === "field" ? model.field : model.mesh;
    const frame = model.display === "geometry" ? model.geometryFrame : model.analysisFrame;
    if (!data) {
      model.renderer.draw(emptySample, "surface", model.camera);
      q("empty").hidden = false; q("legend").hidden = true; return;
    }
    q("empty").hidden = true;
    if (model.sample?.source !== data || model.sample?.frame !== frame) model.sample = { source: data, frame, sample: asSample(data, frame, model.display === "field") };
    const drawn = model.renderer.draw(model.sample.sample, model.display === "graph" ? "mesh" : model.display === "field" ? "field" : "surface", model.camera);
    const points = coordinates(data).length, faceCount = flatten(data.faces).length / 3, edgeCount = flatten(data.edges).length / 2;
    q("topology").textContent = [`${points.toLocaleString()} nodes`, faceCount ? `${faceCount.toLocaleString()} triangles` : null, model.display === "graph" && edgeCount ? `${edgeCount.toLocaleString()} edges` : null].filter(Boolean).join(" · ");
    q("scene-title").textContent = model.display === "geometry" ? `${titles[model.profile] || profile()?.title || model.profile} · generated geometry` : model.display === "graph" ? "DeepJEB · surface graph" : "DeepJEB · Z displacement";
    q("shown-alpha").textContent = model.generation ? `α ${alphaLabel(model.generation.alpha ?? model.mesh?.alpha ?? q("alpha").value)}` : "";
    q("legend").hidden = model.display !== "field";
    if (model.display === "field") {
      const unit = model.field.unit || "mm";
      q("legend-title").textContent = `Z displacement · ${unit}`;
      q("legend-low").textContent = compact(drawn.domain[0]); q("legend-high").textContent = compact(drawn.domain[1]);
      q("legend-bar").style.background = `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map(stop => `${turboColor(stop)} ${stop * 100}%`).join(",")})`;
    }
  }
  async function poll(job, live, handler, interval = 220) {
    while (live()) {
      const next = await apiRequest(`${API}/jobs?id=${encodeURIComponent(job.id)}`);
      if (!live()) return null;
      handler(next);
      if (next.status === "completed") return next;
      if (next.status === "failed") throw new Error(typeof next.error === "string" ? next.error : next.error?.message || "The native model could not complete this request.");
      if (next.status === "superseded" || next.superseded) return null;
      await sleep(interval);
    }
    return null;
  }
  function generationMessage(job) {
    const phase = String(job.phase || job.status || "").toLowerCase();
    if (job.status === "queued") return "Waiting for the model";
    if (/load|checkpoint/.test(phase)) return "Loading SDFFlow model";
    if (/march|extract|decode|mesh/.test(phase)) return "Building generated surface";
    return "Generating geometry";
  }
  /** Show a generated shape (fresh or reused) as the current geometry. */
  function show(entry, seconds = null) {
    model.mesh = entry.mesh; model.generation = entry.generation;
    model.shown = { revision: model.revision };
    if (!model.geometryFrame || model.reframeNext) model.geometryFrame = frameFor(entry.mesh);
    model.reframeNext = false; model.display = "geometry";
    pending(); refreshDisplay(); updateDetails();
    const generated = renderStops();
    status(`Geometry ready · α ${alphaLabel(entry.generation.alpha)} · ${seconds === null ? "generated earlier" : `${seconds} s`} · ${generated}/${STOPS.length} stops generated`);
  }
  /** One native job at a time, in priority order: the shape the slider asks
   *  for, the requested analysis, then the remaining stops of this pair. */
  function pump() {
    if (model.inflight || model.disposed || model.controlsDirty || !profile()?.available) return;
    const wanted = model.wanted;
    if (wanted) {
      model.wanted = null;
      const key = cacheKey(wanted.request), hit = model.cache.get(key);
      if (!hit) { dispatchGeneration(wanted); return; }
      if (key === model.requestedKey && hit.mesh && model.shown.revision !== model.revision) show(hit);
    }
    if (autoAnalyze() || model.analysisJob) return;
    let base;
    try { base = params(); } catch { return; }
    const next = [...STOPS].sort((a, b) => Math.abs(a - base.alpha) - Math.abs(b - base.alpha))
      .map(alpha => ({ ...base, alpha })).find(request => !model.cache.has(cacheKey(request)));
    if (next) dispatchGeneration({ request: next, prefetch: true });
  }
  /** Returns true while an analysis (or the generation it needs) is underway. */
  function autoAnalyze() {
    if (model.profile !== "deepjeb" || !model.generation || model.shown.revision !== model.revision || model.controlsDirty) return false;
    const asked = model.manualAnalysis === model.revision || (q("auto-analysis").checked && model.releaseRevision === model.revision);
    if (!asked) return false;
    if (model.analysisJob) return true;
    if (model.analysisRevision === model.revision) return false;
    const entry = model.cache.get(model.requestedKey);
    if (entry?.analysis) { showAnalysis(entry.analysis); return false; }
    // The server analyzes only its newest generation; a reused stop is
    // regenerated (same seed, same shape) so the analysis reads its own STL.
    if (model.generation.generation_id !== model.serverLatest) {
      try { dispatchGeneration({ request: params(), refresh: true }); } catch (error) { status(error.message, "error"); return false; }
      return true;
    }
    startAnalysis(model.revision);
    return true;
  }
  async function dispatchGeneration({ request, prefetch = false, refresh = false }) {
    const epoch = model.epoch, key = cacheKey(request);
    model.inflight = { key, prefetch };
    if (!prefetch) {
      model.generationJob = { status: "queued" }; model.controlsDirty = false;
      q("generate").textContent = "Regenerate geometry";
      if (!model.mesh) pending("Generating geometry");
    }
    refreshDisplay(); renderStops();
    try {
      const job = await apiRequest(`${API}/generate`, { method: "POST", body: { ...request, request_revision: model.revision } });
      if (!alive(epoch)) return;
      if (!prefetch) model.generationJob = job;
      const completed = await poll(job, () => alive(epoch), next => {
        if (prefetch && key !== model.requestedKey) return;
        model.generationJob = next;
        const message = refresh ? "Preparing geometry for analysis" : generationMessage(next);
        if (!model.mesh) pending(message);
        status(`${message} · α ${alphaLabel(request.alpha)}`, "busy");
      }, 60);
      if (!alive(epoch) || !completed) return;
      const mesh = completed.result?.mesh;
      if (!coordinates(mesh).length) throw new Error("Generation finished without a viewable mesh.");
      const entry = { mesh, generation: { ...request, ...completed.result, generation_id: completed.result.generation_id || completed.id } };
      remember(key, entry); model.serverLatest = entry.generation.generation_id;
      const seconds = Number(completed.result.timings?.total_s || completed.elapsed_s || 0).toFixed(2);
      if (key === model.requestedKey && (refresh || model.shown.revision !== model.revision)) show(entry, seconds);
      else if (model.shown.revision === model.revision && !model.analysisJob && !model.wanted) {
        status(`Geometry ready · α ${alphaLabel(q("alpha").value)} · ${renderStops()}/${STOPS.length} stops generated`);
      }
    } catch (error) {
      if (!alive(epoch)) return;
      // A failed stop is remembered so idle generation does not retry it forever.
      remember(key, { error: error.message });
      if (key === model.requestedKey) { pending(); status(error.message, "error"); }
    } finally {
      if (alive(epoch)) {
        model.inflight = null; model.generationJob = null;
        refreshDisplay(); renderStops(); pump();
      }
    }
  }
  function scheduleGeneration(released = false, force = false) {
    if (!profile()?.available) return;
    let request;
    try { request = params(); } catch (error) { status(error.message, "error"); return; }
    model.controlsDirty = false;
    const key = cacheKey(request);
    if (force || key !== model.requestedKey) {
      model.revision += 1; model.requestedKey = key;
      resetAnalysis();
    }
    if (released) model.releaseRevision = model.revision;
    if (force) model.cache.delete(key);
    const hit = model.cache.get(key);
    if (hit?.mesh) {
      model.wanted = null;
      if (model.shown.revision !== model.revision) show(hit);
      pump(); return;
    }
    if (hit?.error) { model.wanted = null; status(hit.error, "error"); return; }
    if (!model.mesh) pending("Geometry update requested");
    status(`Generating α ${alphaLabel(request.alpha)}${model.mesh ? " · previous geometry remains visible" : ""}`, "busy");
    model.wanted = { request };
    pump();
  }
  function analysisProgress(job) {
    const phase = String(job.phase || job.status || "").toLowerCase();
    const graph = job.graph || job.progress?.graph || job.result?.graph;
    if (graph && coordinates(graph).length) {
      const first = !model.graph;
      model.graph = graph; model.analysisFrame ||= frameFor(graph);
      if (first) model.display = "graph";
    }
    const finished = job.status === "completed";
    const inference = /infer|read|result/.test(phase) || Boolean(model.graph);
    container.querySelector('[data-ge-stage="mesh"]').dataset.state = inference || finished ? "done" : job.status === "queued" ? "waiting" : "active";
    container.querySelector('[data-ge-stage="inference"]').dataset.state = finished ? "done" : inference ? "active" : "waiting";
    container.querySelector('[data-ge-stage="result"]').dataset.state = finished ? "done" : "waiting";
    const message = job.status === "queued" ? "Waiting for structural model" : inference ? "Running HI-MGN prediction" : "Building surface graph";
    pending(message); status(`${message} · ${Math.floor(job.elapsed_s || 0)} s`, "busy");
    refreshDisplay();
  }
  function showAnalysis(analysis, seconds = null) {
    model.graph = analysis.graph; model.field = analysis.field;
    model.analysisFrame ||= frameFor(model.graph || model.field);
    model.analysisRevision = model.revision;
    container.querySelectorAll("[data-ge-stage]").forEach(item => { item.dataset.state = "done"; });
    model.display = "field"; pending(); refreshDisplay();
    const unit = model.field.unit || "mm", stats = model.field.stats || model.sample?.sample.stats;
    q("result-summary").textContent = `u_z range  ${compact(stats?.min)} to ${compact(stats?.max)} ${unit}`;
    q("result-summary").hidden = false;
    status(`Structural response ready · ${seconds === null ? "predicted earlier for this shape" : `${seconds} s`}`);
    updateDetails();
  }
  async function startAnalysis(revision) {
    const key = model.requestedKey, generationId = model.generation.generation_id;
    model.analysisRevision = revision;
    model.analysisJob = { status: "queued" }; refreshDisplay();
    pending("Building surface graph");
    try {
      const job = await apiRequest(`${API}/analyze`, { method: "POST", body: { generation_id: generationId, client_id: model.clientId, request_revision: model.revision } });
      if (!current(revision)) return;
      model.analysisJob = job;
      const completed = await poll(job, () => current(revision), next => { model.analysisJob = next; analysisProgress(next); });
      if (!current(revision)) return;
      model.analysisJob = null;
      if (!completed) { pending(); refreshDisplay(); return; }
      const analysis = { graph: completed.result?.graph || model.graph, field: completed.result?.field };
      if (!analysis.field?.values?.length && !analysis.field?.u_z?.length) throw new Error("Analysis completed without a displacement field.");
      const entry = model.cache.get(key);
      if (entry?.generation?.generation_id === generationId) entry.analysis = analysis;
      showAnalysis(analysis, Number(completed.elapsed_s || completed.result.timings?.total_s || 0).toFixed(1));
    } catch (error) {
      if (!current(revision)) return;
      model.analysisJob = null; pending(); status(error.message, "error"); refreshDisplay();
    } finally {
      if (!model.disposed) {
        if (current(revision)) model.analysisJob = null;
        pump();
      }
    }
  }
  function updateDetails() {
    const item = profile(), source = model.mesh?.provenance || {};
    const pairs = [["Model", item?.title || titles[model.profile] || model.profile], ["Flow checkpoint", item?.checkpoint], ["VAE checkpoint", item?.vae_checkpoint], ["Interpolation", "Flow noise · spherical interpolation"], ["Seed", model.generation?.seed ?? q("seed").value], ["Endpoints", `${q("index-a").value} → ${q("index-b").value}`], ["Resolution", source.mc_resolution || item?.default?.mc_resolution], ["ODE steps", source.ode_steps || item?.default?.ode_steps], ["Generated STL", model.generation?.stl_path]];
    q("details").innerHTML = pairs.filter(([, value]) => value !== undefined && value !== null).map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd>`).join("");
  }
  function endpointLabels() {
    q("endpoint-a").textContent = `A · ${q("index-a").value}`; q("endpoint-b").textContent = `B · ${q("index-b").value}`;
  }
  function selectProfile(id) {
    const item = model.profiles.find(candidate => candidate.id === id);
    if (!item) return;
    model.epoch += 1; model.revision += 1; model.requestedKey = "";
    model.wanted = null; model.inflight = null; model.shown = { revision: -1 }; model.manualAnalysis = -1;
    model.profile = id; model.generation = null; model.mesh = null; model.geometryFrame = null; model.sample = null;
    model.generationJob = null; model.camera = defaultCamera(id); model.controlsDirty = false;
    resetAnalysis(); updateAlpha(0);
    container.querySelectorAll("[data-ge-profile]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.geProfile === id)));
    q("scene-title").textContent = item.title || titles[id] || id; q("shown-alpha").textContent = "";
    q("profile-description").textContent = descriptions[id] || "Generated geometry";
    q("seed").value = item.default?.seed ?? 42;
    q("index-a").value = item.default?.index_a ?? 0; q("index-b").value = item.default?.index_b ?? 1;
    q("index-a").max = q("index-b").max = (item.default?.source_num_samples || 16) - 1;
    endpointLabels(); renderStops();
    q("alpha").disabled = q("generate").disabled = !item.available;
    q("generate").textContent = "Generate geometry";
    q("analysis-section").hidden = id !== "deepjeb";
    q("topology").textContent = "No geometry generated";
    pending(); refreshDisplay(); updateDetails();
    if (!item.available) {
      status("This profile needs its trained checkpoints before geometry generation.", "error");
      q("slider-note").textContent = (item.missing_checkpoints || []).join(" · ") || "Model checkpoints are unavailable.";
    } else {
      status(`${item.title || titles[id] || id} is ready to generate.`);
      q("slider-note").textContent = `The slider snaps to ${STOPS.length} stops. Each stop is generated once by the model at full quality, then reused, so moving between generated stops is instant.`;
      if (autoGenerate) scheduleGeneration(true);
    }
  }
  try { model.renderer = createRenderer(container.querySelector(".ge-canvas")) || createFallbackRenderer(container.querySelector(".ge-canvas")); }
  catch (error) { status(error.message, "error"); }
  const viewport = container.querySelector(".ge-viewport");
  on(q("alpha"), "input", event => { updateAlpha(event.target.value); scheduleGeneration(false); });
  on(q("alpha"), "change", () => scheduleGeneration(true));
  container.querySelectorAll("[data-ge-stop]").forEach(button => on(button, "click", () => { updateAlpha(button.dataset.geStop); scheduleGeneration(true); }));
  on(q("generate"), "click", () => { model.reframeNext = true; model.camera = defaultCamera(model.profile); scheduleGeneration(true, true); });
  on(q("analyze"), "click", () => { model.manualAnalysis = model.revision; pump(); });
  on(q("field-select"), "change", () => { model.display = "field"; refreshDisplay(); });
  container.querySelectorAll("[data-ge-display]").forEach(button => on(button, "click", () => { model.display = button.dataset.geDisplay; refreshDisplay(); }));
  for (const name of ["seed", "index-a", "index-b"]) on(q(name), "input", () => {
    model.epoch += 1; model.revision += 1; model.requestedKey = ""; model.generationJob = null;
    model.wanted = null; model.inflight = null; model.shown = { revision: -1 };
    model.controlsDirty = true; resetAnalysis(); pending(); refreshDisplay(); endpointLabels(); renderStops();
    status("Settings changed. Generate geometry to apply them.");
  });
  const fitView = () => {
    const data = model.display === "geometry" ? model.mesh : model.display === "graph" ? model.graph : model.field;
    if (!data) return;
    if (model.display === "geometry") model.geometryFrame = frameFor(data); else model.analysisFrame = frameFor(data);
    model.camera = defaultCamera(model.profile); draw();
  };
  on(q("fit"), "click", fitView); on(viewport, "dblclick", fitView);
  on(viewport, "wheel", event => { if (!model.mesh) return; event.preventDefault(); model.camera.zoom = Math.max(.2, Math.min(6, model.camera.zoom * Math.exp(-event.deltaY * .001))); draw(); }, { passive: false });
  on(viewport, "pointerdown", event => {
    if (!model.mesh || ![0, 1, 2].includes(event.button)) return;
    event.preventDefault(); model.pointer = { id: event.pointerId, x: event.clientX, y: event.clientY, pan: event.shiftKey || event.button !== 0 };
    viewport.setPointerCapture(event.pointerId); viewport.classList.add("is-dragging");
  });
  on(viewport, "pointermove", event => {
    const pointer = model.pointer; if (!pointer || pointer.id !== event.pointerId) return;
    const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
    pointer.x = event.clientX; pointer.y = event.clientY;
    if (pointer.pan) { model.camera.panX += dx; model.camera.panY += dy; }
    else { model.camera.yaw += dx * .008; model.camera.pitch = Math.max(-1.45, Math.min(1.45, model.camera.pitch + dy * .008)); }
    draw();
  });
  for (const eventName of ["pointerup", "pointercancel"]) on(viewport, eventName, event => {
    if (model.pointer?.id !== event.pointerId) return;
    viewport.releasePointerCapture(event.pointerId); model.pointer = null; viewport.classList.remove("is-dragging");
  });
  on(viewport, "contextmenu", event => event.preventDefault());
  on(viewport, "keydown", event => {
    if (!model.mesh) return;
    if (event.key === "ArrowLeft") model.camera.yaw -= .08;
    else if (event.key === "ArrowRight") model.camera.yaw += .08;
    else if (event.key === "ArrowUp") model.camera.pitch = Math.max(-1.45, model.camera.pitch - .08);
    else if (event.key === "ArrowDown") model.camera.pitch = Math.min(1.45, model.camera.pitch + .08);
    else if (event.key === "+" || event.key === "=") model.camera.zoom = Math.min(6, model.camera.zoom * 1.12);
    else if (event.key === "-") model.camera.zoom = Math.max(.2, model.camera.zoom / 1.12);
    else if (event.key === "0") { fitView(); event.preventDefault(); return; }
    else return;
    event.preventDefault(); draw();
  });
  model.resize = new ResizeObserver(draw); model.resize.observe(viewport);
  model.observer = new MutationObserver(() => { if (!container.isConnected || !container.querySelector(".geometry-explorer")) closeGeometryExplorer(container); });
  model.observer.observe(document.body, { childList: true, subtree: true });
  try {
    const payload = await apiRequest(`${API}/profiles`);
    if (model.disposed || !container.isConnected) return model;
    model.profiles = payload.profiles || [];
    q("runtime").textContent = "Native SDFFlow";
    container.querySelector(".ge-profiles").innerHTML = model.profiles.map(item => `<button type="button" class="ge-profile" data-ge-profile="${escapeHtml(item.id)}" aria-pressed="false"><span class="ge-profile-icon" aria-hidden="true">${item.id === "car" ? "▱" : item.id === "deepjeb" ? "⌑" : "◉"}</span><span><strong>${escapeHtml(titles[item.id] || item.title || item.id)}</strong><small>${escapeHtml(descriptions[item.id] || "Geometry model")}</small></span><i class="${item.available ? "" : "unavailable"}" title="${item.available ? "Checkpoints available" : "Checkpoints unavailable"}"></i></button>`).join("");
    container.querySelectorAll("[data-ge-profile]").forEach(button => on(button, "click", () => selectProfile(button.dataset.geProfile)));
    const first = model.profiles.find(item => item.available) || model.profiles[0];
    if (first) selectProfile(first.id); else status("No geometry model profiles are configured.", "error");
  } catch (error) {
    if (!model.disposed) { q("runtime").textContent = "Runtime unavailable"; status(error.message, "error"); }
  }
  return model;
}

export const openGeometryExplorer = mountGeometryExplorer;

/** Stop background polling and free the viewport when leaving this workspace. */
export function closeGeometryExplorer(container) {
  const model = instances.get(container); if (!model) return;
  model.disposed = true; model.revision += 1; model.epoch += 1;
  model.controller.abort(); model.resize?.disconnect(); model.observer?.disconnect(); model.renderer?.dispose();
  instances.delete(container); container.classList.remove("geometry-explorer-host");
}
