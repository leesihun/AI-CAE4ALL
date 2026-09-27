import { $, $$, escapeHtml, toast } from "./dom.js";
import { state } from "./state.js";
import { BLOCK_SPECS } from "./constants.js";
import { apiRequest, requireRuntime, refreshNavCounts } from "./api.js";
import { graphErrorNodes, validateGraph, executableSteps, preflightConfigText, preflightMessages, inferenceDatasetWarnings } from "./validate.js";
import { render } from "./graph.js";
import { jumpToFailingField } from "./config.js";
import { schedulePipelineSave, pipelineDocument } from "./persistence.js";

// "interrupted" is what recovery records for a run whose launcher died with
// the server. Missing from the terminal list, a polled job that ended that way
// was polled forever and its blocks kept showing "running".
const REJECTED_STATUSES = ["failed", "cancelled", "interrupted"];
const ACTIVE_STATUSES = ["queued", "running"];
const isFinished = job => !ACTIVE_STATUSES.includes(job.status);

/**
 * Whether the user wants the drawer collapsed when nothing else is covering it.
 *
 * The drawer is `position: fixed` in the bottom-right corner at z-index 110,
 * deliberately above the modals (80) so a running job stays visible while you
 * browse. The cost is that its 360px body physically covers the bottom-right of
 * every modal -- the artifact viewer's timeline scrubber, the config modal's
 * footer buttons. Making it click-through fixed reachability but not the fact
 * that you simply cannot see what is underneath.
 *
 * So the collapsed state is the OR of two independent inputs: what the user
 * asked for, and whether a modal currently needs the corner. Collapsing to the
 * header keeps the job title, status dot, and controls on screen while giving
 * the modal its content back; closing the modal restores the user's choice.
 */
let drawerCollapsePreference = false;

export function setDrawerCollapsed(collapsed) {
  drawerCollapsePreference = collapsed;
  applyDrawerCollapse();
}

/**
 * Mirror the drawer's geometry onto <body> so the canvas can get out of its way.
 *
 * The drawer is fixed to the viewport's bottom-right corner and the zoom cluster
 * is absolutely positioned in the same corner of the stage, so the drawer used to
 * sit on top of +, −, and "fit" for the whole duration of a run -- exactly when
 * you most want to zoom in on the block that is executing. The two classes here
 * let CSS lift the cluster by the drawer's actual height, and drop it back to a
 * pill's worth of clearance once the drawer is minimized.
 */
function syncDrawerBodyState(drawer, collapsed) {
  const open = drawer.classList.contains("open");
  document.body.classList.toggle("drawer-open", open);
  document.body.classList.toggle("drawer-mini", open && collapsed);
}

export function applyDrawerCollapse() {
  const drawer = $("#runtimeDrawer");
  const button = $("#runtimeMinimize");
  if (!drawer || !button) return;
  const modalOpen = $$(".overlay.open").length > 0;
  const collapsed = drawerCollapsePreference || modalOpen;
  drawer.classList.toggle("minimized", collapsed);
  drawer.classList.toggle("modal-yielded", modalOpen && !drawerCollapsePreference);
  syncDrawerBodyState(drawer, collapsed);
  button.textContent = drawerCollapsePreference ? "+" : "−";
  button.setAttribute("aria-expanded", String(!drawerCollapsePreference));
  const label = drawerCollapsePreference ? "Expand runtime log" : "Minimize runtime log";
  button.setAttribute("aria-label", label);
  button.title = modalOpen && !drawerCollapsePreference
    ? "Runtime log is collapsed while a dialog is open"
    : label;
}

export function toggleDrawerCollapsed() {
  setDrawerCollapsed(!drawerCollapsePreference);
}

/**
 * Re-evaluate the collapse whenever any overlay opens or closes. Watching the
 * class attribute rather than patching every open/close call site means a
 * future modal gets the behaviour for free.
 */
export function watchModalsForDrawer() {
  const observer = new MutationObserver(applyDrawerCollapse);
  $$(".overlay").forEach(overlay => observer.observe(overlay, { attributes: true, attributeFilter: ["class"] }));
  applyDrawerCollapse();
}

export function renderRuntimeJob(job, { reveal = true } = {}) {
  state.api.activeJob = job;
  if (reveal) {
    $("#runtimeDrawer").classList.add("open");
    setDrawerCollapsed(false);
  }
  const rejected = REJECTED_STATUSES.includes(job.status);
  $("#runtimeJobTitle").textContent = job.label || "AI-CAE4ALL job";
  $("#runtimeJobTitle").classList.toggle("status-failed", rejected);
  $("#runtimeJobMeta").textContent = `${job.status} · job ${job.id || "pending"}${job.pid ? ` · PID ${job.pid}` : ""}`;
  $("#runtimeStatusDot").className = `runtime-status-dot ${job.status || ""}`;
  $("#runtimeStep").textContent = job.total_steps
    ? `Step ${job.current_step || 0}/${job.total_steps}${job.step_label ? ` · ${job.step_label}` : ""}`
    : "Preparing";
  const logEl = $("#runtimeLog");
  if (job.diagnostics?.length) {
    logEl.innerHTML = job.diagnostics.map((item, index) => {
      const severity = item.severity === "error" ? "error" : item.severity === "warning" ? "warn" : "";
      const clickable = item.nodeId ? " diagnostic-clickable" : "";
      // A notice (e.g. STEP-ANALYSIS) has nothing to fix; the row still opens
      // the block, so say that instead of promising a fix.
      const jumpLabel = severity ? "Fix now →" : "Open block →";
      return `<div class="diagnostic ${severity}${clickable}" data-jump-diagnostic="${index}"><i></i><span>${item.stepLabel ? `<strong>${escapeHtml(item.stepLabel)}</strong> · ` : ""}[${escapeHtml(item.code || "")}]${item.field ? ` ${escapeHtml(item.field)}:` : ""} ${escapeHtml(item.message || "")}${item.hint ? ` <em>Hint: ${escapeHtml(item.hint)}</em>` : ""}${item.nodeId ? ` <b class="diagnostic-jump">${jumpLabel}</b>` : ""}</span></div>`;
    }).join("");
    $$("[data-jump-diagnostic]", logEl).forEach(row => row.addEventListener("click", () => {
      const item = job.diagnostics[Number(row.dataset.jumpDiagnostic)];
      if (!item?.nodeId) return;
      setDrawerCollapsed(true);
      jumpToFailingField(item.nodeId, item.field);
    }));
  } else {
    logEl.textContent = job.log || "Waiting for launcher output…";
  }
  // Tailing is right for a streaming log, where the newest line matters most.
  // A diagnostics list is the opposite: the first error is the one to act on,
  // and tailing it opened the drawer already scrolled past the only row that
  // explained the failure. Bring the first error into view instead.
  const firstError = logEl.querySelector(".diagnostic.error");
  if (firstError) firstError.scrollIntoView({ block: "nearest" });
  else logEl.scrollTop = logEl.scrollHeight;
  $("#runtimeCancel").disabled = !["queued", "running"].includes(job.status);
  // A Validate result is rendered through this same drawer as a synthetic
  // "preflight" job, but it is not a process: re-showing the banner here left
  // "Real job · completed" with a spinner and a live Stop button on the canvas
  // after every Validate, with nothing ever clearing it.
  if (job.id === "preflight") return;
  if (job.status === "running" || job.status === "queued") {
    $("#runBanner").classList.add("show");
    $("#runTitle").textContent = `Real job · ${job.status}`;
    $("#runDetail").textContent = job.step_label || "preflight → native launcher";
  } else if (REJECTED_STATUSES.includes(job.status) || job.status === "completed") {
    // The banner used to be written only while a job was live, so after a run
    // ended it kept claiming "Real job · running · MeshGraphNets · train" over a
    // job that had already failed two steps later. Only the toast was correct.
    $("#runBanner").classList.add("show");
    $("#runTitle").textContent = `Real job · ${job.status}`;
    $("#runDetail").textContent = job.total_steps
      ? `step ${job.current_step || 0}/${job.total_steps}${job.step_label ? ` · ${job.step_label}` : ""}`
      : job.step_label || "";
  }
}

export function dismissRuntimeJob() {
  const job = state.api.activeJob;
  if (job && ["queued", "running"].includes(job.status)) {
    toast("Stop the active process before dismissing it.", "warn");
    return;
  }
  state.api.activeJob = null;
  $("#runtimeDrawer").classList.remove("open", "minimized");
  document.body.classList.remove("drawer-open", "drawer-mini");
  $("#runtimeJobTitle").textContent = "No active job";
  $("#runtimeJobTitle").classList.remove("status-failed");
  $("#runtimeJobMeta").textContent = "Actual launcher output appears here.";
  $("#runtimeStatusDot").className = "runtime-status-dot";
  $("#runtimeStep").textContent = "Idle";
  $("#runtimeCancel").disabled = true;
  setDrawerCollapsed(false);
  $("#runtimeLog").textContent = "Connect with START_STUDIO.bat to enable the real AI-CAE4ALL runtime.";
}

/**
 * Whether a job's steps are blocks of the graph on the canvas.
 *
 * Node ids are only unique within one graph, and the templates reuse them
 * ("trainer", "export", ...), so a job rejoined from another pipeline used to
 * paint its status onto whatever block shared the id: an SDFFlow training run
 * marked a fresh HI-MGN graph's MeshGraphNets block "Running" and bound its
 * Train Metrics to the SDFFlow job. A step whose id names a block of a
 * different type proves the job came from another graph. A missing block does
 * not: deleting one while its run is live must not freeze the others.
 */
/**
 * Whether this canvas owns a job, when that is knowable: true or false for a
 * job that recorded the canvas it was launched from (or was reopened here from
 * Runs), null for an untagged job from before canvases had ids.
 */
export function canvasOwnership(jobId, jobCanvasId) {
  if (state.ownedJobs.has(jobId)) return true;
  // Node ids cannot separate two canvases made from one template, so a job
  // tagged with the canvas that launched it is owned by that canvas alone.
  const canvasId = jobCanvasId || state.api.launchedJobs.get(jobId);
  return canvasId ? canvasId === state.canvasId : null;
}

function jobBelongsToCanvas(job) {
  const owned = canvasOwnership(job.id, job.canvas_id);
  if (owned !== null) return owned;
  // A legacy canvas claims an untagged run by node ids only while the run is
  // in flight. Node ids cannot tell its own finished runs from those of every
  // other canvas built on the same template, and claiming them all replayed the
  // newest untagged run of any such pipeline onto this one (its results paths
  // included, then saved). A claimed run is recorded as owned, so it is still
  // this canvas's once it finishes.
  if (!state.legacyCanvas || isFinished(job)) return false;
  const steps = (job.steps || []).filter(step => step.node_id);
  let matched = 0;
  for (const step of steps) {
    const node = state.nodes.find(item => item.id === step.node_id);
    if (!node) continue;
    if (step.node_type && node.type !== step.node_type) return false;
    matched += 1;
  }
  if (!matched) return false;
  state.ownedJobs.add(job.id);
  schedulePipelineSave();
  return true;
}

// Keys a run writes onto its blocks, which a later workspace action (Evaluate,
// Export) may overwrite with the user's own evidence.
const RUN_EVIDENCE_KEYS = ["results_path", "results_samples", "results_dir", "report_path",
  "export_path", "evaluated_samples"];

function metricsBlocksFedBy(nodeId) {
  return state.edges
    .filter(edge => edge.fromNode === nodeId && edge.fromPort === "metrics")
    .map(edge => state.nodes.find(node => node.id === edge.toNode))
    .filter(node => node?.type === "evaluate.training_metrics");
}

function carriesRunEvidence(node) {
  return RUN_EVIDENCE_KEYS.some(key => String(node.config?.[key] ?? "") !== "")
    || metricsBlocksFedBy(node.id).some(metrics => metrics.config?.job_id);
}

/** Whether what a block carries is exactly what this run's step wrote. */
function carriesOutputsOf(node, step, job) {
  const paths = [step.results, step.results_dir].filter(Boolean);
  return paths.some(path => RUN_EVIDENCE_KEYS.some(key => node.config?.[key] === path))
    || metricsBlocksFedBy(node.id).some(metrics => metrics.config?.job_id === job.id);
}

/** Marks every unmarked block as having received no run yet, so each finished
 * run it belongs to is delivered on the next replay. For a document known to
 * predate the runs (a run's own launch snapshot, reopened from Runs). */
export function markBlocksUndelivered() {
  state.nodes.forEach(node => { if (!node.resultsFrom) node.resultsFrom = { job: "", at: "" }; });
}

/**
 * Whether a replayed finished run still has outputs to deliver to a block:
 * only a run newer than the last one whose outputs the block received. Its
 * status is always repainted; its paths are not, so a report, an export or a
 * Train Metrics run chosen after that run survives every page load and undo.
 */
function runIsUndelivered(node, job) {
  const from = node.resultsFrom;
  if (!from) return true;
  if (from.job === job.id) return false;
  return String(job.finished_at || "") > String(from.at || "");
}

/**
 * Paint one job's state onto the blocks of the canvas it belongs to: status,
 * progress, the Train Metrics binding, and each step's results. Touches
 * nothing else, so a finished run can be painted without being tracked.
 * Returns whether any block of this canvas was painted.
 */
function paintJobOntoCanvas(job, { replay = false } = {}) {
  const terminal = isFinished(job);
  const ownsCanvas = jobBelongsToCanvas(job);
  const exactNodeIds = new Set(ownsCanvas ? (job.steps || []).map(step => step.node_id).filter(Boolean) : []);
  const hasNodeIds = (job.steps || []).some(step => step.node_id);
  // Which step index each block is, so a failure can say where it stopped.
  // Collapsing every block to "idle" on failure threw that away: after a run
  // that trained for 25 minutes and inferred 87 rollouts before the evaluation
  // step failed, the canvas showed three untouched blocks and no clue which one
  // broke -- while the job record knew exactly.
  const stepIndexByNode = new Map(
    (job.steps || []).map((step, index) => [step.node_id, index + 1]).filter(([id]) => id)
  );
  const failedAt = Number(job.current_step || 0);
  state.nodes.forEach(node => {
    const label = BLOCK_SPECS[node.type]?.label;
    const legacyMatch = !hasNodeIds && label && job.steps?.some(step => step.label?.startsWith(label));
    if (exactNodeIds.has(node.id) || legacyMatch) {
      const position = stepIndexByNode.get(node.id) || 0;
      if (job.status === "running") {
        node.status = "running";
        node.progress = 58;
      } else if (job.status === "completed") {
        node.status = "complete";
        node.progress = 100;
      } else if (terminal && position && failedAt) {
        // Steps before the stopping point really did finish; the stopping one is
        // the failure; anything after it never started.
        node.status = position < failedAt ? "complete" : position === failedAt ? "failed" : "idle";
        node.progress = position < failedAt ? 100 : 0;
      } else {
        node.status = "idle";
        node.progress = 0;
      }
    }
  });
  // The blocks this run may write its outputs onto: all of them live, and on a
  // replay only those it has not already reached (runIsUndelivered).
  const delivers = new Set([...exactNodeIds].filter(id => {
    const node = state.nodes.find(item => item.id === id);
    return node && (!replay || runIsUndelivered(node, job));
  }));
  delivers.forEach(sourceNodeId => {
    metricsBlocksFedBy(sourceNodeId).forEach(node => { node.config.job_id = job.id; });
  });
  // The backend resolves where each step actually wrote its predictions (the
  // epoch-numbered directory is not derivable from the config). Carry it onto
  // the block so Inspect opens this run's own results instead of guessing at
  // whatever prediction file happens to be lying around the repository.
  (job.steps || []).forEach(step => {
    if (!ownsCanvas || !(step.results || step.results_dir) || !step.node_id) return;
    if (!delivers.has(step.node_id)) return;
    const node = state.nodes.find(item => item.id === step.node_id);
    if (!node) return;
    if (step.kind === "analysis") {
      // An evaluation writes a report, an export writes an archive; both belong
      // on the block so the canvas shows the evidence and the next block
      // downstream can read it without the user re-entering a path.
      if (node.type === "evaluate.predictions") {
        node.config.report_path = step.results;
        const scored = step.analysis?.evaluated_samples;
        if (scored != null) node.config.evaluated_samples = String(scored);
      } else if (node.type === "output.export") {
        node.config.export_path = step.results;
      } else {
        node.config.results_path = step.results;
      }
      return;
    }
    if (step.results) {
      node.config.results_path = step.results;
      node.config.results_samples = String(step.results_samples ?? "");
    } else {
      // A generator run that wrote geometry but no table: the previous run's
      // table must not stay on the block as if it described this one.
      delete node.config.results_path;
      delete node.config.results_samples;
    }
    // A CAD Generator's whole output folder (STLs, report, figures), beside
    // the table the Optimization block reads. Cleared when absent for the same
    // reason.
    if (step.results_dir) node.config.results_dir = step.results_dir;
    else delete node.config.results_dir;
  });
  // A finished run's outputs are final: record them as the block's latest.
  if (terminal) {
    delivers.forEach(id => {
      const node = state.nodes.find(item => item.id === id);
      if (node) node.resultsFrom = { job: job.id, at: String(job.finished_at || "") };
    });
  }
  return exactNodeIds.size > 0;
}

export function applyJobStatus(job) {
  const terminal = isFinished(job);
  // Remembered so a /api/jobs listing fetched before this poll cannot bring
  // the run back as in flight (reconcileFinishedJobs).
  if (terminal) state.api.finishedJobs.set(job.id, job);
  if (paintJobOntoCanvas(job)) schedulePipelineSave();
  if (terminal) state.api.trackedJobs.delete(job.id);
  else state.api.trackedJobs.set(job.id, job);
  state.running = state.api.trackedJobs.size > 0;
  // Several jobs may be polling at once; only the focused one owns the drawer.
  if (!state.api.activeJob || state.api.activeJob.id === job.id) {
    renderRuntimeJob(job, { reveal: false });
  }
  renderActiveJobCount();
  render({ background: true });
  if (!terminal) return;
  if (!state.api.trackedJobs.size) {
    window.clearInterval(state.api.pollTimer);
    state.api.pollTimer = null;
    $("#runBanner").classList.remove("show");
  }
  $("#savedState").textContent = `Job ${job.status} · ${job.finished_at || "now"}`;
  refreshNavCounts();
  toast(
    job.status === "completed"
      ? `Completed: ${job.label || "AI-CAE4ALL job"}.`
      : `${job.label || "Job"} ${job.status}. Open the runtime log for details.`,
    job.status === "completed" ? "" : "error"
  );
}

/**
 * Repaint the canvas from every run it owns.
 *
 * Only live jobs are polled, so a run that finished while the page was closed
 * -- or before its graph was reopened from Runs, imported, or brought back by
 * undo -- never reached the canvas: its blocks stayed idle and its results
 * table never landed on the block Optimization and Export read from. Block
 * status is derived from runs alone, so it is reset and rebuilt: finished runs
 * oldest first, so the latest run of each block is the one left showing, then
 * the runs still in flight. A replayed run writes its paths only onto blocks it
 * has not reached yet (runIsUndelivered): a report, export or Train Metrics run
 * the user chose after it is theirs, not the replay's. No toasts; a finished
 * run was already announced. `jobs` is a /api/jobs listing when the caller
 * already has one.
 */
export async function reconcileFinishedJobs(jobs = null) {
  if (!state.api.connected) return;
  const canvasId = state.canvasId;
  let items = jobs;
  if (!Array.isArray(items)) {
    try {
      items = (await apiRequest("/api/jobs")).items || [];
    } catch {
      return;
    }
  }
  // The canvas changed while the listing was in flight: its own call follows.
  if (state.canvasId !== canvasId) return;
  // A run the poller saw finish while the listing was in flight is still
  // "running" in it; its terminal record is the true one. Taken at face value
  // it was polled again, repainted as running and announced a second time.
  items = items.map(job => state.api.finishedJobs.get(job.id) || job);
  const owned = items.filter(job => jobBelongsToCanvas(job));
  const finished = owned
    .filter(job => isFinished(job) && !state.api.trackedJobs.has(job.id))
    .sort((left, right) => String(left.finished_at || "").localeCompare(String(right.finished_at || "")));
  // A document saved before blocks recorded the run they last received. A
  // block holding exactly what one of these runs wrote was delivered by that
  // run (the newest such), so only later runs deliver to it; a block holding
  // anything else holds the user's own choice, made after every run here.
  const seeds = new Map();
  finished.forEach(job => (job.steps || []).forEach(step => {
    const node = step.node_id && state.nodes.find(item => item.id === step.node_id);
    if (!node || node.resultsFrom || !carriesRunEvidence(node)) return;
    const seed = seeds.get(node) || { match: null, newest: null };
    seed.newest = job;
    if (carriesOutputsOf(node, step, job)) seed.match = job;
    seeds.set(node, seed);
  }));
  seeds.forEach(({ match, newest }, node) => {
    const job = match || newest;
    node.resultsFrom = { job: job.id, at: String(job.finished_at || "") };
  });
  state.nodes.forEach(node => {
    node.status = "idle";
    node.progress = 0;
  });
  let painted = seeds.size > 0;
  finished.forEach(job => { painted = paintJobOntoCanvas(job, { replay: true }) || painted; });
  // In-flight runs last, from the freshest record there is. One started from
  // another tab of this canvas is not polled yet, so start polling it.
  owned.filter(job => !isFinished(job) && !state.api.trackedJobs.has(job.id))
    .forEach(job => beginCommandJob(job, { focus: false }));
  state.api.trackedJobs.forEach(job => { paintJobOntoCanvas(job); });
  if (painted) schedulePipelineSave();
  render();
}

/** Shows how many other pipelines are still running behind the focused one. */
export function renderActiveJobCount() {
  const badge = $("#runtimeOtherJobs");
  if (!badge) return;
  const others = [...state.api.trackedJobs.keys()].filter(id => id !== state.api.activeJob?.id).length;
  badge.textContent = others ? `+${others} running` : "";
  badge.style.display = others ? "" : "none";
}

export async function pollActiveJob() {
  const ids = [...state.api.trackedJobs.keys()];
  if (!ids.length) {
    window.clearInterval(state.api.pollTimer);
    state.api.pollTimer = null;
    return;
  }
  for (const jobId of ids) {
    try {
      const job = await apiRequest(`/api/jobs/${encodeURIComponent(jobId)}`);
      applyJobStatus(job);
    } catch (error) {
      // One unreachable job must not stop the others from being polled.
      state.api.trackedJobs.delete(jobId);
      toast(`Job polling failed: ${error.message}`, "error");
    }
  }
}

export function beginCommandJob(job, { focus = true } = {}) {
  state.api.trackedJobs.set(job.id, job);
  state.running = true;
  if (focus) renderRuntimeJob(job);
  renderActiveJobCount();
  if (!state.api.pollTimer) state.api.pollTimer = window.setInterval(pollActiveJob, 900);
  refreshNavCounts();
}

/** Shown once a full submission preflight passes, in both the log and the
 * diagnostics list, so the two renderings cannot drift apart. */
const PREFLIGHT_PASS_NOTE = "Every native step will receive a full filesystem, dataset, environment, and native launch gate against its exact saved config immediately before it starts.";

export async function validatePipeline(targetId = null) {
  const errors = validateGraph(false);
  if (errors.length) {
    // A graph error used to be a toast and nothing else: it vanished after a
    // few seconds, named only the first of possibly several problems, and could
    // not be clicked. Config errors, meanwhile, got a persistent drawer with a
    // "Fix now" that opens the offending block. Same treatment for both.
    toast(`Graph validation failed: ${errors[0]}`, "error");
    renderRuntimeJob({
      id: "preflight",
      label: `${$("#pipelineName").value} · preflight`,
      status: "failed",
      current_step: 0,
      total_steps: 0,
      diagnostics: errors.map((message, index) => ({
        severity: "error",
        code: "GRAPH-001",
        stepLabel: "Pipeline graph",
        nodeId: graphErrorNodes[index] || "",
        message,
        hint: "Fix the block's inputs on the canvas, then validate again."
      })),
      log: errors.map(message => `GRAPH ${message}`).join("\n")
    });
    return false;
  }
  if (!requireRuntime()) return false;
  const steps = executableSteps(targetId);
  if (!steps.length) {
    toast("This graph has no executable model or inference step.", "error");
    return false;
  }
  // Predicting the training set passes preflight cleanly and reports excellent
  // metrics, so it has to be called out here or it never gets noticed.
  inferenceDatasetWarnings().forEach(message => toast(message, "warn"));
  $("#runBanner").classList.add("show");
  $("#runTitle").textContent = "Submission preflight";
  const lines = [];
  // Validate is the button pressed *before* spending GPU hours, so its findings
  // deserve the same treatment a real run's do. renderRuntimeJob already turns
  // structured diagnostics into rows that name the failing block and jump
  // straight to the offending field; this used to hand it a flat text blob
  // instead, so a failed check said only "failed" and left the user reading a
  // log to work out which of seven blocks to fix.
  const diagnostics = [];
  let passed = true;
  for (let index = 0; index < steps.length; index += 1) {
    const step = steps[index];
    $("#runDetail").textContent = `Checking ${index + 1}/${steps.length} · ${step.label}`;
    if (step.kind === "analysis") {
      // Analysis steps carry no flat config, so the launcher's preflight has
      // nothing to parse. Report what they will read instead of pretending a
      // check ran; an unresolved @results reference is reported by the backend
      // at execution time, when the producing step has actually written.
      const inputs = Object.entries(step.payload || {})
        .filter(([key]) => key === "path" || key.endsWith("_path"))
        .map(([, value]) => value)
        .filter(Boolean);
      lines.push(`SKIP ${step.label}: analysis step · reads ${inputs.join(", ") || "graph output"}`);
      diagnostics.push({
        severity: "notice", code: "STEP-ANALYSIS", stepLabel: step.label, nodeId: step.nodeId,
        message: `Analysis step - reads ${inputs.join(", ") || "graph output"}`
      });
      continue;
    }
    const result = await preflightConfigText(step.config, step.label, {
      skipFilesystem: index > 0,
      skipNative: index > 0
    });
    if (!result) {
      passed = false;
      diagnostics.push({
        severity: "error", code: "PREFLIGHT-UNREACHABLE", stepLabel: step.label, nodeId: step.nodeId,
        message: "The launcher preflight could not be reached for this step.",
        hint: "Check that the Studio runtime is still running, then validate again."
      });
      break;
    }
    const summary = result.report?.summary || { errors: 1, warnings: 0, notices: 0 };
    lines.push(`${result.ok ? "PASS" : "FAIL"} ${step.label}: ${summary.errors} errors, ${summary.warnings} warnings, ${summary.notices} notices${index > 0 ? " · dependency checks deferred until this step launches" : ""}`);
    result.report?.diagnostics?.forEach(item => {
      lines.push(`  [${item.code}] ${item.message}`);
      // step.nodeId is what makes the row clickable: without it the diagnostic
      // renders, but "Fix now" has nowhere to go.
      diagnostics.push({ ...item, stepLabel: step.label, nodeId: step.nodeId });
    });
    if (!result.ok) passed = false;
  }
  $("#runBanner").classList.remove("show");
  if (passed) {
    lines.push("\n" + PREFLIGHT_PASS_NOTE);
    diagnostics.push({ severity: "notice", code: "PREFLIGHT-PASS", message: PREFLIGHT_PASS_NOTE });
  }
  renderRuntimeJob({
    id: "preflight",
    label: `${$("#pipelineName").value} · preflight`,
    status: passed ? "completed" : "failed",
    current_step: steps.length,
    total_steps: steps.length,
    diagnostics,
    log: lines.join("\n")
  });
  toast(passed ? "Submission checks passed; every native step will be fully rechecked at launch." : "Preflight failed. Read the real diagnostics.", passed ? "" : "error");
  return passed;
}

export async function runGraph(targetId = null) {
  // Concurrent pipelines are supported: the backend already gives each job its
  // own thread, so nothing here blocks a second submission.
  const errors = validateGraph(false);
  if (errors.length) {
    toast(`Cannot run: ${errors[0]}`, "error");
    return;
  }
  if (!requireRuntime()) return;
  const steps = executableSteps(targetId);
  if (!steps.length) {
    toast("This graph has no executable model or inference step.", "error");
    return;
  }
  const preview = steps.map((step, index) => `${index + 1}. ${step.label}`).join("\n");
  if (!window.confirm(`Execute the real AI-CAE4ALL launcher?\n\n${preview}\n\nThis may use CUDA, write checkpoints, and run for a long time. Submission checks run now, and each native step is fully revalidated against its saved config immediately before launch.`)) return;
  state.running = true;
  $("#runBanner").classList.add("show");
  $("#runTitle").textContent = "Submitting real pipeline";
  $("#runDetail").textContent = "submission checks → per-step launch gate → native process";
  try {
    // Saved with the run so the exact graph can be reloaded from Runs later.
    // Its canvas id is read now: the user can switch canvases while the
    // request is in flight, and the run belongs to the one it was launched from.
    const pipeline = pipelineDocument();
    const launchCanvas = pipeline.canvas_id;
    const job = await apiRequest("/api/pipeline/run", {
      method: "POST",
      allowError: true,
      body: {
        label: $("#pipelineName").value,
        strict: false,
        target_node_id: targetId || "",
        pipeline,
        steps: steps.map(step => ({
          label: step.label,
          kind: step.kind || "launcher",
          config: step.config || "",
          action: step.action || "",
          payload: step.payload || null,
          node_id: step.nodeId,
          node_type: state.nodes.find(node => node.id === step.nodeId)?.type || ""
        }))
      }
    });
    if (!job.httpOk) {
      state.running = state.api.trackedJobs.size > 0;
      if (!state.running) $("#runBanner").classList.remove("show");
      const failures = job.failures || [];
      const diagnostics = failures.length
        ? failures.flatMap(failure => (failure.preflight?.report?.diagnostics || []).map(item => ({
            ...item,
            nodeId: steps[failure.step]?.nodeId,
            stepLabel: failure.label
          })))
        : [{ severity: "error", code: "REQUEST", message: job.error || "The pipeline request failed." }];
      renderRuntimeJob({
        id: "rejected",
        label: `${$("#pipelineName").value} · rejected`,
        status: "failed",
        current_step: 0,
        total_steps: steps.length,
        log: (job.failures?.flatMap(failure => preflightMessages(failure.preflight).map(item => item.text)) || [job.error]).join("\n"),
        diagnostics
      });
      toast("Pipeline was not started because real preflight failed. Click a diagnostic to fix it.", "error");
      const firstFix = diagnostics.find(item => item.nodeId && item.field);
      if (firstFix) {
        window.setTimeout(() => {
          setDrawerCollapsed(true);
          jumpToFailingField(firstFix.nodeId, firstFix.field);
        }, 80);
      }
      return;
    }
    state.api.launchedJobs.set(job.id, job.canvas_id || launchCanvas);
    beginCommandJob(job);
  } catch (error) {
    state.running = state.api.trackedJobs.size > 0;
    if (!state.running) $("#runBanner").classList.remove("show");
    toast(`Could not start the real pipeline: ${error.message}`, "error");
  }
}

export async function stopRun() {
  // Stops the job the drawer is focused on, which with concurrent runs is not
  // necessarily the only active one.
  const jobId = state.api.activeJob?.id;
  if (!jobId || ["preflight", "rejected"].includes(jobId)) return;
  if (!state.api.trackedJobs.has(jobId)) {
    toast("That job has already finished.", "warn");
    return;
  }
  if (!window.confirm(`Stop job ${jobId} and its child model processes?`)) return;
  try {
    const job = await apiRequest(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: "POST",
      body: {}
    });
    applyJobStatus(job);
    toast("Stop requested for the real process tree.", "warn");
  } catch (error) {
    toast(`Could not stop job: ${error.message}`, "error");
  }
}
