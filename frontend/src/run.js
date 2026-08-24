import { $, $$, escapeHtml, toast } from "./dom.js";
import { state } from "./state.js";
import { BLOCK_SPECS } from "./constants.js";
import { apiRequest, requireRuntime, refreshNavCounts } from "./api.js";
import { validateGraph, executableSteps, preflightConfigText, preflightMessages, inferenceDatasetWarnings } from "./validate.js";
import { render } from "./graph.js";
import { jumpToFailingField } from "./config.js";
import { schedulePipelineSave, pipelineDocument } from "./persistence.js";

const REJECTED_STATUSES = ["failed", "cancelled"];

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
      return `<div class="diagnostic ${severity}${clickable}" data-jump-diagnostic="${index}"><i></i><span>${item.stepLabel ? `<strong>${escapeHtml(item.stepLabel)}</strong> · ` : ""}[${escapeHtml(item.code || "")}]${item.field ? ` ${escapeHtml(item.field)}:` : ""} ${escapeHtml(item.message || "")}${item.hint ? ` <em>Hint: ${escapeHtml(item.hint)}</em>` : ""}${item.nodeId ? ' <b class="diagnostic-jump">Fix now →</b>' : ""}</span></div>`;
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
  logEl.scrollTop = logEl.scrollHeight;
  $("#runtimeCancel").disabled = !["queued", "running"].includes(job.status);
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

export function applyJobStatus(job) {
  const terminal = ["completed", "failed", "cancelled"].includes(job.status);
  const exactNodeIds = new Set((job.steps || []).map(step => step.node_id).filter(Boolean));
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
    const legacyMatch = !exactNodeIds.size && label && job.steps?.some(step => step.label?.startsWith(label));
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
  exactNodeIds.forEach(sourceNodeId => {
    state.edges
      .filter(edge => edge.fromNode === sourceNodeId && edge.fromPort === "metrics")
      .map(edge => state.nodes.find(node => node.id === edge.toNode))
      .filter(node => node?.type === "evaluate.training_metrics")
      .forEach(node => { node.config.job_id = job.id; });
  });
  // The backend resolves where each step actually wrote its predictions (the
  // epoch-numbered directory is not derivable from the config). Carry it onto
  // the block so Inspect opens this run's own results instead of guessing at
  // whatever prediction file happens to be lying around the repository.
  (job.steps || []).forEach(step => {
    if (!step.results || !step.node_id) return;
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
    node.config.results_path = step.results;
    node.config.results_samples = String(step.results_samples ?? "");
  });
  if (exactNodeIds.size) schedulePipelineSave();
  if (terminal) state.api.trackedJobs.delete(job.id);
  else state.api.trackedJobs.set(job.id, job);
  state.running = state.api.trackedJobs.size > 0;
  // Several jobs may be polling at once; only the focused one owns the drawer.
  if (!state.api.activeJob || state.api.activeJob.id === job.id) {
    renderRuntimeJob(job, { reveal: false });
  }
  renderActiveJobCount();
  render();
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

export async function validatePipeline(targetId = null) {
  const errors = validateGraph(false);
  if (errors.length) {
    toast(`Graph validation failed: ${errors[0]}`, "error");
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
  $("#runTitle").textContent = "Authoritative preflight";
  const lines = [];
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
      continue;
    }
    const result = await preflightConfigText(step.config, step.label, {
      skipFilesystem: index > 0,
      skipNative: index > 0
    });
    if (!result) {
      passed = false;
      break;
    }
    const summary = result.report?.summary || { errors: 1, warnings: 0, notices: 0 };
    lines.push(`${result.ok ? "PASS" : "FAIL"} ${step.label}: ${summary.errors} errors, ${summary.warnings} warnings, ${summary.notices} notices${index > 0 ? " · dependency checks deferred" : ""}`);
    result.report?.diagnostics?.forEach(item => lines.push(`  [${item.code}] ${item.message}`));
    if (!result.ok) passed = false;
  }
  $("#runBanner").classList.remove("show");
  renderRuntimeJob({
    id: "preflight",
    label: `${$("#pipelineName").value} · preflight`,
    status: passed ? "completed" : "failed",
    current_step: steps.length,
    total_steps: steps.length,
    log: lines.join("\n")
  });
  toast(passed ? "Authoritative preflight passed." : "Preflight failed. Read the real diagnostics.", passed ? "" : "error");
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
  if (!window.confirm(`Execute the real AI-CAE4ALL launcher?\n\n${preview}\n\nThis may use CUDA, write checkpoints, and run for a long time. Preflight runs before the process starts.`)) return;
  state.running = true;
  $("#runBanner").classList.add("show");
  $("#runTitle").textContent = "Submitting real pipeline";
  $("#runDetail").textContent = "authoritative preflight → native launcher";
  try {
    const job = await apiRequest("/api/pipeline/run", {
      method: "POST",
      allowError: true,
      body: {
        label: $("#pipelineName").value,
        strict: false,
        target_node_id: targetId || "",
        // Saved with the run so the exact graph can be reloaded from Runs later.
        pipeline: pipelineDocument(),
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
