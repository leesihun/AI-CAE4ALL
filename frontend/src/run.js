import { $, toast } from "./dom.js";
import { state } from "./state.js";
import { BLOCK_SPECS } from "./constants.js";
import { apiRequest, requireRuntime, refreshNavCounts } from "./api.js";
import { validateGraph, executableSteps, preflightConfigText, preflightMessages } from "./validate.js";
import { render } from "./graph.js";

export function renderRuntimeJob(job) {
  state.api.activeJob = job;
  $("#runtimeDrawer").classList.add("open");
  $("#runtimeJobTitle").textContent = job.label || "AI-CAE4ALL job";
  $("#runtimeJobMeta").textContent = `${job.status} · job ${job.id || "pending"}${job.pid ? ` · PID ${job.pid}` : ""}`;
  $("#runtimeStatusDot").className = `runtime-status-dot ${job.status || ""}`;
  $("#runtimeStep").textContent = job.total_steps
    ? `Step ${job.current_step || 0}/${job.total_steps}${job.step_label ? ` · ${job.step_label}` : ""}`
    : "Preparing";
  $("#runtimeLog").textContent = job.log || "Waiting for launcher output…";
  $("#runtimeLog").scrollTop = $("#runtimeLog").scrollHeight;
  $("#runtimeCancel").disabled = !["queued", "running"].includes(job.status);
  if (job.status === "running" || job.status === "queued") {
    $("#runBanner").classList.add("show");
    $("#runTitle").textContent = `Real job · ${job.status}`;
    $("#runDetail").textContent = job.step_label || "preflight → native launcher";
  }
}

export function applyJobStatus(job) {
  const terminal = ["completed", "failed", "cancelled"].includes(job.status);
  state.nodes.forEach(node => {
    const label = BLOCK_SPECS[node.type]?.label;
    if (label && job.steps?.some(step => step.label?.startsWith(label))) {
      node.status = job.status === "running" ? "running" : job.status === "completed" ? "complete" : "idle";
      node.progress = job.status === "completed" ? 100 : job.status === "running" ? 58 : 0;
    }
  });
  renderRuntimeJob(job);
  render();
  if (!terminal) return;
  state.running = false;
  window.clearInterval(state.api.pollTimer);
  state.api.pollTimer = null;
  $("#runBanner").classList.remove("show");
  $("#savedState").textContent = `Job ${job.status} · ${job.finished_at || "now"}`;
  refreshNavCounts();
  toast(
    job.status === "completed" ? "Real AI-CAE4ALL job completed." : `Job ${job.status}. Open the runtime log for details.`,
    job.status === "completed" ? "" : "error"
  );
}

export async function pollActiveJob() {
  const jobId = state.api.activeJob?.id;
  if (!jobId) return;
  try {
    const job = await apiRequest(`/api/jobs/${encodeURIComponent(jobId)}`);
    applyJobStatus(job);
  } catch (error) {
    window.clearInterval(state.api.pollTimer);
    state.api.pollTimer = null;
    toast(`Job polling failed: ${error.message}`, "error");
  }
}

export function beginCommandJob(job) {
  state.running = true;
  renderRuntimeJob(job);
  window.clearInterval(state.api.pollTimer);
  state.api.pollTimer = window.setInterval(pollActiveJob, 900);
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
  $("#runBanner").classList.add("show");
  $("#runTitle").textContent = "Authoritative preflight";
  const lines = [];
  let passed = true;
  for (let index = 0; index < steps.length; index += 1) {
    const step = steps[index];
    $("#runDetail").textContent = `Checking ${index + 1}/${steps.length} · ${step.label}`;
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
  if (state.running) {
    toast("A real AI-CAE4ALL job is already active.", "warn");
    return;
  }
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
        steps: steps.map(step => ({ label: step.label, config: step.config }))
      }
    });
    if (!job.httpOk) {
      state.running = false;
      $("#runBanner").classList.remove("show");
      const diagnostics = job.failures?.flatMap(failure => preflightMessages(failure.preflight).map(item => item.text)) || [job.error];
      renderRuntimeJob({
        id: "rejected",
        label: `${$("#pipelineName").value} · rejected`,
        status: "failed",
        current_step: 0,
        total_steps: steps.length,
        log: diagnostics.join("\n")
      });
      toast("Pipeline was not started because real preflight failed.", "error");
      return;
    }
    renderRuntimeJob(job);
    state.api.pollTimer = window.setInterval(pollActiveJob, 900);
    refreshNavCounts();
  } catch (error) {
    state.running = false;
    $("#runBanner").classList.remove("show");
    toast(`Could not start the real pipeline: ${error.message}`, "error");
  }
}

export async function stopRun() {
  const jobId = state.api.activeJob?.id;
  if (!state.running || !jobId || ["preflight", "rejected"].includes(jobId)) return;
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
