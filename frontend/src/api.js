import { $ } from "./dom.js";
import { toast } from "./dom.js";
import { state } from "./state.js";
import { MODEL_CATALOG } from "./constants.js";

export async function apiRequest(path, options = {}) {
  const response = await fetch(path, {
    method: options.method || "GET",
    headers: options.body ? { "Content-Type": "application/json" } : undefined,
    body: options.body ? JSON.stringify(options.body) : undefined
  });
  const payload = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
  payload.httpStatus = response.status;
  payload.httpOk = response.ok;
  if (!response.ok && !options.allowError) {
    const error = new Error(payload.error || `HTTP ${response.status}`);
    error.payload = payload;
    throw error;
  }
  return payload;
}

export function requireRuntime() {
  if (state.api.connected) return true;
  toast("Real runtime is offline. Close this page and launch frontend/START_STUDIO.bat.", "error");
  return false;
}

export async function refreshNavCounts() {
  try {
    const jobs = await apiRequest("/api/jobs");
    const active = jobs.items.filter(job => ["queued", "running"].includes(job.status)).length;
    const navCount = $(".nav-count");
    if (navCount) navCount.textContent = String(active || jobs.items.length);
  } catch {
    // Nav badge is a convenience; a failed refresh should not surface an error toast.
  }
}

export async function connectRuntime(onConnected) {
  const badge = $(".route-health");
  try {
    const [health, models] = await Promise.all([
      apiRequest("/api/health"),
      apiRequest("/api/models")
    ]);
    state.api.connected = Boolean(health.ok);
    state.api.health = health;
    state.api.models = models.items || [];
    state.api.models.forEach(model => {
      const local = MODEL_CATALOG[model.model];
      if (!local) return;
      local.keys = model.known_keys;
      local.modes = model.modes;
      local.dataset = model.dataset_kind || local.dataset;
      local.backend = model;
    });
    const healthy = state.api.models.filter(model => model.healthy).length;
    badge.classList.remove("offline");
    badge.innerHTML = `<i></i> ${healthy}/${state.api.models.length} routes live`;
    badge.title = `Connected to ${health.python} · ${health.gpus?.length || 0} GPU(s)`;
    $(".coverage-pill").textContent = `${state.api.models.length} live routes · ${state.api.models.reduce((sum, model) => sum + model.modes.length, 0)} route modes`;
    refreshNavCounts();
    onConnected?.();
    return true;
  } catch (error) {
    state.api.connected = false;
    badge.classList.add("offline");
    badge.innerHTML = "<i></i> Runtime offline";
    badge.title = error.message;
    $("#savedState").textContent = "Runtime offline · use START_STUDIO.bat";
    return false;
  }
}

