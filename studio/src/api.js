import { $ } from "./dom.js";
import { toast } from "./dom.js";
import { state } from "./state.js";
import { registerLiveModel } from "./constants.js";

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
  toast("Real runtime is offline. Close this page and launch studio/START_STUDIO.bat.", "error");
  return false;
}

export async function refreshNavCounts(prefetched = null) {
  try {
    const jobs = prefetched || await apiRequest("/api/jobs");
    const active = jobs.items.filter(job => ["queued", "running"].includes(job.status)).length;
    const navCount = $(".nav-count");
    if (navCount) {
      // A badge reads as "needs attention", so it counts queued/running jobs only and
      // disappears at zero; the all-time total stays available in the tooltip.
      navCount.textContent = String(active);
      navCount.style.display = active ? "" : "none";
      navCount.title = active ? `${active} active · ${jobs.items.length} total` : `${jobs.items.length} total runs`;
    }
    return jobs;
  } catch {
    // Nav badge is a convenience; a failed refresh should not surface an error toast.
    return null;
  }
}

export async function connectRuntime(onConnected) {
  const badge = $(".route-health");
  try {
    const [health, models, jobs] = await Promise.all([
      apiRequest("/api/health"),
      apiRequest("/api/models"),
      apiRequest("/api/jobs")
    ]);
    state.api.connected = Boolean(health.ok);
    state.api.health = health;
    state.api.models = models.items || [];
    // The live MethodSpec is authoritative. It also installs a generic model
    // block for a newly added trainable route, so registry growth cannot leave
    // a healthy backend route invisible until the next frontend release.
    state.api.models.forEach(registerLiveModel);
    const healthy = state.api.models.filter(model => model.healthy).length;
    badge.classList.remove("offline");
    badge.innerHTML = `<i></i> ${healthy}/${state.api.models.length} routes live`;
    badge.title = `Connected to ${health.python} · ${health.gpus?.length || 0} GPU(s)`;
    $(".coverage-pill").textContent = `${state.api.models.length} live routes · ${state.api.models.reduce((sum, model) => sum + model.modes.length, 0)} route modes`;
    refreshNavCounts(jobs);
    onConnected?.();
    // Rejoin every job still in flight, not just the first — several pipelines
    // can be running when the page is (re)opened.
    const activeJobs = jobs.items.filter(job => ["queued", "running"].includes(job.status));
    if (activeJobs.length) {
      const { beginCommandJob } = await import("./run.js");
      activeJobs.forEach((job, index) => beginCommandJob(job, { focus: index === activeJobs.length - 1 }));
    }
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

/**
 * What a saved .pth records about itself, fetched once per path.
 *
 * An Inference block used to be dead unless the model block that trained the
 * checkpoint was also on the canvas -- the graph carried the file but nothing
 * knew which method to launch or what architecture the weights expect. The
 * checkpoint knows both, and every native inference path already overlays its
 * `model_config` over the config file, so reading it here reproduces exactly
 * what the run would have used.
 *
 * Returns null (never throws) when the runtime is offline or the file cannot be
 * read; callers treat that as "not resolved yet", not as an error.
 */
export function checkpointMetadata(path) {
  const key = String(path || "").trim();
  if (!key) return null;
  // Deliberately NOT async: an async function hands back a *new* promise on
  // every call, so a caller that subscribes to the result once per autofill
  // pass ends up with an unbounded fan-out of callbacks -- which re-enter
  // autofill, subscribe again, and take the renderer down with them. Returning
  // the one shared in-flight promise (or the settled record) keeps it linear.
  if (state.checkpointMeta.has(key)) return state.checkpointMeta.get(key);
  if (!state.api.connected) return null;
  const pending = apiRequest(`/api/checkpoint?path=${encodeURIComponent(key)}`, { allowError: true })
    .then(payload => {
      const record = payload?.ok
        ? payload
        : { ok: false, error: payload?.error || `HTTP ${payload?.httpStatus || "error"}`, path: key };
      state.checkpointMeta.set(key, record);
      return record;
    })
    .catch(error => {
      const record = { ok: false, error: error.message, path: key };
      state.checkpointMeta.set(key, record);
      return record;
    });
  state.checkpointMeta.set(key, pending);
  return pending;
}

/** The resolved record for `path`, or null while it is still in flight. */
export function checkpointMetaNow(path) {
  const cached = state.checkpointMeta.get(String(path || "").trim());
  return cached && !(cached instanceof Promise) ? cached : null;
}
