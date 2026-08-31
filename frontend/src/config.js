import { $, $$, escapeHtml, toast, closeOverlay } from "./dom.js";
import { keys } from "./text.js";
import { state, snapshot } from "./state.js";
import {
  ICONS, BLOCK_SPECS, MODEL_CATALOG, REQUIRED, CHOICES, BOOLEAN_KEYS,
  OPERATOR_REMOVED, TRANSOLVER_REJECTED, CHI_FLOW_REMOVED, CONFIG_SECTIONS, HELP
} from "./constants.js";
import { apiRequest, requireRuntime } from "./api.js";
import { addBlock, selectNode } from "./graph.js";
import {
  applyGraphAutofill, autoFillCount, autoFillMeta,
  markManualConfigValue, resetManualConfigValues
} from "./autofill.js";

function retainExplicitConfig(node) {
  resetManualConfigValues(node);
  Object.entries(node.config).forEach(([key, value]) => markManualConfigValue(node, key, value));
  applyGraphAutofill();
}

export function requiredFor(modelId, mode) {
  const canonicalModel = String(modelId || "").toLowerCase();
  const canonicalMode = String(mode || "").toLowerCase();
  // Prefer the live spec (fetched in connectRuntime): the hardcoded map drifts, and
  // over-claiming is worse than under-claiming — it demanded param_dir for every
  // SimulGen mode even though the spec only needs it for lc_data_type csv/image,
  // pushing users away from the hdf5/cond_var conditioner path. Conditional rules
  // stay with the authoritative preflight rather than being mirrored here.
  const liveRequired = MODEL_CATALOG[canonicalModel]?.required?.[canonicalMode];
  if (liveRequired) return new Set(liveRequired);
  const modelRequired = REQUIRED[canonicalModel]?.[canonicalMode];
  if (modelRequired) return new Set(modelRequired);
  return new Set(canonicalMode === "inference"
    ? keys(`model mode gpu_ids modelpath infer_dataset input_var output_var`)
    : keys(`model mode gpu_ids dataset_dir modelpath input_var output_var training_epochs batch_size learningr`));
}

/** Defaults published by the live MethodSpec, kept separate from the Studio's
 * opinionated new-block values so the raw config remains explicit. */
export function backendDefaultsFor(modelId, mode) {
  const model = MODEL_CATALOG[String(modelId || "").toLowerCase()] || {};
  const canonicalMode = String(mode || "").toLowerCase();
  return {
    ...(model.backendDefaults || {}),
    ...(model.backendDefaultsByMode?.[canonicalMode] || {})
  };
}

export function keyDisposition(modelId, key, config = null) {
  // The multiscale trainer prints "[message_passing_num is IGNORED when
  // use_multiscale=True]" and then obeys mp_per_level instead, but the sheet
  // showed the key as a live, "set" value -- so anyone tuning HI-MGN would
  // reasonably turn that number and see nothing change. Both shipped HI-MGN
  // paths leave message_passing_num in the config, which makes it worse.
  if (key === "message_passing_num" && config && isTruthyConfigValue(config.use_multiscale)) return "inactive";
  if (modelId === "transolver" && TRANSOLVER_REJECTED.has(key)) return "removed";
  if (modelId === "chi-mgnflow" && CHI_FLOW_REMOVED.has(key)) return "removed";
  if (["point_deeponet", "deeponet", "fno", "gino"].includes(modelId)) {
    if (OPERATOR_REMOVED.has(key)) return "removed";
    const owner = key.startsWith("point") ? "point_deeponet" : key.startsWith("deeponet_") ? "deeponet" : key.startsWith("fno_") ? "fno" : key.startsWith("gino_") ? "gino" : "";
    if (owner && owner !== modelId) return "inactive";
  }
  if (key.startsWith("_") || ["num_timesteps", "num_node_types", "log_dir"].includes(key)) return "runtime";
  return "active";
}

function isTruthyConfigValue(value) {
  return ["true", "1", "yes"].includes(String(value ?? "").trim().toLowerCase());
}

/**
 * Warn about keys a *switch you just flipped* made mandatory.
 *
 * `requiredFor` deliberately mirrors only the spec's per-mode required set and
 * leaves conditional rules to the authoritative preflight -- a good call, since
 * mirroring the full rule set here would drift. The cost showed up in practice:
 * turning on use_multiscale makes voronoi_clusters and mp_per_level mandatory
 * (MGN-MULTI-REQ), yet nothing in the sheet said so, and the shipped HI-MGN
 * preset shipped without them, so the first sign of trouble was four errors
 * from a preflight six seconds later -- or a failed run.
 *
 * This stays a hint, not a badge: preflight remains the authority, and the text
 * names the rule so it is obvious where the requirement comes from.
 */
function conditionallyMissing(modelId, config) {
  const notes = [];
  const isMesh = ["meshgraphnets", "meshgraphnets-v", "chi-mgnflow"].includes(modelId);
  if (isMesh && isTruthyConfigValue(config.use_multiscale)) {
    const needed = ["coarsening_type", "multiscale_levels", "voronoi_clusters", "mp_per_level"]
      .filter(key => String(config[key] ?? "").trim() === "");
    if (needed.length) {
      notes.push({
        type: "warn",
        text: `use_multiscale is True, so MGN-MULTI-REQ also requires: ${needed.join(", ")}`
      });
    }
    const levels = Number(config.multiscale_levels);
    const entries = String(config.mp_per_level ?? "").split(",").map(item => item.trim()).filter(Boolean);
    if (Number.isFinite(levels) && levels > 0 && entries.length && entries.length !== 2 * levels + 1) {
      notes.push({
        type: "warn",
        text: `mp_per_level needs ${2 * levels + 1} entries for multiscale_levels ${levels}; it has ${entries.length}.`
      });
    }
  }
  return notes;
}

export function sectionFor(modelId, key, required, config = null) {
  if (keyDisposition(modelId, key, config) !== "active") return "Inactive / rejected";
  if (required.has(key)) return "Required";
  if (key.startsWith("opt_")) return "Optimization";
  if (/dataset|modelpath|output_dir|log_file|pipeline_log|param_dir|input_mesh|sidecar|split_seed/.test(key)) return "Data & output";
  if (/^(point_|pointnet_|deeponet_|fno_|gino_|encoder_|decoder_|fm_arch|fm_blocks|fm_hidden|fm_cond_hidden|flow_time_freqs|latent_|latent_dim|message_passing|slice_num|num_layers|num_heads|attention_kernel|mlp_ratio|coarsening|multiscale|mp_per_level|positional|fourier|operator_dim|global_condition|num_filter|lc_filter|network_size)/.test(key)) return "Architecture";
  // Network-shape keys the prefix-anchored test above misses. Without these,
  // MLP showed an "Architecture / 0" tab while hidden_layers, activation,
  // output_activation, and norm sat in "Advanced", and every mesh route filed
  // its auxiliary encoder/prior sizes there too. Each name is matched exactly
  // so the later Training / Resources / Inference tests keep the keys they own.
  if (/^(hidden_layers|activation|output_activation|norm|pool_type|unpool_type|pool_heads|bipartite_unpool|residual_scale|use_spatial_attention|fm_heads|vae_latent_dim|vae_mp_layers|prior_hidden_dim|prior_mp_layers)$/.test(key)) return "Architecture";
  if (/training_epochs|learningr|weight_decay|warmup|batch_size|loss|dropout|grad_|noise|ema|kl_|beta|eikonal|surface_weight|normal_weight|alpha|lambda_|free_bits|mmd|recon_loss|recon_iter|flow_t_|flow_loss_weighting|flow_det_prob/.test(key)) return "Training";
  if (/gpu|parallel|worker|prefetch|amp|compile|checkpointing|cache|profile|fsdp|pipeline_microbatches|chunk_size|load_all/.test(key)) return "Resources & runtime";
  if (/^infer|^test|^val_|display|plot|histogram|num_samples|candidate_multiplier|cfg_scale|condition_|cond_|ode_steps|flow_steps|flow_solver|flow_predict|best_by|mc_resolution|sample_index|source_num_samples|posterior_noise|timesteps_reduced/.test(key)) return "Inference & evaluation";
  return "Advanced";
}

export function choicesFor(modelId, key) {
  if (key === "model") return [modelId];
  if (key === "mode") return MODEL_CATALOG[modelId].modes;
  if (BOOLEAN_KEYS.has(key) || /^use_/.test(key)) return ["False", "True"];
  if (key === "parallel_mode") {
    // Mirrors each spec's own validator exactly: sdfflow/simulgenvae take
    // single|ddp|fsdp ("single" being the right pick on a one-GPU box);
    // Transolver takes ddp|node_shard (TRANS-PARALLEL-001 rejects model_split);
    // model_split is fno/gino-only among the operators (NOVAR-PARALLEL-002),
    // so deeponet/point_deeponet are ddp-only; the mesh methods take ddp|model_split.
    if (["simulgenvae", "sdfflow"].includes(modelId)) return ["single", "ddp", "fsdp"];
    if (modelId === "transolver") return ["ddp", "node_shard"];
    if (["fno", "gino", "meshgraphnets", "meshgraphnets-v", "chi-mgnflow"].includes(modelId)) return ["ddp", "model_split"];
    return ["ddp"];
  }
  return CHOICES[key] || null;
}

function canonicalConfigValue(key, value) {
  const text = String(value ?? "").trim();
  if (key === "model" || key === "mode") return text.toLowerCase();
  if ((BOOLEAN_KEYS.has(key) || /^use_/.test(key)) && /^(true|false)$/i.test(text)) {
    return text.toLowerCase() === "true" ? "True" : "False";
  }
  const choices = CHOICES[key];
  return choices?.find(choice => String(choice).toLowerCase() === text.toLowerCase()) ?? text;
}

/** Match the authoritative suite parser: config keys are case-insensitive. */
export function normalizeConfigValues(values = {}) {
  const normalized = {};
  Object.entries(values).forEach(([rawKey, value]) => {
    const key = String(rawKey).trim().toLowerCase();
    if (key) normalized[key] = canonicalConfigValue(key, value);
  });
  return normalized;
}

// The flat format has no representation for an empty value: "key" with nothing
// after it fails the native parser as CFG-SYNTAX-001, which then masks the
// actionable CFG-REQ-001 for that same key. Omitting it reports "required" instead.
const hasValue = value => String(value ?? "").trim() !== "";

/**
 * Keys the Studio writes onto a block for its own bookkeeping.
 *
 * They are not config: `run.js` stamps the last run's results onto the block so
 * the canvas can show the evidence, and `rawConfig` then emitted them as
 * "unknown keys" into the flat .txt handed to the launcher -- which answered
 * with CFG-UNKNOWN-001 for each, and would reject the config outright under
 * --strict. Every config the Studio exported after a successful run carried
 * them. Genuinely unknown keys (a user's typo) must still be emitted, so this
 * is a fixed list rather than a catch-all filter.
 */
const STUDIO_ONLY_KEYS = new Set([
  "results_path", "results_samples", "report_path", "evaluated_samples",
  "export_path", "job_id", "model_id", "dataset_path", "checkpoint_path",
  "parameters_path", "prediction_path", "truth_path", "compatibility", "binding"
]);

export function rawConfig(values, catalog) {
  const normalized = normalizeConfigValues(values);
  const ordered = catalog.filter(key => Object.hasOwn(normalized, key) && hasValue(normalized[key]));
  const unknown = Object.keys(normalized)
    .filter(key => !catalog.includes(key) && hasValue(normalized[key]) && !STUDIO_ONLY_KEYS.has(key))
    .sort();
  return [...ordered, ...unknown].map(key => `${key.padEnd(29, " ")}${normalized[key]}`).join("\n");
}

export function parseConfig(text) {
  const values = {};
  const messages = [];
  const seen = new Set();
  text.split(/\r?\n/).forEach((line, index) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("%") || trimmed.startsWith("#") || trimmed === "'") return;
    const clean = line.split("#")[0].trim();
    const match = clean.match(/^(\S+)\s+(.+)$/);
    if (!match) {
      messages.push({ type: "warn", text: `Line ${index + 1} could not be parsed: ${trimmed}` });
      return;
    }
    const key = match[1].toLowerCase();
    if (seen.has(key)) messages.push({ type: "warn", text: `Duplicate ${key} on line ${index + 1}; config keys are case-insensitive and the last value wins.` });
    seen.add(key);
    values[key] = canonicalConfigValue(key, match[2]);
  });
  return { values, messages };
}

export function presetOptions(modelId) {
  const options = [["repository", "Checked-in example"], ["smoke", "Smoke test"], ["low_vram", "Low VRAM"]];
  if (modelId === "meshgraphnets") options.push(["mgn_flat", "Flat MGN"], ["mgn_hi", "HI-MGN"], ["mgn_bsms", "BSMS-GNN"]);
  if (modelId === "sdfflow") options.push(["sdfflow_full", "Full VAE + FM"], ["sdfflow_vae", "VAE only"], ["sdfflow_fm", "Flow matching only"], ["sdfflow_optimize", "Checked-in closed-loop optimization"]);
  if (modelId === "simulgenvae") options.push(["simulgen_full", "VAE → LC pipeline"], ["simulgen_vae", "VAE only"], ["simulgen_lc", "LC only"], ["simulgen_reconstruct", "Reconstruct fields"]);
  return options;
}

export function openConfig(nodeId) {
  const node = state.nodes.find(item => item.id === nodeId);
  const spec = node && BLOCK_SPECS[node.type];
  if (!node || !spec?.isModel) return;
  node.config = normalizeConfigValues(node.config);
  applyGraphAutofill();
  state.configNode = nodeId;
  state.configSection = "Required";
  state.configSearch = "";
  state.configMessages = [];
  $("#configSearch").value = "";
  $("#changedOnly").checked = false;
  $("#showInactive").checked = true;
  $("#configIcon").textContent = ICONS.model;
  $("#configTitle").textContent = `${spec.label} · full configuration`;
  $("#configSubtitle").textContent = `${MODEL_CATALOG[spec.modelId].keys.length} accepted keys · ${MODEL_CATALOG[spec.modelId].modes.length} modes · ${MODEL_CATALOG[spec.modelId].dataset}`;
  $("#configMode").innerHTML = MODEL_CATALOG[spec.modelId].modes.map(mode => `<option value="${mode}">${mode}</option>`).join("");
  $("#configMode").value = node.config.mode || MODEL_CATALOG[spec.modelId].modes[0];
  $("#configPreset").innerHTML = presetOptions(spec.modelId).map(([value, label]) => `<option value="${value}">${label}</option>`).join("");
  $("#configOverlay").classList.add("open");
  renderConfig();
}

export function renderConfig() {
  const node = state.nodes.find(item => item.id === state.configNode);
  if (!node) return;
  applyGraphAutofill();
  const spec = BLOCK_SPECS[node.type];
  const model = MODEL_CATALOG[spec.modelId];
  node.config = normalizeConfigValues(node.config);
  const configuredMode = String(node.config.mode || "").toLowerCase();
  const mode = model.modes.includes(configuredMode) ? configuredMode : $("#configMode").value || model.modes[0];
  $("#configMode").value = mode;
  node.config.model = spec.modelId;
  node.config.mode = mode;
  const required = requiredFor(spec.modelId, mode);
  const backendDefaults = backendDefaultsFor(spec.modelId, mode);
  const search = state.configSearch.toLowerCase().trim();
  const changedOnly = $("#changedOnly").checked;
  const showInactive = $("#showInactive").checked;
  const groups = Object.fromEntries(CONFIG_SECTIONS.map(section => [section, []]));
  model.keys.forEach(key => {
    const disposition = keyDisposition(spec.modelId, key, node.config);
    if (!showInactive && disposition !== "active") return;
    if (changedOnly && !Object.hasOwn(node.config, key)) return;
    if (search && !key.toLowerCase().includes(search)) return;
    groups[sectionFor(spec.modelId, key, required, node.config)].push(key);
  });
  // Studio bookkeeping keys are excluded here too: they are not config the user
  // typed, they are never sent to the launcher, and listing them as "rejected"
  // invites someone to go hunting for a problem that does not exist.
  Object.keys(node.config)
    .filter(key => !model.keys.includes(key) && !STUDIO_ONLY_KEYS.has(key))
    .forEach(key => {
      if (!showInactive) return;
      if (search && !key.toLowerCase().includes(search)) return;
      groups["Inactive / rejected"].push(key);
    });
  // Only offer sections that actually hold a key. An empty tab is still a tab:
  // clicking "Architecture / 0" on MLP or "Inactive / rejected / 0" on the mesh
  // routes opened a blank panel reading "0 visible keys" with nothing to say why.
  const populated = CONFIG_SECTIONS.filter(section => groups[section].length);
  if (!groups[state.configSection]?.length) state.configSection = populated[0] || "Required";

  $("#configSectionList").innerHTML = populated.map(section => `<button class="config-section-button${state.configSection === section ? " active" : ""}" data-config-section="${section}"><span>${section}</span><small>${groups[section].length}</small></button>`).join("");
  $$("[data-config-section]").forEach(button => button.addEventListener("click", () => {
    state.configSection = button.dataset.configSection;
    renderConfig();
  }));

  // A search spans every section. Filtering the search *within* the selected
  // section meant a key you had just typed the exact name of stayed invisible
  // whenever it lived elsewhere and the current section happened to hold some
  // other match: searching "modelpath" from "Data & output" showed
  // init_modelpath and hid `modelpath` itself, which sits in "Required".
  const visible = search
    ? CONFIG_SECTIONS.flatMap(section => groups[section])
    : groups[state.configSection];
  $("#configSectionTitle").textContent = search ? `Search: ${search}` : state.configSection;
  $("#configSectionMeta").textContent = search
    ? `${visible.length} matching keys across all sections · ${mode} mode`
    : `${visible.length} visible keys · ${mode} mode`;
  const automaticCount = autoFillCount(node);
  const defaultedRequiredCount = [...required].filter(key =>
    !Object.hasOwn(node.config, key) && Object.hasOwn(backendDefaults, key)
  ).length;
  $("#configBadges").innerHTML = `<span class="badge">${model.keys.length} accepted</span><span class="badge warn">${required.size} required</span>${defaultedRequiredCount ? `<span class="badge">${defaultedRequiredCount} backend-defaulted</span>` : ""}${automaticCount ? `<span class="badge auto">${automaticCount} graph-filled</span>` : ""}`;
  $("#schemaNote").innerHTML = `<strong>${model.keys.length} live keys</strong><br>All MethodSpec keys are present. Closed choices use dropdowns; paths, widths, lists, and open family values remain manual.<br><br>${spec.modelId === "simulgenvae" ? "The live SimulGen route has separate VAE, LC, combined, and reconstruction requirements." : "Shared-family inactive and rejected keys remain visible for diagnostic honesty."}`;

  $("#configFields").innerHTML = visible.length ? visible.map(key => {
    const accepted = model.keys.includes(key);
    const disposition = accepted ? keyDisposition(spec.modelId, key, node.config) : "unknown";
    const set = Object.hasOwn(node.config, key);
    const requiredKey = required.has(key);
    const hasBackendDefault = !set && Object.hasOwn(backendDefaults, key);
    const backendDefault = hasBackendDefault ? backendDefaults[key] : "";
    const rejectedByPreflight = state.configRejectedNode === node.id && state.configRejectedField === key;
    const status = rejectedByPreflight || disposition === "removed" || disposition === "unknown" ? "rejected" : disposition === "inactive" ? "inactive" : disposition === "runtime" ? "runtime" : hasBackendDefault ? "defaulted" : requiredKey ? "required" : set ? "set" : "optional";
    const value = set ? node.config[key] : "";
    const automatic = autoFillMeta(node, key);
    const choices = choicesFor(spec.modelId, key);
    const disabled = disposition === "removed" || disposition === "runtime";
    const unsetLabel = hasBackendDefault ? `backend default: ${backendDefault}` : "not set";
    const control = choices
      ? `<select class="config-control full-config-control" data-key="${key}"${disabled ? " disabled" : ""}><option value="">— ${escapeHtml(unsetLabel)} —</option>${choices.map(choice => `<option value="${escapeHtml(choice)}"${String(value).toLowerCase() === String(choice).toLowerCase() ? " selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select>`
      : `<input class="config-control full-config-control" data-key="${key}" value="${escapeHtml(value)}" placeholder="${hasBackendDefault ? `backend default: ${escapeHtml(backendDefault)}` : "manual value"}"${disabled ? " disabled" : ""}>`;
    // Why a key is dead outranks what the key means: the generic help for
    // message_passing_num reads as though the value still does something, which
    // is the whole reason nobody noticed the trainer ignores it under multiscale.
    const dispositionHelp = key === "message_passing_num" && disposition === "inactive"
      ? "Ignored while use_multiscale is True — the trainer prints this and uses mp_per_level for depth instead. Set use_multiscale False for this value to take effect."
      : disposition === "unknown" ? "This key is not accepted by the selected model. Clear its value to remove it before running preflight again."
      : disposition === "removed" ? "Known by a shared diagnostic schema, but rejected for this selected model."
      : disposition === "inactive" ? "Accepted by the shared family schema but configures a different variant."
      : "";
    const baseHelp = dispositionHelp || HELP[key] || "Manual input is retained because the live spec does not publish a closed value set for this field.";
    const help = automatic
      ? `Auto-filled from ${automatic.sourceLabel}: ${automatic.reason}. Edit it to keep a manual override, or clear it to follow the graph again.`
      : hasBackendDefault && disposition === "active"
        ? `${baseHelp} Backend default for ${mode}: ${backendDefault}; leave this unset to use it or enter a value to override it.`
        : baseHelp;
    return `<article class="config-card ${disposition}${automatic ? " graph-autofilled" : ""}${rejectedByPreflight ? " preflight-rejected" : ""}"><header class="config-card-head"><span class="config-key">${key}</span><span class="config-card-states">${automatic ? `<span class="config-status autofill">auto · ${escapeHtml(automatic.sourceLabel)}</span>` : ""}<span class="config-status ${status}">${status}</span></span></header>${control}<p class="config-help">${escapeHtml(help)}</p></article>`;
  }).join("") : `<div class="inspect-empty" style="height:auto;grid-column:1/-1"><p>No keys match this filter.</p></div>`;

  $$(".full-config-control").forEach(control => control.addEventListener("change", () => {
    snapshot();
    if (control.value === "") delete node.config[control.dataset.key];
    else node.config[control.dataset.key] = control.value;
    markManualConfigValue(node, control.dataset.key, control.value);
    applyGraphAutofill();
    if (state.configRejectedNode === node.id && state.configRejectedField === control.dataset.key) {
      state.configRejectedNode = null;
      state.configRejectedField = null;
    }
    $("#savedState").textContent = "Unsaved changes";
    $("#configRaw").value = rawConfig(node.config, model.keys);
    renderConfig();
  }));

  $("#configRaw").value = rawConfig(node.config, model.keys);
  const missing = [...required].filter(key =>
    (!Object.hasOwn(node.config, key) || node.config[key] === "")
    && !Object.hasOwn(backendDefaults, key)
  );
  // Same exclusion as rawConfig: the Studio's own bookkeeping keys are never
  // emitted, so warning that they "will fail preflight" was false -- and the
  // authoritative preflight beside it disagreed, reporting 0 errors.
  const unknown = Object.keys(node.config)
    .filter(key => !model.keys.includes(key) && !STUDIO_ONLY_KEYS.has(key));
  const conditional = conditionallyMissing(spec.modelId, node.config);
  const diagnostics = [
    { type: "", text: `${model.keys.length} accepted keys loaded; ${Object.keys(node.config).length} currently set.` },
    ...(missing.length ? [{ type: "warn", text: `Missing required for ${mode}: ${missing.join(", ")}` }] : [{ type: "", text: `All required ${mode} keys have explicit values or published backend defaults.` }]),
    ...conditional,
    ...(unknown.length ? [{ type: "warn", text: `Unknown keys will fail preflight: ${unknown.join(", ")}` }] : []),
    ...state.configMessages
  ];
  $("#configDiagnostics").innerHTML = diagnostics.map(item => `<div class="diagnostic ${item.type}"><i></i><span>${escapeHtml(item.text)}</span></div>`).join("");
}

export async function applyPreset() {
  const node = state.nodes.find(item => item.id === state.configNode);
  if (!node) return;
  const spec = BLOCK_SPECS[node.type];
  const model = MODEL_CATALOG[spec.modelId];
  const preset = $("#configPreset").value;
  if (preset === "smoke" && spec.modelId === "simulgenvae") {
    if (!requireRuntime()) return;
    if (!window.confirm("Create a tiny fixed-geometry HDF5 and condition CSV under frontend/runtime, then load a one-epoch CPU SimulGen VAE config? This fixture exercises real code but is not scientific evidence.")) return;
    try {
      const fixture = await apiRequest("/api/simulgen/smoke-fixture", { method: "POST", body: {} });
      const parsed = parseConfig(fixture.config);
      snapshot();
      node.config = { ...parsed.values };
      retainExplicitConfig(node);
      $("#configMode").value = fixture.mode;
      state.configSection = "Required";
      state.configMessages = [
        { type: "", text: `Runnable smoke fixture created: ${fixture.dataset}` },
        { type: "warn", text: fixture.note }
      ];
      $("#savedState").textContent = "Runnable smoke config · not scientific evidence";
      renderConfig();
      toast("Created and loaded a real one-epoch SimulGen smoke configuration.");
    } catch (error) {
      toast(`Could not create SimulGen smoke fixture: ${error.message}`, "error");
    }
    return;
  }
  let values = {};
  if (preset === "repository" || preset === "simulgen_full" || preset === "sdfflow_full") values = { ...model.defaults };
  if (preset === "smoke") values = {
    training_epochs: "2", batch_size: "1", vae_training_epochs: "2", lc_training_epochs: "2", fm_training_epochs: "2",
    vae_batch_size: "1", lc_batch_size: "2", fm_batch_size: "2", test_max_batches: "1", num_test_shapes: "2"
  };
  if (preset === "low_vram") values = {
    batch_size: "1", vae_batch_size: "1", lc_batch_size: "8", fm_batch_size: "4", grad_accum_steps: "4",
    use_amp: "True", vae_use_amp: "True", lc_use_amp: "True", use_checkpointing: "True", load_all: "False",
    chunk_size: "1024", infer_chunk_size: "1024", train_query_chunk_size: "1024", infer_query_chunk_size: "1024"
  };
  if (preset === "mgn_flat") values = { use_multiscale: "False", coarsening_type: "none" };
  // MGN-MULTI-REQ makes voronoi_clusters and mp_per_level mandatory the moment
  // use_multiscale is True -- for bfs as well -- and mp_per_level must hold
  // 2*levels+1 entries. Emitting only the three switches left every multiscale
  // preset four errors deep in preflight, with the two missing keys carrying no
  // "required" badge anywhere in the sheet to hint at it. These mirror the
  // checked-in configs/MeshGraphNets/ex9 pair, which are known to run.
  if (preset === "mgn_hi") values = {
    use_multiscale: "True", coarsening_type: "voronoi_seedmean", multiscale_levels: "2",
    voronoi_clusters: "500, 100", mp_per_level: "4, 6, 8, 6, 4"
  };
  if (preset === "mgn_bsms") values = {
    use_multiscale: "True", coarsening_type: "bfs", multiscale_levels: "2",
    voronoi_clusters: "500, 100", mp_per_level: "4, 6, 8, 6, 4"
  };
  if (preset === "sdfflow_vae") values = { mode: "train_vae" };
  if (preset === "sdfflow_fm") values = { mode: "train_fm" };
  if (preset === "sdfflow_optimize") {
    if (!requireRuntime()) return;
    try {
      const payload = await apiRequest("/api/config?path=configs%2FGeometry_generation%2Fconfig_optimize.txt");
      values = parseConfig(payload.text).values;
    } catch (error) {
      toast(`Could not load the checked-in SDFFlow optimization config: ${error.message}`, "error");
      return;
    }
  }
  if (preset === "simulgen_vae") values = { mode: "train_vae", training_epochs: model.defaults.vae_training_epochs, batch_size: model.defaults.vae_batch_size, learningr: model.defaults.vae_learningr };
  if (preset === "simulgen_lc") values = { mode: "train_lc", training_epochs: model.defaults.lc_training_epochs, batch_size: model.defaults.lc_batch_size, learningr: model.defaults.lc_learningr };
  if (preset === "simulgen_reconstruct") values = { mode: "reconstruct", batch_size: "16", output_dir: "../output/simulgenvae/ex1/reconstruct" };
  values = Object.fromEntries(Object.entries(values).filter(([key]) => model.keys.includes(key)));
  const changes = Object.entries(values).filter(([key, value]) => String(node.config[key] ?? "") !== String(value));
  if (!changes.length) {
    toast("Preset already matches the current configuration.");
    return;
  }
  const preview = changes.slice(0, 9).map(([key, value]) => `${key} → ${value}`).join("\n");
  if (!window.confirm(`Apply ${changes.length} preset changes?\n\n${preview}${changes.length > 9 ? "\n…" : ""}\n\nOther manual values are preserved.`)) return;
  snapshot();
  Object.assign(node.config, values);
  if (values.mode) $("#configMode").value = values.mode;
  state.configMessages = [{ type: "", text: `Applied ${preset} with ${changes.length} explicit changes.` }];
  $("#savedState").textContent = "Unsaved changes";
  renderConfig();
}

export async function loadConfigExample(modelId, path) {
  try {
    const payload = await apiRequest(`/api/config?path=${encodeURIComponent(path)}`);
    const parsed = parseConfig(payload.text);
    const type = `model.${modelId}`;
    if (!BLOCK_SPECS[type]) {
      toast(`${modelId} is a tool route; its config can be inspected from the checked-in file but it is not a trainable model block.`, "warn");
      return;
    }
    let node = state.nodes.find(item => item.type === type);
    if (!node) {
      addBlock(type);
      node = state.nodes.find(item => item.id === state.selectedNode);
    }
    snapshot();
    node.config = { ...parsed.values };
    retainExplicitConfig(node);
    node.loadedConfigPath = path;
    closeOverlay("studioOverlay");
    selectNode(node.id);
    openConfig(node.id);
    state.configMessages = [
      { type: "", text: `Loaded checked-in configuration: ${path}` },
      ...parsed.messages
    ];
    renderConfig();
    toast(`Loaded ${path} into the real ${modelId} block.`);
  } catch (error) {
    toast(`Could not load config: ${error.message}`, "error");
  }
}

/** Formats /api/config/explain's bucketed response as diagnostic-style lines, mirroring `--explain-config`. */
export function explainMessages(payload) {
  if (!payload || payload.error) {
    return [{ type: "error", text: payload?.error || "Explain-config did not return a result." }];
  }
  const list = (title, names) => `${title} (${names.length}): ${names.length ? names.join(", ") : "<none>"}`;
  return [
    { type: "", text: `Route: ${payload.route ? `${payload.route.model} · ${payload.route.mode}` : "not resolved"}` },
    { type: "", text: list("Required and present", payload.required_present) },
    { type: payload.required_missing.length ? "error" : "", text: list("Required and missing", payload.required_missing) },
    { type: payload.recommended_missing.length ? "warn" : "", text: list("Recommended but missing", payload.recommended_missing) },
    { type: "", text: list("Not explicit; published default applies", payload.optional_defaulted) },
    { type: payload.inactive_or_removed.length ? "warn" : "", text: list("Inactive/ignored/removed for this model", payload.inactive_or_removed) },
    { type: "", text: list("Checkpoint-owned or checkpoint-validated", payload.checkpoint_owned) },
    { type: payload.unknown_keys.length ? "warn" : "", text: list("Unknown keys", payload.unknown_keys) },
    { type: payload.malformed_lines.length ? "error" : "", text: `Malformed lines (${payload.malformed_lines.length}): ${payload.malformed_lines.length ? payload.malformed_lines.map(item => `line ${item.line}`).join(", ") : "<none>"}` }
  ];
}

export async function configureViaLlm() {
  const node = state.nodes.find(item => item.id === state.configNode);
  if (!node || !requireRuntime()) return;
  const instruction = window.prompt("Describe the change you want the LLM to make to this configuration:");
  if (!instruction || !instruction.trim()) return;
  const modelId = BLOCK_SPECS[node.type].modelId;
  const text = `${rawConfig(node.config, MODEL_CATALOG[modelId].keys)}\n`;
  if (!window.confirm(
    `Send this block's complete ${text.split("\n").filter(Boolean).length}-line configuration and your instruction to the LLM endpoint configured under System?\n\n` +
    "Paths, dataset names, checkpoint locations, and every configured value in this block are included."
  )) return;
  const button = $("#llmConfigure");
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Asking LLM…";
  try {
    const result = await apiRequest("/api/llm/configure", {
      method: "POST",
      body: { config_text: text, instruction: instruction.trim() }
    });
    const preview = result.text.length > 700 ? `${result.text.slice(0, 700)}\n…` : result.text;
    if (!window.confirm(`Apply the LLM's updated configuration for this block?\n\n${preview}`)) {
      toast("LLM suggestion discarded.");
      return;
    }
    const parsed = parseConfig(result.text);
    snapshot();
    node.config = { ...parsed.values, model: modelId, mode: parsed.values.mode || node.config.mode };
    retainExplicitConfig(node);
    state.configMessages = [
      { type: "", text: `LLM applied instruction: ${instruction.trim()}` },
      ...parsed.messages
    ];
    $("#configMode").value = node.config.mode;
    $("#savedState").textContent = "Unsaved changes · LLM edited";
    renderConfig();
    toast("LLM updated the configuration. Review it before saving.");
  } catch (error) {
    toast(`LLM configure failed: ${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

/**
 * Selects the block behind a failed-preflight diagnostic and, for a model
 * block, opens its full config editor scrolled to the exact failing field so
 * a rejected run can be fixed immediately instead of hunting for it.
 */
export function jumpToFailingField(nodeId, field) {
  const node = state.nodes.find(item => item.id === nodeId);
  if (!node) return;
  selectNode(nodeId);
  const spec = BLOCK_SPECS[node.type];
  if (!spec?.isModel || !field) {
    toast(field ? `Selected the failing block for "${field}".` : "Selected the failing block.", "warn");
    return;
  }
  field = String(field).toLowerCase();
  const modelId = spec.modelId;
  const model = MODEL_CATALOG[modelId];
  openConfig(nodeId);
  state.configRejectedNode = nodeId;
  state.configRejectedField = field;
  const mode = node.config.mode || model.modes[0];
  const required = requiredFor(modelId, mode);
  state.configSearch = "";
  $("#configSearch").value = "";
  state.configSection = model.keys.includes(field) ? sectionFor(modelId, field, required) : "Inactive / rejected";
  renderConfig();
  requestAnimationFrame(() => {
    const control = document.querySelector(`.full-config-control[data-key="${CSS.escape(field)}"]`);
    const card = control?.closest(".config-card");
    if (!card) {
      toast(`Opened ${modelId} configuration; "${field}" is not a listed key.`, "warn");
      return;
    }
    card.scrollIntoView({ block: "center", behavior: "smooth" });
    card.classList.add("field-flash");
    control.focus();
    setTimeout(() => card.classList.remove("field-flash"), 1800);
  });
}

export async function explainConfig() {
  const node = state.nodes.find(item => item.id === state.configNode);
  if (!node || !requireRuntime()) return null;
  const modelId = BLOCK_SPECS[node.type].modelId;
  const text = `${rawConfig(node.config, MODEL_CATALOG[modelId].keys)}\n`;
  const result = await apiRequest("/api/config/explain", {
    method: "POST",
    allowError: true,
    body: {
      config: text,
      label: `${modelId}-${node.config.mode || "config"}`,
      skip_filesystem: true,
      skip_native: true,
      skip_environment: true
    }
  });
  return result;
}
