import { $, $$, escapeHtml, toast, closeOverlay } from "./dom.js";
import { keys } from "./text.js";
import { state } from "./state.js";
import {
  ICONS, BLOCK_SPECS, MODEL_CATALOG, REQUIRED, CHOICES, BOOLEAN_KEYS,
  OPERATOR_REMOVED, TRANSOLVER_REJECTED, CONFIG_SECTIONS, HELP
} from "./constants.js";
import { apiRequest, requireRuntime } from "./api.js";
import { addBlock, selectNode } from "./graph.js";

export function requiredFor(modelId, mode) {
  const modelRequired = REQUIRED[modelId]?.[mode];
  if (modelRequired) return new Set(modelRequired);
  return new Set(mode === "inference"
    ? keys(`model mode gpu_ids modelpath infer_dataset input_var output_var`)
    : keys(`model mode gpu_ids dataset_dir modelpath input_var output_var training_epochs batch_size learningr`));
}

export function keyDisposition(modelId, key) {
  if (modelId === "transolver" && TRANSOLVER_REJECTED.has(key)) return "removed";
  if (["point_deeponet", "deeponet", "fno", "gino"].includes(modelId)) {
    if (OPERATOR_REMOVED.has(key)) return "removed";
    const owner = key.startsWith("point") ? "point_deeponet" : key.startsWith("deeponet_") ? "deeponet" : key.startsWith("fno_") ? "fno" : key.startsWith("gino_") ? "gino" : "";
    if (owner && owner !== modelId) return "inactive";
  }
  if (key.startsWith("_") || ["num_timesteps", "num_node_types", "log_dir"].includes(key)) return "runtime";
  return "active";
}

export function sectionFor(modelId, key, required) {
  if (keyDisposition(modelId, key) !== "active") return "Inactive / rejected";
  if (required.has(key)) return "Required";
  if (/dataset|modelpath|output_dir|log_file|pipeline_log|param_dir|input_mesh|sidecar|split_seed/.test(key)) return "Data & output";
  if (/^(point_|pointnet_|deeponet_|fno_|gino_|encoder_|decoder_|fm_arch|fm_blocks|fm_hidden|fm_cond_hidden|latent_|latent_dim|message_passing|slice_num|num_layers|num_heads|attention_kernel|mlp_ratio|coarsening|multiscale|mp_per_level|positional|fourier|operator_dim|global_condition|num_filter|lc_filter|network_size)/.test(key)) return "Architecture";
  if (/training_epochs|learningr|weight_decay|warmup|batch_size|loss|dropout|grad_|noise|ema|kl_|beta|eikonal|surface_weight|normal_weight|alpha|lambda_|free_bits|mmd|recon_loss|recon_iter/.test(key)) return "Training";
  if (/gpu|parallel|worker|prefetch|amp|compile|checkpointing|cache|profile|fsdp|pipeline_microbatches|chunk_size|load_all/.test(key)) return "Resources & runtime";
  if (/^infer|^test|^val_|display|plot|histogram|num_samples|candidate_multiplier|cfg_scale|condition_|cond_|ode_steps|mc_resolution|sample_index|source_num_samples|posterior_noise|timesteps_reduced/.test(key)) return "Inference & evaluation";
  return "Advanced";
}

export function choicesFor(modelId, key) {
  if (key === "model") return [modelId];
  if (key === "mode") return MODEL_CATALOG[modelId].modes;
  if (BOOLEAN_KEYS.has(key) || /^use_/.test(key)) return ["False", "True"];
  if (key === "parallel_mode") {
    if (modelId === "simulgenvae") return ["single", "ddp", "fsdp"];
    if (modelId === "transolver") return ["ddp", "node_shard"];
    if (["fno", "gino"].includes(modelId)) return ["ddp", "model_split"];
    return ["ddp"];
  }
  return CHOICES[key] || null;
}

export function rawConfig(values, catalog) {
  const ordered = catalog.filter(key => Object.hasOwn(values, key));
  const unknown = Object.keys(values).filter(key => !catalog.includes(key)).sort();
  return [...ordered, ...unknown].map(key => `${key.padEnd(29, " ")}${values[key]}`).join("\n");
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
    if (seen.has(match[1])) messages.push({ type: "warn", text: `Duplicate ${match[1]} on line ${index + 1}; last value wins.` });
    seen.add(match[1]);
    values[match[1]] = match[2].trim();
  });
  return { values, messages };
}

export function presetOptions(modelId) {
  const options = [["repository", "Checked-in example"], ["smoke", "Smoke test"], ["low_vram", "Low VRAM"]];
  if (modelId === "meshgraphnets") options.push(["mgn_flat", "Flat MGN"], ["mgn_hi", "HI-MGN"], ["mgn_bsms", "BSMS-GNN"]);
  if (modelId === "sdfflow") options.push(["sdfflow_full", "Full VAE + FM"], ["sdfflow_vae", "VAE only"], ["sdfflow_fm", "Flow matching only"]);
  if (modelId === "simulgenvae") options.push(["simulgen_full", "VAE → LC pipeline"], ["simulgen_vae", "VAE only"], ["simulgen_lc", "LC only"], ["simulgen_reconstruct", "Reconstruct fields"]);
  return options;
}

export function openConfig(nodeId) {
  const node = state.nodes.find(item => item.id === nodeId);
  const spec = node && BLOCK_SPECS[node.type];
  if (!node || !spec?.isModel) return;
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
  const spec = BLOCK_SPECS[node.type];
  const model = MODEL_CATALOG[spec.modelId];
  const mode = $("#configMode").value || node.config.mode || model.modes[0];
  node.config.model = spec.modelId;
  node.config.mode = mode;
  const required = requiredFor(spec.modelId, mode);
  const search = state.configSearch.toLowerCase().trim();
  const changedOnly = $("#changedOnly").checked;
  const showInactive = $("#showInactive").checked;
  const groups = Object.fromEntries(CONFIG_SECTIONS.map(section => [section, []]));
  model.keys.forEach(key => {
    const disposition = keyDisposition(spec.modelId, key);
    if (!showInactive && disposition !== "active") return;
    if (changedOnly && !Object.hasOwn(node.config, key)) return;
    if (search && !key.toLowerCase().includes(search)) return;
    groups[sectionFor(spec.modelId, key, required)].push(key);
  });
  if (!groups[state.configSection]?.length && search) state.configSection = CONFIG_SECTIONS.find(section => groups[section].length) || "Required";

  $("#configSectionList").innerHTML = CONFIG_SECTIONS.map(section => `<button class="config-section-button${state.configSection === section ? " active" : ""}" data-config-section="${section}"><span>${section}</span><small>${groups[section].length}</small></button>`).join("");
  $$("[data-config-section]").forEach(button => button.addEventListener("click", () => {
    state.configSection = button.dataset.configSection;
    renderConfig();
  }));

  const visible = groups[state.configSection];
  $("#configSectionTitle").textContent = state.configSection;
  $("#configSectionMeta").textContent = `${visible.length} visible keys · ${mode} mode`;
  $("#configBadges").innerHTML = `<span class="badge">${model.keys.length} accepted</span><span class="badge warn">${required.size} required</span>`;
  $("#schemaNote").innerHTML = `<strong>${model.keys.length} live keys</strong><br>All MethodSpec keys are present. Closed choices use dropdowns; paths, widths, lists, and open family values remain manual.<br><br>${spec.modelId === "simulgenvae" ? "The live SimulGen route has separate VAE, LC, combined, and reconstruction requirements." : "Shared-family inactive and rejected keys remain visible for diagnostic honesty."}`;

  $("#configFields").innerHTML = visible.length ? visible.map(key => {
    const disposition = keyDisposition(spec.modelId, key);
    const set = Object.hasOwn(node.config, key);
    const requiredKey = required.has(key);
    const status = disposition === "removed" ? "rejected" : disposition === "inactive" ? "inactive" : disposition === "runtime" ? "runtime" : requiredKey ? "required" : set ? "set" : "optional";
    const value = set ? node.config[key] : "";
    const choices = choicesFor(spec.modelId, key);
    const disabled = disposition === "removed" || disposition === "runtime";
    const control = choices
      ? `<select class="config-control full-config-control" data-key="${key}"${disabled ? " disabled" : ""}><option value="">— not set —</option>${choices.map(choice => `<option value="${escapeHtml(choice)}"${String(value).toLowerCase() === String(choice).toLowerCase() ? " selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select>`
      : `<input class="config-control full-config-control" data-key="${key}" value="${escapeHtml(value)}" placeholder="manual value"${disabled ? " disabled" : ""}>`;
    const help = HELP[key] || (disposition === "removed" ? "Known by a shared diagnostic schema, but rejected for this selected model." : disposition === "inactive" ? "Accepted by the shared family schema but configures a different variant." : "Manual input is retained because the live spec does not publish a closed value set for this field.");
    return `<article class="config-card ${disposition}"><header class="config-card-head"><span class="config-key">${key}</span><span class="config-status ${status}">${status}</span></header>${control}<p class="config-help">${escapeHtml(help)}</p></article>`;
  }).join("") : `<div class="inspect-empty" style="height:auto;grid-column:1/-1"><p>No keys match this filter.</p></div>`;

  $$(".full-config-control").forEach(control => control.addEventListener("change", () => {
    if (control.value === "") delete node.config[control.dataset.key];
    else node.config[control.dataset.key] = control.value;
    $("#savedState").textContent = "Unsaved changes";
    $("#configRaw").value = rawConfig(node.config, model.keys);
    renderConfig();
  }));

  $("#configRaw").value = rawConfig(node.config, model.keys);
  const missing = [...required].filter(key => !Object.hasOwn(node.config, key) || node.config[key] === "");
  const unknown = Object.keys(node.config).filter(key => !model.keys.includes(key));
  const diagnostics = [
    { type: "", text: `${model.keys.length} accepted keys loaded; ${Object.keys(node.config).length} currently set.` },
    ...(missing.length ? [{ type: "warn", text: `Missing required for ${mode}: ${missing.join(", ")}` }] : [{ type: "", text: `All required ${mode} keys currently have values.` }]),
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
      node.config = { ...parsed.values };
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
  if (preset === "mgn_hi") values = { use_multiscale: "True", coarsening_type: "voronoi", multiscale_levels: "3" };
  if (preset === "mgn_bsms") values = { use_multiscale: "True", coarsening_type: "bfs", multiscale_levels: "3" };
  if (preset === "sdfflow_vae") values = { mode: "train_vae" };
  if (preset === "sdfflow_fm") values = { mode: "train_fm" };
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
    node.config = { ...parsed.values };
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
    { type: "", text: list("Optional; native default applies", payload.optional_defaulted) },
    { type: payload.inactive_or_removed.length ? "warn" : "", text: list("Inactive/ignored/removed for this model", payload.inactive_or_removed) },
    { type: "", text: list("Checkpoint-owned or checkpoint-validated", payload.checkpoint_owned) },
    { type: payload.unknown_keys.length ? "warn" : "", text: list("Unknown keys", payload.unknown_keys) },
    { type: payload.malformed_lines.length ? "error" : "", text: `Malformed lines (${payload.malformed_lines.length}): ${payload.malformed_lines.length ? payload.malformed_lines.map(item => `line ${item.line}`).join(", ") : "<none>"}` }
  ];
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
