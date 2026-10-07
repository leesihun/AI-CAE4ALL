/**
 * What a block *is*, in the handful of facts a person scanning the canvas
 * wants: the dataset's size and fields, the model's variant and scale, the run
 * a block is wired to and what it produced.
 *
 * The card preview used to be an illustration per block kind -- the same ex9
 * plasticity field on every Inference block, a fake mesh grid on every dataset,
 * a made-up parity scatter on every evaluation -- so two different datasets
 * drew identical cards and nothing on the canvas said what was in either. Every
 * value here is read from the block's own config, its connections, or the
 * dataset file itself (`/api/hdf5/facts`, attributes only), and a fact that is
 * not known yet says so instead of being drawn.
 */
import { state } from "./state.js";
import { BLOCK_SPECS, MODEL_CATALOG, REQUIRED } from "./constants.js";
import { apiRequest } from "./api.js";
import { cadGeneratorValue } from "./validate.js";

const FACTS = new Map();          // repo-relative path -> facts | {error} | "pending"

function fileName(value) {
  const text = typeof value === "string" ? value.trim().replaceAll("\\", "/") : "";
  return text ? text.split("/").filter(Boolean).pop() || text : "";
}

function formatCount(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  if (number >= 1e6) return `${(number / 1e6).toFixed(number >= 1e7 ? 0 : 1)}M`;
  if (number >= 1e4) return `${Math.round(number / 1e3)}k`;
  return number.toLocaleString("en-US");
}

/** The upstream block wired into one input port, if any. */
export function upstream(node, portId) {
  const edge = state.edges.find(item => item.toNode === node.id && item.toPort === portId);
  return edge ? state.nodes.find(item => item.id === edge.fromNode) || null : null;
}

function downstreamPorts(node) {
  return state.edges
    .filter(edge => edge.fromNode === node.id)
    .map(edge => ({ edge, target: state.nodes.find(item => item.id === edge.toNode) }))
    .filter(item => item.target);
}

/** Repo-relative dataset path a source block points at (method-relative "../../" stripped). */
function datasetPath(node) {
  const raw = String(node?.config?.path || node?.config?.output_dataset || "").trim().replaceAll("\\", "/");
  if (!/\.(h5|hdf5)$/i.test(raw)) return "";
  return raw.replace(/^(\.\.\/)+/, "");
}

/**
 * Cached dataset facts for a block, fetched once per path. `onReady` runs when
 * a fetch this call started resolves, so the card can redraw itself.
 */
export function datasetFacts(node, onReady) {
  const path = datasetPath(node);
  if (!path) return null;
  const cached = FACTS.get(path);
  if (cached && cached !== "pending") return cached;
  if (!cached && state.api?.connected) {
    FACTS.set(path, "pending");
    apiRequest(`/api/hdf5/facts?path=${encodeURIComponent(path)}`)
      .then(facts => FACTS.set(path, facts))
      .catch(error => FACTS.set(path, { error: error.message }))
      .finally(() => onReady?.());
  }
  return null;
}

function modelMode(node) {
  const spec = BLOCK_SPECS[node.type];
  return (value(node, "mode") || MODEL_CATALOG[spec?.modelId]?.modes?.[0] || "train").toLowerCase();
}

/**
 * Whether a model block's current mode opens the file on its data port. The
 * spec's per-mode `required` fields are the evidence: SDFFlow's sample,
 * reconstruct, interpolate and optimize need neither dataset_dir nor
 * infer_dataset and never read it, yet a pipeline trained in one mode and
 * switched to optimize kept the wire, and its dataset card said "Held-out set".
 */
function modeReadsData(node) {
  const modelId = BLOCK_SPECS[node.type]?.modelId;
  const mode = modelMode(node);
  const required = MODEL_CATALOG[modelId]?.required?.[mode] || REQUIRED[modelId]?.[mode];
  if (!required) return true;
  const fields = new Set(required);
  return fields.has("dataset_dir") || fields.has("infer_dataset");
}

/** Which role a dataset block plays, from where it is wired (then its name). */
export function datasetRole(node) {
  const roles = new Set();
  downstreamPorts(node).forEach(({ edge, target }) => {
    const spec = BLOCK_SPECS[target.type];
    if (spec?.isModel && edge.toPort === "data") {
      const mode = modelMode(target);
      if (!modeReadsData(target)) roles.add(`Not read in ${mode} mode`);
      else roles.add(mode.startsWith("train") ? "Training set" : "Held-out set");
    } else if (target.type === "run.inference" && edge.toPort === "data") roles.add("Held-out set");
    else if (target.type === "evaluate.predictions" && edge.toPort === "truth") roles.add("Ground truth");
    else if (target.type === "deploy.api") roles.add("Sample input");
  });
  if (roles.size) return [...roles].join(" + ");
  const name = fileName(node.config?.path || "");
  if (/_infer\.h5$|_test\.h5$/i.test(name)) return "Held-out set";
  return name ? "Dataset" : "";
}

function value(node, key) {
  const config = node.config || {};
  const raw = key in config ? config[key] : BLOCK_SPECS[node.type]?.defaults?.[key];
  return raw == null ? "" : String(raw).trim();
}

function truthy(text) {
  return ["true", "1", "yes"].includes(String(text).toLowerCase());
}

/** A model's variant in words: "HI-MGN · 2 levels", "Transolver3 · 64 slices". */
export function modelVariant(node) {
  const spec = BLOCK_SPECS[node.type];
  const id = spec?.modelId;
  if (!id) return "";
  if (id === "meshgraphnets" || id === "meshgraphnets-v" || id === "chi-mgnflow") {
    const levels = value(node, "multiscale_levels");
    const hier = truthy(value(node, "use_multiscale"));
    const base = id === "meshgraphnets" ? (hier ? "HI-MGN" : "Flat MGN") : MODEL_CATALOG[id].label;
    return hier && levels ? `${base} · ${levels}-level V-cycle` : base;
  }
  if (id === "sdfflow") return `VecSet VAE + ${value(node, "fm_arch") === "dit" ? "DiT" : "MLP"} flow`;
  return MODEL_CATALOG[id]?.label || id;
}

/** Architecture / scale rows for a model block, ordered by what distinguishes the family. */
function modelRows(node) {
  const id = BLOCK_SPECS[node.type].modelId;
  const rows = [];
  const add = (label, key, suffix = "") => {
    const text = value(node, key);
    if (text) rows.push([label, `${text}${suffix}`]);
  };
  if (["meshgraphnets", "meshgraphnets-v", "chi-mgnflow"].includes(id)) {
    if (truthy(value(node, "use_multiscale")) && value(node, "mp_per_level")) add("MP per level", "mp_per_level");
    else add("message passing", "message_passing_num", " steps");
    add("latent width", "latent_dim");
    if (id === "meshgraphnets-v") add("VAE latent", "vae_latent_dim");
  } else if (id === "transolver") {
    add("layers × heads", "num_layers");
    if (rows.length && value(node, "num_heads")) rows[rows.length - 1][1] += ` × ${value(node, "num_heads")}`;
    add("slices", "slice_num");
    add("latent width", "latent_dim");
  } else if (id === "fno") {
    add("layers", "fno_layers"); add("modes", "fno_modes"); add("width", "fno_hidden_channels");
  } else if (id === "deeponet") {
    add("basis", "deeponet_basis_dim"); add("width", "deeponet_hidden_channels");
  } else if (id === "point_deeponet") {
    add("sensors", "point_sensor_count"); add("width", "point_hidden_channels");
  } else if (id === "mlp") {
    add("hidden layers", "hidden_layers"); add("activation", "activation");
  } else if (id === "simulgenvae") {
    add("latent", "latent_dim"); add("hier. latent", "latent_dim_end");
  }
  const mode = value(node, "mode").toLowerCase() || "train";
  if (id === "sdfflow") {
    add("latent tokens", "latent_tokens");
    if (value(node, "latent_dim")) rows[rows.length - 1] = ["latent", `${value(node, "latent_tokens")} × ${value(node, "latent_dim")}`];
    // kl_weight is a training key; outside training its row would push the
    // checkpoint rows past the card's four.
    if (mode.startsWith("train")) add("KL weight", "kl_weight");
  }
  if (mode.startsWith("train")) {
    const epochs = value(node, "training_epochs") || value(node, "vae_training_epochs");
    const lr = value(node, "learningr") || value(node, "vae_learningr");
    if (id === "sdfflow" && mode === "train") {
      rows.push(["epochs", `VAE ${value(node, "vae_training_epochs") || "?"} · FM ${value(node, "fm_training_epochs") || "?"}`]);
    } else if (id === "simulgenvae" && mode === "train") {
      rows.push(["epochs", `VAE ${value(node, "vae_training_epochs") || "?"} · LC ${value(node, "lc_training_epochs") || "?"}`]);
    } else if (epochs) rows.push(["epochs · lr", `${epochs}${lr ? ` · ${lr}` : ""}`]);
  } else if (id === "sdfflow") {
    // Every non-training mode decodes through the VAE; only these read the FM
    // (evaluate does unless eval_task is reconstruction, its default).
    const task = value(node, "eval_task").toLowerCase() || "reconstruction";
    const readsFm = ["sample", "interpolate", "optimize"].includes(mode) || (mode === "evaluate" && task !== "reconstruction");
    const vae = fileName(value(node, "vae_modelpath"));
    const fm = fileName(value(node, "fm_modelpath"));
    if (vae) rows.push(["VAE checkpoint", vae]);
    if (readsFm && fm) rows.push(["FM checkpoint", fm]);
  } else {
    const ckpt = fileName(value(node, "modelpath") || value(node, "vae_modelpath"));
    if (ckpt) rows.push(["checkpoint", ckpt]);
  }
  return rows;
}

function datasetRows(node, facts) {
  if (!facts) {
    return datasetPath(node)
      ? [["file", fileName(node.config.path || node.config.output_dataset)], ["contents", state.api?.connected ? "reading…" : "runtime offline"]]
      : [["file", "none selected"]];
  }
  if (facts.error) return [["file", fileName(node.config.path || node.config.output_dataset)], ["contents", "not readable yet"]];
  const rows = [];
  if (facts.contract === "mesh_state") {
    const nodes = facts.nodes_min == null ? "" : facts.nodes_min === facts.nodes_max
      ? `${formatCount(facts.nodes_min)} nodes`
      : `${formatCount(facts.nodes_min)}–${formatCount(facts.nodes_max)} nodes`;
    rows.push(["samples", `${formatCount(facts.num_samples)}${nodes ? ` · ${nodes}` : ""}${facts.num_timesteps > 1 ? ` · T=${facts.num_timesteps}` : " · static"}`]);
    const names = facts.feature_names || [];
    const state0 = 3;
    const inVar = facts.input_var ?? null;
    const condVar = facts.cond_var ?? 0;
    if (inVar != null && names.length >= state0 + inVar) {
      rows.push(["fields", names.slice(state0, state0 + inVar).join(", ") || "—"]);
      if (condVar) rows.push(["conditions", names.slice(state0 + inVar, state0 + inVar + condVar).join(", ")]);
    } else if (names.length > state0) {
      rows.push(["channels", names.slice(state0).join(", ")]);
    }
  } else if (facts.contract === "sdf_shapes") {
    rows.push(["shapes", formatCount(facts.num_samples)]);
    if (facts.sdf_points) rows.push(["SDF points", `${formatCount(facts.sdf_points)} / shape`]);
    const conditions = facts.condition_names || [];
    if (conditions.length) rows.push(["conditions", conditions.length > 5 ? `${conditions.slice(0, 4).join(", ")} +${conditions.length - 4}` : conditions.join(", ")]);
  } else if (facts.contract === "table") {
    rows.push(["rows", formatCount(facts.num_samples)]);
    rows.push(["inputs", `${(facts.input_names || []).length}: ${(facts.input_names || []).slice(0, 3).join(", ")}${(facts.input_names || []).length > 3 ? "…" : ""}`]);
    rows.push(["outputs", `${(facts.output_names || []).length}: ${(facts.output_names || []).slice(0, 3).join(", ")}${(facts.output_names || []).length > 3 ? "…" : ""}`]);
  } else if (facts.contract === "operator_grid") {
    rows.push(["samples", `${formatCount(facts.num_samples)} · ${formatCount(facts.grid_points)} grid points`]);
  } else {
    rows.push(["contents", "no recognised dataset layout"]);
  }
  return rows;
}

function linkedLabel(node) {
  if (!node) return "";
  const spec = BLOCK_SPECS[node.type];
  if (spec?.isModel) return modelVariant(node);
  if (node.type === "source.checkpoint") return fileName(node.config?.path) || "checkpoint";
  if (node.type === "source.hdf5") return fileName(node.config?.path) || "dataset";
  return spec?.label || node.type;
}

/** Up to four [label, value] rows describing this block. */
export function blockFacts(node, onReady) {
  const spec = BLOCK_SPECS[node.type];
  const config = node.config || {};
  if (!spec) return [];
  if (node.type === "source.hdf5") return datasetRows(node, datasetFacts(node, onReady));
  if (node.type === "prep.geometry") {
    const rows = [["mode", `${value(node, "mode")} · ${value(node, "emit")}`], ["mesher", value(node, "reader")]];
    if (value(node, "mode") === "ingest") rows.push(["writes", fileName(value(node, "output_dataset"))]);
    return rows;
  }
  if (spec.isModel) {
    const rows = modelRows(node);
    const data = upstream(node, "data");
    if (data) {
      const name = fileName(data.config?.path) || linkedLabel(data);
      rows.unshift(["data", modeReadsData(node) ? name : `not read in ${modelMode(node)}`]);
    }
    return rows.slice(0, 4);
  }
  if (node.type === "run.inference") {
    const model = upstream(node, "model");
    const family = model && BLOCK_SPECS[model.type]?.isModel ? modelVariant(model) : config.model_id ? (MODEL_CATALOG[config.model_id]?.label || config.model_id) : "";
    const data = upstream(node, "data");
    const rows = [["model", family || (model ? linkedLabel(model) : "not connected")], ["predicts on", data ? linkedLabel(data) : "not connected"]];
    if (config.results_path) rows.push(["results", config.results_samples ? `${config.results_samples} samples · ${fileName(config.results_path)}` : fileName(config.results_path)]);
    return rows;
  }
  if (node.type === "evaluate.predictions") {
    const prediction = upstream(node, "prediction");
    const truth = upstream(node, "truth");
    const rows = [["predictions", prediction ? linkedLabel(upstream(prediction, "model") || prediction) : "not connected"], ["truth", truth ? linkedLabel(truth) : "not connected"]];
    if (config.report_path) rows.push(["report", `${config.evaluated_samples || "?"} samples scored`]);
    return rows;
  }
  if (node.type === "evaluate.training_metrics") {
    const source = upstream(node, "metrics");
    return [["run", source ? linkedLabel(source) : "not connected"], ["job", config.job_id ? String(config.job_id).slice(0, 12) : "no run yet"]];
  }
  if (node.type === "evaluate.compare") {
    const runs = state.edges.filter(edge => edge.toNode === node.id).length;
    return [["runs", runs ? `${runs} connected` : "none connected"], ["rule", "same held-out set"]];
  }
  if (node.type === "optimize.design") {
    const rows = [["objectives", `${config.objectives || "—"} (${config.directions || "—"})`]];
    rows.push(["constraints", config.constraints || "none"]);
    rows.push(["keep", `top ${config.top_k || "?"} feasible Pareto`]);
    return rows;
  }
  if (node.type === "run.cad_generator") {
    const mode = value(node, "mode");
    const rows = [["mode", mode]];
    if (mode === "optimize") {
      // The values the run uses: an opt_* key left on the SDFFlow block by a
      // graph saved before these rows moved here still counts.
      const cad = key => String(cadGeneratorValue(node, key) ?? "").trim();
      rows.push(["analysis", cad("opt_analysis") === "surrogate" ? "HI-MGN surrogate" : "FEA (gmsh + linear static)"]);
      // opt_budget 0 is the screen that writes screening.csv for the
      // Optimization block; anything above it is a CMA-ES search.
      const budget = cad("opt_budget");
      rows.push(["search", budget === "" ? "native default budget"
        : Number(budget) === 0 ? `screening · ${cad("opt_baseline_size") || "default"} designs`
          : `CMA-ES · ${budget} evaluations`]);
    } else {
      // cfg_scale only acts on a conditional request: with no cond_values the
      // sampler integrates the null-condition branch and the scale is inert.
      const conditioned = String(config.cond_values || "").trim() !== "";
      rows.push(["candidates", `${value(node, "num_samples")} · ${conditioned ? `cfg ${value(node, "cfg_scale")}` : "unconditional"}`]);
    }
    rows.push(["mesh", `marching cubes ${value(node, "mc_resolution")}³`]);
    if (config.results_path || config.results_dir) {
      const noun = mode === "optimize" ? "designs" : "candidates";
      rows.push(["results", config.results_samples
        ? `${config.results_samples} ${noun} · ${fileName(config.results_dir || config.results_path)}`
        : fileName(config.results_dir || config.results_path)]);
    }
    return rows;
  }
  if (node.type === "source.checkpoint") {
    return [["file", fileName(config.path) || "none selected"], ["family", config.compatibility && !/auto/i.test(config.compatibility) ? config.compatibility : "read from the checkpoint"]];
  }
  if (node.type === "source.cad") return [["file", fileName(config.path) || "none selected"]];
  if (node.type === "output.export") {
    const input = upstream(node, "input");
    return [["exports", input ? linkedLabel(input) : "not connected"], ["as", "copy / ZIP, no conversion"], ...(config.export_path ? [["written", fileName(config.export_path)]] : [])];
  }
  if (node.type === "deploy.api") {
    const model = upstream(node, "model");
    return [["model", model ? linkedLabel(model) : fileName(config.checkpoint_path) || "not connected"], ["name", config.output_name || "—"]];
  }
  return [];
}

/** One-line subtitle under a card title. */
export function blockSubtitle(node) {
  const spec = BLOCK_SPECS[node.type];
  if (!spec) return "";
  if (spec.isModel) return `${modelVariant(node)} · ${value(node, "mode") || MODEL_CATALOG[spec.modelId].modes[0]}`;
  if (node.type === "source.hdf5") return [datasetRole(node), fileName(node.config?.path)].filter(Boolean).join(" · ");
  return spec.category;
}

export function factsTable(rows, className = "block-facts") {
  const escape = text => String(text ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
  if (!rows.length) return "";
  return `<dl class="${className}">${rows.map(([label, text]) => `<div><dt>${escape(label)}</dt><dd title="${escape(text)}">${escape(text)}</dd></div>`).join("")}</dl>`;
}
