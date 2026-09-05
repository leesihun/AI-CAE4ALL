import { BLOCK_SPECS, MODEL_CATALOG } from "./constants.js";
import { state } from "./state.js";
import { checkpointMetadata, checkpointMetaNow } from "./api.js";

const CSV_EXT = /\.csv$/i;
const IMAGE_EXT = /\.(?:png|jpe?g|bmp|tiff?|webp)$/i;

function text(value) {
  return String(value ?? "").trim();
}

export function toMethodPath(value, modelId = "") {
  const normalized = text(value).replaceAll("\\", "/");
  if (!normalized || normalized.startsWith("../") || /^[A-Za-z]:\//.test(normalized)) return normalized;
  const live = state.api.models.find(item => item.model === modelId);
  const depth = text(live?.repository).split("/").filter(Boolean).length || 2;
  return `${"../".repeat(depth)}${normalized.replace(/^\.\//, "")}`;
}

export function toGeometryPath(value) {
  const normalized = text(value).replaceAll("\\", "/");
  if (!normalized || normalized.startsWith("../") || /^[A-Za-z]:\//.test(normalized)) return normalized;
  return `../../${normalized.replace(/^\.\//, "")}`;
}

function resolveMethodRelative(value, repository) {
  const normalized = text(value).replaceAll("\\", "/");
  if (!normalized || /^[A-Za-z]:\//.test(normalized)) return normalized;
  const parts = `${repository}/${normalized}`.split("/");
  const resolved = [];
  parts.forEach(part => {
    if (!part || part === ".") return;
    if (part === "..") resolved.pop();
    else resolved.push(part);
  });
  return resolved.join("/");
}

/** Convert a path stored relative to a method repository back to the suite
 * root contract used by Studio APIs.  Most shipped model configs write shared
 * artifacts as ../output/..., while a method-local outputs/... path needs the
 * live registry's repository prefix. */
export function fromMethodPath(value, modelId) {
  const normalized = text(value).replaceAll("\\", "/");
  if (!normalized || /^[A-Za-z]:\//.test(normalized)) return normalized;
  const live = state.api.models.find(item => item.model === modelId);
  const repository = text(live?.repository).replaceAll("\\", "/").replace(/^\.\//, "").replace(/\/$/, "");
  return repository
    ? resolveMethodRelative(normalized, repository)
    : normalized.replace(/^(?:\.\.\/){2}/, "").replace(/^\.\//, "");
}

/** Inverse of toGeometryPath: geometry_ingest's own output_dataset default is
 * stored method-relative (relative to methods/GeometryIngest/, two levels
 * below the suite root). Other blocks (Export, Evaluate, HDF5 viewers) all
 * resolve paths suite-root-relative, so propagating that raw value verbatim
 * escapes the allowed repository roots. */
export function fromGeometryPath(value) {
  const normalized = text(value).replaceAll("\\", "/");
  if (!normalized || /^[A-Za-z]:\//.test(normalized)) return normalized;
  return resolveMethodRelative(normalized, "methods/GeometryIngest");
}

function configPath(node, keys) {
  for (const key of keys) {
    const value = text(node?.config?.[key]);
    if (value) return value;
  }
  return "";
}

function linkedInputs(node, portId = null) {
  return state.edges
    .filter(edge => edge.toNode === node.id && (!portId || edge.toPort === portId))
    .map(edge => ({
      edge,
      node: state.nodes.find(candidate => candidate.id === edge.fromNode)
    }))
    .filter(item => item.node);
}

function linkedOutputs(node, portId = null) {
  return state.edges
    .filter(edge => edge.fromNode === node.id && (!portId || edge.fromPort === portId))
    .map(edge => ({
      edge,
      node: state.nodes.find(candidate => candidate.id === edge.toNode)
    }))
    .filter(item => item.node);
}

function labelFor(node, port = "") {
  const label = BLOCK_SPECS[node.type]?.label || node.id;
  return port ? `${label} · ${port}` : label;
}

function candidate(value, source, reason, port = "") {
  const normalized = text(value);
  if (!normalized || !source) return null;
  return {
    value: normalized,
    sourceNodeId: source.id,
    sourceLabel: labelFor(source, port),
    reason
  };
}

function put(desired, node, key, item) {
  if (!item) return;
  if (!desired.has(node.id)) desired.set(node.id, {});
  desired.get(node.id)[key] = item;
}

function parameterColumns(node) {
  try {
    const table = JSON.parse(node?.config?.parameter_table || "null");
    if (!Array.isArray(table?.columns)) return { inputs: [], outputs: [] };
    return {
      inputs: table.columns.filter(column => column?.kind === "input"),
      outputs: table.columns.filter(column => column?.kind === "output")
    };
  } catch {
    return { inputs: [], outputs: [] };
  }
}

const FINITE_DECIMAL = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;

/**
 * Resolve the one spreadsheet row explicitly chosen as an execution candidate.
 *
 * SDFFlow's native `cond_values` contract is positional, so Input columns are
 * read left-to-right exactly as they appear in `parameter_table.columns`.
 * Output columns are deliberately excluded. Keeping the resolution here gives
 * the viewer, inspector, validation, and executable-config paths one meaning of
 * "selected candidate" instead of four subtly different parsers.
 */
export function selectedParameterCandidate(node) {
  let table = null;
  try { table = JSON.parse(node?.config?.parameter_table || "null"); } catch { /* Report the empty selection below. */ }
  const selectedSampleId = text(table?.selected_sample_id);
  const rows = Array.isArray(table?.rows) ? table.rows : [];
  const row = selectedSampleId
    ? rows.find(candidate => text(candidate?.sample_id) === selectedSampleId) || null
    : null;
  const columns = Array.isArray(table?.columns)
    ? table.columns.filter(column => column?.kind === "input")
    : [];
  const inputs = columns.map(column => {
    const name = text(column.name);
    const rawValue = text(row?.values?.[column.id]);
    const numericValue = FINITE_DECIMAL.test(rawValue) ? Number(rawValue) : Number.NaN;
    return {
      id: String(column.id || ""),
      name,
      value: rawValue,
      numericValue,
      valid: Boolean(rawValue) && Number.isFinite(numericValue)
    };
  });
  const normalizedNames = inputs.map(input => input.name.toLowerCase()).filter(Boolean);
  const duplicateNames = [...new Set(normalizedNames.filter((name, index) => normalizedNames.indexOf(name) !== index))];
  const missingNames = inputs.filter(input => !input.name);
  const invalidInputs = inputs.filter(input => !input.valid);
  const ready = Boolean(row) && inputs.length > 0
    && !missingNames.length && !duplicateNames.length && !invalidInputs.length;
  return {
    table,
    selectedSampleId,
    row,
    inputs,
    missingNames,
    duplicateNames,
    invalidInputs,
    conditionNames: inputs.map(input => input.name).join(","),
    condValues: ready ? inputs.map(input => input.value).join(",") : "",
    ready
  };
}

function commaNames(node, key) {
  return text(node?.config?.[key]).split(",").map(value => value.trim()).filter(Boolean);
}

function datasetPath(node) {
  if (!node) return "";
  if (node.type === "source.hdf5") return configPath(node, ["path"]);
  if (node.type === "prep.geometry") return fromGeometryPath(configPath(node, ["output_dataset"]));
  return configPath(node, ["dataset_path", "truth_path", "path"]);
}

/**
 * Ask the runtime what a checkpoint is, and re-run autofill once it answers.
 *
 * Synchronous by necessity -- `desiredAutofill` cannot await -- so the first
 * pass sees nothing and requests the read; the resolution re-enters autofill
 * with the answer cached. Both `checkpointMetadata` and the notifier are
 * idempotent per path, so this cannot loop.
 */
let onCheckpointResolved = null;

export function registerCheckpointRefresh(hook) {
  onCheckpointResolved = typeof hook === "function" ? hook : null;
}

const checkpointSubscribed = new Set();

function checkpointFacts(path) {
  const value = text(path);
  if (!value) return null;
  const known = checkpointMetaNow(value);
  if (known) return known.ok ? known : null;
  if (checkpointSubscribed.has(value)) return null;
  const pending = checkpointMetadata(value);
  if (pending && typeof pending.then === "function") {
    // One subscription per path, ever. Autofill runs several passes per call
    // and is itself re-entered by the refresh, so subscribing per call would
    // multiply the callbacks instead of adding one.
    checkpointSubscribed.add(value);
    pending.then(() => onCheckpointResolved?.());
  }
  return null;
}

function checkpointPaths(node) {
  if (!node) return {};
  if (node.type === "source.checkpoint") {
    const path = configPath(node, ["path"]);
    // The saved model names its own family. Without this, an Inference block
    // fed from a checkpoint block had no way to know which method to launch,
    // so it emitted no run step at all.
    return { checkpoint_path: path, model_id: checkpointFacts(path)?.model || "" };
  }
  const spec = BLOCK_SPECS[node.type];
  if (spec?.isModel) {
    return {
      model_id: spec.modelId,
      checkpoint_path: configPath(node, ["modelpath"]),
      vae_modelpath: configPath(node, ["vae_modelpath"]),
      lc_modelpath: configPath(node, ["lc_modelpath"]),
      fm_modelpath: configPath(node, ["fm_modelpath"])
    };
  }
  return {
    checkpoint_path: configPath(node, ["checkpoint_path", "modelpath", "path"]),
    vae_modelpath: configPath(node, ["vae_modelpath"]),
    lc_modelpath: configPath(node, ["lc_modelpath"]),
    fm_modelpath: configPath(node, ["fm_modelpath"]),
    model_id: configPath(node, ["model_id"])
  };
}

function outputArtifactPath(node) {
  if (!node) return "";
  if (node.type === "source.hdf5" || node.type === "source.cad" || node.type === "source.checkpoint") return configPath(node, ["path"]);
  if (node.type === "prep.geometry") return fromGeometryPath(configPath(node, ["output_dataset"]));
  // `results_path` first: it is the only key that names where a finished run
  // actually wrote its predictions, so without it the Evaluate Predictions block
  // downstream of an Inference block received no prediction_path at all and had
  // nothing to score.
  return configPath(node, [
    "results_path", "report_path", "metrics_csv", "prediction_path", "output_path",
    "output_csv", "candidate_csv", "csv_path", "source_path", "path"
  ]) || text(node.optimizationReport);
}

function actualBindingPath(node) {
  const value = configPath(node, ["binding"]);
  if (!value || /^(?:csv condition columns|no input selected)$/i.test(value)) return "";
  return /[\\/]/.test(value) || /\.[a-z0-9]{2,6}$/i.test(value) || /^[A-Za-z]:/.test(value) ? value : "";
}

function modelAutofill(desired, node) {
  const spec = BLOCK_SPECS[node.type];
  const model = MODEL_CATALOG[spec.modelId];
  const accepted = new Set(model.keys);
  const mode = text(node.config.mode || model.modes[0]).toLowerCase();
  const dataLink = linkedInputs(node, "data")[0];
  const parameterLink = linkedInputs(node, "parameters")[0];
  const checkpointLink = linkedInputs(node, "resume")[0];

  if (dataLink) {
    const path = datasetPath(dataLink.node);
    const key = mode === "inference" && accepted.has("infer_dataset") ? "infer_dataset" : accepted.has("dataset_dir") ? "dataset_dir" : "";
    // Same rule validate.js applies when building the inference step: the graph
    // is authoritative EXCEPT when what it says is "predict the data you
    // trained on". Switching a template block to inference used to overwrite
    // its held-out infer_dataset with the training file wired to the data port,
    // and switching back then withdrew the autofill and left the key deleted --
    // silently, with the diagnostics panel still reporting everything present.
    const wired = key && path ? toMethodPath(path, spec.modelId) : "";
    // Deliberately not compared against node.config.dataset_dir: by the time
    // this runs in inference mode that key has already been withdrawn, so the
    // comparison always passed and the overwrite happened anyway. A block that
    // already names a held-out split keeps it -- no model's defaults set
    // infer_dataset, so this only ever protects a value a template or the user
    // put there, and editing the field directly still wins (manual keys).
    const configuredSplit = text(node.config.infer_dataset);
    const wouldPredictTrainingData = key === "infer_dataset"
      && configuredSplit && wired && wired !== configuredSplit;
    if (wired && !wouldPredictTrainingData) {
      put(desired, node, key, candidate(wired, dataLink.node, `${mode} dataset from graph`, dataLink.edge.fromPort));
    }

    const featureNames = commaNames(dataLink.node, "feature_names");
    if (spec.modelId === "simulgenvae" && accepted.has("num_var") && featureNames.length) {
      put(desired, node, "num_var", candidate(featureNames.length, dataLink.node, "physical field count from HDF5 metadata", dataLink.edge.fromPort));
    }
    const conditionNames = commaNames(dataLink.node, "condition_names");
    if (spec.modelId === "sdfflow" && accepted.has("condition_names") && conditionNames.length) {
      put(desired, node, "condition_names", candidate(conditionNames.join(","), dataLink.node, "condition names from HDF5 metadata", dataLink.edge.fromPort));
    }
  }

  if (parameterLink) {
    const source = parameterLink.node;
    const binding = actualBindingPath(source);
    const columns = parameterColumns(source);
    const inputNames = columns.inputs.map(column => text(column.name)).filter(Boolean);
    const outputNames = columns.outputs.map(column => text(column.name)).filter(Boolean);
    const conditions = inputNames.length ? inputNames : commaNames(source, "condition_names");
    const features = outputNames.length ? outputNames : commaNames(source, "feature_names");

    if (accepted.has("param_dir") && binding) {
      put(desired, node, "param_dir", candidate(toMethodPath(binding, spec.modelId), source, "condition data from graph", parameterLink.edge.fromPort));
    }
    if (accepted.has("lc_data_type") && binding) {
      const kind = CSV_EXT.test(binding) ? "csv" : IMAGE_EXT.test(binding) || !/\.[a-z0-9]{2,6}$/i.test(binding) ? "image" : "";
      if (kind) put(desired, node, "lc_data_type", candidate(kind, source, "condition encoding inferred from path", parameterLink.edge.fromPort));
    }
    if (accepted.has("condition_names") && conditions.length) {
      put(desired, node, "condition_names", candidate(conditions.join(","), source, "condition columns from spreadsheet", parameterLink.edge.fromPort));
    }
    if (spec.modelId === "mlp") {
      if (columns.inputs.length) put(desired, node, "input_var", candidate(columns.inputs.length, source, "MLP input column count", parameterLink.edge.fromPort));
      if (columns.outputs.length) put(desired, node, "output_var", candidate(columns.outputs.length, source, "MLP output column count", parameterLink.edge.fromPort));
    }
    if (spec.modelId === "simulgenvae" && features.length) {
      put(desired, node, "num_var", candidate(features.length, source, "physical output column count", parameterLink.edge.fromPort));
    }
  }

  if (checkpointLink) {
    const source = checkpointLink.node;
    const path = checkpointPaths(source).checkpoint_path;
    const filename = path.split(/[\\/]/).at(-1)?.toLowerCase() || "";
    if (path && accepted.has("modelpath")) {
      put(desired, node, "modelpath", candidate(toMethodPath(path, spec.modelId), source, "resume checkpoint from graph", checkpointLink.edge.fromPort));
    } else if (path && spec.modelId === "simulgenvae") {
      const key = /(?:^|[_-])lc(?:[_-]|\.)|condition/.test(filename)
        ? "lc_modelpath"
        : /vae/.test(filename) || mode === "train_lc" ? "vae_modelpath" : "";
      if (key) put(desired, node, key, candidate(toMethodPath(path, spec.modelId), source, `${key === "vae_modelpath" ? "VAE" : "conditioner"} checkpoint from graph`, checkpointLink.edge.fromPort));
    } else if (path && spec.modelId === "sdfflow") {
      const key = /(?:^|[_-])fm(?:[_-]|\.)|flow/.test(filename)
        ? "fm_modelpath"
        : /vae/.test(filename) || ["train_fm", "reconstruct"].includes(mode) ? "vae_modelpath" : "";
      if (key) put(desired, node, key, candidate(toMethodPath(path, spec.modelId), source, `${key === "vae_modelpath" ? "VAE" : "flow"} checkpoint from graph`, checkpointLink.edge.fromPort));
    }
  }
}

function designParameterAutofill(desired, node) {
  let dataset = linkedOutputs(node, "parameters").find(item => item.node.type === "source.hdf5");
  if (!dataset) {
    const modelLink = linkedOutputs(node, "parameters").find(item => BLOCK_SPECS[item.node.type]?.isModel);
    if (modelLink) dataset = linkedInputs(modelLink.node, "data")[0];
  }
  const path = datasetPath(dataset?.node);
  if (path) put(desired, node, "parameter_dataset", candidate(path, dataset.node, "row order locked to connected HDF5 dataset", dataset.edge.fromPort));
}

function genericAutofill(desired, node) {
  if (node.type === "source.checkpoint") {
    // Read back what the file actually is, so a saved model on the canvas says
    // which method trained it instead of the placeholder "auto-detect".
    const facts = checkpointFacts(configPath(node, ["path"]));
    if (facts?.model) {
      const label = MODEL_CATALOG[facts.model]?.label || facts.model;
      const epoch = Number.isFinite(Number(facts.epoch)) ? ` · epoch ${facts.epoch}` : "";
      put(desired, node, "model_id", candidate(facts.model, node, "model family recorded in the checkpoint"));
      put(desired, node, "compatibility", candidate(`${label}${epoch}`, node, `resolved from ${facts.model_source || "the checkpoint"}`));
    }
    return;
  }

  if (node.type === "prep.geometry") {
    const link = linkedInputs(node, "geometry")[0];
    const path = configPath(link?.node, ["path"]);
    if (path) put(desired, node, "input_geometry", candidate(toGeometryPath(path), link.node, "geometry input from graph", link.edge.fromPort));
    return;
  }

  if (node.type === "run.inference") {
    const data = linkedInputs(node, "data")[0];
    const model = linkedInputs(node, "model")[0];
    const parameters = linkedInputs(node, "parameters")[0];
    const dataValue = datasetPath(data?.node);
    if (dataValue) put(desired, node, "dataset_path", candidate(dataValue, data.node, "inference dataset from graph", data.edge.fromPort));
    if (model) {
      const paths = checkpointPaths(model.node);
      for (const key of ["checkpoint_path", "vae_modelpath", "lc_modelpath", "fm_modelpath", "model_id"]) {
        if (paths[key]) put(desired, node, key, candidate(paths[key], model.node, "inference model from graph", model.edge.fromPort));
      }
      // Name the output directory explicitly, beside the checkpoint -- the
      // convention every checked-in benchmark config already follows
      // (.../gino/model.pth -> .../gino/inference). Left blank, the native
      // default ("../../output/<slug>/rollout") is invisible in the config text
      // the Studio reads back, and its fallback scan could not find it either
      // once the natives stopped writing inside their own repositories -- so a
      // pipeline trained for hours, inferred 87 scenes, and then failed at
      // Evaluate with "prediction_path is empty". A Studio-run inference now
      // always says where it wrote.
      const accepts = MODEL_CATALOG[paths.model_id]?.keys?.includes("inference_output_dir");
      const checkpoint = String(paths.checkpoint_path || "").replaceAll("\\", "/");
      const checkpointDir = checkpoint.replace(/\/[^/]*$/, "");
      if (accepts && checkpointDir && checkpointDir !== checkpoint) {
        put(desired, node, "inference_output_dir", candidate(
          toMethodPath(`${checkpointDir}/inference`, paths.model_id), model.node, "beside the checkpoint from graph", model.edge.fromPort));
      }
    }
    const binding = actualBindingPath(parameters?.node);
    if (binding) put(desired, node, "parameter_path", candidate(binding, parameters.node, "inference parameters from graph", parameters.edge.fromPort));
    return;
  }

  if (node.type === "run.cad_generator") {
    const model = linkedInputs(node, "model")[0];
    const parameters = linkedInputs(node, "parameters")[0];
    if (model) {
      const paths = checkpointPaths(model.node);
      for (const key of ["checkpoint_path", "vae_modelpath", "fm_modelpath", "model_id"]) {
        if (paths[key]) put(desired, node, key, candidate(paths[key], model.node, "generator model from graph", model.edge.fromPort));
      }
    }
    const binding = actualBindingPath(parameters?.node);
    if (binding) put(desired, node, "parameter_path", candidate(binding, parameters.node, "generation conditions from graph", parameters.edge.fromPort));
    const selected = selectedParameterCandidate(parameters?.node);
    const tableNames = selected.inputs.map(input => input.name).filter(Boolean);
    const names = tableNames.length ? tableNames : commaNames(parameters?.node, "condition_names");
    if (names.length) put(desired, node, "condition_names", candidate(names.join(","), parameters.node, "condition columns from spreadsheet", parameters.edge.fromPort));
    if (selected.row) {
      const selectionLabel = `${selected.row.sample_label || `Dataset row ${selected.selectedSampleId}`} (ID: ${selected.row.sample_id})`;
      put(desired, node, "condition_sample", candidate(selectionLabel, parameters.node, "selected spreadsheet generation candidate", parameters.edge.fromPort));
    }
    if (selected.ready) {
      put(desired, node, "cond_values", candidate(
        selected.condValues,
        parameters.node,
        `selected row ${selected.selectedSampleId}; Input columns in spreadsheet order`,
        parameters.edge.fromPort
      ));
    }
    return;
  }

  if (node.type === "optimize.design") {
    const link = linkedInputs(node, "candidates")[0];
    const path = outputArtifactPath(link?.node);
    if (CSV_EXT.test(path)) put(desired, node, "csv_path", candidate(path, link.node, "candidate table from graph", link.edge.fromPort));
    return;
  }

  if (node.type === "evaluate.predictions") {
    const prediction = linkedInputs(node, "prediction")[0];
    const truth = linkedInputs(node, "truth")[0];
    const predictionPath = outputArtifactPath(prediction?.node);
    const truthPath = datasetPath(truth?.node);
    if (predictionPath) put(desired, node, "prediction_path", candidate(predictionPath, prediction.node, "prediction artifact from graph", prediction.edge.fromPort));
    if (truthPath) put(desired, node, "truth_path", candidate(truthPath, truth.node, "ground-truth dataset from graph", truth.edge.fromPort));
    return;
  }

  if (node.type === "output.export") {
    const link = linkedInputs(node, "input")[0];
    const path = outputArtifactPath(link?.node);
    if (path) put(desired, node, "source_path", candidate(path, link.node, "export artifact from graph", link.edge.fromPort));
    return;
  }

  if (node.type === "deploy.api") {
    const model = linkedInputs(node, "model")[0];
    const data = linkedInputs(node, "data")[0];
    if (model) {
      const paths = checkpointPaths(model.node);
      const rawPath = paths.checkpoint_path || paths.lc_modelpath || paths.fm_modelpath || paths.vae_modelpath;
      const path = BLOCK_SPECS[model.node.type]?.isModel
        ? fromMethodPath(rawPath, paths.model_id)
        : rawPath;
      if (path) put(desired, node, "checkpoint_path", candidate(path, model.node, "deployment checkpoint from graph", model.edge.fromPort));
      if (paths.model_id) put(desired, node, "model_id", candidate(paths.model_id, model.node, "deployment model family from graph", model.edge.fromPort));
    }
    const dataPath = datasetPath(data?.node);
    if (dataPath) put(desired, node, "input_path", candidate(dataPath, data.node, "deployment sample data from graph", data.edge.fromPort));
  }
}

function desiredAutofill() {
  const desired = new Map();
  state.nodes.forEach(node => {
    if (BLOCK_SPECS[node.type]?.isModel) modelAutofill(desired, node);
    else if (node.type === "source.parameters") designParameterAutofill(desired, node);
    else genericAutofill(desired, node);
  });
  return desired;
}

function manualSet(node) {
  return new Set(Array.isArray(node.manualConfigKeys) ? node.manualConfigKeys : []);
}

export function markManualConfigValue(node, key, value) {
  if (!node || !key) return;
  const manual = manualSet(node);
  if (text(value)) manual.add(key);
  else manual.delete(key);
  node.manualConfigKeys = [...manual].sort();
  if (node.autoFill) delete node.autoFill[key];
}

export function resetManualConfigValues(node) {
  if (!node) return;
  node.manualConfigKeys = [];
  node.autoFill = {};
}

export function applyGraphAutofill() {
  let changed = 0;
  const maxPasses = Math.min(8, Math.max(2, state.nodes.length));
  for (let pass = 0; pass < maxPasses; pass += 1) {
    const desired = desiredAutofill();
    let passChanged = 0;
    state.nodes.forEach(node => {
      node.config ||= {};
      const previous = node.autoFill && typeof node.autoFill === "object" ? node.autoFill : {};
      const manual = manualSet(node);
      const next = {};
      const wanted = desired.get(node.id) || {};

      Object.entries(previous).forEach(([key, meta]) => {
        if (text(node.config[key]) !== text(meta?.value)) manual.add(key);
        if (!wanted[key] && !manual.has(key) && text(node.config[key]) === text(meta?.value)) {
          // Put back whatever this autofill displaced. Deleting outright lost
          // template- and user-authored values for good the moment a link or a
          // mode changed.
          if (text(meta?.displaced)) node.config[key] = meta.displaced;
          else delete node.config[key];
          passChanged += 1;
        }
      });

      Object.entries(wanted).forEach(([key, meta]) => {
        if (manual.has(key)) return;
        // Remember the value being replaced once, so a later withdrawal can
        // restore it; carry it forward while the same autofill stays in place.
        const displaced = Object.hasOwn(previous, key)
          ? previous[key]?.displaced
          : (text(node.config[key]) && text(node.config[key]) !== meta.value ? node.config[key] : "");
        if (text(node.config[key]) !== meta.value) {
          node.config[key] = meta.value;
          passChanged += 1;
        }
        next[key] = { ...meta, displaced: displaced || "" };
      });

      node.manualConfigKeys = [...manual].sort();
      node.autoFill = next;
    });
    changed += passChanged;
    if (!passChanged) break;
  }
  return changed;
}

export function autoFillMeta(node, key) {
  const meta = node?.autoFill?.[key];
  return meta && text(node?.config?.[key]) === text(meta.value) ? meta : null;
}

export function autoFillCount(node) {
  return Object.keys(node?.autoFill || {}).filter(key => autoFillMeta(node, key)).length;
}
