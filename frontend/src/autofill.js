import { BLOCK_SPECS, MODEL_CATALOG } from "./constants.js";
import { state } from "./state.js";

const CSV_EXT = /\.csv$/i;
const IMAGE_EXT = /\.(?:png|jpe?g|bmp|tiff?|webp)$/i;

function text(value) {
  return String(value ?? "").trim();
}

export function toMethodPath(value) {
  const normalized = text(value).replaceAll("\\", "/");
  if (!normalized || normalized.startsWith("../") || /^[A-Za-z]:\//.test(normalized)) return normalized;
  return `../${normalized.replace(/^\.\//, "")}`;
}

export function toGeometryPath(value) {
  const normalized = text(value).replaceAll("\\", "/");
  if (!normalized || normalized.startsWith("../") || /^[A-Za-z]:\//.test(normalized)) return normalized;
  if (normalized.startsWith("dataset/")) return `../${normalized.slice("dataset/".length)}`;
  return `../../${normalized.replace(/^\.\//, "")}`;
}

/** Inverse of toGeometryPath: geometry_ingest's own output_dataset default is
 * stored method-relative (relative to dataset/geometry_ingest/, two levels
 * below the suite root). Other blocks (Export, Evaluate, HDF5 viewers) all
 * resolve paths suite-root-relative, so propagating that raw value verbatim
 * escapes the allowed repository roots. */
export function fromGeometryPath(value) {
  const normalized = text(value).replaceAll("\\", "/");
  if (!normalized || /^[A-Za-z]:\//.test(normalized)) return normalized;
  if (normalized.startsWith("../../")) return normalized.slice("../../".length);
  if (normalized.startsWith("../")) return `dataset/${normalized.slice("../".length)}`;
  return normalized;
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

function commaNames(node, key) {
  return text(node?.config?.[key]).split(",").map(value => value.trim()).filter(Boolean);
}

function datasetPath(node) {
  if (!node) return "";
  if (node.type === "source.hdf5") return configPath(node, ["path"]);
  if (node.type === "prep.geometry") return fromGeometryPath(configPath(node, ["output_dataset"]));
  return configPath(node, ["dataset_path", "truth_path", "path"]);
}

function checkpointPaths(node) {
  if (!node) return {};
  if (node.type === "source.checkpoint") return { checkpoint_path: configPath(node, ["path"]) };
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
  return configPath(node, [
    "report_path", "metrics_csv", "prediction_path", "output_path", "output_csv",
    "candidate_csv", "csv_path", "source_path", "path"
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
    if (key && path) put(desired, node, key, candidate(toMethodPath(path), dataLink.node, `${mode} dataset from graph`, dataLink.edge.fromPort));

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
      put(desired, node, "param_dir", candidate(toMethodPath(binding), source, "condition data from graph", parameterLink.edge.fromPort));
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
      put(desired, node, "modelpath", candidate(toMethodPath(path), source, "resume checkpoint from graph", checkpointLink.edge.fromPort));
    } else if (path && spec.modelId === "simulgenvae") {
      const key = /(?:^|[_-])lc(?:[_-]|\.)|condition/.test(filename)
        ? "lc_modelpath"
        : /vae/.test(filename) || mode === "train_lc" ? "vae_modelpath" : "";
      if (key) put(desired, node, key, candidate(toMethodPath(path), source, `${key === "vae_modelpath" ? "VAE" : "conditioner"} checkpoint from graph`, checkpointLink.edge.fromPort));
    } else if (path && spec.modelId === "sdfflow") {
      const key = /(?:^|[_-])fm(?:[_-]|\.)|flow/.test(filename)
        ? "fm_modelpath"
        : /vae/.test(filename) || ["train_fm", "reconstruct"].includes(mode) ? "vae_modelpath" : "";
      if (key) put(desired, node, key, candidate(toMethodPath(path), source, `${key === "vae_modelpath" ? "VAE" : "flow"} checkpoint from graph`, checkpointLink.edge.fromPort));
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
    const names = commaNames(parameters?.node, "condition_names");
    if (names.length) put(desired, node, "condition_names", candidate(names.join(","), parameters.node, "condition columns from spreadsheet", parameters.edge.fromPort));
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
      const path = paths.checkpoint_path || paths.lc_modelpath || paths.fm_modelpath || paths.vae_modelpath;
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
          delete node.config[key];
          passChanged += 1;
        }
      });

      Object.entries(wanted).forEach(([key, meta]) => {
        if (manual.has(key)) return;
        if (text(node.config[key]) !== meta.value) {
          node.config[key] = meta.value;
          passChanged += 1;
        }
        next[key] = { ...meta };
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
