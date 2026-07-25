import { toast } from "./dom.js";
import { state } from "./state.js";
import { BLOCK_SPECS, MODEL_CATALOG, TYPE_META } from "./constants.js";
import { apiRequest, requireRuntime } from "./api.js";
import { rawConfig } from "./config.js";

export function typeColor(type) {
  return (TYPE_META[type] || TYPE_META.artifact).color;
}

export function compatible(fromType, toType) {
  return fromType === toType || fromType === "artifact" || toType === "artifact";
}

export function configTextForNode(node, overrides = {}) {
  const spec = BLOCK_SPECS[node.type];
  if (!spec?.isModel) return "";
  const model = MODEL_CATALOG[spec.modelId];
  const values = { ...node.config, ...overrides, model: spec.modelId };
  const linkedSources = state.edges
    .filter(edge => edge.toNode === node.id)
    .map(edge => state.nodes.find(candidate => candidate.id === edge.fromNode))
    .filter(Boolean);
  const datasetSource = linkedSources.find(source => source.type === "source.hdf5");
  const parameterSource = linkedSources.find(source => source.type === "source.parameters");
  const checkpointSource = linkedSources.find(source => source.type === "source.checkpoint");
  const toMethodPath = value => {
    const text = String(value || "").replaceAll("\\", "/");
    if (!text || text.startsWith("../") || /^[A-Za-z]:\//.test(text)) return text;
    return `../${text.replace(/^\.\//, "")}`;
  };
  if (datasetSource?.config.path) {
    if (values.mode === "inference" && model.keys.includes("infer_dataset")) values.infer_dataset = toMethodPath(datasetSource.config.path);
    else if (model.keys.includes("dataset_dir")) values.dataset_dir = toMethodPath(datasetSource.config.path);
  }
  if (parameterSource?.config.binding && model.keys.includes("param_dir")) {
    const binding = String(parameterSource.config.binding).trim();
    if (/\.(csv|png|jpg|jpeg)$/i.test(binding) || /^[A-Za-z]:[\\/]/.test(binding)) values.param_dir = toMethodPath(binding);
  }
  if (checkpointSource?.config.path && model.keys.includes("modelpath")) {
    values.modelpath = toMethodPath(checkpointSource.config.path);
  }
  return `${rawConfig(values, model.keys)}\n`;
}

export function geometryConfigText(node) {
  const upstream = state.edges
    .filter(edge => edge.toNode === node.id)
    .map(edge => state.nodes.find(candidate => candidate.id === edge.fromNode))
    .find(candidate => candidate?.type === "source.cad");
  const rawInput = String(upstream?.config.path || node.config.input_geometry || "").replaceAll("\\", "/");
  const inputGeometry = /^[A-Za-z]:\//.test(rawInput)
    ? rawInput
    : rawInput.startsWith("dataset/")
      ? `../${rawInput.slice("dataset/".length)}`
      : `../../${rawInput.replace(/^\.\//, "")}`;
  const values = {
    model: "geometry_ingest",
    mode: node.config.mode || "inspect",
    input_geometry: inputGeometry,
    reader: node.config.reader || "auto",
    mesh_type: node.config.mesh_type || "surface",
    emit: node.config.emit || "graph",
    num_fields: node.config.num_fields || "3",
    num_points: node.config.num_points || "4096",
    resample_method: node.config.resample_method || "fps",
    seed: node.config.seed || "42"
  };
  if (values.mode === "ingest") values.output_dataset = node.config.output_dataset;
  if (String(node.config.limit || "0") !== "0") values.limit = node.config.limit;
  return `${Object.entries(values).map(([key, value]) => `${key.padEnd(29, " ")}${value}`).join("\n")}\n`;
}

export function topologicalNodes(targetId = null) {
  const ordered = [];
  const permanent = new Set();
  const temporary = new Set();
  const visit = node => {
    if (!node || permanent.has(node.id)) return;
    if (temporary.has(node.id)) throw new Error(`Pipeline cycle detected at ${BLOCK_SPECS[node.type]?.label || node.id}.`);
    temporary.add(node.id);
    state.edges
      .filter(edge => edge.toNode === node.id)
      .map(edge => state.nodes.find(candidate => candidate.id === edge.fromNode))
      .filter(Boolean)
      .forEach(visit);
    temporary.delete(node.id);
    permanent.add(node.id);
    ordered.push(node);
  };
  if (targetId) visit(state.nodes.find(node => node.id === targetId));
  else state.nodes.forEach(visit);
  return ordered;
}

export function executableSteps(targetId = null) {
  const target = targetId ? state.nodes.find(node => node.id === targetId) : null;
  const available = topologicalNodes(targetId);
  const modelNodes = available.filter(node => BLOCK_SPECS[node.type]?.isModel);
  if (target && BLOCK_SPECS[target.type]?.isModel) {
    return [{
      label: `${BLOCK_SPECS[target.type].label} · ${target.config.mode || "train"}`,
      nodeId: target.id,
      config: configTextForNode(target)
    }];
  }
  const steps = [];
  available.forEach(node => {
    const spec = BLOCK_SPECS[node.type];
    if (node.type === "prep.geometry") {
      steps.push({
        label: `${spec.label} · ${node.config.mode || "inspect"}`,
        nodeId: node.id,
        config: geometryConfigText(node)
      });
      return;
    }
    if (spec?.isModel) {
      steps.push({
        label: `${spec.label} · ${node.config.mode || "train"}`,
        nodeId: node.id,
        config: configTextForNode(node)
      });
      return;
    }
    if (!["run.inference", "run.cad_generator"].includes(node.type)) return;
    const upstream = [...modelNodes].reverse().find(modelNode =>
      state.edges.some(edge => edge.fromNode === modelNode.id && edge.toNode === node.id)
    ) || modelNodes.at(-1);
    if (!upstream) return;
    const modelId = BLOCK_SPECS[upstream.type].modelId;
    const mode = node.type === "run.cad_generator"
      ? "sample"
      : modelId === "simulgenvae"
        ? "reconstruct"
        : modelId === "sdfflow"
          ? "reconstruct"
          : "inference";
    steps.push({
      label: `${BLOCK_SPECS[upstream.type].label} · ${mode}`,
      nodeId: node.id,
      config: configTextForNode(upstream, { mode })
    });
  });
  return steps;
}

export function validateGraph(showToast = true) {
  const errors = [];
  state.nodes.forEach(node => {
    const spec = BLOCK_SPECS[node.type];
    spec.inputs.filter(port => port.required).forEach(port => {
      const linked = state.edges.some(edge => edge.toNode === node.id && edge.toPort === port.id);
      if (!linked) errors.push(`${spec.label}: missing ${port.label}`);
    });
  });
  state.edges.forEach(edge => {
    const source = state.nodes.find(node => node.id === edge.fromNode);
    const target = state.nodes.find(node => node.id === edge.toNode);
    if (!source || !target) return;
    const out = BLOCK_SPECS[source.type].outputs.find(port => port.id === edge.fromPort);
    const input = BLOCK_SPECS[target.type].inputs.find(port => port.id === edge.toPort);
    if (out && input && !compatible(out.type, input.type)) errors.push(`Type mismatch: ${out.type} → ${input.type}`);
  });
  try {
    topologicalNodes();
  } catch (error) {
    errors.push(error.message);
  }
  if (showToast) toast(errors.length ? `${errors.length} graph issue${errors.length === 1 ? "" : "s"}: ${errors[0]}` : "Validation passed: typed graph and required links are complete.", errors.length ? "error" : "");
  return errors;
}

export function preflightMessages(payload) {
  const report = payload?.report || payload?.failures?.[0]?.preflight?.report;
  if (!report) return [{ type: "error", text: payload?.error || "Preflight did not return a diagnostic report." }];
  return [
    {
      type: report.summary.errors ? "error" : report.summary.warnings ? "warn" : "",
      text: `Authoritative preflight: ${report.summary.errors} errors, ${report.summary.warnings} warnings, ${report.summary.notices} notices.`
    },
    ...report.diagnostics.map(item => ({
      type: item.severity === "error" ? "error" : item.severity === "warning" ? "warn" : "",
      text: `[${item.code}]${item.field ? ` ${item.field}:` : ""} ${item.message}${item.hint ? ` Hint: ${item.hint}` : ""}`
    }))
  ];
}

export async function preflightConfigText(text, label, options = {}) {
  if (!requireRuntime()) return null;
  return apiRequest("/api/preflight", {
    method: "POST",
    allowError: true,
    body: {
      config: text,
      label,
      strict: Boolean(options.strict),
      skip_filesystem: Boolean(options.skipFilesystem),
      skip_native: Boolean(options.skipNative),
      skip_environment: Boolean(options.skipEnvironment)
    }
  });
}
