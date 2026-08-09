import { toast } from "./dom.js";
import { state } from "./state.js";
import { BLOCK_SPECS, MODEL_CATALOG, TYPE_META } from "./constants.js";
import { apiRequest, requireRuntime } from "./api.js";
import { normalizeConfigValues, rawConfig } from "./config.js";
import { applyGraphAutofill, toGeometryPath, toMethodPath } from "./autofill.js";

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
  applyGraphAutofill();
  const values = normalizeConfigValues({ ...node.config, ...overrides, model: spec.modelId });
  return `${rawConfig(values, model.keys)}\n`;
}

export function geometryConfigText(node) {
  applyGraphAutofill();
  const upstream = state.edges
    .filter(edge => edge.toNode === node.id)
    .map(edge => state.nodes.find(candidate => candidate.id === edge.fromNode))
    .find(candidate => candidate?.type === "source.cad");
  const rawInput = String(node.config.input_geometry || upstream?.config.path || "").replaceAll("\\", "/");
  const inputGeometry = toGeometryPath(rawInput);
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
    // The run.inference/run.cad_generator node's own graph-linked fields
    // (dataset_path, checkpoint_path, ...) are what the Inspector shows and
    // what the user edits, but the upstream model node's OWN persisted
    // config (still whatever mode it was last trained/edited in) is what
    // actually gets serialized below. Without this merge, infer_dataset and
    // friends are silently absent unless the model node happens to already
    // be sitting in inference mode with those keys manually set.
    const overrides = { mode };
    const catalogKeys = MODEL_CATALOG[modelId].keys;
    if (catalogKeys.includes("infer_dataset") && node.config.dataset_path) {
      overrides.infer_dataset = toMethodPath(node.config.dataset_path);
    }
    if (catalogKeys.includes("modelpath") && node.config.checkpoint_path) {
      // toMethodPath is a no-op on paths already relative to the owning
      // method repo (e.g. a checkpoint fed back from the same model node
      // that just trained it) and only adds the "../" prefix when the value
      // came from a source.checkpoint block's suite-relative browse/upload
      // path, so it is safe to apply unconditionally here.
      overrides.modelpath = toMethodPath(node.config.checkpoint_path);
    }
    ["vae_modelpath", "lc_modelpath", "fm_modelpath"].forEach(key => {
      if (catalogKeys.includes(key) && node.config[key]) overrides[key] = toMethodPath(node.config[key]);
    });
    steps.push({
      label: `${BLOCK_SPECS[upstream.type].label} · ${mode}`,
      nodeId: node.id,
      config: configTextForNode(upstream, overrides)
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
