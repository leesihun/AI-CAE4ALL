import { BLOCK_SPECS, FIT_MIN_ZOOM, MAX_ZOOM } from "./constants.js";
import { state, registerMutationHook } from "./state.js";
import { applyGraphAutofill } from "./autofill.js";

export const PIPELINE_STORAGE_KEY = "ai-cae4all.studio.pipeline.v1";
const PIPELINE_FORMAT = "ai-cae4all-pipeline";
const PIPELINE_VERSION = 1;
let saveTimer = null;
let lastFingerprint = "";

function pipelineName() {
  return document.getElementById("pipelineName")?.value?.trim() || "Untitled pipeline";
}

function finite(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function pipelineDocument() {
  applyGraphAutofill();
  return {
    format: PIPELINE_FORMAT,
    version: PIPELINE_VERSION,
    saved_at: new Date().toISOString(),
    name: pipelineName(),
    node_counter: state.nodeCounter,
    view: {
      x: finite(state.view.x, 22),
      y: finite(state.view.y, 34),
      scale: Math.min(MAX_ZOOM, Math.max(FIT_MIN_ZOOM, finite(state.view.scale, .9)))
    },
    nodes: state.nodes.map(node => ({
      id: node.id,
      type: node.type,
      x: finite(node.x, 24),
      y: finite(node.y, 54),
      config: node.config && typeof node.config === "object" ? node.config : {},
      ...(node.autoFill && Object.keys(node.autoFill).length ? { auto_fill: node.autoFill } : {}),
      ...(Array.isArray(node.manualConfigKeys) && node.manualConfigKeys.length ? { manual_config_keys: node.manualConfigKeys } : {}),
      ...(node.loadedConfigPath ? { loaded_config_path: node.loadedConfigPath } : {}),
      ...(node.savedConfigPath ? { saved_config_path: node.savedConfigPath } : {}),
      ...(node.optimizationReport ? { optimization_report: node.optimizationReport } : {})
    })),
    edges: state.edges.map(edge => ({
      id: edge.id,
      fromNode: edge.fromNode,
      fromPort: edge.fromPort,
      toNode: edge.toNode,
      toPort: edge.toPort
    }))
  };
}

function validateDocument(payload) {
  if (!payload || payload.format !== PIPELINE_FORMAT || payload.version !== PIPELINE_VERSION) {
    throw new Error("This is not a supported AI-CAE4ALL pipeline JSON file.");
  }
  if (!Array.isArray(payload.nodes) || !Array.isArray(payload.edges)) {
    throw new Error("Pipeline JSON must contain node and edge arrays.");
  }
  if (payload.nodes.length > 500 || payload.edges.length > 2000) {
    throw new Error("Pipeline JSON exceeds the local editor safety limit.");
  }
}

export function applyPipelineDocument(payload) {
  validateDocument(payload);
  const ids = new Set();
  const nodes = payload.nodes.map((raw, index) => {
    const type = String(raw?.type || "");
    const spec = BLOCK_SPECS[type];
    if (!spec) throw new Error(`Node ${index + 1} uses unknown block type ${type || "<empty>"}.`);
    const id = String(raw?.id || "").trim();
    if (!id || ids.has(id)) throw new Error(`Node ${index + 1} has a missing or duplicate ID.`);
    ids.add(id);
    const config = raw.config && typeof raw.config === "object" && !Array.isArray(raw.config) ? raw.config : {};
    return {
      id,
      type,
      x: Math.min(100000, Math.max(10, finite(raw.x, 24))),
      y: Math.min(100000, Math.max(10, finite(raw.y, 54))),
      config: { ...spec.defaults, ...config },
      autoFill: raw.auto_fill && typeof raw.auto_fill === "object" && !Array.isArray(raw.auto_fill) ? { ...raw.auto_fill } : {},
      manualConfigKeys: Array.isArray(raw.manual_config_keys) ? raw.manual_config_keys.map(String) : [],
      status: "idle",
      progress: 0,
      ...(raw.loaded_config_path ? { loadedConfigPath: String(raw.loaded_config_path) } : {}),
      ...(raw.saved_config_path ? { savedConfigPath: String(raw.saved_config_path) } : {}),
      ...(raw.optimization_report ? { optimizationReport: String(raw.optimization_report) } : {})
    };
  });
  const nodeById = new Map(nodes.map(node => [node.id, node]));
  const edgeIds = new Set();
  const occupiedInputs = new Set();
  const edges = payload.edges.map((raw, index) => {
    const source = nodeById.get(String(raw?.fromNode || ""));
    const target = nodeById.get(String(raw?.toNode || ""));
    const sourcePort = source && BLOCK_SPECS[source.type].outputs.find(port => port.id === raw.fromPort);
    const targetPort = target && BLOCK_SPECS[target.type].inputs.find(port => port.id === raw.toPort);
    if (!source || !target || !sourcePort || !targetPort) {
      throw new Error(`Connection ${index + 1} references a missing node or port.`);
    }
    if (source.id === target.id) throw new Error(`Connection ${index + 1} cannot connect a block to itself.`);
    const compatible = sourcePort.type === targetPort.type || sourcePort.type === "artifact" || targetPort.type === "artifact";
    if (!compatible) throw new Error(`Connection ${index + 1} links incompatible port types.`);
    const id = String(raw.id || `edge_imported_${index + 1}`);
    if (edgeIds.has(id)) throw new Error(`Connection ${index + 1} has a duplicate ID.`);
    edgeIds.add(id);
    const inputKey = `${target.id}:${targetPort.id}`;
    if (!targetPort.multiple && occupiedInputs.has(inputKey)) {
      throw new Error(`Connection ${index + 1} duplicates a single-input port.`);
    }
    occupiedInputs.add(inputKey);
    return {
      id,
      fromNode: source.id,
      fromPort: sourcePort.id,
      toNode: target.id,
      toPort: targetPort.id
    };
  });
  const indegree = new Map(nodes.map(node => [node.id, 0]));
  const outgoing = new Map(nodes.map(node => [node.id, []]));
  edges.forEach(edge => {
    indegree.set(edge.toNode, indegree.get(edge.toNode) + 1);
    outgoing.get(edge.fromNode).push(edge.toNode);
  });
  const queue = nodes.filter(node => indegree.get(node.id) === 0).map(node => node.id);
  let visited = 0;
  while (queue.length) {
    const current = queue.shift();
    visited += 1;
    outgoing.get(current).forEach(next => {
      indegree.set(next, indegree.get(next) - 1);
      if (indegree.get(next) === 0) queue.push(next);
    });
  }
  if (visited !== nodes.length) throw new Error("Pipeline JSON contains a dependency cycle.");
  state.nodes = nodes;
  state.edges = edges;
  state.selectedNode = null;
  state.selectedEdge = null;
  state.pendingPort = null;
  state.history = [];
  state.nodeCounter = Math.min(1_000_000, Math.max(1, Math.floor(finite(payload.node_counter, nodes.length + 1))));
  state.view = {
    x: finite(payload.view?.x, 22),
    y: finite(payload.view?.y, 34),
    scale: Math.min(MAX_ZOOM, Math.max(FIT_MIN_ZOOM, finite(payload.view?.scale, .9)))
  };
  applyGraphAutofill();
  const nameInput = document.getElementById("pipelineName");
  if (nameInput) nameInput.value = String(payload.name || "Untitled pipeline");
  return payload;
}

export function savePipelineState({ announce = false } = {}) {
  window.clearTimeout(saveTimer);
  saveTimer = null;
  try {
    const documentValue = pipelineDocument();
    const serialized = JSON.stringify(documentValue);
    const fingerprint = JSON.stringify({ ...documentValue, saved_at: "" });
    if (fingerprint !== lastFingerprint) localStorage.setItem(PIPELINE_STORAGE_KEY, serialized);
    lastFingerprint = fingerprint;
    const status = document.getElementById("savedState");
    if (status) {
      const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      status.textContent = `Saved locally · ${time}`;
    }
    if (announce) document.dispatchEvent(new CustomEvent("pipeline-saved"));
    return documentValue;
  } catch (error) {
    const status = document.getElementById("savedState");
    if (status) status.textContent = "Local save failed";
    throw error;
  }
}

export function schedulePipelineSave() {
  const status = document.getElementById("savedState");
  if (status) status.textContent = "Unsaved changes";
  window.clearTimeout(saveTimer);
  saveTimer = window.setTimeout(() => {
    try { savePipelineState(); } catch { /* The persistent status already exposes the failure. */ }
  }, 350);
}

export function restorePipelineState() {
  try {
    const serialized = localStorage.getItem(PIPELINE_STORAGE_KEY);
    if (!serialized) return false;
    const payload = JSON.parse(serialized);
    applyPipelineDocument(payload);
    lastFingerprint = JSON.stringify({ ...payload, saved_at: "" });
    return true;
  } catch (error) {
    localStorage.removeItem(PIPELINE_STORAGE_KEY);
    const status = document.getElementById("savedState");
    if (status) status.textContent = `Saved workspace ignored · ${error.message}`;
    return false;
  }
}

export function downloadPipelineJson() {
  const payload = savePipelineState();
  const url = URL.createObjectURL(new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${payload.name.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "pipeline"}.ai-cae.json`;
  link.click();
  URL.revokeObjectURL(url);
}

export async function importPipelineJson(file) {
  if (!file) throw new Error("Choose a pipeline JSON file.");
  if (file.size > 5 * 1024 * 1024) throw new Error("Pipeline JSON must be 5 MiB or smaller.");
  const payload = JSON.parse(await file.text());
  applyPipelineDocument(payload);
  savePipelineState();
  return payload;
}

registerMutationHook(schedulePipelineSave);
