import { toast } from "./dom.js";
import { BLOCK_SPECS, FIT_MIN_ZOOM, MAX_ZOOM } from "./constants.js";
import { state, registerMutationHook, snapshot } from "./state.js";
import { applyGraphAutofill } from "./autofill.js";

export const PIPELINE_STORAGE_KEY = "ai-cae4all.studio.pipeline.v1";
const PIPELINE_FORMAT = "ai-cae4all-pipeline";
const PIPELINE_VERSION = 1;
let saveTimer = null;
let lastFingerprint = "";
// The label for the copy actually in localStorage, reused when a save is a no-op.
let lastSavedLabel = "";

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

export class PipelineLoadCancelledError extends Error {
  constructor() {
    super("Pipeline replacement cancelled.");
    this.name = "PipelineLoadCancelledError";
  }
}

function confirmPipelineReplacement(payload) {
  if (typeof window === "undefined" || typeof window.confirm !== "function") return true;
  const name = String(payload.name || "Untitled pipeline");
  return window.confirm(`Replace the current pipeline with "${name}"?\n\nYour current pipeline will be kept as one Undo step.`);
}

export function applyPipelineDocument(payload, {
  confirmReplacement = true,
  recordHistory = true,
  resetHistory = false
} = {}) {
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
      // NOT `{...spec.defaults, ...config}`. pipelineDocument() runs
      // applyGraphAutofill() and writes the full effective config, so the file
      // is already complete; merging defaults on top injected whatever the
      // catalog had gained since the export (parallel_mode ddp,
      // use_checkpointing False) as if the user had typed them, which made
      // Export -> Import not a round trip. Missing keys are the launcher's
      // business: preflight reports them with their own diagnostic codes.
      config: { ...config },
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
  // A port that a block SPEC no longer declares is version drift, not a
  // malformed file: removing an inert port (optimize.design lost two) would
  // otherwise make every stored workspace that touched it unloadable, and
  // restorePipelineState would throw away the user's whole canvas. Those edges
  // are dropped and named; anything else is still a hard error.
  const droppedEdges = [];
  const edges = payload.edges.map((raw, index) => {
    const source = nodeById.get(String(raw?.fromNode || ""));
    const target = nodeById.get(String(raw?.toNode || ""));
    if (!source || !target) {
      throw new Error(`Connection ${index + 1} references a missing node.`);
    }
    const sourcePort = BLOCK_SPECS[source.type].outputs.find(port => port.id === raw.fromPort);
    const targetPort = BLOCK_SPECS[target.type].inputs.find(port => port.id === raw.toPort);
    if (!sourcePort || !targetPort) {
      droppedEdges.push(`${source.type}.${raw.fromPort} → ${target.type}.${raw.toPort}`);
      return null;
    }
    if (source.id === target.id) throw new Error(`Connection ${index + 1} cannot connect a block to itself.`);
    // Same rule as validate.js::compatible -- the artifact wildcard is
    // RECEIVE-side only. This copy still had the send-side half, so a document
    // could reintroduce exactly the Export-into-training-data edge the canvas
    // now refuses.
    const compatible = sourcePort.type === targetPort.type || targetPort.type === "artifact";
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
  }).filter(Boolean);
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
  // Build and validate the entire incoming graph before asking for confirmation
  // or touching history. A malformed/cancelled load must leave both untouched.
  if (confirmReplacement && !confirmPipelineReplacement(payload)) {
    throw new PipelineLoadCancelledError();
  }
  if (recordHistory) snapshot();
  state.nodes = nodes;
  state.edges = edges;
  state.selectedNode = null;
  state.selectedEdge = null;
  state.pendingPort = null;
  if (resetHistory) state.history = [];
  state.nodeCounter = Math.min(1_000_000, Math.max(1, Math.floor(finite(payload.node_counter, nodes.length + 1))));
  state.view = {
    x: finite(payload.view?.x, 22),
    y: finite(payload.view?.y, 34),
    scale: Math.min(MAX_ZOOM, Math.max(FIT_MIN_ZOOM, finite(payload.view?.scale, .9)))
  };
  applyGraphAutofill();
  const nameInput = document.getElementById("pipelineName");
  if (nameInput) nameInput.value = String(payload.name || "Untitled pipeline");
  if (droppedEdges.length) {
    toast(`${droppedEdges.length} connection${droppedEdges.length === 1 ? "" : "s"} to a port that no longer exists ${droppedEdges.length === 1 ? "was" : "were"} dropped: ${droppedEdges.join(", ")}.`, "warn");
  }
  return payload;
}

export function savePipelineState({ announce = false } = {}) {
  window.clearTimeout(saveTimer);
  saveTimer = null;
  try {
    const documentValue = pipelineDocument();
    const serialized = JSON.stringify(documentValue);
    const fingerprint = JSON.stringify({ ...documentValue, saved_at: "" });
    // The timestamp must describe the stored copy, not this call. The
    // fingerprint guard means an unchanged pipeline writes nothing, and
    // refreshing the label anyway claimed a save that never happened -- so the
    // stored saved_at could be minutes older than the line above it.
    const wrote = fingerprint !== lastFingerprint;
    if (wrote) localStorage.setItem(PIPELINE_STORAGE_KEY, serialized);
    lastFingerprint = fingerprint;
    const status = document.getElementById("savedState");
    if (status && wrote) {
      lastSavedLabel = `Saved locally · ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
      status.textContent = lastSavedLabel;
    } else if (status && lastSavedLabel) {
      status.textContent = lastSavedLabel;
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
    // Startup restore is not a user replacement and should start a fresh undo
    // session. Every interactive caller keeps the default protected behavior.
    applyPipelineDocument(payload, {
      confirmReplacement: false,
      recordHistory: false,
      resetHistory: true
    });
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
