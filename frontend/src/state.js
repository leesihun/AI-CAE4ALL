import { BLOCK_SPECS } from "./constants.js";
import { NODE_HEADER_HEIGHT, NODE_PREVIEW_HEIGHT, PORT_GAP, NODE_FOOTER_HEIGHT } from "./constants.js";

export function nodePortRows(node) {
  const spec = BLOCK_SPECS[node.type];
  return Math.max(1, spec.inputs.length, spec.outputs.length);
}

export function nodeHeight(node) {
  return NODE_HEADER_HEIGHT + NODE_PREVIEW_HEIGHT + nodePortRows(node) * PORT_GAP + NODE_FOOTER_HEIGHT;
}

export const state = {
  nodes: [],
  edges: [],
  selectedNode: null,
  selectedEdge: null,
  pendingPort: null,
  history: [],
  view: { x: 22, y: 34, scale: .9 },
  drag: null,
  pan: null,
  running: false,
  runTimer: null,
  nodeCounter: 1,
  configNode: null,
  configSection: "Required",
  configSearch: "",
  configMessages: [],
  configRejectedNode: null,
  configRejectedField: null,
  studioSection: "models",
  studioNode: null,
  artifactNode: null,
  artifactSample: null,
  viewerMode: "field",
  viewerPlaying: false,
  viewerPlayTimer: null,
  viewerCamera: { yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0 },
  viewerDraw: null,
  viewerPointer: null,
  viewerDatasetChoices: [],
  realArtifact: null,
  pendingEvaluationPrediction: "",
  pendingComparisonPaths: [],
  // path -> what /api/checkpoint said about that .pth. This is what lets an
  // Inference block run against a saved model whose trainer is not on the
  // canvas: the checkpoint records the architecture its weights were fit under,
  // which is exactly what the launcher's inference mode requires.
  checkpointMeta: new Map(),
  api: {
    connected: false,
    health: null,
    models: [],
    // activeJob is only the job the runtime drawer is currently showing.
    // trackedJobs holds every non-terminal job being polled, so several
    // pipelines can run concurrently (the backend already runs each in its
    // own thread; the single-job limit was purely client-side).
    activeJob: null,
    trackedJobs: new Map(),
    pollTimer: null,
    lastPreflight: null
  }
};

let mutationHook = null;

export function registerMutationHook(hook) {
  mutationHook = typeof hook === "function" ? hook : null;
}

function pipelineNameInput() {
  return typeof document === "undefined" ? null : document.getElementById("pipelineName");
}

/**
 * Restore one entry created by snapshot(). Older entries contained only nodes
 * and edges, so every newer field is optional for backwards compatibility.
 */
export function restoreSnapshot(serialized) {
  const previous = typeof serialized === "string" ? JSON.parse(serialized) : serialized;
  if (!previous || !Array.isArray(previous.nodes) || !Array.isArray(previous.edges)) {
    throw new Error("Pipeline history entry is invalid.");
  }
  state.nodes = previous.nodes;
  state.edges = previous.edges;
  if (previous.view && typeof previous.view === "object") {
    state.view = { ...previous.view };
  }
  if (Number.isFinite(Number(previous.nodeCounter))) {
    state.nodeCounter = Number(previous.nodeCounter);
  }
  const name = Object.hasOwn(previous, "pipelineName") ? previous.pipelineName : previous.name;
  const nameInput = pipelineNameInput();
  if (nameInput && typeof name === "string") nameInput.value = name;
  return previous;
}

export function snapshot() {
  state.history.push(JSON.stringify({
    nodes: state.nodes,
    edges: state.edges,
    pipelineName: pipelineNameInput()?.value ?? "Untitled pipeline",
    view: state.view,
    nodeCounter: state.nodeCounter
  }));
  if (state.history.length > 25) state.history.shift();
  const savedState = typeof document === "undefined" ? null : document.getElementById("savedState");
  if (savedState) savedState.textContent = "Unsaved changes";
  queueMicrotask(() => mutationHook?.());
}
