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
  studioSection: "models",
  artifactNode: null,
  artifactSample: null,
  viewerMode: "field",
  viewerPlaying: false,
  viewerPlayTimer: null,
  viewerCamera: { yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0 },
  viewerPointer: null,
  viewerDatasetChoices: [],
  realArtifact: null,
  api: {
    connected: false,
    health: null,
    models: [],
    activeJob: null,
    pollTimer: null,
    lastPreflight: null
  }
};

export function snapshot() {
  state.history.push(JSON.stringify({ nodes: state.nodes, edges: state.edges }));
  if (state.history.length > 25) state.history.shift();
  const savedState = document.getElementById("savedState");
  if (savedState) savedState.textContent = "Unsaved changes";
}
