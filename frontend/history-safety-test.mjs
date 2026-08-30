import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath, pathToFileURL } from "node:url";

const elements = new Map([
  ["pipelineName", { value: "Untitled pipeline" }],
  ["savedState", { textContent: "" }]
]);
const storage = new Map();

globalThis.document = {
  getElementById: id => elements.get(id) || null,
  querySelector: selector => selector.startsWith("#") ? elements.get(selector.slice(1)) || null : null,
  querySelectorAll: () => [],
  dispatchEvent: () => true
};
globalThis.window = {
  clearTimeout: () => {},
  setTimeout: () => 1,
  confirm: () => true
};
globalThis.localStorage = {
  getItem: key => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, String(value)),
  removeItem: key => storage.delete(key)
};
globalThis.CustomEvent = class CustomEvent {
  constructor(type) { this.type = type; }
};

// The browser sources intentionally have no package.json. Load their ES module
// graph directly without changing how the CommonJS Playwright runners execute.
const context = vm.createContext({
  console,
  document: globalThis.document,
  window: globalThis.window,
  localStorage: globalThis.localStorage,
  CustomEvent: globalThis.CustomEvent,
  queueMicrotask,
  setTimeout,
  clearTimeout,
  fetch: globalThis.fetch,
  URL,
  URLSearchParams,
  Blob,
  FormData,
  TextEncoder,
  TextDecoder,
  AbortController
});
const modulePromises = new Map();
function browserModule(filename) {
  const resolved = path.resolve(filename);
  if (!modulePromises.has(resolved)) {
    modulePromises.set(resolved, readFile(resolved, "utf8").then(source => new vm.SourceTextModule(source, {
      context,
      identifier: pathToFileURL(resolved).href
    })));
  }
  return modulePromises.get(resolved);
}
const frontendDirectory = path.dirname(fileURLToPath(import.meta.url));
const sourcePath = filename => path.join(frontendDirectory, "src", filename);
const graphPath = sourcePath("graph.js");
const graphModule = await browserModule(graphPath);
await graphModule.link(async (specifier, referencingModule) => {
  if (!specifier.startsWith(".")) throw new Error(`Unexpected browser import: ${specifier}`);
  return browserModule(fileURLToPath(new URL(specifier, referencingModule.identifier)));
});
await graphModule.evaluate();

const stateModule = await browserModule(sourcePath("state.js"));
const persistenceModule = await browserModule(sourcePath("persistence.js"));
const { state, snapshot, restoreSnapshot } = stateModule.namespace;
const {
  PIPELINE_STORAGE_KEY,
  PipelineLoadCancelledError,
  applyPipelineDocument,
  restorePipelineState
} = persistenceModule.namespace;
const { duplicateNodeRecord, loadTemplate } = graphModule.namespace;
const plain = value => JSON.parse(JSON.stringify(value));

function pipeline(name, view, nodeId = "cad_1") {
  return {
    format: "ai-cae4all-pipeline",
    version: 1,
    saved_at: "2026-08-29T00:00:00.000Z",
    name,
    node_counter: 8,
    view,
    nodes: [{ id: nodeId, type: "source.cad", x: 24, y: 54, config: { path: "input/part.step" } }],
    edges: []
  };
}

// A history entry is a complete visible workspace checkpoint, not just graph data.
state.nodes = [{ id: "before", type: "source.hdf5", x: 10, y: 20, config: { path: "dataset/ex2.h5" } }];
state.edges = [];
state.view = { x: 143, y: -27, scale: 1.17 };
state.nodeCounter = 41;
state.history = ["existing-user-change"];
elements.get("pipelineName").value = "Before load";
snapshot();
state.nodes = [];
state.view = { x: 0, y: 0, scale: 0.5 };
state.nodeCounter = 1;
elements.get("pipelineName").value = "After load";
restoreSnapshot(state.history.pop());
assert.equal(state.nodes[0].id, "before");
assert.deepEqual(plain(state.view), { x: 143, y: -27, scale: 1.17 });
assert.equal(state.nodeCounter, 41);
assert.equal(elements.get("pipelineName").value, "Before load");

// Template selection asks before replacing anything and cancellation is a true
// no-op, including history.
state.history = ["existing-user-change"];
let templateConfirmation = "";
context.window.confirm = message => {
  templateConfirmation = message;
  return false;
};
assert.equal(loadTemplate("himgn"), false);
assert.match(templateConfirmation, /Undo step/);
assert.equal(state.nodes[0].id, "before");
assert.deepEqual(plain(state.history), ["existing-user-change"]);
assert.equal(elements.get("pipelineName").value, "Before load");

// Interactive document replacement validates first, confirms, and changes no
// state/history when the user declines.
state.history = [];
let confirmation = "";
context.window.confirm = message => {
  confirmation = message;
  return false;
};
assert.throws(
  () => applyPipelineDocument(pipeline("Saved run", { x: 5, y: 6, scale: 0.8 })),
  PipelineLoadCancelledError
);
assert.match(confirmation, /Saved run/);
assert.equal(state.nodes[0].id, "before");
assert.equal(state.history.length, 0);

// Accepting the same replacement captures exactly one undo entry and restores
// the incoming name/view; that entry can restore the previous visible state.
context.window.confirm = () => true;
applyPipelineDocument(pipeline("Saved run", { x: 5, y: 6, scale: 0.8 }));
assert.equal(state.history.length, 1);
assert.equal(state.nodes[0].id, "cad_1");
assert.deepEqual(plain(state.view), { x: 5, y: 6, scale: 0.8 });
assert.equal(elements.get("pipelineName").value, "Saved run");
restoreSnapshot(state.history.pop());
assert.equal(state.nodes[0].id, "before");
assert.deepEqual(plain(state.view), { x: 143, y: -27, scale: 1.17 });
assert.equal(elements.get("pipelineName").value, "Before load");

// Startup persistence restore is intentionally non-interactive and begins a
// fresh history session.
storage.set(PIPELINE_STORAGE_KEY, JSON.stringify(pipeline("Restored", { x: 8, y: 9, scale: 1.05 }, "restored_1")));
state.history = ["old-entry"];
context.window.confirm = () => { throw new Error("startup restore prompted"); };
assert.equal(restorePipelineState(), true);
assert.equal(state.history.length, 0);
assert.equal(state.nodes[0].id, "restored_1");
assert.equal(elements.get("pipelineName").value, "Restored");

// Duplicates keep explicit source inputs, shed graph-derived inputs and stale
// evidence, and receive a non-colliding output destination.
const source = {
  id: "run_inference_1",
  type: "run.inference",
  x: 100,
  y: 120,
  status: "complete",
  progress: 100,
  config: {
    dataset_path: "dataset/ex2.h5",
    checkpoint_path: "output/models/best.pth",
    parameter_path: "dataset/conditions.csv",
    inference_output_dir: "output/predictions",
    job_id: "job-123",
    results_path: "output/predictions/epoch_4",
    results_samples: "20",
    report_path: "output/report.json",
    metrics_csv: "output/metrics.csv"
  },
  autoFill: {
    parameter_path: { value: "dataset/conditions.csv", sourceNodeId: "parameters_1" }
  },
  manualConfigKeys: ["dataset_path", "checkpoint_path", "job_id", "results_path"],
  loadedConfigPath: "configs/infer.txt",
  savedConfigPath: "frontend/runtime/configs/saved.txt",
  optimizationReport: "frontend/runtime/optimization/report.json"
};
const occupied = {
  id: "occupied",
  type: "run.inference",
  config: { inference_output_dir: "output/predictions-copy-7" }
};
const copy = duplicateNodeRecord(source, "run_inference_1_copy_7", [source, occupied]);
assert.equal(copy.status, "idle");
assert.equal(copy.progress, 0);
assert.equal(copy.config.dataset_path, source.config.dataset_path);
assert.equal(copy.config.checkpoint_path, source.config.checkpoint_path);
assert.equal(copy.config.parameter_path, undefined);
assert.equal(copy.config.job_id, undefined);
assert.equal(copy.config.results_path, undefined);
assert.equal(copy.config.results_samples, undefined);
assert.equal(copy.config.report_path, undefined);
assert.equal(copy.config.metrics_csv, undefined);
assert.equal(copy.config.inference_output_dir, "output/predictions-copy-7-2");
assert.equal(copy.loadedConfigPath, source.loadedConfigPath);
assert.equal(copy.savedConfigPath, undefined);
assert.equal(copy.optimizationReport, undefined);
assert.deepEqual(plain(copy.autoFill), {});
assert.deepEqual(plain(copy.manualConfigKeys), ["dataset_path", "checkpoint_path"]);

console.log("history safety tests passed");
