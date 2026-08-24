import { $ } from "./dom.js";
import { state } from "./state.js";
import { BLOCK_SPECS, MODEL_CATALOG, TEMPLATES } from "./constants.js";
import { connectRuntime } from "./api.js";
import { paletteRender, loadTemplate, render, selectNode } from "./graph.js";
import { validateGraph } from "./validate.js";
import { openConfig } from "./config.js";
import { openStudio } from "./studio.js";
import { openArtifact } from "./viewer.js";
import { defaultCamera } from "./render3d.js";
import { renderRuntimeJob, dismissRuntimeJob } from "./run.js";
import { bindEvents } from "./events.js";
import { restorePipelineState, savePipelineState } from "./persistence.js";
import { applyGraphAutofill, autoFillMeta, markManualConfigValue, registerCheckpointRefresh } from "./autofill.js";

function templateOptionsRender() {
  const select = $("#templateSelect");
  const saved = select.querySelector('option[value="saved"]');
  select.innerHTML = "";
  if (saved) select.append(saved);
  Object.entries(TEMPLATES).forEach(([key, template]) => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = template.name;
    select.append(option);
  });
}

function initialize() {
  // Checkpoint metadata arrives after the first paint, so the canvas and the
  // inspector have to be told to pick it up; without this a saved model block
  // stays "auto-detect" until the next unrelated interaction.
  registerCheckpointRefresh(() => {
    applyGraphAutofill();
    render();
  });
  paletteRender();
  templateOptionsRender();
  bindEvents();
  const params = new URLSearchParams(location.search);
  const review = params.get("review");
  const template = review === "optimization" ? "generative" : "himgn";
  const restored = !review && restorePipelineState();
  if (restored) {
    $("#templateSelect").value = "saved";
    render();
    savePipelineState();
  } else {
    $("#templateSelect").value = template;
    loadTemplate(template, false);
    savePipelineState();
  }
  if (review === "optimization") {
    window.setTimeout(() => {
      const node = state.nodes.find(item => item.type === "optimize.design");
      if (node) selectNode(node.id);
    }, 80);
  }
  if (review === "config") {
    window.setTimeout(() => {
      const node = state.nodes.find(item => item.type === "model.simulgenvae");
      if (node) openConfig(node.id);
    }, 100);
  }
  connectRuntime();
}

window.__AI_CAE_FRONTEND__ = {
  state,
  BLOCK_SPECS,
  MODEL_CATALOG,
  TEMPLATES,
  loadTemplate,
  validateGraph,
  openConfig,
  openStudio,
  openArtifact,
  defaultCamera,
  renderRuntimeJob,
  dismissRuntimeJob,
  applyGraphAutofill,
  autoFillMeta,
  markManualConfigValue
};

initialize();
