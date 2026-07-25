import { $ } from "./dom.js";
import { state } from "./state.js";
import { BLOCK_SPECS, MODEL_CATALOG, TEMPLATES } from "./constants.js";
import { connectRuntime } from "./api.js";
import { paletteRender, loadTemplate, selectNode } from "./graph.js";
import { validateGraph } from "./validate.js";
import { openConfig } from "./config.js";
import { openStudio } from "./studio.js";
import { openArtifact } from "./viewer.js";
import { bindEvents } from "./events.js";

function initialize() {
  paletteRender();
  bindEvents();
  const params = new URLSearchParams(location.search);
  const review = params.get("review");
  const template = review === "optimization" ? "generative" : "simulgen";
  $("#templateSelect").value = template;
  loadTemplate(template, false);
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
  openArtifact
};

initialize();
