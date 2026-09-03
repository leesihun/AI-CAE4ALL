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

/**
 * Group a template by what its blocks are, not by a hand-kept tag. Fifteen
 * templates in one flat list made the picker a scan of near-identical
 * "(ex9 plasticity)" names; grouping by the kind of block each one trains is
 * what a user actually chooses between. Deriving the group from the node types
 * means a template added to TEMPLATES lands in the right group with no extra
 * metadata to forget.
 */
const TEMPLATE_GROUP_ORDER = [
  "Mesh field surrogates",
  "Fixed-geometry and tabular surrogates",
  "Generative geometry (SDFFlow)",
  "Data preparation",
  "Start from scratch",
  "Other"
];

function templateGroup(template) {
  if (!template.nodes?.length) return "Start from scratch";
  const types = new Set(template.nodes.map(node => node[1]));
  if (types.has("model.sdfflow")) return "Generative geometry (SDFFlow)";
  if (types.has("model.simulgenvae") || types.has("model.mlp")) return "Fixed-geometry and tabular surrogates";
  if ([...types].some(type => type.startsWith("model."))) return "Mesh field surrogates";
  if (types.has("prep.geometry")) return "Data preparation";
  return "Other";
}

function templateOptionsRender() {
  const select = $("#templateSelect");
  const saved = select.querySelector('option[value="saved"]');
  select.innerHTML = "";
  if (saved) select.append(saved);
  const groups = new Map();
  Object.entries(TEMPLATES).forEach(([key, template]) => {
    const group = templateGroup(template);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push([key, template]);
  });
  TEMPLATE_GROUP_ORDER.filter(group => groups.has(group)).forEach(group => {
    const optgroup = document.createElement("optgroup");
    optgroup.label = group;
    groups.get(group).forEach(([key, template]) => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = template.name;
      optgroup.append(option);
    });
    select.append(optgroup);
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
  const template = review === "optimization"
    ? "generative"
    : review === "config"
      ? "simulgen"
      : "himgn";
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
  connectRuntime(() => {
    paletteRender();
    render();
  });
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
