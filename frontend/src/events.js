import { $, $$, toast, closeOverlay } from "./dom.js";
import { state, snapshot } from "./state.js";
import { BLOCK_SPECS, MODEL_CATALOG, NODE_WIDTH, NODE_HEADER_HEIGHT } from "./constants.js";
import { apiRequest, requireRuntime } from "./api.js";
import { rawConfig, parseConfig, applyPreset, renderConfig, explainConfig, explainMessages, configureViaLlm } from "./config.js";
import { preflightConfigText, preflightMessages } from "./validate.js";
import {
  paletteRender, loadTemplate, arrangeGraph, setZoom, fitGraphView,
  setPanelVisibility, dragNode, panCanvas, stopNodeDrag, stopCanvasPan,
  addBlock, deleteSelected, render, selectNode, startCanvasPan, renderEdges
} from "./graph.js";
import { validatePipeline, runGraph, stopRun, dismissRuntimeJob } from "./run.js";
import { openStudio } from "./studio.js";
import { downloadPipelineJson, importPipelineJson, savePipelineState, schedulePipelineSave } from "./persistence.js";
import { applyGraphAutofill, markManualConfigValue, resetManualConfigValues } from "./autofill.js";

function retainExplicitConfig(node) {
  resetManualConfigValues(node);
  Object.entries(node.config).forEach(([key, value]) => markManualConfigValue(node, key, value));
  applyGraphAutofill();
}
import {
  compareCurrentSample, downloadCurrentSample, copyArtifactId,
  useArtifactInPipeline, toggleViewerPlayback, renderViewerMode,
  openArtifactDatasetPicker, closeArtifactDatasetPicker, uploadArtifactDataset,
  renderArtifactDatasetChoices, renderArtifactCatalog,
  bindViewerInteractions, resetViewerCamera
} from "./viewer.js";

export function bindEvents() {
  $("#blockSearch").addEventListener("input", event => paletteRender(event.target.value));
  $("#templateSelect").addEventListener("change", event => loadTemplate(event.target.value));
  $("#savePipeline").addEventListener("click", () => {
    try {
      savePipelineState({ announce: true });
      toast("Pipeline saved in this browser.");
    } catch (error) {
      toast(`Pipeline save failed: ${error.message}`, "error");
    }
  });
  $("#exportPipeline").addEventListener("click", () => {
    try {
      downloadPipelineJson();
      toast("Pipeline JSON exported.");
    } catch (error) {
      toast(`Pipeline export failed: ${error.message}`, "error");
    }
  });
  $("#importPipeline").addEventListener("click", () => $("#pipelineFile").click());
  $("#pipelineFile").addEventListener("change", async event => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      snapshot();
      await importPipelineJson(file);
      $("#templateSelect").value = "saved";
      render();
      toast(`Imported ${file.name}.`);
    } catch (error) {
      toast(`Pipeline import failed: ${error.message}`, "error");
    }
  });
  $("#arrangeGraph").addEventListener("click", arrangeGraph);
  $("#undoGraph").addEventListener("click", () => {
    const previous = state.history.pop();
    if (!previous) {
      toast("Nothing to undo.", "warn");
      return;
    }
    const parsed = JSON.parse(previous);
    state.nodes = parsed.nodes;
    state.edges = parsed.edges;
    applyGraphAutofill();
    state.selectedNode = null;
    state.selectedEdge = null;
    render();
    schedulePipelineSave();
    toast("Undid the last graph change.");
  });
  $("#pipelineName").addEventListener("input", schedulePipelineSave);
  $("#validateTop").addEventListener("click", () => validatePipeline());
  $("#runTop").addEventListener("click", () => runGraph());
  $("#stopRun").addEventListener("click", stopRun);
  $("#runtimeCancel").addEventListener("click", stopRun);
  $("#runtimeMinimize").addEventListener("click", () => {
    $("#runtimeDrawer").classList.toggle("minimized");
    const minimized = $("#runtimeDrawer").classList.contains("minimized");
    $("#runtimeMinimize").textContent = minimized ? "+" : "−";
    $("#runtimeMinimize").setAttribute("aria-expanded", String(!minimized));
    $("#runtimeMinimize").setAttribute("aria-label", minimized ? "Expand runtime log" : "Minimize runtime log");
  });
  $("#runtimeExperiments").addEventListener("click", () => openStudio("experiments"));
  $("#runtimeDismiss").addEventListener("click", dismissRuntimeJob);
  $("#runtimeCopyLog").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("#runtimeLog").textContent);
      toast("Runtime log copied.");
    } catch {
      toast("Clipboard access was unavailable; select the log text manually.", "warn");
    }
  });
  $("#buildExe").addEventListener("click", () => openStudio("deploy"));
  $("#brandHome").addEventListener("click", () => {
    closeOverlay("studioOverlay");
    toast("Pipeline workspace ready.");
  });
  document.addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      try {
        savePipelineState({ announce: true });
        toast("Pipeline saved in this browser.");
      } catch (error) {
        toast(`Pipeline save failed: ${error.message}`, "error");
      }
    }
  });
  $$(".nav-item").forEach(button => button.addEventListener("click", () => {
    $$(".nav-item").forEach(item => item.classList.toggle("active", item === button));
    if (button.dataset.section === "pipeline") closeOverlay("studioOverlay");
    else openStudio(button.dataset.section);
  }));
  $$("[data-close]").forEach(button => button.addEventListener("click", () => closeOverlay(button.dataset.close)));
  $$(".overlay").forEach(overlay => overlay.addEventListener("pointerdown", event => {
    if (event.target === overlay) closeOverlay(overlay.id);
  }));
  $$("[data-view-mode]").forEach(button => button.addEventListener("click", () => {
    state.viewerMode = button.dataset.viewMode;
    renderViewerMode();
  }));
  $("#studioPipeline").addEventListener("click", () => {
    closeOverlay("studioOverlay");
    $("#blockSearch").focus();
    toast("Search or drag any block into the pipeline.");
  });

  // Artifact viewer: play/pause timestep playback, plus the four buttons that
  // rendered but were never bound to anything (Compare / Download sample /
  // Copy artifact ID / Use in pipeline).
  $("#viewerPlay").addEventListener("click", toggleViewerPlayback);
  $("#viewerReset").addEventListener("click", () => resetViewerCamera());
  $("#artifactAddDataset").addEventListener("click", openArtifactDatasetPicker);
  $("#artifactDatasetPickerClose").addEventListener("click", closeArtifactDatasetPicker);
  $("#artifactDatasetSearch").addEventListener("input", event =>
    renderArtifactDatasetChoices(event.target.value)
  );
  $("#artifactSampleSearch").addEventListener("input", event =>
    renderArtifactCatalog(event.target.value)
  );
  $("#artifactUploadDataset").addEventListener("click", () => $("#artifactDatasetFile").click());
  $("#artifactDatasetFile").addEventListener("change", event => {
    const file = event.target.files?.[0];
    if (file) uploadArtifactDataset(file);
    event.target.value = "";
  });
  bindViewerInteractions();
  $("#artifactCompare").addEventListener("click", compareCurrentSample);
  $("#artifactDownload").addEventListener("click", downloadCurrentSample);
  $("#artifactCopyId").addEventListener("click", copyArtifactId);
  $("#artifactUseInPipeline").addEventListener("click", () => useArtifactInPipeline(addBlock, selectNode));

  $("#configSearch").addEventListener("input", event => {
    state.configSearch = event.target.value;
    renderConfig();
  });
  $("#changedOnly").addEventListener("change", renderConfig);
  $("#showInactive").addEventListener("change", renderConfig);
  $("#configMode").addEventListener("change", () => {
    const node = state.nodes.find(item => item.id === state.configNode);
    if (node) {
      snapshot();
      node.config.mode = $("#configMode").value;
    }
    state.configSection = "Required";
    renderConfig();
  });
  $("#applyPreset").addEventListener("click", applyPreset);
  $("#llmConfigure").addEventListener("click", configureViaLlm);
  $("#loadTxt").addEventListener("click", () => $("#configFile").click());
  $("#configFile").addEventListener("change", event => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const parsed = parseConfig(String(reader.result || ""));
      const node = state.nodes.find(item => item.id === state.configNode);
      if (!node) return;
      snapshot();
      node.config = { ...parsed.values };
      retainExplicitConfig(node);
      state.configMessages = [{ type: "", text: `Loaded ${file.name} with ${Object.keys(parsed.values).length} values.` }, ...parsed.messages];
      $("#savedState").textContent = "Unsaved changes";
      renderConfig();
    };
    reader.readAsText(file);
    event.target.value = "";
  });
  $("#exportTxt").addEventListener("click", () => {
    const node = state.nodes.find(item => item.id === state.configNode);
    if (!node) return;
    const modelId = BLOCK_SPECS[node.type].modelId;
    const text = `${rawConfig(node.config, MODEL_CATALOG[modelId].keys)}\n`;
    const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `config_${node.config.mode || "train"}_${modelId}.txt`;
    link.click();
    URL.revokeObjectURL(url);
    toast("Exported the current flat configuration.");
  });
  $("#parseRaw").addEventListener("click", () => {
    const node = state.nodes.find(item => item.id === state.configNode);
    if (!node) return;
    const parsed = parseConfig($("#configRaw").value);
    snapshot();
    node.config = { ...parsed.values };
    retainExplicitConfig(node);
    state.configMessages = parsed.messages.length ? parsed.messages : [{ type: "", text: "Parsed raw .txt without syntax errors." }];
    $("#savedState").textContent = "Unsaved changes";
    renderConfig();
  });
  $("#regenerateRaw").addEventListener("click", () => {
    state.configMessages = [{ type: "", text: "Regenerated text from the current form values." }];
    renderConfig();
  });
  $("#preflightConfig").addEventListener("click", async () => {
    const node = state.nodes.find(item => item.id === state.configNode);
    if (!node || !requireRuntime()) return;
    const button = $("#preflightConfig");
    button.disabled = true;
    button.textContent = "Running real preflight…";
    try {
      const modelId = BLOCK_SPECS[node.type].modelId;
      const text = `${rawConfig(node.config, MODEL_CATALOG[modelId].keys)}\n`;
      const result = await preflightConfigText(text, `${modelId}-${node.config.mode || "config"}`);
      state.api.lastPreflight = result;
      state.configMessages = preflightMessages(result);
      renderConfig();
      toast(result?.ok ? "Authoritative launcher preflight passed." : "Real preflight found blocking errors.", result?.ok ? "" : "error");
    } catch (error) {
      state.configMessages = [{ type: "error", text: `Preflight request failed: ${error.message}` }];
      renderConfig();
      toast(`Preflight request failed: ${error.message}`, "error");
    } finally {
      button.disabled = false;
      button.textContent = "Run preflight";
    }
  });
  $("#explainConfig").addEventListener("click", async () => {
    const node = state.nodes.find(item => item.id === state.configNode);
    if (!node) return;
    const button = $("#explainConfig");
    button.disabled = true;
    button.textContent = "Explaining…";
    try {
      const result = await explainConfig();
      state.configMessages = explainMessages(result);
      renderConfig();
      toast(result?.error ? `Explain-config failed: ${result.error}` : "Explained every configured, required, inactive, and unknown key.", result?.error ? "error" : "");
    } catch (error) {
      state.configMessages = [{ type: "error", text: `Explain-config request failed: ${error.message}` }];
      renderConfig();
      toast(`Explain-config request failed: ${error.message}`, "error");
    } finally {
      button.disabled = false;
      button.textContent = "Explain config";
    }
  });
  $("#saveConfig").addEventListener("click", async () => {
    const node = state.nodes.find(item => item.id === state.configNode);
    if (!node) return;
    const modelId = BLOCK_SPECS[node.type].modelId;
    if (state.api.connected) {
      try {
        const result = await apiRequest("/api/config/save", {
          method: "POST",
          body: {
            label: `${modelId}-${node.config.mode || "config"}`,
            config: `${rawConfig(node.config, MODEL_CATALOG[modelId].keys)}\n`
          }
        });
        snapshot();
        node.savedConfigPath = result.path;
        $("#savedState").textContent = `Saved · ${result.path}`;
      } catch (error) {
        toast(`Could not persist config: ${error.message}`, "error");
        return;
      }
    } else {
      $("#savedState").textContent = "Saved in browser only · runtime offline";
    }
    closeOverlay("configOverlay");
    render();
    toast(node.savedConfigPath ? `Configuration saved to ${node.savedConfigPath}.` : "Configuration retained in this browser session.");
  });

  $("#zoomIn").addEventListener("click", () => setZoom(state.view.scale * 1.12));
  $("#zoomOut").addEventListener("click", () => setZoom(state.view.scale / 1.12));
  $("#fitGraph").addEventListener("click", fitGraphView);
  $("#hideLibrary").addEventListener("click", () => setPanelVisibility("library", false));
  $("#showLibrary").addEventListener("click", () => setPanelVisibility("library", true));
  $("#hideInspector").addEventListener("click", () => setPanelVisibility("inspector", false));
  $("#showInspector").addEventListener("click", () => setPanelVisibility("inspector", true));
  $("#stage").addEventListener("wheel", event => {
    event.preventDefault();
    const factor = Math.exp(-event.deltaY * .0014);
    setZoom(state.view.scale * factor, { x: event.clientX, y: event.clientY });
  }, { passive: false });
  $("#stage").addEventListener("dragover", event => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  });
  $("#stage").addEventListener("drop", event => {
    event.preventDefault();
    const type = event.dataTransfer.getData("application/x-ai-cae-block");
    if (!BLOCK_SPECS[type]) return;
    const rect = $("#stage").getBoundingClientRect();
    addBlock(type, {
      x: (event.clientX - rect.left - state.view.x) / state.view.scale - NODE_WIDTH / 2,
      y: (event.clientY - rect.top - state.view.y) / state.view.scale - NODE_HEADER_HEIGHT / 2
    });
  });
  $("#stage").addEventListener("pointerdown", event => {
    if (event.target === $("#stage") || event.target === $("#canvasWorld")) {
      startCanvasPan(event);
    }
  });
  document.addEventListener("pointermove", dragNode);
  document.addEventListener("pointermove", panCanvas);
  document.addEventListener("pointerup", event => {
    stopNodeDrag();
    stopCanvasPan(event);
  });
  document.addEventListener("pointercancel", event => {
    stopNodeDrag();
    stopCanvasPan(event);
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") {
      const open = [...$$(".overlay.open")].pop();
      if (open) closeOverlay(open.id);
      else {
        state.pendingPort = null;
        toast("Pending link cancelled.");
      }
    }
    if ((event.key === "Delete" || event.key === "Backspace") && !event.target.matches("input,textarea,select")) deleteSelected();
  });
  window.addEventListener("resize", renderEdges);
}
