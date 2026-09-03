import { $, $$, toast, closeOverlay, watchOverlayOrder, topOverlayId } from "./dom.js";
import { state, snapshot, restoreSnapshot } from "./state.js";
import { BLOCK_SPECS, MODEL_CATALOG, NODE_WIDTH, NODE_HEADER_HEIGHT } from "./constants.js";
import { apiRequest, requireRuntime } from "./api.js";
import { rawConfig, parseConfig, applyPreset, renderConfig, explainConfig, explainMessages, configureViaLlm } from "./config.js";
import { preflightConfigText, preflightMessages } from "./validate.js";
import {
  paletteRender, loadTemplate, arrangeGraph, setZoom, fitGraphView,
  setPanelVisibility, dragNode, panCanvas, stopNodeDrag, stopCanvasPan,
  addBlock, deleteSelected, render, selectNode, startCanvasPan, renderEdges
} from "./graph.js";
import { validatePipeline, runGraph, stopRun, dismissRuntimeJob, toggleDrawerCollapsed, watchModalsForDrawer } from "./run.js";
import { openStudio } from "./studio.js";
import {
  PipelineLoadCancelledError, downloadPipelineJson, importPipelineJson,
  savePipelineState, schedulePipelineSave
} from "./persistence.js";
import { applyGraphAutofill, markManualConfigValue, resetManualConfigValues } from "./autofill.js";

function undoGraphChange() {
  const previous = state.history.pop();
  if (!previous) {
    toast("Nothing to undo.", "warn");
    return;
  }
  restoreSnapshot(previous);
  applyGraphAutofill();
  state.selectedNode = null;
  state.selectedEdge = null;
  state.pendingPort = null;
  $("#templateSelect").value = "saved";
  render();
  schedulePipelineSave();
  toast("Undid the last graph change.");
}

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
  watchOverlayOrder();
  $("#blockSearch").addEventListener("input", event => paletteRender(event.target.value));
  $("#templateSelect").addEventListener("change", event => {
    if (loadTemplate(event.target.value) === false) event.target.value = "saved";
  });
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
      await importPipelineJson(file);
      $("#templateSelect").value = "saved";
      render();
      toast(`Imported ${file.name}.`);
    } catch (error) {
      if (error instanceof PipelineLoadCancelledError) toast("Pipeline import cancelled.", "warn");
      else toast(`Pipeline import failed: ${error.message}`, "error");
    }
  });
  $("#arrangeGraph").addEventListener("click", arrangeGraph);
  $("#undoGraph").addEventListener("click", undoGraphChange);
  $("#pipelineName").addEventListener("input", schedulePipelineSave);
  $("#validateTop").addEventListener("click", () => validatePipeline());
  $("#runTop").addEventListener("click", () => runGraph());
  $("#stopRun").addEventListener("click", stopRun);
  $("#runtimeCancel").addEventListener("click", stopRun);
  $("#runtimeMinimize").addEventListener("click", toggleDrawerCollapsed);
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
    // The canvas shell remains inert until the overlay observer processes the
    // close mutation. Focusing synchronously therefore fails and the generic
    // return-to-trigger rule wins. Move focus after inert has been removed.
    window.setTimeout(() => $("#blockSearch").focus(), 0);
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
    // The panel used to keep the previous verdict on screen for the ~6s the
    // authoritative preflight takes, so a confident answer about fields that had
    // since been edited was indistinguishable from a fresh one.
    state.configMessages = [{ type: "", text: "Running the authoritative launcher preflight… the verdict below is from the previous run until it returns." }];
    renderConfig();
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
      // Clicking bare canvas has to take focus off whatever field held it.
      // The canvas shortcuts are suppressed while a field is focused, and the
      // canvas is not itself focusable, so without this the user renames the
      // pipeline, clicks away, presses F to fit -- and instead of fitting, an
      // "f" is appended to the pipeline name while every shortcut stays dead.
      const active = document.activeElement;
      if (active && active !== document.body && active.matches("input,textarea,select")) active.blur();
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
      const open = topOverlayId();
      if (open) closeOverlay(open);
      else {
        state.pendingPort = null;
        toast("Pending link cancelled.");
      }
      return;
    }

    // Everything below is a canvas shortcut, so it must never fire while the
    // user is typing a config value, a pipeline name, or a search term.
    const typing = event.target.matches("input,textarea,select") || event.target.isContentEditable;
    const chord = event.ctrlKey || event.metaKey;

    if (chord && event.key.toLowerCase() === "z" && !typing) {
      event.preventDefault();
      undoGraphChange();
      return;
    }
    if (chord && event.key === "Enter") {
      event.preventDefault();
      runGraph();
      return;
    }
    if (chord) return;                       // leave every other browser chord alone

    if (event.key === "?" ) {
      event.preventDefault();
      toggleShortcutsOverlay();
      return;
    }
    if (typing) return;

    if (event.key === "Delete" || event.key === "Backspace") { deleteSelected(); return; }
    if (event.key === "/") {
      event.preventDefault();
      setPanelVisibility("library", true);
      $("#blockSearch").focus();
      $("#blockSearch").select();
      return;
    }
    const key = event.key.toLowerCase();
    if (key === "f") { fitGraphView(); return; }
    if (key === "l") { arrangeGraph(); return; }
    if (key === "v") { validatePipeline(); return; }
    if (event.key === "+" || event.key === "=") { setZoom(state.view.scale * 1.12); return; }
    if (event.key === "-" || event.key === "_") { setZoom(state.view.scale / 1.12); }
  });

  $("#shortcutsTop").addEventListener("click", toggleShortcutsOverlay);
  $("#welcomeDismiss").addEventListener("click", () => closeOverlay("welcomeOverlay"));
  $("#welcomeTour").addEventListener("click", () => {
    closeOverlay("welcomeOverlay");
    toggleShortcutsOverlay();
  });
  watchModalsForDrawer();
  maybeShowWelcome();
  window.addEventListener("resize", renderEdges);
}

const WELCOME_STORAGE_KEY = "ai-cae4all.studio.welcomed.v1";

/**
 * The studio boots straight into a populated pipeline, so there is no empty
 * state in which to explain what a block is or what Validate does. Show the
 * orientation card once per browser instead. A private-mode browser that throws
 * on localStorage should still get a usable studio, so failures are swallowed.
 */
function maybeShowWelcome() {
  // ?welcome=0 suppresses it and ?welcome=1 forces it. Smoke runners and demo
  // machines start from a clean profile every time, so they would otherwise
  // meet a modal on every launch; forcing it back is how you review the card
  // itself without clearing storage by hand.
  const wanted = new URLSearchParams(location.search).get("welcome");
  if (wanted === "0") return;
  if (wanted === "1") {
    $("#welcomeOverlay")?.classList.add("open");
    return;
  }
  let seen = true;
  try {
    seen = localStorage.getItem(WELCOME_STORAGE_KEY) === "1";
  } catch {
    return;                                  // storage blocked: never nag
  }
  if (seen) return;
  $("#welcomeOverlay")?.classList.add("open");
  try {
    localStorage.setItem(WELCOME_STORAGE_KEY, "1");
  } catch {
    /* shown this session is better than a modal on every reload */
  }
}

/**
 * The canvas grew a real set of shortcuts, and an undiscoverable shortcut is
 * worth about as much as no shortcut. The `?` panel is the only place that
 * lists them, so it is generated from one table that doubles as the reference.
 */
const SHORTCUTS = [
  ["Canvas", [
    ["F", "Fit the whole pipeline in view"],
    ["L", "Auto layout the blocks"],
    ["+ / −", "Zoom in / out"],
    ["Del", "Delete the selected block or link"],
    ["Esc", "Close a dialog, or cancel a half-drawn link"]
  ]],
  ["Pipeline", [
    ["Ctrl + Enter", "Run the pipeline"],
    ["V", "Validate without running"],
    ["Ctrl + S", "Save the pipeline in this browser"],
    ["Ctrl + Z", "Undo the last graph change"]
  ]],
  ["Finding things", [
    ["/", "Jump to the block search box"],
    ["?", "Show or hide this list"]
  ]]
];

function toggleShortcutsOverlay() {
  const overlay = $("#shortcutsOverlay");
  if (!overlay) return;
  if (overlay.classList.contains("open")) {
    closeOverlay("shortcutsOverlay");
    return;
  }
  $("#shortcutsBody").innerHTML = SHORTCUTS.map(([group, rows]) => `<section class="shortcut-group">
    <h3>${group}</h3>
    ${rows.map(([keys, label]) => `<div class="shortcut-row"><kbd>${keys}</kbd><span>${label}</span></div>`).join("")}
  </section>`).join("");
  overlay.classList.add("open");
}
