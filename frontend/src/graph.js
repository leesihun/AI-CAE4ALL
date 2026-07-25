import { $, $$, escapeHtml, toast } from "./dom.js";
import { state, snapshot, nodePortRows, nodeHeight } from "./state.js";
import {
  ICONS, BLOCK_SPECS, MODEL_CATALOG, TYPE_META, TEMPLATES,
  NODE_WIDTH, PORT_START_Y, PORT_GAP
} from "./constants.js";
import { previewGraphic, nodeVisualLabel } from "./graphics.js";
import { typeColor, compatible, validateGraph } from "./validate.js";
import { openArtifact } from "./viewer.js";
import { runGraph } from "./run.js";
import { renderInspector } from "./inspector.js";

export function paletteRender(query = "") {
  const normalized = query.trim().toLowerCase();
  const order = ["Sources", "Preparation", "Models", "Execution", "Optimization", "Evaluation", "Deployment", "Outputs"];
  $("#paletteList").innerHTML = order.map(category => {
    const entries = Object.entries(BLOCK_SPECS)
      .filter(([, spec]) => {
        if (spec.category !== category) return false;
        const haystack = `${spec.label} ${spec.description} ${spec.modelId || ""}`.toLowerCase();
        return !normalized || haystack.includes(normalized);
      })
      .sort(([, left], [, right]) =>
        left.label.localeCompare(right.label, undefined, { sensitivity: "base" })
      );
    if (!entries.length) return "";
    return `<section class="palette-group">
      <header class="palette-group-head"><span>${category}</span><span>${entries.length}</span></header>
      ${entries.map(([type, spec]) => `<button class="palette-item" draggable="true" data-block-type="${type}" style="--accent:${spec.accent}">
        <span class="palette-icon">${ICONS[spec.icon]}</span>
        <span class="palette-copy">
          <span class="palette-name">${escapeHtml(spec.label)}</span>
          <span class="palette-desc">${escapeHtml(spec.description)}</span>
          <span class="maturity ${spec.maturity}">${spec.maturity}</span>
        </span>
        <span class="palette-add">+</span>
      </button>`).join("")}
    </section>`;
  }).join("") || `<div class="inspect-empty" style="height:auto;padding:36px 10px"><p>No blocks match “${escapeHtml(query)}”.</p></div>`;

  $$(".palette-item").forEach(button => {
    button.addEventListener("click", () => addBlock(button.dataset.blockType));
    button.addEventListener("dragstart", event => {
      event.dataTransfer.setData("application/x-ai-cae-block", button.dataset.blockType);
      event.dataTransfer.effectAllowed = "copy";
    });
  });
}

export function loadTemplate(name, saveHistory = true) {
  const template = TEMPLATES[name] || TEMPLATES.simulgen;
  if (saveHistory && state.nodes.length) snapshot();
  state.nodes = template.nodes.map(([id, type, x, y, config]) => ({
    id, type, x, y, config: { ...BLOCK_SPECS[type].defaults, ...(config || {}) },
    status: "idle", progress: 0
  }));
  state.edges = template.edges.map((edge, index) => ({
    id: `edge_${index + 1}`,
    fromNode: edge[0], fromPort: edge[1], toNode: edge[2], toPort: edge[3]
  }));
  state.selectedNode = null;
  state.pendingPort = null;
  layoutGraph(false, false);
  state.view = { x: 26, y: 54, scale: .78 };
  $("#pipelineName").value = template.name;
  $("#savedState").textContent = "Saved locally · just now";
  render();
}

export function addBlock(type, position) {
  const spec = BLOCK_SPECS[type];
  if (!spec) return;
  snapshot();
  const id = `${type.replaceAll(".", "_")}_${state.nodeCounter++}`;
  const existing = state.nodes.length;
  const visibleWorldLeft = -state.view.x / state.view.scale;
  const visibleWorldTop = -state.view.y / state.view.scale;
  const viewportPosition = {
    x: visibleWorldLeft + 72 + (existing % 3) * 340,
    y: visibleWorldTop + 96 + Math.floor(existing / 3) * 370
  };
  state.nodes.push({
    id,
    type,
    x: Math.max(24, position?.x ?? viewportPosition.x),
    y: Math.max(54, position?.y ?? viewportPosition.y),
    config: { ...spec.defaults },
    status: "idle",
    progress: 0
  });
  state.selectedNode = id;
  render();
  toast(`${spec.label} added. Click either socket first, then choose a highlighted compatible socket.`);
}

export function duplicateNode(id) {
  const source = state.nodes.find(node => node.id === id);
  if (!source) return;
  snapshot();
  const copy = JSON.parse(JSON.stringify(source));
  copy.id = `${source.id}_copy_${state.nodeCounter++}`;
  copy.x += 35;
  copy.y += 35;
  copy.status = "idle";
  copy.progress = 0;
  state.nodes.push(copy);
  state.selectedNode = copy.id;
  render();
}

export function deleteSelected() {
  if (!state.selectedNode) return;
  snapshot();
  const id = state.selectedNode;
  state.nodes = state.nodes.filter(node => node.id !== id);
  state.edges = state.edges.filter(edge => edge.fromNode !== id && edge.toNode !== id);
  state.selectedNode = null;
  render();
  toast("Removed the selected block.", "warn");
}

export function portTop(index) {
  return PORT_START_Y + index * PORT_GAP;
}

export function applyViewTransform() {
  $("#canvasWorld").style.transform = `translate(${state.view.x}px, ${state.view.y}px) scale(${state.view.scale})`;
  if ($("#zoomLevel")) $("#zoomLevel").textContent = `${Math.round(state.view.scale * 100)}%`;
}

export function portDetail(element) {
  return {
    nodeId: element.dataset.node,
    portId: element.dataset.port,
    type: element.dataset.portType,
    direction: element.dataset.direction
  };
}

export function portsCanLink(first, second) {
  if (!first || !second || first.nodeId === second.nodeId || first.direction === second.direction) return false;
  const output = first.direction === "output" ? first : second;
  const input = first.direction === "input" ? first : second;
  return compatible(output.type, input.type);
}

export function portStateClass(nodeId, port, direction) {
  const pending = state.pendingPort;
  if (!pending) return "";
  if (pending.nodeId === nodeId && pending.portId === port.id && pending.direction === direction) return " link-source";
  const candidate = { nodeId, portId: port.id, type: port.type, direction };
  return portsCanLink(pending, candidate) ? " link-target" : " link-blocked";
}

export function refreshPortHighlights() {
  $$(".port").forEach(port => {
    const detail = portDetail(port);
    const pending = state.pendingPort;
    const isSource = Boolean(
      pending
      && pending.nodeId === detail.nodeId
      && pending.portId === detail.portId
      && pending.direction === detail.direction
    );
    const canLink = portsCanLink(pending, detail);
    port.classList.toggle("link-source", isSource);
    port.classList.toggle("link-target", canLink);
    port.classList.toggle("link-blocked", Boolean(pending) && !isSource && !canLink);
  });
}

export function connectPortDetails(first, second) {
  if (!first || !second) return false;
  if (first.nodeId === second.nodeId) {
    toast("A block cannot link to itself.", "warn");
    return false;
  }
  if (first.direction === second.direction) {
    state.pendingPort = second;
    renderNodes();
    renderEdges();
    toast(`Selected ${second.direction} port. Now choose a compatible ${second.direction === "output" ? "input" : "output"} port.`);
    return false;
  }
  const output = first.direction === "output" ? first : second;
  const input = first.direction === "input" ? first : second;
  if (!compatible(output.type, input.type)) {
    toast(`Cannot link ${output.type} to ${input.type}. Compatible ports are highlighted in green.`, "error");
    return false;
  }
  snapshot();
  state.edges = state.edges.filter(edge => !(edge.toNode === input.nodeId && edge.toPort === input.portId));
  state.edges.push({
    id: `edge_${Date.now()}`,
    fromNode: output.nodeId,
    fromPort: output.portId,
    toNode: input.nodeId,
    toPort: input.portId
  });
  state.pendingPort = null;
  render();
  toast("Blocks linked.");
  return true;
}

export function renderNodes() {
  applyViewTransform();
  $("#nodeLayer").innerHTML = state.nodes.map(node => {
    const spec = BLOCK_SPECS[node.type];
    const portRows = nodePortRows(node);
    const inputs = spec.inputs.map((port, index) => `<button class="port input${portStateClass(node.id, port, "input")}" draggable="true" data-node="${node.id}" data-direction="input" data-port="${port.id}" data-port-type="${port.type}" style="top:${portTop(index) - 13}px;--port:${typeColor(port.type)}" aria-label="${escapeHtml(port.label)} input" title="Connect ${escapeHtml(port.label)} input"><span class="port-label">${escapeHtml(port.label)}${port.required ? " *" : ""}</span></button>`).join("");
    const outputs = spec.outputs.map((port, index) => `<button class="port output${portStateClass(node.id, port, "output")}" draggable="true" data-node="${node.id}" data-direction="output" data-port="${port.id}" data-port-type="${port.type}" style="top:${portTop(index) - 13}px;--port:${typeColor(port.type)}" aria-label="${escapeHtml(port.label)} output" title="Connect ${escapeHtml(port.label)} output"><span class="port-label">${escapeHtml(port.label)}</span></button>`).join("");
    return `<article class="node ${node.status}${state.selectedNode === node.id ? " selected" : ""}" data-node-id="${node.id}" style="left:${node.x}px;top:${node.y}px;--node-accent:${spec.accent};--progress:${node.progress}%">
      ${inputs}${outputs}
      <header class="node-head" data-drag-handle>
        <span class="node-icon">${ICONS[spec.icon]}</span>
        <span><span class="node-title">${escapeHtml(spec.label)}</span><span class="node-kind">${spec.isModel ? "Model · " + MODEL_CATALOG[spec.modelId].modes.join(" / ") : `${spec.category} · ${spec.maturity}`}</span></span>
        <button class="node-menu" aria-label="More actions">•••</button>
      </header>
      <div class="node-preview" data-preview="${node.id}">${previewGraphic(spec.visual, node.id.length)}<span class="preview-label">${escapeHtml(nodeVisualLabel(spec))}</span></div>
      <div class="node-port-space" style="height:${portRows * PORT_GAP + 6}px" aria-hidden="true"></div>
      <div class="node-summary"><span class="status"><i></i>${node.status === "idle" ? "ready" : node.status}</span><span>${escapeHtml(spec.sampleLabel)}</span></div>
      <div class="node-progress"><i></i></div>
      <div class="node-actions">
        <button class="button" data-inspect="${node.id}">Inspect</button>
        <button class="button" data-run="${node.id}">${spec.isModel ? "Train" : "Run"}</button>
      </div>
    </article>`;
  }).join("");

  $$(".node").forEach(element => {
    const id = element.dataset.nodeId;
    element.addEventListener("pointerdown", event => {
      if (event.target.closest("button,.node-preview")) return;
      selectNode(id);
    });
    $("[data-drag-handle]", element).addEventListener("pointerdown", event => startNodeDrag(event, id));
  });
  $$("[data-preview]").forEach(element => element.addEventListener("click", () => openArtifact(element.dataset.preview)));
  $$("[data-inspect]").forEach(button => button.addEventListener("click", () => openArtifact(button.dataset.inspect)));
  $$("[data-run]").forEach(button => button.addEventListener("click", () => runGraph(button.dataset.run)));
  $$(".port").forEach(port => {
    port.addEventListener("click", event => handlePortClick(event, port));
    port.addEventListener("dragstart", event => {
      const detail = portDetail(port);
      state.pendingPort = detail;
      event.dataTransfer.setData("application/x-ai-cae-port", JSON.stringify(detail));
      event.dataTransfer.effectAllowed = "link";
      refreshPortHighlights();
    });
    port.addEventListener("dragover", event => {
      if (!state.pendingPort || !portsCanLink(state.pendingPort, portDetail(port))) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "link";
    });
    port.addEventListener("drop", event => {
      event.preventDefault();
      event.stopPropagation();
      let source = state.pendingPort;
      try {
        source = JSON.parse(event.dataTransfer.getData("application/x-ai-cae-port")) || source;
      } catch {
        // Keep the in-memory source selected by dragstart.
      }
      connectPortDetails(source, portDetail(port));
    });
  });
}

export function renderEdges() {
  const paths = state.edges.map((edge, edgeIndex) => {
    const source = state.nodes.find(node => node.id === edge.fromNode);
    const target = state.nodes.find(node => node.id === edge.toNode);
    if (!source || !target) return "";
    const sourceSpec = BLOCK_SPECS[source.type];
    const targetSpec = BLOCK_SPECS[target.type];
    const outIndex = Math.max(0, sourceSpec.outputs.findIndex(port => port.id === edge.fromPort));
    const inIndex = Math.max(0, targetSpec.inputs.findIndex(port => port.id === edge.toPort));
    const sourcePort = sourceSpec.outputs[outIndex];
    const x1 = source.x + NODE_WIDTH;
    const y1 = source.y + portTop(outIndex);
    const x2 = target.x;
    const y2 = target.y + portTop(inIndex);
    const span = x2 - x1;
    let path;
    const isBus = span > 340;
    if (isBus) {
      const laneOffset = 44 + (edgeIndex % 3) * 18;
      const routeAbove = source.y <= target.y;
      const laneY = routeAbove
        ? Math.max(28, Math.min(source.y, target.y) - laneOffset)
        : Math.max(source.y + nodeHeight(source), target.y + nodeHeight(target)) + laneOffset;
      const lead = Math.min(72, Math.max(42, span * .14));
      path = `M${x1} ${y1} C${x1 + 24} ${y1},${x1 + lead - 16} ${laneY},${x1 + lead} ${laneY} L${x2 - lead} ${laneY} C${x2 - lead + 16} ${laneY},${x2 - 24} ${y2},${x2} ${y2}`;
    } else {
      const bend = Math.max(58, Math.abs(span) * .44);
      path = `M${x1} ${y1} C${x1 + bend} ${y1},${x2 - bend} ${y2},${x2} ${y2}`;
    }
    const selected = state.selectedNode && (edge.fromNode === state.selectedNode || edge.toNode === state.selectedNode);
    const color = typeColor(sourcePort?.type || "artifact");
    return `<path class="edge-shadow${isBus ? " bus" : ""}" d="${path}"/><path class="edge${isBus ? " bus" : ""}${selected ? " selected" : ""}" style="--edge-color:${color}" d="${path}"/>`;
  }).join("");
  $("#edgeLayer").innerHTML = `<defs><marker id="edgeArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 10 5 0 10Z" fill="context-stroke"/></marker></defs>${paths}`;
  $("#edgeLayer").style.transform = "none";
}

export function startNodeDrag(event, id) {
  if (event.button !== 0) return;
  event.preventDefault();
  event.stopPropagation();
  const node = state.nodes.find(item => item.id === id);
  if (!node) return;
  state.selectedNode = id;
  setPanelVisibility("inspector", true);
  state.drag = {
    id,
    startX: event.clientX,
    startY: event.clientY,
    nodeX: node.x,
    nodeY: node.y,
    started: false
  };
  event.currentTarget.setPointerCapture?.(event.pointerId);
  $$(".node").forEach(element => element.classList.toggle("selected", element.dataset.nodeId === id));
  renderEdges();
  renderInspector();
}

export function dragNode(event) {
  if (!state.drag) return;
  const node = state.nodes.find(item => item.id === state.drag.id);
  if (!node) return;
  const deltaX = event.clientX - state.drag.startX;
  const deltaY = event.clientY - state.drag.startY;
  if (!state.drag.started && Math.abs(deltaX) + Math.abs(deltaY) < 3) return;
  if (!state.drag.started) {
    snapshot();
    state.drag.started = true;
  }
  node.x = Math.max(10, state.drag.nodeX + deltaX / state.view.scale);
  node.y = Math.max(10, state.drag.nodeY + deltaY / state.view.scale);
  const element = $(`[data-node-id="${state.drag.id}"]`);
  if (element) {
    element.style.left = `${node.x}px`;
    element.style.top = `${node.y}px`;
  }
  renderEdges();
}

export function stopNodeDrag() {
  state.drag = null;
}

export function startCanvasPan(event) {
  if (![0, 1].includes(event.button)) return;
  event.preventDefault();
  state.pan = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    viewX: state.view.x,
    viewY: state.view.y,
    moved: false
  };
  $("#stage").classList.add("panning");
  if (event.isTrusted) {
    try {
      $("#stage").setPointerCapture?.(event.pointerId);
    } catch {
      // Pointer capture is an enhancement; document-level listeners still pan.
    }
  }
}

export function panCanvas(event) {
  if (!state.pan || event.pointerId !== state.pan.pointerId) return;
  const dx = event.clientX - state.pan.startX;
  const dy = event.clientY - state.pan.startY;
  if (Math.abs(dx) + Math.abs(dy) > 4) state.pan.moved = true;
  state.view.x = state.pan.viewX + dx;
  state.view.y = state.pan.viewY + dy;
  applyViewTransform();
}

export function stopCanvasPan(event) {
  if (!state.pan || (event?.pointerId !== undefined && event.pointerId !== state.pan.pointerId)) return;
  const wasClick = !state.pan.moved;
  state.pan = null;
  $("#stage").classList.remove("panning");
  if (wasClick) {
    state.selectedNode = null;
    state.pendingPort = null;
    render();
  }
}

export function setPanelVisibility(panel, visible) {
  const shell = $("#studioShell");
  const className = panel === "library" ? "library-collapsed" : "inspector-collapsed";
  shell.classList.toggle(className, !visible);
  const hideButton = panel === "library" ? $("#hideLibrary") : $("#hideInspector");
  const showButton = panel === "library" ? $("#showLibrary") : $("#showInspector");
  hideButton?.setAttribute("aria-expanded", String(visible));
  showButton?.setAttribute("aria-expanded", String(visible));
  window.setTimeout(renderEdges, 240);
}

export function handlePortClick(event, element) {
  event.stopPropagation();
  const detail = portDetail(element);
  if (!state.pendingPort) {
    state.pendingPort = detail;
    renderNodes();
    renderEdges();
    toast(`Selected ${TYPE_META[detail.type]?.label || detail.type} ${detail.direction}. Compatible ports are highlighted; click or drag to connect.`);
    return;
  }
  connectPortDetails(state.pendingPort, detail);
}

export function selectNode(id) {
  state.selectedNode = id;
  setPanelVisibility("inspector", true);
  renderNodes();
  renderEdges();
  renderInspector();
}

export function renderGraphMeta() {
  const container = $("#graphStats");
  if (!container) return;
  const errors = validateGraph(false);
  const running = state.nodes.filter(node => node.status === "running").length;
  const complete = state.nodes.filter(node => node.status === "complete").length;
  const statusClass = errors.length ? "graph-warning" : running ? "graph-running" : "graph-ready";
  const statusText = errors.length
    ? `${errors.length} issue${errors.length === 1 ? "" : "s"}`
    : running
      ? `${running} running`
      : complete
        ? `${complete} completed`
        : "Ready";
  container.innerHTML = `
    <span><strong>${state.nodes.length}</strong> blocks</span>
    <span><strong>${state.edges.length}</strong> links</span>
    <span class="${statusClass}"><i></i>${statusText}</span>`;
}

export function render() {
  renderNodes();
  renderEdges();
  renderInspector();
  renderGraphMeta();
}

export function layoutGraph(saveHistory = true, shouldRender = true) {
  if (!state.nodes.length) return;
  if (saveHistory) snapshot();

  const levels = new Map(state.nodes.map(node => [node.id, 0]));
  for (let pass = 0; pass < state.nodes.length; pass += 1) {
    state.edges.forEach(edge => {
      const sourceLevel = levels.get(edge.fromNode);
      const targetLevel = levels.get(edge.toNode);
      if (sourceLevel === undefined || targetLevel === undefined) return;
      levels.set(edge.toNode, Math.min(state.nodes.length - 1, Math.max(targetLevel, sourceLevel + 1)));
    });
  }

  const columns = new Map();
  state.nodes.forEach(node => {
    const level = levels.get(node.id) || 0;
    if (!columns.has(level)) columns.set(level, []);
    columns.get(level).push(node);
  });
  const columnHeights = new Map();
  columns.forEach((nodes, level) => {
    columnHeights.set(level, nodes.reduce((sum, node, index) =>
      sum + nodeHeight(node) + (index ? 70 : 0), 0));
  });
  const tallestColumn = Math.max(...columnHeights.values(), 0);
  [...columns.entries()].sort(([left], [right]) => left - right).forEach(([level, nodes]) => {
    let y = 76 + Math.max(0, (tallestColumn - columnHeights.get(level)) / 2);
    nodes.forEach(node => {
      node.x = 70 + level * 360;
      node.y = y;
      y += nodeHeight(node) + 70;
    });
  });

  if (shouldRender) {
    render();
    toast("Pipeline arranged by dependency level.");
  }
}

export function arrangeGraph() {
  layoutGraph(true, true);
}

export function setZoom(value, anchor = null) {
  const previous = state.view.scale;
  const next = Math.min(1.6, Math.max(.45, value));
  if (anchor && next !== previous) {
    const rect = $("#stage").getBoundingClientRect();
    const localX = anchor.x - rect.left;
    const localY = anchor.y - rect.top;
    const worldX = (localX - state.view.x) / previous;
    const worldY = (localY - state.view.y) / previous;
    state.view.x = localX - worldX * next;
    state.view.y = localY - worldY * next;
  }
  state.view.scale = next;
  applyViewTransform();
}

export function fitGraphView() {
  if (!state.nodes.length) {
    state.view = { x: 22, y: 34, scale: 1 };
    applyViewTransform();
    return;
  }
  const minX = Math.min(...state.nodes.map(node => node.x));
  const minY = Math.min(...state.nodes.map(node => node.y));
  const maxX = Math.max(...state.nodes.map(node => node.x + NODE_WIDTH));
  const maxY = Math.max(...state.nodes.map(node => node.y + nodeHeight(node)));
  const rect = $("#stage").getBoundingClientRect();
  const scale = Math.min(1.1, Math.max(.45, Math.min(
    (rect.width - 90) / Math.max(1, maxX - minX),
    (rect.height - 110) / Math.max(1, maxY - minY)
  )));
  state.view.scale = scale;
  state.view.x = (rect.width - (maxX - minX) * scale) / 2 - minX * scale;
  state.view.y = (rect.height - (maxY - minY) * scale) / 2 - minY * scale;
  applyViewTransform();
}
