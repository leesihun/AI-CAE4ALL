import { state } from "./state.js";

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

export const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
})[char]);

/**
 * Bind an event to an element that a workspace has just rendered.
 *
 * The workspace renderers are async: they await the API, write innerHTML, then
 * bind by document-wide id. If the user switches section (or closes the modal)
 * while that await is in flight, the element is gone by the time the binding
 * runs and `$(...).addEventListener` throws. Nothing useful can be bound to a
 * view that is no longer on screen, so this no-ops instead of crashing.
 */
export function on(selector, event, handler, options) {
  const element = $(selector);
  if (!element) return false;
  element.addEventListener(event, handler, options);
  return true;
}

export function toast(message, type = "") {
  const stack = $("#toastStack");
  if (!stack) return;          // a toast fired against a torn-down document
  while (stack.children.length >= 4) stack.firstElementChild?.remove();
  const item = document.createElement("div");
  item.className = `toast ${type}`;
  item.setAttribute("role", type === "error" ? "alert" : "status");
  item.textContent = message;
  stack.appendChild(item);
  window.setTimeout(() => item.remove(), 3300);
}

export function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
}

/**
 * The ids of every open overlay, oldest first.
 *
 * Escape has to close the modal the user is actually looking at, and the only
 * record of which one that is used to be `.overlay.open` in *document* order --
 * i.e. the order the overlays happen to be authored in index.html. That made
 * the behaviour depend on markup position: adding an overlay to the end of the
 * file silently stole Escape from every overlay above it. Track the real open
 * order instead, and stack the paint order to match so the newest modal is also
 * the visible one. The ceiling of 89 keeps every overlay under the topbar (90),
 * which deliberately stays clickable while a modal is open.
 */
const overlayOrder = [];

export function watchOverlayOrder() {
  $$(".overlay").forEach(overlay => {
    const sync = () => {
      const at = overlayOrder.indexOf(overlay.id);
      const open = overlay.classList.contains("open");
      if (open && at === -1) overlayOrder.push(overlay.id);
      else if (!open && at !== -1) overlayOrder.splice(at, 1);
      else return;
      overlayOrder.forEach((id, index) => {
        const element = $(`#${id}`);
        if (element) element.style.zIndex = String(Math.min(89, 80 + index));
      });
      if (!open) overlay.style.zIndex = "";
    };
    new MutationObserver(sync).observe(overlay, { attributes: true, attributeFilter: ["class"] });
    sync();
  });
}

export function topOverlayId() {
  return overlayOrder[overlayOrder.length - 1] || null;
}

export function closeOverlay(id) {
  $(`#${id}`)?.classList.remove("open");
  if (id === "configOverlay") state.configNode = null;
}
