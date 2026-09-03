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
 * the visible one. Modals start above the runtime drawer and topbar; toast
 * notices retain their separate, higher layer.
 */
const overlayOrder = [];
const overlayFocusOrigins = new Map();
const OVERLAY_Z_INDEX = 130;
const FOCUSABLE_SELECTOR = [
  "a[href]",
  "area[href]",
  "button:not([disabled])",
  "input:not([disabled]):not([type='hidden'])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "iframe",
  "[contenteditable]:not([contenteditable='false'])",
  "[tabindex]:not([tabindex='-1'])"
].join(",");

let overlayObserver = null;
let overlayKeyboardBound = false;

function canReceiveFocus(element) {
  if (!(element instanceof HTMLElement) || !element.isConnected || element.tabIndex < 0) return false;
  if (element.closest("[inert], [hidden], [aria-hidden='true']")) return false;
  const style = window.getComputedStyle(element);
  return style.display !== "none" && style.visibility !== "hidden" && element.getClientRects().length > 0;
}

function focusableElements(overlay) {
  return $$(FOCUSABLE_SELECTOR, overlay).filter(canReceiveFocus);
}

function focusElement(element) {
  if (!canReceiveFocus(element)) return false;
  element.focus({ preventScroll: true });
  return document.activeElement === element;
}

function focusInsideOverlay(overlay, fromEnd = false) {
  if (!overlay) return false;
  const focusable = focusableElements(overlay);
  const target = focusable[fromEnd ? focusable.length - 1 : 0];
  if (target) return focusElement(target);
  if (!overlay.hasAttribute("tabindex")) overlay.setAttribute("tabindex", "-1");
  overlay.focus({ preventScroll: true });
  return document.activeElement === overlay;
}

function applyOverlayState() {
  const topId = topOverlayId();
  const modalOpen = Boolean(topId);

  // The Studio workspace overlay is opened *by* the top nav and mirrors that nav
  // in its own sidebar, so it is a workspace, not a dialog. Inerting the whole
  // .app for it buried the nav under the scrim: every item stayed visible and
  // painted "active" while doing nothing, and switching workspaces silently
  // required closing the modal first. Keep the topbar live for that one overlay
  // and inert only the canvas shell behind it; real dialogs still take the lot.
  const workspaceOverlay = topId === "studioOverlay";
  $$("#runtimeDrawer").forEach(element => element.toggleAttribute("inert", modalOpen));
  $$(".app").forEach(element => element.toggleAttribute("inert", modalOpen && !workspaceOverlay));
  $$(".shell").forEach(element => element.toggleAttribute("inert", modalOpen && workspaceOverlay));
  $$(".overlay").forEach(overlay => {
    const open = overlay.classList.contains("open");
    const top = open && overlay.id === topId;
    overlay.toggleAttribute("inert", open && !top);
    if (open && !top) overlay.setAttribute("aria-hidden", "true");
    else overlay.removeAttribute("aria-hidden");
    overlay.style.zIndex = open
      ? String(OVERLAY_Z_INDEX + overlayOrder.indexOf(overlay.id))
      : "";
  });
  // The Studio workspace overlay is opened *by* the top nav and mirrors it in
  // its own sidebar, so it is a workspace, not a dialog. Letting its scrim
  // cover the topbar left every nav item visible, still painted "active", and
  // completely dead: moving from Models to Data meant closing the modal first
  // with no hint that was required. Keep the topbar above this one overlay.
  document.body.classList.toggle("workspace-overlay", workspaceOverlay);
}

function trapOverlayFocus(event) {
  if (event.key !== "Tab" || event.altKey || event.ctrlKey || event.metaKey) return;
  const overlay = $(`#${topOverlayId()}`);
  if (!overlay) return;

  const focusable = focusableElements(overlay);
  if (!focusable.length) {
    event.preventDefault();
    focusInsideOverlay(overlay);
    return;
  }

  const activeIndex = focusable.indexOf(document.activeElement);
  const atStart = activeIndex <= 0;
  const atEnd = activeIndex === focusable.length - 1;
  if (activeIndex === -1 || (event.shiftKey && atStart) || (!event.shiftKey && atEnd)) {
    event.preventDefault();
    focusElement(focusable[event.shiftKey ? focusable.length - 1 : 0]);
  }
}

export function watchOverlayOrder() {
  if (overlayObserver) return;
  const overlays = $$(".overlay");

  overlays.forEach(overlay => {
    if (overlay.classList.contains("open")) overlayOrder.push(overlay.id);
  });
  applyOverlayState();
  focusInsideOverlay($(`#${topOverlayId()}`));

  overlayObserver = new MutationObserver(records => {
    const previousTopId = topOverlayId();
    const opened = [];
    const closed = [];

    records.forEach(record => {
      const overlay = record.target;
      const at = overlayOrder.indexOf(overlay.id);
      const open = overlay.classList.contains("open");
      if (open && at === -1) {
        const active = document.activeElement;
        overlayFocusOrigins.set(
          overlay,
          active instanceof HTMLElement && !overlay.contains(active) ? active : null
        );
        overlayOrder.push(overlay.id);
        opened.push({ overlay, alreadyFocused: overlay.contains(active) });
      } else if (!open && at !== -1) {
        overlayOrder.splice(at, 1);
        closed.push(overlay);
      }
    });

    applyOverlayState();
    const nextTopId = topOverlayId();
    if (nextTopId !== previousTopId) {
      const nextTop = $(`#${nextTopId}`);
      const newlyOpenedTop = opened.find(item => item.overlay.id === nextTopId);
      if (newlyOpenedTop) {
        if (!newlyOpenedTop.alreadyFocused || !canReceiveFocus(document.activeElement)) {
          focusInsideOverlay(nextTop);
        }
      } else {
        const closedTop = closed.find(overlay => overlay.id === previousTopId);
        const origin = closedTop && overlayFocusOrigins.get(closedTop);
        // Some closing actions intentionally move focus somewhere more useful
        // before the MutationObserver runs (Studio's "Open block library" sends
        // it straight to block search). Do not overwrite that explicit choice
        // with the generic return-to-trigger behaviour. A normal close button
        // still has focus inside the now-closed overlay, so it continues to
        // return to its origin as before.
        const active = document.activeElement;
        const explicitlyFocusedOutside = Boolean(
          closedTop
          && active instanceof HTMLElement
          && !closedTop.contains(active)
          && canReceiveFocus(active)
        );
        if (
          !explicitlyFocusedOutside
          && !focusElement(origin)
          && !focusInsideOverlay(nextTop)
        ) {
          // The first-run welcome card opens without a trigger, so it has no
          // origin to restore. Do not leave focus on its now-hidden button.
          focusElement($("#brandHome"));
        }
      }
    }
    closed.forEach(overlay => overlayFocusOrigins.delete(overlay));
  });

  overlays.forEach(overlay => overlayObserver.observe(overlay, {
    attributes: true,
    attributeFilter: ["class"]
  }));
  if (!overlayKeyboardBound) {
    document.addEventListener("keydown", trapOverlayFocus, true);
    overlayKeyboardBound = true;
  }
}

export function topOverlayId() {
  return overlayOrder[overlayOrder.length - 1] || null;
}

export function closeOverlay(id) {
  $(`#${id}`)?.classList.remove("open");
  if (id === "configOverlay") state.configNode = null;
  if (id === "studioOverlay") {
    $$(".nav-item").forEach(item => {
      const active = item.dataset.section === "pipeline";
      item.classList.toggle("active", active);
      if (active) item.setAttribute("aria-current", "page");
      else item.removeAttribute("aria-current");
    });
  }
}
