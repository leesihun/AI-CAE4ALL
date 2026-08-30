const { chromium } = require(process.argv[2] || "playwright");

const studioUrl = process.argv[3] || "http://127.0.0.1:8081/index.html?welcome=0";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1366, height: 768 } });
  const errors = [];
  page.on("pageerror", error => errors.push(`pageerror: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });

  try {
    await page.addInitScript(() => {
      try { localStorage.setItem("ai-cae4all.studio.welcomed.v1", "1"); } catch { /* storage blocked */ }
    });
    await page.goto(studioUrl);
    await page.waitForSelector("#shortcutsTop");

    // Keep a real background control surface present while exercising the
    // modal. Its visual and DOM state should both yield to the dialog.
    await page.locator("#runtimeDrawer").evaluate(element => element.classList.add("open"));
    await page.locator("#shortcutsTop").click();
    await page.waitForFunction(() => {
      const overlay = document.querySelector("#shortcutsOverlay");
      return overlay?.classList.contains("open") && overlay.contains(document.activeElement);
    });

    const initial = await page.evaluate(() => {
      const overlay = document.querySelector("#shortcutsOverlay");
      const topbar = document.querySelector(".topbar");
      const drawer = document.querySelector("#runtimeDrawer");
      const app = document.querySelector(".app");
      const overlayRect = overlay.getBoundingClientRect();
      const topbarRect = topbar.getBoundingClientRect();
      const trigger = document.querySelector("#validateTop");
      trigger.focus();
      const backgroundFocusBlocked = document.activeElement !== trigger;
      return {
        appInert: app.inert,
        drawerInert: drawer.inert,
        overlayInert: overlay.inert,
        activeClose: document.activeElement?.matches('[data-close="shortcutsOverlay"]') || false,
        backgroundFocusBlocked,
        coversTopbar: overlayRect.top <= topbarRect.top
          && overlayRect.bottom >= topbarRect.bottom
          && overlayRect.left <= topbarRect.left
          && overlayRect.right >= topbarRect.right,
        overlayZ: Number(getComputedStyle(overlay).zIndex),
        topbarZ: Number(getComputedStyle(topbar).zIndex),
        drawerZ: Number(getComputedStyle(drawer).zIndex)
      };
    });
    assert(initial.appInert, "The background app was not inert");
    assert(initial.drawerInert, "The runtime drawer was not inert");
    assert(!initial.overlayInert, "The top modal was inert");
    assert(initial.activeClose, "Initial focus did not move into the modal");
    assert(initial.backgroundFocusBlocked, "Programmatic focus escaped into the inert app");
    assert(initial.coversTopbar, "The modal overlay did not cover the topbar");
    assert(
      initial.overlayZ > initial.topbarZ && initial.overlayZ > initial.drawerZ,
      `The modal layer was not above background chrome: ${JSON.stringify(initial)}`
    );

    const shortcutsClose = page.locator('[data-close="shortcutsOverlay"]');
    assert(await shortcutsClose.count() === 1, "Expected one shortcuts close button");
    await page.keyboard.press("Tab");
    assert(await shortcutsClose.evaluate(element => document.activeElement === element), "Tab escaped a one-control modal");
    await page.keyboard.press("Shift+Tab");
    assert(await shortcutsClose.evaluate(element => document.activeElement === element), "Shift+Tab escaped a one-control modal");

    // Open a second overlay without closing the first to exercise the real
    // overlay stack rather than only the single-dialog happy path.
    await page.locator("#welcomeOverlay").evaluate(element => element.classList.add("open"));
    await page.waitForFunction(() => {
      const overlay = document.querySelector("#welcomeOverlay");
      return overlay?.classList.contains("open") && overlay.contains(document.activeElement);
    });
    const stacked = await page.evaluate(() => {
      const lower = document.querySelector("#shortcutsOverlay");
      const top = document.querySelector("#welcomeOverlay");
      const lowerClose = lower.querySelector('[data-close="shortcutsOverlay"]');
      lowerClose.focus();
      return {
        lowerInert: lower.inert,
        lowerHidden: lower.getAttribute("aria-hidden"),
        topInert: top.inert,
        topHasFocus: top.contains(document.activeElement),
        lowerFocusBlocked: document.activeElement !== lowerClose,
        topZ: Number(getComputedStyle(top).zIndex),
        lowerZ: Number(getComputedStyle(lower).zIndex)
      };
    });
    assert(stacked.lowerInert, "The lower modal remained interactive");
    assert(stacked.lowerHidden === "true", "The lower modal remained exposed to assistive technology");
    assert(!stacked.topInert && stacked.topHasFocus, "The top modal did not own interaction");
    assert(stacked.lowerFocusBlocked, "Focus moved into the inert lower modal");
    assert(stacked.topZ > stacked.lowerZ, "The newest modal was not painted above the older modal");

    const welcomeTour = page.locator("#welcomeTour");
    const welcomeDismiss = page.locator("#welcomeDismiss");
    assert(await welcomeTour.evaluate(element => document.activeElement === element), "Initial stacked-modal focus missed the first control");
    await page.keyboard.press("Shift+Tab");
    assert(await welcomeDismiss.evaluate(element => document.activeElement === element), "Shift+Tab did not wrap to the final control");
    await page.keyboard.press("Tab");
    assert(await welcomeTour.evaluate(element => document.activeElement === element), "Tab did not wrap to the first control");

    await page.keyboard.press("Escape");
    await page.waitForFunction(() => {
      const welcome = document.querySelector("#welcomeOverlay");
      const close = document.querySelector('[data-close="shortcutsOverlay"]');
      return !welcome.classList.contains("open") && document.activeElement === close;
    });
    assert(!(await page.locator("#shortcutsOverlay").getAttribute("inert")), "The revealed modal retained an inert attribute");

    await page.keyboard.press("Escape");
    await page.waitForFunction(() => {
      const shortcuts = document.querySelector("#shortcutsOverlay");
      return !shortcuts.classList.contains("open")
        && document.activeElement === document.querySelector("#shortcutsTop");
    });
    const restored = await page.evaluate(() => ({
      appInert: document.querySelector(".app").inert,
      drawerInert: document.querySelector("#runtimeDrawer").inert,
      triggerFocused: document.activeElement === document.querySelector("#shortcutsTop")
    }));
    assert(!restored.appInert && !restored.drawerInert, "Background inertness remained after the final modal closed");
    assert(restored.triggerFocused, "Focus was not restored to the original trigger");
    assert(errors.length === 0, `Browser errors: ${errors.join(" | ")}`);

    console.log("Modal accessibility smoke test passed.");
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
