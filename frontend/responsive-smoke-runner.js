const { chromium } = require(process.argv[2] || "playwright");

const studioUrl = process.argv[3] || "http://127.0.0.1:8080/index.html";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const context = await browser.newContext({ viewport: { width: 600, height: 800 } });
  const page = await context.newPage();
  await page.addInitScript(() => localStorage.setItem("ai-cae4all.studio.welcomed.v1", "1"));
  try {
    await page.goto(studioUrl);
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));
    const initial = await page.evaluate(() => ({
      viewport: innerWidth,
      documentWidth: document.documentElement.scrollWidth,
      topRight: document.querySelector(".top-actions").getBoundingClientRect().right,
      paletteRight: document.querySelector(".palette").getBoundingClientRect().right
    }));
    assert(initial.documentWidth <= initial.viewport, `Compact layout is horizontally clipped (${initial.documentWidth} > ${initial.viewport})`);
    assert(initial.topRight <= initial.viewport, "Top actions extend beyond the compact viewport");
    assert(initial.paletteRight <= initial.viewport, "Block-library drawer extends beyond the compact viewport");

    await page.locator("#hideLibrary").click();
    await page.waitForFunction(() => document.querySelector("#studioShell")?.classList.contains("library-collapsed"));
    const workspace = await page.locator(".workspace").boundingBox();
    assert(workspace && Math.round(workspace.width) === 600, `Collapsed compact canvas is not full width: ${workspace?.width}`);

    await page.locator("#shortcutsTop").click();
    const overlay = await page.locator("#shortcutsOverlay").boundingBox();
    assert(overlay && overlay.x === 0 && overlay.y === 0 && overlay.width === 600 && overlay.height === 800, "Compact modal does not cover the viewport");
    await page.locator('[data-close="shortcutsOverlay"]').click();

    await page.locator('.topnav [data-section="system"]').click();
    await page.waitForSelector("#llmScheme");
    const studioShell = await page.locator(".studio-shell").boundingBox();
    assert(studioShell && studioShell.x >= 0 && studioShell.x + studioShell.width <= 600, "System workspace escapes the compact viewport");
    assert(await page.locator("#llmScheme").count() === 1, "System workspace lost LLM transport controls on compact screens");
    await page.locator('[data-close="studioOverlay"]').click();

    const configPage = await context.newPage();
    await configPage.addInitScript(() => localStorage.setItem("ai-cae4all.studio.welcomed.v1", "1"));
    await configPage.goto(`${studioUrl}${studioUrl.includes("?") ? "&" : "?"}review=config`);
    await configPage.waitForSelector("#configOverlay.open");
    const configLayout = await configPage.evaluate(() => {
      const shell = document.querySelector(".config-shell").getBoundingClientRect();
      const body = document.querySelector(".config-body");
      return {
        shellLeft: shell.left,
        shellRight: shell.right,
        display: getComputedStyle(body).display,
        scrollable: body.scrollHeight > body.clientHeight
      };
    });
    assert(configLayout.shellLeft >= 0 && configLayout.shellRight <= 600, "Config editor escapes the compact viewport");
    assert(configLayout.display === "block" && configLayout.scrollable, "Compact config editor did not become a scrollable stacked layout");
    await configPage.locator(".raw-panel").scrollIntoViewIfNeeded();
    assert(await configPage.locator("#configRaw").isVisible(), "Raw configuration editor is unreachable on a compact screen");
    await configPage.close();

    console.log("PASS: 600px pipeline, navigation, modal, System, and full-config layouts remain reachable");
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
