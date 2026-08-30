const playwrightModule = process.argv[2] || "playwright";
const { chromium } = require(playwrightModule);

const studioUrl = process.argv[3] || "http://127.0.0.1:8097/index.html";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.addInitScript(() => {
    try { localStorage.setItem("ai-cae4all.studio.welcomed.v1", "1"); } catch { /* storage blocked */ }
  });
  const browserErrors = [];
  page.on("pageerror", error => browserErrors.push(`pageerror: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error") browserErrors.push(`console: ${message.text()}`);
  });

  try {
    await page.goto(studioUrl);
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__?.state.api.connected === true);
    await page.locator('[data-section="system"]').click();
    await page.waitForSelector("#runConfigAudit");
    assert(await page.locator("#llmPassword").getAttribute("type") === "password", "LLM session secret is not a password control");
    assert(await page.locator("#llmPassword").inputValue() === "", "A stored LLM password was reflected into the GUI");

    await page.locator("#runConfigAudit").click();
    await page.waitForFunction(() => {
      const button = document.querySelector("#runConfigAudit");
      return button && !button.disabled && document.querySelector("#auditResults .live-summary");
    }, null, { timeout: 30000 });
    const summary = await page.locator("#auditResults .live-summary strong").allTextContents();
    const files = Number(summary[0]);
    const failing = Number(summary[2]);
    assert(files > 0, "Configuration audit scanned no configs");
    assert(await page.locator("#auditFileList article.live-row").count() === Math.min(files, 200), "Initial audit page is not bounded");
    assert((await page.locator("#auditCatalogStatus").innerText()).includes(`${files} were audited in total`), "Audit catalog status is incomplete");

    if (files > 200) {
      await page.locator("#auditShowMore").click();
      assert(await page.locator("#auditFileList article.live-row").count() === Math.min(files, 400), "Audit show-more did not reveal the next page");
    }

    await page.locator("#auditSearch").fill("chi-mgnflow");
    const chiRows = page.locator("#auditFileList article.live-row");
    assert(await chiRows.count() > 0, "cHI-MGNflow configs are missing from the repository audit");
    for (const text of await chiRows.allTextContents()) {
      assert(text.toLowerCase().includes("chi-mgnflow"), `Audit search leaked a nonmatching row: ${text}`);
    }

    await page.locator("#auditSearch").fill("");
    await page.locator("#auditFailuresOnly").check();
    const failureRows = page.locator("#auditFileList article.live-row");
    assert(await failureRows.count() === Math.min(failing, 200), "Failures-only filter disagrees with the authoritative audit summary");
    if (failing) {
      for (const text of await failureRows.allTextContents()) {
        assert(text.includes("FAIL"), `Failures-only filter showed a passing config: ${text}`);
      }
      await page.locator("[data-audit-detail]").first().click();
      assert(await page.locator("#auditDetail .diagnostic").count() > 0, "Failing config did not expose its diagnostics");
    } else {
      assert((await page.locator("#auditFileList").innerText()).includes("No audited configurations match"), "Empty audit filter has no explanation");
    }
    await page.locator("#auditFailuresOnly").uncheck();

    const detail = page.locator("[data-audit-detail]").first();
    if (await detail.count()) {
      await detail.click();
      assert(await page.locator("#auditDetail .diagnostic").count() > 0, "Audit Diagnostics did not reveal the real messages");
    }
    assert(browserErrors.length === 0, `Browser errors: ${browserErrors.join(" | ")}`);
    console.log(`PASS: System audit searched and paged ${files} real configs and accurately rendered ${failing} structural failures`);
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack || error.message);
  process.exit(1);
});
