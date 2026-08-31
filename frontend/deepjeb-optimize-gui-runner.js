// Drives the real Studio GUI (no shortcuts) through the DeepJEB closed-loop
// design optimization: a config-holding SDFFlow block wired into a CAD
// Generator block running `mode optimize`, submitted with the same
// destructive-execution confirmation any user click goes through, then polled
// to completion against the actual native launcher.
//
// Usage: node deepjeb-optimize-gui-runner.js [path-to-playwright] [studio-url]
const { chromium } = require(process.argv[2] || "playwright");
const fs = require("fs");
const path = require("path");

const studioUrl = process.argv[3] || "http://127.0.0.1:8200/index.html";
const vaePath = "../output/geometry_generation/ex1/sdfflow_vae.pth";
const fmPath = "../output/geometry_generation/ex1/sdfflow_fm.pth";
const outputDir = `../output/geometry_generation/ex1/optimization_gui_${Date.now()}`;
const surrogateCheckpoint = "../output/deepjeb_himgn/deepjeb_himgn.pth";
const surrogateConfig = "../configs/cHI-MGNflow/deepjeb/config_infer.txt";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function setInspectorControl(page, key, value) {
  const control = page.locator(`.inspector-config[data-key="${key}"]`);
  assert(await control.count() === 1, `Inspector is missing field "${key}" (not in this node's visible config)`);
  const tag = await control.evaluate(element => element.tagName);
  if (tag === "SELECT") await control.selectOption(value);
  else {
    await control.fill(value);
    await control.dispatchEvent("change");
  }
}

async function setFullConfigControl(page, nodeId, key, value) {
  await page.locator(`[data-node-id="${nodeId}"] .node-head`).click();
  await page.locator("#openFullConfig").click();
  await page.waitForSelector("#configOverlay.open");
  await page.locator("#configSearch").fill(key);
  const control = page.locator(`.full-config-control[data-key="${key}"]`);
  await control.waitFor({ timeout: 10_000 });
  const tag = await control.evaluate(element => element.tagName);
  if (tag === "SELECT") await control.selectOption(value);
  else {
    await control.fill(value);
    await control.dispatchEvent("change");
  }
  await page.locator("#saveConfig").click();
  await page.waitForFunction(() => !document.querySelector("#configOverlay")?.classList.contains("open"));
}

async function jobDetail(page, jobId) {
  const response = await page.request.get(`${new URL(studioUrl).origin}/api/jobs/${encodeURIComponent(jobId)}`);
  assert(response.ok(), `Job status request failed with HTTP ${response.status()}`);
  return response.json();
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const context = await browser.newContext({ viewport: { width: 1680, height: 1000 } });
  const page = await context.newPage();
  await page.addInitScript(() => {
    localStorage.setItem("ai-cae4all.studio.welcomed.v1", "1");
  });

  const browserErrors = [];
  const confirmations = [];
  page.on("pageerror", error => browserErrors.push(`pageerror: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error") browserErrors.push(`console: ${message.text()}`);
  });
  page.on("dialog", async dialog => {
    confirmations.push(dialog.message());
    await dialog.accept();
  });

  let submittedJob = null;
  try {
    await page.goto(studioUrl);
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));
    await page.screenshot({ path: path.join(__dirname, "runtime", "deepjeb-gui-01-loaded.png") });

    // ---- Build the graph the way a person clicking through the palette would ----
    await page.locator("#templateSelect").selectOption("blank");
    await page.locator('.palette-item[data-block-type="model.sdfflow"]').click();
    await page.locator('.palette-item[data-block-type="run.cad_generator"]').click();
    const ids = await page.evaluate(() => Object.fromEntries(
      window.__AI_CAE_FRONTEND__.state.nodes.map(node => [node.type, node.id])
    ));
    assert(ids["model.sdfflow"] && ids["run.cad_generator"], "Palette did not add both blocks");

    await page.locator(`[data-node="${ids["model.sdfflow"]}"][data-port="model"][data-direction="output"]`).click();
    await page.locator(`[data-node="${ids["run.cad_generator"]}"][data-port="model"][data-direction="input"]`).click();
    const wired = await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.edges.length);
    assert(wired === 1, `Expected exactly one edge after wiring, found ${wired}`);
    await page.screenshot({ path: path.join(__dirname, "runtime", "deepjeb-gui-02-wired.png") });

    // ---- Configure the SDFFlow config-holder: the trained ex1 checkpoints ----
    // and a fast surrogate-backed optimize run (see design_loop/surrogate.py's
    // accuracy notice -- this proves the AI-replaces-FEA path executes end to
    // end through the browser, not that its numbers are yet trustworthy).
    await page.locator(`[data-node-id="${ids["model.sdfflow"]}"] .node-head`).click();
    await setInspectorControl(page, "mode", "optimize");
    await setFullConfigControl(page, ids["model.sdfflow"], "vae_modelpath", vaePath);
    await setFullConfigControl(page, ids["model.sdfflow"], "fm_modelpath", fmPath);
    await setFullConfigControl(page, ids["model.sdfflow"], "output_dir", outputDir);
    // Small budget: this run demonstrates the GUI mechanism, not a production
    // search -- the 214-evaluation FEA search and the 34-evaluation surrogate
    // search were already run and reported from the CLI.
    await setFullConfigControl(page, ids["model.sdfflow"], "opt_baseline_size", "6");
    await setFullConfigControl(page, ids["model.sdfflow"], "opt_budget", "12");
    await setFullConfigControl(page, ids["model.sdfflow"], "opt_popsize", "4");

    // ---- Configure the CAD Generator: run this SDFFlow config in optimize mode ----
    await page.locator(`[data-node-id="${ids["run.cad_generator"]}"] .node-head`).click();
    await setInspectorControl(page, "mode", "optimize");
    await setInspectorControl(page, "opt_analysis", "surrogate");
    const missingSurrogateIssues = await page.evaluate(async () => {
      const { validateGraph } = await import("./src/validate.js");
      return validateGraph(false);
    });
    assert(missingSurrogateIssues.some(issue => issue.includes("opt_surrogate_checkpoint")
      && issue.includes("opt_surrogate_config")),
    `Surrogate selection did not explain its missing files: ${missingSurrogateIssues.join(" | ")}`);

    // Supply the conditionally required pair through the visible Full config
    // sheet, then prove the same graph-level guard clears before submission.
    await page.locator(`[data-node-id="${ids["model.sdfflow"]}"] .node-head`).click();
    await setFullConfigControl(page, ids["model.sdfflow"], "opt_surrogate_checkpoint", surrogateCheckpoint);
    await setFullConfigControl(page, ids["model.sdfflow"], "opt_surrogate_config", surrogateConfig);
    await page.locator(`[data-node-id="${ids["run.cad_generator"]}"] .node-head`).click();
    const readyIssues = await page.evaluate(async () => {
      const { validateGraph } = await import("./src/validate.js");
      return validateGraph(false);
    });
    assert(readyIssues.length === 0, `Configured DeepJEB graph still has issues: ${readyIssues.join(" | ")}`);
    await page.screenshot({ path: path.join(__dirname, "runtime", "deepjeb-gui-03-configured.png") });

    // ---- Confirm the exact native config the GUI will launch ----
    const serialized = await page.evaluate(async id => {
      const { executableSteps } = await import("./src/validate.js");
      return executableSteps(id)[0] || null;
    }, ids["run.cad_generator"]);
    assert(serialized, "The CAD Generator block produced no executable step");
    assert(serialized.label.includes("SDFFlow"), `Wrong native route label: ${serialized.label}`);
    // Parse key/value pairs rather than matching whole padded lines -- the
    // padding width is a config.js implementation detail, not something worth
    // this script being brittle against.
    const parsed = Object.fromEntries(
      serialized.config.split(/\r?\n/)
        .map(line => line.match(/^(\S+)\s+(.+)$/))
        .filter(Boolean)
        .map(match => [match[1], match[2]])
    );
    const expected = {
      model: "sdfflow", mode: "optimize", vae_modelpath: vaePath, fm_modelpath: fmPath,
      output_dir: outputDir, opt_analysis: "surrogate",
      opt_surrogate_checkpoint: surrogateCheckpoint, opt_surrogate_config: surrogateConfig,
      opt_baseline_size: "6", opt_budget: "12", opt_popsize: "4",
    };
    for (const [key, value] of Object.entries(expected)) {
      assert(parsed[key] === value,
        `Native config key "${key}" is ${JSON.stringify(parsed[key])}, expected ${JSON.stringify(value)}\n--- actual config ---\n${serialized.config}`);
    }
    console.log("Native config verified before submission:\n" + serialized.config);

    // ---- Run it for real, through the same button and confirmation a user sees ----
    const responsePromise = page.waitForResponse(response =>
      response.url().endsWith("/api/pipeline/run") && response.request().method() === "POST",
      { timeout: 60_000 });
    await page.locator("#inspectorRun").click();
    const response = await responsePromise;
    const submitted = await response.json();
    assert(response.status() === 201, `Pipeline submission returned HTTP ${response.status()}: ${JSON.stringify(submitted)}`);
    submittedJob = submitted.id;
    assert(submitted.steps?.length === 1, "Expected exactly one native step");
    assert(submitted.steps[0].route?.model === "sdfflow", `Wrong resolved route: ${JSON.stringify(submitted.steps[0].route)}`);
    assert(confirmations.some(message => message.includes("Execute the real AI-CAE4ALL launcher")),
      "The destructive execution confirmation was not shown");
    await page.screenshot({ path: path.join(__dirname, "runtime", "deepjeb-gui-04-running.png") });

    let job = submitted;
    const deadline = Date.now() + 12 * 60_000;
    while (["queued", "running"].includes(job.status) && Date.now() < deadline) {
      await page.waitForTimeout(3_000);
      job = await jobDetail(page, submitted.id);
    }
    assert(!["queued", "running"].includes(job.status), `Run did not finish within 12 minutes (job ${job.id})`);
    assert(job.status === "completed", `Run ${job.status}:\n${job.log?.slice(-4000) || "(no log)"}`);
    assert(job.returncode === 0, `Native process returned ${job.returncode}`);
    assert(job.log.includes("Wrote results to"), "The optimize run did not reach its own completion banner");
    assert(job.log.includes("HI-MGN surrogate"), "The run did not select the surrogate analysis backend");
    const actualOutputDir = path.resolve(__dirname, outputDir);
    const summary = JSON.parse(fs.readFileSync(path.join(actualOutputDir, "summary.json"), "utf8"));
    assert(summary.analysis_backend === "surrogate", "Summary lost the selected surrogate backend");
    assert(!Object.hasOwn(summary, "mesh_sensitivity"), "Surrogate summary invented FEA mesh-sensitivity evidence");
    const verifiedRows = [summary.verified?.baseline, summary.verified?.typical, summary.verified?.optimized].filter(Boolean);
    assert(verifiedRows.length === 3 && verifiedRows.every(row => Number(row.num_nodes) > 0 && !Object.hasOwn(row, "num_tets")),
      `Surrogate summary did not preserve real graph-node counts: ${JSON.stringify(verifiedRows)}`);
    const summaryHeader = fs.readFileSync(path.join(actualOutputDir, "optimize_summary.csv"), "utf8").split(/\r?\n/, 1)[0].split(",");
    assert(summaryHeader.includes("num_nodes") && !summaryHeader.includes("num_tets"),
      `GUI result table exposes the wrong mesh cardinality: ${summaryHeader.join(",")}`);

    await page.reload();
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));
    await page.locator(`[data-node-id="${ids["run.cad_generator"]}"] .node-head`).click();
    await page.screenshot({ path: path.join(__dirname, "runtime", "deepjeb-gui-05-completed.png") });

    console.log("\n=== PASS: DeepJEB closed-loop optimization ran end to end through the Studio GUI ===");
    console.log(`Job ${job.id} | results: ${job.steps?.[0]?.results} | samples: ${job.steps?.[0]?.results_samples}`);
    console.log(`Output dir: ${outputDir}`);
  } catch (error) {
    console.error("\n=== FAIL ===");
    console.error(error);
    await page.screenshot({ path: path.join(__dirname, "runtime", "deepjeb-gui-FAILURE.png") }).catch(() => {});
    if (submittedJob) {
      try {
        const job = await jobDetail(page, submittedJob);
        console.error(`Job ${submittedJob} status=${job.status} returncode=${job.returncode}`);
        console.error((job.log || "").slice(-4000));
      } catch {
        // best effort
      }
    }
    if (browserErrors.length) console.error("Browser errors:\n" + browserErrors.join("\n"));
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
