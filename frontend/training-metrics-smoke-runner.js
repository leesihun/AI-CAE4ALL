const { chromium } = require(process.argv[2] || "playwright");
const path = require("path");

const studioUrl = process.argv[3] || "http://127.0.0.1:8081/index.html";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function replaceTemplate(page, name) {
  const option = page.locator(`#templateSelect option[value="${name}"]`);
  const expectedLabel = (await option.textContent())?.trim();
  const dialogPromise = page.waitForEvent("dialog");
  const selectionPromise = page.locator("#templateSelect").selectOption(name);
  const dialog = await dialogPromise;
  const message = dialog.message();
  try {
    assert(dialog.type() === "confirm", `Template replacement opened an unexpected ${dialog.type()} dialog`);
    assert(
      expectedLabel && message.includes(`"${expectedLabel}"`) && message.includes("Undo step"),
      `Unexpected template replacement confirmation: ${message}`
    );
  } catch (error) {
    await dialog.dismiss();
    await selectionPromise.catch(() => {});
    throw error;
  }
  await dialog.accept();
  await selectionPromise;
}

const metric = (key, label, values) => ({
  key,
  label,
  event: "epoch",
  points: values.map((y, x) => ({ x, y, line: x + 1, event: "epoch" })),
  count: values.length,
  min: Math.min(...values),
  max: Math.max(...values),
  last: values.at(-1)
});

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 960 } });
  // The studio shows a one-time orientation card to first-run users, and every
  // smoke run starts from a clean browser profile -- so without this it would
  // meet that modal on every launch. Seed the "already welcomed" flag instead of
  // clicking the card away, so the runs exercise the studio a returning user sees.
  await page.addInitScript(() => {
    try { localStorage.setItem("ai-cae4all.studio.welcomed.v1", "1"); } catch { /* storage blocked */ }
  });

  const errors = [];
  page.on("pageerror", error => errors.push(`pageerror: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });

  await page.route("**/api/training-metrics", route => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      count: 2,
      source: "Studio job logs",
      items: [{
        job_id: "metric-smoke",
        label: "SimulGen metric smoke",
        status: "completed",
        models: ["simulgenvae"],
        log_path: "frontend/runtime/jobs/metric-smoke/run.log",
        x_label: "epoch",
        metrics: [
          metric("recon", "Recon", [.62, .48, .39]),
          metric("kl", "KL", [.44, .31, .27]),
          metric("lr", "LR", [.001, .0008, .0006])
        ],
        metric_count: 3,
        point_count: 9
      }, {
        job_id: "metric-smoke-b",
        label: "SimulGen metric smoke B",
        status: "completed",
        models: ["simulgenvae"],
        node_ids: ["simulgen_b"],
        lineage: [{ node_id: "simulgen_b", node_type: "model.simulgenvae", model_id: "simulgenvae", mode: "train_vae" }],
        log_path: "frontend/runtime/jobs/metric-smoke-b/run.log",
        x_label: "epoch",
        metrics: [
          metric("recon", "Recon", [.71, .55, .43]),
          metric("kl", "KL", [.51, .37, .3]),
          metric("lr", "LR", [.001, .00075, .0005])
        ],
        metric_count: 3,
        point_count: 9
      }]
    })
  }));
  await page.route(/\/api\/files\?kind=artifact$/, route => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({ items: [{ path: "frontend/runtime/evaluation/schema-smoke/per_sample_metrics.csv", extension: ".csv" }] })
  }));
  await page.route("**/api/comparison/schema", route => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      sources: [{ path: "frontend/runtime/evaluation/schema-smoke/per_sample_metrics.csv", columns: ["prediction_file", "relative_l2", "mae"], rows_sampled: 8 }],
      common_columns: ["prediction_file", "relative_l2", "mae"],
      numeric_columns: ["relative_l2", "mae"],
      group_columns: ["prediction_file"]
    })
  }));

  try {
    await page.goto(studioUrl);
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));
    assert(await page.locator('[data-node-id="train_metrics"]').count() === 1, "Default template is missing Train Metrics");
    // Match the trainer by role, not by node id: the default template is HI-MGN
    // (node id "trainer") rather than the SimulGen one this used to hard-code.
    const edge = await page.evaluate(() => {
      const app = window.__AI_CAE_FRONTEND__;
      return app.state.edges.find(item =>
        item.fromPort === "metrics" && item.toNode === "train_metrics"
        && app.BLOCK_SPECS[app.state.nodes.find(node => node.id === item.fromNode)?.type || ""]?.isModel);
    });
    assert(Boolean(edge), "Model training metrics are not connected to the Train Metrics block");

    await page.locator('[data-node-id="train_metrics"] .node-preview').click();
    await page.waitForSelector(".training-metric-option");
    assert(await page.locator("#studioOverlay").evaluate(element => element.classList.contains("open")), "Metrics workspace did not open");
    assert(!(await page.locator("#artifactOverlay").evaluate(element => element.classList.contains("open"))), "Train Metrics incorrectly opened the dataset viewer");
    assert(await page.locator(".training-metric-option").count() === 3, "Every discovered metric was not listed");
    assert(await page.locator(".training-metric-option input:checked").count() === 3, "All discovered metrics must be plotted by default");
    assert(await page.locator("[data-metric-plot]").count() === 3, "All default metric plots are missing");

    await page.locator('[data-metric-toggle="kl"]').uncheck();
    assert(await page.locator("[data-metric-plot]").count() === 2, "Excluding one metric did not remove exactly one plot");
    assert(await page.locator('[data-metric-plot="kl"]').count() === 0, "Excluded KL plot is still visible");
    const excluded = await page.evaluate(() =>
      window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.id === "train_metrics").config.excluded_metrics
    );
    assert(excluded.split(",").includes("kl"), "Excluded metric selection was not persisted on the block");

    await page.locator("#metricsSelectNone").click();
    assert(await page.locator("[data-metric-plot]").count() === 0, "Plot none did not clear all plots");
    assert((await page.locator(".training-no-plots").innerText()).includes("No metrics selected"), "Empty selection guidance is missing");

    await page.locator("#metricsSelectAll").click();
    assert(await page.locator("[data-metric-plot]").count() === 3, "Plot all did not restore every metric");
    assert(await page.locator(".training-metric-svg polyline").count() === 3, "Metric series were not drawn as real SVG polylines");
    await page.locator("#metricsSmoothing").fill("0.8");
    assert(await page.locator("#metricsSmoothing").inputValue() === "0.8", "Smoothing control did not retain its value");
    assert(await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.id === "train_metrics").config.smoothing) === "0.8", "Smoothing setting was not persisted on the block");
    await page.screenshot({ path: path.join(__dirname, "runtime", "training-metrics.png"), fullPage: false });

    await page.locator('[data-close="studioOverlay"]').click();
    await replaceTemplate(page, "blank");
    await page.locator('.palette-item[data-block-type="evaluate.training_metrics"]').click();
    await page.locator('.palette-item[data-block-type="evaluate.training_metrics"]').click();
    await page.locator('.palette-item[data-block-type="evaluate.compare"]').click();
    const nodes = await page.evaluate(() => ({
      metrics: window.__AI_CAE_FRONTEND__.state.nodes.filter(node => node.type === "evaluate.training_metrics").map(node => node.id),
      compare: window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.type === "evaluate.compare")?.id
    }));
    assert(nodes.metrics.length === 2 && nodes.compare, "Could not create the connected-run comparison fixture");

    await page.locator(`[data-node="${nodes.metrics[1]}"][data-port="metrics"][data-direction="input"]`).click();
    await page.locator(`[data-node="${nodes.metrics[0]}"][data-port="metrics"][data-direction="output"]`).click();
    assert(await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.edges.length) === 1, "Initial metrics link was not created");
    await page.locator(`[data-node="${nodes.metrics[0]}"][data-port="metrics"][data-direction="input"]`).click();
    await page.locator(`[data-node="${nodes.metrics[1]}"][data-port="metrics"][data-direction="output"]`).click();
    assert(await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.edges.length) === 1, "Cycle-producing link was not rejected before graph mutation");
    assert(await page.locator('.toast[role="alert"]').count() >= 1, "Rejected connection was not exposed as an accessible alert");

    await page.evaluate(({ metrics }) => {
      const stateNodes = window.__AI_CAE_FRONTEND__.state.nodes;
      stateNodes.find(node => node.id === metrics[0]).config.job_id = "metric-smoke";
      stateNodes.find(node => node.id === metrics[1]).config.job_id = "metric-smoke-b";
    }, nodes);
    for (const metricsId of nodes.metrics) {
      await page.locator(`[data-node="${nodes.compare}"][data-port="metrics"][data-direction="input"]`).click();
      await page.locator(`[data-node="${metricsId}"][data-port="metrics"][data-direction="output"]`).click();
    }
    assert(await page.evaluate(id => window.__AI_CAE_FRONTEND__.state.edges.filter(edge => edge.toNode === id).length, nodes.compare) === 2, "Compare Models did not retain both connected metric runs");

    await page.locator(`[data-node-id="${nodes.compare}"] .node-preview`).click();
    await page.waitForSelector(".connected-run-cards article");
    assert(await page.locator(".connected-run-cards article").count() === 2, "Connected comparison did not resolve both graph runs");
    assert(await page.locator(".connected-run-chart polyline").count() === 2, "Connected runs were not overlaid in the comparison plot");
    assert(!(await page.locator("#artifactOverlay").evaluate(element => element.classList.contains("open"))), "Compare Models incorrectly opened the artifact viewer");
    await page.locator("[data-run-select]").selectOption("frontend/runtime/evaluation/schema-smoke/per_sample_metrics.csv");
    await page.waitForFunction(() => document.querySelector("#comparisonMetric")?.value === "relative_l2");
    assert(await page.locator("#comparisonGroup").inputValue() === "prediction_file", "CSV schema did not suggest the real group column");
    assert(await page.locator("#runComparison").isEnabled(), "CSV ranking stayed disabled after a numeric metric was detected");
    assert((await page.locator("#comparisonSchemaStatus").innerText()).includes("2 numeric metrics"), "CSV schema evidence is not visible");
    await page.screenshot({ path: path.join(__dirname, "runtime", "connected-run-comparison.png"), fullPage: false });
    await page.locator('[data-close="studioOverlay"]').click();
    await page.locator('.palette-item[data-block-type="evaluate.predictions"]').click();
    const evaluationId = await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.type === "evaluate.predictions")?.id);
    await page.locator(`[data-node-id="${evaluationId}"] .node-preview`).click();
    await page.waitForSelector("#evaluationPrediction");
    assert(!(await page.locator("#artifactOverlay").evaluate(element => element.classList.contains("open"))), "Evaluate Predictions incorrectly opened the artifact viewer");
    assert(await page.locator('#savedState[role="status"]').count() === 1, "Save status is not exposed as an accessible status message");

    assert(errors.length === 0, `Browser reported errors: ${errors.join(" | ")}`);
    console.log("PASS: metrics selection/smoothing, cycle prevention, and graph-connected multi-run comparison");
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(`FAIL: ${error.message}`);
  process.exitCode = 1;
});
