const playwrightModule = process.argv[2] || "playwright";
const { chromium } = require(playwrightModule);

const studioUrl = process.argv[3] || "http://127.0.0.1:8097/index.html";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const schemaFor = predictionPath => ({
  compatible: true,
  errors: [],
  warnings: ["Field names are absent from the prediction file; positional mapping requires confirmation."],
  prediction: {
    contract: "table",
    field_count: 2,
    field_names: ["prediction_0", "prediction_1"],
    target_indices: [0, 1]
  },
  truth: {
    contract: "table",
    field_count: 2,
    field_names: ["lift", "drag"],
    target_indices: [0, 1]
  },
  sample_matching: {
    strategy: "id",
    overlap_count: predictionPath.includes("b.h5") ? 3 : 2,
    matched_ids: ["a", "b"],
    incompatible_shape_count: 0
  },
  recommended_mapping: {
    mode: "selected",
    confidence: "confirm",
    basis: "same width; prediction names unavailable",
    prediction_array: "predictions",
    truth_array: "Y",
    field_pairs: [
      { name: "lift", prediction_index: 0, truth_index: 0, prediction_name: "prediction_0", truth_name: "lift" },
      { name: "drag", prediction_index: 1, truth_index: 1, prediction_name: "prediction_1", truth_name: "drag" }
    ]
  }
});

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.addInitScript(() => {
    try { localStorage.setItem("ai-cae4all.studio.welcomed.v1", "1"); } catch { /* storage blocked */ }
  });
  const browserErrors = [];
  const schemaBodies = [];
  const runBodies = [];
  page.on("pageerror", error => browserErrors.push(`pageerror: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error") browserErrors.push(`console: ${message.text()}`);
  });

  await page.route("**/api/files?kind=artifact", route => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({
      items: [
        { path: "frontend/runtime/prediction.h5", extension: ".h5" },
        { path: "frontend/runtime/prediction-b.h5", extension: ".h5" }
      ],
      matched: 2,
      limit: 250,
      truncated: false
    })
  }));
  await page.route("**/api/files?kind=dataset", route => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({
      items: [
        { path: "dataset/truth.h5", extension: ".h5" },
        { path: "dataset/truth-alt.h5", extension: ".h5" }
      ],
      matched: 2,
      limit: 250,
      truncated: false
    })
  }));
  await page.route("**/api/evaluation/schema", async route => {
    const body = route.request().postDataJSON();
    schemaBodies.push(body);
    if (String(body.prediction_path).includes("prediction-b.h5")) {
      await new Promise(resolve => setTimeout(resolve, 250));
    }
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(schemaFor(body.prediction_path)) });
  });
  await page.route("**/api/evaluation/run", async route => {
    const body = route.request().postDataJSON();
    runBodies.push(body);
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        contract: "table",
        truth_source: "selected",
        evaluated_samples: 2,
        skipped: [],
        field_pairs: body.field_pairs,
        aggregate: {
          relative_l2: { mean: 0.01 },
          mae: { mean: 0.02 },
          rmse: { mean: 0.03 }
        },
        per_sample_csv: "frontend/runtime/evaluation/per_sample.csv",
        report_path: "frontend/runtime/evaluation/report.json"
      })
    });
  });
  await page.route("**/api/docs", route => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ items: [{ name: "GUI guide", path: "frontend/README.md", size: 100, modified: "now" }], matched: 1, limit: 250, truncated: false })
  }));

  try {
    await page.goto(studioUrl);
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__?.state.api.connected === true);
    const nodeId = await page.evaluate(() => {
      const app = window.__AI_CAE_FRONTEND__;
      const node = app.state.nodes.find(item => item.type === "evaluate.predictions");
      if (!node) throw new Error("The default pipeline has no evaluation node");
      Object.assign(node.config, {
        prediction_path: "frontend/runtime/prediction.h5",
        truth_path: "dataset/truth.h5",
        field_pairs: "",
        mapping_mode: "schema",
        mapping_confirmed: "False"
      });
      return node.id;
    });
    await page.evaluate(id => window.__AI_CAE_FRONTEND__.openStudio("evaluation", id), nodeId);
    await page.waitForSelector(".evaluation-field-row");

    const runButton = page.locator("#runFieldEvaluation");
    assert(await runButton.isDisabled(), "A positional field mapping ran without explicit confirmation");
    assert(await page.locator(".evaluation-field-row").count() === 2, "The detected field mapping rows were not rendered");
    assert(await page.locator(".evaluation-contract-summary > span").nth(2).locator("strong").innerText() === "2", "The sample-ID overlap is missing");
    assert((await page.locator("#evaluationSchema").innerText()).includes("id-matched samples"), "The sample matching strategy is missing");

    await page.locator("#evaluationTruth").selectOption("dataset/truth-alt.h5");
    await page.waitForFunction(
      () => document.querySelectorAll(".evaluation-field-row").length === 2
        && !document.querySelector("#evaluationSchema .live-empty")
    );
    assert(schemaBodies.at(-1)?.truth_path === "dataset/truth-alt.h5", "The visible truth selector did not refresh the selected schema");
    const truthSelection = await page.evaluate(id => {
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.id === id);
      return {
        truthPath: node?.config.truth_path,
        fieldPairs: node?.config.field_pairs,
        confirmed: node?.config.mapping_confirmed
      };
    }, nodeId);
    assert(truthSelection.truthPath === "dataset/truth-alt.h5", "The visible truth selection was not persisted on the evaluation node");
    assert(truthSelection.fieldPairs === "", "Changing truth data did not clear the stale field mapping");
    assert(truthSelection.confirmed === "False", "Changing truth data did not clear mapping confirmation");

    await page.locator("#evaluationConfirmMapping").check();
    assert(!(await runButton.isDisabled()), "Confirmed unique mapping did not enable evaluation");

    await page.locator('[data-evaluation-field="1"]').uncheck();
    assert(!(await page.locator("#evaluationConfirmMapping").isChecked()), "Deselecting a prediction field did not clear mapping confirmation");
    const oneFieldMapping = await page.evaluate(id => {
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.id === id);
      return JSON.parse(node?.config.field_pairs || "[]");
    }, nodeId);
    assert(oneFieldMapping.length === 1 && oneFieldMapping[0].prediction_index === 0, "The field checkbox did not persist the reduced mapping");
    await page.locator('[data-evaluation-field="1"]').check();
    assert(!(await page.locator("#evaluationConfirmMapping").isChecked()), "Restoring a prediction field did not keep confirmation cleared");
    await page.locator("#evaluationConfirmMapping").check();
    assert(!(await runButton.isDisabled()), "Restoring and confirming the full mapping did not enable evaluation");

    await page.locator('[data-evaluation-truth="0"]').selectOption("1");
    assert(!(await page.locator("#evaluationConfirmMapping").isChecked()), "Changing a mapping did not clear its confirmation");
    await page.locator("#evaluationConfirmMapping").check();
    assert(await runButton.isDisabled(), "Duplicate truth fields were accepted");

    await page.locator('[data-evaluation-truth="0"]').selectOption("0");
    await page.locator("#evaluationConfirmMapping").check();
    assert(!(await runButton.isDisabled()), "A repaired unique mapping stayed disabled");
    await runButton.click();
    await page.waitForSelector("#evaluationResults .live-summary");
    assert(runBodies.length === 1, "Evaluation endpoint was not called exactly once");
    assert(runBodies[0].truth_path === "dataset/truth-alt.h5", "Evaluation did not use the truth file selected through the GUI");
    assert(runBodies[0].confirm_mapping === true, "The confirmation was not sent to the backend");
    assert(new Set(runBodies[0].field_pairs.map(pair => pair.truth_index)).size === 2, "The submitted truth mapping was not unique");
    assert((await page.locator("#evaluationResults").innerText()).includes("2\nevaluated samples"), "Real evaluation results were not rendered");

    const saved = await page.evaluate(id => {
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.id === id);
      return node?.config;
    }, nodeId);
    assert(saved.mapping_mode === "schema", "Schema mapping mode was not persisted");
    assert(saved.report_path.endsWith("report.json"), "Evaluation evidence path was not persisted");
    const pipelineEvaluation = await page.evaluate(async id => {
      const module = await import("./src/validate.js");
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.id === id);
      return module.analysisStep(node, module.executableSteps());
    }, nodeId);
    assert(pipelineEvaluation.payload.field_pairs.length === 2, "Pipeline execution dropped the saved field mapping");
    assert(pipelineEvaluation.payload.confirm_mapping === true, "Pipeline execution dropped mapping confirmation");
    assert(!Object.hasOwn(pipelineEvaluation.payload, "prediction_start"), "Schema-mode pipeline silently fell back to legacy rows");

    // Start a slow schema request, leave the workspace, and prove its eventual
    // completion cannot replace or mutate the new Docs workspace.
    await page.locator("#evaluationPrediction").selectOption("frontend/runtime/prediction-b.h5");
    await page.waitForTimeout(20);
    await page.evaluate(() => window.__AI_CAE_FRONTEND__.openStudio("docs"));
    await page.waitForSelector('[data-studio-section="docs"]');
    await page.waitForTimeout(320);
    const mainText = await page.locator("#studioMain").innerText();
    assert(mainText.includes("GUI guide"), "Docs workspace did not render after switching");
    assert(!mainText.includes("Fields to score"), "A stale evaluation response overwrote the active workspace");

    assert(browserErrors.length === 0, `Browser errors: ${browserErrors.join(" | ")}`);
    console.log("Evaluation UI contract and workspace-race smoke test passed.");
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack || error.message);
  process.exit(1);
});
