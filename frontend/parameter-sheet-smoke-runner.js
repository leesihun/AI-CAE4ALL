const { chromium } = require(process.argv[2] || "playwright");
const path = require("path");

const studioUrl = process.argv[3] || "http://127.0.0.1:8080/index.html";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 930 } });
  // The studio shows a one-time orientation card to first-run users, and every
  // smoke run starts from a clean browser profile -- so without this it would
  // meet that modal on every launch. Seed the "already welcomed" flag instead of
  // clicking the card away, so the runs exercise the studio a returning user sees.
  await page.addInitScript(() => {
    try { localStorage.setItem("ai-cae4all.studio.welcomed.v1", "1"); } catch { /* storage blocked */ }
  });

  const browserErrors = [];
  page.on("pageerror", error => browserErrors.push(error.message));
  page.on("console", message => {
    if (message.type() === "error") browserErrors.push(message.text());
  });

  let resolveInitialHealthGate = () => {};
  let initialHealthGateReleased = false;
  let initialHealthRequestHeld = false;
  const initialHealthGate = new Promise(resolve => {
    resolveInitialHealthGate = resolve;
  });
  const releaseInitialHealthGate = () => {
    if (initialHealthGateReleased) return;
    initialHealthGateReleased = true;
    resolveInitialHealthGate();
  };

  try {
    await page.route("**/api/health", async route => {
      if (initialHealthRequestHeld) {
        await route.continue();
        return;
      }
      initialHealthRequestHeld = true;
      await initialHealthGate;
      await route.continue();
    });
    const initialHealthRequest = page.waitForRequest(request =>
      new URL(request.url()).pathname === "/api/health"
    );
    await page.goto(studioUrl);
    await initialHealthRequest;
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__?.state?.nodes?.length);
    assert(
      await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.api.connected) === false,
      "Initial runtime health gate did not keep the Studio disconnected"
    );
    await page.evaluate(() => window.__AI_CAE_FRONTEND__.loadTemplate("parametric", false));
    const mlpParametersId = await page.evaluate(() =>
      window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.type === "source.parameters")?.id
    );
    assert(mlpParametersId, "Parametric template Design Parameters node is missing");

    await page.locator(`[data-node-id="${mlpParametersId}"] .node-head`).click();
    const mlpPreviewResponse = page.waitForResponse(response => {
      const url = new URL(response.url());
      return url.pathname === "/api/preview/samples"
        && url.searchParams.get("path") === "dataset/mlp/train.h5";
    });
    await page.locator("#openParameterSpreadsheet").click();
    await page.locator(".parameter-sheet").waitFor({ state: "visible" });
    assert(
      await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.api.connected) === false,
      "Spreadsheet opened only after the runtime health gate was released"
    );
    const previewResponse = await mlpPreviewResponse;
    assert(previewResponse.ok(), `MLP preview request failed with HTTP ${previewResponse.status()}`);
    await page.waitForFunction(() => document.querySelectorAll("#parameterSheetRows tr").length === 512);
    assert((await page.locator("#artifactTitle").innerText()).includes("MLP paired dataset"), "MLP mapping profile is missing");
    assert(await page.locator("#parameterSheetRows tr").count() === 512, "MLP sheet must have one row for every HDF5 sample");
    assert(
      await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.api.connected) === false,
      "Runtime connected before the gated spreadsheet preview rendered"
    );
    releaseInitialHealthGate();
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.api.connected);
    const initialInputColumns = await page.locator('.parameter-column-kind.input').count();
    assert(initialInputColumns === 3, `Expected three HDF5 input columns, found ${initialInputColumns}`);
    assert(await page.locator('.parameter-column-kind.output').count() === 2, "Expected two HDF5 output columns");
    assert(await page.locator('[data-add-parameter-column="input"]').count() === 1, "Add Input column is missing");
    assert(await page.locator('[data-add-parameter-column="output"]').count() === 1, "Add Output column is missing for MLP");
    assert(await page.locator("#artifactAddDataset").isVisible(), "Add dataset must remain visible");
    assert(await page.locator("#viewerVisual canvas").count() === 0, "3D canvas must not appear in Design Parameters");

    const firstTwoRows = await page.locator('#parameterSheetRows tr').evaluateAll(rows => rows.slice(0, 2).map(row => ({
      sampleId: row.dataset.sampleId,
      label: row.querySelector('.parameter-sample-cell strong')?.textContent,
      values: [...row.querySelectorAll('.parameter-sheet-value')].map(input => input.value)
    })));
    assert(firstTwoRows[0].sampleId === "0" && firstTwoRows[1].sampleId === "1", `Rows do not follow dataset order: ${JSON.stringify(firstTwoRows)}`);
    assert(firstTwoRows[0].label === "row 0" && firstTwoRows[1].label === "row 1", "Dataset row labels are not locked");
    assert(firstTwoRows[0].values.length === 5 && firstTwoRows[0].values.every(Boolean), "MLP X/Y values were not imported into row 1");

    await page.locator('[data-add-parameter-column="input"]').click();
    assert(await page.locator('.parameter-column-kind.input').count() === 4, "Input column was not added");
    await page.locator('[data-column-heading="input_4"] .parameter-column-name').fill("temperature");
    await page.locator('.parameter-sheet-value[data-row="0"][data-column-id="input_4"]').fill("650");

    await page.locator('[data-add-parameter-column="output"]').click();
    assert(await page.locator('.parameter-column-kind.output').count() === 3, "Output column was not added");
    await page.locator('[data-column-heading="output_3"] .parameter-column-name').fill("safety_factor");
    await page.locator('.parameter-sheet-value[data-row="0"][data-column-id="output_3"]').fill("1.8");

    const stored = await page.evaluate(id => {
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.id === id);
      return {
        table: JSON.parse(node.config.parameter_table),
        dataset: node.config.parameter_dataset,
        inputs: node.config.condition_names,
        outputs: node.config.feature_names
      };
    }, mlpParametersId);
    assert(stored.table.profile === "mlp_paired", "MLP table profile was not stored");
    assert(stored.dataset === "dataset/mlp/train.h5", `Wrong matched dataset: ${stored.dataset}`);
    assert(stored.table.rows[0].sample_id === "0" && stored.table.rows[1].sample_id === "1", "Stored rows lost sample IDs");
    assert(stored.table.rows[0].values.input_4 === "650", "Added Input value was not stored per sample");
    assert(stored.table.rows[0].values.output_3 === "1.8", "Added Output value was not stored per sample");
    assert(stored.inputs.includes("temperature") && stored.outputs.includes("safety_factor"), "Added column names were not stored");

    await page.waitForTimeout(500);
    await page.locator('[data-close="artifactOverlay"]').click();
    await page.reload();
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));
    await page.evaluate(id => window.__AI_CAE_FRONTEND__.openArtifact(id), mlpParametersId);
    await page.locator(".parameter-sheet").waitFor({ state: "visible" });
    assert(
      await page.locator('.parameter-sheet-value[data-row="0"][data-column-id="input_4"]').inputValue() === "650",
      "Reload lost the edited dataset-row Input value"
    );
    assert(
      await page.locator('.parameter-sheet-value[data-row="0"][data-column-id="output_3"]').inputValue() === "1.8",
      "Reload lost the edited dataset-row Output value"
    );

    await page.screenshot({
      path: path.join(__dirname, "runtime", "design-parameters-spreadsheet.png"),
      fullPage: false
    });
    await page.locator('[data-remove-parameter-column="input_4"]').click();
    assert(await page.locator('[data-column-heading="input_4"]').count() === 0, "Remove column did not remove the added Input column");

    await page.evaluate(() => window.__AI_CAE_FRONTEND__.loadTemplate("simulgen", false));
    const conditionParametersId = await page.evaluate(() =>
      window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.type === "source.parameters")?.id
    );
    await page.evaluate(id => window.__AI_CAE_FRONTEND__.openArtifact(id), conditionParametersId);
    await page.locator(".parameter-sheet").waitFor({ state: "visible" });
    assert((await page.locator("#artifactTitle").innerText()).includes("SimulGen-VAE conditions"), "Non-MLP condition profile is missing");
    assert(await page.locator('[data-add-parameter-column="output"]').count() === 0, "Non-MLP parameters must not force MLP Output columns");

    await page.evaluate(() => window.__AI_CAE_FRONTEND__.loadTemplate("generative", false));
    await page.evaluate(() => window.__AI_CAE_FRONTEND__.openArtifact("parameters"));
    await page.locator(".parameter-sheet.generative").waitFor({ state: "visible" });
    assert(await page.locator('[data-select-parameter-row]').count() === 14, "Generative parameter rows do not expose an explicit candidate selector");
    const beforeSelection = await page.evaluate(() => window.__AI_CAE_FRONTEND__.validateGraph(false));
    assert(beforeSelection.some(message => message.includes("choose one Design Parameters spreadsheet row")), "Conditional generation accepted an unselected spreadsheet");
    await page.locator('[data-column-heading="input_1"] .parameter-column-name').fill("bbox_x");
    await page.locator('.parameter-sheet-value[data-row="0"][data-column-id="input_1"]').fill("1.25");
    await page.locator('[data-select-parameter-row="0"]').check();
    const selected = await page.evaluate(async () => {
      const frontend = window.__AI_CAE_FRONTEND__;
      const { executableSteps } = await import("./src/validate.js");
      const generator = frontend.state.nodes.find(node => node.id === "generator");
      const parameters = frontend.state.nodes.find(node => node.id === "parameters");
      return {
        generator: generator.config,
        table: JSON.parse(parameters.config.parameter_table),
        errors: frontend.validateGraph(false),
        config: executableSteps().find(step => step.nodeId === "generator")?.config || ""
      };
    });
    assert(selected.table.selected_sample_id === "pending:0", "Selected generation row was not persisted");
    assert(selected.generator.cond_values === "1.25", "Selected numeric Input value did not materialize as cond_values");
    assert(selected.generator.condition_sample.includes("pending:0"), "Generator does not identify the source spreadsheet row");
    assert(selected.errors.length === 0, `Selected generation row did not validate: ${selected.errors.join("; ")}`);
    assert(selected.config.split("\n").some(line => line.trim().split(/\s+/).join(" ") === "cond_values 1.25"), "Native SDFFlow config omitted the selected condition value");

    await page.locator('.parameter-sheet-value[data-row="0"][data-column-id="input_1"]').fill("not-a-number");
    const invalid = await page.evaluate(() => ({
      errors: window.__AI_CAE_FRONTEND__.validateGraph(false),
      value: window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.id === "generator").config.cond_values || ""
    }));
    assert(invalid.errors.some(message => message.includes("missing or non-numeric")), "Invalid selected condition did not block execution");
    assert(invalid.value === "", "Invalid spreadsheet edit left stale cond_values on the generator");
    await page.locator('.parameter-sheet-value[data-row="0"][data-column-id="input_1"]').fill("1.5");
    assert(await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.id === "generator").config.cond_values) === "1.5", "Editing the selected row did not refresh cond_values");
    assert(browserErrors.length === 0, `Browser errors: ${browserErrors.join(" | ")}`);

    console.log("PASS: parameter sheets preserve MLP pairs and explicitly materialize one validated SDFFlow generation row.");
  } finally {
    releaseInitialHealthGate();
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
