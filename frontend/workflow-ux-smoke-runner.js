const { chromium } = require(process.argv[2] || "playwright");
const path = require("path");

const studioUrl = process.argv[3] || "http://127.0.0.1:8081/index.html";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const context = await browser.newContext({ viewport: { width: 1600, height: 960 } });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(`pageerror: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });

  try {
    await page.goto(studioUrl);
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));
    await page.locator("#templateSelect").selectOption("blank");
    await page.locator("#pipelineName").fill("Persistence smoke");
    await page.locator('.palette-item[data-block-type="source.hdf5"]').click();
    await page.locator('.palette-item[data-block-type="model.mlp"]').click();
    const ids = await page.evaluate(() => Object.fromEntries(
      window.__AI_CAE_FRONTEND__.state.nodes.map(node => [node.type, node.id])
    ));
    await page.locator(`[data-node="${ids["source.hdf5"]}"][data-port="data"][data-direction="output"]`).click();
    await page.locator(`[data-node="${ids["model.mlp"]}"][data-port="data"][data-direction="input"]`).click();
    await page.locator("#savePipeline").click();
    const stored = await page.evaluate(() => JSON.parse(
      localStorage.getItem("ai-cae4all.studio.pipeline.v1")
    ));
    assert(stored.name === "Persistence smoke", "Pipeline name was not persisted");
    assert(stored.nodes.length === 2 && stored.edges.length === 1, "Saved pipeline lost graph data");

    await page.reload();
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));
    assert(await page.locator("#pipelineName").inputValue() === "Persistence smoke", "Reload did not restore the pipeline name");
    assert(await page.locator(".node").count() === 2, "Reload did not restore both blocks");
    assert(await page.locator("#edgeLayer .edge-hit").count() === 1, "Reload did not restore the connection");
    assert(await page.locator("#templateSelect").inputValue() === "saved", "Restored workspace is not identified in the template control");
    const downloadPromise = page.waitForEvent("download");
    await page.locator("#exportPipeline").click();
    const pipelineDownload = await downloadPromise;
    assert(pipelineDownload.suggestedFilename().endsWith(".ai-cae.json"), "Pipeline Export did not create a portable JSON file");
    const invalid = {
      ...stored,
      edges: [...stored.edges, {
        id: "invalid-self-edge",
        fromNode: ids["source.hdf5"],
        fromPort: "data",
        toNode: ids["source.hdf5"],
        toPort: "parameters"
      }]
    };
    await page.locator("#pipelineFile").setInputFiles({
      name: "invalid-cycle.ai-cae.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(invalid))
    });
    await page.waitForFunction(() => [...document.querySelectorAll(".toast")].some(item => item.textContent.includes("cannot connect a block to itself")));
    assert(await page.locator(".node").count() === 2 && await page.locator("#edgeLayer .edge-hit").count() === 1, "Rejected import mutated the current pipeline");

    await page.locator(`[data-node-id="${ids["model.mlp"]}"] .node-preview`).click();
    await page.waitForFunction(() => document.querySelector("#studioMain")?.textContent.includes("Full configuration contract"));
    const modelDetail = await page.locator("#studioMain").innerText();
    assert(modelDetail.includes("Training status"), "Model preview did not show training status");
    assert(modelDetail.includes("Configure via LLM"), "Model preview did not expose LLM configuration");
    assert(!modelDetail.includes("Add dataset"), "Models workspace still exposes the dataset-add action");
    await page.locator('[data-close="studioOverlay"]').click();

    const hdf5Node = page.locator(`[data-node-id="${ids["source.hdf5"]}"]`);
    assert(await hdf5Node.locator("[data-run]").innerText() === "Open", "A data source still presents a misleading Run action");
    await hdf5Node.locator("[data-node-menu]").click();
    assert(await hdf5Node.locator('[role="menu"]').evaluate(element => element.classList.contains("open")), "Block action menu did not open");
    await page.keyboard.press("ArrowDown");
    assert(await page.evaluate(() => document.activeElement?.hasAttribute("data-menu-duplicate")), "Arrow keys did not move through block menu actions");
    await page.keyboard.press("Enter");
    assert(await page.locator(".node").count() === 3, "Duplicate block action did not create a copy");
    const copyId = await page.evaluate(() =>
      window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.id.includes("_copy_"))?.id
    );
    await page.locator(`[data-node-id="${copyId}"] [data-node-menu]`).click();
    await page.locator(`[data-node-id="${copyId}"] [data-menu-delete]`).click();
    assert(await page.locator(".node").count() === 2, "Delete block menu action did not remove the copy");

    await page.locator("#edgeLayer .edge-hit").dispatchEvent("click");
    const connectionState = await page.evaluate(() => ({
      selectedEdge: window.__AI_CAE_FRONTEND__.state.selectedEdge,
      selectedNode: window.__AI_CAE_FRONTEND__.state.selectedNode,
      inspector: document.querySelector("#inspectorContent")?.textContent || "",
      collapsed: document.querySelector("#studioShell")?.classList.contains("inspector-collapsed")
    }));
    assert(connectionState.inspector.includes("exact typed connection is selected"), `Connection selection did not show its contract: ${JSON.stringify(connectionState)}`);
    await page.locator("#deleteConnection").click();
    assert(await page.locator("#edgeLayer .edge-hit").count() === 0, "Delete connection did not remove the edge");
    assert(await page.locator(".node").count() === 2, "Deleting a connection also removed a block");

    await page.locator('.palette-item[data-block-type="optimize.design"]').click();
    const optimizationId = await page.evaluate(() =>
      window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.type === "optimize.design")?.id
    );
    await page.locator(`[data-node-id="${optimizationId}"] .node-preview`).click();
    await page.waitForSelector("#optimizationCsv");
    const csvValues = await page.locator("#optimizationCsv option").evaluateAll(options =>
      options.map(option => option.value).filter(Boolean)
    );
    assert(csvValues.length > 0, "Optimization workspace found no repository CSV files");
    await page.locator("#optimizationCsv").selectOption(csvValues[0]);
    await page.waitForSelector("[data-optimization-objective]");
    assert(await page.locator("[data-optimization-objective]").count() > 0, "Optimization did not expose detected numeric objectives");
    assert(await page.locator("#runOptimization").isDisabled(), "Optimization ran before an objective was chosen");
    await page.locator("[data-optimization-objective]").first().check();
    assert(!(await page.locator("#runOptimization").isDisabled()), "Choosing an objective did not enable optimization");
    const savedOptimization = await page.evaluate(id => {
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.id === id);
      return { csv: node.config.csv_path, objectives: node.config.objectives, directions: node.config.directions };
    }, optimizationId);
    assert(savedOptimization.csv === csvValues[0], "Optimization CSV selection was not retained on the exact block");
    assert(savedOptimization.objectives, "Optimization objective selection was not retained");
    assert(savedOptimization.directions === "min", "Optimization direction was not retained");

    await page.screenshot({ path: path.join(__dirname, "runtime", "workflow-ux-smoke.png"), fullPage: false });

    await page.evaluate(() => window.__AI_CAE_FRONTEND__.openStudio("data"));
    await page.waitForSelector("[data-use-file]");
    const chosenFile = await page.locator("[data-use-file]").first().evaluate(button => ({
      path: button.dataset.useFile,
      type: button.dataset.useType
    }));
    await page.locator("[data-use-file]").first().click();
    const fileNode = await page.evaluate(() => {
      const state = window.__AI_CAE_FRONTEND__.state;
      const node = state.nodes.find(item => item.id === state.selectedNode);
      return node && { type: node.type, path: node.config.path };
    });
    assert(fileNode?.type === chosenFile.type && fileNode.path === chosenFile.path, "Use in pipeline did not bind the selected repository file");

    await page.evaluate(() => window.__AI_CAE_FRONTEND__.openStudio("benchmarks"));
    await page.waitForSelector("[data-benchmark-preflight]");
    await page.locator("[data-benchmark-preflight]").first().click();
    await page.waitForFunction(() => [...document.querySelectorAll("[data-benchmark-result]")].some(item => item.textContent.trim()), null, { timeout: 30000 });
    assert((await page.locator("[data-benchmark-result]").first().innerText()).length > 0, "Benchmark preflight produced no visible result");

    assert(errors.length === 0, `Browser errors: ${errors.join(" | ")}`);
    console.log("PASS: persistence, menus, edge deletion, model details, schema optimization, file binding, and benchmark preflight");
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
