const playwrightModule = process.argv[2] || "playwright";
const { chromium } = require(playwrightModule);
const { pathToFileURL } = require("url");
const path = require("path");

const studioUrl = process.argv[3] || pathToFileURL(path.join(__dirname, "index.html")).href;
const liveRuntime = studioUrl.startsWith("http://") || studioUrl.startsWith("https://");

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 960 } });
  const browserErrors = [];
  page.on("pageerror", error => browserErrors.push(`pageerror: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error") browserErrors.push(`console: ${message.text()}`);
  });

  try {
    await page.goto(studioUrl);
    if (liveRuntime) {
      await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));
      assert((await page.locator(".route-health").innerText()).toLowerCase().includes("11/11 routes live"), "Live registry did not connect");
    }

    const contract = await page.evaluate(() => ({
      models: Object.keys(window.__AI_CAE_FRONTEND__.MODEL_CATALOG).length,
      simulgenKeys: window.__AI_CAE_FRONTEND__.MODEL_CATALOG.simulgenvae.keys.length,
      simulgenModes: window.__AI_CAE_FRONTEND__.MODEL_CATALOG.simulgenvae.modes.length
    }));
    assert(contract.models === 10, "Expected all ten live trainable model routes");
    assert(contract.simulgenKeys === 67, "Expected all 67 SimulGen keys");
    assert(contract.simulgenModes === 4, "Expected all four SimulGen modes");

    assert(await page.locator(".node").count() === 7, "Expected seven blocks in the SimulGen pipeline");
    assert(await page.locator("#edgeLayer path.edge").count() === 9, "Expected nine semantically valid default links");
    assert(
      await page.locator("#edgeLayer").evaluate(element => getComputedStyle(element).transform === "none"),
      "Edge layer must not apply the canvas transform twice"
    );
    const defaultGraph = await page.evaluate(() => ({
      edges: window.__AI_CAE_FRONTEND__.state.edges,
      scale: window.__AI_CAE_FRONTEND__.state.view.scale
    }));
    assert(
      !defaultGraph.edges.some(edge => edge.fromNode === "conditions" && edge.toNode === "dataset"),
      "Design parameters must not loop backward into the default dataset"
    );
    assert((await page.locator('[data-node-id="simulgen"]').innerText()).includes("SimulGen-VAE"), "SimulGen block is missing");
    assert(
      await page.locator("#studioShell").evaluate(element => element.classList.contains("inspector-collapsed")),
      "The pipeline-first default should keep the inspector collapsed"
    );
    await page.screenshot({ path: path.join(__dirname, "runtime", "pipeline-fixed.png"), fullPage: false });

    const panResult = await page.locator("#stage").evaluate(element => {
      const state = window.__AI_CAE_FRONTEND__.state;
      const before = { x: state.view.x, y: state.view.y };
      const rect = element.getBoundingClientRect();
      const pointerId = 91;
      element.dispatchEvent(new PointerEvent("pointerdown", {
        bubbles: true,
        cancelable: true,
        button: 0,
        buttons: 1,
        pointerId,
        clientX: rect.left + 300,
        clientY: rect.top + 300
      }));
      document.dispatchEvent(new PointerEvent("pointermove", {
        bubbles: true,
        buttons: 1,
        pointerId,
        clientX: rect.left + 350,
        clientY: rect.top + 330
      }));
      document.dispatchEvent(new PointerEvent("pointerup", {
        bubbles: true,
        button: 0,
        pointerId,
        clientX: rect.left + 350,
        clientY: rect.top + 330
      }));
      return { before, after: { x: state.view.x, y: state.view.y } };
    });
    assert(
      panResult.after.x - panResult.before.x === 50 && panResult.after.y - panResult.before.y === 30,
      `Canvas drag did not pan predictably: ${JSON.stringify(panResult)}`
    );

    await page.locator('[data-node-id="simulgen"] .node-head').click();
    assert(
      !(await page.locator("#studioShell").evaluate(element => element.classList.contains("inspector-collapsed"))),
      "Selecting a block did not open the inspector"
    );
    assert((await page.locator("#inspectorContent").innerText()).includes("67 live keys"), "67-key inspector summary is missing");
    assert((await page.locator("#inspectorContent").innerText()).includes("Dataset gate"), "Fixed-geometry contract is missing");
    await page.waitForTimeout(300);
    await page.screenshot({ path: path.join(__dirname, "runtime", "pipeline-inspector.png"), fullPage: false });
    await page.locator("#hideInspector").click();
    assert(
      await page.locator("#studioShell").evaluate(element => element.classList.contains("inspector-collapsed")),
      "Inspector close control did not return space to the pipeline"
    );
    await page.locator("#showInspector").click();
    assert(
      !(await page.locator("#studioShell").evaluate(element => element.classList.contains("inspector-collapsed"))),
      "Inspector reopen control did not restore the panel"
    );

    const paletteModels = await page.locator('.palette-item[data-block-type^="model."] .palette-name').allInnerTexts();
    const sortedPaletteModels = [...paletteModels].sort((left, right) =>
      left.localeCompare(right, undefined, { sensitivity: "base" })
    );
    assert(
      JSON.stringify(paletteModels) === JSON.stringify(sortedPaletteModels),
      `Model palette is not alphabetical: ${paletteModels.join(", ")}`
    );

    const fontSizes = await page.evaluate(() => ({
      palette: Number.parseFloat(getComputedStyle(document.querySelector(".palette-name")).fontSize),
      description: Number.parseFloat(getComputedStyle(document.querySelector(".palette-desc")).fontSize),
      nodeTitle: Number.parseFloat(getComputedStyle(document.querySelector(".node-title")).fontSize),
      port: Number.parseFloat(getComputedStyle(document.querySelector(".port")).width)
    }));
    assert(fontSizes.palette >= 11, `Palette font is still too small: ${fontSizes.palette}px`);
    assert(fontSizes.description >= 9, `Palette description is still too small: ${fontSizes.description}px`);
    assert(fontSizes.nodeTitle >= 12, `Node title is still too small: ${fontSizes.nodeTitle}px`);
    assert(fontSizes.port >= 20, `Ports are still too small: ${fontSizes.port}px`);

    const zoomBefore = defaultGraph.scale;
    await page.locator("#stage").evaluate(element => {
      const rect = element.getBoundingClientRect();
      element.dispatchEvent(new WheelEvent("wheel", {
        bubbles: true,
        cancelable: true,
        clientX: rect.left + rect.width / 2,
        clientY: rect.top + rect.height / 2,
        deltaY: -140
      }));
    });
    const zoomAfter = await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.view.scale);
    assert(zoomAfter > zoomBefore, `Mouse-wheel zoom did not change scale: ${zoomBefore} -> ${zoomAfter}`);
    assert((await page.locator("#zoomLevel").innerText()).includes("%"), "Visible zoom percentage is missing");

    await page.evaluate(async () => {
      const { renderRuntimeJob } = await import("./src/run.js");
      renderRuntimeJob({ id: "drawer-poll-smoke", label: "Drawer polling smoke", status: "running", current_step: 1, total_steps: 2, log: "running" });
    });
    await page.locator("#runtimeMinimize").click();
    assert(await page.locator("#runtimeDrawer").evaluate(element => element.classList.contains("minimized")), "Runtime drawer did not minimize");
    await page.evaluate(async () => {
      const { applyJobStatus } = await import("./src/run.js");
      applyJobStatus({ id: "drawer-poll-smoke", label: "Drawer polling smoke", status: "running", current_step: 2, total_steps: 2, log: "poll update" });
    });
    assert(await page.locator("#runtimeDrawer").evaluate(element => element.classList.contains("minimized")), "Polling reopened the minimized runtime drawer");
    assert(await page.locator("#runtimeMinimize").innerText() === "+", "Minimized runtime control was reset by polling");
    await page.evaluate(async () => {
      const { applyJobStatus, dismissRuntimeJob } = await import("./src/run.js");
      applyJobStatus({ id: "drawer-poll-smoke", label: "Drawer polling smoke", status: "completed", current_step: 2, total_steps: 2, log: "completed", finished_at: "now" });
      dismissRuntimeJob();
    });

    await page.locator("#openFullConfig").click();
    assert(await page.locator("#configOverlay").evaluate(element => element.classList.contains("open")), "Configuration did not open");
    assert((await page.locator("#configBadges").innerText()).includes("67 accepted"), "Accepted-key count is wrong");
    assert((await page.locator("#configBadges").innerText()).includes("18 required"), "Train required-key count is wrong");
    assert((await page.locator("#configRaw").inputValue()).includes("vae_modelpath"), "VAE checkpoint key is missing");
    assert((await page.locator("#configRaw").inputValue()).includes("lc_modelpath"), "LC checkpoint key is missing");
    await page.screenshot({ path: path.join(__dirname, "runtime", "config-v2.png"), fullPage: false });
    await page.locator("#configRaw").fill("MODEL SimulGenVAE\nMoDe TRAIN_VAE\nBATCH_SIZE 7\nbatch_size 9\nDATASET_DIR ../Dataset/CaseSensitive.H5");
    await page.locator("#parseRaw").click();
    const caseInsensitiveConfig = await page.evaluate(() => {
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.id === "simulgen");
      return { config: node.config, keys: Object.keys(node.config) };
    });
    assert(caseInsensitiveConfig.config.model === "simulgenvae", "Model value was not normalized case-insensitively");
    assert(caseInsensitiveConfig.config.mode === "train_vae", "Mode value was not normalized case-insensitively");
    assert(caseInsensitiveConfig.config.batch_size === "9", "Case-insensitive duplicate keys did not use the last value");
    assert(caseInsensitiveConfig.config.dataset_dir === "../Dataset/CaseSensitive.H5", "Free-form path casing was not preserved");
    assert(!caseInsensitiveConfig.keys.some(key => /[A-Z]/.test(key)), "Config keys were not canonicalized to lowercase");
    assert((await page.locator("#configDiagnostics").innerText()).includes("case-insensitive"), "Case-insensitive duplicate diagnostic is missing");

    if (liveRuntime) {
      await page.locator("#configPreset").selectOption("smoke");
      page.once("dialog", dialog => dialog.accept());
      await page.locator("#applyPreset").click();
      await page.waitForFunction(() => document.querySelector("#configRaw")?.value.includes("frontend/runtime/simulgen-smoke"));
      await page.locator("#preflightConfig").click();
      await page.waitForFunction(() => document.querySelector("#configDiagnostics")?.textContent.includes("Authoritative preflight"));
      assert((await page.locator("#configDiagnostics").innerText()).includes("0 errors"), "Real SimulGen preflight did not pass");
    }

    await page.locator("#configMode").selectOption("reconstruct");
    assert((await page.locator("#configBadges").innerText()).includes("13 required"), "Reconstruct required-key count is wrong");
    await page.locator('[data-close="configOverlay"]').click();

    if (liveRuntime) {
      await page.locator('[data-node-id="dataset"] .node-head').click();
      await page.locator("#inspectorSamples").click();
      assert(await page.locator("#artifactOverlay").evaluate(element => element.classList.contains("open")), "Sample viewer did not open");
      await page.waitForFunction(() => document.querySelectorAll("[data-real-sample]").length > 0);
      assert(
        await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.realArtifact.currentSample === null),
        "Sample viewer must start without a default sample"
      );
      await page.locator("[data-real-sample]").first().click();
      await page.waitForFunction(() => document.querySelector("#sampleInfo")?.textContent.includes("actual mean"));
      assert((await page.locator("#artifactSubtitle").innerText()).includes("actual repository values"), "Sample viewer is not using actual repository data");
      assert(await page.locator('[data-view-mode="field"]').evaluate(element => element.classList.contains("active")), "Field mode is not the default");
      const drawn = () => page.evaluate(() => ({ ...window.__AI_CAE_FRONTEND__.state.viewerDraw }));
      assert((await drawn()).drewFaces, "Field view did not render reconstructed elements");
      assert((await page.locator("#viewerModeMeta").innerText()).includes("elements"), "Mesh topology metadata is missing");

      await page.locator('[data-view-mode="mesh"]').click();
      assert(await page.locator('[data-view-mode="mesh"]').evaluate(element => element.classList.contains("active")), "Mesh mode did not activate");
      assert((await drawn()).drewEdges, "Mesh view did not render real mesh edges");

      await page.locator('[data-view-mode="points"]').click();
      assert(await page.locator('[data-view-mode="points"]').evaluate(element => element.classList.contains("active")), "Points mode did not activate");
      const pointDraw = await drawn();
      assert(!pointDraw.drewEdges && !pointDraw.drewFaces, "Points mode should not render mesh topology");
      assert(pointDraw.drewPoints, "Points mode did not render sampled nodes");

      await page.locator('[data-view-mode="field"]').click();
      await page.screenshot({ path: path.join(__dirname, "runtime", "mesh-field-viewer.png"), fullPage: false });
      const comparedArtifactPath = await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.realArtifact?.path);
      await page.locator("#artifactCompare").click();
      await page.waitForSelector("#evaluationPrediction");
      assert(await page.locator("#evaluationPrediction").inputValue() === comparedArtifactPath, "Artifact Compare did not carry the current HDF5 into Evaluation");
      await page.locator('[data-close="studioOverlay"]').click();

      await page.locator('[data-section="models"]').click();
      await page.waitForFunction(() => document.querySelector("#studioMain")?.textContent.includes("registered routes"));
      assert((await page.locator("#studioMain").innerText()).includes("simulgenvae"), "Live Models workspace is missing SimulGen");
      await page.locator('[data-close="studioOverlay"]').click();

      await page.locator("#templateSelect").selectOption("blank");
      await page.locator('.palette-item[data-block-type="source.hdf5"]').click();
      await page.locator('.palette-item[data-block-type="model.mlp"]').click();
      const added = await page.evaluate(() => Object.fromEntries(
        window.__AI_CAE_FRONTEND__.state.nodes.map(node => [node.type, node.id])
      ));
      assert(added["source.hdf5"] && added["model.mlp"], "Could not add HDF5 and MLP blocks to a blank pipeline");
      await page.locator(`[data-node="${added["model.mlp"]}"][data-port="data"][data-direction="input"]`).click();
      assert(await page.locator(".port.link-target").count() > 0, "Compatible link targets were not highlighted");
      await page.locator(`[data-node="${added["source.hdf5"]}"][data-port="data"][data-direction="output"]`).click();
      const linked = await page.evaluate(() => ({
        edges: window.__AI_CAE_FRONTEND__.state.edges,
        pending: window.__AI_CAE_FRONTEND__.state.pendingPort
      }));
      assert(linked.edges.length === 1, "New blocks did not create exactly one link");
      assert(
        linked.edges[0].fromNode === added["source.hdf5"] && linked.edges[0].toNode === added["model.mlp"],
        "Input-first linking created the wrong direction"
      );
      assert(linked.pending === null, "Linking left a stale pending port");

      await page.locator(`[data-node-id="${added["source.hdf5"]}"] .node-head`).click();
      assert(await page.locator("#browseInputSource").count() === 1, "HDF5 source does not expose an input picker");
      assert(await page.locator("#uploadInputSource").count() === 1, "HDF5 source does not expose local upload");
      await page.locator("#browseInputSource").click();
      await page.waitForFunction(() => document.querySelectorAll("[data-use-input]").length > 0);
      await page.locator("[data-use-input]").first().click();
      const selectedInput = await page.evaluate(id =>
        window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.id === id)?.config.path,
        added["source.hdf5"]
      );
      assert(/\.(h5|hdf5)$/i.test(selectedInput || ""), `HDF5 picker did not set a valid path: ${selectedInput}`);
    }

    await page.locator('[data-section="optimization"]').click();
    assert(await page.locator("#studioOverlay").evaluate(element => element.classList.contains("open")), "Optimization workspace did not open");
    assert((await page.locator("#studioMain").innerText()).includes("Pareto"), "Optimization decision support is missing");

    if (liveRuntime) {
      await page.locator('[data-close="studioOverlay"]').click();
      await page.locator("#templateSelect").selectOption("geometry");
      if (await page.locator("#artifactOverlay").evaluate(element => element.classList.contains("open"))) {
        await page.locator('[data-close="artifactOverlay"]').click({ force: true });
      }
      await page.locator('[data-node-id="ingest"] .node-head').click();
      const geometryErrors = await page.evaluate(() => window.__AI_CAE_FRONTEND__.validateGraph(false));
      assert(geometryErrors.length === 0, `Geometry template validation failed: ${geometryErrors.join("; ")}`);
      page.once("dialog", dialog => dialog.accept());
      await page.locator("#inspectorRun").click();
      await page.waitForFunction(
        () => document.querySelector("#runtimeJobMeta")?.textContent.includes("completed"),
        null,
        { timeout: 30000 }
      );
      assert((await page.locator("#runtimeLog").innerText()).includes("[studio] Pipeline completed"), "Geometry ingest did not execute through the real launcher");
      const lineage = await page.evaluate(() => ({
        target: window.__AI_CAE_FRONTEND__.state.api.activeJob?.target_node_id,
        nodeId: window.__AI_CAE_FRONTEND__.state.api.activeJob?.steps?.[0]?.node_id,
        nodeType: window.__AI_CAE_FRONTEND__.state.api.activeJob?.steps?.[0]?.node_type
      }));
      assert(lineage.target === "ingest" && lineage.nodeId === "ingest" && lineage.nodeType === "prep.geometry", `Job lost exact pipeline-node lineage: ${JSON.stringify(lineage)}`);

      await page.evaluate(() => window.__AI_CAE_FRONTEND__.openStudio("evaluation"));
      assert((await page.locator("#studioMain").innerText()).includes("Actual HDF5 field comparison"), "Evaluation workspace is not live");
      await page.evaluate(() => window.__AI_CAE_FRONTEND__.openStudio("comparison"));
      assert((await page.locator("#studioMain").innerText()).includes("Connected run comparison"), "Comparison workspace is not live");
      await page.evaluate(() => window.__AI_CAE_FRONTEND__.openStudio("export"));
      assert((await page.locator("#studioMain").innerText()).includes("Isolated artifact handoff"), "Export workspace is not live");
    }

    const workspaceSections = [
      "models",
      "data",
      "experiments",
      "optimization",
      "benchmarks",
      "artifacts",
      "deploy",
      "system",
      "docs"
    ];
    if (await page.locator("#studioOverlay").evaluate(element => element.classList.contains("open"))) {
      await page.locator('[data-close="studioOverlay"]').click();
    }
    for (const section of workspaceSections) {
      await page.locator(`.topnav [data-section="${section}"]`).click();
      assert(
        await page.locator("#studioOverlay").evaluate(element => element.classList.contains("open")),
        `${section} workspace did not open from the top navigation`
      );
      await page.waitForFunction(() => (document.querySelector("#studioMain")?.textContent || "").trim().length > 40);
      assert(
        (await page.locator("#studioMain").innerText()).trim().length > 40,
        `${section} workspace rendered no useful content`
      );
      await page.locator('[data-close="studioOverlay"]').click();
    }

    const compactContext = await browser.newContext({ viewport: { width: 1366, height: 768 } });
    const compactPage = await compactContext.newPage();
    compactPage.on("pageerror", error => browserErrors.push(`compact pageerror: ${error.message}`));
    compactPage.on("console", message => {
      if (message.type() === "error") browserErrors.push(`compact console: ${message.text()}`);
    });
    await compactPage.goto(studioUrl);
    if (liveRuntime) {
      await compactPage.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));
    }
    const compactLayout = await compactPage.evaluate(() => ({
      innerWidth: window.innerWidth,
      scrollWidth: document.documentElement.scrollWidth,
      workspaceWidth: document.querySelector(".workspace").getBoundingClientRect().width,
      nodes: document.querySelectorAll(".node").length
    }));
    assert(compactLayout.scrollWidth <= compactLayout.innerWidth, `Compact layout overflows horizontally: ${JSON.stringify(compactLayout)}`);
    assert(compactLayout.workspaceWidth >= 950, `Pipeline canvas is too narrow at 1366px: ${compactLayout.workspaceWidth}`);
    assert(compactLayout.nodes === 7, "Compact pipeline did not render all seven blocks");
    await compactPage.screenshot({ path: path.join(__dirname, "runtime", "pipeline-1366.png"), fullPage: false });
    await compactContext.close();

    assert(browserErrors.length === 0, `Browser reported errors: ${browserErrors.join(" | ")}`);
    console.log(`PASS: ${liveRuntime ? "live runtime, real preflight, real HDF5 samples, geometry ingest, " : ""}SimulGen config, Evaluation, Comparison, Export, and Optimization`);
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(`FAIL: ${error.message}`);
  process.exitCode = 1;
});
