const playwrightModule = process.argv[2] || "playwright";
const { chromium } = require(playwrightModule);
const path = require("path");

const studioUrl = process.argv[3] || "http://127.0.0.1:8097/index.html";
const fixturePath = path.join(__dirname, "..", "dataset", "ex1_infer.h5");

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function fakeJob(id, label, { autoComplete = false, steps = [] } = {}) {
  return {
    id,
    label,
    status: "running",
    created_at: "2026-08-31T12:00:00+09:00",
    started_at: "2026-08-31T12:00:00+09:00",
    finished_at: null,
    current_step: steps.length ? 1 : 0,
    total_steps: Math.max(1, steps.length),
    step_label: steps[0]?.label || label,
    returncode: null,
    steps,
    log: `${label}\ncontrol-surface smoke log`,
    autoComplete,
    polls: 0
  };
}

function publicJob(job) {
  const { autoComplete, polls, ...payload } = job;
  return payload;
}

function fileItem(index, extension = ".h5") {
  const name = `catalog-${String(index).padStart(4, "0")}${extension}`;
  return {
    name,
    path: `frontend/runtime/control-catalog/${name}`,
    extension,
    kind: "dataset",
    size: 1024 + index,
    modified: "2026-08-31 12:00:00"
  };
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const context = await browser.newContext({ viewport: { width: 1600, height: 960 } });
  await context.grantPermissions(["clipboard-read", "clipboard-write"], {
    origin: new URL(studioUrl).origin
  });
  const page = await context.newPage();
  await page.addInitScript(() => {
    localStorage.setItem("ai-cae4all.studio.welcomed.v1", "1");
    if (sessionStorage.getItem("ai-cae4all.control-surface.initialized.v1") !== "1") {
      localStorage.removeItem("ai-cae4all.studio.pipeline.v1");
      sessionStorage.setItem("ai-cae4all.control-surface.initialized.v1", "1");
    }
  });

  const browserErrors = [];
  const dialogMessages = [];
  const requests = {
    llmConfigure: [],
    llmSettings: [],
    audits: [],
    preflights: [],
    pipelines: [],
    inference: [],
    builds: 0,
    evaluationSchemas: [],
    evaluations: [],
    comparisonSchemas: [],
    comparisons: [],
    optimizationSchemas: [],
    optimizations: []
  };
  const jobs = new Map();
  let phase = "editor";
  let llmSettings = {
    configured: false,
    ready: false,
    scheme: "https",
    master_ip: "",
    port: 10002,
    username: "",
    password_configured: false,
    allow_insecure_http: false,
    base_url: ""
  };

  page.on("pageerror", error => browserErrors.push(`pageerror: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error") browserErrors.push(`console: ${message.text()}`);
  });
  page.on("dialog", async dialog => {
    dialogMessages.push({ type: dialog.type(), message: dialog.message() });
    await dialog.accept(dialog.type() === "prompt" ? "Set batch_size to 13" : undefined);
  });

  await context.route("**/api/**", async route => {
    const request = route.request();
    const url = new URL(request.url());
    const endpoint = url.pathname;
    const json = (body, status = 200) => route.fulfill({
      status,
      contentType: "application/json",
      body: JSON.stringify(body)
    });

    if (endpoint.endsWith("/api/llm/settings")) {
      if (request.method() === "POST") {
        const body = request.postDataJSON();
        requests.llmSettings.push(body);
        llmSettings = {
          configured: true,
          ready: true,
          scheme: body.scheme,
          master_ip: body.master_ip,
          port: Number(body.port),
          username: body.username,
          password_configured: Boolean(body.password),
          allow_insecure_http: Boolean(body.allow_insecure_http),
          base_url: `${body.scheme}://${body.master_ip}:${body.port}/v1`
        };
      }
      return json(llmSettings);
    }
    if (endpoint.endsWith("/api/llm/configure")) {
      const body = request.postDataJSON();
      requests.llmConfigure.push(body);
      const withoutBatch = String(body.config_text || "")
        .split(/\r?\n/)
        .filter(line => !/^\s*batch_size\s+/i.test(line))
        .join("\n")
        .trim();
      return json({ text: `${withoutBatch}\nbatch_size 13\n` });
    }
    if (endpoint.endsWith("/api/audit-configs")) {
      requests.audits.push(url.searchParams.get("strict"));
      return json({
        summary: { files: 2, errors: 1, warnings: 1, notices: 1 },
        files: [{
          path: "configs/control/pass.txt", model: "mlp", mode: "train",
          status: "PASS", errors: 0, warnings: 0,
          report: { diagnostics: [{ severity: "notice", code: "CONTROL-PASS", message: "Pass fixture" }] }
        }, {
          path: "configs/control/broken.txt", model: "transolver", mode: "train",
          status: "FAIL", errors: 1, warnings: 1,
          report: { diagnostics: [{ severity: "error", code: "CONTROL-FAIL", field: "dataset_dir", message: "Missing fixture path", hint: "Choose a dataset" }] }
        }]
      });
    }
    if (endpoint.endsWith("/api/preflight")) {
      const body = request.postDataJSON();
      requests.preflights.push(body);
      return json({
        ok: true,
        route: { model: "meshgraphnets", mode: "train" },
        report: {
          summary: { errors: 0, warnings: 0, notices: 1 },
          diagnostics: [{ severity: "notice", code: "CONTROL-PREFLIGHT", message: "Contained UI preflight" }]
        }
      });
    }
    if (endpoint.endsWith("/api/deploy")) {
      return json({
        existing_exe: null,
        pyinstaller_available: true,
        models: ["mlp", "meshgraphnets", "transolver", "deeponet", "fno", "simulgenvae", "sdfflow", "meshgraphnets-variational"],
        driver_families: ["mlp", "graph", "operator", "generative", "geometry"],
        api_endpoint: "/api/inference/run"
      });
    }
    if (endpoint.endsWith("/api/checkpoint") && url.searchParams.get("path") === "output/control/transolver.pth") {
      return json({
        ok: true,
        path: "output/control/transolver.pth",
        model: "transolver",
        model_source: "checkpoint metadata",
        standalone_inference: true,
        portable_inference: true,
        model_config: { hidden_dim: 64 }
      });
    }
    if (endpoint.endsWith("/api/pipeline/run")) {
      const body = request.postDataJSON();
      requests.pipelines.push(body);
      const id = `control-pipeline-${requests.pipelines.length}`;
      const job = fakeJob(id, body.label || "Control pipeline", { steps: body.steps || [] });
      jobs.set(id, job);
      return json(publicJob(job));
    }
    if (endpoint.endsWith("/api/inference/run")) {
      const body = request.postDataJSON();
      requests.inference.push(body);
      const id = `control-inference-${requests.inference.length}`;
      const job = fakeJob(id, "Portable inference control", { autoComplete: true });
      jobs.set(id, job);
      return json(publicJob(job));
    }
    if (endpoint.endsWith("/api/build/exe")) {
      requests.builds += 1;
      const id = `control-build-${requests.builds}`;
      const job = fakeJob(id, "Portable executable control", { autoComplete: true });
      jobs.set(id, job);
      return json(publicJob(job));
    }
    const cancelMatch = endpoint.match(/\/api\/jobs\/(control-[^/]+)\/cancel$/);
    if (cancelMatch && jobs.has(cancelMatch[1])) {
      const job = jobs.get(cancelMatch[1]);
      Object.assign(job, {
        status: "cancelled",
        returncode: -15,
        finished_at: "2026-08-31T12:00:01+09:00",
        log: `${job.log}\nCancellation requested by control-surface smoke.`
      });
      return json(publicJob(job));
    }
    const jobMatch = endpoint.match(/\/api\/jobs\/(control-[^/]+)$/);
    if (jobMatch && jobs.has(jobMatch[1])) {
      const job = jobs.get(jobMatch[1]);
      job.polls += 1;
      if (job.autoComplete) Object.assign(job, {
        status: "completed",
        current_step: job.total_steps,
        returncode: 0,
        finished_at: "2026-08-31T12:00:01+09:00",
        log: `${job.label}\ncontrol-surface smoke completed`
      });
      return json(publicJob(job));
    }
    if (endpoint.endsWith("/api/docs")) {
      const items = Array.from({ length: 260 }, (_, index) => ({
        name: `doc-${String(index).padStart(4, "0")}.md`,
        path: `frontend/runtime/control-docs/doc-${String(index).padStart(4, "0")}.md`,
        size: 100 + index,
        modified: "2026-08-31 12:00:00"
      }));
      return json({ items, matched: items.length, limit: 500, truncated: false });
    }
    if (endpoint.endsWith("/api/doc")) {
      const docPath = url.searchParams.get("path") || "";
      return json({ path: docPath, text: `# Control document\n\n${docPath}\n` });
    }
    if (endpoint.endsWith("/api/files") && url.searchParams.get("kind") === "dataset" && phase === "data-pagination") {
      const items = Array.from({ length: 260 }, (_, index) => fileItem(index));
      return json({ items, matched: items.length, limit: 500, truncated: false });
    }
    if (endpoint.endsWith("/api/files") && url.searchParams.get("kind") === "artifact" && ["comparison", "optimization"].includes(phase)) {
      const items = [0, 1, 2].map(index => ({
        ...fileItem(index, ".csv"),
        kind: "artifact",
        path: `frontend/runtime/control-results/run-${index}.csv`
      }));
      return json({ items, matched: items.length, limit: 500, truncated: false });
    }
    if (endpoint.endsWith("/api/training-metrics") && phase === "comparison") {
      return json({ count: 0, source: "control", items: [] });
    }
    if (endpoint.endsWith("/api/evaluation/schema")) {
      const body = request.postDataJSON();
      requests.evaluationSchemas.push(body);
      return json({
        compatible: true,
        errors: [],
        warnings: [],
        prediction: { contract: "table", field_count: 2, field_names: ["lift", "drag"], target_indices: [0, 1] },
        truth: { contract: "table", field_count: 2, field_names: ["lift", "drag"], target_indices: [0, 1] },
        sample_matching: { strategy: "id", overlap_count: 2, matched_ids: ["0", "1"], incompatible_shape_count: 0 },
        recommended_mapping: {
          mode: "selected",
          confidence: "exact",
          basis: "matching field names",
          prediction_array: "predictions",
          truth_array: "Y",
          field_pairs: [
            { name: "lift", prediction_index: 0, truth_index: 0, prediction_name: "lift", truth_name: "lift" },
            { name: "drag", prediction_index: 1, truth_index: 1, prediction_name: "drag", truth_name: "drag" }
          ]
        }
      });
    }
    if (endpoint.endsWith("/api/evaluation/run")) {
      const body = request.postDataJSON();
      requests.evaluations.push(body);
      return json({
        contract: "table",
        truth_source: "selected",
        evaluated_samples: 2,
        skipped: [],
        field_pairs: body.field_pairs || [],
        aggregate: { relative_l2: { mean: 0.01 }, mae: { mean: 0.02 }, rmse: { mean: 0.03 } },
        per_sample_csv: "frontend/runtime/evaluation/control/per_sample.csv",
        report_path: "frontend/runtime/evaluation/control/report.json"
      });
    }
    if (endpoint.endsWith("/api/comparison/schema")) {
      const body = request.postDataJSON();
      requests.comparisonSchemas.push(body);
      return json({
        sources: body.csv_paths.map(csvPath => ({ path: csvPath, columns: ["model", "relative_l2", "mae"], rows_sampled: 4 })),
        common_columns: ["model", "relative_l2", "mae"],
        numeric_columns: ["relative_l2", "mae"],
        group_columns: ["model"]
      });
    }
    if (endpoint.endsWith("/api/comparison/run")) {
      const body = request.postDataJSON();
      requests.comparisons.push(body);
      return json({
        numeric_rows: 8,
        runs: body.csv_paths.length,
        best: { name: "control-b", value: 0.1 },
        metric: body.metric,
        direction: body.direction,
        report_path: "frontend/runtime/comparison/control/report.json",
        sources: body.csv_paths.map((csvPath, index) => ({ run: `run-${index}`, path: csvPath, rows: 4, numeric_rows: 4 })),
        ranked: [{ rank: 1, name: "control-b", value: 0.1, source: "run-1", index: 1 }]
      });
    }
    if (endpoint.endsWith("/api/optimization/schema")) {
      const body = request.postDataJSON();
      requests.optimizationSchemas.push(body);
      return json({
        rows_sampled: 12,
        numeric_columns: ["mass", "drag", "score"],
        objective_columns: ["mass", "drag"],
        identifier_columns: ["design_id"],
        numeric_counts: { mass: 12, drag: 12, score: 12 }
      });
    }
    if (endpoint.endsWith("/api/optimization/run")) {
      const body = request.postDataJSON();
      requests.optimizations.push(body);
      return json({
        rows: 12,
        numeric_candidates: 12,
        feasible: 8,
        pareto: 3,
        report_path: "frontend/runtime/optimization/control/report.json",
        selected: [{ id: "design-7", index: 7, objectives: { mass: 12.5 }, crowding: null }]
      });
    }
    return route.continue();
  });

  async function openWorkspace(section) {
    if (await page.locator("#studioOverlay").evaluate(element => element.classList.contains("open"))) {
      await page.locator(`[data-studio-id="${section}"]`).click();
    } else {
      await page.locator(`[data-section="${section}"]`).click();
    }
    await page.waitForFunction(expected =>
      document.querySelector("#studioOverlay")?.classList.contains("open")
      && window.__AI_CAE_FRONTEND__?.state.studioSection === expected,
    section);
  }

  async function addBlock(type) {
    if (await page.locator("#studioOverlay").evaluate(element => element.classList.contains("open"))) {
      await page.locator("#studioPipeline").click();
    }
    if (await page.locator("#studioShell").evaluate(element => element.classList.contains("library-collapsed"))) {
      await page.locator("#showLibrary").click();
    }
    await page.locator("#blockSearch").fill("");
    await page.locator(`.palette-item[data-block-type="${type}"]`).click();
    return page.evaluate(expectedType => {
      const app = window.__AI_CAE_FRONTEND__;
      return app.state.nodes.find(node => node.id === app.state.selectedNode && node.type === expectedType)?.id || "";
    }, type);
  }

  async function nodePosition(id) {
    return page.evaluate(nodeId => {
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.id === nodeId);
      return node ? { x: node.x, y: node.y } : null;
    }, id);
  }

  async function dragNodeBy(id, deltaX, deltaY) {
    const handle = page.locator(`[data-node-id="${id}"] [data-drag-handle]`);
    await handle.scrollIntoViewIfNeeded();
    const box = await handle.boundingBox();
    assert(box, `Drag handle for ${id} has no visible bounding box`);
    const startX = box.x + Math.min(64, box.width * .3);
    const startY = box.y + box.height / 2;
    await page.mouse.move(startX, startY);
    await page.mouse.down();
    await page.mouse.move(startX + deltaX, startY + deltaY, { steps: 8 });
    await page.mouse.up();
    await page.waitForFunction(() => !window.__AI_CAE_FRONTEND__.state.drag);
    return nodePosition(id);
  }

  async function dragElementTo(source, target) {
    await source.scrollIntoViewIfNeeded();
    await target.scrollIntoViewIfNeeded();
    const dataTransfer = await page.evaluateHandle(() => new DataTransfer());
    try {
      await source.dispatchEvent("dragstart", { dataTransfer });
      await target.dispatchEvent("dragenter", { dataTransfer });
      await target.dispatchEvent("dragover", { dataTransfer });
      await target.dispatchEvent("drop", { dataTransfer });
      await source.dispatchEvent("dragend", { dataTransfer });
    } finally {
      await dataTransfer.dispose();
    }
  }

  async function focusCanvas() {
    const box = await page.locator("#stage").boundingBox();
    assert(box, "Canvas stage has no visible bounding box");
    await page.mouse.click(box.x + 12, box.y + box.height - 12);
    await page.waitForFunction(() => !document.activeElement?.matches("input,textarea,select,[contenteditable=true]"));
  }

  try {
    await page.goto(studioUrl);
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));

    // Editor controls: palette search, panels, zoom/layout, inspector actions,
    // and the visible import affordance (including its real file chooser).
    await page.locator("#blockSearch").fill("MLP");
    assert(await page.locator('.palette-item[data-block-type="model.mlp"]').count() === 1, "Block search did not expose MLP");
    await page.locator("#blockSearch").fill("");
    await page.locator("#hideLibrary").click();
    assert(await page.locator("#studioShell").evaluate(element => element.classList.contains("library-collapsed")), "Hide library did not collapse the panel");
    await page.locator("#showLibrary").click();
    assert(!(await page.locator("#studioShell").evaluate(element => element.classList.contains("library-collapsed"))), "Show library did not restore the panel");
    await page.locator("#templateSelect").selectOption("blank");
    const sourceId = await addBlock("source.hdf5");
    const modelId = await addBlock("model.mlp");
    const scaleBefore = await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.view.scale);
    await page.locator("#zoomIn").click();
    assert(await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.view.scale) > scaleBefore, "Zoom in did not change the graph scale");
    await page.locator("#zoomOut").click();
    await page.locator("#arrangeGraph").click();
    await page.locator("#fitGraph").click();
    assert(Number.isFinite(await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.view.scale)), "Fit graph produced an invalid scale");

    // Exercise the actual pointer/HTML5 interaction paths rather than their
    // click alternatives: palette drop, typed port drop, and node movement.
    const stage = page.locator("#stage");
    const stageBox = await stage.boundingBox();
    assert(stageBox, "Canvas stage has no visible drop target");
    const nodeCountBeforePaletteDrop = await page.locator(".node").count();
    await page.locator('.palette-item[data-block-type="source.parameters"]').dragTo(stage, {
      targetPosition: { x: stageBox.width * .78, y: stageBox.height * .76 }
    });
    await page.waitForFunction(expected => window.__AI_CAE_FRONTEND__.state.nodes.length === expected, nodeCountBeforePaletteDrop + 1);
    const parameterId = await page.evaluate(() =>
      window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.type === "source.parameters")?.id || ""
    );
    assert(parameterId, "Palette drag/drop did not create the Design Parameters block");
    await page.locator("#fitGraph").click();

    const edgesBeforePortDrag = await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.edges.length);
    await dragElementTo(
      page.locator(`[data-node="${sourceId}"][data-port="data"][data-direction="output"]`),
      page.locator(`[data-node="${modelId}"][data-port="data"][data-direction="input"]`)
    );
    await page.waitForFunction(({ sourceId, modelId, expected }) =>
      window.__AI_CAE_FRONTEND__.state.edges.length === expected
      && window.__AI_CAE_FRONTEND__.state.edges.some(edge => edge.fromNode === sourceId && edge.toNode === modelId),
    { sourceId, modelId, expected: edgesBeforePortDrag + 1 });
    const compatibleEdges = await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.edges.length);
    await dragElementTo(
      page.locator(`[data-node="${parameterId}"][data-port="parameters"][data-direction="output"]`),
      page.locator(`[data-node="${modelId}"][data-port="data"][data-direction="input"]`)
    );
    assert(await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.edges.length) === compatibleEdges, "Incompatible parameter-to-dataset drag created an edge");
    assert(await page.evaluate(({ sourceId, modelId }) =>
      window.__AI_CAE_FRONTEND__.state.edges.some(edge => edge.fromNode === sourceId && edge.toNode === modelId),
    { sourceId, modelId }), "Incompatible drop replaced the existing compatible edge");
    await page.keyboard.press("Escape");

    const dragOrigin = await nodePosition(sourceId);
    const draggedOnce = await dragNodeBy(sourceId, 92, 58);
    assert(draggedOnce.x > dragOrigin.x + 20 && draggedOnce.y > dragOrigin.y + 20, "Node drag handle did not move the source block");
    await focusCanvas();
    await page.keyboard.press("Control+z");
    await page.waitForFunction(({ id, origin }) => {
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.id === id);
      return node && Math.abs(node.x - origin.x) < .01 && Math.abs(node.y - origin.y) < .01;
    }, { id: sourceId, origin: dragOrigin });
    const persistedPosition = await dragNodeBy(sourceId, 74, 46);
    await page.waitForFunction(({ id, expected }) => {
      const saved = JSON.parse(localStorage.getItem("ai-cae4all.studio.pipeline.v1") || "null");
      const node = saved?.nodes?.find(item => item.id === id);
      return node && Math.abs(node.x - expected.x) < .01 && Math.abs(node.y - expected.y) < .01;
    }, { id: sourceId, expected: persistedPosition }, { timeout: 5000 });
    await page.reload();
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));
    const reloadedPosition = await nodePosition(sourceId);
    assert(reloadedPosition && Math.abs(reloadedPosition.x - persistedPosition.x) < .01 && Math.abs(reloadedPosition.y - persistedPosition.y) < .01, "Dragged node position did not survive local reload");
    assert(await page.evaluate(({ sourceId, modelId }) =>
      window.__AI_CAE_FRONTEND__.state.edges.some(edge => edge.fromNode === sourceId && edge.toNode === modelId),
    { sourceId, modelId }), "Compatible drag-created edge did not survive local reload");

    // Keyboard shortcuts must execute from the canvas and stay inert while a
    // user is typing in a field.
    const guardedView = await page.evaluate(() => ({ ...window.__AI_CAE_FRONTEND__.state.view }));
    await page.locator("#pipelineName").fill("Keyboard guard");
    await page.keyboard.press("f");
    const typingGuard = await page.evaluate(() => ({
      name: document.querySelector("#pipelineName")?.value,
      view: { ...window.__AI_CAE_FRONTEND__.state.view }
    }));
    assert(typingGuard.name === "Keyboard guardf", "Typing guard did not leave the focused pipeline-name field editable");
    assert(JSON.stringify(typingGuard.view) === JSON.stringify(guardedView), "Canvas Fit shortcut fired while an input was focused");

    await page.locator("#hideLibrary").click();
    await focusCanvas();
    await page.keyboard.press("/");
    await page.waitForFunction(() =>
      !document.querySelector("#studioShell")?.classList.contains("library-collapsed")
      && document.activeElement?.id === "blockSearch"
    );

    await focusCanvas();
    const shortcutScale = await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.view.scale);
    await page.keyboard.press("Equal");
    await page.waitForFunction(previous => window.__AI_CAE_FRONTEND__.state.view.scale > previous, shortcutScale);
    const shortcutScaleUp = await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.view.scale);
    await page.keyboard.press("Minus");
    await page.waitForFunction(previous => window.__AI_CAE_FRONTEND__.state.view.scale < previous, shortcutScaleUp);
    await page.locator("#zoomIn").click();
    const beforeFitShortcut = await page.evaluate(() => ({ ...window.__AI_CAE_FRONTEND__.state.view }));
    await focusCanvas();
    await page.keyboard.press("f");
    await page.waitForFunction(previous => JSON.stringify(window.__AI_CAE_FRONTEND__.state.view) !== JSON.stringify(previous), beforeFitShortcut);

    const beforeLayoutShortcut = await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.nodes.map(node => ({ id: node.id, x: node.x, y: node.y })));
    await focusCanvas();
    await page.keyboard.press("l");
    await page.waitForFunction(previous => {
      const positions = new Map(previous.map(item => [item.id, item]));
      return window.__AI_CAE_FRONTEND__.state.nodes.some(node => {
        const before = positions.get(node.id);
        return before && (Math.abs(node.x - before.x) > .01 || Math.abs(node.y - before.y) > .01);
      });
    }, beforeLayoutShortcut);

    const beforeDeleteShortcut = await page.locator(".node").count();
    await page.locator(`[data-node-id="${parameterId}"] .node-head`).click();
    await page.keyboard.press("Delete");
    await page.waitForFunction(id => !window.__AI_CAE_FRONTEND__.state.nodes.some(node => node.id === id), parameterId);
    assert(await page.locator(".node").count() === beforeDeleteShortcut - 1, "Delete shortcut did not remove the selected block");
    await focusCanvas();
    await page.keyboard.press("Control+z");
    await page.waitForFunction(id => window.__AI_CAE_FRONTEND__.state.nodes.some(node => node.id === id), parameterId);

    const shortcutPreflights = requests.preflights.length;
    const shortcutPreflightResponse = page.waitForResponse(response =>
      new URL(response.url()).pathname.endsWith("/api/preflight")
      && response.request().method() === "POST"
    );
    await focusCanvas();
    await page.keyboard.press("v");
    await shortcutPreflightResponse;
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.api.activeJob?.id === "preflight");
    assert(requests.preflights.length === shortcutPreflights + 1, "V shortcut did not submit validation");
    await page.locator("#runtimeDismiss").click();

    const shortcutPipelineCount = requests.pipelines.length;
    await focusCanvas();
    await page.keyboard.press("Control+Enter");
    await page.waitForFunction(expected => window.__AI_CAE_FRONTEND__.state.api.activeJob?.id === expected, `control-pipeline-${shortcutPipelineCount + 1}`);
    assert(requests.pipelines.length === shortcutPipelineCount + 1, "Ctrl+Enter did not submit the pipeline");
    await page.locator("#runtimeCancel").click();
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.api.activeJob?.status === "cancelled");
    await page.locator("#runtimeDismiss").click();

    // Render a contained structured failure, then activate its real Fix now
    // row and verify the full node-to-field focus contract.
    await page.evaluate(async ({ nodeId }) => {
      const { renderRuntimeJob } = await import("./src/run.js");
      renderRuntimeJob({
        id: "control-diagnostic",
        label: "Control diagnostic",
        status: "failed",
        current_step: 1,
        total_steps: 1,
        diagnostics: [{
          severity: "error",
          code: "CONTROL-FIELD",
          stepLabel: "Simple MLP",
          nodeId,
          field: "batch_size",
          message: "Choose a valid batch size."
        }]
      });
    }, { nodeId: modelId });
    await page.locator('[data-jump-diagnostic="0"]').click();
    await page.waitForFunction(expected =>
      window.__AI_CAE_FRONTEND__.state.selectedNode === expected.nodeId
      && window.__AI_CAE_FRONTEND__.state.configRejectedNode === expected.nodeId
      && window.__AI_CAE_FRONTEND__.state.configRejectedField === expected.field
      && document.querySelector("#configOverlay")?.classList.contains("open")
      && document.activeElement?.matches(`.full-config-control[data-key="${expected.field}"]`),
    { nodeId: modelId, field: "batch_size" });
    assert(await page.locator("#runtimeDrawer").evaluate(element => element.classList.contains("minimized")), "Fix now did not minimize the runtime drawer");
    await page.locator('#configOverlay [data-close="configOverlay"]').click();
    await page.locator("#runtimeDismiss").click();
    await page.locator("#runBanner").evaluate(element => element.classList.remove("show"));

    await page.locator(`[data-node-id="${sourceId}"] .node-head`).click();
    const nodeCount = await page.locator(".node").count();
    await page.locator("#duplicateNode").click();
    assert(await page.locator(".node").count() === nodeCount + 1, "Inspector Duplicate did not create a block");
    await page.locator("#deleteNode").click();
    assert(await page.locator(".node").count() === nodeCount, "Inspector Delete did not remove the copy");
    await page.locator("#undoGraph").click();
    assert(await page.locator(".node").count() === nodeCount + 1, "Undo did not restore the deleted block");

    const imported = {
      format: "ai-cae4all-pipeline",
      version: 1,
      name: "Control surface import",
      node_counter: 2,
      view: { x: 26, y: 54, scale: 0.78 },
      nodes: [{
        id: "control_data",
        type: "source.hdf5",
        x: 40,
        y: 60,
        config: { path: "dataset/ex1_infer.h5" },
        auto_fill: {},
        manual_config_keys: ["path"]
      }],
      edges: []
    };
    const importChooser = page.waitForEvent("filechooser");
    await page.locator("#importPipeline").click();
    await (await importChooser).setFiles({
      name: "control-surface.ai-cae.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(imported))
    });
    await page.waitForFunction(() => document.querySelector("#pipelineName")?.value === "Control surface import");

    for (const action of ["inspect", "menu", "run"]) {
      const node = page.locator('[data-node-id="control_data"]');
      if (action === "inspect") await node.locator("[data-inspect]").click();
      if (action === "menu") {
        await node.locator("[data-node-menu]").click();
        await node.locator("[data-menu-open]").click();
      }
      if (action === "run") await node.locator("[data-run]").click();
      await page.waitForFunction(() =>
        document.querySelector("#artifactOverlay")?.classList.contains("open")
        && document.querySelector("#artifactSubtitle")?.textContent.includes("dataset/ex1_infer.h5"),
      null, { timeout: 30000 });
      await page.locator('[data-close="artifactOverlay"]').click();
    }

    await openWorkspace("models");
    await page.waitForSelector('[data-model-row="mlp"]');
    await page.locator("#brandHome").click();
    assert(!(await page.locator("#studioOverlay").evaluate(element => element.classList.contains("open"))), "Brand Home did not close the Studio workspace");
    await openWorkspace("models");
    await page.locator("#studioPipeline").click();
    await page.waitForTimeout(50);
    const pipelineFocus = await page.evaluate(() => ({
      id: document.activeElement?.id || "",
      tag: document.activeElement?.tagName || "",
      studioOpen: document.querySelector("#studioOverlay")?.classList.contains("open")
    }));
    assert(pipelineFocus.id === "blockSearch", `Open block library did not return focus to search: ${JSON.stringify(pipelineFocus)}`);

    // Top-level Run/Stop and runtime drawer controls use a contained fake job;
    // the graph and all clicks are real, while no trainer or CUDA process starts.
    await page.locator("#templateSelect").selectOption("himgn");
    const preflightResponse = page.waitForResponse(response =>
      new URL(response.url()).pathname.endsWith("/api/preflight")
      && response.request().method() === "POST"
    );
    await page.locator("#validateTop").click();
    await preflightResponse;
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.api.activeJob?.id === "preflight");
    assert(requests.preflights.length > 0, "Validate did not submit contained authoritative preflight checks");
    await page.locator("#runtimeDismiss").click();
    const topPipelineCount = requests.pipelines.length;
    await page.locator("#runTop").click();
    await page.waitForFunction(expected => window.__AI_CAE_FRONTEND__.state.api.activeJob?.id === expected, `control-pipeline-${topPipelineCount + 1}`);
    await page.locator("#stopRun").click();
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.api.activeJob?.status === "cancelled");
    await page.locator("#runtimeCopyLog").click();
    assert((await page.evaluate(() => navigator.clipboard.readText())).includes("Cancellation requested"), "Runtime Copy log did not reach the clipboard");
    await page.locator("#runtimeExperiments").click();
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.studioSection === "experiments");
    await page.locator("#brandHome").click();
    await page.locator("#runtimeDismiss").click();
    await page.locator("#runTop").click();
    await page.waitForFunction(expected => window.__AI_CAE_FRONTEND__.state.api.activeJob?.id === expected, `control-pipeline-${topPipelineCount + 2}`);
    await page.locator("#runtimeCancel").click();
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.api.activeJob?.status === "cancelled");
    await page.locator("#runtimeDismiss").click();
    assert(requests.pipelines.length === topPipelineCount + 2, "Top-level Run did not submit two contained pipeline jobs");

    // Configuration actions, including visible file chooser/download, real
    // explain/save endpoints, and a mocked LLM transport that cannot leak data.
    await page.locator("#templateSelect").selectOption("blank");
    const mlpId = await addBlock("model.mlp");
    await page.locator(`[data-node-id="${mlpId}"] .node-head`).click();
    await page.locator("#openFullConfig").click();
    const configSection = page.locator("[data-config-section]").nth(1);
    const configSectionName = await configSection.getAttribute("data-config-section");
    await configSection.click();
    assert(await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.configSection) === configSectionName, "Config section navigation did not change the visible section");
    await page.locator("#changedOnly").check();
    await page.locator("#showInactive").uncheck();
    await page.locator("#showInactive").check();
    const configChooser = page.waitForEvent("filechooser");
    await page.locator("#loadTxt").click();
    await (await configChooser).setFiles({
      name: "control-mlp.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("model mlp\nmode train\ndataset_dir dataset/ex1_infer.h5\nresult_dir frontend/runtime/control-mlp\nbatch_size 4\n")
    });
    await page.waitForFunction(id =>
      window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.id === id)?.config.batch_size === "4",
    mlpId);
    await page.locator("#regenerateRaw").click();
    const configDownload = page.waitForEvent("download");
    await page.locator("#exportTxt").click();
    assert((await configDownload).suggestedFilename().endsWith(".txt"), "Export .txt did not download a flat config");
    await page.locator("#explainConfig").click();
    await page.waitForFunction(() => document.querySelector("#configDiagnostics")?.textContent.includes("Route:"));
    const llmCallsBefore = requests.llmConfigure.length;
    const llmResponse = page.waitForResponse(response =>
      new URL(response.url()).pathname.endsWith("/api/llm/configure")
      && response.request().method() === "POST"
    );
    await page.locator("#llmConfigure").click();
    await llmResponse;
    await page.waitForFunction(id =>
      window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.id === id)?.config.batch_size === "13",
    mlpId);
    assert(requests.llmConfigure.length === llmCallsBefore + 1, "Configure via LLM did not call the isolated endpoint");
    await page.locator("#saveConfig").click();
    await page.waitForFunction(() => !document.querySelector("#configOverlay")?.classList.contains("open"));

    await openWorkspace("models");
    await page.waitForSelector('[data-live-configs="mlp"]');
    await page.locator('[data-live-configs="mlp"]').click();
    await page.waitForSelector("#liveBackModels");
    await page.locator("#liveBackModels").click();
    await page.locator('[data-live-configs="mlp"]').click();
    await page.waitForSelector("[data-load-config]");
    await page.locator("[data-load-config]").first().click();
    await page.waitForFunction(() => document.querySelector("#configOverlay")?.classList.contains("open"));
    assert((await page.locator("#configDiagnostics").innerText()).includes("Loaded checked-in configuration"), "Load config did not open the checked-in example");
    await page.locator('#configOverlay [data-close="configOverlay"]').click();
    await openWorkspace("models");
    await page.locator('[data-model-details="mlp"]').click();
    await page.waitForSelector("#modelConfigSearch");
    await page.locator("#modelConfigSearch").fill("batch_size");
    assert(await page.locator(".model-config-row").count() === 1, "Model config search did not filter the full contract");
    await page.locator("#modelEditConfig").click();
    await page.waitForFunction(() => document.querySelector("#configOverlay")?.classList.contains("open"));
    await page.locator('#configOverlay [data-close="configOverlay"]').click();
    await openWorkspace("models");
    await page.locator('[data-model-details="mlp"]').click();
    const modelLlmBefore = requests.llmConfigure.length;
    const modelLlmResponse = page.waitForResponse(response =>
      new URL(response.url()).pathname.endsWith("/api/llm/configure")
      && response.request().method() === "POST"
    );
    await page.locator("#modelConfigureLlm").click();
    await modelLlmResponse;
    await page.waitForFunction(() =>
      window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.type === "model.mlp")?.config.batch_size === "13"
    );
    assert(requests.llmConfigure.length === modelLlmBefore + 1, "Model-detail LLM action did not call the isolated endpoint");
    await page.locator('#configOverlay [data-close="configOverlay"]').click();

    // System settings and the strict audit are isolated so no user credential
    // or full-repository audit state is changed by the control-surface test.
    await openWorkspace("system");
    await page.waitForSelector("#saveLlmSettings");
    await page.locator("#llmScheme").selectOption("http");
    await page.locator("#llmMasterIp").fill("127.0.0.1");
    await page.locator("#llmMasterPort").fill("10002");
    await page.locator("#llmUsername").fill("control-user");
    await page.locator("#llmPassword").fill("session-only-secret");
    await page.locator("#llmAllowHttp").check();
    await page.locator("#saveLlmSettings").click();
    await page.waitForFunction(() =>
      document.querySelector("#llmMasterIp")?.value === "127.0.0.1"
      && document.querySelector("#llmPassword")?.value === ""
      && !document.querySelector("#saveLlmSettings")?.disabled
    );
    assert(requests.llmSettings.length === 1 && requests.llmSettings[0].allow_insecure_http === true, "LLM settings did not submit every visible control");
    await page.locator("#auditStrict").check();
    const auditResponse = page.waitForResponse(response =>
      new URL(response.url()).pathname.endsWith("/api/audit-configs")
    );
    await page.locator("#runConfigAudit").click();
    await auditResponse;
    try {
      await page.waitForSelector("#auditSearch", { timeout: 5000 });
    } catch (error) {
      const auditState = await page.evaluate(() => ({
        results: document.querySelector("#auditResults")?.textContent || "",
        button: document.querySelector("#runConfigAudit")?.textContent || "",
        section: window.__AI_CAE_FRONTEND__?.state.studioSection || ""
      }));
      throw new Error(`Audit response did not render controls: ${JSON.stringify({ auditState, audits: requests.audits, browserErrors })}`);
    }
    await page.locator("#auditSearch").fill("broken");
    await page.locator("#auditFailuresOnly").check();
    await page.locator("[data-audit-detail]").click();
    assert((await page.locator("#auditDetail").innerText()).includes("CONTROL-FAIL"), "Audit diagnostics action did not open details");
    assert(requests.audits.at(-1) === "1", "Strict audit checkbox did not reach the endpoint");

    // Deployment controls submit complete bodies; expensive native PyInstaller
    // and inference commands are represented by auto-completing local jobs.
    await openWorkspace("deploy");
    await page.waitForSelector("#deployCheckpoint");
    await page.locator("#deployCheckpoint").evaluate(select => select.add(new Option("output/control/transolver.pth", "output/control/transolver.pth")));
    await page.locator("#deployCheckpoint").selectOption("output/control/transolver.pth");
    await page.waitForFunction(() => !document.querySelector("#runPortableInference")?.disabled);
    await page.locator("#deployInput").evaluate(select => {
      if (![...select.options].some(option => option.value === "dataset/ex1_infer.h5")) {
        select.add(new Option("dataset/ex1_infer.h5", "dataset/ex1_infer.h5"));
      }
    });
    await page.locator("#deployInput").selectOption("dataset/ex1_infer.h5");
    for (const [selector, value] of [
      ["#deployOutput", "control-portable"],
      ["#deployTimesteps", "1,3"],
      ["#deploySamples", "2"],
      ["#deployOdeSteps", "4"],
      ["#deployConditions", "0.1,0.2"]
    ]) {
      await page.locator(selector).fill(value);
      await page.locator(selector).dispatchEvent("change");
    }
    await page.locator("#runPortableInference").click();
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.api.activeJob?.id === "control-inference-1");
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.api.activeJob?.status === "completed", null, { timeout: 5000 });
    assert(requests.inference[0].timesteps === "1,3" && requests.inference[0].cond_values === "0.1,0.2", "Portable inference dropped runtime controls");
    await page.locator("#buildPortableExe").click();
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.api.activeJob?.id === "control-build-1");
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.api.activeJob?.status === "completed", null, { timeout: 5000 });
    assert(requests.builds === 1, "Build .exe did not submit its isolated job");

    // Force both pagination controls, then return to the real backend for HDF5
    // inspection, source upload, viewer controls, download, copy, and handoff.
    phase = "data-pagination";
    await openWorkspace("data");
    await page.waitForSelector("#liveFileShowMore");
    await page.locator("#liveFileShowMore").click();
    assert(await page.locator("#liveFileList .live-row").count() === 260, "Data Show more did not reveal the remaining catalog rows");
    await page.locator("#liveFileSearch").fill("catalog-0259");
    assert(await page.locator("#liveFileList .live-row").count() === 1, "Data search did not filter the loaded catalog");
    phase = "real-data";
    await openWorkspace("data");
    await page.waitForSelector("#liveFileSearch");
    await page.locator("#liveFileSearch").fill("ex1_infer.h5");
    await page.waitForSelector('[data-inspect-hdf5="dataset/ex1_infer.h5"]');
    await page.locator('[data-inspect-hdf5="dataset/ex1_infer.h5"]').click();
    await page.waitForSelector("#liveBackFiles");
    assert((await page.locator("#studioMain").innerText()).includes("dataset/ex1_infer.h5"), "HDF5 inspection did not show the selected file");
    await page.locator("#liveBackFiles").click();
    await page.locator("#liveFileSearch").fill("ex1_infer.h5");
    await page.locator('[data-use-file="dataset/ex1_infer.h5"]').click();
    await page.locator("#runtimeDismiss").click();
    const dataNodeId = await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.selectedNode);
    await page.locator(`[data-node-id="${dataNodeId}"] .node-head`).click();
    const sourceUploadChooser = page.waitForEvent("filechooser");
    await page.locator("#uploadInputSource").click();
    await (await sourceUploadChooser).setFiles(fixturePath);
    await page.waitForFunction(id => window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.id === id)?.config.path.includes("frontend/runtime/uploads/"), dataNodeId, { timeout: 30000 });
    const uploadedSource = await page.evaluate(id => window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.id === id)?.config.path, dataNodeId);
    await page.locator("#artifactStrip").click();
    try {
      await page.waitForFunction(expected => document.querySelector("#artifactSubtitle")?.textContent.includes(expected), uploadedSource, { timeout: 5000 });
    } catch (error) {
      const viewerState = await page.evaluate(() => ({
        overlay: document.querySelector("#artifactOverlay")?.classList.contains("open"),
        subtitle: document.querySelector("#artifactSubtitle")?.textContent || "",
        sampleInfo: document.querySelector("#sampleInfo")?.textContent || "",
        toasts: [...document.querySelectorAll(".toast")].map(item => item.textContent),
        realPath: window.__AI_CAE_FRONTEND__?.state.realArtifact?.path || "",
        studioOpen: document.querySelector("#studioOverlay")?.classList.contains("open")
      }));
      throw new Error(`Artifact Mini did not open uploaded source ${uploadedSource}: ${JSON.stringify({ viewerState, browserErrors })}`);
    }
    await page.locator('[data-close="artifactOverlay"]').click();
    await page.locator("#artifactMini").click();
    await page.waitForFunction(expected => document.querySelector("#artifactSubtitle")?.textContent.includes(expected), uploadedSource, { timeout: 30000 });
    await page.waitForSelector("[data-real-sample]");
    await page.locator("#artifactSampleSearch").fill("no-such-control-sample");
    assert(await page.locator("[data-real-sample]").count() === 0, "Artifact sample search did not filter samples");
    await page.locator("#artifactSampleSearch").fill("");
    await page.locator("[data-real-sample]").first().click();
    await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.realArtifact?.currentSample, null, { timeout: 30000 });
    if (await page.locator("#loadRealField").count()) {
      if (!(await page.locator("#realTimestep").isDisabled())) {
        await page.locator("#realTimestep").fill("1");
        await page.locator("#loadRealField").click();
        await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.realArtifact?.currentSample?.timestep === 1);
      } else {
        await page.locator("#loadRealField").click();
      }
      if (await page.locator("[data-load-feature]").count() > 1) {
        await page.locator("#realFeature").selectOption("1");
        await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.realArtifact?.currentSample?.feature === 1);
        await page.locator('[data-load-feature="0"]').click();
        await page.waitForFunction(() => window.__AI_CAE_FRONTEND__.state.realArtifact?.currentSample?.feature === 0);
      }
      if (!(await page.locator("#viewerPlay").isDisabled())) {
        const timeline = page.locator(".viewer-timeline input");
        const timelineMax = Number(await timeline.getAttribute("max"));
        const timelineCurrent = Number(await timeline.inputValue());
        assert(timelineMax > 0, "Temporal viewer fixture did not expose a usable timeline range");
        const timelineTarget = timelineCurrent === 0 ? timelineMax : 0;
        await timeline.press(timelineTarget === 0 ? "Home" : "End");
        await page.waitForFunction(expected =>
          window.__AI_CAE_FRONTEND__.state.realArtifact?.currentSample?.timestep === expected,
        timelineTarget, { timeout: 5000 });
        assert(Number(await page.locator(".viewer-timeline input").inputValue()) === timelineTarget, "Direct timeline range input did not render the selected timestep");
        const timelineBefore = await page.locator(".viewer-timeline input").inputValue();
        await page.locator("#viewerPlay").click();
        await page.waitForFunction(previous => document.querySelector(".viewer-timeline input")?.value !== previous, timelineBefore, { timeout: 5000 });
        await page.locator("#viewerPlay").click();
      }
    }
    const sampleDownload = page.waitForEvent("download");
    await page.locator("#artifactDownload").click();
    assert((await sampleDownload).suggestedFilename().endsWith(".json"), "Artifact Download did not produce sample JSON");
    await page.locator("#artifactCopyId").click();
    assert((await page.evaluate(() => navigator.clipboard.readText())).includes(uploadedSource), "Copy artifact ID did not copy the current source");
    await page.locator("#artifactAddDataset").click();
    await page.locator("#artifactDatasetSearch").fill("ex1_infer.h5");
    await page.locator("#artifactDatasetPickerClose").click();
    assert(await page.locator("#artifactDatasetPicker").evaluate(element => element.hidden), "Dataset picker Close did not hide the picker");
    await page.locator("#artifactAddDataset").click();
    const artifactUploadChooser = page.waitForEvent("filechooser");
    await page.locator("#artifactUploadDataset").click();
    await (await artifactUploadChooser).setFiles(fixturePath);
    await page.waitForFunction(() => document.querySelector("#artifactSubtitle")?.textContent.includes("frontend/runtime/uploads/"), null, { timeout: 30000 });
    await page.locator("#artifactUseInPipeline").click();
    assert(!(await page.locator("#artifactOverlay").evaluate(element => element.classList.contains("open"))), "Use in pipeline did not close the artifact viewer");

    phase = "docs";
    await openWorkspace("docs");
    await page.waitForSelector("#liveDocShowMore");
    await page.locator("#liveDocShowMore").click();
    assert(await page.locator("[data-open-doc]").count() === 260, "Docs Show more did not reveal the remaining documents");
    await page.locator("#liveDocSearch").fill("doc-0259");
    await page.locator("[data-open-doc]").click();
    await page.waitForSelector("#liveBackDocs");
    assert((await page.locator(".live-document").innerText()).includes("doc-0259"), "Open document did not render its text");
    await page.locator("#liveBackDocs").click();

    // Evaluation legacy controls, comparison, optimization, and export each
    // execute through isolated deterministic APIs and persist their evidence.
    await page.locator("#studioPipeline").click();
    const evaluationId = await addBlock("evaluate.predictions");
    await page.evaluate(id => {
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.id === id);
      Object.assign(node.config, { prediction_path: "dataset/ex1_infer.h5", truth_path: "dataset/ex1_infer.h5" });
    }, evaluationId);
    await page.locator(`[data-node-id="${evaluationId}"] .node-head`).click();
    await page.locator("#artifactMini").click();
    await page.waitForSelector("[data-evaluation-field]");
    const schemaCount = requests.evaluationSchemas.length;
    const schemaResponse = page.waitForResponse(response =>
      new URL(response.url()).pathname.endsWith("/api/evaluation/schema")
      && response.request().method() === "POST"
    );
    await page.locator("#refreshEvaluationSchema").click();
    await schemaResponse;
    assert(requests.evaluationSchemas.length > schemaCount, "Refresh contract did not repeat schema inspection");
    await page.locator(".evaluation-legacy summary").click();
    await page.locator("#evaluationUseLegacy").check();
    await page.locator("#evaluationPredictionStart").fill("1");
    await page.locator("#evaluationPredictionStart").dispatchEvent("change");
    await page.locator("#evaluationTruthStart").fill("2");
    await page.locator("#evaluationTruthStart").dispatchEvent("change");
    await page.locator("#evaluationFields").fill("2");
    await page.locator("#evaluationFields").dispatchEvent("change");
    assert(!(await page.locator("#runFieldEvaluation").isDisabled()), "Valid legacy mapping did not enable evaluation");
    await page.locator("#runFieldEvaluation").click();
    await page.waitForSelector("#exportEvaluation");
    assert(requests.evaluations.at(-1).prediction_start === 1 && requests.evaluations.at(-1).num_fields === 2, "Legacy evaluation dropped row controls");
    assert(!Object.hasOwn(requests.evaluations.at(-1), "field_pairs"), "Legacy evaluation incorrectly submitted schema pairs");
    await page.locator("#exportEvaluation").click();
    await page.waitForSelector("#exportPath");

    await page.locator("#studioPipeline").click();
    phase = "comparison";
    const comparisonId = await addBlock("evaluate.compare");
    await page.locator(`[data-node-id="${comparisonId}"] .node-head`).click();
    await page.locator("#artifactMini").click();
    await page.waitForSelector("#comparisonAddRun");
    await page.locator("#comparisonAddRun").click();
    const runSelects = page.locator("[data-run-select]");
    await runSelects.nth(0).selectOption("frontend/runtime/control-results/run-0.csv");
    await runSelects.nth(1).selectOption("frontend/runtime/control-results/run-1.csv");
    await page.waitForFunction(() => !document.querySelector("#runComparison")?.disabled);
    await page.locator("#comparisonGroup").fill("model");
    await page.locator("#comparisonGroup").dispatchEvent("change");
    await page.locator("#comparisonMetric").selectOption("relative_l2");
    await page.locator("#comparisonDirection").selectOption("max");
    await page.locator("#runComparison").click();
    await page.waitForSelector("#comparisonResults .live-summary");
    assert(requests.comparisons.at(-1).csv_paths.length === 2 && requests.comparisons.at(-1).direction === "max", "Comparison dropped selected runs or direction");
    assert((await page.locator("#comparisonResults").innerText()).includes("control-b"), "Comparison ranking was not rendered");
    await page.locator("[data-remove-run]").last().click();
    assert(await page.locator("[data-run-row]").count() === 1, "Remove comparison run did not remove its row");

    await page.locator("#studioPipeline").click();
    phase = "optimization";
    const optimizationId = await addBlock("optimize.design");
    await page.locator(`[data-node-id="${optimizationId}"] .node-head`).click();
    await page.locator("#artifactMini").click();
    await page.waitForSelector("#optimizationCsv");
    await page.locator("#optimizationCsv").selectOption("frontend/runtime/control-results/run-0.csv");
    await page.waitForSelector("[data-optimization-objective]");
    await page.locator("[data-optimization-objective]").first().check();
    await page.locator("[data-objective-direction]").first().selectOption("max");
    await page.locator("#optimizationConstraints").fill("mass <= 20");
    await page.locator("#optimizationConstraints").dispatchEvent("change");
    await page.locator("#optimizationTopK").fill("5");
    await page.locator("#optimizationTopK").dispatchEvent("change");
    await page.locator("#runOptimization").click();
    await page.waitForSelector("#optimizationResults .live-summary");
    assert(requests.optimizations.at(-1).directions === "max" && requests.optimizations.at(-1).top_k === 5, "Optimization dropped direction or top-k");
    const optimizationEvidence = await page.evaluate(id => {
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.id === id);
      return { report: node.config.report_path, constraints: node.config.constraints };
    }, optimizationId);
    assert(optimizationEvidence.report.endsWith("report.json") && optimizationEvidence.constraints === "mass <= 20", "Optimization evidence was not persisted on its block");

    await page.locator("#studioPipeline").click();
    phase = "export";
    const exportId = await addBlock("output.export");
    await page.locator(`[data-node-id="${exportId}"] .node-head`).click();
    await page.locator("#artifactMini").click();
    await page.waitForSelector("#exportPath");
    await page.locator("#exportPath").fill("dataset/ex1_infer.h5");
    await page.locator("#exportPath").dispatchEvent("change");
    await page.locator("#exportLabel").fill("control-surface-export");
    await page.locator("#exportLabel").dispatchEvent("change");
    await page.locator("#runExport").click();
    await page.waitForSelector("#exportResults a[download]", { timeout: 30000 });
    const exportEvidence = await page.evaluate(id => window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.id === id)?.config, exportId);
    assert(exportEvidence.export_path && exportEvidence.export_label === "control-surface-export", "Export result was not persisted on its block");

    // Capability cards are the non-live overview behind each workspace. Render
    // one overview deliberately, click its real pipeline action, then exercise
    // every top-navigation item as a user would.
    await page.evaluate(async () => {
      const studio = await import("./src/studio.js");
      studio.activateStudioWorkspace("data");
    });
    await page.waitForSelector('[data-capability-block="source.hdf5"]');
    await page.locator('[data-capability-block="source.hdf5"]').first().click();
    assert(await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.nodes.some(node => node.type === "source.hdf5")), "Capability action did not select or add its pipeline block");
    for (const section of ["models", "data", "experiments", "optimization", "benchmarks", "artifacts", "deploy", "system", "docs"]) {
      await page.locator(`[data-section="${section}"]`).click();
      await page.waitForFunction(expected => window.__AI_CAE_FRONTEND__.state.studioSection === expected, section);
      await page.locator("#brandHome").click();
    }
    await page.locator('[data-section="pipeline"]').click();
    assert(!(await page.locator("#studioOverlay").evaluate(element => element.classList.contains("open"))), "Pipeline navigation did not return to the canvas");

    assert(dialogMessages.some(item => item.type === "prompt"), "No visible LLM prompt was exercised");
    assert(dialogMessages.filter(item => item.type === "confirm").length >= 6, "Destructive/expensive confirmations were not exercised");
    assert(browserErrors.length === 0, `Browser errors: ${browserErrors.join(" | ")}`);
    console.log("PASS: editor, runtime, config, models, system, deploy, catalogs, viewer, evaluation, comparison, optimization, and export controls were clicked end to end");
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
