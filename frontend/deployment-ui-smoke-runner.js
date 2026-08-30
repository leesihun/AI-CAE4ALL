const { chromium } = require(process.argv[2] || "playwright");

const studioUrl = process.argv[3] || "http://127.0.0.1:8080/index.html";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1366, height: 768 } });
  await page.addInitScript(() => localStorage.setItem("ai-cae4all.studio.welcomed.v1", "1"));
  await page.route("**/api/checkpoint?*", async route => {
    const path = new URL(route.request().url()).searchParams.get("path") || "";
    const isChi = path.includes("chi_mgnflow");
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        ok: true,
        path,
        model: isChi ? "chi-mgnflow" : "transolver",
        model_source: "checkpoint metadata",
        standalone_inference: true,
        portable_inference: !isChi,
        model_config: {}
      })
    });
  });
  try {
    await page.goto(studioUrl);
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));
    await page.locator("#templateSelect").selectOption("blank");
    await page.locator('.palette-item[data-block-type="source.checkpoint"]').click();
    await page.locator('.palette-item[data-block-type="source.hdf5"]').click();
    await page.locator('.palette-item[data-block-type="deploy.api"]').click();
    const ids = await page.evaluate(() => Object.fromEntries(
      window.__AI_CAE_FRONTEND__.state.nodes.map(node => [node.type, node.id])
    ));
    await page.locator(`[data-node="${ids["source.checkpoint"]}"][data-port="model"][data-direction="output"]`).click();
    await page.locator(`[data-node="${ids["deploy.api"]}"][data-port="model"][data-direction="input"]`).click();
    await page.locator(`[data-node="${ids["source.hdf5"]}"][data-port="data"][data-direction="output"]`).click();
    await page.locator(`[data-node="${ids["deploy.api"]}"][data-port="data"][data-direction="input"]`).click();
    await page.evaluate(({ checkpointId, dataId }) => {
      const frontend = window.__AI_CAE_FRONTEND__;
      frontend.state.nodes.find(node => node.id === checkpointId).config.path = "output/transolver/example.pth";
      frontend.state.nodes.find(node => node.id === dataId).config.path = "dataset/ex1_infer.h5";
      frontend.applyGraphAutofill();
    }, { checkpointId: ids["source.checkpoint"], dataId: ids["source.hdf5"] });
    await page.locator(`[data-node-id="${ids["deploy.api"]}"] .node-head`).click();
    const keys = await page.locator(".inspector-config").evaluateAll(controls => controls.map(control => control.dataset.key));
    for (const key of ["checkpoint_path", "input_path", "output_name", "timesteps", "num_samples", "ode_steps", "cond_values"]) {
      assert(keys.includes(key), `Deploy inspector is missing live field ${key}`);
    }
    for (const dead of ["target", "device", "auth", "openapi"]) {
      assert(!keys.includes(dead), `Deploy inspector still advertises inert field ${dead}`);
    }
    await page.locator("#inspectorRun").click();
    await page.waitForSelector("#runPortableInference");
    assert(await page.locator("#deployCheckpoint").inputValue() === "output/transolver/example.pth", "Connected checkpoint did not reach deployment workspace");
    assert(await page.locator("#deployInput").inputValue() === "dataset/ex1_infer.h5", "Connected HDF5 did not reach deployment workspace");
    await page.waitForFunction(() => !document.querySelector("#runPortableInference")?.disabled);
    assert((await page.locator("#deployCheckpointWarning").textContent()).includes("transolver"), "Supported checkpoint metadata was not shown");
    await page.locator("#deployCheckpoint").evaluate(select => {
      select.add(new Option("output/chi_mgnflow/best.pth", "output/chi_mgnflow/best.pth"));
    });
    await page.locator("#deployCheckpoint").selectOption("output/chi_mgnflow/best.pth");
    await page.waitForFunction(() => document.querySelector("#deployCheckpointWarning")?.textContent.includes("cHI-MGNflow model"));
    assert(await page.locator("#runPortableInference").isDisabled(), "Unsupported cHI checkpoint was still runnable through the portable bundle");
    assert((await page.locator("#deployCheckpointWarning").textContent()).includes("native Inference block"), "cHI portable rejection did not give the native alternative");
    await page.locator("#deployCheckpoint").selectOption("output/transolver/example.pth");
    await page.waitForFunction(() => !document.querySelector("#runPortableInference")?.disabled);
    await page.locator("#deployOutput").fill("portable-smoke-output");
    await page.locator("#deployOutput").dispatchEvent("change");
    assert(await page.evaluate(id =>
      window.__AI_CAE_FRONTEND__.state.nodes.find(node => node.id === id).config.output_name,
      ids["deploy.api"]
    ) === "portable-smoke-output", "Deployment workspace edit did not persist to its block");
    console.log("PASS: deployment graph autofill, checkpoint verification, and cHI portable/native boundary share one live contract");
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
