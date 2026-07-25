const playwrightModule = process.argv[2] || "playwright";
const { chromium } = require(playwrightModule);
const path = require("path");

const studioUrl = process.argv[3] || "http://127.0.0.1:8080/index.html";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function openNodeCatalog(page, nodeId, expectedPath) {
  await page.evaluate(id => window.__AI_CAE_FRONTEND__.openArtifact(id), nodeId);
  await page.waitForFunction(
    expected => document.querySelector("#artifactSubtitle")?.textContent.includes(expected),
    expectedPath,
    { timeout: 30000 }
  );
  await page.waitForFunction(
    () => document.querySelectorAll("[data-real-sample]").length > 0,
    null,
    { timeout: 30000 }
  );
  assert(
    (await page.locator(".sample-item.active").count()) === 0,
    `${expectedPath} selected a default sample`
  );
  assert(
    (await page.locator("#viewerVisual svg").count()) === 0,
    `${expectedPath} rendered before explicit sample selection`
  );
  assert(
    await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.realArtifact.currentSample === null),
    `${expectedPath} loaded a current sample implicitly`
  );
  assert(
    await page.locator("#artifactDownload").evaluate(element => element.disabled),
    `${expectedPath} enabled sample download without a selected sample`
  );
}

async function selectSample(page, index = 0) {
  await page.locator(`[data-real-sample="${index}"]`).click();
  await page.waitForFunction(
    () => document.querySelector("#viewerVisual svg")
      && window.__AI_CAE_FRONTEND__.state.realArtifact.currentSample,
    null,
    { timeout: 30000 }
  );
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
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));

    await page.locator("#templateSelect").selectOption("geometry");
    await openNodeCatalog(page, "cad", "processed-car-pressure-data/data");
    await page.screenshot({ path: path.join(__dirname, "runtime", "viewer-no-default-sample.png"), fullPage: false });
    await selectSample(page);
    assert(
      await page.locator("#viewerVisual svg polygon").count() > 0,
      "CAD preview did not render real mesh faces"
    );
    assert(
      await page.locator('[data-view-mode="mesh"]').evaluate(element => element.classList.contains("active")),
      "CAD preview did not select Mesh mode"
    );
    assert(
      await page.locator('[data-view-mode="field"]').evaluate(element => element.disabled),
      "CAD preview incorrectly exposes a scalar field"
    );
    assert(
      (await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.realArtifact.source_kind)) === "geometry",
      "CAD preview resolved to a non-geometry artifact"
    );
    const viewport = await page.locator("#viewerVisual").boundingBox();
    assert(viewport, "CAD viewport has no browser bounds");
    const center = { x: viewport.x + viewport.width / 2, y: viewport.y + viewport.height / 2 };
    await page.mouse.move(center.x, center.y);
    await page.mouse.down({ button: "left" });
    await page.mouse.move(center.x + 80, center.y + 35, { steps: 4 });
    await page.mouse.up({ button: "left" });
    const rotated = await page.evaluate(() => ({ ...window.__AI_CAE_FRONTEND__.state.viewerCamera }));
    assert(Math.abs(rotated.yaw) > 0.1 && Math.abs(rotated.pitch) > 0.1, "Left-drag did not rotate the CAD camera");
    await page.mouse.move(center.x, center.y);
    await page.mouse.down({ button: "right" });
    await page.mouse.move(center.x + 55, center.y - 25, { steps: 4 });
    await page.mouse.up({ button: "right" });
    const panned = await page.evaluate(() => ({ ...window.__AI_CAE_FRONTEND__.state.viewerCamera }));
    assert(Math.abs(panned.panX) > 1 && Math.abs(panned.panY) > 1, "Right-drag did not pan the CAD camera");
    await page.mouse.move(center.x, center.y);
    await page.mouse.wheel(0, -300);
    const zoomed = await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.viewerCamera.zoom);
    assert(zoomed > 1, "Mouse wheel did not zoom the CAD camera");
    await page.locator("#viewerReset").click();
    const reset = await page.evaluate(() => ({ ...window.__AI_CAE_FRONTEND__.state.viewerCamera }));
    assert(
      reset.yaw === 0 && reset.pitch === 0 && reset.zoom === 1 && reset.panX === 0 && reset.panY === 0,
      "Reset view did not restore the camera"
    );
    await page.screenshot({ path: path.join(__dirname, "runtime", "viewer-cad-fixed.png"), fullPage: false });
    await page.locator('[data-close="artifactOverlay"]').click();

    await page.locator("#templateSelect").selectOption("simulgen");
    await openNodeCatalog(page, "dataset", "dataset/ex1.h5");
    await selectSample(page);
    assert(
      await page.locator("#viewerVisual svg line").count() > 0,
      "Mesh HDF5 preview did not render real mesh_edge topology"
    );
    assert(
      await page.locator('[data-view-mode="field"]').evaluate(element => element.classList.contains("active")),
      "Mesh HDF5 preview did not select Field mode"
    );
    await page.screenshot({ path: path.join(__dirname, "runtime", "viewer-hdf5-mesh-fixed.png"), fullPage: false });

    await page.locator("#artifactAddDataset").click();
    await page.locator("#artifactDatasetPicker:not([hidden])").waitFor();
    await page.locator("#artifactDatasetSearch").fill("deepjeb.h5");
    await page.locator('[data-preview-dataset="dataset/deepjeb.h5"]').waitFor();
    await page.screenshot({ path: path.join(__dirname, "runtime", "viewer-add-dataset.png"), fullPage: false });
    await page.locator('[data-preview-dataset="dataset/deepjeb.h5"]').click();
    await page.waitForFunction(
      () => document.querySelector("#artifactSubtitle")?.textContent.includes("dataset/deepjeb.h5")
        && document.querySelector("#artifactDatasetPicker")?.hidden,
      null,
      { timeout: 30000 }
    );
    assert(
      await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.realArtifact.currentSample === null),
      "Add dataset implicitly selected a DeepJEB sample"
    );
    assert((await page.locator(".sample-item.active").count()) === 0, "Added dataset has a default sample");
    await selectSample(page);
    assert(
      await page.locator("#viewerVisual svg circle").count() > 100,
      "SDFFlow shapes/{id} preview did not render the surface point cloud"
    );
    assert(
      await page.locator('[data-view-mode="points"]').evaluate(element => element.classList.contains("active")),
      "SDFFlow point-cloud preview did not select Points mode"
    );
    assert(
      await page.locator('[data-view-mode="field"]').evaluate(element => element.disabled),
      "SDFFlow surface preview incorrectly exposes a scalar field"
    );
    await page.screenshot({ path: path.join(__dirname, "runtime", "viewer-hdf5-sdf-fixed.png"), fullPage: false });

    await page.evaluate(() => {
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.id === "dataset");
      node.config.path = "dataset/benchmarks/deeponet_fractional2d/fractional2d_released.h5";
    });
    await openNodeCatalog(page, "dataset", "fractional2d_released.h5");
    await selectSample(page);
    assert(
      await page.locator("#viewerVisual svg circle").count() === 225,
      "Operator-grid HDF5 preview did not render all query points"
    );

    await page.evaluate(() => {
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.id === "dataset");
      node.config.path = "dataset/mlp/train.h5";
    });
    await openNodeCatalog(page, "dataset", "dataset/mlp/train.h5");
    await selectSample(page);
    assert(
      await page.locator("#viewerVisual svg polyline").count() === 1,
      "Table HDF5 preview did not render the normalized row series"
    );

    assert(browserErrors.length === 0, `Browser reported errors: ${browserErrors.join(" | ")}`);
    console.log("PASS: explicit sample selection, add-dataset flow, camera controls, and shared CAD/HDF5 rendering");
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(`FAIL: ${error.message}`);
  process.exitCode = 1;
});
