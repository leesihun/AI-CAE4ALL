const playwrightModule = process.argv[2] || "playwright";
const { chromium } = require(playwrightModule);

const studioUrl = process.argv[3] || "http://127.0.0.1:8080/index.html";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

function delay(milliseconds) {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}

async function waitFor(predicate, message, timeout = 5000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await delay(25);
  }
  throw new Error(message);
}

function catalog(path) {
  return {
    path,
    source_kind: "hdf5",
    default_mode: "field",
    feature_names: ["value"],
    condition_names: [],
    samples: [0, 1].map(index => ({
      id: String(index),
      label: `sample ${index}`,
      default_feature: 0,
      datasets: [{ name: "nodal_data", shape: [3, 3, 1] }]
    }))
  };
}

function samplePayload(path, sample, timestep) {
  const offset = Number(sample) * 10 + Number(timestep);
  return {
    path,
    sample: String(sample),
    dataset: "nodal_data",
    source_kind: "hdf5",
    preview_kind: "field",
    shape: [3, 3, 1],
    total_points: 3,
    returned_points: 3,
    feature: 0,
    feature_name: "value",
    feature_names: ["value"],
    feature_count: 1,
    timestep: Number(timestep),
    timestep_count: 3,
    x: [0, 1, 0],
    y: [0, 0, 1],
    z: [0, 0, 0],
    values: [offset, offset + 1, offset + 2],
    supports: { field: true, points: true, mesh: false },
    stats: { min: offset, max: offset + 2, mean: offset + 1 },
    mesh: null,
    parameters: [],
    metadata: {}
  };
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  await page.addInitScript(() => {
    try { localStorage.setItem("ai-cae4all.studio.welcomed.v1", "1"); } catch { /* storage blocked */ }
  });

  const firstCatalogSeen = deferred();
  const releaseFirstCatalog = deferred();
  const firstSampleSeen = deferred();
  const releaseFirstSample = deferred();
  let delayedCatalog = false;
  let delayedSample = false;
  let trackPlayback = false;
  let activePlaybackRequests = 0;
  let maxPlaybackRequests = 0;
  let playbackRequests = 0;
  const playbackStarts = [];
  const playbackEnds = [];

  await page.route("**/api/preview/samples?*", async route => {
    const url = new URL(route.request().url());
    const path = url.searchParams.get("path");
    if (!["dataset/race-a.h5", "dataset/race-b.h5"].includes(path)) {
      await route.continue();
      return;
    }
    if (path === "dataset/race-a.h5" && !delayedCatalog) {
      delayedCatalog = true;
      firstCatalogSeen.resolve();
      await releaseFirstCatalog.promise;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(catalog(path))
    });
  });

  await page.route("**/api/preview/sample?*", async route => {
    const url = new URL(route.request().url());
    const path = url.searchParams.get("path");
    if (path !== "dataset/race-b.h5") {
      await route.continue();
      return;
    }
    const sample = url.searchParams.get("sample");
    const timestep = Number(url.searchParams.get("timestep") || 0);
    if (sample === "0" && !delayedSample) {
      delayedSample = true;
      firstSampleSeen.resolve();
      await releaseFirstSample.promise;
    }
    if (trackPlayback) {
      playbackRequests += 1;
      activePlaybackRequests += 1;
      maxPlaybackRequests = Math.max(maxPlaybackRequests, activePlaybackRequests);
      playbackStarts.push(Date.now());
      await delay(650);
      activePlaybackRequests -= 1;
      playbackEnds.push(Date.now());
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(samplePayload(path, sample, timestep))
    });
  });

  const browserErrors = [];
  page.on("pageerror", error => browserErrors.push(`pageerror: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error") browserErrors.push(`console: ${message.text()}`);
  });

  try {
    await page.goto(studioUrl);
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));
    await page.evaluate(() => window.__AI_CAE_FRONTEND__.loadTemplate("simulgen", false));

    await page.evaluate(() => {
      const frontend = window.__AI_CAE_FRONTEND__;
      frontend.state.nodes.find(node => node.id === "dataset").config.path = "dataset/race-a.h5";
      window.__viewerRaceFirstCatalog = frontend.openArtifact("dataset");
    });
    await firstCatalogSeen.promise;
    await page.evaluate(async () => {
      const frontend = window.__AI_CAE_FRONTEND__;
      frontend.state.nodes.find(node => node.id === "dataset").config.path = "dataset/race-b.h5";
      await frontend.openArtifact("dataset");
    });
    releaseFirstCatalog.resolve();
    await page.evaluate(() => window.__viewerRaceFirstCatalog);
    assert(
      await page.evaluate(() => window.__AI_CAE_FRONTEND__.state.realArtifact?.path) === "dataset/race-b.h5",
      "A stale artifact catalog replaced the newer selection"
    );

    await page.evaluate(async () => {
      const viewer = await import("./src/viewer.js");
      window.__viewerRaceFirstSample = viewer.renderRealArtifactSample(0, null, 0);
    });
    await firstSampleSeen.promise;
    await page.evaluate(async () => {
      const viewer = await import("./src/viewer.js");
      await viewer.renderRealArtifactSample(1, null, 0);
    });
    releaseFirstSample.resolve();
    await page.evaluate(() => window.__viewerRaceFirstSample);
    const selected = await page.evaluate(() => ({
      index: window.__AI_CAE_FRONTEND__.state.artifactSample,
      sample: window.__AI_CAE_FRONTEND__.state.realArtifact?.currentSample?.sample
    }));
    assert(selected.index === 1 && selected.sample === "1", "A stale sample response replaced the newer sample");

    trackPlayback = true;
    await page.evaluate(async () => {
      const viewer = await import("./src/viewer.js");
      viewer.toggleViewerPlayback();
    });
    await waitFor(
      () => playbackRequests >= 2 && activePlaybackRequests === 0,
      "Playback did not complete two delayed frames",
      6000
    );
    await page.evaluate(async () => {
      const viewer = await import("./src/viewer.js");
      viewer.stopViewerPlayback();
    });
    assert(maxPlaybackRequests === 1, `Playback overlapped ${maxPlaybackRequests} sample requests`);
    assert(
      playbackStarts[1] - playbackEnds[0] >= 400,
      "Playback scheduled the next frame before the previous render settled"
    );
    assert(browserErrors.length === 0, `Browser reported errors: ${browserErrors.join(" | ")}`);
    console.log("PASS: stale artifact/sample responses ignored and playback frames serialized");
  } finally {
    releaseFirstCatalog.resolve();
    releaseFirstSample.resolve();
    await browser.close();
  }
})().catch(error => {
  console.error(`FAIL: ${error.message}`);
  process.exitCode = 1;
});
