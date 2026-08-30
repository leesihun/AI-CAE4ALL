const { chromium } = require(process.argv[2] || "playwright");

const studioUrl = process.argv[3] || "http://127.0.0.1:8080/index.html";
const checkpointPath = "output/chi-mgnflow/ex1_smoke/ex1_smoke.pth";
const datasetPath = "dataset/ex1_infer.h5";
const outputDir = `../frontend/runtime/chi-native-inference-${Date.now()}`;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function chooseRepositoryFile(page, nodeId, query, expectedPath) {
  await page.locator(`[data-node-id="${nodeId}"] .node-head`).click();
  await page.locator("#browseInputSource").click();
  await page.waitForSelector("#inputPickerSearch");
  await page.locator("#inputPickerSearch").fill(query);
  const choice = page.locator(`[data-use-input="${expectedPath}"]`);
  await choice.waitFor();
  await choice.click();
  await page.waitForFunction(() => !document.querySelector("#studioOverlay")?.classList.contains("open"));
}

async function setInspectorControl(page, key, value) {
  const control = page.locator(`.inspector-config[data-key="${key}"]`);
  assert(await control.count() === 1, `Inference Inspector is missing ${key}`);
  const tag = await control.evaluate(element => element.tagName);
  if (tag === "SELECT") await control.selectOption(value);
  else await control.fill(value);
  await control.dispatchEvent("change");
  await page.waitForFunction(
    ({ key: field, value: expected }) => {
      const node = window.__AI_CAE_FRONTEND__.state.nodes.find(item => item.type === "run.inference");
      return String(node?.config?.[field] ?? "") === expected;
    },
    { key, value }
  );
}

async function jobDetail(page, jobId) {
  const response = await page.request.get(`${new URL(studioUrl).origin}/api/jobs/${encodeURIComponent(jobId)}`);
  assert(response.ok(), `Job status request failed with HTTP ${response.status()}`);
  return response.json();
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const context = await browser.newContext({ viewport: { width: 1600, height: 960 } });
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

    // Build the three-block checkpoint-only graph using the same clicks a user
    // uses. The native route must not depend on keeping the training block on
    // the canvas after a checkpoint has been saved.
    await page.locator("#templateSelect").selectOption("blank");
    await page.locator('.palette-item[data-block-type="source.checkpoint"]').click();
    await page.locator('.palette-item[data-block-type="source.hdf5"]').click();
    await page.locator('.palette-item[data-block-type="run.inference"]').click();
    const ids = await page.evaluate(() => Object.fromEntries(
      window.__AI_CAE_FRONTEND__.state.nodes.map(node => [node.type, node.id])
    ));

    await page.locator(`[data-node="${ids["source.checkpoint"]}"][data-port="model"][data-direction="output"]`).click();
    await page.locator(`[data-node="${ids["run.inference"]}"][data-port="model"][data-direction="input"]`).click();
    await page.locator(`[data-node="${ids["source.hdf5"]}"][data-port="data"][data-direction="output"]`).click();
    await page.locator(`[data-node="${ids["run.inference"]}"][data-port="data"][data-direction="input"]`).click();

    await chooseRepositoryFile(page, ids["source.checkpoint"], "ex1_smoke.pth", checkpointPath);
    await chooseRepositoryFile(page, ids["source.hdf5"], "ex1_infer.h5", datasetPath);

    // Metadata is read from the real 3.5 MB checkpoint. Wait for its async
    // authority to reach the connected Inference block before configuring it.
    await page.waitForFunction(
      ({ inferenceId, checkpoint, dataset }) => {
        const app = window.__AI_CAE_FRONTEND__;
        const node = app.state.nodes.find(item => item.id === inferenceId);
        const facts = app.state.checkpointMeta.get(checkpoint);
        return node?.config.model_id === "chi-mgnflow"
          && node.config.checkpoint_path === checkpoint
          && node.config.dataset_path === dataset
          && facts?.ok === true
          && facts.model === "chi-mgnflow";
      },
      { inferenceId: ids["run.inference"], checkpoint: checkpointPath, dataset: datasetPath },
      { timeout: 45_000 }
    );

    await page.locator(`[data-node-id="${ids["run.inference"]}"] .node-head`).click();
    for (const [key, value] of [
      ["gpu_ids", "-1"],
      ["infer_timesteps", "1"],
      ["inference_output_dir", outputDir],
      ["batch_size", "1"],
      ["num_workers", "0"],
      ["num_vae_samples", "1"],
      ["vae_batch_size", "1"],
      ["flow_steps", "1"],
      ["flow_solver", "euler"],
      ["flow_predict", "mean"]
    ]) {
      await setInspectorControl(page, key, value);
    }

    const serialized = await page.evaluate(async inferenceId => {
      const { executableSteps } = await import("./src/validate.js");
      return executableSteps(inferenceId)[0] || null;
    }, ids["run.inference"]);
    assert(serialized, "The checkpoint-only Inference block produced no executable step");
    assert(serialized.label.includes("cHI-MGNflow"), `Wrong native route label: ${serialized.label}`);
    for (const line of [
      "model                        chi-mgnflow",
      "mode                         inference",
      "gpu_ids                      -1",
      "modelpath                    ../output/chi-mgnflow/ex1_smoke/ex1_smoke.pth",
      "infer_dataset                ../dataset/ex1_infer.h5",
      "infer_timesteps              1",
      `inference_output_dir         ${outputDir}`,
      "num_vae_samples              1",
      "vae_batch_size               1",
      "flow_steps                   1",
      "flow_solver                  euler",
      "flow_predict                 mean"
    ]) {
      assert(serialized.config.split(/\r?\n/).includes(line), `Native config omitted or changed: ${line}`);
    }

    const responsePromise = page.waitForResponse(response =>
      response.url().endsWith("/api/pipeline/run") && response.request().method() === "POST",
      { timeout: 60_000 }
    );
    await page.locator("#inspectorRun").click();
    const response = await responsePromise;
    const submitted = await response.json();
    assert(response.status() === 201, `Pipeline submission returned HTTP ${response.status()}: ${JSON.stringify(submitted)}`);
    submittedJob = submitted.id;
    assert(submitted.steps?.length === 1, "The selected Inference block did not submit exactly one native step");
    assert(submitted.steps[0].route?.model === "chi-mgnflow", `Submission resolved the wrong route: ${JSON.stringify(submitted.steps[0].route)}`);
    assert(confirmations.some(message => message.includes("Execute the real AI-CAE4ALL launcher")), "The destructive execution confirmation was not shown");

    let job = submitted;
    const deadline = Date.now() + 10 * 60_000;
    while (["queued", "running"].includes(job.status) && Date.now() < deadline) {
      await page.waitForTimeout(1_000);
      job = await jobDetail(page, submitted.id);
    }
    assert(!["queued", "running"].includes(job.status), `Native cHI inference did not finish within 10 minutes (job ${job.id})`);
    assert(job.status === "completed", `Native cHI inference ${job.status}:\n${job.log || "(no log)"}`);
    assert(job.returncode === 0, `Native cHI process returned ${job.returncode}`);
    assert(job.steps?.[0]?.route?.model === "chi-mgnflow", "Launch-time preflight changed the model family");
    assert(job.steps?.[0]?.launch_preflight?.ok === true, "Exact saved-config launch preflight did not pass");
    assert(job.log.includes("Using device: cpu"), "The run did not honor the GUI's CPU selection");
    assert(job.log.includes("Flow readout: DETERMINISTIC mean"), "The run did not honor deterministic mean flow readout");
    assert(job.log.includes("Rollout inference complete"), "The native cHI rollout did not reach completion");
    assert(job.steps?.[0]?.results?.startsWith("frontend/runtime/chi-native-inference-"), `The runtime did not publish the configured result directory: ${job.steps?.[0]?.results}`);
    assert(Number(job.steps?.[0]?.results_samples) > 0, "The native run published no HDF5 result files");

    await page.waitForFunction(
      ({ inferenceId, jobId }) => {
        const app = window.__AI_CAE_FRONTEND__;
        const node = app.state.nodes.find(item => item.id === inferenceId);
        return app.state.api.activeJob?.id === jobId
          && app.state.api.activeJob.status === "completed"
          && node?.status === "complete"
          && String(node.config.results_path || "").startsWith("frontend/runtime/chi-native-inference-");
      },
      { inferenceId: ids["run.inference"], jobId: job.id },
      { timeout: 15_000 }
    );

    // The result path written back onto the block must be consumable by the
    // same GUI, not merely present on disk. Open Samples, choose the produced
    // rollout, and require a real mesh/field draw from its native HDF5 arrays.
    await page.locator("#inspectorSamples").click();
    await page.waitForFunction(
      expected => document.querySelector("#artifactSubtitle")?.textContent.includes(expected),
      job.steps[0].results,
      { timeout: 30_000 }
    );
    await page.waitForFunction(() => document.querySelectorAll("[data-real-sample]").length > 0, null, { timeout: 30_000 });
    await page.locator("[data-real-sample]").first().click();
    await page.waitForFunction(() => {
      const app = window.__AI_CAE_FRONTEND__;
      return document.querySelector("#viewerVisual .viewer-canvas:not([hidden])")
        && app.state.realArtifact.currentSample
        && app.state.viewerDraw?.drewFaces;
    }, null, { timeout: 45_000 });
    const viewed = await page.evaluate(() => {
      const app = window.__AI_CAE_FRONTEND__;
      const sample = app.state.realArtifact.currentSample;
      return {
        points: sample?.geometry?.points?.length || sample?.points?.length || 0,
        edges: sample?.mesh?.returned_edges || 0,
        faces: sample?.mesh?.returned_faces || 0,
        drew_faces: Boolean(app.state.viewerDraw?.drewFaces)
      };
    });
    assert(viewed.edges > 0 && viewed.faces > 0 && viewed.drew_faces, `Native result viewer did not draw mesh fields: ${JSON.stringify(viewed)}`);
    await page.locator('[data-close="artifactOverlay"]').click();

    assert(browserErrors.length === 0, `Browser reported errors: ${browserErrors.join(" | ")}`);
    console.log(JSON.stringify({
      ok: true,
      job_id: job.id,
      route: job.steps[0].route,
      config_path: job.steps[0].config_path,
      results: job.steps[0].results,
      results_samples: job.steps[0].results_samples,
      device: "cpu",
      flow_predict: "mean",
      viewer: viewed
    }, null, 2));
    console.log("PASS: actual Chrome clicks selected real files, resolved cHI-MGNflow, and completed native CPU inference");
  } catch (error) {
    if (submittedJob) {
      try {
        const job = await jobDetail(page, submittedJob);
        console.error(`JOB ${submittedJob} ${job.status}:\n${job.log || "(no log)"}`);
      } catch { /* preserve the original failure */ }
    }
    throw error;
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
