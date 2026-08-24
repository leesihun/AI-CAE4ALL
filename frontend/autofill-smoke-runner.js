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

  try {
    await page.goto(studioUrl);
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));

    const first = await page.evaluate(() => {
      const app = window.__AI_CAE_FRONTEND__;
      const make = (id, type, config = {}) => ({
        id, type, x: 50, y: 50, status: "idle", progress: 0,
        config: { ...app.BLOCK_SPECS[type].defaults, ...config }
      });
      const table = JSON.stringify({
        version: 2,
        columns: [
          { id: "input_1", kind: "input", name: "length" },
          { id: "input_2", kind: "input", name: "width" },
          { id: "input_3", kind: "input", name: "load" },
          { id: "output_1", kind: "output", name: "mass" },
          { id: "output_2", kind: "output", name: "stress" }
        ],
        rows: []
      });
      app.state.nodes = [
        make("dataset", "source.hdf5", { path: "dataset/custom/train.h5", feature_names: "ux, uy" }),
        make("parameters", "source.parameters", { binding: "dataset/custom/conditions.csv", parameter_table: table }),
        make("checkpoint", "source.checkpoint", { path: "output/mlp/best.pth" }),
        make("mlp", "model.mlp", { mode: "train" }),
        make("inference", "run.inference", { output_path: "output/predictions.h5" }),
        make("evaluation", "evaluate.predictions", { report_path: "frontend/runtime/evaluation/report.json" }),
        make("export", "output.export"),
        make("generator", "run.cad_generator", { output_csv: "frontend/runtime/candidates.csv" }),
        make("optimization", "optimize.design"),
        make("deploy", "deploy.api"),
        make("cad", "source.cad", { path: "dataset/cad/bracket.step" }),
        make("prep", "prep.geometry"),
        make("sim_params", "source.parameters", { binding: "dataset/simul/conditions.csv", condition_names: "force, angle" }),
        make("sim_checkpoint", "source.checkpoint", { path: "output/simul/simulgen_vae.pth" }),
        make("simulgen", "model.simulgenvae", { mode: "train_lc" }),
        make("sdf_params", "source.parameters", { binding: "dataset/sdf/conditions.csv", condition_names: "bbox_x, volume" }),
        make("sdf_checkpoint", "source.checkpoint", { path: "output/sdf/fm_best.pth" }),
        make("sdfflow", "model.sdfflow", { mode: "train_fm" })
      ];
      const edge = (id, fromNode, fromPort, toNode, toPort) => ({ id, fromNode, fromPort, toNode, toPort });
      app.state.edges = [
        edge("e1", "dataset", "data", "mlp", "data"),
        edge("e2", "parameters", "parameters", "mlp", "parameters"),
        edge("e3", "checkpoint", "model", "mlp", "resume"),
        edge("e4", "dataset", "data", "inference", "data"),
        edge("e5", "mlp", "model", "inference", "model"),
        edge("e6", "parameters", "parameters", "inference", "parameters"),
        edge("e7", "inference", "prediction", "evaluation", "prediction"),
        edge("e8", "dataset", "data", "evaluation", "truth"),
        edge("e9", "evaluation", "report", "export", "input"),
        edge("e10", "checkpoint", "model", "deploy", "model"),
        edge("e11", "dataset", "data", "deploy", "data"),
        edge("e12", "cad", "geometry", "prep", "geometry"),
        edge("e13", "dataset", "data", "simulgen", "data"),
        edge("e14", "sim_params", "parameters", "simulgen", "parameters"),
        edge("e15", "sim_checkpoint", "model", "simulgen", "resume"),
        edge("e16", "dataset", "data", "sdfflow", "data"),
        edge("e17", "sdf_params", "parameters", "sdfflow", "parameters"),
        edge("e18", "sdf_checkpoint", "model", "sdfflow", "resume"),
        edge("e19", "generator", "candidates", "optimization", "candidates")
      ];
      app.state.selectedNode = "mlp";
      app.state.selectedEdge = null;
      app.state.nodeCounter = 30;
      app.applyGraphAutofill();
      return Object.fromEntries(app.state.nodes.map(node => [node.id, {
        config: node.config,
        auto: node.autoFill,
        manual: node.manualConfigKeys
      }]));
    });

    assert(first.mlp.config.dataset_dir === "../dataset/custom/train.h5", "HDF5 path did not fill the MLP dataset_dir");
    assert(first.mlp.config.input_var === "3" && first.mlp.config.output_var === "2", "MLP spreadsheet column counts did not fill input_var/output_var");
    assert(first.mlp.config.modelpath === "../output/mlp/best.pth", "Checkpoint did not fill modelpath");
    assert(first.parameters.config.parameter_dataset === "dataset/custom/train.h5", "Design Parameters was not bound to the model's HDF5 dataset");
    assert(first.inference.config.dataset_path === "dataset/custom/train.h5", "Inference dataset was not filled");
    assert(first.inference.config.checkpoint_path === "../output/mlp/best.pth", `Inference checkpoint was not filled through the model: ${JSON.stringify(first.inference.config)}`);
    assert(first.inference.config.parameter_path === "dataset/custom/conditions.csv", "Inference parameter path was not filled");
    assert(first.evaluation.config.prediction_path === "output/predictions.h5", "Evaluation prediction path was not filled");
    assert(first.evaluation.config.truth_path === "dataset/custom/train.h5", "Evaluation truth path was not filled");
    assert(first.export.config.source_path === "frontend/runtime/evaluation/report.json", "Export source path was not filled");
    assert(first.optimization.config.csv_path === "frontend/runtime/candidates.csv", "Optimization CSV was not filled");
    assert(first.deploy.config.checkpoint_path === "output/mlp/best.pth", "Deploy checkpoint was not filled");
    assert(first.deploy.config.input_path === "dataset/custom/train.h5", "Deploy input dataset was not filled");
    assert(first.prep.config.input_geometry === "../cad/bracket.step", "CAD path did not fill geometry ingest");
    assert(first.simulgen.config.param_dir === "../dataset/simul/conditions.csv", "SimulGen param_dir was not filled");
    assert(first.simulgen.config.lc_data_type === "csv", "SimulGen lc_data_type was not inferred");
    assert(first.simulgen.config.vae_modelpath === "../output/simul/simulgen_vae.pth", "SimulGen VAE checkpoint heuristic failed");
    assert(first.simulgen.config.num_var === "2", "SimulGen field count was not inferred from dataset metadata");
    assert(first.sdfflow.config.condition_names === "bbox_x,volume", "SDFFlow condition names were not filled");
    assert(first.sdfflow.config.fm_modelpath === "../output/sdf/fm_best.pth", "SDFFlow FM checkpoint heuristic failed");

    const override = await page.evaluate(() => {
      const app = window.__AI_CAE_FRONTEND__;
      const model = app.state.nodes.find(node => node.id === "mlp");
      const dataset = app.state.nodes.find(node => node.id === "dataset");
      model.config.dataset_dir = "../dataset/manual/keep.h5";
      app.markManualConfigValue(model, "dataset_dir", model.config.dataset_dir);
      dataset.config.path = "dataset/changed/train.h5";
      app.applyGraphAutofill();
      const kept = model.config.dataset_dir;
      delete model.config.dataset_dir;
      app.markManualConfigValue(model, "dataset_dir", "");
      app.applyGraphAutofill();
      const followedAgain = model.config.dataset_dir;
      app.state.edges = app.state.edges.filter(edge => edge.id !== "e1");
      app.applyGraphAutofill();
      const clearedAfterDisconnect = Object.hasOwn(model.config, "dataset_dir");
      app.state.edges.push({ id: "e1b", fromNode: "dataset", fromPort: "data", toNode: "mlp", toPort: "data" });
      app.applyGraphAutofill();
      const restored = model.config.dataset_dir;
      model.config.mode = "inference";
      app.applyGraphAutofill();
      const inferenceMode = {
        infer_dataset: model.config.infer_dataset,
        has_dataset_dir: Object.hasOwn(model.config, "dataset_dir")
      };
      model.config.mode = "train";
      app.applyGraphAutofill();
      return { kept, followedAgain, clearedAfterDisconnect, restored, inferenceMode };
    });
    assert(override.kept === "../dataset/manual/keep.h5", "Manual model path was overwritten");
    assert(override.followedAgain === "../dataset/changed/train.h5", "Clearing a manual override did not resume graph filling");
    assert(!override.clearedAfterDisconnect, "Disconnect did not clear the graph-owned value");
    assert(override.restored === "../dataset/changed/train.h5", "Reconnect did not restore graph filling");
    assert(override.inferenceMode.infer_dataset === "../dataset/changed/train.h5" && !override.inferenceMode.has_dataset_dir, "Mode change did not move the auto path from dataset_dir to infer_dataset");

    await page.evaluate(() => {
      document.querySelector("#pipelineName").value = "Autofill persistence smoke";
      document.querySelector("#savePipeline").click();
    });
    await page.reload();
    await page.waitForFunction(() => document.querySelector(".route-health")?.textContent.includes("routes live"));
    const persisted = await page.evaluate(() => {
      const app = window.__AI_CAE_FRONTEND__;
      const model = app.state.nodes.find(node => node.id === "mlp");
      return {
        value: model.config.dataset_dir,
        source: app.autoFillMeta(model, "dataset_dir")?.sourceNodeId,
        name: document.querySelector("#pipelineName").value
      };
    });
    assert(persisted.value === "../dataset/changed/train.h5" && persisted.source === "dataset", "Autofill provenance did not survive persistence");
    assert(persisted.name === "Autofill persistence smoke", "Pipeline persistence did not restore the test document");

    await page.evaluate(() => window.__AI_CAE_FRONTEND__.openConfig("mlp"));
    const autoCard = page.locator('.config-card:has([data-key="dataset_dir"])');
    assert(await autoCard.count() === 1, "dataset_dir config card was not rendered");
    assert(await autoCard.evaluate(element => element.classList.contains("graph-autofilled")), "Auto-filled config card is not visually marked");
    assert((await autoCard.innerText()).includes("HDF5 Dataset"), "Auto-filled config card does not show its source");
    await page.screenshot({ path: path.join(__dirname, "runtime", "autofill-config.png"), fullPage: false });

    assert(errors.length === 0, errors.join("\n"));
    console.log("Autofill smoke test passed: graph paths, model dimensions, provenance, manual overrides, disconnect cleanup, persistence, and UI markers.");
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
