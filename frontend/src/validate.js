import { toast } from "./dom.js";
import { state } from "./state.js";
import { BLOCK_SPECS, INPUT_SOURCE_META, MODEL_CATALOG, TYPE_META } from "./constants.js";
import { apiRequest, checkpointMetaNow, requireRuntime } from "./api.js";
import { normalizeConfigValues, rawConfig, requiredFor } from "./config.js";
import { applyGraphAutofill, selectedParameterCandidate, toGeometryPath, toMethodPath } from "./autofill.js";

export function typeColor(type) {
  return (TYPE_META[type] || TYPE_META.artifact).color;
}

export function compatible(fromType, toType) {
  return fromType === toType || fromType === "artifact" || toType === "artifact";
}

export function configTextForNode(node, overrides = {}) {
  const spec = BLOCK_SPECS[node.type];
  if (!spec?.isModel) return "";
  const model = MODEL_CATALOG[spec.modelId];
  applyGraphAutofill();
  const values = normalizeConfigValues({ ...node.config, ...overrides, model: spec.modelId });
  return `${rawConfig(values, model.keys)}\n`;
}

export function geometryConfigText(node) {
  applyGraphAutofill();
  const upstream = state.edges
    .filter(edge => edge.toNode === node.id)
    .map(edge => state.nodes.find(candidate => candidate.id === edge.fromNode))
    .find(candidate => candidate?.type === "source.cad");
  const rawInput = String(node.config.input_geometry || upstream?.config.path || "").replaceAll("\\", "/");
  const inputGeometry = toGeometryPath(rawInput);
  const values = {
    model: "geometry_ingest",
    mode: node.config.mode || "inspect",
    input_geometry: inputGeometry,
    reader: node.config.reader || "auto",
    mesh_type: node.config.mesh_type || "surface",
    emit: node.config.emit || "graph",
    num_fields: node.config.num_fields || "3",
    num_points: node.config.num_points || "4096",
    resample_method: node.config.resample_method || "fps",
    seed: node.config.seed || "42"
  };
  if (values.mode === "ingest") values.output_dataset = node.config.output_dataset;
  if (String(node.config.limit || "0") !== "0") values.limit = node.config.limit;
  // Emitted only when set, so leaving them at 0 keeps gmsh's own default sizing
  // rather than pinning it to an explicit 0 in the generated config.
  ["mesh_size_min", "mesh_size_max"].forEach(key => {
    if (Number(node.config[key] || 0) > 0) values[key] = node.config[key];
  });
  // Same reason as rawConfig: an empty value serializes to a keyless line that the
  // native parser rejects, hiding the real "input_geometry is required" diagnostic.
  return `${Object.entries(values)
    .filter(([, value]) => String(value ?? "").trim() !== "")
    .map(([key, value]) => `${key.padEnd(29, " ")}${value}`).join("\n")}\n`;
}

/**
 * The model family an Inference block should launch, and where that came from.
 *
 * A trainer wired into the `model` port names it directly. Without one, the
 * saved checkpoint does: every native inference path overlays the checkpoint's
 * `model_config` over the config file, so the checkpoint is the authority on
 * both the family and the architecture -- and `/api/checkpoint` reads it back.
 */
export function inferenceModel(node) {
  const trainer = [...state.edges]
    .filter(edge => edge.toNode === node.id && edge.toPort === "model")
    .map(edge => state.nodes.find(item => item.id === edge.fromNode))
    .find(item => item && BLOCK_SPECS[item.type]?.isModel);
  if (trainer) return { modelId: BLOCK_SPECS[trainer.type].modelId, trainer, source: "trainer" };
  const checkpoint = String(node.config.checkpoint_path || "").trim();
  const facts = checkpoint ? checkpointMetaNow(checkpoint) : null;
  const declared = String(node.config.model_id || "").trim().toLowerCase();
  const modelId = declared || (facts?.ok ? String(facts.model || "") : "");
  return {
    modelId: MODEL_CATALOG[modelId] ? modelId : "",
    trainer: null,
    facts: facts?.ok ? facts : null,
    checkpointError: facts && !facts.ok ? facts.error : "",
    source: "checkpoint"
  };
}

// SimulGen-VAE and SDFFlow need a dozen architecture keys their checkpoints do
// not record (latent_dim_end, lc_filter, num_filter_enc, ...), so their
// non-training modes still require the model block that owns those values.
export const STANDALONE_INFERENCE_MODEL_IDS = [
  "meshgraphnets", "meshgraphnets-v", "transolver", "fno", "gino", "deeponet", "point_deeponet", "mlp"
];
const STANDALONE_INFERENCE_MODELS = new Set(STANDALONE_INFERENCE_MODEL_IDS);

/**
 * Build a runnable inference config from a checkpoint alone.
 *
 * This is the "put in the inference dataset and infer" path. Previously an
 * Inference block fed by a `source.checkpoint` produced **no step at all** --
 * `executableSteps` needed a model node on the canvas and returned early
 * without one -- so a graph holding a trained model and a held-out dataset
 * validated cleanly and then did nothing but report "no executable step".
 *
 * The architecture keys are taken from the checkpoint rather than asked for,
 * because that is what the native loader will use regardless of what the config
 * says; anything typed here that disagreed would be silently overridden.
 */
// The operator repo records the data contract separately from the architecture,
// under DataSpec field names. Mapping them back to config keys is what supplies
// input_var/output_var for FNO/GINO/DeepONet, which model_config does not hold.
const DATA_SPEC_KEYS = {
  input_var: "input_var",
  output_var: "output_var",
  num_timesteps: "num_timesteps",
  operator_dim: "operator_dim",
  positional_dim: "positional_features",
  condition_dim: "cond_var"
};

/** Everything a checkpoint records about how to rebuild and feed its model. */
export function checkpointArchitecture(facts) {
  const architecture = { ...(facts?.model_config || {}) };
  // data_config last, matching the native loader's own order: it applies the
  // DataSpec *after* the model_config overlay, so it wins where both speak.
  Object.entries(facts?.data_config || {}).forEach(([key, value]) => {
    const mapped = DATA_SPEC_KEYS[key];
    if (mapped) architecture[mapped] = value;
  });
  return architecture;
}

export function standaloneInferenceConfig(node, modelId, facts) {
  const model = MODEL_CATALOG[modelId];
  if (!model || !facts?.model_config) return "";
  const accepted = new Set(model.keys);
  const values = { model: modelId, mode: "inference" };
  const architecture = checkpointArchitecture(facts);
  Object.entries(architecture).forEach(([key, value]) => {
    if (accepted.has(key) && key !== "model" && key !== "mode") values[key] = value;
  });
  values.gpu_ids = String(node.config.gpu_ids || "0");
  values.modelpath = toMethodPath(node.config.checkpoint_path);
  values.infer_dataset = toMethodPath(node.config.dataset_path);
  // num_timesteps records what the model was TRAINED on; the rollout length is
  // a property of this run. Defaulting it to the trained span minus the given
  // initial condition reproduces what the paired training config would do.
  const trained = Number(architecture.num_timesteps || 0);
  const requested = String(node.config.infer_timesteps || "").trim();
  if (accepted.has("infer_timesteps")) {
    values.infer_timesteps = requested || (trained > 1 ? String(trained - 1) : "");
  }
  ["inference_output_dir", "input_var", "output_var", "cond_var", "edge_var", "batch_size", "num_workers"].forEach(key => {
    const override = String(node.config[key] ?? "").trim();
    if (override && accepted.has(key)) values[key] = override;
  });
  return `${rawConfig(values, model.keys)}\n`;
}

export function topologicalNodes(targetId = null) {
  const ordered = [];
  const permanent = new Set();
  const temporary = new Set();
  const visit = node => {
    if (!node || permanent.has(node.id)) return;
    if (temporary.has(node.id)) throw new Error(`Pipeline cycle detected at ${BLOCK_SPECS[node.type]?.label || node.id}.`);
    temporary.add(node.id);
    state.edges
      .filter(edge => edge.toNode === node.id)
      .map(edge => state.nodes.find(candidate => candidate.id === edge.fromNode))
      .filter(Boolean)
      .forEach(visit);
    temporary.delete(node.id);
    permanent.add(node.id);
    ordered.push(node);
  };
  if (targetId) visit(state.nodes.find(node => node.id === targetId));
  else state.nodes.forEach(visit);
  return ordered;
}

export function executableSteps(targetId = null) {
  const target = targetId ? state.nodes.find(node => node.id === targetId) : null;
  const available = topologicalNodes(targetId);
  const modelNodes = available.filter(node => BLOCK_SPECS[node.type]?.isModel);
  // A model configured for a runtime-only mode (for example SDFFlow `sample`)
  // is the configuration provider for its connected run block, not a second
  // independent execution. Training modes remain separate steps so a full
  // train -> inference pipeline still trains before it predicts.
  const runtimeProviderIds = new Set(state.edges
    .filter(edge => edge.toPort === "model"
      && ["run.inference", "run.cad_generator"].includes(
        state.nodes.find(node => node.id === edge.toNode)?.type
      ))
    .map(edge => edge.fromNode));
  if (target && BLOCK_SPECS[target.type]?.isModel) {
    return [{
      label: `${BLOCK_SPECS[target.type].label} · ${target.config.mode || "train"}`,
      nodeId: target.id,
      config: configTextForNode(target)
    }];
  }
  // Running one execution block means "run this block against what already
  // exists upstream", not "rebuild everything it depends on". Targeting it used
  // to fall through to the whole-graph branch below, so pressing Run on an
  // Inference block launched a full training run first and only then inferred.
  // The upstream model node is still walked -- inference needs its key set and
  // checkpoint -- it just is not executed again.
  const executeOnly = target && ["run.inference", "run.cad_generator"].includes(target.type)
    ? target.id
    : null;
  const steps = [];
  available.forEach(node => {
    if (executeOnly && node.id !== executeOnly) return;
    const spec = BLOCK_SPECS[node.type];
    if (node.type === "prep.geometry") {
      steps.push({
        label: `${spec.label} · ${node.config.mode || "inspect"}`,
        nodeId: node.id,
        config: geometryConfigText(node)
      });
      return;
    }
    if (spec?.isModel) {
      const mode = String(node.config?.mode || "train").toLowerCase();
      const trainingMode = ["train", "train_vae", "train_fm", "train_lc"].includes(mode);
      if (node.id !== target?.id && runtimeProviderIds.has(node.id) && !trainingMode) return;
      steps.push({
        label: `${spec.label} · ${node.config.mode || "train"}`,
        nodeId: node.id,
        config: configTextForNode(node)
      });
      return;
    }
    if (!["run.inference", "run.cad_generator"].includes(node.type)) return;
    const connected = [...modelNodes].reverse().find(modelNode =>
      state.edges.some(edge => edge.fromNode === modelNode.id && edge.toNode === node.id)
    );
    // The fallback to "whatever model is on the canvas" only makes sense when
    // nothing else names one. A block fed by a saved-model source has already
    // said which checkpoint to run, and borrowing an unrelated trainer's key
    // set there would run the wrong architecture under the right file name.
    const fedByCheckpoint = state.edges.some(edge =>
      edge.toNode === node.id && edge.toPort === "model"
      && state.nodes.find(item => item.id === edge.fromNode)?.type === "source.checkpoint"
    );
    const upstream = connected || (fedByCheckpoint ? null : modelNodes.at(-1));
    if (!upstream) {
      // No trainer on the canvas: run the saved checkpoint directly. This is
      // the whole point of the `source.checkpoint` -> Inference wiring, which
      // used to fall through here and emit nothing at all.
      const resolved = node.type === "run.inference" ? inferenceModel(node) : { modelId: "" };
      const config = STANDALONE_INFERENCE_MODELS.has(resolved.modelId)
        && node.config.checkpoint_path && node.config.dataset_path
        ? standaloneInferenceConfig(node, resolved.modelId, resolved.facts)
        : "";
      if (config) {
        steps.push({
          label: `${MODEL_CATALOG[resolved.modelId].label} · inference`,
          nodeId: node.id,
          config
        });
      }
      return;
    }
    const modelId = BLOCK_SPECS[upstream.type].modelId;
    const mode = node.type === "run.cad_generator"
      ? "sample"
      : modelId === "simulgenvae"
        ? "reconstruct"
        : modelId === "sdfflow"
          ? "reconstruct"
          : "inference";
    // The run.inference/run.cad_generator node's own graph-linked fields
    // (dataset_path, checkpoint_path, ...) are what the Inspector shows and
    // what the user edits, but the upstream model node's OWN persisted
    // config (still whatever mode it was last trained/edited in) is what
    // actually gets serialized below. Without this merge, infer_dataset and
    // friends are silently absent unless the model node happens to already
    // be sitting in inference mode with those keys manually set.
    const overrides = { mode };
    const catalogKeys = MODEL_CATALOG[modelId].keys;
    if (node.type === "run.cad_generator") {
      // Serialize the controls shown on the CAD Generator block into the native
      // SDFFlow configuration. The aliases keep previously saved Studio graphs
      // (`candidates`, `guidance`) runnable after the UI adopted native names.
      const generatorValues = {
        num_samples: node.config.num_samples || node.config.candidates,
        cfg_scale: node.config.cfg_scale || node.config.guidance,
        ode_steps: node.config.ode_steps,
        mc_resolution: node.config.mc_resolution,
        seed: node.config.seed,
        cond_values: node.config.cond_values,
        candidate_multiplier: node.config.candidate_multiplier
      };
      Object.entries(generatorValues).forEach(([key, value]) => {
        if (catalogKeys.includes(key) && String(value ?? "").trim()) overrides[key] = value;
      });
    }
    if (catalogKeys.includes("infer_dataset") && node.config.dataset_path) {
      // The graph is authoritative -- except when what it says is "run
      // inference on the training set". That happens whenever the same HDF5
      // source block feeds both the trainer and the Inference block, and
      // silently overwriting a configured held-out infer_dataset with the
      // training file is never what anyone means by it. Keep the model's own
      // held-out split in that one case; `inferenceDatasetWarnings` surfaces it.
      const wired = toMethodPath(node.config.dataset_path);
      const trainingDataset = upstream.config.dataset_dir;
      const configuredSplit = upstream.config.infer_dataset;
      if (!(configuredSplit && wired === trainingDataset)) overrides.infer_dataset = wired;
    }
    if (catalogKeys.includes("modelpath") && node.config.checkpoint_path) {
      // toMethodPath is a no-op on paths already relative to the owning
      // method repo (e.g. a checkpoint fed back from the same model node
      // that just trained it) and only adds the "../" prefix when the value
      // came from a source.checkpoint block's suite-relative browse/upload
      // path, so it is safe to apply unconditionally here.
      overrides.modelpath = toMethodPath(node.config.checkpoint_path);
    }
    ["vae_modelpath", "lc_modelpath", "fm_modelpath"].forEach(key => {
      if (catalogKeys.includes(key) && node.config[key]) overrides[key] = toMethodPath(node.config[key]);
    });
    steps.push({
      label: `${BLOCK_SPECS[upstream.type].label} · ${mode}`,
      nodeId: node.id,
      config: configTextForNode(upstream, overrides)
    });
  });

  // Analysis blocks close the loop the canvas already draws. They do not go
  // through the native launcher -- they call the Studio analysis APIs -- so they
  // are emitted as a second kind of step and executed in-process by the backend.
  available.forEach(node => {
    if (executeOnly && node.id !== executeOnly) return;
    const step = analysisStep(node, steps);
    if (step) steps.push(step);
  });
  return steps;
}

/** The node feeding `port`, if any. */
function upstreamOf(node, port) {
  const edge = state.edges.find(item => item.toNode === node.id && item.toPort === port);
  return edge ? state.nodes.find(item => item.id === edge.fromNode) : null;
}

/**
 * Name an analysis input by the block that produces it.
 *
 * When the producing block runs in this same pipeline its output does not exist
 * yet -- inference writes to an epoch-numbered directory created during the run
 * -- so the path cannot be resolved here. `@results:<nodeId>` defers that to the
 * backend. Only when the producer is NOT part of this run do we fall back to the
 * path it recorded earlier, which would otherwise be a stale directory.
 */
function analysisInput(source, steps) {
  if (!source) return "";
  const willRun = steps.some(step => step.nodeId === source.id);
  if (willRun) return `@results:${source.id}`;
  return source.config?.results_path || outputPathOf(source);
}

function outputPathOf(node) {
  const config = node?.config || {};
  return config.results_path || config.report_path || config.export_path
    || config.metrics_csv || config.path || "";
}

export function analysisStep(node, steps) {
  if (node.type === "evaluate.predictions") {
    const prediction = analysisInput(upstreamOf(node, "prediction"), steps);
    const truth = node.config.truth_path || upstreamOf(node, "truth")?.config?.path || "";
    if (!prediction || !truth) return null;
    let fieldPairs = [];
    try {
      const parsed = JSON.parse(node.config.field_pairs || "[]");
      if (Array.isArray(parsed)) fieldPairs = parsed;
    } catch {
      // The evaluation workspace will surface malformed saved mappings; do not
      // silently fall back to positional rows in a graph run.
    }
    const payload = {
      prediction_path: prediction,
      truth_path: truth
    };
    if (node.config.mapping_mode === "legacy") {
      Object.assign(payload, {
        prediction_start: Number(node.config.prediction_start ?? 3),
        truth_start: Number(node.config.truth_start ?? 3),
        num_fields: Number(node.config.num_fields ?? 1)
      });
    } else if (fieldPairs.length) {
      payload.field_pairs = fieldPairs;
      payload.confirm_mapping = String(node.config.mapping_confirmed || "").toLowerCase() === "true";
    }
    return {
      kind: "analysis",
      action: "evaluation",
      label: `${BLOCK_SPECS[node.type].label} · score`,
      nodeId: node.id,
      payload
    };
  }
  if (node.type === "output.export") {
    const source = analysisInput(upstreamOf(node, "input"), steps);
    if (!source) return null;
    return {
      kind: "analysis",
      action: "export",
      label: `${BLOCK_SPECS[node.type].label} · package`,
      nodeId: node.id,
      payload: { path: source, label: node.config.path || "studio-export" }
    };
  }
  if (node.type === "optimize.design") {
    // Without this the block was inert: the canvas drew it, the inspector
    // configured it, `/api/optimization` implemented it, and a pipeline run
    // reported "completed" having quietly dropped the step -- taking the
    // downstream Export with it, since its input then resolved to "".
    // `@results:<id>` is what analysisInput returns when the producing block runs
    // in this same pipeline -- the path does not exist yet, so it cannot be
    // pattern-matched, and requiring a literal ".csv" here silently dropped the
    // step for exactly the case the template is built around. The backend
    // publishes the generator's candidates.csv as that step's result, so the
    // reference resolves to the table at execution time.
    const upstream = String(analysisInput(upstreamOf(node, "candidates"), steps) || "");
    const csv = upstream.startsWith("@results:") || /\.csv$/i.test(upstream)
      ? upstream
      : node.config.csv_path;
    const objectives = String(node.config.objectives || "").split(",").map(item => item.trim()).filter(Boolean);
    if (!csv || !objectives.length) return null;
    return {
      kind: "analysis",
      action: "optimization",
      label: `${BLOCK_SPECS[node.type].label} · select`,
      nodeId: node.id,
      payload: {
        csv_path: csv,
        objectives: objectives.join(","),
        directions: String(node.config.directions || ""),
        constraints: String(node.config.constraints || ""),
        top_k: Number(node.config.top_k ?? 10)
      }
    };
  }
  return null;
}

/**
 * Flag every Inference block that is pointed at its model's training data.
 *
 * Not an error -- predicting the training set is a legitimate sanity check --
 * but it is almost always an accident, and the resulting metrics look great for
 * the wrong reason. Returned as text so both the validate toast and the run
 * path can show it.
 */
export function inferenceDatasetWarnings() {
  const warnings = [];
  state.nodes.filter(node => node.type === "run.inference").forEach(node => {
    const wired = node.config.dataset_path;
    if (!wired) return;
    const model = state.edges
      .filter(edge => edge.toNode === node.id && edge.toPort === "model")
      .map(edge => state.nodes.find(item => item.id === edge.fromNode))
      .find(item => item && BLOCK_SPECS[item.type]?.isModel);
    if (!model) return;
    const training = String(model.config.dataset_dir || "");
    if (training && toMethodPath(wired) === training) {
      warnings.push(`${BLOCK_SPECS[node.type].label} is pointed at the training dataset (${wired}). Connect the held-out inference dataset for a real test.`);
    }
  });
  return warnings;
}

/**
 * Is this block's required `data` port actually required *in its current mode*?
 *
 * `required` is a static property of the block spec, so a model block demanded a
 * training dataset even in a mode that reads none -- SDFFlow `sample` generates
 * from a checkpoint and noise, and the launcher's own required set for it is
 * {vae_modelpath, fm_modelpath, output_dir, num_samples, seed, ode_steps,
 * mc_resolution}. The result was "Cannot run: SDFFlow: missing training data"
 * for a configuration the authoritative preflight passed with zero errors, and
 * the shipped generative template -- which ships no dataset block at all --
 * could never be run.
 *
 * The launcher's per-mode required set is the authority: if it does not ask for
 * a dataset in this mode, neither do we.
 */
function portRequiredInMode(node, port) {
  if (!port.required) return false;
  const spec = BLOCK_SPECS[node.type];
  if (port.id !== "data" || !spec?.isModel) return true;
  const mode = String(node.config?.mode || "").toLowerCase();
  const required = requiredFor(spec.modelId, mode);
  if (!required?.size) return true;
  return required.has("dataset_dir");
}

export function validateGraph(showToast = true) {
  // Validation must inspect the same graph-derived values that execution will
  // serialize, including a selected Design Parameters spreadsheet row.
  applyGraphAutofill();
  const errors = [];
  state.nodes.forEach(node => {
    const spec = BLOCK_SPECS[node.type];
    spec.inputs.filter(port => portRequiredInMode(node, port)).forEach(port => {
      const edge = state.edges.find(candidate => candidate.toNode === node.id && candidate.toPort === port.id);
      if (!edge) {
        errors.push(`${spec.label}: missing ${port.label}`);
        return;
      }
      const source = state.nodes.find(candidate => candidate.id === edge.fromNode);
      const sourceMeta = source && INPUT_SOURCE_META[source.type];
      // A wire is only topology. CAD, HDF5, and checkpoint source blocks still
      // need a concrete artifact before the downstream block can run. Catch the
      // empty-source case here so Validate and Run give an immediate instruction
      // instead of reporting a clean graph and failing later in server preflight.
      // Design Parameters is different: its spreadsheet can be stored directly
      // on the block without an external CSV/JSON binding.
      if (sourceMeta && source.type !== "source.parameters"
          && !String(source.config?.[sourceMeta.key] || "").trim()) {
        errors.push(`${spec.label}: ${port.label} is connected, but ${sourceMeta.label} is not selected.`);
      }
    });
  });
  state.edges.forEach(edge => {
    const source = state.nodes.find(node => node.id === edge.fromNode);
    const target = state.nodes.find(node => node.id === edge.toNode);
    if (!source || !target) return;
    const out = BLOCK_SPECS[source.type].outputs.find(port => port.id === edge.fromPort);
    const input = BLOCK_SPECS[target.type].inputs.find(port => port.id === edge.toPort);
    if (out && input && !compatible(out.type, input.type)) errors.push(`Type mismatch: ${out.type} → ${input.type}`);
  });
  // An Inference block that cannot name its model family is not runnable, and
  // saying so here is the difference between an actionable message and the old
  // "this graph has no executable model or inference step" -- which was thrown
  // by the Run button after validation had already reported the graph clean.
  state.nodes.filter(node => node.type === "run.inference").forEach(node => {
    const linkedTrainer = state.edges.some(edge =>
      edge.toNode === node.id && edge.toPort === "model"
      && BLOCK_SPECS[state.nodes.find(item => item.id === edge.fromNode)?.type || ""]?.isModel
    );
    if (linkedTrainer) return;
    const resolved = inferenceModel(node);
    const label = BLOCK_SPECS[node.type].label;
    if (resolved.checkpointError) {
      errors.push(`${label}: the saved model could not be read (${resolved.checkpointError}).`);
    } else if (!resolved.modelId) {
      errors.push(`${label}: cannot tell which method this saved model belongs to. Set "model id" on the block, or connect the model that trained it.`);
    } else if (!STANDALONE_INFERENCE_MODELS.has(resolved.modelId)) {
      errors.push(`${label}: ${MODEL_CATALOG[resolved.modelId]?.label || resolved.modelId} needs its model block on the canvas — its checkpoint does not record every key that mode requires.`);
    } else if (!node.config.dataset_path) {
      errors.push(`${label}: connect the dataset to infer on.`);
    }
  });
  state.nodes.filter(node => node.type === "run.cad_generator").forEach(node => {
    const label = BLOCK_SPECS[node.type].label;
    const rawConditions = String(node.config.cond_values || "").split(",").map(item => item.trim()).filter(Boolean);
    if (rawConditions.some(value => !Number.isFinite(Number(value)))) {
      errors.push(`${label}: condition values must all be finite numbers.`);
      return;
    }
    const parameterEdge = state.edges.find(edge => edge.toNode === node.id && edge.toPort === "parameters");
    const source = parameterEdge && state.nodes.find(candidate => candidate.id === parameterEdge.fromNode);
    if (!source || source.type !== "source.parameters") return;
    const selected = selectedParameterCandidate(source);
    if (!selected.table) return; // The connected block is intentionally unused for unconditional sampling.
    const manualConditions = (node.manualConfigKeys || []).includes("cond_values") && rawConditions.length > 0;
    if (manualConditions) {
      if (selected.inputs.length && rawConditions.length !== selected.inputs.length) {
        errors.push(`${label}: manual condition values must match the spreadsheet's ${selected.inputs.length} Input columns.`);
      }
      return;
    }
    if (!selected.selectedSampleId) {
      errors.push(`${label}: choose one Design Parameters spreadsheet row for generation.`);
    } else if (!selected.row) {
      errors.push(`${label}: the selected generation row no longer exists in the bound dataset.`);
    } else if (selected.missingNames.length) {
      errors.push(`${label}: every generation Input column needs a name.`);
    } else if (selected.duplicateNames.length) {
      errors.push(`${label}: generation Input column names must be unique (${selected.duplicateNames.join(", ")}).`);
    } else if (selected.invalidInputs.length) {
      errors.push(`${label}: selected row has missing or non-numeric values for ${selected.invalidInputs.map(item => item.name || item.id).join(", ")}.`);
    }
  });
  state.nodes.filter(node => node.type === "optimize.design").forEach(node => {
    const label = BLOCK_SPECS[node.type].label;
    const objectives = String(node.config.objectives || "").split(",").map(item => item.trim()).filter(Boolean);
    const directions = String(node.config.directions || "").split(",").map(item => item.trim().toLowerCase()).filter(Boolean);
    if (!objectives.length) {
      errors.push(`${label}: add at least one objective column.`);
    } else if (directions.length !== objectives.length || directions.some(item => !["min", "max"].includes(item))) {
      errors.push(`${label}: directions must contain one min or max entry for every objective.`);
    }
  });
  try {
    topologicalNodes();
  } catch (error) {
    errors.push(error.message);
  }
  if (showToast) toast(errors.length ? `${errors.length} graph issue${errors.length === 1 ? "" : "s"}: ${errors[0]}` : "Validation passed: typed graph and required links are complete.", errors.length ? "error" : "");
  return errors;
}

export function preflightMessages(payload) {
  const report = payload?.report || payload?.failures?.[0]?.preflight?.report;
  if (!report) return [{ type: "error", text: payload?.error || "Preflight did not return a diagnostic report." }];
  return [
    {
      type: report.summary.errors ? "error" : report.summary.warnings ? "warn" : "",
      text: `Authoritative preflight: ${report.summary.errors} errors, ${report.summary.warnings} warnings, ${report.summary.notices} notices.`
    },
    ...report.diagnostics.map(item => ({
      type: item.severity === "error" ? "error" : item.severity === "warning" ? "warn" : "",
      text: `[${item.code}]${item.field ? ` ${item.field}:` : ""} ${item.message}${item.hint ? ` Hint: ${item.hint}` : ""}`
    }))
  ];
}

export async function preflightConfigText(text, label, options = {}) {
  if (!requireRuntime()) return null;
  return apiRequest("/api/preflight", {
    method: "POST",
    allowError: true,
    body: {
      config: text,
      label,
      strict: Boolean(options.strict),
      skip_filesystem: Boolean(options.skipFilesystem),
      skip_native: Boolean(options.skipNative),
      skip_environment: Boolean(options.skipEnvironment)
    }
  });
}
