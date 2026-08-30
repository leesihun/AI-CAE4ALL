# AI-CAE4ALL Studio workflow audit

Date: 2026-08-30
Scope: `frontend/` plus narrow cHI checkpoint and portable-inference integration

## Research-derived requirements

The Studio is a workflow and experiment system, not only a collection of
visual cards. The following contracts are therefore required.

1. **A run must retain exact lineage.** MLflow defines a run as one execution
   with metadata, parameters, metrics, timestamps, and artifacts. Dataset
   identity is also part of fair run search and comparison.
   - https://mlflow.org/docs/latest/tracking
   - https://mlflow.org/docs/latest/ml/tracking/tracking-api/
2. **Scalar metrics need raw step history, run selection, comparison, and an
   optional visual smoothing control.** TensorBoard explicitly supports
   scalar history, selecting and comparing runs, and adjusting smoothing while
   retaining the unsmoothed measurements.
   - https://www.tensorflow.org/tensorboard/get_started
   - https://www.tensorflow.org/tensorboard/scalars_and_keras
3. **Graph connections must be typed, handle-specific, multi-connectable where
   the input contract permits it, and rejected before mutation when invalid.**
   React Flow documents unique handle IDs, connection validation, and cycle
   prevention as editor responsibilities.
   - https://reactflow.dev/learn/customization/handles
   - https://reactflow.dev/examples/interaction/validation
   - https://reactflow.dev/examples/interaction/prevent-cycles
4. **A workflow editor must expose selection/deletion and save/restore as real
   operations.** React Flow's public contracts expose element deletion,
   keyboard deletion, and serializable viewport/graph state; controls that only
   resemble those operations are not sufficient.
   - https://reactflow.dev/api-reference/types/delete-elements
   - https://reactflow.dev/api-reference/react-flow
   - https://v9.reactflow.dev/examples/save-and-restore/
5. **Menu buttons require keyboard-equivalent operation.** WAI-ARIA's menu
   button pattern includes focus transfer, arrow-key movement, Home/End, and
   Escape-to-trigger behavior.
   - https://www.w3.org/WAI/ARIA/apg/patterns/menu-button/
6. **Dynamic status must be programmatically exposed.** WCAG 2.2 status-message
   guidance requires status changes to be available through roles or live
   regions rather than color alone.
   - https://www.w3.org/WAI/WCAG22/Techniques/failures/F103

## Live-code findings and repairs

| Finding | Why it was wrong | Implemented repair | Evidence gate |
|---|---|---|---|
| Jobs stored labels and routes but no pipeline node ID. | Two blocks of the same model family could not be distinguished; model-card matching was fuzzy text matching. | Pipeline submissions now persist `target_node_id`, per-step `node_id`, `node_type`, route model, and mode. New jobs bind downstream Train Metrics blocks by exact node lineage. Legacy jobs fall back to route metadata. | Backend unit test plus a real graph job metadata assertion. |
| Compare Models accepted multiple graph links but ignored them in its workspace. | The visible connection implied a data flow that did not exist. | The Compare workspace resolves every connected Model/Train Metrics source to a persisted run, finds common metric keys, overlays raw histories, and ranks last raw values. Qualified evaluation CSV ranking remains separate for cross-family comparison. | Browser test with two connected Train Metrics blocks and two distinct runs. |
| New links could create a cycle and only fail later during full validation. | Invalid state was accepted even though the editor already had enough information to reject it. | Connection creation now performs reachability checking and rejects a cycle before adding the edge. | Browser connection test asserts the second edge is rejected. |
| Several non-artifact cards opened the generic sample viewer. | Evaluation, comparison, optimization, export, and deployment own dedicated information surfaces. | Block specifications now declare their owning workspace; preview, Inspect, and primary actions route consistently. | Browser routing assertions verify Comparison and Evaluation do not open the artifact viewer. |
| Train Metrics plotted real values but lacked smoothing, export, and direct access from jobs/model details. | Users could inspect a run but could not reproduce a smoothed view, export the raw observations, or move directly from run status to metrics. | Added visual-only smoothing, raw long-form CSV download, exact run selectors, and Metrics actions in Models and Experiments. | Focused metrics browser test and CSV download assertion. |
| Qualified CSV comparison guessed `model` and `mean_relative_l2` columns. | Real evaluation CSVs expose different schemas, including `relative_l2`, so the default could reject valid evidence. | Added a schema inspection endpoint and automatic common numeric metric/group suggestions for every selected run set. | Backend schema/ranking unit test and live endpoint check. |
| Connected artifacts only changed a model config in the final runtime text. | The visible editor, saved pipeline, model-detail view, and downstream blocks could disagree with the command that actually ran. | Added one graph-derived resolver with persisted provenance for datasets, Design Parameters, model-family checkpoints, CAD ingest, inference, evaluation, optimization, export, and deploy inputs. Manual edits remain authoritative until cleared; mode/source/link changes update graph-owned values and disconnects clean them up. | `autofill-smoke-runner.js` covers multi-hop propagation, MLP dimensions, SimulGen/SDFFlow keys, manual overrides, disconnect/reconnect, mode switching, persistence, and config UI markers. |
| The browser treated config-key casing differently from the suite parser. | `MODEL`, `Model`, and `model` could appear as separate UI fields even though authoritative parsing is case-insensitive. | Canonicalize config keys and closed values at import, raw-edit, render, save, preflight, and pipeline serialization boundaries; preserve free-form path casing. | Full browser regression covers mixed-case keys, values, duplicates, and a case-sensitive-looking path. |
| Job polling reopened the minimized Process status drawer every 900 ms. | Status rendering coupled data refresh to an unconditional reveal and removed the user's `minimized` state. | Explicit job/open-log actions reveal the drawer; polling and terminal updates refresh content while preserving display state. Dismiss now resets all stale job metadata. | Browser regression minimizes a running job, applies polling and terminal updates, and asserts the drawer stays minimized. |
| “Saved locally” was shown without any persistence implementation. | Reload discarded the graph, viewport, block configs, Design Parameter cells, and workspace selections. | Added versioned local persistence, debounced autosave, explicit Save, validated JSON Import/Export, viewport/name/config restoration, cycle/single-input validation, and persistence after Undo and job-lineage updates. Running processes are deliberately restored as idle graph state and rejoined through persisted jobs. | Browser regression saves, reloads, rejects a malformed self-link import without mutation, and restores edited spreadsheet cells. |
| The block `•••` control did nothing and graph edges could neither be selected nor deleted. | Basic workflow-editor operations were visually advertised but absent. | Added accessible Open/Duplicate/Delete block menus, arrow/Home/End/Escape keyboard operation, selectable wide edge hit targets, exact connection-contract inspection, and edge-only Delete/Backspace behavior. | Focused browser test duplicates/deletes a block and removes an edge while retaining both endpoint blocks. |
| Every card footer called the pipeline runner, including datasets, parameters, evaluation, comparison, export, and deployment. | Non-executable blocks produced irrelevant “no executable step” errors. | Block specs now distinguish executable steps from information/workspace blocks. Card actions say Train/Run only for launcher-backed steps and Open/Sheet/Metrics for their actual information surface. | Browser test asserts an HDF5 source exposes Open and model blocks retain Train. |
| Optimization required guessed comma-separated column names and defaulted to invented phrases such as “min peak stress”. | Real CSV schemas differ, and a plausible-looking label was not evidence that the column existed. | Added `POST /api/optimization/schema`, finite-value sampling, identifier/metadata exclusion, explicit objective checkboxes, per-objective min/max direction, constraint suggestions, exact-node persistence, and a disabled Run action until a real objective is selected. | Unit tests cover schema classification and non-finite rows; browser test drives an actual evaluation CSV. |
| Evaluation, deployment, optimization, comparison, and export frequently ignored the exact block that opened them. | Two blocks of the same type shared the first matching node, and graph connections often did not preselect their artifact. | Each workspace now receives `state.studioNode`, resolves compatible connected paths, persists controls on that exact node, and retains generated report/job/export lineage. The viewer Compare action also carries the current CSV or HDF5 into the correct comparison/evaluation workspace. | Browser/API regression covers exact optimization state, graph-connected multi-run comparison, and repository-file binding. |
| Data/Artifacts listed files but could not place them in the graph. | Repository discovery ended in a dead-end list. | Supported HDF5, checkpoint, and geometry rows now expose Use in pipeline and create a correctly typed source with the selected path. | Browser test selects a real repository row and verifies the new exact block/path. |
| Benchmark rows claimed they were “qualified by repository workflow” while only listing checked-in configs. | Presence of a config is not benchmark execution evidence. | Rows now state “checked in · not executed here” and expose a real launcher preflight result. | Browser test runs the first checked-in benchmark preflight and requires a visible PASS/FAIL result. |
| Static capability counts and buttons implied implementations that did not exist. | Hard-coded sample/checkpoint/session totals drifted from the repository and information-only cards displayed clickable actions. | Live APIs own runtime counts; offline capability summaries report declared/actionable/information-only/roadmap counts. Cards without a real block action are disabled and explicitly marked No live action or Not implemented. Node evidence labels now use configured paths, run IDs, and reports rather than fictional epoch/sample totals. | Static/live browser review plus repository scan for remaining fake/demo claims. |
| Save/run/toast state was primarily visual. | Dynamic status was not consistently announced to assistive technology. | Added status/live-region semantics and alert roles for errors while retaining visible text. | DOM accessibility assertions. |
| Prediction evaluation assumed matching contiguous mesh rows. | That can score coordinates or condition channels, misalign table rows, and silently compare incompatible node/sample shapes. | Added `POST /api/evaluation/schema`; mesh, table, operator, and native-result contracts now expose sample-ID/shape checks, exact named mappings, confirmation-gated positional mappings, embedded truth, and an explicit legacy-row override. Pipeline execution persists and reuses the verified mapping. | Four backend contract tests plus Chrome mapping, duplicate-field, confirmation, persistence, and stale-response tests. |
| `chi-mgnflow` was healthy in the live registry but absent from the model palette/config UI. | A runnable method had no block, modes, key sheet, defaults, help, or removal diagnostics. Hard-coded route totals also failed as soon as the registry grew. | Added a curated cHI-MGNflow block and flow controls, valid multiscale defaults, exact removed-latent diagnostics, and live registry reconciliation. Any future trainable route receives a generic usable block automatically; counts are registry-derived. | Main Chrome regression compares every trainable registry route/key contract to GUI blocks and opens the real cHI-MGNflow flow-solver control. |
| Studio and viewer requests could complete after the user changed workspace, sample, file, or playback frame. | Late responses replaced newer UI and playback could render frames out of order. A custom source picker also stopped rendering after workspace request tokens were introduced. | Added workspace/render generations, viewer catalog/sample/upload generations, serialized playback, detached-container guards, and a properly activated custom source-picker shell. | Dedicated viewer and Studio race tests deliberately reverse response order and switch workspaces mid-request. |
| The local HTTP server exposed every file beneath `frontend/` through the static handler. | Backend source, runtime job/config state, and any session material could be fetched by path even though the UI only needs the app bundle and exports. | Static GET/HEAD now allow only `index.html`, `styles.css`, `src/*.js`, and individual `runtime/exports/*` downloads; traversal and directory listing are rejected. CSP, frame denial, permissions policy, MIME sniffing protection, and same-origin resource policy are emitted. | Live HTTP allowlist test covers allowed assets/exports, source/runtime denial, encoded traversal, directories, GET, HEAD, and headers. |
| HDF5/contracts, files, docs, and config-audit lists hid catalog limits or rendered unbounded result sets. | Users could mistake a server cap for “no match,” while hundreds of rows made the System audit hard to use. | APIs report matched/limit/truncated counts; the UI explains server coverage and adds filtering plus bounded Show more pagination. Config audit now searches path/model/mode and supports Failures only. | System Chrome test runs the full current config audit, pages it, filters cHI-MGNflow, filters failures, and opens diagnostics. |
| Modal focus, small-window layout, and history replacement were unsafe. | Keyboard focus escaped stacked dialogs; narrow screens hid panels; template/import/run replacement could destroy work without a recoverable snapshot; duplicated blocks retained stale run evidence. | Added stacked modal inertness/focus trap/Escape/focus return, drawer layouts down to 600 px, confirmation plus complete Undo snapshots for destructive loads, and evidence stripping on duplicate. | Focus, responsive, history-safety, and template-dialog regressions pass. |
| Deployment controls and LLM credentials looked functional without a safe complete contract. | Deployment fields were not shared with graph execution; cHI could be misclassified as deterministic MGN by the portable bundle; LLM secrets could be persisted or sent over HTTP without a deliberate decision. | Deployment now shares one graph/autofill/metadata-gated portable-inference contract. Unsupported cHI/MLP/SimulGen checkpoints are directed to native routes before job creation. LLM settings persist no password, default to HTTPS, require explicit insecure-HTTP consent, and keep the secret only in process memory/environment. | Deployment Chrome test, checkpoint-support backend tests, and LLM backend security tests pass. |
| Inference exposed family-specific controls that were not consistently serialized into a connected model or checkpoint-only run. | A cHI flow solver/readout, MGN-V ensemble count, or operator chunk field could change on screen while the native config retained a different value. | The Inference block now owns every nonblank run-time key accepted by the resolved model, while checkpoint identity and graph-linked dataset/model paths remain authoritative. | `chi-native-inference-smoke-runner.js` selects real `ex1_smoke.pth` and `ex1_infer.h5`, sets CPU, one timestep, Euler, and deterministic mean through Chrome, completes one HDF5 rollout, then opens it in the GUI and requires a real mesh/field draw. |
| The browser suite covered core workflows but left many visible controls reachable only by inspection, not actual clicks. Uploaded paths containing a nested `dataset/` segment were also rewritten to the wrong repository root. | A button could remain unbound or silently drop values even while the main happy path passed; a locally uploaded HDF5 could not be reopened in the viewer. | Added an inventory-driven control-surface regression for editor, config, model, System, deploy, catalogs, viewer, evaluation, comparison, optimization, export, navigation, and runtime actions. Path normalization now selects the earliest exact repository-root segment, preserving `frontend/runtime/uploads/...`; model rows expose a real keyboard-accessible Details button. | Fourteen Chrome runners pass. The inventory finds 137 ID-bearing interactive controls; the only three not named directly are hidden file inputs, each exercised through its visible file-chooser button. |

## Deliberate boundaries

- Training-loss values from different model families are not treated as a fair
  accuracy comparison merely because both are numeric. If connected runs have
  no common metric key, the Studio explains that a qualified held-out
  evaluation is required and keeps the existing real-CSV ranking workflow.
- Metrics are parsed from persisted Studio job logs. This preserves current
  method repositories unchanged. A future structured event writer can replace
  the parser without changing the browser contract.
- Authentication, remote multi-user isolation, quotas, and remote artifact
  storage remain outside this local Studio scope.

## Completion checklist

- [x] Python parser, lineage, and API unit tests pass.
- [x] JavaScript syntax and scoped `git diff --check` pass; unrelated user worktree whitespace is untouched.
- [x] Focused Train Metrics test passes.
- [x] Connected-run comparison and cycle-prevention test passes.
- [x] Existing parameter spreadsheet and shared artifact viewer tests pass.
- [x] Versioned persistence, Import/Export validation, node menu, and edge deletion tests pass.
- [x] Schema-driven optimization, repository-file binding, and benchmark preflight tests pass.
- [x] Full live Studio regression passes with all registered routes.
- [x] Schema-aware evaluation and stale-response race regressions pass.
- [x] Modal accessibility, 600 px responsive layout, history safety, deployment,
  System audit, parameter binding, viewer, and viewer-race regressions pass.
- [x] Static server allowlist/security and just-in-time launch preflight tests pass.
- [x] Live registry contains 12 healthy routes; all 11 trainable routes have GUI blocks.
- [x] Real checkpoint-only cHI-MGNflow CPU inference completes through Chrome clicks and publishes one HDF5 result.
- [x] Inventory-driven control-surface smoke clicks every distinct visible action family, including all workspace execution and export paths.
