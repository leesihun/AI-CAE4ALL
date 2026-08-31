"""Prove the Studio GUI's Train button actually launches the native trainer,
for both models this session's optimize-mode loop depends on.

This does NOT reproduce the full-scale checkpoints already used for the
optimize-mode demonstration (500+500 SDFFlow epochs, 400 HI-MGN epochs would
take hours). It launches a real but tiny run through real GUI clicks: one VAE
epoch plus one FM epoch for SDFFlow, or one epoch for HI-MGN. It uses palette
add, config-modal edits, and the same "Start / resume" button a user clicks,
then confirms the native process starts,
completes, and writes a real checkpoint file. The optimize-mode runs already
reported use the properly-trained ex1 / deepjeb_himgn checkpoints; this script
exists only to close the "was Train ever actually clicked" gap.

Usage: python deepjeb_train_gui_runner.py <sdfflow|chi-mgnflow> [studio-url]

One model per process invocation, run in the foreground (not backgrounded with
nohup+&): a backgrounded shell on Windows does not reliably keep a detached
Chromium/Playwright process tree alive past the point where the launching
tool call itself returns, which silently truncated the first attempts at this
mid-run with no Python-level exception at all.
"""

import io
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)

TARGET = sys.argv[1] if len(sys.argv) > 1 else "sdfflow"
require_arg = TARGET in ("sdfflow", "chi-mgnflow")
if not require_arg:
    print(f"unknown target {TARGET!r}; expected sdfflow or chi-mgnflow", file=sys.stderr)
    sys.exit(2)
STUDIO_URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8200/index.html?welcome=0"
SUITE_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
RUNTIME_DIR.mkdir(exist_ok=True)

SDFFLOW_OUT_DIR = f"../output/geometry_generation/gui_train_smoke_{int(time.time())}"
CHI_MODELPATH = f"../output/deepjeb_himgn/gui_train_smoke_{int(time.time())}.pth"


class GuiRunError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GuiRunError(message)


def open_full_config(page, node_id):
    page.locator(f'[data-node-id="{node_id}"] .node-head').click()
    page.locator("#openFullConfig").click()
    page.wait_for_selector("#configOverlay.open")


def fill_full_config_field(page, key, value):
    # Reopening the modal once per field triggered a real app bug (a blur
    # handler racing a DOM replacement, surfaced as a NotFoundError on
    # innerHTML) that silently broke the Run button afterward. Opening once
    # and searching+filling repeatedly inside that one session avoids it.
    page.locator("#configSearch").fill(key)
    control = page.locator(f'.full-config-control[data-key="{key}"]')
    control.wait_for(timeout=10_000)
    tag = control.evaluate("el => el.tagName")
    if tag == "SELECT":
        control.select_option(value)
    else:
        control.fill(value)
    # Root cause (found via the pageerror stack trace): the app's own 'change'
    # handler calls renderConfig(), which replaces #configFields' innerHTML --
    # removing `control` itself. If `control` is still focused at that moment,
    # Chrome forces an implicit blur mid-removal and throws "node to be
    # removed is no longer a child of this node... moved in a blur event
    # handler". Blurring BEFORE dispatch (not after, which is too late) keeps
    # the removal from happening on a focused node in the first place.
    control.evaluate("el => el.blur()")
    control.dispatch_event("change")


def save_full_config(page):
    page.locator("#saveConfig").click()
    page.wait_for_function(
        "() => !document.querySelector('#configOverlay')?.classList.contains('open')")
    page.wait_for_timeout(300)


def set_full_config_controls(page, node_id, overrides, browser_errors=None):
    open_full_config(page, node_id)
    for key, value in overrides.items():
        before = len(browser_errors) if browser_errors is not None else 0
        fill_full_config_field(page, key, value)
        if browser_errors is not None and len(browser_errors) > before:
            print(f"  [diagnostic] field {key!r} <- {value!r} triggered "
                 f"{len(browser_errors) - before} new browser error(s)")
    save_full_config(page)


def job_detail(page, origin, job_id):
    response = page.request.get(f"{origin}/api/jobs/{job_id}")
    require(response.ok, f"Job status request failed with HTTP {response.status}")
    return response.json()


def choose_repository_file(page, node_id, query, expected_path):
    """Set a source block's path via the real browse-and-pick UI (not by
    typing the path directly), matching how a user actually attaches a
    dataset file.
    """
    page.locator(f'[data-node-id="{node_id}"] .node-head').click()
    page.locator("#browseInputSource").click()
    page.wait_for_selector("#inputPickerSearch")
    page.locator("#inputPickerSearch").fill(query)
    choice = page.locator(f'[data-use-input="{expected_path}"]')
    choice.wait_for(timeout=10_000)
    choice.click()
    page.wait_for_function(
        "() => !document.querySelector('#studioOverlay')?.classList.contains('open')")


def run_training_job(page, origin, block_type, dataset_path, overrides, tag, confirmations,
                     browser_errors, timeout_min=15):
    """Add a dataset source wired into a model block (required: a model's
    `data` input port must be wired for any mode whose spec requires
    `dataset_dir`, typing the path into full-config alone does not satisfy the
    client-side graph validation -- see validate.js's portRequiredInMode),
    apply the rest of the overrides, click Train, and poll the real job to
    completion. Returns the finished job dict.
    """
    page.locator("#templateSelect").select_option("blank")
    page.locator('.palette-item[data-block-type="source.hdf5"]').click()
    page.locator(f'.palette-item[data-block-type="{block_type}"]').click()
    ids = page.evaluate(
        "(t) => Object.fromEntries(window.__AI_CAE_FRONTEND__.state.nodes.map(n => [n.type, n.id]))",
        block_type)
    require("source.hdf5" in ids and block_type in ids, f"Palette did not add source.hdf5 + {block_type}")

    choose_repository_file(page, ids["source.hdf5"], dataset_path.rsplit("/", 1)[-1], dataset_path)
    page.locator(f'[data-node="{ids["source.hdf5"]}"][data-port="data"][data-direction="output"]').click()
    page.locator(f'[data-node="{ids[block_type]}"][data-port="data"][data-direction="input"]').click()
    wired = page.evaluate("() => window.__AI_CAE_FRONTEND__.state.edges.length")
    require(wired == 1, f"[{tag}] expected exactly one edge after wiring the dataset, found {wired}")
    node_id = ids[block_type]

    set_full_config_controls(page, node_id, overrides, browser_errors=browser_errors)
    page.screenshot(path=str(RUNTIME_DIR / f"deepjeb-train-{tag}-configured.png"))

    serialized = page.evaluate(
        """async (id) => {
            const { executableSteps } = await import('./src/validate.js');
            return executableSteps(id)[0] || null;
        }""", node_id)
    require(serialized, f"{block_type} produced no executable step")
    parsed = {}
    for line in serialized["config"].splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            parsed[parts[0]] = parts[1]
    for key, value in overrides.items():
        require(parsed.get(key) == value,
               f'{tag}: config key "{key}" is {parsed.get(key)!r}, expected {value!r}\n'
               f'--- actual config ---\n{serialized["config"]}')
    expected_dataset_dir = "../" + dataset_path
    require(parsed.get("dataset_dir") == expected_dataset_dir,
           f'{tag}: dataset_dir is {parsed.get("dataset_dir")!r} (autofilled from the wired '
           f'source.hdf5 block), expected {expected_dataset_dir!r}\n'
           f'--- actual config ---\n{serialized["config"]}')
    print(f"[{tag}] native config verified:\n{serialized['config']}\n")

    page.locator(f'[data-node-id="{node_id}"] .node-head').click()
    page.wait_for_timeout(300)
    graph_errors = page.evaluate(
        """async () => {
            const { validateGraph } = await import('./src/validate.js');
            return validateGraph(false);
        }""")
    if graph_errors:
        print(f"[{tag}] validateGraph errors before Run click: {graph_errors}")
    confirmations.clear()
    with page.expect_response(
        lambda r: r.url.endswith("/api/pipeline/run") and r.request.method == "POST",
        timeout=60_000,
    ) as response_info:
        page.locator("#inspectorRun").click()
    response = response_info.value
    submitted = response.json()
    require(response.status == 201,
           f"[{tag}] Pipeline submission returned HTTP {response.status}: {submitted}")
    require(any("Execute the real AI-CAE4ALL launcher" in m for m in confirmations),
           f"[{tag}] destructive execution confirmation was not shown")
    page.screenshot(path=str(RUNTIME_DIR / f"deepjeb-train-{tag}-running.png"))

    job = submitted
    deadline = time.time() + timeout_min * 60
    while job.get("status") in ("queued", "running") and time.time() < deadline:
        page.wait_for_timeout(3_000)
        job = job_detail(page, origin, submitted["id"])
    require(job.get("status") not in ("queued", "running"),
           f"[{tag}] did not finish within {timeout_min} min (job {job.get('id')})")
    require(job.get("status") == "completed",
           f"[{tag}] {job.get('status')}:\n{(job.get('log') or '')[-4000:]}")
    require(job.get("returncode") == 0, f"[{tag}] native process returned {job.get('returncode')}")
    page.screenshot(path=str(RUNTIME_DIR / f"deepjeb-train-{tag}-completed.png"))
    return job


def main():
    origin = f"{urlsplit(STUDIO_URL).scheme}://{urlsplit(STUDIO_URL).netloc}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1680, "height": 1000})
        page = context.new_page()
        browser_errors = []
        confirmations = []
        page.on("pageerror", lambda exc: browser_errors.append(
            f"pageerror: {exc}\nstack:\n{getattr(exc, 'stack', '(no stack)')}"))
        page.on("console", lambda msg: browser_errors.append(f"console: {msg.text}")
               if msg.type == "error" else None)

        def on_dialog(dialog):
            confirmations.append(dialog.message)
            dialog.accept()

        page.on("dialog", on_dialog)

        try:
            page.goto(STUDIO_URL)
            page.wait_for_function(
                "() => document.querySelector('.route-health')?.textContent.includes('routes live')")

            if TARGET == "sdfflow":
                # ---- SDFFlow: 1-epoch VAE+FM smoke train on the real deepjeb.h5 ----
                print("=== Training SDFFlow (Geometry generation), 1+1 epoch, real deepjeb.h5 ===")
                job = run_training_job(page, origin, "model.sdfflow", "dataset/deepjeb.h5", {
                    "output_dir": SDFFLOW_OUT_DIR,
                    "vae_modelpath": f"{SDFFLOW_OUT_DIR}/sdfflow_vae.pth",
                    "fm_modelpath": f"{SDFFLOW_OUT_DIR}/sdfflow_fm.pth",
                    "vae_training_epochs": "1", "fm_training_epochs": "1",
                }, tag="sdfflow", confirmations=confirmations, browser_errors=browser_errors, timeout_min=8)
                # Both SDFFlow's and cHI-MGNflow's native cwd is one level below
                # the suite root, so a "../output/..." config value is
                # "output/..." relative to SUITE_ROOT.
                out_dir_from_root = SUITE_ROOT / SDFFLOW_OUT_DIR.replace("../", "", 1)
                vae_ckpt = out_dir_from_root / "sdfflow_vae.pth"
                fm_ckpt = out_dir_from_root / "sdfflow_fm.pth"
                require(vae_ckpt.is_file(), f"SDFFlow VAE checkpoint missing at {vae_ckpt}")
                require(fm_ckpt.is_file(), f"SDFFlow FM checkpoint missing at {fm_ckpt}")
                print(f"SDFFlow job {job['id']} completed; checkpoints written: "
                     f"{vae_ckpt.stat().st_size} + {fm_ckpt.stat().st_size} bytes\n")
            else:
                # ---- HI-MGN: 1-epoch smoke train on the real deepjeb_mgn.h5 ----
                print("=== Training HI-MGN (cHI-MGNflow), 1 epoch, real deepjeb_mgn.h5 ===")
                job = run_training_job(page, origin, "model.chi-mgnflow", "dataset/deepjeb_mgn.h5", {
                    "modelpath": CHI_MODELPATH,
                    "input_var": "2", "output_var": "2", "cond_var": "4",
                    "positional_features": "4", "feature_loss_weights": "1.0, 1.0",
                    "training_epochs": "1", "num_workers": "2",
                }, tag="chi-mgnflow", confirmations=confirmations, browser_errors=browser_errors, timeout_min=8)
                chi_ckpt = SUITE_ROOT / CHI_MODELPATH.replace("../", "", 1)
                require(chi_ckpt.is_file(), f"HI-MGN checkpoint missing at {chi_ckpt}")
                print(f"HI-MGN job {job['id']} completed; "
                     f"checkpoint written: {chi_ckpt.stat().st_size} bytes\n")

            require(not browser_errors,
                    "Browser errors occurred during the successful-looking training flow:\n"
                    + "\n".join(browser_errors))
            print(f"=== PASS: {TARGET} Train button launched, ran, and completed for real, through the GUI ===")
            return 0

        except Exception as exc:
            print("\n=== FAIL ===")
            print(repr(exc))
            try:
                page.screenshot(path=str(RUNTIME_DIR / "deepjeb-train-FAILURE.png"))
            except Exception:
                pass
            if browser_errors:
                print("Browser errors:\n" + "\n".join(browser_errors))
            return 1
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
