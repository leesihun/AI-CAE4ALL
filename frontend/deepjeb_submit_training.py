"""Submit a real training job through the Studio GUI, then detach.

A Playwright/Chromium process does not reliably survive being backgrounded
with nohup+& on Windows (see memory: deepjeb-gui-training-proof-2026-09-01),
which rules out keeping a browser open for an hours-long full training run.
This script does only the GUI part -- build the graph, wire the dataset,
configure every field, click the real "Start / resume" button, capture the
job id -- and exits. The submitted job keeps running server-side regardless of
whether the browser that submitted it is still open; poll it afterward with
plain HTTP (see poll_job.py), no browser required.

Usage: python deepjeb_submit_training.py <sdfflow|chi-mgnflow> [studio-url]
Prints the job id on the last line of stdout on success.
"""

import io
import json
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
if TARGET not in ("sdfflow", "chi-mgnflow"):
    print(f"unknown target {TARGET!r}; expected sdfflow or chi-mgnflow", file=sys.stderr)
    sys.exit(2)
STUDIO_URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8200/index.html?welcome=0"
SUITE_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
RUNTIME_DIR.mkdir(exist_ok=True)

STAMP = int(time.time())
SDFFLOW_OUT_DIR = f"../output/geometry_generation/ex1_gui_{STAMP}"
CHI_MODELPATH = f"../output/deepjeb_himgn/deepjeb_himgn_full_{STAMP}.pth"


class GuiRunError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GuiRunError(message)


def choose_repository_file(page, node_id, query, expected_path):
    page.locator(f'[data-node-id="{node_id}"] .node-head').click()
    page.locator("#browseInputSource").click()
    page.wait_for_selector("#inputPickerSearch")
    page.locator("#inputPickerSearch").fill(query)
    choice = page.locator(f'[data-use-input="{expected_path}"]')
    choice.wait_for(timeout=10_000)
    choice.click()
    page.wait_for_function(
        "() => !document.querySelector('#studioOverlay')?.classList.contains('open')")


def open_full_config(page, node_id):
    page.locator(f'[data-node-id="{node_id}"] .node-head').click()
    page.locator("#openFullConfig").click()
    page.wait_for_selector("#configOverlay.open")


def fill_full_config_field(page, key, value):
    page.locator("#configSearch").fill(key)
    control = page.locator(f'.full-config-control[data-key="{key}"]')
    control.wait_for(timeout=10_000)
    tag = control.evaluate("el => el.tagName")
    if tag == "SELECT":
        control.select_option(value)
    else:
        control.fill(value)
    # Root cause of a real app bug (see memory): the 'change' handler calls
    # renderConfig(), which replaces #configFields' innerHTML and removes this
    # very input. Chrome throws if that input is still focused when removed.
    # Blur BEFORE dispatch, not after.
    control.evaluate("el => el.blur()")
    control.dispatch_event("change")


def save_full_config(page):
    page.locator("#saveConfig").click()
    page.wait_for_function(
        "() => !document.querySelector('#configOverlay')?.classList.contains('open')")
    page.wait_for_timeout(300)


def set_full_config_controls(page, node_id, overrides):
    open_full_config(page, node_id)
    for key, value in overrides.items():
        fill_full_config_field(page, key, value)
    save_full_config(page)


def submit_training(page, block_type, dataset_path, overrides, confirmations):
    """Wire a source.hdf5 into the model, configure it, click Train, and
    return the submitted job dict without waiting for it to finish.
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
    require(wired == 1, f"expected exactly one edge after wiring the dataset, found {wired}")
    node_id = ids[block_type]

    set_full_config_controls(page, node_id, overrides)
    page.screenshot(path=str(RUNTIME_DIR / f"deepjeb-submit-{block_type}-{STAMP}-configured.png"))

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
               f'config key "{key}" is {parsed.get(key)!r}, expected {value!r}\n'
               f'--- actual config ---\n{serialized["config"]}')
    expected_dataset_dir = "../" + dataset_path
    require(parsed.get("dataset_dir") == expected_dataset_dir,
           f'dataset_dir is {parsed.get("dataset_dir")!r}, expected {expected_dataset_dir!r}')
    print(f"native config verified:\n{serialized['config']}\n")

    page.locator(f'[data-node-id="{node_id}"] .node-head').click()
    page.wait_for_timeout(300)
    confirmations.clear()
    with page.expect_response(
        lambda r: r.url.endswith("/api/pipeline/run") and r.request.method == "POST",
        timeout=60_000,
    ) as response_info:
        page.locator("#inspectorRun").click()
    response = response_info.value
    submitted = response.json()
    require(response.status == 201, f"Pipeline submission returned HTTP {response.status}: {submitted}")
    require(any("Execute the real AI-CAE4ALL launcher" in m for m in confirmations),
           "destructive execution confirmation was not shown")
    page.screenshot(path=str(RUNTIME_DIR / f"deepjeb-submit-{block_type}-{STAMP}-running.png"))
    return submitted


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1680, "height": 1000})
        page = context.new_page()
        browser_errors = []
        confirmations = []
        page.on("pageerror", lambda exc: browser_errors.append(f"pageerror: {exc}"))
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
                print("=== Submitting SDFFlow full training (500 VAE + 500 FM epochs), real deepjeb.h5 ===")
                submitted = submit_training(page, "model.sdfflow", "dataset/deepjeb.h5", {
                    "output_dir": SDFFLOW_OUT_DIR,
                    "vae_modelpath": f"{SDFFLOW_OUT_DIR}/sdfflow_vae.pth",
                    "fm_modelpath": f"{SDFFLOW_OUT_DIR}/sdfflow_fm.pth",
                    "vae_training_epochs": "500", "fm_training_epochs": "500",
                }, confirmations)
                extra = {"out_dir": SDFFLOW_OUT_DIR}
            else:
                print("=== Submitting HI-MGN full training (400 epochs), real deepjeb_mgn.h5 ===")
                submitted = submit_training(page, "model.chi-mgnflow", "dataset/deepjeb_mgn.h5", {
                    "modelpath": CHI_MODELPATH,
                    "input_var": "2", "output_var": "2", "cond_var": "4",
                    "positional_features": "4", "feature_loss_weights": "1.0, 1.0",
                    "training_epochs": "400", "num_workers": "2",
                }, confirmations)
                extra = {"modelpath": CHI_MODELPATH}

            require(not browser_errors,
                   "Browser errors occurred during submission:\n" + "\n".join(browser_errors))
            result = {"job_id": submitted["id"], "target": TARGET, **extra}
            print("SUBMITTED_JSON:" + json.dumps(result))
            return 0

        except Exception as exc:
            print("\n=== FAIL ===")
            print(repr(exc))
            try:
                page.screenshot(path=str(RUNTIME_DIR / f"deepjeb-submit-{TARGET}-FAILURE.png"))
            except Exception:
                pass
            if browser_errors:
                print("Browser errors:\n" + "\n".join(browser_errors))
            return 1
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
