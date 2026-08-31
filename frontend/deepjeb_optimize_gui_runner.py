"""Drive the real Studio GUI (no shortcuts) through the DeepJEB closed-loop
design optimization: a config-holding SDFFlow block wired into a CAD Generator
block running `mode optimize`, submitted with the same destructive-execution
confirmation any user click goes through, then polled to completion against
the actual native launcher.

Port of deepjeb-optimize-gui-runner.js to Python's playwright bindings -- no
Node `playwright` npm package is installed in this repo, only the Python one.

Usage: python deepjeb_optimize_gui_runner.py [studio-url] [surrogate|fea]
"""

import io
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

# The Windows console's active code page can't encode em dashes and similar
# characters this script's own error messages contain; force UTF-8 so a
# genuine failure is reported instead of masked by an encoding crash.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

STUDIO_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8200/index.html?welcome=0"
# "surrogate" (fast, unproven -- ~3 min for a 12-eval demo) or "fea" (exact,
# slow -- ~13s/eval, so a small budget still costs a few minutes).
BACKEND = sys.argv[2] if len(sys.argv) > 2 else "surrogate"
VAE_PATH = "../output/geometry_generation/ex1/sdfflow_vae.pth"
FM_PATH = "../output/geometry_generation/ex1/sdfflow_fm.pth"
OUTPUT_DIR = f"../output/geometry_generation/ex1/optimization_gui_{BACKEND}_{int(time.time() * 1000)}"
SURROGATE_CHECKPOINT = "../output/deepjeb_himgn/deepjeb_himgn.pth"
SURROGATE_CONFIG = "../configs/cHI-MGNflow/deepjeb/config_infer.txt"
BUDGETS = {
    "surrogate": {"opt_baseline_size": "6", "opt_budget": "12", "opt_popsize": "4"},
    "fea": {"opt_baseline_size": "4", "opt_budget": "8", "opt_popsize": "4"},
}[BACKEND]

RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
RUNTIME_DIR.mkdir(exist_ok=True)


class GuiRunError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GuiRunError(message)


def set_inspector_control(page, key, value):
    control = page.locator(f'.inspector-config[data-key="{key}"]')
    require(control.count() == 1,
           f'Inspector is missing field "{key}" (not in this node\'s visible config)')
    tag = control.evaluate("el => el.tagName")
    if tag == "SELECT":
        control.select_option(value)
    else:
        control.fill(value)
        control.dispatch_event("change")


def set_full_config_control(page, node_id, key, value):
    page.locator(f'[data-node-id="{node_id}"] .node-head').click()
    page.locator("#openFullConfig").click()
    page.wait_for_selector("#configOverlay.open")
    page.locator("#configSearch").fill(key)
    control = page.locator(f'.full-config-control[data-key="{key}"]')
    control.wait_for(timeout=10_000)
    tag = control.evaluate("el => el.tagName")
    if tag == "SELECT":
        control.select_option(value)
    else:
        control.fill(value)
        control.dispatch_event("change")
    page.locator("#saveConfig").click()
    page.wait_for_function(
        "() => !document.querySelector('#configOverlay')?.classList.contains('open')")


def job_detail(page, origin, job_id):
    response = page.request.get(f"{origin}/api/jobs/{job_id}")
    require(response.ok, f"Job status request failed with HTTP {response.status}")
    return response.json()


def main():
    origin = f"{urlsplit(STUDIO_URL).scheme}://{urlsplit(STUDIO_URL).netloc}"
    browser_errors = []
    confirmations = []
    submitted_job = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1680, "height": 1000})
        page = context.new_page()
        page.add_init_script(
            "() => localStorage.setItem('ai-cae4all.studio.welcomed.v1', '1')")
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
            page.screenshot(path=str(RUNTIME_DIR / f"deepjeb-gui-{BACKEND}-01-loaded.png"))

            # ---- Build the graph the way a person clicking the palette would ----
            page.locator("#templateSelect").select_option("blank")
            page.locator('.palette-item[data-block-type="model.sdfflow"]').click()
            page.locator('.palette-item[data-block-type="run.cad_generator"]').click()
            ids = page.evaluate(
                "() => Object.fromEntries(window.__AI_CAE_FRONTEND__.state.nodes.map(n => [n.type, n.id]))")
            require("model.sdfflow" in ids and "run.cad_generator" in ids,
                   "Palette did not add both blocks")

            page.locator(f'[data-node="{ids["model.sdfflow"]}"][data-port="model"][data-direction="output"]').click()
            page.locator(f'[data-node="{ids["run.cad_generator"]}"][data-port="model"][data-direction="input"]').click()
            wired = page.evaluate("() => window.__AI_CAE_FRONTEND__.state.edges.length")
            require(wired == 1, f"Expected exactly one edge after wiring, found {wired}")
            page.screenshot(path=str(RUNTIME_DIR / f"deepjeb-gui-{BACKEND}-02-wired.png"))

            # ---- Configure the SDFFlow config-holder: the trained ex1 checkpoints ----
            # and a fast surrogate-backed optimize run (see design_loop/surrogate.py's
            # accuracy notice -- this proves the AI-replaces-FEA path executes end to
            # end through the browser, not that its numbers are yet trustworthy).
            page.locator(f'[data-node-id="{ids["model.sdfflow"]}"] .node-head').click()
            set_inspector_control(page, "mode", "optimize")
            set_full_config_control(page, ids["model.sdfflow"], "vae_modelpath", VAE_PATH)
            set_full_config_control(page, ids["model.sdfflow"], "fm_modelpath", FM_PATH)
            set_full_config_control(page, ids["model.sdfflow"], "output_dir", OUTPUT_DIR)
            # Small budget: demonstrates the GUI mechanism, not a production search --
            # the 214-eval FEA search and 34-eval surrogate search already ran from the CLI.
            for key, value in BUDGETS.items():
                set_full_config_control(page, ids["model.sdfflow"], key, value)

            # ---- Configure the CAD Generator: run this SDFFlow config in optimize mode ----
            page.locator(f'[data-node-id="{ids["run.cad_generator"]}"] .node-head').click()
            set_inspector_control(page, "mode", "optimize")
            set_inspector_control(page, "opt_analysis", BACKEND)
            if BACKEND == "surrogate":
                missing_surrogate_issues = page.evaluate(
                    """async () => {
                        const { validateGraph } = await import('./src/validate.js');
                        return validateGraph(false);
                    }""")
                require(any("opt_surrogate_checkpoint" in issue
                            and "opt_surrogate_config" in issue
                            for issue in missing_surrogate_issues),
                        "Surrogate selection did not explain its missing checkpoint/config")
                page.locator(f'[data-node-id="{ids["model.sdfflow"]}"] .node-head').click()
                set_full_config_control(page, ids["model.sdfflow"], "opt_surrogate_checkpoint",
                                        SURROGATE_CHECKPOINT)
                set_full_config_control(page, ids["model.sdfflow"], "opt_surrogate_config",
                                        SURROGATE_CONFIG)
                page.locator(f'[data-node-id="{ids["run.cad_generator"]}"] .node-head').click()
            ready_issues = page.evaluate(
                """async () => {
                    const { validateGraph } = await import('./src/validate.js');
                    return validateGraph(false);
                }""")
            require(not ready_issues,
                    f"Configured DeepJEB graph still has issues: {' | '.join(ready_issues)}")
            page.screenshot(path=str(RUNTIME_DIR / f"deepjeb-gui-{BACKEND}-03-configured.png"))

            # ---- Confirm the exact native config the GUI will launch ----
            serialized = page.evaluate(
                """async (id) => {
                    const { executableSteps } = await import('./src/validate.js');
                    return executableSteps(id)[0] || null;
                }""", ids["run.cad_generator"])
            require(serialized, "The CAD Generator block produced no executable step")
            require("SDFFlow" in serialized["label"],
                   f"Wrong native route label: {serialized['label']}")

            parsed = {}
            for line in serialized["config"].splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2:
                    parsed[parts[0]] = parts[1]
            expected = {
                "model": "sdfflow", "mode": "optimize", "vae_modelpath": VAE_PATH,
                "fm_modelpath": FM_PATH, "output_dir": OUTPUT_DIR,
                "opt_analysis": BACKEND, **BUDGETS,
            }
            if BACKEND == "surrogate":
                expected["opt_surrogate_checkpoint"] = SURROGATE_CHECKPOINT
                expected["opt_surrogate_config"] = SURROGATE_CONFIG
            for key, value in expected.items():
                require(parsed.get(key) == value,
                       f'Native config key "{key}" is {parsed.get(key)!r}, expected {value!r}\n'
                       f'--- actual config ---\n{serialized["config"]}')
            print("Native config verified before submission:\n" + serialized["config"])

            # ---- Run it for real, through the same button and confirmation a user sees ----
            with page.expect_response(
                lambda r: r.url.endswith("/api/pipeline/run") and r.request.method == "POST",
                timeout=60_000,
            ) as response_info:
                page.locator("#inspectorRun").click()
            response = response_info.value
            submitted = response.json()
            require(response.status == 201,
                   f"Pipeline submission returned HTTP {response.status}: {submitted}")
            submitted_job = submitted["id"]
            require(len(submitted.get("steps") or []) == 1, "Expected exactly one native step")
            require(submitted["steps"][0].get("route", {}).get("model") == "sdfflow",
                   f"Wrong resolved route: {submitted['steps'][0].get('route')}")
            require(any("Execute the real AI-CAE4ALL launcher" in m for m in confirmations),
                   "The destructive execution confirmation was not shown")
            page.screenshot(path=str(RUNTIME_DIR / f"deepjeb-gui-{BACKEND}-04-running.png"))

            job = submitted
            deadline = time.time() + 12 * 60
            while job.get("status") in ("queued", "running") and time.time() < deadline:
                page.wait_for_timeout(3_000)
                job = job_detail(page, origin, submitted["id"])
            require(job.get("status") not in ("queued", "running"),
                   f"Run did not finish within 12 minutes (job {job.get('id')})")
            require(job.get("status") == "completed",
                   f"Run {job.get('status')}:\n{(job.get('log') or '')[-4000:]}")
            require(job.get("returncode") == 0, f"Native process returned {job.get('returncode')}")
            require("Wrote results to" in (job.get("log") or ""),
                   "The optimize run did not reach its own completion banner")
            backend_banner = "HI-MGN surrogate" if BACKEND == "surrogate" else "Analysis backend: FEA"
            require(backend_banner in (job.get("log") or ""),
                   f"The run did not select the {BACKEND} analysis backend")
            actual_output_dir = (Path(__file__).resolve().parent / OUTPUT_DIR).resolve()
            summary = json.loads((actual_output_dir / "summary.json").read_text(encoding="utf-8"))
            require(summary.get("analysis_backend") == BACKEND,
                    "Summary lost the selected analysis backend")
            if BACKEND == "surrogate":
                require("mesh_sensitivity" not in summary,
                        "Surrogate summary invented FEA mesh-sensitivity evidence")
                verified_rows = [summary.get("verified", {}).get(tag)
                                 for tag in ("baseline", "typical", "optimized")]
                require(all(row and int(row.get("num_nodes", 0)) > 0
                            and "num_tets" not in row for row in verified_rows),
                        f"Surrogate summary lost real graph-node counts: {verified_rows}")
                header = (actual_output_dir / "optimize_summary.csv").read_text(
                    encoding="utf-8").splitlines()[0].split(",")
                require("num_nodes" in header and "num_tets" not in header,
                        f"GUI result table exposes the wrong mesh cardinality: {header}")

            page.reload()
            page.wait_for_function(
                "() => document.querySelector('.route-health')?.textContent.includes('routes live')")
            page.locator(f'[data-node-id="{ids["run.cad_generator"]}"] .node-head').click()
            page.screenshot(path=str(RUNTIME_DIR / f"deepjeb-gui-{BACKEND}-05-completed.png"))

            print("\n=== PASS: DeepJEB closed-loop optimization ran end to end through the Studio GUI ===")
            step0 = (job.get("steps") or [{}])[0]
            print(f"Job {job.get('id')} | results: {step0.get('results')} | "
                 f"samples: {step0.get('results_samples')}")
            print(f"Output dir: {OUTPUT_DIR}")
            return 0

        except Exception as exc:
            print("\n=== FAIL ===")
            print(repr(exc))
            try:
                page.screenshot(path=str(RUNTIME_DIR / f"deepjeb-gui-{BACKEND}-FAILURE.png"))
            except Exception:
                pass
            if submitted_job:
                try:
                    job = job_detail(page, origin, submitted_job)
                    print(f"Job {submitted_job} status={job.get('status')} "
                         f"returncode={job.get('returncode')}")
                    print((job.get("log") or "")[-4000:])
                except Exception:
                    pass
            if browser_errors:
                print("Browser errors:\n" + "\n".join(browser_errors))
            return 1
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
