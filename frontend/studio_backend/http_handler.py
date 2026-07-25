"""The Studio's one HTTP surface: request parsing/dispatch only.

Every route handler here is a thin call into another studio_backend module —
this file owns no domain logic itself, just JSON in/out, status codes, and
path-based dispatch.
"""

from __future__ import annotations

import json
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from studio_backend.artifact_preview import artifact_sample, artifact_samples
from studio_backend.audit import audit_configs, explain_config
from studio_backend.analysis import (
    export_artifact,
    run_field_evaluation,
    run_model_comparison,
    run_optimization,
)
from studio_backend.hdf5_preview import hdf5_sample, hdf5_samples, hdf5_summary
from studio_backend.native_jobs import (
    UPLOAD_SUFFIXES,
    create_exe_build_job,
    create_inference_job,
    create_simulgen_smoke_fixture,
)
from studio_backend.paths import (
    CONFIG_RUNTIME,
    FRONTEND_ROOT,
    MAX_BODY,
    MAX_TEXT,
    MAX_UPLOAD,
    RUNTIME_ROOT,
    SUITE_ROOT,
    json_safe,
    relative,
    safe_repo_path,
    slug,
)
from studio_backend.state import STATE, PreflightFailure
from studio_backend.suite_bridge import config_catalog, documentation_catalog, file_catalog
from studio_backend.system_info import deployment_status, gpu_inventory


class StudioRequestHandler(SimpleHTTPRequestHandler):
    server_version = "AI-CAE4ALL-Studio/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if self.path.startswith("/api/") or self.path.endswith((".js", ".css", ".html")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(json_safe(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length.") from exc
        if length <= 0 or length > MAX_BODY:
            raise ValueError("Request body must be between 1 byte and 2 MiB.")
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON request body must be an object.")
        return payload

    def receive_upload(self, kind: str) -> dict[str, Any]:
        suffixes = UPLOAD_SUFFIXES.get(kind)
        if suffixes is None:
            raise ValueError("Upload kind must be dataset, geometry, or checkpoint.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid upload Content-Length.") from exc
        if length <= 0 or length > MAX_UPLOAD:
            raise ValueError("Upload must be between 1 byte and 8 GiB.")
        raw_name = unquote(self.headers.get("X-Filename", "")).strip()
        from pathlib import Path

        source_name = Path(raw_name).name
        suffix = Path(source_name).suffix.lower()
        if not source_name or suffix not in suffixes:
            raise ValueError(f"Unsupported {kind} file type. Allowed: {', '.join(sorted(suffixes))}")
        destination_dir = RUNTIME_ROOT / "uploads" / kind
        destination_dir.mkdir(parents=True, exist_ok=True)
        import uuid

        destination = destination_dir / f"{slug(Path(source_name).stem)}-{uuid.uuid4().hex[:8]}{suffix}"
        remaining = length
        try:
            with destination.open("xb") as handle:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Upload ended before the advertised content length.")
                    handle.write(chunk)
                    remaining -= len(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return {
            "ok": True,
            "kind": kind,
            "path": relative(destination),
            "name": source_name,
            "size": destination.stat().st_size,
        }

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            return super().do_GET()
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self.send_json({**STATE.health(), "gpus": gpu_inventory()})
            elif parsed.path == "/api/models":
                self.send_json({"items": STATE.models()})
            elif parsed.path == "/api/configs":
                self.send_json(config_catalog(query.get("model", [""])[0]))
            elif parsed.path == "/api/config":
                path = safe_repo_path(query.get("path", [""])[0], (SUITE_ROOT / "configs", CONFIG_RUNTIME))
                self.send_json({"path": relative(path), "text": path.read_text(encoding="utf-8")})
            elif parsed.path == "/api/audit-configs":
                self.send_json(audit_configs(strict=query.get("strict", ["0"])[0] not in {"0", "", "false"}))
            elif parsed.path == "/api/docs":
                self.send_json(documentation_catalog())
            elif parsed.path == "/api/doc":
                path = safe_repo_path(query.get("path", [""])[0], (SUITE_ROOT,))
                if path.suffix.lower() != ".md":
                    raise ValueError("Only Markdown documents can be opened here.")
                self.send_json({"path": relative(path), "text": path.read_text(encoding="utf-8")[:MAX_TEXT]})
            elif parsed.path == "/api/files":
                self.send_json(file_catalog(query.get("kind", ["artifact"])[0]))
            elif parsed.path == "/api/hdf5":
                path = safe_repo_path(query.get("path", [""])[0], (SUITE_ROOT,))
                if path.suffix.lower() not in {".h5", ".hdf5"}:
                    raise ValueError("Select an .h5 or .hdf5 file.")
                self.send_json(hdf5_summary(path))
            elif parsed.path == "/api/hdf5/samples":
                path = safe_repo_path(query.get("path", [""])[0], (SUITE_ROOT,))
                self.send_json(hdf5_samples(path))
            elif parsed.path == "/api/hdf5/sample":
                path = safe_repo_path(query.get("path", [""])[0], (SUITE_ROOT,))
                self.send_json(
                    hdf5_sample(
                        path,
                        query.get("sample", ["0"])[0],
                        int(query.get("feature", ["3"])[0]),
                        int(query.get("timestep", ["0"])[0]),
                    )
                )
            elif parsed.path == "/api/preview/samples":
                path = safe_repo_path(query.get("path", [""])[0], (SUITE_ROOT,))
                self.send_json(artifact_samples(path))
            elif parsed.path == "/api/preview/sample":
                path = safe_repo_path(query.get("path", [""])[0], (SUITE_ROOT,))
                self.send_json(
                    artifact_sample(
                        path,
                        query.get("sample", ["0"])[0],
                        int(query.get("feature", ["0"])[0]),
                        int(query.get("timestep", ["0"])[0]),
                    )
                )
            elif parsed.path == "/api/jobs":
                self.send_json({"items": STATE.list_jobs()})
            elif parsed.path == "/api/deploy":
                self.send_json(deployment_status())
            elif parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                self.send_json(STATE.get_job(job_id))
            else:
                self.send_json({"error": "Unknown API route."}, HTTPStatus.NOT_FOUND)
        except KeyError:
            self.send_json({"error": "Requested job was not found."}, HTTPStatus.NOT_FOUND)
        except (ValueError, FileNotFoundError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json(
                {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/upload":
                kind = parse_qs(parsed.query).get("kind", [""])[0]
                self.send_json(self.receive_upload(kind), HTTPStatus.CREATED)
                return
            payload = self.read_json()
            if parsed.path == "/api/preflight":
                result, _, _ = STATE.preflight(
                    str(payload.get("config", "")),
                    label=str(payload.get("label", "config")),
                    strict=bool(payload.get("strict", False)),
                    skip_filesystem=bool(payload.get("skip_filesystem", False)),
                    skip_native=bool(payload.get("skip_native", False)),
                    skip_environment=bool(payload.get("skip_environment", False)),
                )
                self.send_json(result, 200 if result["ok"] else 422)
            elif parsed.path == "/api/config/explain":
                self.send_json(
                    explain_config(
                        str(payload.get("config", "")),
                        label=str(payload.get("label", "config")),
                        strict=bool(payload.get("strict", False)),
                        skip_filesystem=bool(payload.get("skip_filesystem", False)),
                        skip_native=bool(payload.get("skip_native", False)),
                        skip_environment=bool(payload.get("skip_environment", False)),
                    )
                )
            elif parsed.path == "/api/config/save":
                path = STATE.save_config(
                    str(payload.get("config", "")),
                    str(payload.get("label", "config")),
                    "saved",
                )
                self.send_json({"ok": True, "path": relative(path)}, HTTPStatus.CREATED)
            elif parsed.path == "/api/pipeline/run":
                job = STATE.create_pipeline_job(
                    list(payload.get("steps") or []),
                    label=str(payload.get("label", "AI-CAE4ALL pipeline")),
                    strict=bool(payload.get("strict", False)),
                )
                self.send_json(job, HTTPStatus.CREATED)
            elif parsed.path == "/api/inference/run":
                self.send_json(create_inference_job(payload), HTTPStatus.CREATED)
            elif parsed.path == "/api/build/exe":
                self.send_json(create_exe_build_job(), HTTPStatus.CREATED)
            elif parsed.path == "/api/optimization/run":
                self.send_json(run_optimization(payload), HTTPStatus.CREATED)
            elif parsed.path == "/api/evaluation/run":
                self.send_json(run_field_evaluation(payload), HTTPStatus.CREATED)
            elif parsed.path == "/api/comparison/run":
                self.send_json(run_model_comparison(payload), HTTPStatus.CREATED)
            elif parsed.path == "/api/export":
                self.send_json(export_artifact(payload), HTTPStatus.CREATED)
            elif parsed.path == "/api/simulgen/smoke-fixture":
                self.send_json(create_simulgen_smoke_fixture(), HTTPStatus.CREATED)
            elif parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
                job_id = parsed.path.split("/")[-2]
                self.send_json(STATE.cancel_job(job_id))
            else:
                self.send_json({"error": "Unknown API route."}, HTTPStatus.NOT_FOUND)
        except PreflightFailure as exc:
            self.send_json({"error": str(exc), "failures": exc.failures}, 422)
        except KeyError:
            self.send_json({"error": "Requested job was not found."}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json(
                {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=5)},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[studio-http] {self.address_string()} - {fmt % args}", flush=True)


def create_server(host: str, port: int) -> ThreadingHTTPServer:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    return ThreadingHTTPServer((host, port), StudioRequestHandler)


def serve(host: str, port: int) -> int:
    try:
        server = create_server(host, port)
    except OSError as exc:
        print(f"\nAI-CAE4ALL Studio could not use port {port}: {exc}", flush=True)
        print(f"Try another port:\n    START_STUDIO.bat {port + 1}", flush=True)
        return 1
    print(f"AI-CAE4ALL Studio API: http://{host}:{port}/api/health", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AI-CAE4ALL Studio...", flush=True)
    finally:
        server.server_close()
    return 0
