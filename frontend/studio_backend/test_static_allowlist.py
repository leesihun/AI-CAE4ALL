from __future__ import annotations

import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from studio_backend.http_handler import create_server
from studio_backend.paths import RUNTIME_ROOT


class StaticAllowlistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.export_dir = RUNTIME_ROOT / "exports"
        cls.export_dir.mkdir(parents=True, exist_ok=True)
        cls.export_file = cls.export_dir / "static-allowlist-test.txt"
        cls.export_file.write_text("download evidence", encoding="utf-8")
        cls.server = create_server("127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.export_file.unlink(missing_ok=True)

    def status(self, path: str, method: str = "GET") -> tuple[int, object]:
        request = Request(self.base + path, method=method)
        try:
            with urlopen(request, timeout=3) as response:
                response.read()
                return response.status, response.headers
        except HTTPError as exc:
            exc.read()
            return exc.code, exc.headers

    def test_public_app_assets_and_export_download_are_served(self) -> None:
        for path in ("/", "/index.html", "/styles.css", "/src/main.js", "/runtime/exports/static-allowlist-test.txt"):
            status, headers = self.status(path)
            self.assertEqual(status, 200, path)
            self.assertEqual(headers["X-Frame-Options"], "DENY")
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

    def test_source_runtime_and_directory_paths_are_not_public(self) -> None:
        denied = (
            "/studio_backend/http_handler.py",
            "/README.md",
            "/runtime/",
            "/runtime/exports/",
            "/runtime/jobs/private.json",
            "/src/../studio_backend/http_handler.py",
            "/src/%2e%2e/studio_backend/http_handler.py",
            "/src%5c..%5cstudio_backend%5chttp_handler.py",
        )
        for path in denied:
            with self.subTest(path=path):
                self.assertEqual(self.status(path)[0], 404)
                self.assertEqual(self.status(path, "HEAD")[0], 404)


if __name__ == "__main__":
    unittest.main()
