from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from studio_backend import llm_configure


class LlmSettingsSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.original_path = llm_configure.SETTINGS_PATH
        self.original_root = llm_configure.RUNTIME_ROOT
        self.original_secret = llm_configure._secret_cache["password"]
        llm_configure.RUNTIME_ROOT = self.root
        llm_configure.SETTINGS_PATH = self.root / "llm_client.local.json"
        llm_configure._secret_cache["password"] = ""
        llm_configure._token_cache.update(token=None, expires_at=0.0)

    def tearDown(self) -> None:
        llm_configure.SETTINGS_PATH = self.original_path
        llm_configure.RUNTIME_ROOT = self.original_root
        llm_configure._secret_cache["password"] = self.original_secret
        llm_configure._token_cache.update(token=None, expires_at=0.0)
        self._temporary.cleanup()

    def test_fresh_checkout_has_no_endpoint_or_default_credentials(self) -> None:
        public = llm_configure.public_settings()
        self.assertEqual(public["scheme"], "https")
        self.assertEqual(public["master_ip"], "")
        self.assertEqual(public["username"], "")
        self.assertFalse(public["password_configured"])
        self.assertFalse(public["ready"])

    def test_password_is_session_only(self) -> None:
        public = llm_configure.save_settings({
            "scheme": "https",
            "master_ip": "llm.internal",
            "port": 10002,
            "username": "studio",
            "password": "session-secret",
        })
        persisted = json.loads(llm_configure.SETTINGS_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("password", persisted)
        self.assertTrue(public["password_configured"])
        self.assertTrue(public["ready"])
        self.assertEqual(public["base_url"], "https://llm.internal:10002")

    def test_http_requires_explicit_transport_consent(self) -> None:
        settings = {
            **llm_configure._DEFAULT_SETTINGS,
            "scheme": "http",
            "master_ip": "10.0.0.5",
            "username": "studio",
        }
        with self.assertRaisesRegex(ValueError, "Refusing to send"):
            llm_configure._base_url(settings, require_safe=True)
        settings["allow_insecure_http"] = True
        self.assertEqual(llm_configure._base_url(settings, require_safe=True), "http://10.0.0.5:10002")

    def test_legacy_plaintext_password_is_removed_on_read(self) -> None:
        llm_configure.SETTINGS_PATH.write_text(json.dumps({
            "master_ip": "legacy.internal",
            "port": 10002,
            "username": "legacy",
            "password": "old-secret",
        }), encoding="utf-8")
        settings = llm_configure._load_settings()
        persisted = json.loads(llm_configure.SETTINGS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(settings["master_ip"], "legacy.internal")
        self.assertEqual(llm_configure._secret_cache["password"], "old-secret")
        self.assertNotIn("password", persisted)


if __name__ == "__main__":
    unittest.main()
