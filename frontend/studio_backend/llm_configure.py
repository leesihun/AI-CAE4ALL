"""Client for the LLM_API_fast master node's OpenAI-compatible chat endpoint.

This lets the Studio ask an LLM to rewrite a block's flat `.txt` configuration
in place. It only speaks LLM_API_fast's HTTP API (`/api/auth/login` then
`/v1/chat/completions`, see that repo's README) - nothing from that separate
repository is imported or vendored here.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from studio_backend.paths import RUNTIME_ROOT

SETTINGS_PATH = RUNTIME_ROOT / "llm_client.local.json"

# A Studio checkout must never ship working administrator credentials. The
# endpoint is opt-in, HTTPS is the default, and the password lives only in this
# process (or AI_CAE_LLM_PASSWORD) rather than in the settings file.
_DEFAULT_SETTINGS = {
    "master_ip": "",
    "port": 10002,
    "username": "",
    "scheme": "https",
    "allow_insecure_http": False,
}

_token_cache: dict[str, Any] = {"token": None, "expires_at": 0.0}
_secret_cache: dict[str, str] = {"password": os.environ.get("AI_CAE_LLM_PASSWORD", "")}
_PERSISTED_KEYS = frozenset(_DEFAULT_SETTINGS)


def _load_settings() -> dict[str, Any]:
    if SETTINGS_PATH.is_file():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # Migrate old Studio files that wrote the password in plaintext:
                # retain it for this process, then immediately rewrite only the
                # non-secret connection metadata.
                legacy_password = data.get("password")
                if isinstance(legacy_password, str) and legacy_password and not _secret_cache["password"]:
                    _secret_cache["password"] = legacy_password
                clean = {key: data[key] for key in _PERSISTED_KEYS if key in data}
                if "password" in data:
                    try:
                        SETTINGS_PATH.write_text(json.dumps(clean, indent=2), encoding="utf-8")
                    except OSError:
                        pass
                return {**_DEFAULT_SETTINGS, **clean}
        except (ValueError, OSError):
            pass
    return dict(_DEFAULT_SETTINGS)


def _base_url(settings: dict[str, Any], *, require_safe: bool = False) -> str:
    host = str(settings.get("master_ip") or "").strip()
    if not host:
        return ""
    scheme = str(settings.get("scheme") or "https").lower()
    if scheme not in {"https", "http"}:
        raise ValueError("LLM transport must be https or http.")
    if require_safe and scheme == "http" and not settings.get("allow_insecure_http"):
        raise ValueError(
            "Refusing to send credentials or configuration over plain HTTP. "
            "Use HTTPS, or explicitly allow insecure HTTP in System settings."
        )
    return f"{scheme}://{host}:{settings['port']}"


def public_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = settings or _load_settings()
    password_configured = bool(_secret_cache["password"])
    return {
        "master_ip": settings["master_ip"],
        "port": settings["port"],
        "username": settings["username"],
        "scheme": settings["scheme"],
        "allow_insecure_http": bool(settings["allow_insecure_http"]),
        "base_url": _base_url(settings),
        "configured": bool(settings["master_ip"] and settings["username"]),
        "password_configured": password_configured,
        "ready": bool(
            settings["master_ip"] and settings["username"] and password_configured
            and (settings["scheme"] == "https" or settings["allow_insecure_http"])
        ),
    }


def save_settings(update: dict[str, Any]) -> dict[str, Any]:
    current = _load_settings()
    for key in ("master_ip", "username"):
        if key in update and isinstance(update[key], str):
            current[key] = update[key].strip()
    if "scheme" in update:
        scheme = str(update["scheme"]).strip().lower()
        if scheme not in {"https", "http"}:
            raise ValueError("scheme must be https or http.")
        current["scheme"] = scheme
    if "allow_insecure_http" in update:
        value = update["allow_insecure_http"]
        current["allow_insecure_http"] = value is True or (
            isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}
        )
    if update.get("port") not in (None, ""):
        try:
            current["port"] = int(update["port"])
        except (TypeError, ValueError) as exc:
            raise ValueError("port must be a number.") from exc
        if not 1 <= current["port"] <= 65535:
            raise ValueError("port must be between 1 and 65535.")
    if update.get("clear_password"):
        _secret_cache["password"] = ""
    elif isinstance(update.get("password"), str) and update["password"]:
        _secret_cache["password"] = update["password"]
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    persisted = {key: current[key] for key in _PERSISTED_KEYS}
    SETTINGS_PATH.write_text(json.dumps(persisted, indent=2), encoding="utf-8")
    _token_cache["token"] = None
    return public_settings(current)


def _call(url: str, payload: dict[str, Any] | None, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise ValueError(f"LLM master node returned HTTP {exc.code}: {body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(
            f"Could not reach the LLM master node at {url}: {exc.reason}. "
            "Check the master IP/port in System → LLM configuration service."
        ) from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise ValueError(f"LLM master node request failed: {exc}") from exc


def _login(settings: dict[str, Any]) -> str:
    now = time.time()
    if _token_cache["token"] and _token_cache["expires_at"] > now:
        return _token_cache["token"]
    if not str(settings.get("username") or "").strip():
        raise ValueError("Configure an LLM username in System settings first.")
    if not str(settings.get("master_ip") or "").strip():
        raise ValueError("Configure an LLM master host in System settings first.")
    password = _secret_cache["password"]
    if not password:
        raise ValueError(
            "Enter the LLM password in System settings for this Studio session, "
            "or set AI_CAE_LLM_PASSWORD before starting Studio."
        )
    base = _base_url(settings, require_safe=True)
    result = _call(
        f"{base}/api/auth/login",
        {"username": settings["username"], "password": password},
        {},
        15.0,
    )
    token = result.get("access_token")
    if not token:
        raise ValueError("LLM master node did not return an access token.")
    _token_cache["token"] = token
    # Refresh well before the server's 7-day expiry rather than tracking it exactly.
    _token_cache["expires_at"] = now + 6 * 3600
    return token


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.split("\n")
    if lines and lines[-1].strip() == "```":
        lines = lines[1:-1]
    else:
        lines = lines[1:]
    return "\n".join(lines).strip()


def configure_via_llm(config_text: str, instruction: str) -> dict[str, Any]:
    """Ask the master LLM to rewrite one block's flat configuration text."""
    if not instruction.strip():
        raise ValueError("Describe the change you want the LLM to make.")
    if not config_text.strip():
        raise ValueError("There is no configuration text to send to the LLM.")
    settings = _load_settings()
    base = _base_url(settings, require_safe=True)
    token = _login(settings)
    payload = {
        "model": "default",
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You edit AI-CAE4ALL flat `key value` configuration files, one pair per line. "
                    "Apply the user's instruction to the given configuration and return ONLY the "
                    "complete updated file contents - keep every existing key unless the instruction "
                    "says to add or remove one, keep one `key value` pair per line, and do not add "
                    "explanations, comments, or markdown code fences."
                ),
            },
            {
                "role": "user",
                "content": f"Current configuration:\n{config_text}\n\nInstruction: {instruction.strip()}",
            },
        ],
    }
    result = _call(f"{base}/v1/chat/completions", payload, {"Authorization": f"Bearer {token}"}, 180.0)
    choices = result.get("choices") or []
    if not choices:
        raise ValueError("LLM master node returned no completion.")
    text = str(choices[0].get("message", {}).get("content", ""))
    cleaned = _strip_code_fence(text)
    if not cleaned:
        raise ValueError("LLM master node returned an empty response.")
    return {"text": f"{cleaned}\n", "raw": text}
